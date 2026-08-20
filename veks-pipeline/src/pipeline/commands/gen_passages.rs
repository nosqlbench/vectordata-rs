// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: derive passages from an s2orc-format text corpus.
//!
//! Reads record-oriented JSONL (optionally gzip-compressed) shards in the
//! s2orc shape — `corpusid` + body text + `paragraph`/section-header
//! character spans, in the `s2orc_v2` layout (`body.*`), the classic
//! `content.*` nesting, or flat derivatives with a top-level `text` —
//! and chunks each selected document into passages
//! with the deterministic, versioned, section-aware chunker `para-v1`. Emits `passages.parquet` plus a `parents.parquet`
//! manifest via [`veks_core::formats::passage_table`], the schema authority.
//!
//! Row order is **parent blocks**: all passages of a document are contiguous,
//! so prefix windows over the output respect parent boundaries. The global
//! passage ordinal is the output row index; passage identity is the
//! (corpusid, section, ordinal-in-section) triple.
//!
//! Every option that changes output bytes is a declared option and therefore
//! a provenance axis — chunker identity (id + params), document selection,
//! and ordering are all reproducible from the step record.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use serde_json::Value;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, ValueCompletions, render_options_table,
};
use crate::pipeline::rng;
use veks_core::formats::passage_table::{
    ParentRow, ParentTableWriter, PassageRow, PassageTableWriter,
};

/// Pipeline command: derive passages from s2orc-format JSONL shards.
pub struct GeneratePassagesOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GeneratePassagesOp)
}

/// Chunker identifier for the one implemented policy. New policies get new
/// ids — passage identity depends on (chunker, params), so an algorithm
/// change must never reuse an existing id.
const CHUNKER_PARA_V1: &str = "para-v1";

/// Emission order for parent blocks in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DocOrder {
    /// Ascending corpusid (buffers selected documents' passages in memory).
    Corpusid,
    /// Seeded parent-granularity shuffle (buffers like `Corpusid`); this is
    /// the union-design §6.1 ordering that makes prefix strata parent-sampled.
    Shuffle,
    /// Shard stream order (streaming, constant memory) — the at-scale path.
    Source,
}

impl DocOrder {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "corpusid" => Ok(Self::Corpusid),
            "shuffle" => Ok(Self::Shuffle),
            "source" => Ok(Self::Source),
            other => Err(format!(
                "unknown doc-order '{}', expected 'corpusid', 'shuffle', or 'source'",
                other
            )),
        }
    }
}

/// Chunk-budget parameters, in whitespace-delimited words (≈ tokens/1.3).
#[derive(Debug, Clone, Copy)]
struct ChunkParams {
    min_words: usize,
    target_words: usize,
    max_words: usize,
}

impl CommandOp for GeneratePassagesOp {
    fn command_path(&self) -> &str {
        "generate passages"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_GENERATE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Derive passages from an s2orc-format text corpus".into(),
            body: format!(
                r#"# generate passages

Derive passages from an s2orc-format text corpus.

## Description

Reads record-oriented JSONL (optionally `.gz`) shards in the s2orc shape —
`corpusid` plus body text and `paragraph`/section-header character spans —
and chunks each selected document into passages. Three record layouts are
accepted: `s2orc_v2` (`body.text`/`body.annotations`, header spans under
`section_header`; the parallel `bibliography` object is never chunked),
classic nested `s2orc` (`content.text`/`content.annotations.sectionheader`),
and flat derivatives with a top-level `text`. Output is `passages.parquet` (one row per passage) plus a
`parents.parquet` manifest (one row per document), written through the
schema authority in `veks-core`.

## Deterministic chunking

The chunker is versioned (`chunker`, currently only `{v1}`) and all of its
parameters are declared options, so chunker identity is a provenance axis:
the same source, options, and binary produce byte-identical output. `{v1}`
is section-aware — paragraphs are labeled with the nearest preceding section
header and packed greedily into chunks of `target-words`/`max-words` words,
splitting oversized paragraphs at word boundaries and merging a trailing
fragment below `min-words` into its predecessor.

## Document selection and ordering

`doc-limit N` selects the N lowest corpusids among records with non-empty
body text — a deterministic parent-level cap. `doc-order` fixes the parent
block order in the output: `corpusid` (ascending), `shuffle` (seeded
parent-granularity shuffle, which makes prefix windows behave as
parent-sampled strata), or `source` (shard stream order — the only
constant-memory mode; `corpusid` and `shuffle` buffer the selected
documents' passages before writing).

## Output contract

Row order is parent blocks: all passages of a document are contiguous, with
sections in document order. The global passage ordinal is the row index —
downstream embedding must preserve it (row i of the vectors artifact embeds
row i of `passages.parquet`; see `verify alignment`). Passage identity is
(corpusid, section, ordinal), and `char_start`/`char_end` are character
offsets into the source document text.

## Options

{opts}"#,
                v1 = CHUNKER_PARA_V1,
                opts = render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source = match options.require("source") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let output = match options.require("output") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let parents_path = options
            .get("parents")
            .map(|s| resolve_path(s, &ctx.workspace))
            .unwrap_or_else(|| {
                output
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("parents.parquet")
            });

        let chunker = options.get("chunker").unwrap_or(CHUNKER_PARA_V1);
        if chunker != CHUNKER_PARA_V1 {
            return error_result(
                format!("unknown chunker '{}', expected '{}'", chunker, CHUNKER_PARA_V1),
                start,
            );
        }
        let params = {
            let parsed: Result<ChunkParams, String> = (|| {
                Ok(ChunkParams {
                    min_words: options.parse_or("min-words", 40)?,
                    target_words: options.parse_or("target-words", 170)?,
                    max_words: options.parse_or("max-words", 230)?,
                })
            })();
            match parsed {
                Ok(p) => p,
                Err(e) => return error_result(e, start),
            }
        };
        if params.min_words == 0
            || params.target_words < params.min_words
            || params.max_words < params.target_words
        {
            return error_result(
                format!(
                    "invalid chunk budgets: require 0 < min-words ({}) <= target-words ({}) <= max-words ({})",
                    params.min_words, params.target_words, params.max_words
                ),
                start,
            );
        }
        let doc_limit: Option<usize> = match options.get("doc-limit") {
            None => None,
            Some(s) => match s.parse() {
                Ok(n) if n > 0 => Some(n),
                _ => return error_result(format!("invalid doc-limit: '{}'", s), start),
            },
        };
        let doc_order = match DocOrder::parse(options.get("doc-order").unwrap_or("corpusid")) {
            Ok(o) => o,
            Err(e) => return error_result(e, start),
        };
        let seed = rng::parse_seed(options.get("seed"));

        let files_selector = options.get("files").unwrap_or("all").to_string();

        let shards = match enumerate_shards(&source)
            .and_then(|s| select_shards(s, &files_selector))
        {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        if shards.is_empty() {
            return error_result(
                format!("no .jsonl/.jsonl.gz shards found at {}", source.display()),
                start,
            );
        }
        ctx.ui.log(&format!(
            "chunking {} shard(s) from {} with {} (min/target/max words {}/{}/{})",
            shards.len(),
            source.display(),
            chunker,
            params.min_words,
            params.target_words,
            params.max_words
        ));

        // ── Pass 1: scan corpusids + body presence, fix the selected set ──
        let scan_pb = ctx.ui.bar_with_unit(0, "scan", "rec");
        let scan = match scan_shards(&shards, &scan_pb) {
            Ok(s) => s,
            Err(e) => {
                scan_pb.finish();
                return error_result(e, start);
            }
        };
        scan_pb.finish();

        let mut selected_sorted = scan.eligible;
        selected_sorted.sort_unstable();
        if let Some(limit) = doc_limit {
            selected_sorted.truncate(limit);
        }
        let selected: HashSet<i64> = selected_sorted.iter().copied().collect();
        ctx.ui.log(&format!(
            "selected {} of {} eligible parent(s) ({} record(s) scanned, {} without body text, {} duplicate(s), {} parse error(s))",
            selected.len(),
            scan.eligible_count,
            scan.records,
            scan.no_body,
            scan.duplicates,
            scan.parse_errors
        ));
        if selected.is_empty() {
            return error_result("no documents selected — nothing to chunk".to_string(), start);
        }

        // ── Pass 2: chunk selected documents ─────────────────────────────
        let chunk_pb = ctx.ui.bar_with_unit(selected.len() as u64, "chunk", "doc");
        let mut passage_writer = match PassageTableWriter::create(&output) {
            Ok(w) => w,
            Err(e) => return error_result(e, start),
        };
        let mut parent_writer = match ParentTableWriter::create(&parents_path) {
            Ok(w) => w,
            Err(e) => return error_result(e, start),
        };

        // For buffered orders, documents are collected here and emitted
        // afterwards; for `source` order they are written as encountered.
        let mut buffered: HashMap<i64, Vec<PassageRow>> = HashMap::new();
        let mut emitted_parents: Vec<ParentRow> = Vec::new();
        let mut global_row: i64 = 0;
        let mut processed: HashSet<i64> = HashSet::with_capacity(selected.len());
        let mut chunk_errors = 0u64;

        for shard in &shards {
            let reader = match open_shard(shard) {
                Ok(r) => r,
                Err(e) => return error_result(e, start),
            };
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(e) => {
                        return error_result(format!("read error in {}: {}", shard.display(), e), start)
                    }
                };
                if line.trim().is_empty() {
                    continue;
                }
                let record: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => {
                        chunk_errors += 1;
                        continue;
                    }
                };
                let Some(corpusid) = record.get("corpusid").and_then(Value::as_i64) else {
                    continue;
                };
                if !selected.contains(&corpusid) || !processed.insert(corpusid) {
                    continue;
                }
                let doc_passages = chunk_record(corpusid, &record, &params);
                match doc_order {
                    DocOrder::Source => {
                        let parent = ParentRow {
                            corpusid,
                            passage_count: doc_passages.len() as i32,
                            row_start: global_row,
                        };
                        for row in &doc_passages {
                            if let Err(e) = passage_writer.push(row) {
                                return error_result(e, start);
                            }
                            global_row += 1;
                        }
                        if let Err(e) = parent_writer.push(&parent) {
                            return error_result(e, start);
                        }
                        emitted_parents.push(parent);
                    }
                    DocOrder::Corpusid | DocOrder::Shuffle => {
                        buffered.insert(corpusid, doc_passages);
                    }
                }
                chunk_pb.inc(1);
                if processed.len() == selected.len() {
                    break;
                }
            }
            if processed.len() == selected.len() {
                break;
            }
        }
        chunk_pb.finish();

        let missing = selected.len() - processed.len();
        if missing > 0 {
            ctx.ui.log(&format!(
                "warning: {} selected document(s) not found on the second pass",
                missing
            ));
        }

        // ── Emit buffered orders ─────────────────────────────────────────
        if doc_order != DocOrder::Source {
            let mut emit_ids = selected_sorted.clone();
            emit_ids.retain(|id| processed.contains(id));
            if doc_order == DocOrder::Shuffle {
                let mut rng_inst = rng::seeded_rng(seed);
                rng::fisher_yates_shuffle(&mut emit_ids, &mut rng_inst);
            }
            for id in emit_ids {
                let doc_passages = buffered.remove(&id).unwrap_or_default();
                let parent = ParentRow {
                    corpusid: id,
                    passage_count: doc_passages.len() as i32,
                    row_start: global_row,
                };
                for row in &doc_passages {
                    if let Err(e) = passage_writer.push(row) {
                        return error_result(e, start);
                    }
                    global_row += 1;
                }
                if let Err(e) = parent_writer.push(&parent) {
                    return error_result(e, start);
                }
                emitted_parents.push(parent);
            }
        }

        let passage_count = match passage_writer.finish() {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };
        let parent_count = match parent_writer.finish() {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };

        for (name, value) in [
            ("passage_count", passage_count),
            ("parent_count", parent_count),
        ] {
            let _ = crate::pipeline::variables::set_and_save(&ctx.workspace, name, &value.to_string());
            ctx.defaults.insert(name.to_string(), value.to_string());
        }

        let fanout = fanout_summary(&emitted_parents);
        let zero_docs = emitted_parents.iter().filter(|p| p.passage_count == 0).count();
        ctx.ui.log(&format!("passages/doc fan-out: {}", fanout));
        if chunk_errors > 0 {
            ctx.ui.log(&format!("warning: {} unparseable record(s) skipped", chunk_errors));
        }
        if zero_docs > 0 {
            ctx.ui.log(&format!("{} selected document(s) yielded zero passages", zero_docs));
        }

        CommandResult {
            status: if passage_count == 0 { Status::Warning } else { Status::Ok },
            message: format!(
                "chunked {} passages from {} parents (fan-out {}) to {}",
                passage_count,
                parent_count,
                fanout,
                output.display()
            ),
            produced: vec![output, parents_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "s2orc-format JSONL(.gz) shard file or directory of shards".to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output passages.parquet path".to_string(),
                extended_description: None,
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "parents".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Parent-manifest parquet path (default: parents.parquet beside output)"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "files".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("all".to_string()),
                description: "Shard selection over lexically-sorted basenames: first:N (strict), \
                              a glob, or all — same semantics as download s2ag"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "doc-limit".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Select the N lowest corpusids with non-empty body text (default: all)"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "doc-order".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("corpusid".to_string()),
                description: "Parent block order: corpusid, shuffle (seeded), or source (streaming)"
                    .to_string(),
                extended_description: Some(
                    "corpusid and shuffle buffer the selected documents' passages in memory \
                     before writing; source streams in shard order with constant memory. \
                     shuffle makes prefix windows over the output behave as parent-sampled \
                     strata."
                        .to_string(),
                ),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Random seed for doc-order: shuffle".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "chunker".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some(CHUNKER_PARA_V1.to_string()),
                description: "Chunker policy id (versioned; provenance axis)".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "min-words".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("40".to_string()),
                description: "Merge a trailing chunk below this many words into its predecessor"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "target-words".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("170".to_string()),
                description: "Window size in words when splitting oversized paragraphs".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "max-words".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("230".to_string()),
                description: "Maximum words packed into one passage".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
        ]
    }

    fn value_completions(&self) -> HashMap<String, ValueCompletions> {
        let mut map = HashMap::new();
        map.insert(
            "doc-order".to_string(),
            ValueCompletions::enum_values(&["corpusid", "shuffle", "source"]),
        );
        map.insert(
            "chunker".to_string(),
            ValueCompletions::enum_values(&[CHUNKER_PARA_V1]),
        );
        map
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["source"],
            &["output", "parents"],
        )
    }
}

// ── Shard enumeration and streaming ──────────────────────────────────────

/// Apply the `files` selector over lexically-sorted shard basenames —
/// `first:N` (strict: fewer than N present is an error, since a silent
/// shortfall would change the parent set), a glob, or `all`. Same selector
/// semantics as `download s2ag`, so a chunk step can name exactly the shard
/// subset a download step fetched.
pub(crate) fn select_shards(shards: Vec<PathBuf>, selector: &str) -> Result<Vec<PathBuf>, String> {
    if selector == "all" {
        return Ok(shards);
    }
    if let Some(n) = selector.strip_prefix("first:") {
        let n: usize = match n.parse() {
            Ok(n) if n > 0 => n,
            _ => return Err(format!("invalid files selector '{}': first:N needs N > 0", selector)),
        };
        if shards.len() < n {
            return Err(format!(
                "files selector 'first:{}' but only {} shard(s) present — \
                 a silent shortfall would change the parent set",
                n,
                shards.len()
            ));
        }
        return Ok(shards.into_iter().take(n).collect());
    }
    if selector.is_empty() {
        return Err("empty files selector".to_string());
    }
    let filtered: Vec<PathBuf> = shards
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|b| super::fetch_s2ag::glob_match(selector, b))
        })
        .collect();
    if filtered.is_empty() {
        return Err(format!("files selector '{}' matched no shards", selector));
    }
    Ok(filtered)
}

/// List shard files: a file source is itself; a directory yields its
/// `.jsonl` / `.jsonl.gz` / `.gz` entries in lexical filename order
/// (that order is part of the deterministic selection rule).
pub(crate) fn enumerate_shards(source: &Path) -> Result<Vec<PathBuf>, String> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    if !source.is_dir() {
        return Err(format!("source not found: {}", source.display()));
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(source)
        .map_err(|e| format!("failed to read {}: {}", source.display(), e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| {
                        !n.starts_with('.')
                            && (n.ends_with(".jsonl") || n.ends_with(".jsonl.gz") || n.ends_with(".gz"))
                    })
        })
        .collect();
    shards.sort();
    Ok(shards)
}

/// Open a shard as a buffered line reader, transparently gunzipping `.gz`.
pub(crate) fn open_shard(path: &Path) -> Result<BufReader<Box<dyn Read>>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let inner: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(flate2::read::MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(BufReader::with_capacity(1 << 20, inner))
}

// ── Pass 1: eligibility scan ─────────────────────────────────────────────

/// Lean pass-1 record: corpusid plus body presence only. Covers all three
/// accepted layouts: `s2orc_v2` (`body.text`), classic (`content.text`),
/// and flat derivatives (top-level `text`).
#[derive(Deserialize)]
struct ScanRecord {
    corpusid: Option<i64>,
    text: Option<String>,
    body: Option<ScanContent>,
    content: Option<ScanContent>,
}

#[derive(Deserialize)]
struct ScanContent {
    text: Option<String>,
}

struct ScanResult {
    /// First-occurrence corpusids with non-empty body text (unsorted).
    eligible: Vec<i64>,
    eligible_count: usize,
    records: u64,
    no_body: u64,
    duplicates: u64,
    parse_errors: u64,
}

fn scan_shards(shards: &[PathBuf], pb: &veks_core::ui::ProgressHandle) -> Result<ScanResult, String> {
    let mut seen: HashSet<i64> = HashSet::new();
    let mut eligible = Vec::new();
    let (mut records, mut no_body, mut duplicates, mut parse_errors) = (0u64, 0u64, 0u64, 0u64);

    for shard in shards {
        let reader = open_shard(shard)?;
        for line in reader.lines() {
            let line = line.map_err(|e| format!("read error in {}: {}", shard.display(), e))?;
            if line.trim().is_empty() {
                continue;
            }
            records += 1;
            if records % 1000 == 0 {
                pb.set_position(records);
            }
            let record: ScanRecord = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => {
                    parse_errors += 1;
                    continue;
                }
            };
            let Some(corpusid) = record.corpusid else {
                parse_errors += 1;
                continue;
            };
            if !seen.insert(corpusid) {
                duplicates += 1;
                continue;
            }
            let has_body = record
                .text
                .or(record.body.and_then(|b| b.text))
                .or(record.content.and_then(|c| c.text))
                .is_some_and(|t| !t.trim().is_empty());
            if has_body {
                eligible.push(corpusid);
            } else {
                no_body += 1;
            }
        }
    }
    pb.set_position(records);
    let eligible_count = eligible.len();
    Ok(ScanResult { eligible, eligible_count, records, no_body, duplicates, parse_errors })
}

// ── Chunker para-v1 ──────────────────────────────────────────────────────

/// A half-open character-offset span into the document text.
#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
}

/// Parse an s2orc annotation value into spans. The value is either a JSON
/// array of `{start, end}` objects or (as shipped in s2orc releases) a JSON
/// *string* containing that array; both shapes are accepted, anything else
/// yields no spans.
fn parse_spans(value: Option<&Value>) -> Vec<Span> {
    let Some(value) = value else { return Vec::new() };
    let owned;
    let arr = match value {
        Value::Array(a) => a,
        Value::String(s) => {
            owned = serde_json::from_str::<Value>(s).unwrap_or(Value::Null);
            match &owned {
                Value::Array(a) => a,
                _ => return Vec::new(),
            }
        }
        _ => return Vec::new(),
    };
    let mut spans: Vec<Span> = arr
        .iter()
        .filter_map(|o| {
            let start = o.get("start")?.as_f64()?;
            let end = o.get("end")?.as_f64()?;
            if start < 0.0 || end <= start {
                return None;
            }
            Some(Span { start: start as usize, end: end as usize })
        })
        .collect();
    spans.sort_by_key(|s| s.start);
    spans
}

/// Chunk one parsed s2orc record into passage rows (doc-local; the caller
/// assigns row placement). Returns an empty vec for records without body
/// text or paragraph spans.
///
/// Three record shapes are accepted, verified against real releases:
/// `s2orc_v2` (2026 releases) nests the main text as `body.text` /
/// `body.annotations` with header spans under `section_header` (the
/// parallel `bibliography` object is deliberately not chunked); classic
/// `s2orc` nests as `content.text` / `content.annotations.sectionheader`;
/// and cleaned derivatives like peS2o carry a top-level `text`. Identical
/// spans produce identical passages in every shape, so the chunker id is
/// unaffected by the input layout.
fn chunk_record(corpusid: i64, record: &Value, params: &ChunkParams) -> Vec<PassageRow> {
    let body = record.get("body");
    let content = record.get("content");
    let Some(text) = record
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| body.and_then(|b| b.get("text")).and_then(Value::as_str))
        .or_else(|| content.and_then(|c| c.get("text")).and_then(Value::as_str))
    else {
        return Vec::new();
    };
    let annotations = record
        .get("annotations")
        .or_else(|| body.and_then(|b| b.get("annotations")))
        .or_else(|| content.and_then(|c| c.get("annotations")));
    let paragraphs = parse_spans(annotations.and_then(|a| a.get("paragraph")));
    let headers = parse_spans(
        annotations
            .and_then(|a| a.get("section_header"))
            .or_else(|| annotations.and_then(|a| a.get("sectionheader"))),
    );
    chunk_text(corpusid, text, &paragraphs, &headers, params)
}

/// The `para-v1` policy over already-parsed spans (unit-testable core).
fn chunk_text(
    corpusid: i64,
    text: &str,
    paragraphs: &[Span],
    headers: &[Span],
    params: &ChunkParams,
) -> Vec<PassageRow> {
    // Character-offset addressing: s2orc spans index characters, not bytes.
    let char_starts: Vec<usize> = text
        .char_indices()
        .map(|(b, _)| b)
        .chain(std::iter::once(text.len()))
        .collect();
    let char_len = char_starts.len() - 1;
    let slice = |s: usize, e: usize| -> &str { &text[char_starts[s]..char_starts[e]] };

    // Section label per header, in document order.
    let header_labels: Vec<(usize, String)> = headers
        .iter()
        .filter(|h| h.start < char_len)
        .map(|h| {
            let end = h.end.min(char_len);
            let label = slice(h.start, end).split_whitespace().collect::<Vec<_>>().join(" ");
            (h.start, label)
        })
        .collect();
    let label_for = |para_start: usize| -> String {
        header_labels
            .iter()
            .rev()
            .find(|(hs, _)| *hs <= para_start)
            .map(|(_, l)| l.clone())
            .unwrap_or_default()
    };

    // A chunk under construction / emitted: (char_start, char_end, text, words).
    struct Chunk {
        start: usize,
        end: usize,
        text: String,
        words: usize,
    }

    let mut passages: Vec<PassageRow> = Vec::new();
    let mut ordinals: HashMap<String, i32> = HashMap::new();
    let mut group_label: Option<String> = None;
    let mut group_chunks: Vec<Chunk> = Vec::new();
    let mut cur: Vec<Span> = Vec::new();
    let mut cur_words: usize = 0;

    // Emit a whole section group, applying the trailing-fragment merge.
    let flush_group = |label: Option<String>,
                       chunks: &mut Vec<Chunk>,
                       ordinals: &mut HashMap<String, i32>,
                       passages: &mut Vec<PassageRow>| {
        let Some(label) = label else { return };
        if chunks.len() >= 2 && chunks.last().is_some_and(|c| c.words < params.min_words) {
            let tail = chunks.pop().expect("len checked");
            let prev = chunks.last_mut().expect("len checked");
            prev.text.push_str("\n\n");
            prev.text.push_str(&tail.text);
            prev.end = tail.end;
            prev.words += tail.words;
        }
        for chunk in chunks.drain(..) {
            let ordinal = ordinals.entry(label.clone()).or_insert(0);
            passages.push(PassageRow {
                corpusid,
                section: label.clone(),
                ordinal: *ordinal,
                char_start: chunk.start as i64,
                char_end: chunk.end as i64,
                text: chunk.text,
            });
            *ordinal += 1;
        }
    };

    // Close the in-progress packed chunk into the current group.
    let flush_cur = |cur: &mut Vec<Span>, cur_words: &mut usize, chunks: &mut Vec<Chunk>| {
        if cur.is_empty() {
            return;
        }
        let start = cur[0].start;
        let end = cur.last().expect("non-empty").end;
        let text = cur
            .iter()
            .map(|p| slice(p.start, p.end))
            .collect::<Vec<_>>()
            .join("\n\n");
        chunks.push(Chunk { start, end, text, words: *cur_words });
        cur.clear();
        *cur_words = 0;
    };

    for para in paragraphs {
        if para.start >= char_len {
            continue;
        }
        let para = Span { start: para.start, end: para.end.min(char_len) };
        let words = word_spans(slice(para.start, para.end), para.start);
        if words.is_empty() {
            continue;
        }
        let label = label_for(para.start);
        if group_label.as_deref() != Some(label.as_str()) {
            flush_cur(&mut cur, &mut cur_words, &mut group_chunks);
            flush_group(group_label.take(), &mut group_chunks, &mut ordinals, &mut passages);
            group_label = Some(label);
        }

        if words.len() > params.max_words {
            flush_cur(&mut cur, &mut cur_words, &mut group_chunks);
            // Split at word boundaries into target-words windows, absorbing
            // a tail smaller than min-words into the final window.
            let mut i = 0;
            while i < words.len() {
                let mut take = params.target_words.min(words.len() - i);
                let remaining = words.len() - i - take;
                if remaining > 0 && remaining < params.min_words {
                    take = words.len() - i;
                }
                let start = words[i].0;
                let end = words[i + take - 1].1;
                group_chunks.push(Chunk {
                    start,
                    end,
                    text: slice(start, end).to_string(),
                    words: take,
                });
                i += take;
            }
        } else if cur_words > 0 && cur_words + words.len() > params.max_words {
            flush_cur(&mut cur, &mut cur_words, &mut group_chunks);
            cur.push(para);
            cur_words = words.len();
        } else {
            cur.push(para);
            cur_words += words.len();
        }
    }
    flush_cur(&mut cur, &mut cur_words, &mut group_chunks);
    flush_group(group_label, &mut group_chunks, &mut ordinals, &mut passages);
    passages
}

/// Whitespace-delimited word spans of `text`, as absolute character offsets
/// (the slice starts at character offset `base` of the document).
fn word_spans(text: &str, base: usize) -> Vec<(usize, usize)> {
    let mut words = Vec::new();
    let mut start: Option<usize> = None;
    let mut idx = base;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                words.push((s, idx));
            }
        } else if start.is_none() {
            start = Some(idx);
        }
        idx += 1;
    }
    if let Some(s) = start {
        words.push((s, idx));
    }
    words
}

// ── Reporting helpers ────────────────────────────────────────────────────

/// Render the passages/doc distribution as `min/p50/p90/max (mean m)`.
fn fanout_summary(parents: &[ParentRow]) -> String {
    if parents.is_empty() {
        return "n/a".to_string();
    }
    let mut counts: Vec<i32> = parents.iter().map(|p| p.passage_count).collect();
    counts.sort_unstable();
    let pct = |p: f64| counts[((counts.len() - 1) as f64 * p) as usize];
    let mean = counts.iter().map(|&c| c as f64).sum::<f64>() / counts.len() as f64;
    format!(
        "{}/{}/{}/{} (mean {:.1})",
        counts[0],
        pct(0.5),
        pct(0.9),
        counts[counts.len() - 1],
        mean
    )
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veks_core::formats::passage_table::{read_parents, read_passages};

    fn params() -> ChunkParams {
        ChunkParams { min_words: 3, target_words: 6, max_words: 8 }
    }

    fn span(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    #[test]
    fn labels_follow_nearest_preceding_header() {
        // "Intro\naaa bbb ccc\nMethods\nddd eee fff\n"
        let text = "Intro\naaa bbb ccc\nMethods\nddd eee fff\n";
        let headers = [span(0, 5), span(18, 25)];
        let paras = [span(6, 17), span(26, 37)];
        let rows = chunk_text(7, text, &paras, &headers, &params());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].section, "Intro");
        assert_eq!(rows[0].text, "aaa bbb ccc");
        assert_eq!(rows[0].ordinal, 0);
        assert_eq!(rows[1].section, "Methods");
        assert_eq!(rows[1].text, "ddd eee fff");
        assert_eq!(rows[1].ordinal, 0);
        assert_eq!((rows[0].char_start, rows[0].char_end), (6, 17));
    }

    #[test]
    fn packs_consecutive_paragraphs_up_to_max_words() {
        // Three 3-word paragraphs, max 8 → first two pack, third stands alone.
        let text = "a b c\nd e f\ng h i";
        let paras = [span(0, 5), span(6, 11), span(12, 17)];
        let rows = chunk_text(1, text, &paras, &[], &params());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].text, "a b c\n\nd e f");
        assert_eq!(rows[1].text, "g h i");
        assert_eq!(rows[0].section, "");
        assert_eq!(rows[0].ordinal, 0);
        assert_eq!(rows[1].ordinal, 1);
    }

    #[test]
    fn splits_oversized_paragraph_at_word_boundaries() {
        // 14 words, target 6, min 3 → windows of 6, 6, then tail 2 < min
        // absorbed into the second window (6 + 8).
        let words: Vec<String> = (0..14).map(|i| format!("w{}", i)).collect();
        let text = words.join(" ");
        let paras = [span(0, text.chars().count())];
        let rows = chunk_text(1, &text, &paras, &[], &params());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].text.split_whitespace().count(), 6);
        assert_eq!(rows[1].text.split_whitespace().count(), 8);
        // window text is the exact source slice
        let joined = format!("{} {}", rows[0].text, rows[1].text);
        assert_eq!(joined, text);
    }

    #[test]
    fn trailing_fragment_merges_into_predecessor() {
        // An 8-word para fills a chunk (max 8), so the following 2-word para
        // becomes a trailing chunk below min 3 — it must merge back.
        let text = "a b c d e f g h\ni j";
        let paras = [span(0, 15), span(16, 19)];
        let rows = chunk_text(1, text, &paras, &[], &params());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].text, "a b c d e f g h\n\ni j");
        assert_eq!((rows[0].char_start, rows[0].char_end), (0, 19));
    }

    #[test]
    fn char_offsets_are_character_not_byte_indices() {
        // Multibyte characters before the paragraph shift byte offsets away
        // from character offsets; spans must address characters.
        let text = "μμμμμ\nalpha beta gamma";
        let paras = [span(6, 22)];
        let rows = chunk_text(1, text, &paras, &[], &params());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].text, "alpha beta gamma");
    }

    #[test]
    fn same_section_label_continues_ordinals_across_groups() {
        // Header A, para; header B, para; header A again (same label), para:
        // the second A-passage continues the A ordinal sequence.
        let text = "A\np one two three\nB\nq one two three\nA\nr one two three";
        let headers = [span(0, 1), span(18, 19), span(36, 37)];
        let paras = [span(2, 17), span(20, 35), span(38, 53)];
        let rows = chunk_text(1, text, &paras, &headers, &params());
        assert_eq!(rows.len(), 3);
        assert_eq!((rows[0].section.as_str(), rows[0].ordinal), ("A", 0));
        assert_eq!((rows[1].section.as_str(), rows[1].ordinal), ("B", 0));
        assert_eq!((rows[2].section.as_str(), rows[2].ordinal), ("A", 1));
    }

    #[test]
    fn parse_spans_accepts_both_encodings() {
        let direct: Value = serde_json::json!([{"start": 3, "end": 9}]);
        let encoded: Value = Value::String("[{\"start\": 3, \"end\": 9}]".to_string());
        for v in [direct, encoded] {
            let spans = parse_spans(Some(&v));
            assert_eq!(spans.len(), 1);
            assert_eq!((spans[0].start, spans[0].end), (3, 9));
        }
        assert!(parse_spans(Some(&Value::Null)).is_empty());
        assert!(parse_spans(None).is_empty());
    }

    // ── execute()-level tests over synthetic s2orc shards ────────────────

    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    /// Build the text + span arrays for a synthetic doc with `sections`
    /// sections of `paras` paragraphs, each `words_per_para` words.
    fn synth_doc(
        sections: usize,
        paras: usize,
        words_per_para: usize,
    ) -> (String, Vec<Value>, Vec<Value>) {
        let mut text = String::new();
        let mut header_spans = Vec::new();
        let mut para_spans = Vec::new();
        for s in 0..sections {
            let h = format!("Section{}", s);
            let start = text.chars().count();
            text.push_str(&h);
            header_spans.push(serde_json::json!({"start": start, "end": start + h.chars().count()}));
            text.push('\n');
            for p in 0..paras {
                let body = (0..words_per_para)
                    .map(|w| format!("w{}s{}p{}", w, s, p))
                    .collect::<Vec<_>>()
                    .join(" ");
                let start = text.chars().count();
                text.push_str(&body);
                para_spans.push(serde_json::json!({"start": start, "end": start + body.chars().count()}));
                text.push('\n');
            }
        }
        (text, para_spans, header_spans)
    }

    /// One synthetic record in the `s2orc_v2` layout: `body.text` /
    /// `body.annotations` with header spans under `section_header`, plus a
    /// `bibliography` object that must NOT be chunked.
    fn synth_record_v2(corpusid: i64, sections: usize, paras: usize, words_per_para: usize) -> String {
        let (text, para_spans, header_spans) = synth_doc(sections, paras, words_per_para);
        serde_json::json!({
            "corpusid": corpusid,
            "body": {
                "text": text,
                "annotations": {
                    "sentence": serde_json::Value::Null,
                    "paragraph": serde_json::Value::Array(para_spans).to_string(),
                    "section_header": serde_json::Value::Array(header_spans).to_string(),
                }
            },
            "bibliography": {
                "text": "ref one\n\nref two",
                "annotations": { "bib_entry": "[{\"start\":0,\"end\":7}]" }
            }
        })
        .to_string()
    }

    /// One synthetic s2orc record with `sections` sections of `paras`
    /// paragraphs, each `words_per_para` words.
    fn synth_record(corpusid: i64, sections: usize, paras: usize, words_per_para: usize) -> String {
        let (text, para_spans, header_spans) = synth_doc(sections, paras, words_per_para);
        serde_json::json!({
            "corpusid": corpusid,
            "content": {
                "text": text,
                "annotations": {
                    // s2orc ships annotation arrays JSON-encoded as strings
                    "paragraph": serde_json::Value::Array(para_spans).to_string(),
                    "sectionheader": serde_json::Value::Array(header_spans).to_string(),
                }
            }
        })
        .to_string()
    }

    #[test]
    fn v1_and_v2_record_shapes_chunk_identically() {
        let v1: Value = serde_json::from_str(&synth_record(9, 2, 3, 4)).unwrap();
        let v2: Value = serde_json::from_str(&synth_record_v2(9, 2, 3, 4)).unwrap();
        let p = ChunkParams { min_words: 3, target_words: 6, max_words: 8 };
        let from_v1 = chunk_record(9, &v1, &p);
        let from_v2 = chunk_record(9, &v2, &p);
        assert!(!from_v1.is_empty());
        assert_eq!(from_v1, from_v2);
        assert_eq!(from_v1[0].section, "Section0");
    }

    #[test]
    fn execute_handles_v2_flat_records() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        std::fs::write(
            shards.join("part-0.jsonl"),
            format!("{}\n{}\n", synth_record_v2(3, 2, 2, 4), synth_record_v2(4, 1, 1, 4)),
        )
        .unwrap();
        let (result, output, parents) = run(tmp.path(), &[]);
        assert_eq!(result.status, Status::Ok, "{}", result.message);
        let rows = read_passages(&output).unwrap();
        let parent_rows = read_parents(&parents).unwrap();
        assert_eq!(parent_rows.iter().map(|p| p.corpusid).collect::<Vec<_>>(), vec![3, 4]);
        // Doc 3: 2 sections × 2×4-word paras pack pairwise → 2 passages.
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].section, "Section0");
        assert_eq!(rows[1].section, "Section1");
    }

    /// Write a gz shard with the given records plus one no-body record.
    fn write_shard(path: &Path, records: &[String]) {
        use std::io::Write;
        let file = std::fs::File::create(path).unwrap();
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        for r in records {
            writeln!(gz, "{}", r).unwrap();
        }
        writeln!(gz, "{}", serde_json::json!({"corpusid": 999_999, "content": {"text": ""}}))
            .unwrap();
        gz.finish().unwrap();
    }

    fn run(dir: &Path, extra: &[(&str, &str)]) -> (CommandResult, PathBuf, PathBuf) {
        let output = dir.join("out/passages.parquet");
        let parents = dir.join("out/parents.parquet");
        let mut opts = Options::new();
        opts.set("source", dir.join("shards").to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("min-words", "3");
        opts.set("target-words", "6");
        opts.set("max-words", "8");
        for (k, v) in extra {
            opts.set(*k, *v);
        }
        let mut ctx = test_ctx(dir);
        let result = GeneratePassagesOp.execute(&opts, &mut ctx);
        (result, output, parents)
    }

    fn setup_shards(dir: &Path) {
        let shards = dir.join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        // Two shards; ids deliberately out of order across shards so the
        // doc-limit rule (lowest ids) crosses shard boundaries.
        write_shard(
            &shards.join("part-b.jsonl.gz"),
            &[synth_record(10, 2, 2, 4), synth_record(40, 1, 1, 4)],
        );
        write_shard(
            &shards.join("part-a.jsonl.gz"),
            &[synth_record(30, 1, 3, 4), synth_record(20, 2, 1, 4)],
        );
    }

    #[test]
    fn execute_chunks_selects_and_orders_by_corpusid() {
        let tmp = tempfile::tempdir().unwrap();
        setup_shards(tmp.path());
        let (result, output, parents) = run(tmp.path(), &[("doc-limit", "3")]);
        assert_eq!(result.status, Status::Ok, "{}", result.message);

        let rows = read_passages(&output).unwrap();
        let parent_rows = read_parents(&parents).unwrap();
        // doc-limit 3 → lowest ids 10, 20, 30 (40 excluded); corpusid order.
        assert_eq!(parent_rows.iter().map(|p| p.corpusid).collect::<Vec<_>>(), vec![10, 20, 30]);
        // Parent blocks: row_start/passage_count tile the row space exactly.
        let mut expect_start = 0i64;
        for p in &parent_rows {
            assert_eq!(p.row_start, expect_start);
            expect_start += p.passage_count as i64;
        }
        assert_eq!(expect_start, rows.len() as i64);
        // Parent-block contiguity in the passage rows themselves.
        for p in &parent_rows {
            for (i, row) in rows[p.row_start as usize..(p.row_start + p.passage_count as i64) as usize]
                .iter()
                .enumerate()
            {
                assert_eq!(row.corpusid, p.corpusid, "row {} of parent {}", i, p.corpusid);
            }
        }
        // Identity triple uniqueness.
        let mut ids: Vec<(i64, String, i32)> = rows
            .iter()
            .map(|r| (r.corpusid, r.section.clone(), r.ordinal))
            .collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), rows.len());
        // 2 sections × 2 paras of 4 words pack pairwise (8 ≤ max 8) → doc 10
        // yields one passage per section.
        let doc10 = &rows[parent_rows[0].row_start as usize
            ..(parent_rows[0].row_start + parent_rows[0].passage_count as i64) as usize];
        assert_eq!(doc10.len(), 2);
        assert_eq!(doc10[0].section, "Section0");
        assert_eq!(doc10[1].section, "Section1");
    }

    #[test]
    fn execute_is_deterministic_across_runs() {
        let tmp = tempfile::tempdir().unwrap();
        setup_shards(tmp.path());
        let (r1, output, parents) = run(tmp.path(), &[("doc-limit", "3")]);
        assert_eq!(r1.status, Status::Ok);
        let bytes1 = (std::fs::read(&output).unwrap(), std::fs::read(&parents).unwrap());
        let (r2, ..) = run(tmp.path(), &[("doc-limit", "3")]);
        assert_eq!(r2.status, Status::Ok);
        let bytes2 = (std::fs::read(&output).unwrap(), std::fs::read(&parents).unwrap());
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn execute_shuffle_order_is_seeded_and_parent_blocked() {
        let tmp = tempfile::tempdir().unwrap();
        setup_shards(tmp.path());
        let (r1, _, parents) = run(tmp.path(), &[("doc-order", "shuffle"), ("seed", "7")]);
        assert_eq!(r1.status, Status::Ok);
        let order1: Vec<i64> = read_parents(&parents).unwrap().iter().map(|p| p.corpusid).collect();
        let (r2, _, _) = run(tmp.path(), &[("doc-order", "shuffle"), ("seed", "7")]);
        assert_eq!(r2.status, Status::Ok);
        let order2: Vec<i64> = read_parents(&parents).unwrap().iter().map(|p| p.corpusid).collect();
        assert_eq!(order1, order2);
        let mut sorted = order1.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![10, 20, 30, 40]);
    }

    #[test]
    fn execute_source_order_streams_in_shard_order() {
        let tmp = tempfile::tempdir().unwrap();
        setup_shards(tmp.path());
        let (result, _, parents) = run(tmp.path(), &[("doc-order", "source")]);
        assert_eq!(result.status, Status::Ok);
        // Lexical shard order: part-a (30, 20) then part-b (10, 40).
        let order: Vec<i64> = read_parents(&parents).unwrap().iter().map(|p| p.corpusid).collect();
        assert_eq!(order, vec![30, 20, 10, 40]);
    }

    #[test]
    fn execute_plain_jsonl_matches_gz() {
        let tmp = tempfile::tempdir().unwrap();
        let shards = tmp.path().join("shards");
        std::fs::create_dir_all(&shards).unwrap();
        std::fs::write(
            shards.join("part-0.jsonl"),
            format!("{}\n", synth_record(5, 1, 2, 4)),
        )
        .unwrap();
        let (result, output, _) = run(tmp.path(), &[]);
        assert_eq!(result.status, Status::Ok, "{}", result.message);
        let rows = read_passages(&output).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].corpusid, 5);
    }

    #[test]
    fn execute_rejects_unknown_chunker_and_bad_budgets() {
        let tmp = tempfile::tempdir().unwrap();
        setup_shards(tmp.path());
        let (result, ..) = run(tmp.path(), &[("chunker", "nope-v9")]);
        assert_eq!(result.status, Status::Error);
        let (result, ..) = run(tmp.path(), &[("max-words", "2")]);
        assert_eq!(result.status, Status::Error);
    }

    #[test]
    fn select_shards_first_n_glob_and_strictness() {
        let mk = |names: &[&str]| -> Vec<PathBuf> {
            names.iter().map(PathBuf::from).collect()
        };
        let shards = mk(&["a.jsonl.gz", "b.jsonl.gz", "c.jsonl"]);
        // all passes through unchanged.
        assert_eq!(select_shards(shards.clone(), "all").unwrap(), shards);
        // first:N takes the lexical prefix.
        assert_eq!(
            select_shards(shards.clone(), "first:2").unwrap(),
            mk(&["a.jsonl.gz", "b.jsonl.gz"])
        );
        // first:N beyond what exists is an error, not a silent shortfall.
        assert!(select_shards(shards.clone(), "first:4").is_err());
        assert!(select_shards(shards.clone(), "first:0").is_err());
        // Globs filter over basenames; no match is an error.
        assert_eq!(
            select_shards(shards.clone(), "*.jsonl.gz").unwrap(),
            mk(&["a.jsonl.gz", "b.jsonl.gz"])
        );
        assert!(select_shards(shards, "z*").is_err());
    }
}
