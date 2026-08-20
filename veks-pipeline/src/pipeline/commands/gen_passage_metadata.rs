// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: derive per-passage metadata by joining S2AG `papers`
//! records onto a passage table.
//!
//! The M facet of a predicated (PVS) dataset needs one metadata row per
//! base vector, in base row order. This command builds that table for the
//! S2OA passage pipeline: it scans `papers` dataset shards (JSONL/.gz, one
//! record per corpusid) for the parents that appear in `passages.parquet`,
//! then broadcasts each parent's fields to its passages **in passage row
//! order**, emitting `metadata.parquet` through the schema authority in
//! `veks_core::formats::passage_metadata`. Row i of the output describes
//! row i of the passage table and row i of the embedded vectors — the same
//! ordinal-identity contract `verify alignment` gates.
//!
//! Shard scanning fans out across worker threads (shards are independent);
//! records are pre-filtered with a cheap `"corpusid":` scan so only the
//! ~parent-set fraction of records pays a full JSON parse. Parents missing
//! from the papers shards get documented defaults (year 0, counts 0,
//! flags false, empty strings) and are counted in the result message.

use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use serde_json::Value;

use super::gen_passages::{enumerate_shards, open_shard, select_shards};
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use veks_core::formats::passage_metadata::{MetadataRow, MetadataTableWriter};
use veks_core::formats::passage_table::{read_i64_column, read_text_column};

/// Pipeline command: join papers metadata onto passages.
pub struct GeneratePassageMetadataOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GeneratePassageMetadataOp)
}

/// Parent-level fields extracted from one papers record.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ParentMeta {
    year: i32,
    citationcount: i64,
    isopenaccess: bool,
    field: String,
    venue: String,
}

impl CommandOp for GeneratePassageMetadataOp {
    fn command_path(&self) -> &str {
        "generate passage-metadata"
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
            summary: "Join S2AG papers metadata onto a passage table".into(),
            body: format!(
                r#"# generate passage-metadata

Join S2AG papers metadata onto a passage table.

## Description

Scans S2AG `papers` dataset shards (JSONL, optionally `.gz`) for the
parent corpusids of `passages`, then writes `metadata.parquet` with one
row per passage **in passage row order** — the M-facet raw input for a
predicated (PVS) dataset build. Columns: corpusid, section (passage
level), year, citationcount, isopenaccess, field (primary
s2fieldsofstudy category), venue. Parents absent from the papers shards
get defaults (0 / false / "") and are counted in the result.

## Role in dataset pipelines

Row i of the output describes row i of `passages.parquet` and therefore
row i of the embedded vectors — the ordinal contract `verify alignment`
asserts. Hand the output to `veks prepare bootstrap --metadata` (or the
wizard) and the predicated facets (M, P, R, filtered ground truth) are
generated from it.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source = match options.require("source") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let passages = match options.require("passages") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let output = match options.require("output") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let files_selector = options.get("files").unwrap_or("all").to_string();
        let threads: usize = match options.parse_or("threads", 0) {
            Ok(0) => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };

        // ── Passage row order: corpusid + section per row ────────────────
        let corpusids = match read_i64_column(&passages, "corpusid") {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let sections = match read_text_column(&passages, "section") {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        if corpusids.len() != sections.len() || corpusids.is_empty() {
            return error_result(
                format!("bad passage table {} ({} rows)", passages.display(), corpusids.len()),
                start,
            );
        }
        let needed: HashSet<i64> = corpusids.iter().copied().collect();

        // ── Scan papers shards in parallel for the needed parents ────────
        let shards = match enumerate_shards(&source)
            .and_then(|s| select_shards(s, &files_selector))
        {
            Ok(s) if !s.is_empty() => s,
            Ok(_) => return error_result(format!("no shards at {}", source.display()), start),
            Err(e) => return error_result(e, start),
        };
        ctx.ui.log(&format!(
            "scanning {} papers shard(s) for {} parent(s) with {} thread(s)",
            shards.len(),
            needed.len(),
            threads.min(shards.len())
        ));
        let pb = ctx.ui.bar_with_unit(shards.len() as u64, "scan", "shard");
        let next = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let found: Mutex<HashMap<i64, ParentMeta>> = Mutex::new(HashMap::new());
        let scan_err: Mutex<Option<String>> = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..threads.min(shards.len()) {
                scope.spawn(|| {
                    let mut local: HashMap<i64, ParentMeta> = HashMap::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= shards.len() {
                            break;
                        }
                        if let Err(e) = scan_shard(&shards[i], &needed, &mut local) {
                            *scan_err.lock().unwrap() = Some(e);
                            break;
                        }
                        done.fetch_add(1, Ordering::Relaxed);
                    }
                    found.lock().unwrap().extend(local);
                });
            }
            // Main thread drives the progress bar while workers scan.
            loop {
                let d = done.load(Ordering::Relaxed);
                pb.set_position(d as u64);
                if d + usize::from(scan_err.lock().unwrap().is_some()) >= shards.len()
                    || next.load(Ordering::Relaxed) > shards.len() + threads
                {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        });
        pb.finish();
        if let Some(e) = scan_err.into_inner().unwrap() {
            return error_result(e, start);
        }
        let found = found.into_inner().unwrap();

        // ── Broadcast parent fields to passages in row order ─────────────
        let default_meta = ParentMeta::default();
        let missing_parents: usize = needed.iter().filter(|id| !found.contains_key(id)).count();
        let mut writer = match MetadataTableWriter::create(&output) {
            Ok(w) => w,
            Err(e) => return error_result(e, start),
        };
        for (id, section) in corpusids.iter().zip(sections) {
            let m = found.get(id).unwrap_or(&default_meta);
            let row = MetadataRow {
                corpusid: *id,
                section,
                year: m.year,
                citationcount: m.citationcount,
                isopenaccess: m.isopenaccess,
                field: m.field.clone(),
                venue: m.venue.clone(),
            };
            if let Err(e) = writer.push(&row) {
                return error_result(e, start);
            }
        }
        let rows = match writer.finish() {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };

        CommandResult {
            status: Status::Ok,
            message: format!(
                "joined metadata for {} passage(s) from {} parent(s) ({} parent(s) missing from papers, defaulted) to {}",
                rows,
                needed.len(),
                missing_parents,
                output.display()
            ),
            produced: vec![output],
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
                description: "S2AG papers JSONL(.gz) shard file or directory of shards".to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "passages".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "passages.parquet whose row order the output mirrors".to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output metadata.parquet (one row per passage, row-aligned)".to_string(),
                extended_description: None,
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "files".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("all".to_string()),
                description: "Shard selection over lexically-sorted basenames: first:N (strict), \
                              a glob, or all"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "threads".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Shard-scan worker threads (0 = all cores); does not affect output bytes"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["source", "passages"],
            &["output"],
        )
    }
}

/// Scan one papers shard, collecting fields for records whose corpusid is
/// in `needed`. A cheap substring prefilter finds the corpusid before any
/// full JSON parse; only matching records (the ~parent-set fraction) are
/// parsed.
fn scan_shard(
    shard: &Path,
    needed: &HashSet<i64>,
    out: &mut HashMap<i64, ParentMeta>,
) -> Result<(), String> {
    let reader = open_shard(shard)?;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read error in {}: {}", shard.display(), e))?;
        let Some(id) = prefilter_corpusid(&line) else { continue };
        if !needed.contains(&id) {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| format!("bad JSON in {}: {}", shard.display(), e))?;
        out.insert(id, extract_paper_meta(&v));
    }
    Ok(())
}

/// Extract the corpusid from a papers JSONL line without a full parse.
/// Returns None when the marker is absent or malformed (caller skips).
fn prefilter_corpusid(line: &str) -> Option<i64> {
    let pos = line.find("\"corpusid\"")?;
    let rest = line[pos + "\"corpusid\"".len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Pull the metadata fields this pipeline records from one papers record.
/// Absent or differently-typed fields fall back to defaults — the papers
/// corpus has real nulls in all of these.
fn extract_paper_meta(v: &Value) -> ParentMeta {
    ParentMeta {
        year: v.get("year").and_then(Value::as_i64).unwrap_or(0) as i32,
        citationcount: v.get("citationcount").and_then(Value::as_i64).unwrap_or(0),
        isopenaccess: v.get("isopenaccess").and_then(Value::as_bool).unwrap_or(false),
        field: v
            .get("s2fieldsofstudy")
            .and_then(|f| f.get(0))
            .and_then(|f| f.get("category"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        venue: v.get("venue").and_then(Value::as_str).unwrap_or("").to_string(),
    }
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
            progress: crate::pipeline::progress::ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    #[test]
    fn prefilter_finds_corpusid() {
        assert_eq!(prefilter_corpusid(r#"{"corpusid": 42, "x": 1}"#), Some(42));
        assert_eq!(prefilter_corpusid(r#"{"corpusid":7,"y":2}"#), Some(7));
        assert_eq!(prefilter_corpusid(r#"{"other": 1}"#), None);
        assert_eq!(prefilter_corpusid(r#"{"corpusid": "nope"}"#), None);
    }

    #[test]
    fn extract_handles_nulls_and_shapes() {
        let full: Value = serde_json::json!({
            "corpusid": 9, "year": 2021, "citationcount": 14,
            "isopenaccess": true, "venue": "NeurIPS",
            "s2fieldsofstudy": [{"category": "Computer Science", "source": "s2"}]
        });
        assert_eq!(
            extract_paper_meta(&full),
            ParentMeta {
                year: 2021,
                citationcount: 14,
                isopenaccess: true,
                field: "Computer Science".into(),
                venue: "NeurIPS".into(),
            }
        );
        let nulls: Value = serde_json::json!({
            "corpusid": 9, "year": null, "citationcount": null,
            "isopenaccess": null, "venue": null, "s2fieldsofstudy": null
        });
        assert_eq!(extract_paper_meta(&nulls), ParentMeta::default());
    }

    #[test]
    fn join_writes_row_aligned_metadata() {
        use veks_core::formats::passage_table::{PassageRow, PassageTableWriter};
        let tmp = tempfile::tempdir().unwrap();
        // Passage table: 3 passages over 2 parents (7, then 9).
        let passages = tmp.path().join("passages.parquet");
        let mut w = PassageTableWriter::create(&passages).unwrap();
        for (cid, sec, ord) in [(7i64, "A", 0i32), (7, "B", 0), (9, "A", 0)] {
            w.push(&PassageRow {
                corpusid: cid,
                section: sec.into(),
                ordinal: ord,
                char_start: 0,
                char_end: 1,
                text: "t".into(),
            })
            .unwrap();
        }
        w.finish().unwrap();
        // Papers shard: parent 7 present, parent 9 missing.
        let shard = tmp.path().join("papers-0.jsonl");
        std::fs::write(
            &shard,
            r#"{"corpusid": 7, "year": 1999, "citationcount": 3, "isopenaccess": true, "venue": "V", "s2fieldsofstudy": [{"category": "Biology"}]}
{"corpusid": 8, "year": 2000}
"#,
        )
        .unwrap();

        let mut op = GeneratePassageMetadataOp;
        let mut options = Options::new();
        options.set("source", shard.to_string_lossy().to_string());
        options.set("passages", passages.to_string_lossy().to_string());
        options.set(
            "output",
            tmp.path().join("metadata.parquet").to_string_lossy().to_string(),
        );
        let mut ctx = test_ctx(tmp.path());
        let result = op.execute(&options, &mut ctx);
        assert_eq!(result.status, Status::Ok, "{}", result.message);
        assert!(result.message.contains("3 passage(s)"), "{}", result.message);
        assert!(result.message.contains("1 parent(s) missing"), "{}", result.message);

        let years = read_i64_column(&tmp.path().join("metadata.parquet"), "citationcount").unwrap();
        assert_eq!(years, vec![3, 3, 0]); // parent 7 broadcast, parent 9 defaulted
        let sections =
            read_text_column(&tmp.path().join("metadata.parquet"), "section").unwrap();
        assert_eq!(sections, vec!["A", "B", "A"]);
    }
}
