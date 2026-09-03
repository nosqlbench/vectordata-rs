// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `transform enrich-metadata` — joins the derived predicate columns
//! onto a passage-level metadata table, in source order.
//!
//! The metadata a corpus arrives with is paper-level: every passage of
//! a paper carries the paper's fields, so a predicate on any of them
//! draws from tens of millions of papers rather than hundreds of
//! millions of passages, in blocks. This command adds the columns that
//! give a predicate set somewhere else to stand:
//!
//! | column | from | family |
//! |---|---|---|
//! | `topic_l1` … `topic_lN` | topic assignment, code → label | topical |
//! | `section_class` | the section heading, through an ordered prefix table | structural |
//! | `passage_position` | the passage's index within its paper ÷ `passage_count`, 0–99 | structural |
//! | `word_count` | whitespace tokens of the passage text | structural |
//! | `citation_percentile` | rank of the paper's citations within its year, 0–99 | bibliographic |
//! | `sample_bucket` | a seeded hash of `(corpusid, source row)` — the passage, not the paper | control |
//!
//! It is not a pure row-wise map. `citation_percentile` needs the
//! per-year distribution over **papers** first — over papers, not
//! passages, or a paper with many passages would dominate its own
//! percentile — which is one columnar read of the metadata taking the
//! first row of each paper. `passage_position` needs each paper's
//! passage count, which the parent table already carries. Everything
//! else is row-local, so the map itself runs in parallel over the
//! passage table's row groups, with one consumer writing the enriched
//! table in order.
//!
//! Coded values are stored as their **label**, not a code: the value a
//! predicate compares against is the value on the wire, so nothing
//! stands between the stored data and the query a person would write.
//!
//! See the topic-stratified predicate SRD, §4, §6.3, §9.4 and §10.3.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;

use arrow::array::{
    Array, ArrayRef, Int8Array, Int16Array, Int16Builder, Int32Array, Int32Builder, Int64Array,
    LargeStringArray, RecordBatch, StringArray, StringBuilder, UInt8Array, UInt16Array,
    UInt32Array, UInt64Array,
};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use indexmap::IndexMap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use slabtastic::{SlabWriter, WriterConfig};

use vectordata::io::{VectorReader, XvecReader};
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::passage_table::{
    ParentRow, StagedParquetWriter, parquet_row_count, parquet_row_groups, read_columns_row_group,
    read_parents, read_row_range,
};

use crate::pipeline::command::{
    ArtifactManifest, ArtifactState, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, ResourceDesc, Status, StreamContext, render_options_table,
};

use super::compute_topic_labels::read_labels;
use super::compute_topics::workspace_of;
use super::source_window::resolve_path;

/// `transform enrich-metadata`.
pub struct EnrichMetadataOp;

/// Factory used by the pipeline command registry.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(EnrichMetadataOp)
}

/// Default modulus of `sample_bucket`: 2²⁴, so a threshold predicate
/// reaches 6×10⁻⁸, below the design's finest decade.
pub const DEFAULT_BUCKETS: u32 = 16_777_216;

/// Workers in the map pass. Each holds one row group of the passage
/// table (text included) plus the aligned metadata rows, so this is
/// bounded independently of the governor's thread count.
const MAX_MAP_WORKERS: usize = 32;

/// The section classes, in the order of the table below.
pub const SECTION_CLASSES: [&str; 8] = [
    "introduction",
    "background",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "other",
];

/// The ordered prefix table. A normalised heading takes the class of
/// the first entry it starts with; order is significant because
/// `results and discussion` must not match `result` first. Anything
/// unmatched is `other`.
pub const SECTION_RULES: &[(&str, &str)] = &[
    ("results and discussion", "discussion"),
    ("result and discussion", "discussion"),
    ("discussion and conclusion", "conclusion"),
    ("summary and conclusion", "conclusion"),
    ("conclusion", "conclusion"),
    ("concluding", "conclusion"),
    ("summary", "conclusion"),
    ("outlook", "conclusion"),
    ("future", "conclusion"),
    ("introduction", "introduction"),
    ("intro", "introduction"),
    ("overview", "introduction"),
    ("motivation", "introduction"),
    ("abstract", "introduction"),
    ("background", "background"),
    ("related", "background"),
    ("literature", "background"),
    ("prior", "background"),
    ("previous work", "background"),
    ("preliminar", "background"),
    ("theor", "background"),
    ("state of the art", "background"),
    ("review", "background"),
    ("experimental setup", "methods"),
    ("experimental procedure", "methods"),
    ("experimental section", "methods"),
    ("experimental detail", "methods"),
    ("experimental method", "methods"),
    ("method", "methods"),
    ("material", "methods"),
    ("procedure", "methods"),
    ("setup", "methods"),
    ("implementation", "methods"),
    ("approach", "methods"),
    ("model", "methods"),
    ("data", "methods"),
    ("system", "methods"),
    ("design", "methods"),
    ("algorithm", "methods"),
    ("proposed", "methods"),
    ("framework", "methods"),
    ("architecture", "methods"),
    ("measurement", "methods"),
    ("simulation", "methods"),
    ("calculation", "methods"),
    ("participant", "methods"),
    ("patient", "methods"),
    ("sample", "methods"),
    ("study design", "methods"),
    ("statistic", "methods"),
    ("protocol", "methods"),
    ("experiment", "results"),
    ("evaluation", "results"),
    ("result", "results"),
    ("finding", "results"),
    ("performance", "results"),
    ("analysis", "results"),
    ("comparison", "results"),
    ("case stud", "results"),
    ("application", "results"),
    ("numerical", "results"),
    ("discussion", "discussion"),
    ("limitation", "discussion"),
    ("implication", "discussion"),
    ("reference", "references"),
    ("bibliograph", "references"),
    ("acknowledg", "other"),
    ("appendix", "other"),
    ("supplement", "other"),
    ("funding", "other"),
    ("conflict", "other"),
    ("author contribution", "other"),
    ("ethic", "other"),
    ("declaration", "other"),
    ("availability", "other"),
    ("abbreviation", "other"),
    ("nomenclature", "other"),
    ("keyword", "other"),
    ("competing", "other"),
];

// ---------------------------------------------------------------------------
// Derivations
// ---------------------------------------------------------------------------

/// Lower-case, strip leading numbering (`3.`, `3.2.1`, `iii.`, `A)`,
/// `(b)`), strip trailing punctuation, collapse whitespace.
pub fn normalize_heading(raw: &str) -> String {
    let lower = raw.trim().to_lowercase();
    let mut s = lower.as_str();
    loop {
        let stripped = strip_one_numbering(s);
        if stripped.len() == s.len() {
            break;
        }
        s = stripped;
    }
    let s = s.trim_end_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Remove one leading numbering token — digits or roman numerals with
/// optional dots between parts, a single letter, or a parenthesised
/// form — followed by a separator or a space.
fn strip_one_numbering(s: &str) -> &str {
    let s = s.trim_start();
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return s;
    }
    let mut i = 0;
    let open = bytes[0] == b'(';
    if open {
        i = 1;
    }
    let start = i;
    // Numbering body: digits, roman-numeral letters, dots, or one
    // ASCII letter.
    while i < bytes.len()
        && (bytes[i].is_ascii_digit() || bytes[i] == b'.' || b"ivxlc".contains(&bytes[i]))
    {
        i += 1;
    }
    let body_len = i - start;
    let single_letter = body_len == 0
        && i < bytes.len()
        && bytes[i].is_ascii_alphabetic()
        && (i + 1 == bytes.len() || !bytes[i + 1].is_ascii_alphabetic());
    if single_letter {
        i += 1;
    } else if body_len == 0 || body_len > 12 {
        return s;
    }
    // A roman/digit body must not be a word: it needs a separator or
    // to end the string.
    let mut end = i;
    if open {
        if end < bytes.len() && bytes[end] == b')' {
            end += 1;
        } else {
            return s;
        }
    } else if end < bytes.len() && (bytes[end] == b'.' || bytes[end] == b')' || bytes[end] == b':')
    {
        end += 1;
    }
    if end == bytes.len() {
        return "";
    }
    if bytes[end] == b' ' || bytes[end] == b'\t' {
        // A bare alphabetic token that happens to be roman numerals
        // ("civil", "mix") is a word, not a number: require at least
        // one digit or a separator for letter-only bodies.
        let body = &s[start..i];
        let letters_only = body.bytes().all(|b| b.is_ascii_alphabetic());
        if letters_only && !single_letter && end <= i {
            return s;
        }
        return &s[end..];
    }
    s
}

/// The section class of a raw heading.
pub fn section_class(raw: &str) -> &'static str {
    let h = normalize_heading(raw);
    if h.is_empty() {
        return "other";
    }
    for (prefix, class) in SECTION_RULES {
        if h.starts_with(prefix) {
            return class;
        }
    }
    "other"
}

/// `sample_bucket`: splitmix64 over `(seed, corpusid, row)`, reduced
/// modulo `buckets`, where `row` is the passage's source row — its
/// ordinal in `passages.parquet`, unique per passage. Keyed on the
/// passage, not the paper, so the control family is free of paper
/// blocking (TS-80). The upstream `ordinal` column is **not** the
/// passage's identity: it restarts at zero in every section of a
/// paper, so keying on `(corpusid, ordinal)` — as this hash once did —
/// gave the passages of a paper's sections the same bucket and put
/// paper blocking back into the one family built to be free of it
/// (TS-174).
pub fn sample_bucket(seed: u64, corpusid: i64, row: u64, buckets: u32) -> u32 {
    let mut z = seed
        ^ (corpusid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ row.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z % buckets.max(1) as u64) as u32
}

/// `word_count`: whitespace-delimited tokens, capped at `i16::MAX`.
pub fn word_count(text: &str) -> i16 {
    text.split_whitespace().count().min(i16::MAX as usize) as i16
}

/// `passage_position`: `⌊100 · ordinal / passage_count⌋`, so 0–99. A
/// single-passage paper yields 0.
pub fn passage_position(index_in_paper: i64, passage_count: i32) -> i16 {
    if passage_count <= 0 || index_in_paper < 0 {
        return 0;
    }
    ((100 * index_in_paper) / passage_count as i64).clamp(0, 99) as i16
}

/// Per-year citation ranks over papers: for each year, the distinct
/// citation counts ascending with the percentile each maps to, so a
/// lookup is a binary search. Ties take the midpoint rank, so the
/// large mass of zero-citation papers maps to one value.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct CitationRanks {
    by_year: HashMap<i32, (Vec<i64>, Vec<i16>)>,
}

impl CitationRanks {
    /// Build from `(year, citationcount)` per paper.
    pub fn build(papers: &[(i32, i64)]) -> Self {
        let mut grouped: HashMap<i32, Vec<i64>> = HashMap::new();
        for (year, count) in papers {
            grouped.entry(*year).or_default().push(*count);
        }
        let mut by_year = HashMap::with_capacity(grouped.len());
        for (year, mut counts) in grouped {
            counts.sort_unstable();
            let n = counts.len() as f64;
            let mut values = Vec::new();
            let mut pcts = Vec::new();
            let mut i = 0;
            while i < counts.len() {
                let c = counts[i];
                let mut j = i;
                while j < counts.len() && counts[j] == c {
                    j += 1;
                }
                let below = i as f64;
                let ties = (j - i) as f64;
                let midpoint = below + (ties - 1.0) / 2.0;
                let pct = ((100.0 * midpoint / n).floor() as i64).clamp(0, 99) as i16;
                values.push(c);
                pcts.push(pct);
                i = j;
            }
            by_year.insert(year, (values, pcts));
        }
        CitationRanks { by_year }
    }

    /// Percentile of `count` within `year`; a year never seen (which
    /// cannot happen for a paper the ranks were built from) maps to 0.
    pub fn percentile(&self, year: i32, count: i64) -> i16 {
        let Some((values, pcts)) = self.by_year.get(&year) else {
            return 0;
        };
        match values.binary_search(&count) {
            Ok(i) => pcts[i],
            // A count between two observed ones: the rank of the next
            // lower value, which is what its position implies.
            Err(0) => 0,
            Err(i) => pcts[i - 1],
        }
    }

    pub fn years(&self) -> usize {
        self.by_year.len()
    }
}

// ---------------------------------------------------------------------------
// Column access
// ---------------------------------------------------------------------------

/// An integer at row `i` of any integer-typed arrow column.
fn int_at(col: &ArrayRef, i: usize) -> Option<i64> {
    macro_rules! try_int {
        ($t:ty) => {
            if let Some(a) = col.as_any().downcast_ref::<$t>() {
                return if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i) as i64)
                };
            }
        };
    }
    try_int!(Int64Array);
    try_int!(Int32Array);
    try_int!(Int16Array);
    try_int!(Int8Array);
    try_int!(UInt64Array);
    try_int!(UInt32Array);
    try_int!(UInt16Array);
    try_int!(UInt8Array);
    None
}

/// A string at row `i` of a utf8 or large-utf8 column.
fn str_at(col: &ArrayRef, i: usize) -> Option<&str> {
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return if a.is_null(i) { None } else { Some(a.value(i)) };
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return if a.is_null(i) { None } else { Some(a.value(i)) };
    }
    None
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, String> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| format!("column `{}` is missing", name))?;
    Ok(batch.column(idx))
}

// ---------------------------------------------------------------------------
// Labels and the paper table
// ---------------------------------------------------------------------------

/// Code → label per level, from the label slab or positional.
struct LabelTable {
    per_level: Vec<Vec<String>>,
    positional: bool,
}

impl LabelTable {
    fn from_slab(path: &Path, depth: usize) -> Result<Self, String> {
        let mut per_level: Vec<Vec<String>> = vec![Vec::new(); depth];
        for (level, code, label) in read_labels(path)? {
            if level == 0 || level > depth {
                return Err(format!(
                    "label slab names level {} but the assignments carry {} levels",
                    level, depth
                ));
            }
            let v = &mut per_level[level - 1];
            if v.len() <= code {
                v.resize(code + 1, String::new());
            }
            v[code] = label;
        }
        Ok(LabelTable {
            per_level,
            positional: false,
        })
    }

    fn positional(depth: usize) -> Self {
        LabelTable {
            per_level: vec![Vec::new(); depth],
            positional: true,
        }
    }

    fn label(&self, level: usize, code: usize) -> String {
        match self.per_level.get(level).and_then(|v| v.get(code)) {
            Some(l) if !l.is_empty() => l.clone(),
            _ => format!("l{}-{:05}", level + 1, code),
        }
    }
}

/// The parent table as a sorted `row_start` index, for the paper of any
/// row by binary search and then a pointer walk.
struct Papers {
    rows: Vec<ParentRow>,
}

impl Papers {
    fn load(path: &Path) -> Result<Self, String> {
        let mut rows = read_parents(path)?;
        rows.sort_by_key(|p| p.row_start);
        Ok(Papers { rows })
    }

    /// Index of the paper holding global row `row`.
    fn paper_at(&self, row: u64) -> Option<usize> {
        let i = self.rows.partition_point(|p| (p.row_start as u64) <= row);
        if i == 0 {
            return None;
        }
        let p = &self.rows[i - 1];
        if row < p.row_start as u64 + p.passage_count.max(0) as u64 {
            Some(i - 1)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// The map pass
// ---------------------------------------------------------------------------

/// Column names the source metadata is read by.
#[derive(Debug, Clone, PartialEq)]
struct SourceColumns {
    paper: String,
    section: String,
    year: String,
    citations: String,
}

/// What one row group of the map pass produced.
struct MappedGroup {
    index: usize,
    batch: Option<RecordBatch>,
    headings: HashMap<String, (&'static str, u64)>,
    rows: u64,
    error: Option<String>,
}

/// Everything the workers share.
struct MapContext {
    metadata: PathBuf,
    passages: PathBuf,
    assignments: PathBuf,
    columns: SourceColumns,
    labels: LabelTable,
    papers: Papers,
    ranks: CitationRanks,
    seed: u64,
    buckets: u32,
    depth: usize,
    output_schema: SchemaRef,
}

fn new_fields(depth: usize) -> Vec<Field> {
    let mut fields = Vec::with_capacity(depth + 5);
    for l in 0..depth {
        fields.push(Field::new(
            format!("topic_l{}", l + 1),
            DataType::Utf8,
            false,
        ));
    }
    fields.push(Field::new("section_class", DataType::Utf8, false));
    fields.push(Field::new("citation_percentile", DataType::Int16, false));
    fields.push(Field::new("passage_position", DataType::Int16, false));
    fields.push(Field::new("word_count", DataType::Int16, false));
    fields.push(Field::new("sample_bucket", DataType::Int32, false));
    fields
}

fn map_group(ctx: &MapContext, index: usize, start: u64, rows: u64) -> MappedGroup {
    match map_group_inner(ctx, index, start, rows) {
        Ok((batch, headings)) => MappedGroup {
            index,
            batch: Some(batch),
            headings,
            rows,
            error: None,
        },
        Err(e) => MappedGroup {
            index,
            batch: None,
            headings: HashMap::new(),
            rows,
            error: Some(e),
        },
    }
}

fn map_group_inner(
    ctx: &MapContext,
    index: usize,
    start: u64,
    rows: u64,
) -> Result<(RecordBatch, HashMap<String, (&'static str, u64)>), String> {
    let end = start + rows;
    let (pstart, pbatches) = read_columns_row_group(&ctx.passages, &["text"], index)?;
    if pstart != start {
        return Err(format!(
            "passages row group {} starts at {} but was planned at {}",
            index, pstart, start
        ));
    }
    let pschema = pbatches
        .first()
        .map(|b| b.schema())
        .ok_or_else(|| format!("passages row group {} is empty", index))?;
    let passages =
        concat_batches(&pschema, &pbatches).map_err(|e| format!("passages concat: {}", e))?;
    let mbatches = read_row_range(&ctx.metadata, start, end)?;
    let mschema = mbatches
        .first()
        .map(|b| b.schema())
        .ok_or_else(|| format!("metadata rows {}..{} are empty", start, end))?;
    let metadata =
        concat_batches(&mschema, &mbatches).map_err(|e| format!("metadata concat: {}", e))?;
    if metadata.num_rows() != passages.num_rows() {
        return Err(format!(
            "rows {}..{}: {} metadata rows against {} passage rows",
            start,
            end,
            metadata.num_rows(),
            passages.num_rows()
        ));
    }
    let n = passages.num_rows();
    let assignments = XvecReader::<u16>::open_path(&ctx.assignments)
        .map_err(|e| format!("failed to open assignments: {}", e))?;

    let text_col = column(&passages, "text")?;
    let paper_col = column(&metadata, &ctx.columns.paper)?;
    let section_col = column(&metadata, &ctx.columns.section)?;
    let year_col = column(&metadata, &ctx.columns.year)?;
    let cit_col = column(&metadata, &ctx.columns.citations)?;

    let mut topic_builders: Vec<StringBuilder> =
        (0..ctx.depth).map(|_| StringBuilder::new()).collect();
    let mut section_b = StringBuilder::new();
    let mut pct_b = Int16Builder::with_capacity(n);
    let mut pos_b = Int16Builder::with_capacity(n);
    let mut words_b = Int16Builder::with_capacity(n);
    let mut bucket_b = Int32Builder::with_capacity(n);
    let mut headings: HashMap<String, (&'static str, u64)> = HashMap::new();
    let mut section_cache: HashMap<String, &'static str> = HashMap::new();

    let mut paper_ix = ctx.papers.paper_at(start);
    for i in 0..n {
        let row = start + i as u64;
        // Advance the paper pointer; rows are grouped by paper.
        while let Some(p) = paper_ix {
            let pr = &ctx.papers.rows[p];
            if row >= pr.row_start as u64 + pr.passage_count.max(0) as u64 {
                paper_ix = if p + 1 < ctx.papers.rows.len() {
                    Some(p + 1)
                } else {
                    None
                };
                if paper_ix.is_some_and(|q| (ctx.papers.rows[q].row_start as u64) > row) {
                    paper_ix = None;
                }
            } else {
                break;
            }
        }
        if paper_ix.is_none() {
            paper_ix = ctx.papers.paper_at(row);
        }
        let passage_count = paper_ix
            .map(|p| ctx.papers.rows[p].passage_count)
            .unwrap_or(0);
        // The passage's index within its paper comes from the paper's
        // row span, not from the upstream `ordinal`, which is
        // section-local (TS-174).
        let index_in_paper = paper_ix
            .map(|p| row as i64 - ctx.papers.rows[p].row_start)
            .unwrap_or(0);

        let codes = assignments
            .get(row as usize)
            .map_err(|e| format!("failed to read assignment {}: {}", row, e))?;
        for (l, b) in topic_builders.iter_mut().enumerate() {
            b.append_value(ctx.labels.label(l, codes[l] as usize));
        }
        let heading = str_at(section_col, i).unwrap_or("");
        let class = match section_cache.get(heading) {
            Some(c) => *c,
            None => {
                let c = section_class(heading);
                section_cache.insert(heading.to_string(), c);
                c
            }
        };
        let entry = headings.entry(heading.to_string()).or_insert((class, 0));
        entry.1 += 1;
        section_b.append_value(class);
        let year = int_at(year_col, i).unwrap_or(0) as i32;
        let cits = int_at(cit_col, i).unwrap_or(0);
        pct_b.append_value(ctx.ranks.percentile(year, cits));
        pos_b.append_value(passage_position(index_in_paper, passage_count));
        words_b.append_value(word_count(str_at(text_col, i).unwrap_or("")));
        let corpusid = int_at(paper_col, i).unwrap_or(0);
        bucket_b.append_value(sample_bucket(ctx.seed, corpusid, row, ctx.buckets) as i32);
    }

    let mut columns: Vec<ArrayRef> = metadata.columns().to_vec();
    for mut b in topic_builders {
        columns.push(Arc::new(b.finish()));
    }
    columns.push(Arc::new(section_b.finish()));
    columns.push(Arc::new(pct_b.finish()));
    columns.push(Arc::new(pos_b.finish()));
    columns.push(Arc::new(words_b.finish()));
    columns.push(Arc::new(bucket_b.finish()));
    let batch = RecordBatch::try_new(ctx.output_schema.clone(), columns)
        .map_err(|e| format!("failed to assemble enriched rows {}..{}: {}", start, end, e))?;
    Ok((batch, headings))
}

/// Paper-level `(year, citationcount)` from the metadata: the first row
/// of every paper, by the parent table's `row_start`.
fn paper_citations(
    metadata: &Path,
    columns: &SourceColumns,
    papers: &Papers,
    groups: &[(u64, u64)],
    mut progress: impl FnMut(usize),
) -> Result<Vec<(i32, i64)>, String> {
    let mut out = Vec::with_capacity(papers.rows.len());
    let mut next_paper = 0usize;
    for (g, &(start, rows)) in groups.iter().enumerate() {
        let (_, batches) =
            read_columns_row_group(metadata, &[&columns.year, &columns.citations], g)?;
        let mut offset = 0u64;
        for batch in &batches {
            let year_col = column(batch, &columns.year)?;
            let cit_col = column(batch, &columns.citations)?;
            let bstart = start + offset;
            let bend = bstart + batch.num_rows() as u64;
            while next_paper < papers.rows.len() {
                let rs = papers.rows[next_paper].row_start as u64;
                if rs >= bend {
                    break;
                }
                if rs >= bstart {
                    let i = (rs - bstart) as usize;
                    out.push((
                        int_at(year_col, i).unwrap_or(0) as i32,
                        int_at(cit_col, i).unwrap_or(0),
                    ));
                }
                next_paper += 1;
            }
            offset += batch.num_rows() as u64;
        }
        if offset != rows {
            return Err(format!(
                "metadata row group {} yielded {} rows, expected {}",
                g, offset, rows
            ));
        }
        progress(g + 1);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnrichReport {
    pub schema_version: u32,
    pub rows: u64,
    pub papers: usize,
    pub years: usize,
    pub levels: usize,
    pub labels: String,
    pub positional_labels: bool,
    pub distinct_headings: usize,
    pub headings_other_share: f64,
    pub buckets: u32,
    pub seed: u64,
    pub seconds: f64,
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(
    name: &str,
    type_name: &str,
    required: bool,
    default: Option<&str>,
    desc: &str,
    role: OptionRole,
) -> OptionDesc {
    OptionDesc {
        name: name.into(),
        type_name: type_name.into(),
        required,
        default: default.map(str::to_string),
        description: desc.into(),
        extended_description: None,
        role,
    }
}

fn output_schema(source: &Schema, depth: usize) -> Result<SchemaRef, String> {
    let mut fields: Vec<Field> = source.fields().iter().map(|f| f.as_ref().clone()).collect();
    for f in new_fields(depth) {
        if source.index_of(f.name()).is_ok() {
            return Err(format!(
                "the metadata already has a column named `{}`",
                f.name()
            ));
        }
        fields.push(f);
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn read_schema(path: &Path) -> Result<SchemaRef, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("failed to read parquet {}: {}", path.display(), e))?;
    Ok(builder.schema().clone())
}

fn report_path(options: &Options, output: &Path, workspace: &Path) -> PathBuf {
    match options.get("report") {
        Some(s) => resolve_path(s, workspace),
        None => output.with_extension("json"),
    }
}

fn write_section_map(
    path: &Path,
    headings: &HashMap<String, (&'static str, u64)>,
) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {}", parent.display(), e))?;
    }
    let mut entries: Vec<(&String, &(&'static str, u64))> = headings.iter().collect();
    entries.sort_by(|a, b| b.1.1.cmp(&a.1.1).then_with(|| a.0.cmp(b.0)));
    let config =
        WriterConfig::new(512, 4096, u32::MAX, false).map_err(|e| format!("slab config: {}", e))?;
    let mut w = SlabWriter::new(path, config)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    for (heading, (class, count)) in entries {
        let mut fields = IndexMap::new();
        fields.insert("heading".to_string(), MValue::Text(heading.clone()));
        fields.insert(
            "section_class".to_string(),
            MValue::Text((*class).to_string()),
        );
        fields.insert("count".to_string(), MValue::Int(*count as i64));
        w.add_record(&anode::encode(&ANode::MNode(MNode { fields })))
            .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;
    }
    w.finish()
        .map_err(|e| format!("failed to finish {}: {}", path.display(), e))?;
    Ok(())
}

impl CommandOp for EnrichMetadataOp {
    fn command_path(&self) -> &str {
        "transform enrich-metadata"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_TRANSFORM
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary:
                "Join topic, structural, bibliographic and control columns onto passage metadata"
                    .into(),
            body: format!(
                r#"# transform enrich-metadata

Joins derived columns onto a passage-level metadata table, producing an
enriched table in the same row order: `topic_l1`…`topic_lN` (labels
from the topic assignment), `section_class` (the heading through an
ordered prefix table), `citation_percentile` (rank within publication
year over papers, ties at the midpoint), `passage_position`
(the passage's index within its paper ÷ `passage_count`, 0–99), `word_count` (whitespace tokens of
the passage text) and `sample_bucket` (a seeded hash of the passage,
modulo `buckets`).

The per-year citation distribution is computed over papers first —
one columnar read taking the first row of each paper — then the map
runs in parallel over the passage table's row groups, reading the
aligned metadata rows and writing the enriched rows in order. The
heading → class outcomes are published beside the output as an
auditable table.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt(
                "metadata",
                "Path",
                true,
                None,
                "Source metadata table (parquet), one row per passage",
                OptionRole::Input,
            ),
            opt(
                "passages",
                "Path",
                true,
                None,
                "Passage table (parquet): `text`, row-aligned with the metadata",
                OptionRole::Input,
            ),
            opt(
                "parents",
                "Path",
                true,
                None,
                "Parent table (parquet): `passage_count` and `row_start` per paper",
                OptionRole::Input,
            ),
            opt(
                "assignments",
                "Path",
                true,
                None,
                "Topic assignments (u16vecs, one code per level), row-aligned",
                OptionRole::Input,
            ),
            opt(
                "labels",
                "Path",
                false,
                None,
                "Topic label slab from `compute topic-labels`; absent means positional labels",
                OptionRole::Input,
            ),
            opt(
                "paper-column",
                "string",
                false,
                Some("corpusid"),
                "Metadata column identifying the paper",
                OptionRole::Config,
            ),
            opt(
                "section-column",
                "string",
                false,
                Some("section"),
                "Metadata column holding the section heading",
                OptionRole::Config,
            ),
            opt(
                "year-column",
                "string",
                false,
                Some("year"),
                "Metadata column holding the publication year",
                OptionRole::Config,
            ),
            opt(
                "citations-column",
                "string",
                false,
                Some("citationcount"),
                "Metadata column holding the citation count",
                OptionRole::Config,
            ),
            opt(
                "buckets",
                "int",
                false,
                Some("16777216"),
                "Modulus of `sample_bucket` (2^24 by default)",
                OptionRole::Config,
            ),
            opt(
                "seed",
                "int",
                false,
                Some("42"),
                "Hash seed for `sample_bucket`",
                OptionRole::Config,
            ),
            opt(
                "output",
                "Path",
                true,
                None,
                "Enriched metadata table (parquet)",
                OptionRole::Output,
            ),
            opt(
                "section-map-out",
                "Path",
                false,
                None,
                "Slab of every distinct heading with its class and count (default: beside `output` as section_class_map.slab)",
                OptionRole::Output,
            ),
            opt(
                "report",
                "Path",
                false,
                None,
                "Enrichment report JSON (default: beside `output` with a .json extension)",
                OptionRole::Output,
            ),
        ]
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "mem".into(),
                description:
                    "The parent table, the citation ranks, and one row group of text per worker"
                        .into(),
                adjustable: false,
            },
            ResourceDesc {
                name: "threads".into(),
                description: "Parallel row groups in the map pass".into(),
                adjustable: true,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let req = |k: &str| options.require(k).map(str::to_string);
        let (metadata_str, passages_str, parents_str, assignments_str, output_str) = match (
            req("metadata"),
            req("passages"),
            req("parents"),
            req("assignments"),
            req("output"),
        ) {
            (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e)) => (a, b, c, d, e),
            (Err(e), ..)
            | (_, Err(e), ..)
            | (_, _, Err(e), ..)
            | (_, _, _, Err(e), _)
            | (_, _, _, _, Err(e)) => return error_result(e, start),
        };
        let columns = SourceColumns {
            paper: options
                .get("paper-column")
                .unwrap_or("corpusid")
                .to_string(),
            section: options
                .get("section-column")
                .unwrap_or("section")
                .to_string(),
            year: options.get("year-column").unwrap_or("year").to_string(),
            citations: options
                .get("citations-column")
                .unwrap_or("citationcount")
                .to_string(),
        };
        let buckets = match options.parse_or::<u32>("buckets", DEFAULT_BUCKETS) {
            Ok(v) if v > 0 => v,
            Ok(_) => return error_result("buckets must be positive".into(), start),
            Err(e) => return error_result(e, start),
        };
        let seed = match options.parse_or::<u64>("seed", 42) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let ws = &ctx.workspace;
        let metadata_path = resolve_path(&metadata_str, ws);
        let passages_path = resolve_path(&passages_str, ws);
        let parents_path = resolve_path(&parents_str, ws);
        let assignments_path = resolve_path(&assignments_str, ws);
        let labels_path = options.get("labels").map(|s| resolve_path(s, ws));
        let output_path = resolve_path(&output_str, ws);
        let section_map_path = match options.get("section-map-out") {
            Some(s) => resolve_path(s, ws),
            None => output_path.with_file_name("section_class_map.slab"),
        };
        let report_path = report_path(options, &output_path, ws);

        // Inputs and their agreement.
        let groups = match parquet_row_groups(&passages_path) {
            Ok(g) => g,
            Err(e) => return error_result(e, start),
        };
        let rows: u64 = groups.iter().map(|(_, n)| *n).sum();
        let metadata_rows = match parquet_row_count(&metadata_path) {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };
        if metadata_rows != rows {
            return error_result(
                format!(
                    "metadata holds {} rows but the passage table holds {}; they must be row-aligned",
                    metadata_rows, rows
                ),
                start,
            );
        }
        let assignments = match XvecReader::<u16>::open_path(&assignments_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!(
                        "failed to open assignments {}: {}",
                        assignments_path.display(),
                        e
                    ),
                    start,
                );
            }
        };
        if assignments.count() as u64 != rows {
            return error_result(
                format!(
                    "assignments hold {} records but the metadata holds {} rows",
                    assignments.count(),
                    rows
                ),
                start,
            );
        }
        let depth = assignments.dim();
        drop(assignments);
        let labels = match &labels_path {
            Some(p) => match LabelTable::from_slab(p, depth) {
                Ok(t) => t,
                Err(e) => return error_result(e, start),
            },
            None => LabelTable::positional(depth),
        };
        let source_schema = match read_schema(&metadata_path) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        for c in [
            &columns.paper,
            &columns.section,
            &columns.year,
            &columns.citations,
        ] {
            if source_schema.index_of(c).is_err() {
                return error_result(format!("metadata has no column `{}`", c), start);
            }
        }
        let output_schema = match output_schema(&source_schema, depth) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let papers = match Papers::load(&parents_path) {
            Ok(p) => p,
            Err(e) => return error_result(e, start),
        };
        let threads = {
            let t = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
            if t == 0 {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            } else {
                t
            }
        };
        let workers = threads.clamp(1, MAX_MAP_WORKERS);
        ctx.ui.log(&format!(
            "enrich: {} rows, {} papers, {} levels ({} labels), {} passage row groups, {} workers",
            rows,
            papers.rows.len(),
            depth,
            if labels.positional {
                "positional"
            } else {
                "from slab"
            },
            groups.len(),
            workers,
        ));

        // Pass 1: the per-year citation distribution over papers.
        let metadata_groups = match parquet_row_groups(&metadata_path) {
            Ok(g) => g,
            Err(e) => return error_result(e, start),
        };
        let pb = ctx
            .ui
            .bar_with_unit(metadata_groups.len() as u64, "ranking citations", "groups");
        let paper_cits =
            match paper_citations(&metadata_path, &columns, &papers, &metadata_groups, |g| {
                pb.set_position(g as u64)
            }) {
                Ok(v) => v,
                Err(e) => return error_result(e, start),
            };
        pb.finish();
        if paper_cits.len() != papers.rows.len() {
            return error_result(
                format!(
                    "found {} paper first-rows in the metadata but the parent table has {} papers",
                    paper_cits.len(),
                    papers.rows.len()
                ),
                start,
            );
        }
        let ranks = CitationRanks::build(&paper_cits);
        ctx.ui.log(&format!(
            "enrich: citation ranks over {} papers in {} years",
            paper_cits.len(),
            ranks.years()
        ));

        // Pass 2: the map, parallel over passage row groups, written
        // in order.
        let map_ctx = Arc::new(MapContext {
            metadata: metadata_path.clone(),
            passages: passages_path,
            assignments: assignments_path,
            columns,
            labels,
            papers,
            ranks,
            seed,
            buckets,
            depth,
            output_schema: output_schema.clone(),
        });
        let mut writer = match StagedParquetWriter::create(&output_path, output_schema) {
            Ok(w) => w,
            Err(e) => return error_result(e, start),
        };
        let pb = ctx.ui.bar_with_unit(rows, "enriching", "rows");
        let mut headings: HashMap<String, (&'static str, u64)> = HashMap::new();
        let next = AtomicUsize::new(0);
        let (tx, rx) = mpsc::sync_channel::<MappedGroup>(workers);
        let outcome: Result<(), String> = std::thread::scope(|scope| {
            for _ in 0..workers {
                let tx = tx.clone();
                let next = &next;
                let groups = &groups;
                let map_ctx = Arc::clone(&map_ctx);
                scope.spawn(move || {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(start, rows)) = groups.get(index) else {
                            break;
                        };
                        if tx.send(map_group(&map_ctx, index, start, rows)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            let mut held: BTreeMap<usize, MappedGroup> = BTreeMap::new();
            let mut next_to_write = 0usize;
            let mut done = 0u64;
            for g in rx {
                held.insert(g.index, g);
                while let Some(g) = held.remove(&next_to_write) {
                    if let Some(e) = g.error {
                        return Err(e);
                    }
                    if let Some(batch) = g.batch {
                        writer.write_batch(batch)?;
                    }
                    for (h, (class, n)) in g.headings {
                        let e = headings.entry(h).or_insert((class, 0));
                        e.1 += n;
                    }
                    done += g.rows;
                    pb.set_position(done);
                    next_to_write += 1;
                }
            }
            Ok(())
        });
        if let Err(e) = outcome {
            return error_result(e, start);
        }
        pb.finish();
        let written = match writer.finish() {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };
        if written != rows {
            return error_result(format!("wrote {} rows, expected {}", written, rows), start);
        }
        let mut produced = vec![output_path.clone()];

        if let Err(e) = write_section_map(&section_map_path, &headings) {
            return error_result(e, start);
        }
        produced.push(section_map_path);

        let other: u64 = headings
            .values()
            .filter(|(c, _)| *c == "other")
            .map(|(_, n)| *n)
            .sum();
        let report = EnrichReport {
            schema_version: 1,
            rows,
            papers: map_ctx.papers.rows.len(),
            years: map_ctx.ranks.years(),
            levels: depth,
            labels: labels_path
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "positional".into()),
            positional_labels: map_ctx.labels.positional,
            distinct_headings: headings.len(),
            headings_other_share: if rows == 0 {
                0.0
            } else {
                other as f64 / rows as f64
            },
            buckets,
            seed,
            seconds: start.elapsed().as_secs_f64(),
        };
        match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&report_path, json) {
                    return error_result(
                        format!("failed to write {}: {}", report_path.display(), e),
                        start,
                    );
                }
                produced.push(report_path);
            }
            Err(e) => return error_result(format!("report serialisation failed: {}", e), start),
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} rows enriched with {} columns over {} papers in {} years; {} distinct headings, {:.1}% other",
                rows,
                depth + 5,
                report.papers,
                report.years,
                report.distinct_headings,
                100.0 * report.headings_other_share,
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    /// Complete when the enriched table holds exactly as many rows as
    /// the metadata and carries every derived column for the
    /// assignments' depth.
    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState {
        if !output.exists() {
            return ArtifactState::Absent;
        }
        let workspace = workspace_of(output, options.get("output"));
        let resolve = |k: &str| options.get(k).map(|s| resolve_path(s, &workspace));
        let (Some(metadata), Some(assignments)) = (resolve("metadata"), resolve("assignments"))
        else {
            return ArtifactState::Partial;
        };
        let Ok(expected) = parquet_row_count(&metadata) else {
            return ArtifactState::Unknown("metadata table cannot be read".into());
        };
        let Ok(depth) = XvecReader::<u16>::open_path(&assignments).map(|r| r.dim()) else {
            return ArtifactState::Unknown("assignments cannot be opened".into());
        };
        let (Ok(rows), Ok(schema)) = (parquet_row_count(output), read_schema(output)) else {
            return ArtifactState::Partial;
        };
        if rows != expected {
            return ArtifactState::Partial;
        }
        for f in new_fields(depth) {
            match schema.field_with_name(f.name()) {
                Ok(have) if have.data_type() == f.data_type() => {}
                _ => return ArtifactState::Partial,
            }
        }
        ArtifactState::Complete
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        let mut manifest = crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["metadata", "passages", "parents", "assignments", "labels"],
            &["output", "section-map-out", "report"],
        );
        if let Some(o) = options.get("output") {
            let out = PathBuf::from(o);
            if options.get("section-map-out").is_none() {
                manifest.outputs.push(
                    out.with_file_name("section_class_map.slab")
                        .to_string_lossy()
                        .to_string(),
                );
            }
            if options.get("report").is_none() {
                manifest
                    .outputs
                    .push(out.with_extension("json").to_string_lossy().to_string());
            }
        }
        manifest
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_normalise_and_classify_in_table_order() {
        assert_eq!(
            normalize_heading("3.2.1 Experimental Setup:"),
            "experimental setup"
        );
        assert_eq!(
            normalize_heading("III. RESULTS AND DISCUSSION"),
            "results and discussion"
        );
        assert_eq!(normalize_heading("(b) Materials"), "materials");
        assert_eq!(normalize_heading("A) Methods"), "methods");
        assert_eq!(normalize_heading("  Introduction  "), "introduction");
        assert_eq!(
            normalize_heading("Mixed methods"),
            "mixed methods",
            "a word of roman letters is a word"
        );
        assert_eq!(normalize_heading("2"), "");
        assert_eq!(section_class("Results and Discussion"), "discussion");
        assert_eq!(section_class("4. Results"), "results");
        assert_eq!(section_class("Experimental Results"), "results");
        assert_eq!(section_class("2.1 Materials and Methods"), "methods");
        assert_eq!(section_class("Related Work"), "background");
        assert_eq!(section_class("Conclusions and Future Work"), "conclusion");
        assert_eq!(section_class("References"), "references");
        assert_eq!(section_class(""), "other");
        assert_eq!(section_class("Acknowledgements"), "other");
        assert_eq!(section_class("Some Unusual Heading"), "other");
        assert!(
            SECTION_RULES
                .iter()
                .all(|(_, c)| SECTION_CLASSES.contains(c))
        );
    }

    #[test]
    fn citation_percentile_ranks_within_year_with_midpoint_ties() {
        let papers = vec![
            (2020, 0),
            (2020, 0),
            (2020, 0),
            (2020, 5),
            (2020, 10),
            (2021, 7),
        ];
        let ranks = CitationRanks::build(&papers);
        assert_eq!(ranks.years(), 2);
        // Three zeros occupy ranks 0..2, midpoint 1 → 100·1/5 = 20.
        assert_eq!(ranks.percentile(2020, 0), 20);
        assert_eq!(ranks.percentile(2020, 5), 60);
        assert_eq!(ranks.percentile(2020, 10), 80);
        // An unobserved count takes the rank of the next lower value;
        // below every value is 0.
        assert_eq!(ranks.percentile(2020, 7), 60);
        assert_eq!(ranks.percentile(2020, -1), 0);
        assert_eq!(ranks.percentile(2021, 7), 0, "a single paper is at rank 0");
        assert_eq!(ranks.percentile(1999, 7), 0);
    }

    #[test]
    fn row_local_derivations() {
        assert_eq!(passage_position(0, 1), 0);
        assert_eq!(passage_position(0, 30), 0);
        assert_eq!(passage_position(29, 30), 96);
        assert_eq!(passage_position(15, 30), 50);
        assert_eq!(passage_position(3, 0), 0);
        assert_eq!(word_count("grid  integration of\nrenewables"), 4);
        assert_eq!(word_count(""), 0);
        let a = sample_bucket(42, 17, 3, DEFAULT_BUCKETS);
        assert_eq!(
            a,
            sample_bucket(42, 17, 3, DEFAULT_BUCKETS),
            "deterministic"
        );
        assert_ne!(a, sample_bucket(43, 17, 3, DEFAULT_BUCKETS), "seeded");
        assert_ne!(
            a,
            sample_bucket(42, 17, 4, DEFAULT_BUCKETS),
            "keyed on the passage, not the paper"
        );
        assert!(sample_bucket(42, 17, 3, 1000) < 1000);
        // Roughly uniform over a small modulus.
        let mut hist = [0u32; 10];
        for c in 0..2000i64 {
            for o in 0..5u64 {
                hist[sample_bucket(1, c, o, 10) as usize] += 1;
            }
        }
        assert!(hist.iter().all(|h| (800..1200).contains(h)), "{:?}", hist);
    }

    /// TS-80/TS-174: bucket occupancy is Poisson — index of dispersion
    /// one — over a corpus whose upstream `ordinal` restarts in every
    /// section, because the key is the source row, not that column.
    #[test]
    fn buckets_are_not_dispersed_by_section_local_ordinals() {
        const BUCKETS: u32 = 4096;
        let mut counts = vec![0u32; BUCKETS as usize];
        let mut row = 0u64;
        for paper in 0..20_000i64 {
            let sections = 1 + (paper % 6) as usize;
            for _section in 0..sections {
                for _ordinal in 0..3 {
                    // A key of (paper, ordinal) would repeat `sections` times.
                    counts[sample_bucket(7, 100_000 + paper, row, BUCKETS) as usize] += 1;
                    row += 1;
                }
            }
        }
        let mean = row as f64 / BUCKETS as f64;
        let var = counts.iter().map(|&c| (c as f64 - mean).powi(2)).sum::<f64>() / BUCKETS as f64;
        let dispersion = var / mean;
        assert!((0.9..1.1).contains(&dispersion), "index of dispersion {} at mean {}", dispersion, mean);
    }

    #[test]
    fn papers_index_finds_the_paper_of_a_row() {
        let papers = Papers {
            rows: vec![
                ParentRow {
                    corpusid: 1,
                    passage_count: 3,
                    row_start: 0,
                },
                ParentRow {
                    corpusid: 2,
                    passage_count: 1,
                    row_start: 3,
                },
                ParentRow {
                    corpusid: 3,
                    passage_count: 4,
                    row_start: 4,
                },
            ],
        };
        assert_eq!(papers.paper_at(0), Some(0));
        assert_eq!(papers.paper_at(2), Some(0));
        assert_eq!(papers.paper_at(3), Some(1));
        assert_eq!(papers.paper_at(7), Some(2));
        assert_eq!(papers.paper_at(8), None);
    }

    #[test]
    fn label_table_falls_back_to_positional() {
        let t = LabelTable::positional(3);
        assert_eq!(t.label(2, 4187), "l3-04187");
        assert!(t.positional);
    }
}
