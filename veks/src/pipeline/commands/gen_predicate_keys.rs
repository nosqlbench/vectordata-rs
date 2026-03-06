// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: evaluate PNode predicates against MNode metadata records
//! and write per-predicate matching ordinal sets to an output slab.
//!
//! Each output slab record is a sequence of `i32 LE` ordinals identifying the
//! metadata records that satisfy the corresponding predicate. This serves as
//! a "predicate answer key" analogous to KNN ground truth.
//!
//! ## Performance
//!
//! Predicates go through two compilation phases:
//!
//! 1. **Memoize** — PNode trees are flattened into field-indexed conditions
//!    (once, at startup).
//! 2. **Schema-compile** — after reading the first metadata record, the
//!    memoized conditions are mapped to field *positions* in the record layout,
//!    producing a [`CompiledScanPredicates`] that uses positional dispatch
//!    instead of per-field name hashing.
//!
//! During scanning, [`scan_record`] walks raw MNode bytes with zero heap
//! allocation: non-targeted fields are skipped in-place and targeted fields
//! are compared directly against comparand values using
//! [`check_condition_raw`].
//!
//! Large datasets are split into segments (default 1 M records). Each segment
//! is processed independently and its results cached to disk so that
//! interrupted runs resume from the last completed segment.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use slabtastic::{OpenProgress, Page, SlabReader, SlabWriter, WriterConfig};

use crate::formats::mnode::scan::{
    discover_schema, flatten_and, missing_field_passes, scan_record, CompiledScanPredicates,
    RecordSchema,
};
use crate::formats::mnode::{MNode, MValue};
use crate::formats::pnode::eval::evaluate;
use crate::formats::pnode::{Comparand, OpType, PNode};
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: generate predicate answer keys from metadata and predicate slabs.
pub struct GenPredicateKeysOp;

/// Create a boxed `GenPredicateKeysOp` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenPredicateKeysOp)
}

// ---------------------------------------------------------------------------
// Memoized predicates
// ---------------------------------------------------------------------------

/// Predicates flattened into an optimal in-memory form for batch evaluation.
///
/// AND-only predicate trees with named fields are decomposed into per-field
/// condition lists. Other predicates (OR trees, indexed fields) are retained
/// as fallbacks for full tree evaluation.
struct MemoizedPredicates {
    /// Total predicate count.
    count: usize,
    /// `field_name → [(predicate_index, op, comparands)]`
    field_conditions: HashMap<String, Vec<(usize, OpType, Vec<Comparand>)>>,
    /// Number of compiled conditions per predicate (0 for fallback-only predicates).
    condition_counts: Vec<u32>,
    /// Predicates that could not be flattened.
    fallback: Vec<(usize, PNode)>,
}

/// Memoize predicates into a flat field-indexed structure.
///
/// This is the first compilation phase: PNode trees → field conditions.
fn memoize_predicates(predicates: &[PNode]) -> MemoizedPredicates {
    let count = predicates.len();
    let mut field_conditions: HashMap<String, Vec<(usize, OpType, Vec<Comparand>)>> =
        HashMap::new();
    let mut condition_counts = vec![0u32; count];
    let mut fallback = Vec::new();

    for (idx, pnode) in predicates.iter().enumerate() {
        match flatten_and(pnode) {
            Some(conditions) => {
                condition_counts[idx] = conditions.len() as u32;
                for (field_name, op, comparands) in conditions {
                    field_conditions
                        .entry(field_name)
                        .or_default()
                        .push((idx, op, comparands));
                }
            }
            None => {
                fallback.push((idx, pnode.clone()));
            }
        }
    }

    MemoizedPredicates {
        count,
        field_conditions,
        condition_counts,
        fallback,
    }
}

// ---------------------------------------------------------------------------
// Name-based evaluation fallback (for schema-mismatched records)
// ---------------------------------------------------------------------------

/// Check a condition against an MValue (used in the fallback path).
fn check_condition(value: &MValue, op: &OpType, comparands: &[Comparand]) -> bool {
    match op {
        OpType::Eq => comparands.iter().any(|c| cmp_eq(value, c)),
        OpType::Ne => comparands.iter().all(|c| !cmp_eq(value, c)),
        OpType::In => comparands.iter().any(|c| cmp_eq(value, c)),
        OpType::Gt => {
            !comparands.is_empty()
                && cmp_ord(value, &comparands[0]) == Some(std::cmp::Ordering::Greater)
        }
        OpType::Lt => {
            !comparands.is_empty()
                && cmp_ord(value, &comparands[0]) == Some(std::cmp::Ordering::Less)
        }
        OpType::Ge => {
            !comparands.is_empty()
                && matches!(
                    cmp_ord(value, &comparands[0]),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                )
        }
        OpType::Le => {
            !comparands.is_empty()
                && matches!(
                    cmp_ord(value, &comparands[0]),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
        }
        OpType::Matches => false,
    }
}

fn cmp_eq(v: &MValue, c: &Comparand) -> bool {
    match (v, c) {
        (MValue::Null, Comparand::Null) => true,
        (MValue::Null, _) | (_, Comparand::Null) => false,
        (MValue::Bool(a), Comparand::Bool(b)) => a == b,

        (MValue::Int(a), Comparand::Int(b)) => a == b,
        (MValue::Int32(a), Comparand::Int(b)) => *a as i64 == *b,
        (MValue::Short(a), Comparand::Int(b)) => *a as i64 == *b,
        (MValue::Millis(a), Comparand::Int(b)) => a == b,

        (MValue::Int(a), Comparand::Float(b)) => *a as f64 == *b,
        (MValue::Int32(a), Comparand::Float(b)) => *a as f64 == *b,
        (MValue::Short(a), Comparand::Float(b)) => *a as f64 == *b,
        (MValue::Millis(a), Comparand::Float(b)) => *a as f64 == *b,

        (MValue::Float(a), Comparand::Float(b)) => a == b,
        (MValue::Float32(a), Comparand::Float(b)) => *a as f64 == *b,

        (MValue::Float(a), Comparand::Int(b)) => *a == *b as f64,
        (MValue::Float32(a), Comparand::Int(b)) => *a as f64 == *b as f64,

        (MValue::Text(a), Comparand::Text(b)) => a == b,
        (MValue::Ascii(a), Comparand::Text(b)) => a == b,
        (MValue::EnumStr(a), Comparand::Text(b)) => a == b,

        (MValue::Bytes(a), Comparand::Bytes(b)) => a == b,

        _ => false,
    }
}

fn cmp_ord(v: &MValue, c: &Comparand) -> Option<std::cmp::Ordering> {
    match (v, c) {
        (MValue::Int(a), Comparand::Int(b)) => Some(a.cmp(b)),
        (MValue::Int32(a), Comparand::Int(b)) => Some((*a as i64).cmp(b)),
        (MValue::Short(a), Comparand::Int(b)) => Some((*a as i64).cmp(b)),
        (MValue::Millis(a), Comparand::Int(b)) => Some(a.cmp(b)),

        (MValue::Int(a), Comparand::Float(b)) => (*a as f64).partial_cmp(b),
        (MValue::Int32(a), Comparand::Float(b)) => (*a as f64).partial_cmp(b),
        (MValue::Short(a), Comparand::Float(b)) => (*a as f64).partial_cmp(b),
        (MValue::Millis(a), Comparand::Float(b)) => (*a as f64).partial_cmp(b),

        (MValue::Float(a), Comparand::Float(b)) => a.partial_cmp(b),
        (MValue::Float32(a), Comparand::Float(b)) => (*a as f64).partial_cmp(b),

        (MValue::Float(a), Comparand::Int(b)) => a.partial_cmp(&(*b as f64)),
        (MValue::Float32(a), Comparand::Int(b)) => (*a as f64).partial_cmp(&(*b as f64)),

        (MValue::Text(a), Comparand::Text(b)) => Some(a.cmp(b)),
        (MValue::Ascii(a), Comparand::Text(b)) => Some(a.cmp(b)),
        (MValue::EnumStr(a), Comparand::Text(b)) => Some(a.cmp(b)),

        _ => None,
    }
}

/// Evaluate a record via the MNode-based fallback path.
///
/// Used when the record's schema does not match the compiled schema (different
/// field count). Processes compiled conditions + missing-field checks + tree
/// fallbacks.
fn eval_record_fallback(
    mnode: &MNode,
    memo: &MemoizedPredicates,
    pass_counts: &mut [u32],
    local_matches: &mut [Vec<i32>],
    compiled_scan: &CompiledScanPredicates,
    ordinal: i32,
    limit: usize,
) {
    let pred_len = memo.count;

    // Reset pass counts
    for c in pass_counts.iter_mut() {
        *c = 0;
    }

    // Evaluate conditions for fields present in the record
    for (field_name, field_value) in &mnode.fields {
        if let Some(conditions) = memo.field_conditions.get(field_name.as_str()) {
            for (pred_idx, op, comparands) in conditions {
                if check_condition(field_value, op, comparands) {
                    pass_counts[*pred_idx] += 1;
                }
            }
        }
    }

    // Missing fields: Eq Null / In Null / Ne <non-Null> pass
    for (field_name, conditions) in &memo.field_conditions {
        if !mnode.fields.contains_key(field_name.as_str()) {
            for (pred_idx, op, comparands) in conditions {
                if missing_field_passes(op, comparands) {
                    pass_counts[*pred_idx] += 1;
                }
            }
        }
    }

    // Compiled predicates match when all conditions pass
    for i in 0..pred_len {
        let required = memo.condition_counts[i];
        if required > 0 && pass_counts[i] == required {
            if limit == 0 || local_matches[i].len() < limit {
                local_matches[i].push(ordinal);
            }
        }
    }

    // Fallback predicates use full tree evaluation
    for (idx, pnode) in compiled_scan.fallback() {
        if evaluate(pnode, mnode) {
            if limit == 0 || local_matches[*idx].len() < limit {
                local_matches[*idx].push(ordinal);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Segment structure
// ---------------------------------------------------------------------------

struct SegmentInfo {
    /// Index into the page_entries slice: first page in this segment.
    page_start_idx: usize,
    /// Index into the page_entries slice: one past the last page.
    page_end_idx: usize,
    /// First record ordinal in this segment.
    start_ordinal: i64,
    /// One past the last record ordinal.
    end_ordinal: i64,
    /// Cache file path (if caching is enabled).
    cache_path: Option<PathBuf>,
    /// Whether the cache file exists and is valid.
    cached: bool,
}

/// Plan segments from sorted page entries.
fn plan_segments(
    page_entries: &[slabtastic::PageEntry],
    total_records: u64,
    segment_size: usize,
    cache_dir: Option<&Path>,
    step_id: &str,
    pred_count: usize,
) -> Vec<SegmentInfo> {
    if page_entries.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut seg_page_start = 0usize;
    let mut seg_start_ordinal = page_entries[0].start_ordinal;
    let mut current_count = 0usize;

    for (i, entry) in page_entries.iter().enumerate() {
        let page_records = if i + 1 < page_entries.len() {
            (page_entries[i + 1].start_ordinal - entry.start_ordinal) as usize
        } else {
            (total_records as i64 - entry.start_ordinal) as usize
        };
        current_count += page_records;

        if current_count >= segment_size || i + 1 == page_entries.len() {
            let end_ordinal = if i + 1 < page_entries.len() {
                page_entries[i + 1].start_ordinal
            } else {
                total_records as i64
            };

            let (cache_path, cached) = match cache_dir {
                Some(dir) => {
                    let path = dir.join(format!(
                        "{}.seg_{:010}_{:010}.predkeys.slab",
                        step_id, seg_start_ordinal, end_ordinal,
                    ));
                    let valid = is_cache_valid(&path, pred_count);
                    (Some(path), valid)
                }
                None => (None, false),
            };

            segments.push(SegmentInfo {
                page_start_idx: seg_page_start,
                page_end_idx: i + 1,
                start_ordinal: seg_start_ordinal,
                end_ordinal,
                cache_path,
                cached,
            });

            seg_page_start = i + 1;
            seg_start_ordinal = end_ordinal;
            current_count = 0;
        }
    }

    segments
}

/// Check if a cache slab exists and has the expected number of records.
fn is_cache_valid(path: &Path, expected_records: usize) -> bool {
    if !path.exists() {
        return false;
    }
    match SlabReader::open(path) {
        Ok(reader) => reader.total_records() as usize == expected_records,
        Err(_) => false,
    }
}

/// Write segment results to a cache slab.
fn write_segment_cache(
    path: &Path,
    matches: &[Vec<i32>],
) -> Result<(), String> {
    let config = WriterConfig::new(512, 4096, u32::MAX, false)
        .map_err(|e| format!("cache writer config: {}", e))?;
    let mut writer =
        SlabWriter::new(path, config).map_err(|e| format!("cache writer open: {}", e))?;

    for pred_matches in matches {
        let mut buf = Vec::with_capacity(pred_matches.len() * 4);
        for &ord in pred_matches {
            buf.write_i32::<LittleEndian>(ord).unwrap();
        }
        writer
            .add_record(&buf)
            .map_err(|e| format!("cache write: {}", e))?;
    }
    writer
        .finish()
        .map_err(|e| format!("cache finish: {}", e))?;
    Ok(())
}

/// Read segment results from a cache slab.
fn read_segment_cache(path: &Path, pred_count: usize) -> Result<Vec<Vec<i32>>, String> {
    let reader =
        SlabReader::open(path).map_err(|e| format!("cache read open: {}", e))?;
    let mut results = Vec::with_capacity(pred_count);
    for pi in 0..pred_count {
        let data = reader
            .get(pi as i64)
            .map_err(|e| format!("cache read pred {}: {}", pi, e))?;
        results.push(read_ordinals(&data));
    }
    Ok(results)
}

fn read_ordinals(data: &[u8]) -> Vec<i32> {
    let mut cursor = std::io::Cursor::new(data);
    let mut result = Vec::with_capacity(data.len() / 4);
    while let Ok(v) = cursor.read_i32::<LittleEndian>() {
        result.push(v);
    }
    result
}

// ---------------------------------------------------------------------------
// Progress formatting
// ---------------------------------------------------------------------------

/// Format a byte count as a human-readable string (B, KB, MB, GB, TB).
fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;
    let b = bytes as f64;
    if b < KB {
        format!("{} B", bytes)
    } else if b < MB {
        format!("{:.1} KB", b / KB)
    } else if b < GB {
        format!("{:.1} MB", b / MB)
    } else if b < TB {
        format!("{:.1} GB", b / GB)
    } else {
        format!("{:.1} TB", b / TB)
    }
}

// ---------------------------------------------------------------------------
// CommandOp implementation
// ---------------------------------------------------------------------------

impl CommandOp for GenPredicateKeysOp {
    fn command_path(&self) -> &str {
        "generate predicate-keys"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Evaluate predicates against metadata to produce match ordinals".into(),
            body: format!(r#"# generate predicate-keys

Evaluate predicates against metadata to produce match ordinals.

## Description

Evaluates PNode predicates against MNode metadata records and writes
per-predicate matching ordinal sets to an output slab. Each output record
is a sequence of i32 LE ordinals identifying the metadata records that
satisfy the corresponding predicate.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Per-predicate match ordinal vectors".into(), adjustable: true },
            ResourceDesc { name: "threads".into(), description: "Parallel predicate evaluation".into(), adjustable: true },
            ResourceDesc { name: "segments".into(), description: "Maximum concurrent segments in flight".into(), adjustable: true },
            ResourceDesc { name: "segmentsize".into(), description: "Records per processing segment".into(), adjustable: true },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let predicates_str = match options.require("predicates") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let input_path = resolve_path(input_str, &ctx.workspace);
        let predicates_path = resolve_path(predicates_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        let limit: usize = options
            .get("limit")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let segment_size: usize = options
            .get("segment_size")
            .and_then(|s| s.parse().ok())
            .or_else(|| ctx.governor.current("segmentsize").map(|v| v as usize))
            .unwrap_or(1_000_000);

        let selectivity: Option<f64> = options
            .get("selectivity")
            .and_then(|s| s.parse().ok());
        let selectivity_max: Option<f64> = options
            .get("selectivity-max")
            .and_then(|s| s.parse().ok());

        // ── Phase 1: Load predicates and memoize ─────────────────────────

        let pred_reader = match SlabReader::open(&predicates_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open predicates slab: {}", e),
                    start,
                )
            }
        };
        let mut predicates: Vec<PNode> = Vec::new();
        let pred_count = pred_reader.total_records();
        for ord in 0..pred_count {
            match pred_reader.get(ord as i64) {
                Ok(data) => match PNode::from_bytes_named(&data) {
                    Ok(pnode) => predicates.push(pnode),
                    Err(e) => {
                        ctx.display.log(&format!(
                            "generate predicate-keys: skipping predicate {}: {}",
                            ord, e
                        ));
                    }
                },
                Err(e) => {
                    ctx.display.log(&format!(
                        "generate predicate-keys: error reading predicate {}: {}",
                        ord, e
                    ));
                }
            }
        }

        let pred_len = predicates.len();
        ctx.display.log(&format!(
            "generate predicate-keys: loaded {} predicates from {}",
            pred_len,
            predicates_path.display(),
        ));

        // Memoize: flatten into field-indexed conditions
        let memo = memoize_predicates(&predicates);
        ctx.display.log(&format!(
            "generate predicate-keys: memoized {} field-indexed, {} fallback, {} fields",
            memo.count - memo.fallback.len(),
            memo.fallback.len(),
            memo.field_conditions.len(),
        ));

        // ── Phase 2: Open metadata slab ──────────────────────────────────

        let index_pb: std::cell::RefCell<Option<indicatif::ProgressBar>> = std::cell::RefCell::new(None);
        let display = &ctx.display;
        let meta_reader = match SlabReader::open_with_progress(&input_path, |p| match p {
            OpenProgress::PagesPageRead { page_count } => {
                *index_pb.borrow_mut() = Some(display.bar(*page_count as u64, "slab index"));
            }
            OpenProgress::IndexBuild { done, total: _ } => {
                if let Some(ref bar) = *index_pb.borrow() {
                    bar.set_position(*done as u64);
                }
            }
            OpenProgress::IndexComplete { total_records } => {
                if let Some(bar) = index_pb.borrow_mut().take() {
                    bar.finish_and_clear();
                }
                display.log(&format!("    slab index: complete, {} total records", total_records));
            }
        }) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open metadata slab: {}", e),
                    start,
                )
            }
        };

        let total_records = meta_reader.total_records();

        let mut page_entries = meta_reader.page_entries();
        page_entries.sort_by_key(|e| e.start_ordinal);
        let total_pages = page_entries.len();

        // ── Phase 3: Discover schema and compile predicates ──────────────

        // Read first record to discover schema
        let schema: Option<RecordSchema> = if !page_entries.is_empty() {
            let first_buf = meta_reader.page_buf(&page_entries[0]);
            match Page::record_count_from_buf(first_buf) {
                Ok(rc) if rc > 0 => {
                    match Page::get_record_ref_from_buf(first_buf, 0, rc) {
                        Ok(data) => {
                            match discover_schema(data) {
                                Ok(s) => {
                                    ctx.display.log(&format!(
                                        "generate predicate-keys: schema discovered: {} fields",
                                        s.field_count,
                                    ));
                                    Some(s)
                                }
                                Err(e) => {
                                    ctx.display.log(&format!(
                                        "generate predicate-keys: schema discovery failed: {}, using fallback path",
                                        e,
                                    ));
                                    None
                                }
                            }
                        }
                        Err(_) => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };

        // Compile against schema (second compilation phase)
        let compiled_scan = match &schema {
            Some(s) => CompiledScanPredicates::compile(
                pred_len,
                &memo.field_conditions,
                &memo.condition_counts,
                memo.fallback.clone(),
                s,
            ),
            None => {
                // No schema — create a dummy compiled struct with no field conditions.
                // All evaluation will go through the fallback path.
                let dummy_schema = RecordSchema {
                    field_names: Vec::new(),
                    field_count: 0,
                };
                CompiledScanPredicates::compile(
                    pred_len,
                    &memo.field_conditions,
                    &memo.condition_counts,
                    memo.fallback.clone(),
                    &dummy_schema,
                )
            }
        };

        // ── Phase 3b: Memory-aware segment sizing (REQ-RM-09) ────────────
        //
        // Estimate peak memory per segment and reduce segment_size if the
        // estimate exceeds the governor's memory budget.  The formula:
        //   per_segment_bytes = pred_count × est_matches_per_pred × 4 × threads
        // where est_matches_per_pred = min(segment_size × selectivity, limit).
        let segment_size = {
            let est_selectivity = selectivity.unwrap_or(0.10); // conservative 10% default
            let est_matches_per_pred = {
                let raw = (segment_size as f64 * est_selectivity).ceil() as usize;
                if limit > 0 { raw.min(limit) } else { raw }
            };
            // Each match is 4 bytes (i32 ordinal), per predicate, per thread
            let num_threads = {
                let base = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
                base.max(1)
            };
            let bytes_per_segment = pred_len as u64
                * est_matches_per_pred as u64
                * 4
                * num_threads as u64;

            if let Some(mem_ceiling) = ctx.governor.mem_ceiling() {
                let snapshot = super::super::resource::SystemSnapshot::sample();
                let target = (mem_ceiling as f64 * 0.85) as u64; // MEM_TARGET_HIGH
                let available = if snapshot.rss_bytes < target {
                    target - snapshot.rss_bytes
                } else {
                    (mem_ceiling as f64 * 0.10) as u64
                };

                if bytes_per_segment > available && bytes_per_segment > 0 {
                    // Scale segment_size down proportionally
                    let scale = available as f64 / bytes_per_segment as f64;
                    let adjusted = ((segment_size as f64) * scale).floor() as usize;
                    let adjusted = adjusted.max(1000); // floor at 1000
                    ctx.display.log(
                        "generate predicate-keys: memory-aware segment sizing");
                    ctx.display.log(&format!(
                        "    estimated {}/segment ({} preds × {} matches × 4B × {} threads)",
                        format_bytes(bytes_per_segment),
                        pred_len, est_matches_per_pred, num_threads,
                    ));
                    ctx.display.log(&format!(
                        "    available memory: {} (RSS: {}, ceiling: {})",
                        format_bytes(available),
                        format_bytes(snapshot.rss_bytes),
                        format_bytes(mem_ceiling),
                    ));
                    ctx.display.log(&format!(
                        "    reducing segment_size: {} → {}",
                        segment_size, adjusted,
                    ));
                    adjusted
                } else {
                    segment_size
                }
            } else {
                segment_size
            }
        };

        // ── Phase 4: Plan segments ───────────────────────────────────────

        // Always use a cache directory to avoid holding all segment results in
        // memory.  When a step_id is set the cache lives under ctx.cache;
        // otherwise the scratch directory is used.
        let cache_dir: PathBuf = if !ctx.step_id.is_empty() {
            ctx.cache.clone()
        } else {
            ctx.scratch.clone()
        };
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            return error_result(
                format!("cannot create cache dir {}: {}", cache_dir.display(), e),
                start,
            );
        }

        let segments = plan_segments(
            &page_entries,
            total_records,
            segment_size,
            Some(cache_dir.as_path()),
            &ctx.step_id,
            pred_len,
        );

        let cached_count = segments.iter().filter(|s| s.cached).count();
        let uncached_count = segments.len() - cached_count;
        let num_threads = if total_pages == 0 {
            1
        } else {
            let base_threads = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
            base_threads.max(1).min(total_pages)
        };

        // ── Statistical summary ─────────────────────────────────────────
        {
            let total_conditions: usize = memo.field_conditions.values()
                .map(|v| v.len())
                .sum();
            let total_comparands: usize = memo.field_conditions.values()
                .flat_map(|v| v.iter())
                .map(|(_, _, comps)| comps.len())
                .sum();
            let comparisons_per_record = total_comparands;
            let total_comparisons = comparisons_per_record as u64 * total_records;

            let limit_desc = if limit > 0 {
                format!("{}", limit)
            } else {
                "unlimited".to_string()
            };
            let max_ordinals_per_pred = if limit > 0 {
                limit as u64
            } else {
                total_records
            };
            // Each match is a 4-byte i32 ordinal. Worst case: every predicate
            // matches every record → max_ordinals_per_pred × 4 × pred_len.
            let max_output_bytes = max_ordinals_per_pred * 4 * pred_len as u64;

            // Per-predicate condition counts for selectivity estimation
            let (min_conds, max_conds, mean_conds) = {
                let counts = &memo.condition_counts;
                if counts.is_empty() {
                    (0u32, 0u32, 0.0f64)
                } else {
                    let min = *counts.iter().min().unwrap();
                    let max = *counts.iter().max().unwrap();
                    let sum: u64 = counts.iter().map(|&c| c as u64).sum();
                    (min, max, sum as f64 / counts.len() as f64)
                }
            };

            ctx.display.log("generate predicate-keys: summary");
            ctx.display.log(&format!("    predicates:             {}", pred_len));
            ctx.display.log(&format!("    records:                {}", total_records));
            ctx.display.log(&format!("    unique fields targeted: {}", memo.field_conditions.len()));
            ctx.display.log(&format!("    total conditions:       {}", total_conditions));
            ctx.display.log(&format!("    total comparands:       {}", total_comparands));
            ctx.display.log(&format!("    comparisons/record:     {}", comparisons_per_record));
            ctx.display.log(&format!("    total comparisons:      {}", total_comparisons));
            ctx.display.log(&format!("    conditions/predicate:   min={} max={} mean={:.1}",
                min_conds, max_conds, mean_conds));
            ctx.display.log(&format!("    limit/predicate:        {}", limit_desc));
            ctx.display.log(&format!("    segments:               {} ({} cached, {} to scan)",
                segments.len(), cached_count, uncached_count));
            ctx.display.log(&format!("    segment size:           {} records", segment_size));
            ctx.display.log(&format!("    threads:                {}", num_threads));
            ctx.display.log("");

            // Output size math breakdown
            let gib = 1_073_741_824.0f64;
            ctx.display.log(&format!("    max output size:        {:.1} GB", max_output_bytes as f64 / gib));
            ctx.display.log(&format!("      = {} ordinals/pred × 4 bytes × {} preds",
                max_ordinals_per_pred, pred_len));
            ctx.display.log("");

            // Selectivity-based estimates: expected matches per predicate and
            // total output. When selectivity/selectivity-max are provided from
            // the predicate generation step, show a focused estimate; otherwise
            // show a table of common selectivity values.
            let effective_limit = if limit > 0 { Some(limit as u64) } else { None };

            let sel_targets: Vec<f64> = if let Some(sel_min) = selectivity {
                // Known selectivity from predicate generation step
                let sel_max = selectivity_max.unwrap_or(sel_min);
                if (sel_min - sel_max).abs() < f64::EPSILON {
                    vec![sel_min]
                } else {
                    vec![sel_min, (sel_min + sel_max) / 2.0, sel_max]
                }
            } else {
                vec![0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
            };

            let estimate_fn = |sel: f64| -> (u64, u64, u64) {
                let raw_matches = (total_records as f64 * sel).ceil() as u64;
                let matches_per_pred = match effective_limit {
                    Some(lim) => raw_matches.min(lim),
                    None => raw_matches,
                };
                let total_indices = matches_per_pred * pred_len as u64;
                let data_bytes = total_indices * 4;
                (matches_per_pred, total_indices, data_bytes)
            };

            if selectivity.is_some() {
                let sel_min = selectivity.unwrap();
                let sel_max = selectivity_max.unwrap_or(sel_min);
                if (sel_min - sel_max).abs() < f64::EPSILON {
                    ctx.display.log(&format!("    selectivity:            {:.4}%  (from predicate generation)",
                        sel_min * 100.0));
                } else {
                    ctx.display.log(&format!("    selectivity:            {:.4}% .. {:.4}%  (from predicate generation)",
                        sel_min * 100.0, sel_max * 100.0));
                }
                let (matches, indices, bytes) = estimate_fn(sel_min);
                ctx.display.log(&format!("    est. matches/pred:      {}", matches));
                ctx.display.log(&format!("    est. total indices:     {}", indices));
                ctx.display.log(&format!("    est. output size:       {}", format_bytes(bytes)));
                if sel_min != sel_max {
                    let (matches_hi, indices_hi, bytes_hi) = estimate_fn(sel_max);
                    ctx.display.log(&format!("    est. matches/pred (hi): {}", matches_hi));
                    ctx.display.log(&format!("    est. total indices (hi):{}", indices_hi));
                    ctx.display.log(&format!("    est. output size (hi):  {}", format_bytes(bytes_hi)));
                }
            } else {
                ctx.display.log("    estimated output by selectivity (pass selectivity= to narrow):");
                ctx.display.log(&format!("    {:>12}  {:>14}  {:>14}  {:>10}",
                    "selectivity", "matches/pred", "total indices", "data size"));
                for &sel in &sel_targets {
                    let (matches_per_pred, total_indices, data_bytes) = estimate_fn(sel);
                    ctx.display.log(&format!("    {:>11.1}%  {:>14}  {:>14}  {:>10}",
                        sel * 100.0,
                        matches_per_pred,
                        total_indices,
                        format_bytes(data_bytes)));
                }
            }
        }

        // ── Phase 5: Process segments ────────────────────────────────────

        // Governor checkpoint before segment processing
        ctx.governor.checkpoint();

        let records_done = AtomicU64::new(0);
        let errors_done = AtomicU64::new(0);
        let scan_start = Instant::now();
        let next_segment = AtomicUsize::new(0);

        // Progress bar for the scan phase
        let scan_pb = ctx.display.bar(total_records, "scanning records");

        // Scan uncached segments — results are written to cache files and
        // dropped immediately, keeping memory proportional to one segment per
        // thread rather than all segments combined.
        if uncached_count > 0 {
            // Hint sequential access to enable kernel readahead
            meta_reader.advise_sequential();

            let seg_ref = &segments;
            let display = &ctx.display;
            std::thread::scope(|s| {
                // Prefetch thread — walks uncached segments in order, issuing
                // MADV_WILLNEED for each segment's byte range so the kernel
                // pages data in ahead of the scan threads.
                {
                    let reader = &meta_reader;
                    let pe = &page_entries;
                    s.spawn(move || {
                        for seg in seg_ref.iter() {
                            if seg.cached {
                                continue;
                            }
                            let start = pe[seg.page_start_idx].file_offset as usize;
                            let end = if seg.page_end_idx < pe.len() {
                                pe[seg.page_end_idx].file_offset as usize
                            } else {
                                reader.file_len().unwrap_or(0) as usize
                            };
                            reader.prefetch_range(start, end);
                        }
                    });
                }

                let handles: Vec<_> = (0..num_threads)
                    .map(|_| {
                        let reader = &meta_reader;
                        let compiled_scan = &compiled_scan;
                        let memo = &memo;
                        let records_done = &records_done;
                        let errors_done = &errors_done;
                        let next_segment = &next_segment;
                        let scan_pb = &scan_pb;
                        let page_entries = &page_entries;
                        let display = display;

                        s.spawn(move || {
                            let mut pass_counts = vec![0u32; pred_len];

                            loop {
                                let seg_idx =
                                    next_segment.fetch_add(1, Ordering::Relaxed);
                                if seg_idx >= seg_ref.len() {
                                    break;
                                }
                                let seg = &seg_ref[seg_idx];
                                if seg.cached {
                                    continue;
                                }

                                let mut local_matches: Vec<Vec<i32>> =
                                    vec![Vec::new(); pred_len];
                                let mut local_rec_count = 0u64;

                                for pi in seg.page_start_idx..seg.page_end_idx {
                                    let entry = &page_entries[pi];
                                    let page_buf = reader.page_buf(entry);
                                    let rec_count = match Page::record_count_from_buf(page_buf) {
                                        Ok(rc) => rc,
                                        Err(_) => continue,
                                    };

                                    let page_start = entry.start_ordinal;

                                    for rec_idx in 0..rec_count {
                                        let data = match Page::get_record_ref_from_buf(
                                            page_buf, rec_idx, rec_count,
                                        ) {
                                            Ok(d) => d,
                                            Err(_) => continue,
                                        };

                                        let ordinal =
                                            (page_start + rec_idx as i64) as i32;

                                        // Reset pass counts
                                        for c in pass_counts.iter_mut() {
                                            *c = 0;
                                        }

                                        // Try zero-alloc scan
                                        let schema_matched = match scan_record(
                                            data,
                                            compiled_scan,
                                            &mut pass_counts,
                                        ) {
                                            Ok(matched) => matched,
                                            Err(_) => {
                                                errors_done
                                                    .fetch_add(1, Ordering::Relaxed);
                                                continue;
                                            }
                                        };

                                        if schema_matched {
                                            // Check compiled predicates
                                            for i in 0..pred_len {
                                                let required =
                                                    compiled_scan.required_count(i);
                                                if required != u32::MAX
                                                    && pass_counts[i] == required
                                                {
                                                    if limit == 0
                                                        || local_matches[i].len() < limit
                                                    {
                                                        local_matches[i].push(ordinal);
                                                    }
                                                }
                                            }

                                            // Fallback predicates still need MNode
                                            if !compiled_scan.fallback().is_empty() {
                                                match MNode::from_bytes(data) {
                                                    Ok(mnode) => {
                                                        for (idx, pnode) in
                                                            compiled_scan.fallback()
                                                        {
                                                            if evaluate(pnode, &mnode) {
                                                                if limit == 0
                                                                    || local_matches[*idx]
                                                                        .len()
                                                                        < limit
                                                                {
                                                                    local_matches[*idx]
                                                                        .push(ordinal);
                                                                }
                                                            }
                                                        }
                                                    }
                                                    Err(_) => {
                                                        errors_done.fetch_add(
                                                            1,
                                                            Ordering::Relaxed,
                                                        );
                                                    }
                                                }
                                            }
                                        } else {
                                            // Schema mismatch — full fallback
                                            match MNode::from_bytes(data) {
                                                Ok(mnode) => {
                                                    eval_record_fallback(
                                                        &mnode,
                                                        memo,
                                                        &mut pass_counts,
                                                        &mut local_matches,
                                                        compiled_scan,
                                                        ordinal,
                                                        limit,
                                                    );
                                                }
                                                Err(_) => {
                                                    errors_done
                                                        .fetch_add(1, Ordering::Relaxed);
                                                }
                                            }
                                        }

                                        // Progress reporting
                                        local_rec_count += 1;
                                        if local_rec_count % 100 == 0 {
                                            let total = records_done
                                                .fetch_add(100, Ordering::Relaxed)
                                                + 100;
                                            scan_pb.set_position(total);
                                        }
                                    }

                                    // Flush remaining
                                    let leftover = local_rec_count % 100;
                                    if leftover > 0 {
                                        records_done
                                            .fetch_add(leftover, Ordering::Relaxed);
                                        local_rec_count = 0;
                                    }
                                }

                                // Always write to cache — this is how we keep
                                // memory bounded.  local_matches is dropped after
                                // this block, freeing the ordinal vectors.
                                if let Some(ref cache_path) = seg.cache_path {
                                    if let Err(e) =
                                        write_segment_cache(cache_path, &local_matches)
                                    {
                                        display.log(&format!(
                                            "generate predicate-keys: cache write error for segment {}: {}",
                                            seg_idx, e
                                        ));
                                    }
                                }
                                // local_matches dropped here — memory reclaimed
                            }
                        })
                    })
                    .collect();

                // Wait for all threads to finish
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        }

        // Governor checkpoint after segment processing
        if ctx.governor.checkpoint() {
            ctx.display.log("generate predicate-keys: governor throttle signal received");
        }

        scan_pb.finish_and_clear();

        let total_scanned = records_done.load(Ordering::Relaxed);
        let total_errors = errors_done.load(Ordering::Relaxed);
        let scan_elapsed = scan_start.elapsed();
        let final_rate = if scan_elapsed.as_secs_f64() > 0.0 {
            total_scanned as f64 / scan_elapsed.as_secs_f64()
        } else {
            0.0
        };
        ctx.display.log(&format!(
            "    scan complete: {} records in {:.1}s ({:.0} rec/s), {} decode errors",
            total_scanned,
            scan_elapsed.as_secs_f64(),
            final_rate,
            total_errors,
        ));

        // ── Phase 6: Merge segment results from cache ─────────────────────
        //
        // Stream one segment at a time from cache files to keep memory bounded.
        // Only one segment's worth of data is in memory at any point.

        let mut matches: Vec<Vec<i32>> = vec![Vec::new(); pred_len];
        for (seg_idx, seg) in segments.iter().enumerate() {
            if let Some(ref cache_path) = seg.cache_path {
                match read_segment_cache(cache_path, pred_len) {
                    Ok(seg_matches) => {
                        for (pi, seg_vec) in seg_matches.into_iter().enumerate() {
                            if limit > 0 && matches[pi].len() >= limit {
                                continue;
                            }
                            matches[pi].extend(seg_vec);
                            if limit > 0 {
                                matches[pi].truncate(limit);
                            }
                        }
                    }
                    Err(e) => {
                        ctx.display.log(&format!(
                            "generate predicate-keys: error reading cache for segment {}: {}",
                            seg_idx, e
                        ));
                    }
                }
            }
        }

        // ── Phase 7: Write output slab ───────────────────────────────────

        let config = match WriterConfig::new(512, 4096, u32::MAX, false) {
            Ok(c) => c,
            Err(e) => return error_result(format!("writer config error: {}", e), start),
        };
        let mut writer = match SlabWriter::new(&output_path, config) {
            Ok(w) => w,
            Err(e) => {
                return error_result(
                    format!("failed to create output: {}", e),
                    start,
                )
            }
        };

        let mut total_matches: u64 = 0;
        for (pi, pred_matches) in matches.iter().enumerate() {
            total_matches += pred_matches.len() as u64;

            let mut buf = Vec::with_capacity(pred_matches.len() * 4);
            for &ord in pred_matches {
                buf.write_i32::<LittleEndian>(ord).unwrap();
            }
            if let Err(e) = writer.add_record(&buf) {
                return error_result(
                    format!("write error at predicate {}: {}", pi, e),
                    start,
                );
            }
        }

        if let Err(e) = writer.finish() {
            return error_result(format!("finish error: {}", e), start);
        }

        let avg_matches = if pred_len > 0 {
            total_matches / pred_len as u64
        } else {
            0
        };
        let message = format!(
            "{} predicate keys generated from {} predicates × {} records ({} avg matches), {:.1}s",
            pred_len,
            pred_len,
            total_scanned,
            avg_matches,
            start.elapsed().as_secs_f64(),
        );
        ctx.display.log(&format!("generate predicate-keys: {}", message));

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt(
                "input",
                "Path",
                true,
                None,
                "Metadata slab (MNode records)",
            ),
            opt(
                "predicates",
                "Path",
                true,
                None,
                "Predicate slab (PNode records)",
            ),
            opt(
                "output",
                "Path",
                true,
                None,
                "Output slab for answer key records",
            ),
            opt(
                "limit",
                "int",
                false,
                Some("0"),
                "Max matches per predicate (0 = unlimited)",
            ),
            opt(
                "segment_size",
                "int",
                false,
                Some("1000000"),
                "Records per segment for cache partitioning",
            ),
            opt(
                "selectivity",
                "float",
                false,
                None,
                "Target selectivity from predicate generation (0.0–1.0), for output size estimates",
            ),
            opt(
                "selectivity-max",
                "float",
                false,
                None,
                "Upper bound of selectivity range (if predicates used a range)",
            ),
        ]
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

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
) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mnode::{MNode, MValue};
    use crate::formats::pnode::{
        Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
    };
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &std::path::Path) -> StreamContext {
        StreamContext {
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            display: crate::pipeline::display::ProgressDisplay::new(),
        }
    }

    fn create_metadata_slab(
        dir: &std::path::Path,
        name: &str,
        records: &[MNode],
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut writer = SlabWriter::new(&path, config).unwrap();
        for node in records {
            writer.add_record(&node.to_bytes()).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    fn create_predicates_slab(
        dir: &std::path::Path,
        name: &str,
        predicates: &[PNode],
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut writer = SlabWriter::new(&path, config).unwrap();
        for pnode in predicates {
            writer.add_record(&pnode.to_bytes_named()).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    fn make_test_records() -> Vec<MNode> {
        let mut records = Vec::new();
        for i in 0..20 {
            let mut node = MNode::new();
            node.insert("user_id".into(), MValue::Int(i));
            node.insert("name".into(), MValue::Text(format!("user_{}", i % 5)));
            node.insert("score".into(), MValue::Float(i as f64 * 1.5));
            node.insert("active".into(), MValue::Bool(i % 2 == 0));
            records.push(node);
        }
        records
    }

    fn read_test_ordinals(data: &[u8]) -> Vec<i32> {
        read_ordinals(data)
    }

    #[test]
    fn test_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let records = make_test_records();
        let meta_path = create_metadata_slab(ws, "meta.slab", &records);

        // Predicate 0: user_id > 15 → matches ordinals 16,17,18,19
        let p0 = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("user_id".into()),
            op: OpType::Gt,
            comparands: vec![Comparand::Int(15)],
        });
        // Predicate 1: active = true → matches even ordinals
        let p1 = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("active".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Bool(true)],
        });
        // Predicate 2: name = 'user_0' → matches ordinals 0,5,10,15
        let p2 = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("name".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Text("user_0".into())],
        });

        let pred_path = create_predicates_slab(ws, "preds.slab", &[p0, p1, p2]);
        let output_path = ws.join("keys.slab");

        let mut opts = Options::new();
        opts.set("input", meta_path.to_string_lossy().to_string());
        opts.set("predicates", pred_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());

        let mut op = GenPredicateKeysOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        let reader = SlabReader::open(&output_path).unwrap();

        // Predicate 0: user_id > 15 → 16,17,18,19
        let ords0 = read_test_ordinals(&reader.get(0).unwrap());
        assert_eq!(ords0, vec![16, 17, 18, 19]);

        // Predicate 1: active = true → even ordinals
        let ords1 = read_test_ordinals(&reader.get(1).unwrap());
        let expected_even: Vec<i32> = (0..20).filter(|i| i % 2 == 0).collect();
        assert_eq!(ords1, expected_even);

        // Predicate 2: name = 'user_0' → 0,5,10,15
        let ords2 = read_test_ordinals(&reader.get(2).unwrap());
        assert_eq!(ords2, vec![0, 5, 10, 15]);
    }

    #[test]
    fn test_with_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let records = make_test_records();
        let meta_path = create_metadata_slab(ws, "meta.slab", &records);

        // active = true matches 10 records; limit to 3
        let p = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("active".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Bool(true)],
        });
        let pred_path = create_predicates_slab(ws, "preds.slab", &[p]);
        let output_path = ws.join("keys.slab");

        let mut opts = Options::new();
        opts.set("input", meta_path.to_string_lossy().to_string());
        opts.set("predicates", pred_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("limit", "3".to_string());

        let mut op = GenPredicateKeysOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        let reader = SlabReader::open(&output_path).unwrap();
        let data = reader.get(0).unwrap();
        assert!(
            data.len() <= 12,
            "expected at most 12 bytes (3 × 4), got {}",
            data.len()
        );
        let ords = read_test_ordinals(&data);
        assert_eq!(ords.len(), 3);
    }

    #[test]
    fn test_conjugate() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let records = make_test_records();
        let meta_path = create_metadata_slab(ws, "meta.slab", &records);

        // AND: user_id >= 10 AND active = true → even ordinals >= 10
        let p = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("user_id".into()),
                    op: OpType::Ge,
                    comparands: vec![Comparand::Int(10)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("active".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Bool(true)],
                }),
            ],
        });

        let pred_path = create_predicates_slab(ws, "preds.slab", &[p]);
        let output_path = ws.join("keys.slab");

        let mut opts = Options::new();
        opts.set("input", meta_path.to_string_lossy().to_string());
        opts.set("predicates", pred_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());

        let mut op = GenPredicateKeysOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        let reader = SlabReader::open(&output_path).unwrap();
        let ords = read_test_ordinals(&reader.get(0).unwrap());
        // even ordinals in [10..20): 10, 12, 14, 16, 18
        assert_eq!(ords, vec![10, 12, 14, 16, 18]);
    }

    #[test]
    fn test_empty_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Empty metadata slab
        let meta_path = ws.join("empty_meta.slab");
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut w = SlabWriter::new(&meta_path, config).unwrap();
        w.finish().unwrap();

        let p = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("x".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Int(1)],
        });
        let pred_path = create_predicates_slab(ws, "preds.slab", &[p]);
        let output_path = ws.join("keys.slab");

        let mut opts = Options::new();
        opts.set("input", meta_path.to_string_lossy().to_string());
        opts.set("predicates", pred_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());

        let mut op = GenPredicateKeysOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        let reader = SlabReader::open(&output_path).unwrap();
        let data = reader.get(0).unwrap();
        assert_eq!(data.len(), 0, "expected empty record for no matches");
    }
}
