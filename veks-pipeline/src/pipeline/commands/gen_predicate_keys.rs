// Copyright (c) Jonathan Shook
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use slabtastic::{OpenProgress, Page, SlabReader, SlabWriter, WriterConfig};

use veks_core::formats::mnode::scan::{
    discover_schema, flatten_and, missing_field_passes, scan_record, CompiledScanPredicates,
    RecordSchema,
};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::pnode::eval::evaluate;
use veks_core::formats::pnode::{Comparand, OpType, PNode};
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use super::slab::survey_from_json;

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

    for c in pass_counts.iter_mut() {
        *c = 0;
    }

    for (field_name, field_value) in &mnode.fields {
        if let Some(conditions) = memo.field_conditions.get(field_name.as_str()) {
            for (pred_idx, op, comparands) in conditions {
                if check_condition(field_value, op, comparands) {
                    pass_counts[*pred_idx] += 1;
                }
            }
        }
    }

    for (field_name, conditions) in &memo.field_conditions {
        if !mnode.fields.contains_key(field_name.as_str()) {
            for (pred_idx, op, comparands) in conditions {
                if missing_field_passes(op, comparands) {
                    pass_counts[*pred_idx] += 1;
                }
            }
        }
    }

    for i in 0..pred_len {
        let required = memo.condition_counts[i];
        if required > 0 && pass_counts[i] == required {
            if limit == 0 || local_matches[i].len() < limit {
                local_matches[i].push(ordinal);
            }
        }
    }

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

/// Plan segments at exact `segment_size` ordinal boundaries.
///
/// Segments cover fixed ordinal ranges `[0, S), [S, 2S), …` where
/// `S = segment_size`.  Pages that straddle a boundary are included in
/// both adjacent segments; the scan loop filters records to the segment's
/// ordinal range.
///
/// The cache key uses `cache_prefix` (derived from input file stems) rather
/// than the pipeline step ID, so that different profiles with overlapping
/// ordinal ranges can share cached segments.
fn plan_segments(
    page_entries: &[slabtastic::PageEntry],
    end_ordinal: u64,
    segment_size: usize,
    cache_dir: Option<&Path>,
    cache_prefix: &str,
    pred_count: usize,
    first_ordinal: i64,
    compress_cache: bool,
) -> Vec<SegmentInfo> {
    if page_entries.is_empty() || end_ordinal == 0 {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut seg_start: i64 = first_ordinal;

    // Scan for super-segments from smaller profiles: look for cached segments
    // covering a larger range than segment_size, starting at seg_start.
    let find_largest_cached_segment = |start: i64, max_end: i64, dir: &Path| -> Option<i64> {
        let mut try_end = max_end;
        while try_end > start + segment_size as i64 {
            let path = dir.join(format!(
                "{}.seg_{:010}_{:010}.predkeys.slab",
                cache_prefix, start, try_end,
            ));
            if is_cache_valid(&path, pred_count, compress_cache) {
                return Some(try_end);
            }
            // Step down by segment_size
            try_end = ((try_end - start - 1) / segment_size as i64) * segment_size as i64 + start;
            if try_end <= start { break; }
        }
        None
    };

    while (seg_start as u64) < end_ordinal {
        // Check for super-segment first
        let seg_end = if let Some(dir) = cache_dir {
            if let Some(super_end) = find_largest_cached_segment(seg_start, end_ordinal as i64, dir) {
                super_end
            } else {
                ((seg_start as u64) + segment_size as u64).min(end_ordinal) as i64
            }
        } else {
            ((seg_start as u64) + segment_size as u64).min(end_ordinal) as i64
        };

        // First page: last page with start_ordinal <= seg_start (contains seg_start).
        let ps = page_entries.partition_point(|e| e.start_ordinal <= seg_start);
        let page_start_idx = if ps > 0 { ps - 1 } else { 0 };

        // Last page (exclusive): first page with start_ordinal >= seg_end.
        let page_end_idx = page_entries.partition_point(|e| e.start_ordinal < seg_end);

        let (cache_path, cached) = match cache_dir {
            Some(dir) => {
                let path = dir.join(format!(
                    "{}.seg_{:010}_{:010}.predkeys.slab",
                    cache_prefix, seg_start, seg_end,
                ));
                let valid = is_cache_valid(&path, pred_count, compress_cache);
                (Some(path), valid)
            }
            None => (None, false),
        };

        segments.push(SegmentInfo {
            page_start_idx,
            page_end_idx,
            start_ordinal: seg_start,
            end_ordinal: seg_end,
            cache_path,
            cached,
        });

        seg_start = seg_end;
    }

    segments
}

/// Check if a cache slab exists and has the expected number of records.
///
/// When `compress_cache` is true, also checks for the `.gz` compressed form.
/// If the compressed form exists, it is decompressed to the original path so
/// that subsequent `SlabReader::open` calls work normally.
fn is_cache_valid(path: &Path, expected_records: usize, compress_cache: bool) -> bool {
    // If the uncompressed file already exists, validate it directly.
    if path.exists() {
        return match SlabReader::open(path) {
            Ok(reader) => reader.total_records() as usize == expected_records,
            Err(_) => false,
        };
    }

    // If compress-cache is enabled, check for the .gz form.
    if compress_cache && crate::pipeline::gz_cache::gz_exists(path) {
        match crate::pipeline::gz_cache::load_gz(path) {
            Ok(data) => {
                // Write decompressed data to the original path for SlabReader.
                if std::fs::write(path, &data).is_err() {
                    return false;
                }
                let valid = match SlabReader::open(path) {
                    Ok(reader) => reader.total_records() as usize == expected_records,
                    Err(_) => false,
                };
                // Clean up decompressed file if invalid.
                if !valid {
                    let _ = std::fs::remove_file(path);
                }
                valid
            }
            Err(_) => false,
        }
    } else {
        false
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
// Memory profiling
// ---------------------------------------------------------------------------

/// Detailed memory breakdown from /proc/self/status.
#[derive(Clone, Debug)]
struct MemProfile {
    vm_rss: u64,       // VmRSS: total resident
    rss_anon: u64,     // RssAnon: heap + stack (anonymous pages)
    rss_file: u64,     // RssFile: mmap'd file pages (includes slab mmap)
    rss_shmem: u64,    // RssShmem: shared memory
    vm_data: u64,      // VmData: private data segments (heap capacity)
    vm_swap: u64,      // VmSwap: swapped out
}

impl MemProfile {
    fn sample() -> Self {
        #[cfg(target_os = "linux")]
        {
            let mut p = MemProfile {
                vm_rss: 0, rss_anon: 0, rss_file: 0, rss_shmem: 0,
                vm_data: 0, vm_swap: 0,
            };
            if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
                for line in content.lines() {
                    let mut parts = line.split_whitespace();
                    if let (Some(key), Some(val)) = (parts.next(), parts.next()) {
                        let kb: u64 = val.parse().unwrap_or(0);
                        let bytes = kb * 1024;
                        match key {
                            "VmRSS:" => p.vm_rss = bytes,
                            "RssAnon:" => p.rss_anon = bytes,
                            "RssFile:" => p.rss_file = bytes,
                            "RssShmem:" => p.rss_shmem = bytes,
                            "VmData:" => p.vm_data = bytes,
                            "VmSwap:" => p.vm_swap = bytes,
                            _ => {}
                        }
                    }
                }
            }
            p
        }
        #[cfg(not(target_os = "linux"))]
        {
            MemProfile {
                vm_rss: 0, rss_anon: 0, rss_file: 0, rss_shmem: 0,
                vm_data: 0, vm_swap: 0,
            }
        }
    }

    fn summary(&self) -> String {
        format!(
            "VmRSS={} (anon={}, file={}, shmem={}) VmData={} VmSwap={}",
            format_bytes(self.vm_rss),
            format_bytes(self.rss_anon),
            format_bytes(self.rss_file),
            format_bytes(self.rss_shmem),
            format_bytes(self.vm_data),
            format_bytes(self.vm_swap),
        )
    }

    fn delta_from(&self, baseline: &MemProfile) -> String {
        let d = |cur: u64, base: u64| -> String {
            if cur >= base {
                format!("+{}", format_bytes(cur - base))
            } else {
                format!("-{}", format_bytes(base - cur))
            }
        };
        format!(
            "Δ VmRSS={} (anon={}, file={}, shmem={}) VmData={}",
            d(self.vm_rss, baseline.vm_rss),
            d(self.rss_anon, baseline.rss_anon),
            d(self.rss_file, baseline.rss_file),
            d(self.rss_shmem, baseline.rss_shmem),
            d(self.vm_data, baseline.vm_data),
        )
    }
}

// ---------------------------------------------------------------------------
// Format-aware record parsing
// ---------------------------------------------------------------------------

/// Parse records from scalar (u8, i16, etc.) or ivec format into Vec<Vec<i32>>.
fn parse_scalar_records(data: &[u8], ext: &str, fields: usize) -> Vec<Vec<i32>> {
    let elem_size: usize = match ext {
        "u8" | "i8" => 1,
        "u16" | "i16" => 2,
        "u32" | "i32" => 4,
        "u64" | "i64" => 8,
        "ivec" | "ivecs" => 0, // xvec with dim header
        _ => 0,
    };

    let mut records = Vec::new();

    if elem_size > 0 {
        // Scalar format: flat packed, no header
        let record_size = elem_size * fields;
        if record_size == 0 { return records; }
        let total = data.len() / record_size;
        for i in 0..total {
            let offset = i * record_size;
            let mut vals = Vec::with_capacity(fields);
            for f in 0..fields {
                let fo = offset + f * elem_size;
                let val: i32 = match elem_size {
                    1 => data[fo] as i32,
                    2 => i16::from_le_bytes([data[fo], data[fo+1]]) as i32,
                    4 => i32::from_le_bytes(data[fo..fo+4].try_into().unwrap()),
                    8 => i64::from_le_bytes(data[fo..fo+8].try_into().unwrap()) as i32,
                    _ => 0,
                };
                vals.push(val);
            }
            records.push(vals);
        }
    } else {
        // ivec format: [dim:i32, val0:i32, val1:i32, ...]
        let mut offset = 0;
        while offset + 4 <= data.len() {
            let dim = i32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + dim * 4 > data.len() { break; }
            let mut vals = Vec::with_capacity(fields.min(dim));
            for f in 0..fields.min(dim) {
                let fo = offset + f * 4;
                vals.push(i32::from_le_bytes(data[fo..fo+4].try_into().unwrap()));
            }
            records.push(vals);
            offset += dim * 4;
        }
    }
    records
}

// ---------------------------------------------------------------------------
// Simple-int-eq evaluation
// ---------------------------------------------------------------------------

impl GenPredicateKeysOp {
    fn execute_simple_int_eq(
        &self,
        options: &Options,
        ctx: &mut StreamContext,
        start: Instant,
    ) -> CommandResult {
        let input_str = match options.require("source") {
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

        let fields: usize = options.parse_or("fields", 1u32).unwrap_or(1) as usize;

        if let Some(parent) = output_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        // Read metadata: detect format from extension
        let meta_bytes = match std::fs::read(&input_path) {
            Ok(b) => b,
            Err(e) => return error_result(format!("read {}: {}", input_path.display(), e), start),
        };
        let meta_ext = input_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let metadata = parse_scalar_records(&meta_bytes, meta_ext, fields);
        ctx.ui.log(&format!("  evaluate: {} metadata records, {} fields (format={})",
            metadata.len(), fields, meta_ext));

        // Read predicate file
        let pred_bytes = match std::fs::read(&predicates_path) {
            Ok(b) => b,
            Err(e) => return error_result(format!("read {}: {}", predicates_path.display(), e), start),
        };
        let pred_ext = predicates_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let predicates = parse_scalar_records(&pred_bytes, pred_ext, fields);
        ctx.ui.log(&format!("  evaluate: {} predicates (format={})", predicates.len(), pred_ext));

        // Build index: map from field values → matching ordinals
        // For single-field: HashMap<i32, Vec<i32>>
        // For multi-field: HashMap<Vec<i32>, Vec<i32>>
        let pred_count = predicates.len();

        let results: Vec<Vec<i32>> = if fields == 1 {
            // Single-field fast path: hash map from value → ordinals
            let mut index: std::collections::HashMap<i32, Vec<i32>> =
                std::collections::HashMap::new();
            for (mi, meta) in metadata.iter().enumerate() {
                index.entry(meta[0]).or_default().push(mi as i32);
            }
            ctx.ui.log(&format!("  built index: {} distinct values", index.len()));

            let pb = ctx.ui.bar(pred_count as u64, "evaluating predicates");
            let results: Vec<Vec<i32>> = predicates.iter().enumerate().map(|(pi, pred)| {
                if (pi + 1) % 10_000 == 0 { pb.set_position((pi + 1) as u64); }
                index.get(&pred[0]).cloned().unwrap_or_default()
            }).collect();
            pb.finish();
            results
        } else {
            // Multi-field: hash map from value tuple → ordinals
            let mut index: std::collections::HashMap<Vec<i32>, Vec<i32>> =
                std::collections::HashMap::new();
            for (mi, meta) in metadata.iter().enumerate() {
                index.entry(meta.clone()).or_default().push(mi as i32);
            }
            ctx.ui.log(&format!("  built index: {} distinct value tuples", index.len()));

            let pb = ctx.ui.bar(pred_count as u64, "evaluating predicates");
            let results: Vec<Vec<i32>> = predicates.iter().enumerate().map(|(pi, pred)| {
                if (pi + 1) % 10_000 == 0 { pb.set_position((pi + 1) as u64); }
                index.get(pred).cloned().unwrap_or_default()
            }).collect();
            pb.finish();
            results
        };

        // Remove stale index before rewriting the vvec file
        vectordata::io::remove_vvec_index(&output_path);

        // Write output as ivec (each record = matching ordinals)
        {
            use std::io::Write;
            let mut f = match std::fs::File::create(&output_path) {
                Ok(f) => std::io::BufWriter::new(f),
                Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
            };
            for matches in &results {
                let dim = matches.len() as i32;
                f.write_all(&dim.to_le_bytes()).unwrap_or(());
                for &ord in matches {
                    f.write_all(&ord.to_le_bytes()).unwrap_or(());
                }
            }
        }

        // Build offset index for variable-length output (vvec/ivvec)
        let out_ext = output_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if veks_core::formats::VecFormat::from_extension(out_ext)
            .map(|f| f.is_vvec()).unwrap_or(false)
            || out_ext == "ivec" || out_ext == "ivecs"
        {
            match vectordata::io::IndexedXvecReader::open_ivec(&output_path) {
                Ok(r) => ctx.ui.log(&format!("  built offset index ({} records)", r.count())),
                Err(e) => ctx.ui.log(&format!("  WARNING: failed to build offset index: {}", e)),
            }
        }

        let total_matches: usize = results.iter().map(|r| r.len()).sum();
        let avg_matches = if pred_count > 0 { total_matches as f64 / pred_count as f64 } else { 0.0 };
        ctx.ui.log(&format!("  {} predicates evaluated, avg {:.1} matches/predicate",
            pred_count, avg_matches));

        // Write verified count
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &pred_count.to_string());
        ctx.defaults.insert(var_name, pred_count.to_string());

        CommandResult {
            status: Status::Ok,
            message: format!("{} predicates, avg {:.1} matches", pred_count, avg_matches),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }
}

// ---------------------------------------------------------------------------
// CommandOp implementation
// ---------------------------------------------------------------------------

impl CommandOp for GenPredicateKeysOp {
    fn command_path(&self) -> &str {
        "compute evaluate-predicates"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_COMPUTE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Compute predicates against metadata to produce match ordinals".into(),
            body: format!(r#"# compute evaluate-predicates

Compute predicates against metadata to produce match ordinals.

## Description

Evaluates each PNode predicate against every MNode metadata record in
the input slab and writes per-predicate matching ordinal sets to an
output slab. Each output record is a sequence of `i32 LE` ordinals
identifying the metadata records that satisfy the corresponding predicate.
This serves as a **predicate answer key** -- analogous to KNN ground truth
indices, but for metadata filtering.

## How it works

Predicates go through two compilation phases for efficient evaluation:

1. **Memoize** -- PNode trees are flattened into field-indexed condition
   lists (once, at startup). AND-only trees with named fields are
   decomposed into per-field conditions; other predicates (OR trees,
   indexed fields) are retained as fallbacks for full tree evaluation.
2. **Schema-compile** -- after reading the first metadata record, the
   memoized conditions are mapped to field *positions* in the record
   layout, producing a compiled predicate set that uses positional
   dispatch instead of per-field name hashing.

During scanning, each record's raw MNode bytes are walked with zero heap
allocation: non-targeted fields are skipped in-place and targeted fields
are compared directly against comparand values. This makes evaluation
extremely fast even on hundreds of millions of records.

## Segmented processing

Large datasets are split into segments (default 1M records). Each segment
is processed independently and its results cached to disk, so interrupted
runs resume from the last completed segment rather than starting over.
This is essential for multi-billion-record datasets where a full scan can
take hours.

## Output format

The output slab contains one record per predicate. Each record is a
packed array of `i32 LE` ordinals -- the indices of metadata records that
matched the predicate. For a predicate with selectivity 0.1 over 10M
records, the output record contains approximately 1M ordinal values.

## Role in dataset pipelines

This command is typically the final step in building a filtered-search
ground truth dataset:

1. `synthesize predicates` generates PNode predicate trees from metadata
   statistics.
2. `compute predicates` evaluates those predicates against the full
   metadata slab to produce per-predicate answer keys.

The answer keys are then used by filtered-KNN ground truth computation:
for each query, only base vectors whose ordinals appear in the
corresponding answer key are considered as candidate neighbors. This
precomputation avoids re-evaluating predicates during the expensive KNN
sweep.

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

        let mode = options.get("mode").unwrap_or("survey");
        if mode == "simple-int-eq" {
            return self.execute_simple_int_eq(options, ctx, start);
        }

        let input_str = match options.require("source") {
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

        let survey_str = match options.require("survey") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let input_path = resolve_path(input_str, &ctx.workspace);
        let predicates_path = resolve_path(predicates_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let survey_path = resolve_path(survey_str, &ctx.workspace);

        // Load survey data
        let survey = match survey_from_json(&survey_path) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to load survey: {}", e), start),
        };
        ctx.ui.log(&format!(
            "compute predicates: survey loaded: {} fields, {} records sampled, {} total records",
            survey.field_stats.len(),
            survey.sampled,
            survey.total_records,
        ));

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

        // Parse optional range to limit ordinal scanning to a profile subset.
        let range_start: i64;
        let range_end: Option<i64>;
        if let Some(range_str) = options.get("range") {
            match parse_ordinal_range(range_str) {
                Ok((s, e)) => {
                    range_start = s;
                    range_end = e;
                }
                Err(e) => return error_result(format!("invalid range '{}': {}", range_str, e), start),
            }
        } else {
            range_start = 0;
            range_end = None;
        }

        let compress_cache: bool = options
            .get("compress-cache")
            .map(|s| s == "true" || s == "1")
            .unwrap_or(false);

        // ── Phase 1: Load predicates and memoize ─────────────────────────

        let pred_reader = match SlabReader::open(&predicates_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open predicates slab {}: {}", predicates_path.display(), e),
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
                        ctx.ui.log(&format!(
                            "compute predicates: skipping predicate {}: {}",
                            ord, e
                        ));
                    }
                },
                Err(e) => {
                    ctx.ui.log(&format!(
                        "compute predicates: error reading predicate {}: {}",
                        ord, e
                    ));
                }
            }
        }

        let pred_len = predicates.len();
        ctx.ui.log(&format!(
            "compute predicates: loaded {} predicates from {}",
            pred_len,
            predicates_path.display(),
        ));

        // ── Predicate structure analysis ─────────────────────────────────
        if !predicates.is_empty() {
            // Build histogram of structural fingerprints
            let mut fingerprint_counts: indexmap::IndexMap<String, (usize, usize)> =
                indexmap::IndexMap::new();
            for (i, pred) in predicates.iter().enumerate() {
                let fp_str = pred.fingerprint().to_string();
                let entry = fingerprint_counts.entry(fp_str).or_insert((0, i));
                entry.0 += 1;
            }
            // Sort by descending count
            fingerprint_counts.sort_by(|_, a, _, b| b.0.cmp(&a.0));

            let num_forms = fingerprint_counts.len();
            if num_forms == 1 {
                let (fp_str, (count, example_idx)) = fingerprint_counts.iter().next().unwrap();
                ctx.ui.log(&format!(
                    "compute predicates: all {} predicates share one structural form:",
                    count,
                ));
                ctx.ui.log(&format!("    form: {}", fp_str));
                ctx.ui.log(&format!(
                    "    example[{}]: {}",
                    example_idx, predicates[*example_idx],
                ));
            } else {
                ctx.ui.log(&format!(
                    "compute predicates: {} distinct predicate forms across {} predicates:",
                    num_forms, pred_len,
                ));
                for (fp_str, (count, example_idx)) in &fingerprint_counts {
                    let pct = *count as f64 / pred_len as f64 * 100.0;
                    ctx.ui.log(&format!(
                        "    {:>6} ({:5.1}%) — {}",
                        count, pct, fp_str,
                    ));
                    ctx.ui.log(&format!(
                        "             example[{}]: {}",
                        example_idx, predicates[*example_idx],
                    ));
                }
            }
        }

        // Memoize: flatten into field-indexed conditions
        let memo = memoize_predicates(&predicates);
        ctx.ui.log(&format!(
            "compute predicates: memoized {} field-indexed, {} fallback, {} fields",
            memo.count - memo.fallback.len(),
            memo.fallback.len(),
            memo.field_conditions.len(),
        ));

        // ── Phase 2: Open metadata slab ──────────────────────────────────

        let index_pb: std::cell::RefCell<Option<veks_core::ui::ProgressHandle>> = std::cell::RefCell::new(None);
        let ui = &ctx.ui;
        let meta_reader = match SlabReader::open_with_progress(&input_path, |p| match p {
            OpenProgress::PagesPageRead { page_count } => {
                *index_pb.borrow_mut() = Some(ui.bar(*page_count as u64, "slab index"));
            }
            OpenProgress::IndexBuild { done, total: _ } => {
                if let Some(ref bar) = *index_pb.borrow() {
                    bar.set_position(*done as u64);
                }
            }
            OpenProgress::IndexComplete { total_records } => {
                if let Some(bar) = index_pb.borrow_mut().take() {
                    bar.finish();
                }
                ui.log(&format!("    slab index: complete, {} total records", total_records));
            }
        }) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open metadata slab {}: {}", input_path.display(), e),
                    start,
                )
            }
        };

        let slab_total_records = meta_reader.total_records();

        // Apply range to limit scanning to a subset of ordinals.
        let effective_start = range_start.max(0);
        let effective_end = match range_end {
            Some(e) => (e as u64).min(slab_total_records),
            None => slab_total_records,
        };
        let total_records = effective_end.saturating_sub(effective_start as u64);

        if range_end.is_some() || range_start > 0 {
            ctx.ui.log(&format!(
                "compute predicates: range [{}, {}) — scanning {} of {} total records",
                effective_start, effective_end, total_records, slab_total_records,
            ));
        }

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
                                    ctx.ui.log(&format!(
                                        "compute predicates: schema discovered: {} fields",
                                        s.field_count,
                                    ));
                                    Some(s)
                                }
                                Err(e) => {
                                    ctx.ui.log(&format!(
                                        "compute predicates: schema discovery failed: {}, using fallback path",
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
            // Use explicit selectivity, or estimate from survey distinct counts
            let est_selectivity = selectivity.unwrap_or_else(|| {
                let est = estimate_selectivity_from_survey(&memo, &survey);
                ctx.ui.log(&format!(
                    "compute predicates: selectivity estimated from survey: {:.4}%",
                    est * 100.0,
                ));
                est
            });
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
                    ctx.ui.log(
                        "compute predicates: memory-aware segment sizing");
                    ctx.ui.log(&format!(
                        "    estimated {}/segment ({} preds × {} matches × 4B × {} threads)",
                        format_bytes(bytes_per_segment),
                        pred_len, est_matches_per_pred, num_threads,
                    ));
                    ctx.ui.log(&format!(
                        "    available memory: {} (RSS: {}, ceiling: {})",
                        format_bytes(available),
                        format_bytes(snapshot.rss_bytes),
                        format_bytes(mem_ceiling),
                    ));
                    ctx.ui.log(&format!(
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
        // memory.
        let cache_dir: PathBuf = ctx.cache.clone();
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            return error_result(
                format!("cannot create cache dir {}: {}", cache_dir.display(), e),
                start,
            );
        }

        // Cache prefix from input file stems so overlapping profile ranges
        // share cached segments (same fix as compute_knn).
        let cache_prefix = {
            let input_stem = input_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();
            let pred_stem = predicates_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();
            // Include metadata file size in the cache key so cached segments
            // are invalidated when the data changes (e.g., different fraction).
            let input_size = std::fs::metadata(&input_path)
                .map(|m| m.len())
                .unwrap_or(0);
            format!("{}.{}.{}", input_stem, pred_stem, input_size)
        };

        let segments = plan_segments(
            &page_entries,
            effective_end,
            segment_size,
            Some(cache_dir.as_path()),
            &cache_prefix,
            pred_len,
            effective_start,
            compress_cache,
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

            ctx.ui.log("compute predicates: summary");
            ctx.ui.log(&format!("    predicates:             {}", pred_len));
            ctx.ui.log(&format!("    records:                {}", total_records));
            ctx.ui.log(&format!("    unique fields targeted: {}", memo.field_conditions.len()));
            ctx.ui.log(&format!("    total conditions:       {}", total_conditions));
            ctx.ui.log(&format!("    total comparands:       {}", total_comparands));
            ctx.ui.log(&format!("    comparisons/record:     {}", comparisons_per_record));
            ctx.ui.log(&format!("    total comparisons:      {}", total_comparisons));
            ctx.ui.log(&format!("    conditions/predicate:   min={} max={} mean={:.1}",
                min_conds, max_conds, mean_conds));
            ctx.ui.log(&format!("    limit/predicate:        {}", limit_desc));
            ctx.ui.log(&format!("    segments:               {} ({} cached, {} to scan)",
                segments.len(), cached_count, uncached_count));
            ctx.ui.log(&format!("    segment size:           {} records", segment_size));
            ctx.ui.log(&format!("    threads:                {}", num_threads));
            ctx.ui.log("");

            // Output size math breakdown
            let gib = 1_073_741_824.0f64;
            ctx.ui.log(&format!("    max output size:        {:.1} GB", max_output_bytes as f64 / gib));
            ctx.ui.log(&format!("      = {} ordinals/pred × 4 bytes × {} preds",
                max_ordinals_per_pred, pred_len));
            ctx.ui.log("");

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
                    ctx.ui.log(&format!("    selectivity:            {:.4}%  (from predicate generation)",
                        sel_min * 100.0));
                } else {
                    ctx.ui.log(&format!("    selectivity:            {:.4}% .. {:.4}%  (from predicate generation)",
                        sel_min * 100.0, sel_max * 100.0));
                }
                let (matches, indices, bytes) = estimate_fn(sel_min);
                ctx.ui.log(&format!("    est. matches/pred:      {}", matches));
                ctx.ui.log(&format!("    est. total indices:     {}", indices));
                ctx.ui.log(&format!("    est. output size:       {}", format_bytes(bytes)));
                if sel_min != sel_max {
                    let (matches_hi, indices_hi, bytes_hi) = estimate_fn(sel_max);
                    ctx.ui.log(&format!("    est. matches/pred (hi): {}", matches_hi));
                    ctx.ui.log(&format!("    est. total indices (hi):{}", indices_hi));
                    ctx.ui.log(&format!("    est. output size (hi):  {}", format_bytes(bytes_hi)));
                }
            } else {
                ctx.ui.log("    estimated output by selectivity (pass selectivity= to narrow):");
                ctx.ui.log(&format!("    {:>12}  {:>14}  {:>14}  {:>10}",
                    "selectivity", "matches/pred", "total indices", "data size"));
                for &sel in &sel_targets {
                    let (matches_per_pred, total_indices, data_bytes) = estimate_fn(sel);
                    ctx.ui.log(&format!("    {:>11.1}%  {:>14}  {:>14}  {:>10}",
                        sel * 100.0,
                        matches_per_pred,
                        total_indices,
                        format_bytes(data_bytes)));
                }
            }
        }

        // ── Phase 5: Process segments ────────────────────────────────────
        //
        // Each segment covers ~1M metadata records (aligned with KNN partition
        // boundaries for tandem use). Results are cached per-segment so
        // interrupted runs resume and subsets can be recomposed later.

        // Governor checkpoint before segment processing
        ctx.governor.checkpoint();

        let records_done = AtomicU64::new(0);
        let errors_done = AtomicU64::new(0);
        let total_matches_counter = AtomicU64::new(0);
        let scan_start = Instant::now();

        // Memory profile: baseline before scan (before bars, so log doesn't
        // disrupt the multi-progress region).
        let scan_baseline = MemProfile::sample();

        // Progress bars — ghost-line artifacts from sequential bar creation
        // will be resolved by the ratatui migration (ui module).
        let scan_pb = ctx.ui.bar(total_records, "scanning records");
        let seg_pb = ctx.ui.bar(segments.len() as u64, "segments");
        let sel_bar = ctx.ui.ratio(10_000, "selectivity");

        if uncached_count > 0 {
            meta_reader.advise_sequential();

            // Process segments sequentially; parallelise pages *within* each
            // segment.  This gives sequential I/O, bounded memory (one
            // segment's matches at a time), and ordered cache writes.

            let seg_byte_range = |seg: &SegmentInfo| -> (usize, usize) {
                let start = page_entries[seg.page_start_idx].file_offset as usize;
                let end = if seg.page_end_idx < page_entries.len() {
                    page_entries[seg.page_end_idx].file_offset as usize
                } else {
                    meta_reader.file_len().unwrap_or(0) as usize
                };
                (start, end)
            };

            for (seg_idx, seg) in segments.iter().enumerate() {
                if seg.cached {
                    // Count cached records and segments toward progress.
                    let seg_records = (seg.end_ordinal - seg.start_ordinal) as u64;
                    records_done.fetch_add(seg_records, Ordering::Relaxed);
                    scan_pb.set_position(records_done.load(Ordering::Relaxed));
                    seg_pb.set_position((seg_idx + 1) as u64);
                    continue;
                }

                // Prefetch this segment's byte range.
                let (byte_start, byte_end) = seg_byte_range(seg);
                meta_reader.prefetch_range(byte_start, byte_end);

                // Divide pages among threads.
                let seg_pages = seg.page_end_idx - seg.page_start_idx;
                let effective_threads = num_threads.min(seg_pages).max(1);

                // Each thread produces its own per-predicate match vectors.
                let thread_results: Vec<Vec<Vec<i32>>> =
                    std::thread::scope(|s| {
                        let reader = &meta_reader;
                        let compiled_scan = &compiled_scan;
                        let memo = &memo;
                        let records_done = &records_done;
                        let errors_done = &errors_done;
                        let scan_pb = &scan_pb;
                        let page_entries = &page_entries;

                        let handles: Vec<_> = (0..effective_threads)
                            .map(|tid| {
                                // Divide pages for this thread (contiguous slice).
                                let pages_per = seg_pages / effective_threads;
                                let extra = seg_pages % effective_threads;
                                let t_start = seg.page_start_idx
                                    + tid * pages_per
                                    + tid.min(extra);
                                let t_end = t_start
                                    + pages_per
                                    + if tid < extra { 1 } else { 0 };

                                s.spawn(move || {
                                    let mut local_matches: Vec<Vec<i32>> =
                                        vec![Vec::new(); pred_len];
                                    let mut pass_counts = vec![0u32; pred_len];
                                    let mut local_rec_count = 0u64;

                                    for pi in t_start..t_end {
                                        let entry = &page_entries[pi];
                                        let page_buf = reader.page_buf(entry);
                                        let rec_count =
                                            match Page::record_count_from_buf(page_buf) {
                                                Ok(rc) => rc,
                                                Err(_) => continue,
                                            };

                                        let page_start = entry.start_ordinal;

                                        // Clip to segment ordinal range (pages
                                        // straddling a boundary appear in both
                                        // adjacent segments).
                                        let first_rec =
                                            if page_start < seg.start_ordinal {
                                                (seg.start_ordinal - page_start) as usize
                                            } else {
                                                0
                                            };
                                        let last_rec = if page_start + rec_count as i64
                                            > seg.end_ordinal
                                        {
                                            (seg.end_ordinal - page_start) as usize
                                        } else {
                                            rec_count
                                        };

                                        for rec_idx in first_rec..last_rec {
                                            let data =
                                                match Page::get_record_ref_from_buf(
                                                    page_buf, rec_idx, rec_count,
                                                ) {
                                                    Ok(d) => d,
                                                    Err(_) => continue,
                                                };

                                            let ordinal =
                                                (page_start + rec_idx as i64) as i32;

                                            for c in pass_counts.iter_mut() {
                                                *c = 0;
                                            }

                                            let schema_matched = match scan_record(
                                                data,
                                                compiled_scan,
                                                &mut pass_counts,
                                            ) {
                                                Ok(matched) => matched,
                                                Err(_) => {
                                                    errors_done.fetch_add(
                                                        1,
                                                        Ordering::Relaxed,
                                                    );
                                                    continue;
                                                }
                                            };

                                            if schema_matched {
                                                for i in 0..pred_len {
                                                    let required =
                                                        compiled_scan.required_count(i);
                                                    if required != u32::MAX
                                                        && pass_counts[i] == required
                                                    {
                                                        if limit == 0
                                                            || local_matches[i].len()
                                                                < limit
                                                        {
                                                            local_matches[i]
                                                                .push(ordinal);
                                                        }
                                                    }
                                                }

                                                if !compiled_scan
                                                    .fallback()
                                                    .is_empty()
                                                {
                                                    match MNode::from_bytes(data) {
                                                        Ok(mnode) => {
                                                            for (idx, pnode) in
                                                                compiled_scan.fallback()
                                                            {
                                                                if evaluate(
                                                                    pnode, &mnode,
                                                                ) {
                                                                    if limit == 0
                                                                        || local_matches
                                                                            [*idx]
                                                                            .len()
                                                                            < limit
                                                                    {
                                                                        local_matches
                                                                            [*idx]
                                                                            .push(
                                                                                ordinal,
                                                                            );
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
                                                        errors_done.fetch_add(
                                                            1,
                                                            Ordering::Relaxed,
                                                        );
                                                    }
                                                }
                                            }

                                            local_rec_count += 1;
                                            if local_rec_count % 100 == 0 {
                                                let total = records_done
                                                    .fetch_add(100, Ordering::Relaxed)
                                                    + 100;
                                                scan_pb.set_position(total);
                                            }
                                        }

                                        let leftover = local_rec_count % 100;
                                        if leftover > 0 {
                                            records_done.fetch_add(
                                                leftover,
                                                Ordering::Relaxed,
                                            );
                                            local_rec_count = 0;
                                        }
                                    }

                                    local_matches
                                })
                            })
                            .collect();

                        handles
                            .into_iter()
                            .map(|h| h.join().unwrap())
                            .collect()
                    });

                // Merge thread-local matches into segment matches.
                let mut seg_matches: Vec<Vec<i32>> = vec![Vec::new(); pred_len];
                for thread_match in thread_results {
                    for (pi, mut ords) in thread_match.into_iter().enumerate() {
                        seg_matches[pi].append(&mut ords);
                    }
                }

                // Sort ordinals within each predicate (threads processed
                // contiguous page ranges so this is nearly sorted already).
                for ords in &mut seg_matches {
                    ords.sort_unstable();
                    if limit > 0 {
                        ords.truncate(limit);
                    }
                }

                // Tally segment matches for running selectivity.
                let seg_match_count: u64 =
                    seg_matches.iter().map(|v| v.len() as u64).sum();
                total_matches_counter.fetch_add(seg_match_count, Ordering::Relaxed);

                // Update selectivity display (bar scaled to 0.0–1.0 via 0–10000).
                let total_m = total_matches_counter.load(Ordering::Relaxed);
                let total_r = records_done.load(Ordering::Relaxed);
                if total_r > 0 && pred_len > 0 {
                    let avg_matches_per_pred = total_m as f64 / pred_len as f64;
                    let eff_sel = avg_matches_per_pred / total_r as f64;
                    let sel_scaled = (eff_sel * 10_000.0).round().min(10_000.0) as u64;
                    sel_bar.set_position(sel_scaled);
                    sel_bar.set_message(format!(
                        "{:.4}% ({:.0} avg/pred)",
                        eff_sel * 100.0,
                        avg_matches_per_pred,
                    ));
                }

                // Write segment cache.
                match &seg.cache_path {
                    Some(cache_path) => {
                        if let Err(e) = write_segment_cache(cache_path, &seg_matches) {
                            log::warn!(
                                "compute predicates: cache write error for segment {}: {} (path: {})",
                                seg_idx, e, cache_path.display(),
                            );
                        } else if compress_cache {
                            // Read back the slab file, compress, and delete the original.
                            match std::fs::read(cache_path) {
                                Ok(raw) => {
                                    if let Err(e) = crate::pipeline::gz_cache::save_gz(cache_path, &raw) {
                                        log::warn!(
                                            "compute predicates: compress error for segment {}: {}",
                                            seg_idx, e,
                                        );
                                    } else {
                                        let _ = std::fs::remove_file(cache_path);
                                    }
                                }
                                Err(e) => {
                                    log::warn!(
                                        "compute predicates: failed to read cache for compress, segment {}: {}",
                                        seg_idx, e,
                                    );
                                }
                            }
                        }
                    }
                    None => {
                        log::warn!(
                            "compute predicates: BUG — segment {} has no cache_path",
                            seg_idx,
                        );
                    }
                }

                seg_pb.set_position((seg_idx + 1) as u64);

                // Release madvise for this segment's byte range.
                meta_reader.release_range(byte_start, byte_end);

                if ctx.governor.checkpoint() {
                    log::info!(
                        "compute predicates: governor throttle at segment {}",
                        seg_idx,
                    );
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        }

        // Governor checkpoint after segment processing
        if ctx.governor.checkpoint() {
            log::info!("compute predicates: governor throttle signal received");
        }

        scan_pb.finish();
        seg_pb.finish();

        // Finalize selectivity display
        let total_match_count = total_matches_counter.load(Ordering::Relaxed);
        let total_scanned = records_done.load(Ordering::Relaxed);
        let effective_selectivity = if total_scanned > 0 && pred_len > 0 {
            let avg = total_match_count as f64 / pred_len as f64;
            avg / total_scanned as f64
        } else {
            0.0
        };
        sel_bar.finish();
        ctx.ui.log(&format!(
            "    effective selectivity: {:.4}% ({} total matches across {} predicates, {} records scanned)",
            effective_selectivity * 100.0,
            total_match_count,
            pred_len,
            total_scanned,
        ));
        if let Some(target) = selectivity {
            let ratio = if target > 0.0 { effective_selectivity / target } else { 0.0 };
            if ratio > 2.0 {
                ctx.ui.log(&format!(
                    "    WARNING: effective selectivity {:.4}% is {:.0}x higher than target {:.4}% — predicates may be too broad",
                    effective_selectivity * 100.0,
                    ratio,
                    target * 100.0,
                ));
            }
        }

        let total_errors = errors_done.load(Ordering::Relaxed);
        let scan_elapsed = scan_start.elapsed();
        let final_rate = if scan_elapsed.as_secs_f64() > 0.0 {
            total_scanned as f64 / scan_elapsed.as_secs_f64()
        } else {
            0.0
        };
        ctx.ui.log(&format!(
            "    scan complete: {} records in {:.1}s ({:.0} rec/s), {} decode errors",
            total_scanned,
            scan_elapsed.as_secs_f64(),
            final_rate,
            total_errors,
        ));

        // Memory profile dump
        {
            let scan_end = MemProfile::sample();
            log::info!("mem profile (scan end):   {}", scan_end.summary());
            log::info!("mem profile (scan delta): {}", scan_end.delta_from(&scan_baseline));
        }

        // ── Phase 6: Open segment caches ──────────────────────────────────

        let cache_readers: Vec<Option<SlabReader>> = segments
            .iter()
            .enumerate()
            .map(|(seg_idx, seg)| {
                seg.cache_path.as_ref().and_then(|cp| {
                    // If compress-cache is enabled and the uncompressed file is
                    // missing but the .gz form exists, decompress it first.
                    if compress_cache && !cp.exists() && crate::pipeline::gz_cache::gz_exists(cp) {
                        match crate::pipeline::gz_cache::load_gz(cp) {
                            Ok(data) => {
                                if let Err(e) = std::fs::write(cp, &data) {
                                    ctx.ui.log(&format!(
                                        "compute predicates: decompress write error for segment {}: {}",
                                        seg_idx, e
                                    ));
                                    return None;
                                }
                            }
                            Err(e) => {
                                ctx.ui.log(&format!(
                                    "compute predicates: decompress error for segment {}: {}",
                                    seg_idx, e
                                ));
                                return None;
                            }
                        }
                    }
                    match SlabReader::open(cp) {
                        Ok(r) => Some(r),
                        Err(e) => {
                            ctx.ui.log(&format!(
                                "compute predicates: error opening cache for segment {}: {}",
                                seg_idx, e
                            ));
                            None
                        }
                    }
                })
            })
            .collect();

        // ── Phase 7: Stream merge + write output slab ────────────────────

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
        for pi in 0..pred_len {
            let mut buf: Vec<u8> = Vec::new();
            let mut count = 0usize;
            for reader_opt in &cache_readers {
                if limit > 0 && count >= limit {
                    break;
                }
                if let Some(reader) = reader_opt {
                    match reader.get(pi as i64) {
                        Ok(data) => {
                            let ords = read_ordinals(&data);
                            let take = if limit > 0 {
                                ords.len().min(limit - count)
                            } else {
                                ords.len()
                            };
                            for &ord in &ords[..take] {
                                buf.write_i32::<LittleEndian>(ord).unwrap();
                            }
                            count += take;
                        }
                        Err(e) => {
                            ctx.ui.log(&format!(
                                "compute predicates: error reading cache pred {}: {}",
                                pi, e
                            ));
                        }
                    }
                }
            }
            total_matches += count as u64;
            if let Err(e) = writer.add_record(&buf) {
                return error_result(
                    format!("write error at predicate {}: {}", pi, e),
                    start,
                );
            }
        }

        drop(cache_readers);

        if let Err(e) = writer.finish() {
            return error_result(format!("finish error: {}", e), start);
        }

        // Save merged result as a cache super-segment for reuse by larger profiles
        if segments.len() > 1 {
            let super_path = cache_dir.join(format!(
                "{}.seg_{:010}_{:010}.predkeys.slab",
                cache_prefix, effective_start, effective_end,
            ));
            if !super_path.exists() {
                let _ = std::fs::copy(&output_path, &super_path);
                ctx.ui.log(&format!(
                    "  cached merged result as segment [{}, {}) for reuse by larger profiles",
                    effective_start, effective_end,
                ));
            }
        }

        let avg_matches = if pred_len > 0 {
            total_matches / pred_len as u64
        } else {
            0
        };
        let message = format!(
            "{} metadata indices generated from {} predicates × {} records ({} avg matches), {:.1}s",
            pred_len,
            pred_len,
            total_scanned,
            avg_matches,
            start.elapsed().as_secs_f64(),
        );
        ctx.ui.log(&format!("compute predicates: {}", message));

        // Write verified count for the bound checker
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &pred_len.to_string());
        ctx.defaults.insert(var_name, pred_len.to_string());

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
                "source",
                "Path",
                true,
                None,
                "Metadata source (slab or ivec)",
                OptionRole::Input,
            ),
            opt(
                "predicates",
                "Path",
                true,
                None,
                "Predicate slab (PNode records)",
                OptionRole::Input,
            ),
            opt(
                "output",
                "Path",
                true,
                None,
                "Output slab for answer key records",
                OptionRole::Output,
            ),
            opt(
                "survey",
                "Path",
                false,
                None,
                "Survey JSON file (not needed for simple-int-eq mode)",
                OptionRole::Input,
            ),
            opt(
                "mode",
                "string",
                false,
                Some("survey"),
                "Evaluation mode: 'survey' (slab-based) or 'simple-int-eq' (ivec integer equality)",
                OptionRole::Config,
            ),
            opt(
                "fields",
                "int",
                false,
                Some("1"),
                "Number of integer fields (simple-int-eq mode)",
                OptionRole::Config,
            ),
            opt(
                "limit",
                "int",
                false,
                Some("0"),
                "Max matches per predicate (0 = unlimited)",
                OptionRole::Config,
            ),
            opt(
                "segment_size",
                "int",
                false,
                Some("1000000"),
                "Records per segment for cache partitioning",
                OptionRole::Config,
            ),
            opt(
                "selectivity",
                "float",
                false,
                None,
                "Target selectivity from predicate generation (0.0–1.0), for output size estimates",
                OptionRole::Config,
            ),
            opt(
                "selectivity-max",
                "float",
                false,
                None,
                "Upper bound of selectivity range (if predicates used a range)",
                OptionRole::Config,
            ),
            opt(
                "range",
                "String",
                false,
                None,
                "Ordinal range to scan, e.g. '[0,10000000)'. Limits metadata scanning to a profile subset.",
                OptionRole::Config,
            ),
            opt(
                "compress-cache",
                "bool",
                false,
                Some("false"),
                "Gzip-compress segment cache files",
                OptionRole::Config,
            ),
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source", "predicates", "survey"],
            &["output"],
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Estimate selectivity from survey field statistics and memoized predicates.
///
/// For each targeted field, computes `1 / distinct_count` as the probability
/// that a random record matches an equality condition. The overall selectivity
/// is the product across fields (assuming independence), clamped to a
/// conservative range of 0.001–0.50.
fn estimate_selectivity_from_survey(
    memo: &MemoizedPredicates,
    survey: &super::slab::SurveyResult,
) -> f64 {
    let mut selectivities: Vec<f64> = Vec::new();
    for field_name in memo.field_conditions.keys() {
        if let Some(fs) = survey.field_stats.get(field_name) {
            let distinct = fs.distinct.len().max(1) as f64;
            // If distinct values overflowed the sample, use the sample count
            let effective_distinct = if fs.distinct_overflow {
                fs.count.max(1) as f64
            } else {
                distinct
            };
            selectivities.push(1.0 / effective_distinct);
        }
    }

    if selectivities.is_empty() {
        return 0.10; // no field info available, conservative default
    }

    // Product of per-field selectivities (assumes independence)
    let combined: f64 = selectivities.iter().product();
    combined.clamp(0.001, 0.50)
}

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
    role: OptionRole,
) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        role,
}
}

/// Parse an ordinal range specification like `[0,10000000)` or `[0..10M)`.
///
/// Returns `(start, Option<end>)` where `end` is `None` for open-ended ranges.
fn parse_ordinal_range(s: &str) -> Result<(i64, Option<i64>), String> {
    let s = s.trim();

    let left_exclusive = s.starts_with('(');
    let right_inclusive = s.ends_with(']');
    let has_left = s.starts_with('[') || s.starts_with('(');
    let has_right = s.ends_with(')') || s.ends_with(']');

    let inner = if has_left { &s[1..] } else { s };
    let inner = if has_right { &inner[..inner.len() - 1] } else { inner };
    let inner = inner.trim();

    // "all" pattern: empty inner means full range
    if inner == ".." || inner.is_empty() {
        return Ok((0, None));
    }

    let sep = if inner.contains(',') { "," } else { ".." };
    let parts: Vec<&str> = inner.splitn(2, sep).collect();
    if parts.len() != 2 {
        return Err(format!("expected 'start{}end' format", sep));
    }

    let left_str = parts[0].trim();
    let right_str = parts[1].trim();

    let start = if left_str.is_empty() {
        0i64
    } else {
        let v = vectordata::dataset::source::parse_number_with_suffix(left_str)? as i64;
        if left_exclusive { v + 1 } else { v }
    };

    let end = if right_str.is_empty() {
        None
    } else {
        let v = vectordata::dataset::source::parse_number_with_suffix(right_str)? as i64;
        Some(if right_inclusive { v + 1 } else { v })
    };

    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use veks_core::formats::mnode::{MNode, MValue};
    use veks_core::formats::pnode::{
        Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
    };
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &std::path::Path) -> StreamContext {
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

    /// Create a minimal survey JSON for the test records produced by [`make_test_records`].
    fn create_test_survey(dir: &std::path::Path, record_count: usize) -> std::path::PathBuf {
        let path = dir.join("survey.json");
        let json = serde_json::json!({
            "sampled": record_count,
            "total_records": record_count,
            "non_mnode_count": 0,
            "decode_errors": 0,
            "fields": {
                "user_id": {
                    "count": record_count,
                    "null_count": 0,
                    "types": { "int": record_count },
                    "numeric": { "min": 0, "max": record_count - 1, "mean": (record_count - 1) as f64 / 2.0, "count": record_count },
                    "distinct": {},
                    "distinct_overflow": true
                },
                "name": {
                    "count": record_count,
                    "null_count": 0,
                    "types": { "text": record_count },
                    "distinct": { "user_0": 4, "user_1": 4, "user_2": 4, "user_3": 4, "user_4": 4 },
                    "distinct_overflow": false
                },
                "score": {
                    "count": record_count,
                    "null_count": 0,
                    "types": { "float": record_count },
                    "numeric": { "min": 0.0, "max": 28.5, "mean": 14.25, "count": record_count },
                    "distinct": {},
                    "distinct_overflow": true
                },
                "active": {
                    "count": record_count,
                    "null_count": 0,
                    "types": { "bool": record_count },
                    "distinct": { "true": record_count / 2, "false": record_count / 2 },
                    "distinct_overflow": false
                }
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        path
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
        opts.set("source", meta_path.to_string_lossy().to_string());
        opts.set("predicates", pred_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        let survey_path = create_test_survey(ws, 20);
        opts.set("survey", survey_path.to_string_lossy().to_string());

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
        opts.set("source", meta_path.to_string_lossy().to_string());
        opts.set("predicates", pred_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("limit", "3".to_string());
        let survey_path = create_test_survey(ws, 20);
        opts.set("survey", survey_path.to_string_lossy().to_string());

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
        opts.set("source", meta_path.to_string_lossy().to_string());
        opts.set("predicates", pred_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        let survey_path = create_test_survey(ws, 20);
        opts.set("survey", survey_path.to_string_lossy().to_string());

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
        opts.set("source", meta_path.to_string_lossy().to_string());
        opts.set("predicates", pred_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        let survey_path = create_test_survey(ws, 20);
        opts.set("survey", survey_path.to_string_lossy().to_string());

        let mut op = GenPredicateKeysOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        let reader = SlabReader::open(&output_path).unwrap();
        let data = reader.get(0).unwrap();
        assert_eq!(data.len(), 0, "expected empty record for no matches");
    }
}
