// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: deduplicate vectors via external merge-sort.
//!
//! Uses a multi-phase external sort with adaptive prefix keys:
//!
//! 1. **Dimension sampling** — sparse sample to determine how many leading
//!    vector components (1–10) are needed to distinguish vectors. Measures
//!    per-dimension distinctness in the sample.
//!
//! 2. **Sorted run creation** — process input in batches, sort by
//!    **lexicographic order of component values** (not norm), and write
//!    intermediate sorted runs to cache. Each run record is
//!    `(ordinal:u32, prefix[0..N]:f32)`. Runs are cached for resume.
//!
//! 3. **K-way streaming merge** — merge all sorted runs using prefix
//!    components for comparison. Only when two adjacent entries have
//!    identical prefixes does the merge read the full vectors from the
//!    source file for exact equality comparison. This avoids random I/O
//!    for the vast majority of comparisons.
//!
//! 4. **Output** — write the merged (optionally deduplicated) ordinal
//!    sequence as an ivec file, plus a JSON report.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

// ---------------------------------------------------------------------------
// Format-agnostic vector reader (f32 and f16 → f32 upcast)
// ---------------------------------------------------------------------------

/// Uniform read interface that always returns `Vec<f32>`, upcasting f16
/// when the source is an mvec file.
enum VecReader {
    F32(MmapVectorReader<f32>),
    F16(MmapVectorReader<half::f16>),
}

impl VecReader {
    /// Open the appropriate reader based on file extension.
    fn open(path: &Path) -> Result<Self, String> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("mvec") | Some("mvecs") => {
                MmapVectorReader::<half::f16>::open_mvec(path)
                    .map(VecReader::F16)
                    .map_err(|e| format!("failed to open {}: {}", path.display(), e))
            }
            _ => {
                MmapVectorReader::<f32>::open_fvec(path)
                    .map(VecReader::F32)
                    .map_err(|e| format!("failed to open {}: {}", path.display(), e))
            }
        }
    }

    fn count(&self) -> usize {
        match self {
            VecReader::F32(r) => <MmapVectorReader<f32> as VectorReader<f32>>::count(r),
            VecReader::F16(r) => <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(r),
        }
    }

    fn dim(&self) -> usize {
        match self {
            VecReader::F32(r) => <MmapVectorReader<f32> as VectorReader<f32>>::dim(r),
            VecReader::F16(r) => <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(r),
        }
    }

    /// Get vector at index as f32 (upcasting f16 if needed).
    fn get_f32(&self, index: usize) -> Result<Vec<f32>, String> {
        match self {
            VecReader::F32(r) => r.get(index)
                .map_err(|e| format!("failed to read vector {}: {}", index, e)),
            VecReader::F16(r) => {
                let v = r.get(index)
                    .map_err(|e| format!("failed to read vector {}: {}", index, e))?;
                Ok(v.iter().map(|x| x.to_f32()).collect())
            }
        }
    }

    /// Bitwise equality check — works at native precision (no upcast).
    fn vectors_equal(&self, a: usize, b: usize) -> bool {
        match self {
            VecReader::F32(r) => {
                let va = match r.get(a) { Ok(v) => v, Err(_) => return false };
                let vb = match r.get(b) { Ok(v) => v, Err(_) => return false };
                va.len() == vb.len() && va.iter().zip(vb.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            VecReader::F16(r) => {
                let va = match r.get(a) { Ok(v) => v, Err(_) => return false };
                let vb = match r.get(b) { Ok(v) => v, Err(_) => return false };
                va.len() == vb.len() && va.iter().zip(vb.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
            }
        }
    }

    fn format_name(&self) -> &'static str {
        match self {
            VecReader::F32(_) => "f32",
            VecReader::F16(_) => "f16",
        }
    }
}

/// Pipeline command: deduplicate vectors.
pub struct ComputeDedupOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeDedupOp)
}

/// Default batch size for sorted run creation.
const DEFAULT_BATCH_SIZE: usize = 1_000_000;
/// Number of vectors to sample for variance estimation.
const VARIANCE_SAMPLE_SIZE: usize = 10_000;
/// Minimum prefix components.
const MIN_PREFIX: usize = 1;
/// Maximum prefix components.
const MAX_PREFIX: usize = 10;
/// BufReader/BufWriter capacity (1 MiB).
const IO_BUF_SIZE: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Run record: stored in intermediate sorted-run files on disk
// ---------------------------------------------------------------------------

/// A single record in a sorted run file.
///
/// Wire format: `[ordinal:u32 LE][prefix_0:f32 LE]...[prefix_N:f32 LE]`
/// where N = `prefix_width`.
#[derive(Clone)]
struct RunRecord {
    ordinal: u32,
    prefix: Vec<f32>,
}

impl RunRecord {
    /// Compare by prefix components using total lexicographic order.
    fn cmp_prefix(&self, other: &RunRecord) -> Ordering {
        for (a, b) in self.prefix.iter().zip(other.prefix.iter()) {
            let ord = float_to_sortable(*a).cmp(&float_to_sortable(*b));
            if ord != Ordering::Equal {
                return ord;
            }
        }
        // Tie-break by ordinal for stability
        self.ordinal.cmp(&other.ordinal)
    }

    /// Check if prefix components are identical (potential duplicate).
    fn prefix_eq(&self, other: &RunRecord) -> bool {
        self.prefix.len() == other.prefix.len()
            && self.prefix.iter().zip(other.prefix.iter())
                .all(|(a, b)| float_to_sortable(*a) == float_to_sortable(*b))
    }

    /// Write to a buffered writer.
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.ordinal)?;
        for &v in &self.prefix {
            w.write_f32::<LittleEndian>(v)?;
        }
        Ok(())
    }

    /// Read from a buffered reader. Returns None at EOF.
    fn read_from<R: Read>(r: &mut R, prefix_width: usize) -> std::io::Result<Option<Self>> {
        let ordinal = match r.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut prefix = Vec::with_capacity(prefix_width);
        for _ in 0..prefix_width {
            prefix.push(r.read_f32::<LittleEndian>()?);
        }
        Ok(Some(RunRecord { ordinal, prefix }))
    }
}

// ---------------------------------------------------------------------------
// Heap entry for k-way merge
// ---------------------------------------------------------------------------

/// Entry in the merge heap. Wraps a RunRecord with its source-run index.
struct HeapEntry {
    record: RunRecord,
    run_idx: usize,
}

// Min-heap: smallest prefix first
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.record.cmp_prefix(&other.record) == Ordering::Equal
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for min-heap (BinaryHeap is max-heap)
        other.record.cmp_prefix(&self.record)
    }
}

// ---------------------------------------------------------------------------
// CommandOp implementation
// ---------------------------------------------------------------------------

impl CommandOp for ComputeDedupOp {
    fn command_path(&self) -> &str {
        "compute dedup"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Detect and elide duplicate vectors via streaming merge-sort".into(),
            body: format!(r#"# compute dedup

Detect and elide duplicate vectors via streaming merge-sort deduplication.

## Description

Uses an external merge-sort with adaptive prefix keys to detect and
optionally remove exact duplicate vectors. Produces a **sorted-order
index** (ivec file) rather than a materialized vector copy.

### How It Works

1. **Dimension sampling** — reads a sparse sample of vectors to determine
   how many leading dimensions (1–10) are needed to distinguish most
   vectors. Measures per-dimension distinctness: if the first dimension
   alone separates 90%+ of the sample, one prefix component suffices.

2. **Sorted run creation** — processes the input in configurable batches
   (default 1M vectors). For each batch, sorts by **lexicographic order
   of component values** (dim 0 first, then dim 1, etc.), and writes
   a sorted run file to the cache directory. Each run record stores
   `(ordinal, prefix[0..N])` — not the full vector. Run files are
   preserved in `.cache/dedup_runs/` for resume across interrupted runs.

3. **K-way streaming merge** — opens all sorted run files and merges them
   using a min-heap over prefix keys. Adjacent entries with identical
   prefixes are compared by reading the full vectors from the source (via
   mmap random access). This short-circuits the vast majority of
   comparisons to prefix-only, avoiding random I/O for non-duplicates.

4. **Output** — the merged ordinal sequence (with duplicates optionally
   elided) is written as an ivec file. A JSON report records total,
   unique, and duplicate counts plus the prefix width chosen.

### Lexicographic Ordering

The output index is sorted by component values in lexicographic order,
not by norm. This means:
- **Zero vectors sort first** (all components are 0)
- **Binary search** for any specific vector is O(log N)
- **Exact duplicates are always adjacent**
- The index can answer "does vector X exist in this dataset?"

### Role in the Pipeline

Deduplication is a data-quality step that runs after import but before
shuffle, KNN, or metadata alignment. The output index can be fed to
`transform mvec-extract` to materialize a deduplicated dataset.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Sort buffers and run records".into(), adjustable: true },
            ResourceDesc { name: "threads".into(), description: "Batch sort parallelism".into(), adjustable: true },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let report_default = {
            let p = PathBuf::from(&output_str);
            p.with_extension("json").to_string_lossy().to_string()
        };
        let report_str = options.get("report")
            .map(|s| s.to_string())
            .unwrap_or(report_default);
        let duplicates_default = {
            let p = PathBuf::from(&output_str);
            let stem = p.file_stem().unwrap_or_default().to_string_lossy();
            let dir = p.parent().unwrap_or(Path::new("."));
            dir.join(format!("{}_duplicates.ivec", stem)).to_string_lossy().to_string()
        };
        let duplicates_str = options.get("duplicates")
            .map(|s| s.to_string())
            .unwrap_or(duplicates_default);
        let elide = options.get("elide")
            .map(|s| s != "false")
            .unwrap_or(true);
        let explicit_batch_size: Option<usize> = options.get("batch-size")
            .and_then(|s| s.parse().ok());

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let report_path = resolve_path(&report_str, &ctx.workspace);
        let duplicates_path = resolve_path(&duplicates_str, &ctx.workspace);

        // Open source vectors (auto-detect f32/f16 from extension)
        let reader = match VecReader::open(&source_path) {
            Ok(r) => r,
            Err(e) => return error_result(e, start),
        };

        let count = reader.count();
        let dim = reader.dim();

        // Derive batch size from governor memory budget when not explicitly set.
        // Each in-memory sort entry costs: sort key (4 + prefix_width) × 4 bytes
        // + RunRecord overhead (~(1 + prefix_width) × 4 bytes) + Vec overhead.
        // We estimate ~(16 + 8*MAX_PREFIX) bytes per entry conservatively.
        let batch_size = if let Some(explicit) = explicit_batch_size {
            explicit
        } else if let Some(mem_ceiling) = ctx.governor.mem_ceiling() {
            let snapshot = crate::pipeline::resource::SystemSnapshot::sample();
            let target = (mem_ceiling as f64 * 0.70) as u64;
            let available = if snapshot.rss_bytes < target {
                target - snapshot.rss_bytes
            } else {
                (mem_ceiling as f64 * 0.15) as u64
            };
            // Bytes per entry: sort key vec (~48B) + RunRecord (~48B) + overhead
            let bytes_per_entry: u64 = 96 + (MAX_PREFIX as u64 * 8);
            let governor_batch = (available / bytes_per_entry) as usize;
            let clamped = governor_batch.max(10_000).min(10_000_000);
            if clamped != DEFAULT_BATCH_SIZE {
                ctx.ui.log(&format!(
                    "  governor-derived batch size: {} (available: {} MiB, RSS: {} MiB, ceiling: {} MiB)",
                    clamped,
                    available / (1 << 20),
                    snapshot.rss_bytes / (1 << 20),
                    mem_ceiling / (1 << 20),
                ));
            }
            clamped
        } else {
            DEFAULT_BATCH_SIZE
        };

        let source_bytes = std::fs::metadata(&source_path).map(|m| m.len()).unwrap_or(0);

        ctx.ui.log(&format!(
            "Dedup: {} vectors ({}, dim={}), elide={}, batch_size={}",
            format_count(count), reader.format_name(), dim, elide, format_count(batch_size),
        ));
        ctx.ui.log(&format!(
            "  source: {} ({})",
            source_path.file_name().unwrap_or_default().to_string_lossy(),
            format_bytes(source_bytes),
        ));

        if count == 0 {
            ensure_parent(&output_path);
            ensure_parent(&report_path);
            if let Err(e) = write_ivec(&output_path, &[]) {
                return error_result(format!("failed to write output: {}", e), start);
            }
            if let Err(e) = write_report(&report_path, 0, 0, 0, 0) {
                return error_result(format!("failed to write report: {}", e), start);
            }
            return CommandResult {
                status: Status::Ok,
                message: "empty input".to_string(),
                produced: vec![output_path, report_path],
                elapsed: start.elapsed(),
            };
        }

        // ── Phase 0: Variance sampling to determine prefix width ──────
        let phase0_start = Instant::now();
        ctx.ui.log("Phase 0: variance sampling");
        let prefix_width = sample_prefix_width(&reader, count, dim, ctx);
        let phase0_elapsed = phase0_start.elapsed();
        ctx.ui.log(&format!(
            "  prefix width: {} component(s), sampled in {:.1}s",
            prefix_width, phase0_elapsed.as_secs_f64(),
        ));

        // ── Phase 1: Create sorted runs ───────────────────────────────
        let phase1_start = Instant::now();
        ctx.ui.log("Phase 1: creating sorted runs");
        let run_dir = ctx.cache.join("dedup_runs");
        if let Err(e) = std::fs::create_dir_all(&run_dir) {
            return error_result(format!("failed to create run dir: {}", e), start);
        }

        let run_files = match create_sorted_runs(
            &reader, count, dim, prefix_width, batch_size, &run_dir, ctx,
        ) {
            Ok(files) => files,
            Err(e) => return error_result(e, start),
        };

        let phase1_elapsed = phase1_start.elapsed();
        let phase1_secs = phase1_elapsed.as_secs_f64();
        let phase1_rate = if phase1_secs > 0.0 { count as f64 / phase1_secs } else { 0.0 };
        let phase1_mbps = if phase1_secs > 0.0 {
            source_bytes as f64 / (1024.0 * 1024.0) / phase1_secs
        } else { 0.0 };
        ctx.ui.log(&format!(
            "  {} sorted run(s) in {:.1}s ({:.0} vec/s, {:.1} MB/s read)",
            run_files.len(), phase1_secs, phase1_rate, phase1_mbps,
        ));

        // ── Phase 2: K-way streaming merge with dedup ─────────────────
        let phase2_start = Instant::now();
        ctx.ui.log("Phase 2: k-way merge");
        ensure_parent(&output_path);
        ensure_parent(&report_path);

        ensure_parent(&duplicates_path);
        let (unique_count, dup_count) = match merge_runs(
            &run_files, prefix_width, &reader, dim, elide,
            &output_path, &duplicates_path, count, ctx,
        ) {
            Ok(counts) => counts,
            Err(e) => return error_result(e, start),
        };

        let phase2_elapsed = phase2_start.elapsed();
        let phase2_secs = phase2_elapsed.as_secs_f64();
        let merge_rate = if phase2_secs > 0.0 { count as f64 / phase2_secs } else { 0.0 };
        ctx.ui.log(&format!(
            "  merge: {:.1}s ({:.0} vec/s), {} full-vector comparisons avoided by prefix",
            phase2_secs, merge_rate,
            format_count(count - dup_count), // non-dup entries never needed full read
        ));

        // ── Phase 3: Write report and clean up ────────────────────────
        if let Err(e) = write_report(&report_path, count, unique_count, dup_count, prefix_width) {
            return error_result(format!("failed to write report: {}", e), start);
        }

        // Run files are kept in .cache/dedup_runs/ for resume on re-run.
        // They are cleaned up by `veks run --clean` or manual cache deletion.

        let total_elapsed = start.elapsed();
        let total_secs = total_elapsed.as_secs_f64();
        let overall_rate = if total_secs > 0.0 { count as f64 / total_secs } else { 0.0 };
        let overall_mbps = if total_secs > 0.0 {
            source_bytes as f64 / (1024.0 * 1024.0) / total_secs
        } else { 0.0 };
        let dup_pct = if count > 0 { 100.0 * dup_count as f64 / count as f64 } else { 0.0 };

        let msg = format!(
            "{} vectors -> {} unique, {} dup ({:.2}%){}, {:.1}s ({:.0} vec/s, {:.1} MB/s), prefix_width={}",
            format_count(count),
            format_count(unique_count),
            format_count(dup_count),
            dup_pct,
            if elide { " elided" } else { "" },
            total_secs,
            overall_rate,
            overall_mbps,
            prefix_width,
        );
        ctx.ui.log(&msg);

        CommandResult {
            status: Status::Ok,
            message: msg,
            produced: vec![output_path, duplicates_path, report_path],
            elapsed: total_elapsed,
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Input fvec file".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output ivec file (sorted ordinal index)".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "duplicates".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: Some("<output>_duplicates.ivec".to_string()),
                description: "Output ivec file containing ordinals of duplicate vectors".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "report".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: Some("<output>.json".to_string()),
                description: "JSON report with duplicate statistics".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "elide".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("true".to_string()),
                description: "Remove duplicates from the output index".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "batch-size".to_string(),
                type_name: "integer".to_string(),
                required: false,
                default: Some(DEFAULT_BATCH_SIZE.to_string()),
                description: "Vectors per sorted run".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        let mut manifest = crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output", "duplicates", "report"],
        );
        // Sorted run files in .cache/dedup_runs/ are intermediate artifacts
        manifest.intermediates.push("${cache}/dedup_runs/".to_string());
        manifest
    }
}

// ---------------------------------------------------------------------------
// Phase 0: Variance-based prefix width selection
// ---------------------------------------------------------------------------

/// Sample vectors and choose how many prefix components (1–10) to retain
/// in sorted run records.
///
/// Strategy: for each leading dimension, compute the number of distinct
/// values in the sample. As soon as a dimension has enough distinct values
/// to separate the sample (≥ 90% unique among sampled values), that many
/// prefix components suffice. If early dimensions are low-cardinality
/// (e.g., quantized or categorical), more dimensions are needed.
fn sample_prefix_width(
    reader: &VecReader,
    count: usize,
    dim: usize,
    ctx: &mut StreamContext,
) -> usize {
    let sample_count = std::cmp::min(VARIANCE_SAMPLE_SIZE, count);
    let step = if count <= sample_count { 1 } else { count / sample_count };

    let max_dims = MAX_PREFIX.min(dim);

    // Collect sampled vectors (first max_dims components each)
    let mut samples: Vec<Vec<f32>> = Vec::with_capacity(sample_count);

    let pb = ctx.ui.bar_with_unit(sample_count as u64, "sampling dimensions", "vectors");
    let mut sampled = 0usize;
    let mut i = 0usize;
    while i < count && sampled < sample_count {
        if let Ok(vec) = reader.get_f32(i) {
            let prefix: Vec<f32> = vec.iter().take(max_dims).copied().collect();
            samples.push(prefix);
            sampled += 1;
        }
        i += step;
    }
    pb.finish();

    if samples.len() < 2 {
        return max_dims;
    }

    // For each prefix length 1..max_dims, count how many distinct
    // sort keys exist in the sample. Stop when distinctness ≥ 90%.
    let threshold = (samples.len() as f64 * 0.90) as usize;
    let mut chosen_width = max_dims;

    for width in MIN_PREFIX..=max_dims {
        let mut keys: Vec<Vec<u32>> = samples.iter()
            .map(|s| s.iter().take(width).map(|&v| float_to_sortable(v)).collect())
            .collect();
        keys.sort();
        keys.dedup();
        let distinct = keys.len();

        ctx.ui.log(&format!(
            "  dim prefix {}: {} distinct / {} sampled ({:.1}%)",
            width, distinct, samples.len(),
            100.0 * distinct as f64 / samples.len() as f64,
        ));

        if distinct >= threshold {
            chosen_width = width;
            break;
        }
    }

    chosen_width
}

// ---------------------------------------------------------------------------
// Phase 1: Create sorted runs
// ---------------------------------------------------------------------------

/// Encode an f32 as a u32 that sorts in the same total order as the float.
///
/// IEEE 754 positive floats already sort by bit pattern. For negatives,
/// the sign bit is set and the magnitude is inverted. This function maps
/// all f32 values to u32 such that `a < b` iff `float_to_sortable(a) < float_to_sortable(b)`.
#[inline]
fn float_to_sortable(f: f32) -> u32 {
    let bits = f.to_bits();
    if bits & 0x8000_0000 != 0 {
        // Negative: flip all bits (sign + magnitude inversion)
        !bits
    } else {
        // Positive (and +0): flip sign bit so positives sort after negatives
        bits ^ 0x8000_0000
    }
}

/// Build a lexicographic sort key from the full vector components.
///
/// The key contains all dimension values encoded as sortable u32s.
/// This enables binary search for any specific vector and ensures
/// exact duplicates are always adjacent in sorted order.
fn build_sort_key(vec: &[f32], _prefix_width: usize) -> Vec<u32> {
    vec.iter().map(|&v| float_to_sortable(v)).collect()
}

/// Metadata for a set of sorted runs, used for resume validation.
#[derive(serde::Serialize, serde::Deserialize)]
struct RunMeta {
    count: usize,
    prefix_width: usize,
    batch_size: usize,
    num_runs: u32,
}

/// Create sorted run files from batches of the input, with resume support.
///
/// If valid run files from a previous interrupted attempt exist in `run_dir`
/// with matching parameters, they are reused. Only missing runs are created.
fn create_sorted_runs(
    reader: &VecReader,
    count: usize,
    _dim: usize,
    prefix_width: usize,
    batch_size: usize,
    run_dir: &Path,
    ctx: &mut StreamContext,
) -> Result<Vec<PathBuf>, String> {
    let threads = ctx.governor.current_or("threads", ctx.threads as u64)
        .max(1) as usize;
    ctx.ui.log(&format!("  {} threads for parallel read + sort", threads));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let expected_runs = ((count + batch_size - 1) / batch_size) as u32;

    // Check for existing runs from a previous attempt
    let meta_path = run_dir.join("meta.json");
    let can_resume = if meta_path.exists() {
        match std::fs::read_to_string(&meta_path) {
            Ok(content) => match serde_json::from_str::<RunMeta>(&content) {
                Ok(meta) => {
                    meta.count == count
                        && meta.prefix_width == prefix_width
                        && meta.batch_size == batch_size
                        && meta.num_runs == expected_runs
                }
                Err(_) => false,
            },
            Err(_) => false,
        }
    } else {
        false
    };

    // Record size: 4 bytes ordinal + prefix_width * 4 bytes
    let record_bytes = 4 + prefix_width * 4;

    let mut run_files: Vec<PathBuf> = Vec::new();
    let mut skipped = 0u32;
    let pb = ctx.ui.bar_with_unit(count as u64, "building runs", "vectors");

    let mut batch_start = 0;
    let mut run_idx = 0u32;

    while batch_start < count {
        let batch_end = std::cmp::min(batch_start + batch_size, count);
        let batch_len = batch_end - batch_start;

        let run_path = run_dir.join(format!("run_{:04}.bin", run_idx));

        // Resume: skip runs whose .gz file already exists
        if can_resume && crate::pipeline::gz_cache::gz_exists(&run_path) {
            // Validate by checking the gzip ISIZE footer matches expected uncompressed size
            let expected_size = (batch_len * record_bytes) as u32;
            if let Some(isize_hint) = crate::pipeline::gz_cache::gz_uncompressed_size_hint(&run_path) {
                if isize_hint == expected_size {
                    run_files.push(run_path);
                    run_idx += 1;
                    skipped += 1;
                    batch_start = batch_end;
                    pb.set_position(batch_end as u64);
                    continue;
                }
            }
        }

        // Build (sort_key, RunRecord) for this batch using parallel reads.
        // Each vector read is independent (mmap random access), and key
        // computation is pure arithmetic — both parallelize perfectly.
        let mut entries: Vec<(Vec<u32>, RunRecord)>;
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};

            let progress = AtomicU64::new(0);

            let build_fn = || {
                (batch_start..batch_end)
                    .into_par_iter()
                    .map(|i| {
                        let vec = reader.get_f32(i)?;
                        let key = build_sort_key(&vec, prefix_width);
                        let prefix: Vec<f32> = vec.iter().take(prefix_width).copied().collect();
                        let done = progress.fetch_add(1, AtomicOrd::Relaxed) + 1;
                        if done % 500_000 == 0 {
                            pb.set_position(batch_start as u64 + done);
                        }
                        Ok((key, RunRecord { ordinal: i as u32, prefix }))
                    })
                    .collect::<Result<Vec<_>, String>>()
            };

            let entries_result = if let Some(ref p) = pool {
                p.install(build_fn)
            } else {
                build_fn()
            };

            entries = match entries_result {
                Ok(e) => e,
                Err(e) => return Err(e),
            };
        }
        pb.set_position(batch_end as u64);

        // Parallel sort by composite key (governor-limited thread pool)
        {
            use rayon::prelude::*;
            let sort_sp = ctx.ui.spinner(&format!("sorting run {} ({} entries)", run_idx, batch_len));
            let sort_fn = |entries: &mut Vec<(Vec<u32>, RunRecord)>| {
                entries.par_sort_by(|a, b| {
                    for (ka, kb) in a.0.iter().zip(b.0.iter()) {
                        let ord = ka.cmp(kb);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                    }
                    a.1.ordinal.cmp(&b.1.ordinal)
                });
            };
            if let Some(ref p) = pool {
                p.install(|| sort_fn(&mut entries));
            } else {
                sort_fn(&mut entries);
            }
            sort_sp.finish();
        }

        // Write sorted run to memory buffer, then compress and save as .gz
        let mut buf = Vec::with_capacity(batch_len * record_bytes);
        for (_, record) in &entries {
            record.write_to(&mut buf)
                .map_err(|e| format!("failed to write run record: {}", e))?;
        }
        crate::pipeline::gz_cache::save_gz(&run_path, &buf)?;

        run_files.push(run_path);
        run_idx += 1;

        pb.set_position(batch_end as u64);

        if ctx.governor.checkpoint() {
            ctx.ui.log("  governor: throttle active");
        }

        batch_start = batch_end;
    }
    pb.finish();

    if skipped > 0 {
        ctx.ui.log(&format!(
            "  resumed: {} of {} runs from cache, {} created fresh",
            skipped, run_files.len(), run_files.len() as u32 - skipped,
        ));
    }

    // Write metadata for future resume
    let meta = RunMeta {
        count,
        prefix_width,
        batch_size,
        num_runs: run_files.len() as u32,
    };
    if let Ok(json) = serde_json::to_string_pretty(&meta) {
        let _ = std::fs::write(&meta_path, json);
    }

    Ok(run_files)
}

// ---------------------------------------------------------------------------
// Phase 2: K-way streaming merge
// ---------------------------------------------------------------------------

/// Merge all sorted run files, streaming through a min-heap.
///
/// Writes the sorted (optionally deduplicated) ordinals to `output_path`
/// and all duplicate ordinals to `duplicates_path`.
///
/// Returns `(unique_count, duplicate_count)`.
fn merge_runs(
    run_files: &[PathBuf],
    prefix_width: usize,
    reader: &VecReader,
    _dim: usize,
    elide: bool,
    output_path: &Path,
    duplicates_path: &Path,
    total: usize,
    ctx: &mut StreamContext,
) -> Result<(usize, usize), String> {
    // Decompress all run files into memory, then wrap as cursors for streaming read
    let mut run_buffers: Vec<std::io::Cursor<Vec<u8>>> = Vec::with_capacity(run_files.len());
    let decompress_pb = ctx.ui.bar_with_unit(run_files.len() as u64, "loading runs", "files");
    for (i, path) in run_files.iter().enumerate() {
        let data = crate::pipeline::gz_cache::load_gz(path)?;
        run_buffers.push(std::io::Cursor::new(data));
        decompress_pb.set_position((i + 1) as u64);
    }
    decompress_pb.finish();

    // Initialize the min-heap with the first record from each run
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(run_buffers.len());
    for (idx, rdr) in run_buffers.iter_mut().enumerate() {
        if let Some(record) = RunRecord::read_from(rdr, prefix_width)
            .map_err(|e| format!("failed to read run {}: {}", idx, e))?
        {
            heap.push(HeapEntry { record, run_idx: idx });
        }
    }

    // Open outputs: sorted ordinals + duplicate ordinals
    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("failed to create output: {}", e))?;
    let mut out = BufWriter::with_capacity(IO_BUF_SIZE, file);

    ensure_parent(duplicates_path);
    let dup_file = std::fs::File::create(duplicates_path)
        .map_err(|e| format!("failed to create duplicates file: {}", e))?;
    let mut dup_out = BufWriter::with_capacity(IO_BUF_SIZE, dup_file);

    let pb = ctx.ui.bar_with_unit(total as u64, "merging", "vectors");

    let mut unique_count = 0usize;
    let mut dup_count = 0usize;
    let mut prev: Option<RunRecord> = None;
    let mut emitted = 0u64;

    while let Some(entry) = heap.pop() {
        let record = entry.record;
        let run_idx = entry.run_idx;

        // Refill from the same run
        if let Some(next) = RunRecord::read_from(&mut run_buffers[run_idx], prefix_width)
            .map_err(|e| format!("failed to read run {}: {}", run_idx, e))?
        {
            heap.push(HeapEntry { record: next, run_idx });
        }

        // Check for duplicate against previous record
        let is_dup = if let Some(ref prev_rec) = prev {
            if prev_rec.prefix_eq(&record) {
                // Prefixes match — need full vector comparison
                reader.vectors_equal(prev_rec.ordinal as usize, record.ordinal as usize)
            } else {
                false
            }
        } else {
            false
        };

        if is_dup {
            dup_count += 1;
            // Always write duplicate ordinals to the duplicates file
            write_ivec_record(&mut dup_out, record.ordinal)
                .map_err(|e| format!("dup write error: {}", e))?;
            if !elide {
                // Also emit to the main sorted output when not eliding
                write_ivec_record(&mut out, record.ordinal)
                    .map_err(|e| format!("write error: {}", e))?;
            }
        } else {
            unique_count += 1;
            write_ivec_record(&mut out, record.ordinal)
                .map_err(|e| format!("write error: {}", e))?;
        }

        prev = Some(record);
        emitted += 1;

        if emitted % 500_000 == 0 {
            pb.set_position(emitted);
        }
    }

    pb.finish();
    out.flush().map_err(|e| format!("flush error: {}", e))?;
    dup_out.flush().map_err(|e| format!("dup flush error: {}", e))?;

    Ok((unique_count, dup_count))
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

/// Write a single ivec record (dim=1, one i32 ordinal).
fn write_ivec_record<W: Write>(w: &mut W, ordinal: u32) -> std::io::Result<()> {
    w.write_i32::<LittleEndian>(1)?;
    w.write_i32::<LittleEndian>(ordinal as i32)?;
    Ok(())
}

/// Write a complete ivec file from a slice of ordinals.
fn write_ivec(path: &Path, ordinals: &[u32]) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    let mut writer = BufWriter::with_capacity(IO_BUF_SIZE, file);
    for &ord in ordinals {
        write_ivec_record(&mut writer, ord).map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write the JSON deduplication report.
fn write_report(
    path: &Path,
    total: usize,
    unique: usize,
    duplicates: usize,
    prefix_width: usize,
) -> Result<(), String> {
    let content = format!(
        concat!(
            "{{\n",
            "  \"total_vectors\": {},\n",
            "  \"unique_vectors\": {},\n",
            "  \"duplicate_vectors\": {},\n",
            "  \"duplicate_ratio\": {:.6},\n",
            "  \"has_duplicates\": {},\n",
            "  \"prefix_width\": {}\n",
            "}}\n",
        ),
        total,
        unique,
        duplicates,
        if total > 0 { duplicates as f64 / total as f64 } else { 0.0 },
        duplicates > 0,
        prefix_width,
    );
    std::fs::write(path, content)
        .map_err(|e| format!("failed to write {}: {}", path.display(), e))
}

fn ensure_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
}

/// Format a count with thousands separators.
fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

/// Format byte count as human-readable size.
fn format_bytes(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * 1024 * 1024 * 1024;
    if bytes >= TIB {
        format!("{:.1} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
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
