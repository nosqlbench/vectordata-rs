// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: deduplicate vectors via external merge-sort.
//!
//! Uses a multi-phase external sort with adaptive prefix keys:
//!
//! 1. **Variance sampling** — sparse sample to determine how many leading
//!    vector components (1–10) are needed to distinguish vectors. High
//!    variance in the norm means fewer prefix components suffice.
//!
//! 2. **Sorted run creation** — process input in batches, compute a
//!    composite sort key (norm + prefix components), sort in-memory, and
//!    write intermediate sorted runs to cache. Each run record is
//!    `(ordinal:u32, prefix[0..N]:f32)`.
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
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc,
    Status, StreamContext, render_options_table,
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
    /// Compare by prefix components (total order via bit patterns).
    fn cmp_prefix(&self, other: &RunRecord) -> Ordering {
        for (a, b) in self.prefix.iter().zip(other.prefix.iter()) {
            let ord = a.to_bits().cmp(&b.to_bits());
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
                .all(|(a, b)| a.to_bits() == b.to_bits())
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

1. **Variance sampling** — reads a sparse sample of vectors to estimate
   how many leading components (1–10) are needed to distinguish most
   vectors. High variance in the norm means fewer prefix components
   suffice; low variance means more are needed.

2. **Sorted run creation** — processes the input in configurable batches
   (default 1M vectors). For each batch, computes a composite sort key
   from the L2 norm and the adaptive prefix, sorts in-memory, and writes
   a sorted run file to the cache directory. Each run record stores only
   `(ordinal, prefix[0..N])` — not the full vector.

3. **K-way streaming merge** — opens all sorted run files and merges them
   using a min-heap over prefix keys. Adjacent entries with identical
   prefixes are compared by reading the full vectors from the source (via
   mmap random access). This short-circuits the vast majority of
   comparisons to prefix-only, avoiding random I/O for non-duplicates.

4. **Output** — the merged ordinal sequence (with duplicates optionally
   elided) is written as an ivec file. A JSON report records total,
   unique, and duplicate counts plus the prefix width chosen.

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
        let elide = options.get("elide")
            .map(|s| s != "false")
            .unwrap_or(true);
        let explicit_batch_size: Option<usize> = options.get("batch-size")
            .and_then(|s| s.parse().ok());

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let report_path = resolve_path(&report_str, &ctx.workspace);

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

        let (unique_count, dup_count) = match merge_runs(
            &run_files, prefix_width, &reader, dim, elide,
            &output_path, count, ctx,
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

        // Clean up intermediate run files
        for f in &run_files {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&run_dir);

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
            produced: vec![output_path, report_path],
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
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output ivec file (sorted ordinal index)".to_string(),
            },
            OptionDesc {
                name: "report".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: Some("<output>.json".to_string()),
                description: "JSON report with duplicate statistics".to_string(),
            },
            OptionDesc {
                name: "elide".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("true".to_string()),
                description: "Remove duplicates from the output index".to_string(),
            },
            OptionDesc {
                name: "batch-size".to_string(),
                type_name: "integer".to_string(),
                required: false,
                default: Some(DEFAULT_BATCH_SIZE.to_string()),
                description: "Vectors per sorted run".to_string(),
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output", "report"],
        )
    }
}

// ---------------------------------------------------------------------------
// Phase 0: Variance-based prefix width selection
// ---------------------------------------------------------------------------

/// Sample vectors and choose how many prefix components (1–10) to retain
/// in sorted run records.
///
/// Strategy: compute the coefficient of variation (CV = stddev/mean) of
/// the L2 norm across the sample. High CV means the norm alone separates
/// vectors well → fewer prefix components. Low CV means vectors cluster
/// tightly in norm space → more prefix components for disambiguation.
fn sample_prefix_width(
    reader: &VecReader,
    count: usize,
    dim: usize,
    ctx: &mut StreamContext,
) -> usize {
    let sample_count = std::cmp::min(VARIANCE_SAMPLE_SIZE, count);
    let step = if count <= sample_count { 1 } else { count / sample_count };

    let mut norms: Vec<f64> = Vec::with_capacity(sample_count);

    let pb = ctx.ui.bar_with_unit(sample_count as u64, "sampling variance", "vectors");
    let mut sampled = 0usize;
    let mut i = 0usize;
    while i < count && sampled < sample_count {
        if let Ok(vec) = reader.get_f32(i) {
            let mut sum = 0.0f64;
            for &v in &vec {
                let vf = v as f64;
                sum += vf * vf;
            }
            norms.push(sum.sqrt());
            sampled += 1;
        }
        i += step;
    }
    pb.finish();

    if norms.len() < 2 {
        return MAX_PREFIX.min(dim);
    }

    let n = norms.len() as f64;
    let mean = norms.iter().sum::<f64>() / n;
    let variance = norms.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    let stddev = variance.sqrt();
    let cv = if mean.abs() > 1e-12 { stddev / mean.abs() } else { 0.0 };

    ctx.ui.log(&format!(
        "  norm stats: mean={:.4}, stddev={:.4}, cv={:.4} (n={})",
        mean, stddev, cv, norms.len(),
    ));

    // Map CV to prefix width:
    //   CV >= 0.5  → 1 component  (norms spread widely, very discriminating)
    //   CV ~= 0.25 → 3 components
    //   CV ~= 0.1  → 5 components
    //   CV ~= 0.01 → 8 components
    //   CV < 0.005 → 10 components (nearly identical norms, need many dims)
    let width = if cv >= 0.50 {
        1
    } else if cv >= 0.25 {
        2
    } else if cv >= 0.10 {
        3
    } else if cv >= 0.05 {
        5
    } else if cv >= 0.01 {
        7
    } else if cv >= 0.005 {
        8
    } else {
        10
    };

    width.max(MIN_PREFIX).min(MAX_PREFIX).min(dim)
}

// ---------------------------------------------------------------------------
// Phase 1: Create sorted runs
// ---------------------------------------------------------------------------

/// Build a sort key for one vector: `[norm_bits, prefix_0_bits, ..., prefix_N_bits]`.
fn build_sort_key(vec: &[f32], prefix_width: usize) -> Vec<u32> {
    let mut sum = 0.0f64;
    for &v in vec {
        let vf = v as f64;
        sum += vf * vf;
    }
    let norm = sum.sqrt() as f32;

    let mut key = Vec::with_capacity(1 + prefix_width);
    key.push(norm.to_bits());
    for i in 0..prefix_width {
        if i < vec.len() {
            key.push(vec[i].to_bits());
        } else {
            key.push(0);
        }
    }
    key
}

/// Create sorted run files from batches of the input.
fn create_sorted_runs(
    reader: &VecReader,
    count: usize,
    _dim: usize,
    prefix_width: usize,
    batch_size: usize,
    run_dir: &Path,
    ctx: &mut StreamContext,
) -> Result<Vec<PathBuf>, String> {
    let mut run_files: Vec<PathBuf> = Vec::new();
    let pb = ctx.ui.bar_with_unit(count as u64, "building runs", "vectors");

    let mut batch_start = 0;
    let mut run_idx = 0u32;

    while batch_start < count {
        let batch_end = std::cmp::min(batch_start + batch_size, count);
        let batch_len = batch_end - batch_start;

        // Build (sort_key, RunRecord) for this batch
        let mut entries: Vec<(Vec<u32>, RunRecord)> = Vec::with_capacity(batch_len);

        for i in batch_start..batch_end {
            let vec = reader.get_f32(i)?;

            let key = build_sort_key(&vec, prefix_width);

            let mut prefix = Vec::with_capacity(prefix_width);
            for j in 0..prefix_width {
                if j < vec.len() {
                    prefix.push(vec[j]);
                } else {
                    prefix.push(0.0);
                }
            }

            entries.push((key, RunRecord { ordinal: i as u32, prefix }));
        }

        // Sort batch by composite key
        entries.sort_by(|a, b| {
            for (ka, kb) in a.0.iter().zip(b.0.iter()) {
                let ord = ka.cmp(kb);
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            a.1.ordinal.cmp(&b.1.ordinal)
        });

        // Write sorted run to disk
        let run_path = run_dir.join(format!("run_{:04}.bin", run_idx));
        let file = std::fs::File::create(&run_path)
            .map_err(|e| format!("failed to create run file: {}", e))?;
        let mut writer = BufWriter::with_capacity(IO_BUF_SIZE, file);

        for (_, record) in &entries {
            record.write_to(&mut writer)
                .map_err(|e| format!("failed to write run record: {}", e))?;
        }
        writer.flush().map_err(|e| format!("failed to flush run: {}", e))?;

        run_files.push(run_path);
        run_idx += 1;

        pb.set_position(batch_end as u64);

        if ctx.governor.checkpoint() {
            ctx.ui.log("  governor: throttle active");
        }

        batch_start = batch_end;
    }
    pb.finish();

    Ok(run_files)
}

// ---------------------------------------------------------------------------
// Phase 2: K-way streaming merge
// ---------------------------------------------------------------------------

/// Merge all sorted run files, streaming through a min-heap.
///
/// Returns `(unique_count, duplicate_count)`.
fn merge_runs(
    run_files: &[PathBuf],
    prefix_width: usize,
    reader: &VecReader,
    _dim: usize,
    elide: bool,
    output_path: &Path,
    total: usize,
    ctx: &mut StreamContext,
) -> Result<(usize, usize), String> {
    // Open all run files
    let mut run_readers: Vec<BufReader<std::fs::File>> = Vec::with_capacity(run_files.len());
    for path in run_files {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open run {}: {}", path.display(), e))?;
        run_readers.push(BufReader::with_capacity(IO_BUF_SIZE, file));
    }

    // Initialize the min-heap with the first record from each run
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(run_readers.len());
    for (idx, rdr) in run_readers.iter_mut().enumerate() {
        if let Some(record) = RunRecord::read_from(rdr, prefix_width)
            .map_err(|e| format!("failed to read run {}: {}", idx, e))?
        {
            heap.push(HeapEntry { record, run_idx: idx });
        }
    }

    // Open output
    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("failed to create output: {}", e))?;
    let mut out = BufWriter::with_capacity(IO_BUF_SIZE, file);

    let pb = ctx.ui.bar_with_unit(total as u64, "merging", "vectors");

    let mut unique_count = 0usize;
    let mut dup_count = 0usize;
    let mut prev: Option<RunRecord> = None;
    let mut emitted = 0u64;

    while let Some(entry) = heap.pop() {
        let record = entry.record;
        let run_idx = entry.run_idx;

        // Refill from the same run
        if let Some(next) = RunRecord::read_from(&mut run_readers[run_idx], prefix_width)
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
            if !elide {
                // Still emit the ordinal even though it's a duplicate
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
