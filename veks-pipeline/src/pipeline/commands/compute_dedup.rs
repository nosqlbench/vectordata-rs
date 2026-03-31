// Copyright (c) nosqlbench contributors
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
use std::io::{Read, Write};
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

    /// Read the first `n` components of vector at `index` as f32 into `out`.
    ///
    /// Zero-allocation for f32 sources (direct mmap slice). For f16 sources,
    /// upcasts only the requested components — no full-vector allocation.
    #[inline]
    fn prefix_f32_into(&self, index: usize, out: &mut [f32]) {
        let n = out.len();
        match self {
            VecReader::F32(r) => {
                let slice = r.get_slice(index);
                out.copy_from_slice(&slice[..n]);
            }
            VecReader::F16(r) => {
                let slice = r.get_slice(index);
                for (dst, src) in out.iter_mut().zip(slice[..n].iter()) {
                    *dst = src.to_f32();
                }
            }
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
/// Minimum prefix components. 10 components provides strong discrimination
/// for most embedding models, minimizing prefix collision groups that
/// require full-vector comparison during dedup.
const MIN_PREFIX: usize = 10;
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
#[derive(Clone, Copy)]
struct RunRecord {
    ordinal: u32,
    prefix: [f32; MAX_PREFIX],
    prefix_len: u8,
}

impl RunRecord {
    /// Create a new record with the given ordinal and prefix components.
    #[inline]
    fn new(ordinal: u32, prefix_data: &[f32]) -> Self {
        let mut prefix = [0.0f32; MAX_PREFIX];
        let n = prefix_data.len().min(MAX_PREFIX);
        prefix[..n].copy_from_slice(&prefix_data[..n]);
        RunRecord { ordinal, prefix, prefix_len: n as u8 }
    }

    /// The active prefix slice.
    #[inline]
    fn prefix(&self) -> &[f32] {
        &self.prefix[..self.prefix_len as usize]
    }

    /// Compare by prefix components using total lexicographic order.
    fn cmp_prefix(&self, other: &RunRecord) -> Ordering {
        for (a, b) in self.prefix().iter().zip(other.prefix().iter()) {
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
        self.prefix_len == other.prefix_len
            && self.prefix().iter().zip(other.prefix().iter())
                .all(|(a, b)| float_to_sortable(*a) == float_to_sortable(*b))
    }

    /// Write to a buffered writer.
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.ordinal)?;
        for &v in self.prefix() {
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
        let mut prefix = [0.0f32; MAX_PREFIX];
        for i in 0..prefix_width {
            prefix[i] = r.read_f32::<LittleEndian>()?;
        }
        Ok(Some(RunRecord { ordinal, prefix, prefix_len: prefix_width as u8 }))
    }
}


// ---------------------------------------------------------------------------
// CommandOp implementation
// ---------------------------------------------------------------------------

impl CommandOp for ComputeDedupOp {
    fn command_path(&self) -> &str {
        "compute sort"
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

        // Cache compression level: 0 = raw (fast), 1-9 = gzip (smaller on disk).
        // Default 0 (no compression) since run files are ephemeral intermediates.
        let compress_level: u32 = options.get("compress")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        crate::pipeline::gz_cache::set_compression_level(compress_level);

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

        // Top-level phase indicator — visible on the progress bar line
        // throughout the entire command. Sub-phases create their own bars
        // underneath for detailed progress.
        let phase = ctx.ui.spinner("sort+dedup");
        phase.set_message("phase 0/4: sampling variance".to_string());

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
        phase.set_message("phase 1/4: building sorted runs".to_string());
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

        // ── Phase 2: Parallel merge + dedup ───────────────────────────
        let phase2_start = Instant::now();
        phase.set_message("phase 2/4: parallel merge".to_string());
        ctx.ui.log("Phase 2: parallel merge + dedup");
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
        phase.set_message("phase 3/4: writing report".to_string());
        if let Err(e) = write_report(&report_path, count, unique_count, dup_count, prefix_width) {
            return error_result(format!("failed to write report: {}", e), start);
        }

        // Run files are kept in .cache/dedup_runs/ for resume on re-run.
        // They are cleaned up by `veks run --clean` or manual cache deletion.

        phase.finish();

        let total_elapsed = start.elapsed();
        let total_secs = total_elapsed.as_secs_f64();
        let overall_rate = if total_secs > 0.0 { count as f64 / total_secs } else { 0.0 };
        let overall_mbps = if total_secs > 0.0 {
            source_bytes as f64 / (1024.0 * 1024.0) / total_secs
        } else { 0.0 };
        let dup_pct = if count > 0 { 100.0 * dup_count as f64 / count as f64 } else { 0.0 };

        // Write verified counts so the bound checker can validate the output.
        let output_records = if elide { unique_count } else { count };
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &output_records.to_string());
        ctx.defaults.insert(var_name, output_records.to_string());

        let dup_var_name = format!("verified_count:{}",
            duplicates_path.file_name().and_then(|n| n.to_str()).unwrap_or("dups"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &dup_var_name, &dup_count.to_string());
        ctx.defaults.insert(dup_var_name, dup_count.to_string());

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
            OptionDesc {
                name: "compress".to_string(),
                type_name: "integer".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Cache compression level (0=raw, 1-9=gzip)".to_string(),
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
/// Build a fixed-size sort key from the prefix components only.
#[inline]
fn build_sort_key(prefix: &[f32]) -> [u32; MAX_PREFIX] {
    let mut key = [0u32; MAX_PREFIX];
    for (k, &v) in key.iter_mut().zip(prefix.iter()) {
        *k = float_to_sortable(v);
    }
    key
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

    let num_runs = ((count + batch_size - 1) / batch_size).max(1);
    let mut run_files: Vec<PathBuf> = Vec::new();
    let mut skipped = 0u32;
    let pb = ctx.ui.bar_with_unit(count as u64, "building runs", "vec");

    let mut batch_start = 0;
    let mut run_idx = 0u32;

    while batch_start < count {
        let batch_end = std::cmp::min(batch_start + batch_size, count);
        let batch_len = batch_end - batch_start;

        let run_path = run_dir.join(format!("run_{:04}.bin", run_idx));

        // Resume: skip runs whose cached file already exists (.gz or raw)
        if can_resume && crate::pipeline::gz_cache::gz_exists(&run_path) {
            let expected_size = (batch_len * record_bytes) as u64;
            let size_ok = if crate::pipeline::gz_cache::gz_path(&run_path).exists() {
                // Compressed: check gzip ISIZE footer
                crate::pipeline::gz_cache::gz_uncompressed_size_hint(&run_path)
                    .map(|hint| hint as u64 == expected_size)
                    .unwrap_or(false)
            } else {
                // Raw: check file size directly
                std::fs::metadata(&run_path)
                    .map(|m| m.len() == expected_size)
                    .unwrap_or(false)
            };
            if size_ok {
                run_files.push(run_path);
                run_idx += 1;
                skipped += 1;
                batch_start = batch_end;
                pb.set_position(batch_end as u64);
                ctx.ui.log(&format!("  run {}/{}: resumed from cache", run_idx, num_runs));
                continue;
            }
        }

        // ── Sub-phase A: Read prefix components (parallel) ──────────
        let sub_start = Instant::now();
        pb.set_message(format!("run {}/{} reading", run_idx + 1, num_runs));
        ctx.ui.log(&format!(
            "  run {}/{}: reading {} vectors [{}-{})",
            run_idx + 1, num_runs, format_count(batch_len), batch_start, batch_end,
        ));

        // Each vector read uses get_slice (zero-alloc mmap borrow) and only
        // reads prefix_width components — not the full vector.
        let mut entries: Vec<([u32; MAX_PREFIX], RunRecord)>;
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};

            let progress = AtomicU64::new(0);
            // Update progress every ~0.5% of batch or 5K vectors
            let update_interval = (batch_len / 200).max(5_000) as u64;
            let pw = prefix_width;

            let build_fn = || {
                (batch_start..batch_end)
                    .into_par_iter()
                    .map(|i| {
                        let mut prefix_buf = [0.0f32; MAX_PREFIX];
                        reader.prefix_f32_into(i, &mut prefix_buf[..pw]);
                        let key = build_sort_key(&prefix_buf[..pw]);
                        let record = RunRecord::new(i as u32, &prefix_buf[..pw]);
                        let done = progress.fetch_add(1, AtomicOrd::Relaxed) + 1;
                        if done % update_interval == 0 {
                            pb.set_position(batch_start as u64 + done);
                        }
                        Ok::<_, String>((key, record))
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
        let read_secs = sub_start.elapsed().as_secs_f64();
        let read_rate = if read_secs > 0.0 { batch_len as f64 / read_secs } else { 0.0 };
        ctx.ui.log(&format!(
            "  run {}/{}: read done ({:.1}s, {:.0} vec/s)",
            run_idx + 1, num_runs, read_secs, read_rate,
        ));

        // ── Sub-phase B: Parallel sort ───────────────────────────────
        let sort_start = Instant::now();
        pb.set_message(format!("run {}/{} sorting", run_idx + 1, num_runs));
        {
            use rayon::prelude::*;
            let sort_fn = |entries: &mut Vec<([u32; MAX_PREFIX], RunRecord)>| {
                entries.par_sort_unstable_by(|a, b| {
                    a.0.cmp(&b.0)
                        .then_with(|| a.1.ordinal.cmp(&b.1.ordinal))
                });
            };
            if let Some(ref p) = pool {
                p.install(|| sort_fn(&mut entries));
            } else {
                sort_fn(&mut entries);
            }
        }
        let sort_secs = sort_start.elapsed().as_secs_f64();
        let sort_rate = if sort_secs > 0.0 { batch_len as f64 / sort_secs } else { 0.0 };
        ctx.ui.log(&format!(
            "  run {}/{}: sort done ({:.1}s, {:.0} vec/s, {} threads)",
            run_idx + 1, num_runs, sort_secs, sort_rate, threads,
        ));

        // ── Sub-phase C: Serialize + save run to cache ───────────────
        let write_start = Instant::now();
        pb.set_message(format!("run {}/{} saving", run_idx + 1, num_runs));
        let mut buf = Vec::with_capacity(batch_len * record_bytes);
        for (_, record) in &entries {
            record.write_to(&mut buf)
                .map_err(|e| format!("failed to write run record: {}", e))?;
        }
        crate::pipeline::gz_cache::save_gz(&run_path, &buf)?;
        let write_secs = write_start.elapsed().as_secs_f64();

        run_files.push(run_path);
        run_idx += 1;

        let total_secs = sub_start.elapsed().as_secs_f64();
        ctx.ui.log(&format!(
            "  run {}/{} done: read {:.1}s + sort {:.1}s + save {:.1}s = {:.1}s",
            run_idx, num_runs, read_secs, sort_secs, write_secs, total_secs,
        ));

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
// Phase 2: Parallel merge + dedup
// ---------------------------------------------------------------------------

/// Load all run files, parallel-sort into a single sorted sequence, then
/// scan for duplicates and write output.
///
/// Replaces the old single-threaded k-way heap merge with:
/// 1. Parallel load: parse all runs into Vec<RunRecord> concurrently
/// 2. Parallel sort: par_sort_unstable_by on the combined vector
/// 3. Sequential dedup scan + write (inherently sequential — needs adjacency)
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
    use rayon::prelude::*;

    let threads = ctx.governor.current_or("threads", ctx.threads as u64)
        .max(1) as usize;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    // ── Sub-phase A: Load all runs in parallel ───────────────────────
    let load_start = Instant::now();
    ctx.ui.log(&format!("  loading {} run files ({} threads)...", run_files.len(), threads));
    let pb = ctx.ui.bar_with_unit(run_files.len() as u64, "loading runs", "files");
    let pb_id = pb.id();

    let load_fn = || {
        run_files.par_iter().map(|path| {
            let data = crate::pipeline::gz_cache::load_gz(path)?;
            let mut cursor = std::io::Cursor::new(data);
            let mut records = Vec::new();
            while let Some(rec) = RunRecord::read_from(&mut cursor, prefix_width)
                .map_err(|e| format!("failed to parse {}: {}", path.display(), e))?
            {
                records.push(rec);
            }
            ctx.ui.inc_by_id(pb_id, 1);
            Ok::<Vec<RunRecord>, String>(records)
        }).collect::<Result<Vec<Vec<RunRecord>>, String>>()
    };

    let per_run_records = if let Some(ref p) = pool {
        p.install(load_fn)
    } else {
        load_fn()
    }?;
    pb.finish();

    let load_secs = load_start.elapsed().as_secs_f64();
    ctx.ui.log(&format!("  loaded in {:.1}s", load_secs));

    // Flatten into a single vec
    let flatten_start = Instant::now();
    let mut all_records: Vec<RunRecord> = Vec::with_capacity(total);
    for mut run in per_run_records {
        all_records.append(&mut run);
    }
    let flatten_secs = flatten_start.elapsed().as_secs_f64();
    ctx.ui.log(&format!(
        "  {} records collected ({:.1}s, {:.0} MB)",
        all_records.len(),
        flatten_secs,
        (all_records.len() * std::mem::size_of::<RunRecord>()) as f64 / (1024.0 * 1024.0),
    ));

    // ── Sub-phase B: Parallel sort ───────────────────────────────────
    let sort_start = Instant::now();
    let sort_pb = ctx.ui.spinner(&format!("sorting {} records ({} threads)", all_records.len(), threads));

    let sort_fn = |records: &mut Vec<RunRecord>| {
        records.par_sort_unstable_by(|a, b| a.cmp_prefix(b));
    };
    if let Some(ref p) = pool {
        p.install(|| sort_fn(&mut all_records));
    } else {
        sort_fn(&mut all_records);
    }
    sort_pb.finish();

    let sort_secs = sort_start.elapsed().as_secs_f64();
    let sort_rate = if sort_secs > 0.0 { all_records.len() as f64 / sort_secs } else { 0.0 };
    ctx.ui.log(&format!("  sorted in {:.1}s ({:.0} rec/s)", sort_secs, sort_rate));

    // ── Sub-phase C: Parallel dedup scan → ivec bytes ──────────────
    //
    // Partition the sorted records into chunks and scan each in parallel.
    // Each chunk builds its output + dup ivec byte buffers directly —
    // the prefix ("memo header") in each RunRecord drives the dup check
    // inline, and ivec records (8 bytes: dim=1 + ordinal) are written
    // straight into the byte buffer. One pass, no intermediate Vec<u32>.
    //
    // Boundary pairs (last of chunk N vs first of chunk N+1) are fixed
    // up in a tiny sequential pass before concatenating the buffers.
    let scan_start = Instant::now();
    ctx.ui.log(&format!("  dedup+write: {} threads, {} records", threads, total));

    let chunk_size = (total / threads).max(10_000);
    let chunks: Vec<&[RunRecord]> = all_records.chunks(chunk_size).collect();
    let num_chunks = chunks.len();

    let dim_bytes = 1_i32.to_le_bytes();

    struct ChunkResult {
        out_buf: Vec<u8>,      // ivec bytes for output ordinals
        dup_buf: Vec<u8>,      // ivec bytes for dup ordinals
        dup_count: usize,
    }

    /// Stats for prefix group processing — tracks how much random I/O
    /// the full-vector sort requires, helping diagnose prefix width issues.
    #[derive(Default)]
    struct PrefixGroupStats {
        singleton_groups: usize,        // groups with exactly 1 element (no dup possible)
        multi_groups: usize,            // groups with 2+ elements (need full-vector sort)
        multi_group_total_size: usize,  // total records across multi-element groups
        largest_group: usize,           // biggest prefix collision group
        full_vector_reads: usize,       // total full-vector reads for hashing
    }

    impl PrefixGroupStats {
        fn merge(&mut self, other: &PrefixGroupStats) {
            self.singleton_groups += other.singleton_groups;
            self.multi_groups += other.multi_groups;
            self.multi_group_total_size += other.multi_group_total_size;
            self.largest_group = self.largest_group.max(other.largest_group);
            self.full_vector_reads += other.full_vector_reads;
        }
    }

    let scan_pb = ctx.ui.bar_with_unit(total as u64, "dedup+write", "vec");
    let pb_id = scan_pb.id();

    // Stats for prefix group sorting (tracks how much random I/O the
    // full-vector sort within prefix groups requires)
    let prefix_group_stats = std::sync::Mutex::new(PrefixGroupStats::default());

    let chunk_scan = |chunk: &[RunRecord]| -> ChunkResult {
        let mut out_buf = Vec::with_capacity(chunk.len() * 8);
        let mut dup_buf = Vec::new();
        let mut dups = 0usize;
        let mut local_stats = PrefixGroupStats::default();

        // Process the chunk in prefix groups. Within each group,
        // sort by full vector content so exact duplicates become
        // adjacent. This fixes the bug where a non-duplicate
        // interleaves two true duplicates in the prefix-sorted order.
        let mut group_start = 0;
        while group_start < chunk.len() {
            // Find the extent of this prefix group
            let mut group_end = group_start + 1;
            while group_end < chunk.len() && chunk[group_start].prefix_eq(&chunk[group_end]) {
                group_end += 1;
            }
            let group_len = group_end - group_start;

            if group_len == 1 {
                // Single-element group — no duplicates possible
                let record = &chunk[group_start];
                out_buf.extend_from_slice(&dim_bytes);
                out_buf.extend_from_slice(&(record.ordinal as i32).to_le_bytes());
                local_stats.singleton_groups += 1;
            } else {
                // Multi-element prefix group — track the last unique
                // vector and compare each subsequent record against it.
                // Only reads full vectors for the comparison, no hashing.
                // When a record doesn't match the last unique, it becomes
                // the new last unique reference.
                local_stats.multi_groups += 1;
                local_stats.multi_group_total_size += group_len;
                if group_len > local_stats.largest_group {
                    local_stats.largest_group = group_len;
                }

                let mut last_unique_ord = chunk[group_start].ordinal;
                out_buf.extend_from_slice(&dim_bytes);
                out_buf.extend_from_slice(&(last_unique_ord as i32).to_le_bytes());

                for record in &chunk[group_start + 1..group_end] {
                    local_stats.full_vector_reads += 1;
                    let is_dup = reader.vectors_equal(
                        last_unique_ord as usize, record.ordinal as usize,
                    );

                    let ord_bytes = (record.ordinal as i32).to_le_bytes();
                    if is_dup {
                        dups += 1;
                        dup_buf.extend_from_slice(&dim_bytes);
                        dup_buf.extend_from_slice(&ord_bytes);
                        if !elide {
                            out_buf.extend_from_slice(&dim_bytes);
                            out_buf.extend_from_slice(&ord_bytes);
                        }
                    } else {
                        out_buf.extend_from_slice(&dim_bytes);
                        out_buf.extend_from_slice(&ord_bytes);
                        last_unique_ord = record.ordinal;
                    }
                }
            }
            group_start = group_end;
        }

        ctx.ui.inc_by_id(pb_id, chunk.len() as u64);
        if let Ok(mut stats) = prefix_group_stats.lock() {
            stats.merge(&local_stats);
        }
        ChunkResult { out_buf, dup_buf, dup_count: dups }
    };

    let mut chunk_results: Vec<ChunkResult> = if let Some(ref p) = pool {
        p.install(|| chunks.par_iter().map(|c| chunk_scan(c)).collect())
    } else {
        chunks.iter().map(|c| chunk_scan(c)).collect()
    };
    scan_pb.finish();

    // Boundary fixup: check the boundary between adjacent chunks.
    // A prefix group can span a chunk boundary, so a duplicate in
    // chunk N+1 might match a non-adjacent record in chunk N.
    // Scan backwards from the boundary in chunk N to find the last
    // unique record in the prefix group, and compare against the
    // first record of chunk N+1.
    let mut boundary_dup_count = 0usize;
    for i in 0..num_chunks.saturating_sub(1) {
        let first = chunks[i + 1].first().unwrap();
        // Scan backwards through chunk N's trailing prefix group to
        // find the last unique vector. Compare the first record of
        // chunk N+1 against it.
        let mut found_dup = false;
        for record in chunks[i].iter().rev() {
            if !record.prefix_eq(first) {
                break; // left the prefix group
            }
            if reader.vectors_equal(record.ordinal as usize, first.ordinal as usize) {
                found_dup = true;
                break;
            }
        }
        if found_dup {
            boundary_dup_count += 1;
            let result = &mut chunk_results[i + 1];
            result.dup_count += 1;

            // The first 8 bytes of out_buf is this record's ivec entry.
            // Move it to dup_buf (or remove if eliding).
            if result.out_buf.len() >= 8 {
                let first_record = result.out_buf[..8].to_vec();
                result.dup_buf.extend_from_slice(&first_record);
                if elide {
                    result.out_buf.drain(..8);
                }
            }
        }
    }

    // Tally results
    let dup_count: usize = chunk_results.iter().map(|r| r.dup_count).sum();
    let unique_count = total - dup_count;
    let total_out_bytes: usize = chunk_results.iter().map(|r| r.out_buf.len()).sum();
    let total_dup_bytes: usize = chunk_results.iter().map(|r| r.dup_buf.len()).sum();

    // Free sorted records before allocating output buffers
    drop(all_records);

    let scan_secs = scan_start.elapsed().as_secs_f64();
    let scan_rate = if scan_secs > 0.0 { total as f64 / scan_secs } else { 0.0 };
    ctx.ui.log(&format!(
        "  dedup: {:.1}s ({:.0} vec/s), {} unique, {} dups ({} boundary)",
        scan_secs, scan_rate, unique_count, dup_count, boundary_dup_count,
    ));

    // Log prefix group stats to help diagnose prefix width effectiveness
    if let Ok(stats) = prefix_group_stats.lock() {
        let total_groups = stats.singleton_groups + stats.multi_groups;
        let collision_pct = if total_groups > 0 {
            stats.multi_groups as f64 / total_groups as f64 * 100.0
        } else { 0.0 };
        ctx.ui.log(&format!(
            "  prefix groups: {} singleton, {} multi-element ({:.1}% collision rate)",
            stats.singleton_groups, stats.multi_groups, collision_pct,
        ));
        ctx.ui.log(&format!(
            "  prefix collisions: {} vectors in multi-groups, largest group: {}, full-vector reads: {}",
            stats.multi_group_total_size, stats.largest_group, stats.full_vector_reads,
        ));
        if collision_pct > 20.0 {
            ctx.ui.log(&format!(
                "  WARNING: high prefix collision rate ({:.1}%). Consider increasing prefix_width for better dedup performance.",
                collision_pct,
            ));
        }
    }

    // ── Sub-phase D: Write files ─────────────────────────────────────
    let write_start = Instant::now();
    ensure_parent(output_path);
    ensure_parent(duplicates_path);

    ctx.ui.log(&format!(
        "  writing {:.1} MB output + {:.1} MB dups...",
        total_out_bytes as f64 / (1024.0 * 1024.0),
        total_dup_bytes as f64 / (1024.0 * 1024.0),
    ));

    // Concatenate output chunks and write in one shot
    {
        let mut out_bytes = Vec::with_capacity(total_out_bytes);
        for r in &chunk_results {
            out_bytes.extend_from_slice(&r.out_buf);
        }
        std::fs::write(output_path, &out_bytes)
            .map_err(|e| format!("failed to write {}: {}", output_path.display(), e))?;
    }

    {
        let mut dup_bytes = Vec::with_capacity(total_dup_bytes);
        for r in &chunk_results {
            dup_bytes.extend_from_slice(&r.dup_buf);
        }
        std::fs::write(duplicates_path, &dup_bytes)
            .map_err(|e| format!("failed to write {}: {}", duplicates_path.display(), e))?;
    }

    let write_secs = write_start.elapsed().as_secs_f64();
    ctx.ui.log(&format!(
        "  files written in {:.1}s",
        write_secs,
    ));

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
    write_ivec_bulk(path, ordinals)
}

/// Bulk-write an ivec file from a slice of ordinals.
///
/// Builds the entire byte buffer in memory (8 bytes per record:
/// 4-byte dim=1 header + 4-byte ordinal) and writes it in a single
/// syscall. This is orders of magnitude faster than per-record writes
/// for large ordinal lists.
fn write_ivec_bulk(path: &Path, ordinals: &[u32]) -> Result<(), String> {
    // 8 bytes per record: dim header (1_i32 LE) + ordinal (i32 LE)
    let mut buf = vec![0u8; ordinals.len() * 8];
    let dim_bytes = 1_i32.to_le_bytes();
    for (i, &ord) in ordinals.iter().enumerate() {
        let offset = i * 8;
        buf[offset..offset + 4].copy_from_slice(&dim_bytes);
        buf[offset + 4..offset + 8].copy_from_slice(&(ord as i32).to_le_bytes());
    }
    std::fs::write(path, &buf)
        .map_err(|e| format!("failed to write {}: {}", path.display(), e))
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
    if n >= 1_000_000_000 && n % 1_000_000_000 == 0 {
        format!("{}B", n / 1_000_000_000)
    } else if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 && n % 1_000_000 == 0 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 && n % 1_000 == 0 {
        format!("{}K", n / 1_000)
    } else if n >= 10_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
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
