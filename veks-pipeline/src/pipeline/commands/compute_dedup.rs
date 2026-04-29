// Copyright (c) Jonathan Shook
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
use std::sync::Arc;
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
///
/// Carries a separate `pread_file` handle for the phase-1 prefix pass.
/// That pass copies bytes through `read_at` into a per-batch heap
/// buffer rather than touching the mmap, so the source pages never
/// get mapped into this process's address space and never inflate RSS.
/// The mmap is still used for phase-2 random-access full-vector
/// comparisons (where the get_slice zero-copy path is faster than a
/// per-call pread).
enum VecReader {
    F32 { mmap: MmapVectorReader<f32>, pread_file: Arc<std::fs::File>, entry_size: usize },
    F16 { mmap: MmapVectorReader<half::f16>, pread_file: Arc<std::fs::File>, entry_size: usize },
}

impl VecReader {
    /// Open the appropriate reader based on file extension.
    fn open(path: &Path) -> Result<Self, String> {
        let pread_file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open {} for pread: {}", path.display(), e))?;
        let pread_file = Arc::new(pread_file);
        match path.extension().and_then(|e| e.to_str()) {
            Some("mvec") | Some("mvecs") => {
                let mmap = MmapVectorReader::<half::f16>::open_mvec(path)
                    .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
                let entry_size = mmap.entry_size();
                Ok(VecReader::F16 { mmap, pread_file, entry_size })
            }
            _ => {
                let mmap = MmapVectorReader::<f32>::open_fvec(path)
                    .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
                let entry_size = mmap.entry_size();
                Ok(VecReader::F32 { mmap, pread_file, entry_size })
            }
        }
    }

    fn count(&self) -> usize {
        match self {
            VecReader::F32 { mmap, .. } => <MmapVectorReader<f32> as VectorReader<f32>>::count(mmap),
            VecReader::F16 { mmap, .. } => <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(mmap),
        }
    }

    fn dim(&self) -> usize {
        match self {
            VecReader::F32 { mmap, .. } => <MmapVectorReader<f32> as VectorReader<f32>>::dim(mmap),
            VecReader::F16 { mmap, .. } => <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(mmap),
        }
    }

    fn entry_size(&self) -> usize {
        match self {
            VecReader::F32 { entry_size, .. } => *entry_size,
            VecReader::F16 { entry_size, .. } => *entry_size,
        }
    }

    fn pread_file(&self) -> &Arc<std::fs::File> {
        match self {
            VecReader::F32 { pread_file, .. } => pread_file,
            VecReader::F16 { pread_file, .. } => pread_file,
        }
    }

    /// Read the raw bytes (excluding the 4-byte dim header) of one
    /// vector at `index` into a fresh `Vec<u8>` via `pread`. Used by
    /// the phase-2 full-vector compare path so we don't touch the
    /// mmap and don't leak page maps into process RSS — the bytes
    /// flow through the kernel page cache (cache-warm if recently
    /// touched) and are copied into the returned buffer, then dropped
    /// at the end of the comparison call.
    fn pread_one_vector_bytes(&self, index: usize) -> Result<Vec<u8>, String> {
        use std::os::unix::fs::FileExt;
        let entry_size = self.entry_size();
        let value_bytes = entry_size - 4;
        let byte_offset = (index * entry_size + 4) as u64;  // skip i32 dim header
        let mut buf = vec![0u8; value_bytes];
        self.pread_file()
            .read_exact_at(&mut buf, byte_offset)
            .map_err(|e| format!("pread vector {}: {}", index, e))?;
        Ok(buf)
    }

    /// Get vector at index as f32 (upcasting f16 if needed). Uses
    /// `pread` (heap buffer, no mmap touch) so calls in tight loops
    /// don't accumulate mmap RSS.
    fn get_f32(&self, index: usize) -> Result<Vec<f32>, String> {
        let bytes = self.pread_one_vector_bytes(index)?;
        let dim = self.dim();
        match self {
            VecReader::F32 { .. } => {
                let mut out = Vec::with_capacity(dim);
                for i in 0..dim {
                    out.push(f32::from_le_bytes([
                        bytes[i * 4], bytes[i * 4 + 1],
                        bytes[i * 4 + 2], bytes[i * 4 + 3],
                    ]));
                }
                Ok(out)
            }
            VecReader::F16 { .. } => {
                let mut out = Vec::with_capacity(dim);
                for i in 0..dim {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    out.push(half::f16::from_bits(bits).to_f32());
                }
                Ok(out)
            }
        }
    }

    /// Bitwise equality check at native precision — pread both
    /// vectors' raw bytes and compare. Same RSS-safety contract as
    /// `get_f32`: no mmap touch.
    fn vectors_equal(&self, a: usize, b: usize) -> bool {
        let ba = match self.pread_one_vector_bytes(a) { Ok(v) => v, Err(_) => return false };
        let bb = match self.pread_one_vector_bytes(b) { Ok(v) => v, Err(_) => return false };
        ba == bb
    }

    /// Full lexicographic comparison of two vectors via `pread`. Used
    /// by the phase-2 in-group sort. The buffers live for the
    /// duration of one comparison and are dropped on return — no
    /// mmap pages get mapped into this process during the millions of
    /// compares phase 2 may issue.
    fn compare_vectors(&self, a: usize, b: usize) -> std::cmp::Ordering {
        let ba = match self.pread_one_vector_bytes(a) {
            Ok(v) => v,
            Err(_) => return std::cmp::Ordering::Equal,
        };
        let bb = match self.pread_one_vector_bytes(b) {
            Ok(v) => v,
            Err(_) => return std::cmp::Ordering::Equal,
        };
        let dim = self.dim();
        match self {
            VecReader::F32 { .. } => {
                for i in 0..dim {
                    let fa = f32::from_le_bytes([
                        ba[i * 4], ba[i * 4 + 1],
                        ba[i * 4 + 2], ba[i * 4 + 3],
                    ]);
                    let fb = f32::from_le_bytes([
                        bb[i * 4], bb[i * 4 + 1],
                        bb[i * 4 + 2], bb[i * 4 + 3],
                    ]);
                    match fa.partial_cmp(&fb) {
                        Some(std::cmp::Ordering::Equal) => continue,
                        Some(ord) => return ord,
                        None => {
                            if fa.is_nan() && fb.is_nan() { continue; }
                            if fa.is_nan() { return std::cmp::Ordering::Greater; }
                            return std::cmp::Ordering::Less;
                        }
                    }
                }
                std::cmp::Ordering::Equal
            }
            VecReader::F16 { .. } => {
                for i in 0..dim {
                    let fa = half::f16::from_bits(
                        u16::from_le_bytes([ba[i * 2], ba[i * 2 + 1]])
                    ).to_f32();
                    let fb = half::f16::from_bits(
                        u16::from_le_bytes([bb[i * 2], bb[i * 2 + 1]])
                    ).to_f32();
                    match fa.partial_cmp(&fb) {
                        Some(std::cmp::Ordering::Equal) => continue,
                        Some(ord) => return ord,
                        None => {
                            if fa.is_nan() && fb.is_nan() { continue; }
                            if fa.is_nan() { return std::cmp::Ordering::Greater; }
                            return std::cmp::Ordering::Less;
                        }
                    }
                }
                std::cmp::Ordering::Equal
            }
        }
    }

    fn format_name(&self) -> &'static str {
        match self {
            VecReader::F32 { .. } => "f32",
            VecReader::F16 { .. } => "f16",
        }
    }

    /// Hint the kernel to drop the page-cache pages backing source
    /// vectors `[start, end)`. Called after a batch's prefix reads
    /// complete to keep page-cache footprint bounded — without this,
    /// the source mmap holds onto every page touched, growing pcache
    /// linearly with bytes read until the kernel hits memory pressure
    /// (or the governor aborts the run for exceeding RSS ceiling).
    fn release_range(&self, start: usize, end: usize) {
        match self {
            VecReader::F32 { mmap, .. } => mmap.release_range(start, end),
            VecReader::F16 { mmap, .. } => mmap.release_range(start, end),
        }
    }

    /// Disable kernel readahead on the source mmap. Critical for the
    /// prefix-read pass: without it, every 4-byte prefix access faults
    /// in 256 KB of surrounding pages, inflating RSS by 64× more than
    /// the data we actually consume.
    fn advise_random(&self) {
        match self {
            VecReader::F32 { mmap, .. } => mmap.advise_random(),
            VecReader::F16 { mmap, .. } => mmap.advise_random(),
        }
    }

    /// Read a contiguous range of vector bytes via `pread` into the
    /// caller's pre-allocated buffer. Used by the phase-1 prefix
    /// pass — see `pread_vector_range_into`.
    ///
    /// `buf` is resized (via `set_len`) to fit exactly the requested
    /// byte range and then filled by parallel `pread` calls. Caller
    /// owns the allocation and reuses it across batches — critical
    /// to avoid paying ~50M anonymous-page-fault first-touch cost on
    /// every batch.
    fn pread_vector_range_into(
        &self,
        start: usize,
        end: usize,
        buf: &mut Vec<u8>,
    ) -> Result<(), String> {
        use rayon::prelude::*;
        use std::os::unix::fs::FileExt;
        let entry_size = self.entry_size();
        let byte_start = (start * entry_size) as u64;
        let byte_len = (end - start) * entry_size;

        // Resize to exactly the bytes we need. If the buffer's
        // existing capacity is enough, this is a no-op alloc; if
        // it's not, we grow it once (the caller's job to size it
        // for the largest batch up front so we never grow).
        if buf.capacity() < byte_len {
            buf.reserve(byte_len - buf.capacity());
        }
        // Promise the bytes are valid; they'll be filled below. The
        // first batch pays the page-fault cost; subsequent batches
        // hit already-committed pages.
        unsafe { buf.set_len(byte_len); }

        // Pick a parallelism that scales with the read size. Each
        // chunk should be large enough that the storage likes the
        // I/O size (>= 1 MiB) but small enough that we get real
        // parallelism across rayon threads.
        const CHUNK_BYTES: usize = 16 * 1024 * 1024;
        let n_chunks = ((byte_len + CHUNK_BYTES - 1) / CHUNK_BYTES).max(1);
        let chunk_size = (byte_len + n_chunks - 1) / n_chunks;
        let file = self.pread_file();

        buf.par_chunks_mut(chunk_size)
            .enumerate()
            .try_for_each(|(i, slice)| {
                let off = byte_start + (i * chunk_size) as u64;
                file.read_exact_at(slice, off).map_err(|e| format!(
                    "pread chunk {} ({} bytes @ {}): {}",
                    i, slice.len(), off, e,
                ))
            })
    }

    /// Decode the prefix of vector `local_idx` (offset within `batch_buf`)
    /// into `out`. `batch_buf` must be the bytes from
    /// [`pread_vector_range`] for this vector range. Decodes
    /// f16 → f32 in-line for f16 sources.
    fn prefix_from_buf_into(
        &self,
        batch_buf: &[u8],
        local_idx: usize,
        out: &mut [f32],
    ) {
        let entry_size = self.entry_size();
        let entry_offset = local_idx * entry_size + 4;  // skip i32 dim header
        let n = out.len();
        match self {
            VecReader::F32 { .. } => {
                let bytes_needed = n * 4;
                let src = &batch_buf[entry_offset..entry_offset + bytes_needed];
                for i in 0..n {
                    out[i] = f32::from_le_bytes([
                        src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3],
                    ]);
                }
            }
            VecReader::F16 { .. } => {
                let bytes_needed = n * 2;
                let src = &batch_buf[entry_offset..entry_offset + bytes_needed];
                for i in 0..n {
                    let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                    out[i] = half::f16::from_bits(bits).to_f32();
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
/// Fixed prefix width: 10 leading components for sort-key comparison.
/// 10 components provides strong discrimination for all known embedding
/// models, minimizing prefix collision groups that require full-vector
/// comparison during dedup. No variance sampling needed.
const PREFIX_WIDTH: usize = 10;
const MAX_PREFIX: usize = PREFIX_WIDTH;
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_COMPUTE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

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
        // When true, skip the per-ordinal random-access zero scan on
        // the surviving-uniques and duplicate-ordinals files. The
        // scan is a correctness crutch — dedup misattributes
        // (N-1) zeros as duplicates if the source has N identical
        // zero vectors — but it's expensive (one `pread` per
        // ordinal, billions on a billion-vector source) and not
        // every caller needs the precise zero-vs-dup split.
        // `analyze find-duplicates` sets this to true because it
        // only answers the duplicate question and lets
        // `analyze find-zeros` answer the zero question separately.
        let skip_zero_scan = options.get("skip-zero-scan")
            .map(|s| s == "true")
            .unwrap_or(false);
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
            // Aim for the per-batch peak to land at ~40% of ceiling.
            // The other ~25% headroom (we're already capped at 65% of
            // physical RAM) covers:
            //   - phase-2 `all_records` (~48 B/record, can be 30+ GiB
            //     on billion-record inputs, lives concurrently with
            //     output buffers, sort scratch, etc.);
            //   - kernel page cache for the source file that the
            //     governor counts as RSS-equivalent pressure;
            //   - allocator slack and rayon thread-local buffers.
            let target = (mem_ceiling as f64 * 0.40) as u64;
            let available = if snapshot.rss_bytes < target {
                target - snapshot.rss_bytes
            } else {
                (mem_ceiling as f64 * 0.10) as u64
            };
            // True per-batch peak memory cost. Contributions:
            //   - `batch_buf`:   one pread'd vector per entry → entry_size B
            //   - `entries`:     RunRecord + sort key → ~96 B + MAX_PREFIX*8 B
            //   - sort scratch:  rayon's par_sort_unstable_by uses a
            //                    temporary buffer roughly the same
            //                    size as the input → +entries again
            //   - allocator slack: jemalloc/glibc rounds up; conservatively
            //                    add 20% to the heap entries we control
            // Final factor: ~1.4× the strict accounting of (batch_buf +
            // entries) to leave headroom against measurement error.
            let entry_size = reader.entry_size() as u64;
            let strict = entry_size + 2 * (96 + (MAX_PREFIX as u64 * 8));
            let bytes_per_entry: u64 = (strict as f64 * 1.20) as u64;
            let governor_batch = (available / bytes_per_entry) as usize;
            // Floor at 10K so tiny inputs still get a real batch.
            // No hard upper cap — the governor sized this.
            let clamped = governor_batch.max(10_000).min(count.max(1));
            if clamped != DEFAULT_BATCH_SIZE {
                ctx.ui.log(&format!(
                    "  governor-derived batch size: {} (available: {} MiB, RSS: {} MiB, ceiling: {} MiB, per-entry: {} B incl. overhead)",
                    clamped,
                    available / (1 << 20),
                    snapshot.rss_bytes / (1 << 20),
                    mem_ceiling / (1 << 20),
                    bytes_per_entry,
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
        let phase = ctx.ui.spinner("prepare-vectors");
        let prefix_width = PREFIX_WIDTH.min(dim);

        // ── Phase 1: Create sorted runs ───────────────────────────────
        let phase1_start = Instant::now();
        phase.set_message("phase 1/3: building sorted runs".to_string());
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
        phase.set_message("phase 2/3: parallel merge".to_string());
        ctx.ui.log("Phase 2: parallel merge + dedup");
        ensure_parent(&output_path);
        ensure_parent(&report_path);

        ensure_parent(&duplicates_path);
        let (unique_count, dup_count) = match merge_runs(
            &run_files, prefix_width, &reader, dim, elide,
            &output_path, &duplicates_path, count,
            Some(phase.id()),
            ctx,
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

        // ── Zero vector accounting + removal ─────────────────────────
        // Tell the parent step spinner we've left phase 2 so the
        // user sees the new "phase 3/3: zero scan" message instead of
        // the stale "phase 2/3: parallel merge" while this scan runs.
        phase.set_message("phase 3/3: zero scan + report".to_string());
        // Attempting to L2-normalize a zero vector is an error, so
        // every zero must be caught here, not silently dropped during
        // extraction. The subtlety: dedup already ran, and if the
        // dataset contains N identical zero vectors, dedup routed one
        // survivor to the output ordinals and (N-1) to the duplicates
        // ordinals — misattributing (N-1) zeros as "duplicates".
        //
        // To report counts accurately we must scan BOTH files for
        // zero-norm vectors:
        //   - Zeros in `output_path` are removed (rewrite without them).
        //   - Zeros in `duplicates_path` are counted toward `zero_count`
        //     and subtracted from `dup_count`. They can stay in the
        //     duplicates ordinals file — downstream consumers use that
        //     list to *exclude* ordinals anyway, so zero-duplicates
        //     still correctly get excluded. Only the reported counts
        //     need fixing.
        // Parallel zero scan. The scan is one pread + norm² per
        // ordinal, and on a billion-record input that's a billion
        // syscalls — single-threaded this takes 15+ minutes of silent
        // wall time even with all pages cache-warm. Rayon-parallelize
        // and emit a progress bar so the step is observable.
        let scan_zeros = |data: &[u8], label: &str| -> Result<Vec<usize>, String> {
            use rayon::prelude::*;
            let record_size = 4 + 4; // dim=1 (i32) + ordinal (i32)
            let ord_count = data.len() / record_size;
            if ord_count == 0 { return Ok(Vec::new()); }

            let pb = ctx.ui.bar_with_unit(ord_count as u64, label, "vec");
            let pb_id = pb.id();
            let progress = std::sync::atomic::AtomicU64::new(0);
            // Flush ~200 progress updates total over the whole scan.
            let flush_every = ((ord_count / 200).max(50_000)) as u64;

            let result: Result<Vec<usize>, String> = (0..ord_count)
                .into_par_iter()
                .filter_map(|i| {
                    let offset = i * record_size + 4;
                    let ordinal = i32::from_le_bytes([
                        data[offset], data[offset+1],
                        data[offset+2], data[offset+3],
                    ]) as usize;
                    let vec = match reader.get_f32(ordinal) {
                        Ok(v) => v,
                        Err(e) => return Some(Err(format!("zero scan: {}", e))),
                    };
                    // Throttled progress flush.
                    let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if done % flush_every == 0 {
                        ctx.ui.inc_by_id(pb_id, flush_every);
                    }
                    // norm² == 0 ⇔ all components exactly zero in f32;
                    // keep it as a squared-sum (no sqrt needed).
                    let norm_sq: f64 = vec.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    if norm_sq == 0.0 {
                        Some(Ok(i))
                    } else {
                        None
                    }
                })
                .collect();
            // Final flush of any remainder.
            let total_done = progress.load(std::sync::atomic::Ordering::Relaxed);
            let last_full_flush = (total_done / flush_every) * flush_every;
            let tail = total_done.saturating_sub(last_full_flush);
            if tail > 0 {
                ctx.ui.inc_by_id(pb_id, tail);
            }
            pb.finish();
            result
        };

        // Two ways to skip the expensive per-ordinal zero scan:
        //  1. The caller explicitly asked to skip it (`skip-zero-scan=true`).
        //     `analyze find-duplicates` does this — it only answers the
        //     duplicate question; `analyze find-zeros` handles zeros
        //     separately at ~4× the throughput of this scan.
        //  2. A prior `find-zeros` run recorded `zero_count=0` in
        //     variables.yaml or the state map. A zero-free source
        //     can't have zero survivors in either the uniques or
        //     duplicates lists, so there's nothing to find.
        //
        // In either case: exit with `zeros=0` for both outputs. Any
        // positive `zero_count` (or absence of the hint) and no
        // explicit skip means we fall through to the full scan.
        let known_zero_count: Option<u64> = ctx.defaults.get("zero_count")
            .and_then(|v| v.parse().ok())
            .or_else(|| crate::pipeline::variables::load(&ctx.workspace).ok()
                .and_then(|v| v.get("zero_count").and_then(|s| s.parse().ok())));
        let skip_zero_sweep = skip_zero_scan || known_zero_count == Some(0);
        if skip_zero_scan {
            ctx.ui.log("  zero-scan skipped: caller opted out (skip-zero-scan=true)");
        } else if known_zero_count == Some(0) {
            ctx.ui.log("  zero-scan skipped: zero_count=0 from prior analyze find-zeros run");
        }

        // Scan the surviving-uniques path and rewrite it without zeros.
        let output_zero_count = if skip_zero_sweep { 0 } else {
            let ord_data = match std::fs::read(&output_path) {
                Ok(d) => d,
                Err(e) => return error_result(format!("read ordinals for zero scan: {}", e), start),
            };
            let record_size = 4 + 4;
            let ord_count = ord_data.len() / record_size;
            let zeros = match scan_zeros(&ord_data, "zero scan: uniques") {
                Ok(z) => z,
                Err(e) => return error_result(e, start),
            };
            if !zeros.is_empty() {
                let zero_set: std::collections::HashSet<usize> = zeros.iter().copied().collect();
                let mut clean_data = Vec::with_capacity(ord_data.len());
                for i in 0..ord_count {
                    if !zero_set.contains(&i) {
                        let s = i * record_size;
                        clean_data.extend_from_slice(&ord_data[s..s + record_size]);
                    }
                }
                if let Err(e) = std::fs::write(&output_path, &clean_data) {
                    return error_result(format!("rewrite ordinals without zeros: {}", e), start);
                }
                ctx.ui.log(&format!("  removed {} zero vector(s) from ordinals", zeros.len()));
            }
            zeros.len()
        };

        // Scan the duplicates path — every zero found here was a zero
        // that dedup saw as a "duplicate of another zero". For
        // accurate attribution, count it as a zero, not a dup.
        // Same short-circuit applies — see the output_zero_count
        // block for the full reasoning.
        let dup_zero_count = if skip_zero_sweep { 0 } else {
            match std::fs::read(&duplicates_path) {
                Ok(data) if !data.is_empty() => {
                    let zeros = match scan_zeros(&data, "zero scan: dups") {
                        Ok(z) => z,
                        Err(e) => return error_result(e, start),
                    };
                    zeros.len()
                }
                _ => 0,
            }
        };
        if dup_zero_count > 0 {
            ctx.ui.log(&format!(
                "  reclassified {} zero vector(s) from duplicate list (they were duplicates-of-zero)",
                dup_zero_count,
            ));
        }

        // Accurate, non-overlapping counts. The three buckets must
        // partition `count` exactly:
        //   count = unique_count + dup_count + zero_count
        // Re-attribute the zero ordinals dedup misclassified as
        // duplicates of each other:
        //   - `output_zero_count` zeros came from the survivors-of-
        //     dedup bucket (`unique_count`); they're now zeros.
        //   - `dup_zero_count` zeros came from the duplicates-of-
        //     each-other bucket (`dup_count`); they're now zeros.
        let zero_count = output_zero_count + dup_zero_count;
        let unique_count = unique_count.saturating_sub(output_zero_count);
        let dup_count = dup_count.saturating_sub(dup_zero_count);
        debug_assert_eq!(unique_count + dup_count + zero_count, count,
            "bucket partition broken: unique={} + dup={} + zero={} != count={}",
            unique_count, dup_count, zero_count, count);

        // ── Phase 3: Write report and clean up ────────────────────────
        phase.set_message("phase 3/3: writing report".to_string());
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

        // Write verified counts so the bound checker can validate the
        // output. `unique_count` is already post-zero-removal — it
        // excludes the zeros that survived dedup — so the file at
        // `output_path` (with elide=true) has exactly `unique_count`
        // records. With elide=false the file holds (count - zero_count)
        // records: dups stay in but zeros were removed from output_path
        // by the rewrite above.
        let output_records = if elide { unique_count } else { count - zero_count };
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &output_records.to_string());
        ctx.defaults.insert(var_name, output_records.to_string());

        // The duplicates file still physically contains zero ordinals
        // (we didn't move them out — exclusion-by-list still works
        // identically). The reported count is the true non-zero dup
        // count, but the file's record count is dup_count + dup_zero_count.
        let dup_file_records = dup_count + dup_zero_count;
        let dup_var_name = format!("verified_count:{}",
            duplicates_path.file_name().and_then(|n| n.to_str()).unwrap_or("dups"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &dup_var_name, &dup_file_records.to_string());
        ctx.defaults.insert(dup_var_name, dup_file_records.to_string());

        // clean_count = the population available for downstream sampling
        // (shuffle, extract-base, extract-queries). Always equals
        // `unique_count` post-zero-removal regardless of elide setting,
        // since every consumer applies the duplicates list as an
        // exclusion filter.
        let clean_count = unique_count;
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, "clean_count", &clean_count.to_string());
        ctx.defaults.insert("clean_count".into(), clean_count.to_string());

        // Record zero count
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, "zero_count", &zero_count.to_string());
        ctx.defaults.insert("zero_count".into(), zero_count.to_string());

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

        let produced = vec![output_path, duplicates_path, report_path];

        CommandResult {
            status: Status::Ok,
            message: msg,
            produced,
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
        let output_keys = vec!["output", "duplicates", "report"];
        let mut manifest = crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &output_keys,
        );
        // Sorted run files in .cache/dedup_runs/ are intermediate artifacts
        manifest.intermediates.push("${cache}/dedup_runs/".to_string());
        manifest
    }
}

// Variance sampling (Phase 0) removed — prefix width is fixed at
// PREFIX_WIDTH (10 components). This provides strong discrimination
// for all known embedding models.

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
    // The prefix-read pass touches only ~4–40 bytes per vector but
    // those reads are spread across a multi-TB file. Default kernel
    // readahead would map in ~256 KB around each access, inflating
    // process RSS by 64×+ over the actual prefix data — the dominant
    // cause of the "RSS over ceiling" abort on huge sources. Disable
    // readahead for this pass; on-demand single-page faults are what
    // we want here.
    reader.advise_random();

    let pb = ctx.ui.bar_with_unit(count as u64, "building runs", "vec");

    // Reusable batch buffer. Pre-allocate to the largest batch's
    // byte size so it never grows during the run, and the
    // first-touch anonymous-page faults happen ONCE on the first
    // batch instead of on every batch. For a 192 GiB buffer that's
    // ~50 saved batches × ~30s of fault overhead each.
    let max_batch_bytes = batch_size * reader.entry_size();
    let mut batch_buf: Vec<u8> = Vec::with_capacity(max_batch_bytes);

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

        // Pull the entire batch's bytes into our reusable heap
        // buffer via parallel pread. Bypasses the source mmap
        // entirely — the bytes flow through the kernel page cache
        // (cache-warm if the file's been touched recently) but are
        // copied into our owned buffer rather than mapped into our
        // address space.
        //
        // The buffer is owned outside this loop and reused across
        // every batch, so the per-batch first-touch page-fault cost
        // is paid once on iteration 1 and the rest hit committed
        // pages.
        reader.pread_vector_range_into(batch_start, batch_end, &mut batch_buf)?;

        let mut entries: Vec<([u32; MAX_PREFIX], RunRecord)>;
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};

            let progress = AtomicU64::new(0);
            // Update progress every ~0.5% of batch or 5K vectors
            let update_interval = (batch_len / 200).max(5_000) as u64;
            let pw = prefix_width;

            let build_fn = || {
                (0..batch_len)
                    .into_par_iter()
                    .map(|local_i| {
                        let i = batch_start + local_i;
                        let mut prefix_buf = [0.0f32; MAX_PREFIX];
                        reader.prefix_from_buf_into(&batch_buf, local_i, &mut prefix_buf[..pw]);
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
        // batch_buf is owned by the loop scope — it'll be reused
        // for the next batch (committed pages, no faults).
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

        // Drop the page-cache pages backing this batch's source range.
        // Without this, the source mmap accumulates every byte we
        // touched into pcache — for a 1.4 TB input that means pcache
        // grows linearly and eventually pushes the process past the
        // governor's RSS ceiling. The merge phase later does sparse
        // random-access reads into the same file; those will re-fault
        // pages on demand, which is fine — the cost is ~one page per
        // full-vector compare, which only happens for prefix-collision
        // groups (already a small fraction of records).
        reader.release_range(batch_start, batch_end);

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
    phase_id: Option<veks_core::ui::ProgressId>,
    ctx: &mut StreamContext,
) -> Result<(usize, usize), String> {
    use rayon::prelude::*;

    // Macro that updates the parent step spinner's message so the
    // user sees which sub-phase of phase 2 is currently running. The
    // top-line spinner otherwise stays on a stale "phase 2/3:
    // parallel merge" for the full ~minute the merge takes. (Macro
    // rather than closure so it doesn't fight with `ctx`'s mutable
    // borrow elsewhere in this function.)
    macro_rules! set_phase_msg {
        ($msg:expr) => {
            if let Some(id) = phase_id {
                ctx.ui.set_message_by_id(id, $msg.to_string());
            }
        };
    }

    let threads = ctx.governor.current_or("threads", ctx.threads as u64)
        .max(1) as usize;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    // ── Sub-phase A: Load all runs in parallel ───────────────────────
    set_phase_msg!("phase 2/3: loading sorted runs");
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
    set_phase_msg!("phase 2/3: flattening run records");
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
    set_phase_msg!("phase 2/3: sorting all records by prefix");
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
    set_phase_msg!("phase 2/3: dedup scan + emit ordinals");
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

        // Throttled in-chunk progress reporting. With 128 parallel
        // chunks each holding ~7-8M records and the dedup pass taking
        // ~tens of seconds per chunk, a single inc-at-end means the
        // bar appears frozen for the whole phase then jumps to 100%.
        // Flush the local counter every PROGRESS_FLUSH records so the
        // bar advances smoothly while keeping the UI event volume
        // bounded (~20 inc events per chunk per 100k records).
        const PROGRESS_FLUSH: usize = 100_000;
        let mut progress_pending: usize = 0;

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
                // Multi-element prefix group — sort by full vector content
                // within the group so that non-identical vectors are in
                // lexicographic order. Then deduplicate by comparing
                // consecutive entries (identical vectors are now adjacent).
                local_stats.multi_groups += 1;
                local_stats.multi_group_total_size += group_len;
                if group_len > local_stats.largest_group {
                    local_stats.largest_group = group_len;
                }
                local_stats.full_vector_reads += group_len;

                // Sort the group slice by full vector content.
                // We collect ordinals, sort them by vector data, then
                // process in sorted order.
                let mut group_ords: Vec<u32> = chunk[group_start..group_end]
                    .iter()
                    .map(|r| r.ordinal)
                    .collect();
                group_ords.sort_unstable_by(|&a, &b| {
                    reader.compare_vectors(a as usize, b as usize)
                });

                let mut last_unique_ord = group_ords[0];
                out_buf.extend_from_slice(&dim_bytes);
                out_buf.extend_from_slice(&(last_unique_ord as i32).to_le_bytes());

                for &ord in &group_ords[1..] {
                    let is_dup = reader.vectors_equal(
                        last_unique_ord as usize, ord as usize,
                    );

                    let ord_bytes = (ord as i32).to_le_bytes();
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
                        last_unique_ord = ord;
                    }
                }
            }
            // Account every record in the group toward progress —
            // singletons are O(1), multi-element groups did the work
            // of sorting and dedup'ing, but from the user's
            // perspective they've all been "processed".
            progress_pending += group_len;
            if progress_pending >= PROGRESS_FLUSH {
                ctx.ui.inc_by_id(pb_id, progress_pending as u64);
                progress_pending = 0;
            }
            group_start = group_end;
        }

        // Flush any remaining tail.
        if progress_pending > 0 {
            ctx.ui.inc_by_id(pb_id, progress_pending as u64);
        }
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
    set_phase_msg!("phase 2/3: writing output ordinals");
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
/// Deduplication report — shared between writer and reader.
/// Used by `compute sort` (write) and `analyze find-duplicates` (read).
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct DedupReport {
    pub total_vectors: usize,
    pub unique_vectors: usize,
    pub duplicate_vectors: usize,
    pub duplicate_ratio: f64,
    pub has_duplicates: bool,
    pub prefix_width: usize,
}

fn write_report(
    path: &Path,
    total: usize,
    unique: usize,
    duplicates: usize,
    prefix_width: usize,
) -> Result<(), String> {
    let report = DedupReport {
        total_vectors: total,
        unique_vectors: unique,
        duplicate_vectors: duplicates,
        duplicate_ratio: if total > 0 { duplicates as f64 / total as f64 } else { 0.0 },
        has_duplicates: duplicates > 0,
        prefix_width,
    };
    let content = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("serialize report: {}", e))?;
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
