// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Compile-once, drain-at-batch-granularity parquet → xvec extractor.
//!
//! This is the fast-path counterpart to [`super::parquet_compiler`] (the
//! MNode metadata compiler). Both follow the same meta-compiler pattern —
//! analyze an Arrow schema once to build a typed plan, then iterate record
//! batches through that plan — but they optimize for different output shapes:
//!
//! - **Metadata** (`CompiledMnodeWriter`) emits TLV-framed field records,
//!   so the per-row work is unavoidably K writes (one per field).
//! - **Vector** (`CompiledVectorExtractor`, this module) emits a single
//!   homogeneous payload per row with no envelope, so the hot loop can
//!   hoist the Arrow downcast and type match OUT of the row loop and
//!   iterate the underlying Arrow primitive buffer as a contiguous slice.
//!
//! On little-endian platforms the entire batch's value buffer matches the
//! xvec wire format bit-for-bit, so each row is one bounds-checked slice
//! and one [`VecSink::write_record`] call — no downcasts, no per-row
//! allocations, no virtual dispatch beyond the sink.

use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// Maximum number of parquet files in flight at once across all worker
/// threads in the parallel shard path.
///
/// Two competing constraints set this:
///
/// 1. **EBS read concurrency.** Each in-flight decoder holds a source
///    fd open. If the prefetch pool falls behind, the decoder's
///    `read_to_end` triggers its own kernel readahead, adding to
///    in-flight reads against the volume. Each fd typically pipelines
///    ~4 outstanding readahead I/Os, so `in_flight × 4` is a rough
///    upper bound on EBS reads contributed by decoders.
/// 2. **CPU decode concurrency.** Arrow → xvec serialization is
///    single-threaded per worker; decoders below this number leave
///    cores idle and slow the whole pipeline.
///
/// 16 sits at the sweet spot for the billion-vector workload: ~64 EBS
/// reads contributed by decoders (well under the productive knee) +
/// enough CPU parallelism to keep the local-NVMe write side fed.
/// Memory is not the bottleneck — 16 × 360 MiB ≈ 6 GiB resident is
/// trivial on any EBS host.
const MAX_IN_FLIGHT: usize = 16;

/// Hard cap on the number of loader threads, regardless of what the caller
/// passes. Parquet decoding is mostly I/O-bound, so beyond ~16-32 readers
/// you saturate the disk and just pay scheduler overhead. The existing slow
/// `ParquetDirReader` uses the same 64 cap for consistency.
const MAX_LOADER_THREADS: usize = 64;

/// Number of dedicated prefetch threads that warm the page cache for
/// upcoming source parquet files via real `read()` syscalls. Decoupled
/// from decoder workers so reads always have a driver, even when every
/// decoder is busy on CPU-bound batch decoding.
///
/// 4 is enough to saturate EBS gp3/io2 read bandwidth in practice:
/// each thread keeps one source fd open with `POSIX_FADV_SEQUENTIAL`
/// widening the kernel's readahead window, so each prefetcher already
/// drives ~10–20 outstanding 1–4 MiB readahead requests on its own.
/// 4 prefetchers × that pipeline puts the volume's queue depth in the
/// productive 16–32 range. Going higher just over-fills the queue and
/// pushes tail latency up without buying more MB/s.
const PREFETCH_THREADS: usize = 4;

/// Cap on how far ahead the prefetch pool may run beyond completed
/// shards. Bounds the cached-but-not-yet-decoded working set to roughly
/// `MAX_LOOKAHEAD × parquet_size`. At 16 × 120 MiB ≈ 2 GiB resident,
/// well below the dirty-page budget on any EBS host with > 8 GiB RAM.
const MAX_LOOKAHEAD: usize = 16;

/// Discard buffer size for prefetch reads. `posix_fadvise(WILLNEED)`
/// alone doesn't reliably populate large files (it's bounded by the
/// per-BDI `read_ahead_kb`, typically 128 KiB), so each prefetch
/// thread does real `read()`s into a throwaway buffer to force the
/// pages into the page cache.
///
/// 16 MiB amortizes syscall + bookkeeping overhead well, and the
/// kernel coalesces readahead better when handed larger user-space
/// reads (it can keep the pipe full with fewer block-layer requests
/// per MB transferred).
const PREFETCH_CHUNK: usize = 16 * 1024 * 1024;

/// Granularity at which the shard writer chunks `write_all_at` calls
/// and kicks off writeback via `sync_file_range`. Keeps the EBS write
/// queue continuously fed with 16 MiB writes back-to-back instead of
/// letting the kernel accumulate the whole shard's dirty pages and
/// then burst them out. Same value as the xvec_dir_compiler concat
/// path, for the same reasoning.
const WRITE_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// Cadence of progress-callback updates from the parallel-drain ticker
/// thread. The writer thread bumps a shared atomic after each batch; the
/// ticker reads it on this interval and emits a single `cb(delta, total)`
/// per tick. This decouples UI updates from per-batch drain so the writer
/// hot path stays tight (no UI work per record or per batch) while the
/// records-per-second display still updates smoothly.
const PROGRESS_TICK: std::time::Duration = std::time::Duration::from_millis(250);

/// Soft cap on the per-worker scratch buffer's size before we flush it via
/// `pwrite`. EBS reaches its throughput ceiling on large sequential writes;
/// the parallel pwrite path therefore *accumulates* every batch's bytes for
/// a single file into one buffer and issues one large write at the end of
/// the file. For very large parquet shards we still cap the buffer size so
/// per-worker memory stays bounded — when the buffer crosses this threshold
/// we flush mid-file and reset.
///
/// 256 MiB is well above the EBS sweet spot (>= 1 MiB sequential) while
/// keeping per-worker memory predictable: with [`MAX_IN_FLIGHT`] active
/// workers, peak scratch budget is `MAX_IN_FLIGHT × WRITE_FLUSH_BYTES` ≈
/// 8 GiB on top of decoded Arrow batches.
const WRITE_FLUSH_BYTES: usize = 256 * 1024 * 1024;

use arrow::array::{
    Array, ArrayRef, Float16Array, Float32Array, Float64Array,
    FixedSizeListArray, Int16Array, Int32Array, Int64Array, Int8Array,
    ListArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use veks_io::VecSink;

use super::VecFormat;

/// Kind of list structure carrying the vector payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListKind {
    /// `FixedSizeList<T>` — every row has the same dimension (in the type).
    FixedSize,
    /// `List<T>` — dimension is per-row; caller must validate uniformity
    /// if emitting into a fixed-dim xvec output.
    Variable,
}

/// Element type of the vector payload, one-to-one with supported xvec
/// element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorElement {
    F32, F64, F16,
    U8, I8,
    U16, I16,
    U32, I32,
    U64, I64,
}

impl VectorElement {
    /// Bytes per element.
    pub fn element_size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Preferred xvec output format for this element type.
    pub fn preferred_xvec_format(self) -> VecFormat {
        match self {
            Self::F32 => VecFormat::Fvec,
            Self::F64 => VecFormat::Dvec,
            Self::F16 => VecFormat::Mvec,
            Self::U8  => VecFormat::Bvec,
            Self::I8  => VecFormat::I8vec,
            Self::U16 => VecFormat::U16vec,
            Self::I16 => VecFormat::Svec,
            Self::U32 => VecFormat::U32vec,
            Self::I32 => VecFormat::Ivec,
            Self::U64 => VecFormat::U64vec,
            Self::I64 => VecFormat::I64vec,
        }
    }

    fn from_arrow(dt: &DataType) -> Option<Self> {
        match dt {
            DataType::Float32 => Some(Self::F32),
            DataType::Float64 => Some(Self::F64),
            DataType::Float16 => Some(Self::F16),
            DataType::UInt8   => Some(Self::U8),
            DataType::Int8    => Some(Self::I8),
            DataType::UInt16  => Some(Self::U16),
            DataType::Int16   => Some(Self::I16),
            DataType::UInt32  => Some(Self::U32),
            DataType::Int32   => Some(Self::I32),
            DataType::UInt64  => Some(Self::U64),
            DataType::Int64   => Some(Self::I64),
            _ => None,
        }
    }
}

/// Compiled plan for extracting vector records from a parquet source.
///
/// Built once from the Arrow schema (see [`Self::compile`]) and reused for
/// every record batch. All per-row Arrow downcasts and type branches are
/// resolved during compilation.
#[derive(Debug, Clone)]
pub struct CompiledVectorExtractor {
    col_index: usize,
    list_kind: ListKind,
    element: VectorElement,
    /// For `FixedSize` this is set from the column type. For `Variable` it
    /// is 0 until a batch is drained (drain validates all rows match the
    /// first row's dimension).
    dimension: u32,
}

impl CompiledVectorExtractor {
    /// Analyze an Arrow schema and build an extractor.
    ///
    /// If `column_hint` is provided, matches the column by name. Otherwise
    /// picks the first column whose type is `FixedSizeList<T>` or `List<T>`
    /// with `T` a supported numeric primitive.
    pub fn compile(schema: &Schema, column_hint: Option<&str>) -> Result<Self, String> {
        let (col_index, data_type, field_name) = match column_hint {
            Some(name) => {
                let idx = schema
                    .fields()
                    .iter()
                    .position(|f| f.name() == name)
                    .ok_or_else(|| {
                        format!(
                            "column '{}' not found in parquet schema (available: {})",
                            name,
                            schema.fields().iter().map(|f| f.name().as_str())
                                .collect::<Vec<_>>().join(", "),
                        )
                    })?;
                let f = schema.field(idx);
                (idx, f.data_type().clone(), f.name().to_string())
            }
            None => {
                let (idx, f) = schema
                    .fields()
                    .iter()
                    .enumerate()
                    .find(|(_, f)| is_numeric_list(f.data_type()))
                    .ok_or_else(|| {
                        "no list-of-numeric column found in parquet schema".to_string()
                    })?;
                (idx, f.data_type().clone(), f.name().to_string())
            }
        };

        match &data_type {
            DataType::FixedSizeList(inner, size) => {
                let element = VectorElement::from_arrow(inner.data_type()).ok_or_else(|| {
                    format!(
                        "column '{}': unsupported element type {:?}",
                        field_name,
                        inner.data_type()
                    )
                })?;
                if *size <= 0 {
                    return Err(format!(
                        "column '{}': FixedSizeList size must be positive, got {}",
                        field_name, size
                    ));
                }
                Ok(Self {
                    col_index,
                    list_kind: ListKind::FixedSize,
                    element,
                    dimension: *size as u32,
                })
            }
            DataType::List(inner) => {
                let element = VectorElement::from_arrow(inner.data_type()).ok_or_else(|| {
                    format!(
                        "column '{}': unsupported element type {:?}",
                        field_name,
                        inner.data_type()
                    )
                })?;
                Ok(Self {
                    col_index,
                    list_kind: ListKind::Variable,
                    element,
                    dimension: 0,
                })
            }
            other => Err(format!(
                "column '{}' is {:?}, expected FixedSizeList or List",
                field_name, other
            )),
        }
    }

    /// Column position in the parquet schema (for diagnostics).
    pub fn column_index(&self) -> usize { self.col_index }

    /// Element type of the vector payload.
    pub fn element(&self) -> VectorElement { self.element }

    /// Whether the underlying list is fixed-size or variable-length.
    pub fn list_kind(&self) -> ListKind { self.list_kind }

    /// Vector dimension. Zero for `Variable` lists until the first batch
    /// has been drained (at which point the caller can query).
    pub fn dimension(&self) -> u32 { self.dimension }

    /// Preferred xvec output format for this element type.
    pub fn preferred_xvec_format(&self) -> VecFormat {
        self.element.preferred_xvec_format()
    }

    /// Drain every row of `batch` into `sink`, starting at ordinal
    /// `ordinal_start` and incrementing by one per row.
    ///
    /// Returns the number of rows written. Per-row cost is one bounds-checked
    /// slice plus one `write_record` call — Arrow downcasts and the element-type
    /// match happen once per batch, not per row.
    pub fn drain_batch(
        &mut self,
        batch: &RecordBatch,
        ordinal_start: i64,
        sink: &mut dyn VecSink,
    ) -> Result<usize, String> {
        let col = batch.column(self.col_index);
        match self.list_kind {
            ListKind::FixedSize => self.drain_fixed_size(col, ordinal_start, sink),
            ListKind::Variable => self.drain_variable(col, ordinal_start, sink),
        }
    }

    fn drain_fixed_size(
        &self,
        col: &ArrayRef,
        ordinal_start: i64,
        sink: &mut dyn VecSink,
    ) -> Result<usize, String> {
        let arr = col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| "column is not FixedSizeListArray (schema drift?)".to_string())?;
        let row_count = arr.len();
        let dim = self.dimension as usize;
        let row_bytes = dim * self.element.element_size();
        let values = arr.values();

        drain_primitive_rows(self.element, values, row_count, dim, row_bytes, ordinal_start, sink)
    }

    fn drain_variable(
        &mut self,
        col: &ArrayRef,
        ordinal_start: i64,
        sink: &mut dyn VecSink,
    ) -> Result<usize, String> {
        let arr = col
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| "column is not ListArray (schema drift?)".to_string())?;
        let row_count = arr.len();
        if row_count == 0 { return Ok(0); }

        // Infer (or verify) the uniform dimension from the first row.
        let first_len = arr.value(0).len();
        if self.dimension == 0 {
            self.dimension = first_len as u32;
        } else if self.dimension as usize != first_len {
            return Err(format!(
                "list dimension changed across batches: previously {}, now {}",
                self.dimension, first_len
            ));
        }
        let dim = self.dimension as usize;
        let row_bytes = dim * self.element.element_size();

        // For List (as opposed to FixedSizeList) the value buffer is a flat
        // concatenation of all rows. Offsets tell us where each row starts.
        // When every row has the same length, the buffer is exactly the
        // FixedSizeList layout — we can take the same contiguous-slice fast
        // path after validating uniformity.
        let offsets = arr.offsets();
        if offsets.len() != row_count + 1 {
            return Err(format!(
                "ListArray offsets length {} inconsistent with row_count {}",
                offsets.len(), row_count
            ));
        }
        for row in 0..row_count {
            let len = (offsets[row + 1] - offsets[row]) as usize;
            if len != dim {
                return Err(format!(
                    "row {} has dimension {} but expected {} — emit into a \
                     variable-length vvec output instead",
                    row, len, dim
                ));
            }
        }

        // The values buffer may have leading slack (if the batch was sliced)
        // — use the offset of row 0 as the starting index.
        let base_offset = offsets[0] as usize;
        let values = arr.values();
        drain_primitive_rows_offset(
            self.element, values, row_count, dim, row_bytes, base_offset, ordinal_start, sink,
        )
    }
}

fn is_numeric_list(dt: &DataType) -> bool {
    match dt {
        DataType::List(inner) | DataType::FixedSizeList(inner, _) => {
            VectorElement::from_arrow(inner.data_type()).is_some()
        }
        _ => false,
    }
}

/// Drain `row_count` rows of `dim` elements each from a primitive Arrow array.
/// The values are assumed to start at element offset 0 — appropriate for
/// `FixedSizeListArray::values()` and for `ListArray::values()` when row 0's
/// offset is zero (see `drain_primitive_rows_offset` for the general case).
fn drain_primitive_rows(
    element: VectorElement,
    values: &ArrayRef,
    row_count: usize,
    dim: usize,
    row_bytes: usize,
    ordinal_start: i64,
    sink: &mut dyn VecSink,
) -> Result<usize, String> {
    drain_primitive_rows_offset(
        element, values, row_count, dim, row_bytes, 0, ordinal_start, sink,
    )
}

fn drain_primitive_rows_offset(
    element: VectorElement,
    values: &ArrayRef,
    row_count: usize,
    dim: usize,
    row_bytes: usize,
    base_offset: usize,
    ordinal_start: i64,
    sink: &mut dyn VecSink,
) -> Result<usize, String> {
    macro_rules! drain_le_primitive {
        ($array_ty:ty, $prim_ty:ty) => {{
            let arr = values
                .as_any()
                .downcast_ref::<$array_ty>()
                .ok_or_else(|| format!("values not {}", stringify!($array_ty)))?;
            let raw: &[$prim_ty] = arr.values();
            let expected_elems = base_offset + row_count * dim;
            if raw.len() < expected_elems {
                return Err(format!(
                    "values buffer has {} elements, need {} (offset {}, {} rows × {} dim)",
                    raw.len(), expected_elems, base_offset, row_count, dim,
                ));
            }
            write_rows_primitive::<$prim_ty>(raw, base_offset, row_count, dim, row_bytes, ordinal_start, sink);
            Ok(row_count)
        }};
    }

    match element {
        VectorElement::F32 => drain_le_primitive!(Float32Array, f32),
        VectorElement::F64 => drain_le_primitive!(Float64Array, f64),
        VectorElement::U8  => drain_le_primitive!(UInt8Array, u8),
        VectorElement::I8  => drain_le_primitive!(Int8Array, i8),
        VectorElement::U16 => drain_le_primitive!(UInt16Array, u16),
        VectorElement::I16 => drain_le_primitive!(Int16Array, i16),
        VectorElement::U32 => drain_le_primitive!(UInt32Array, u32),
        VectorElement::I32 => drain_le_primitive!(Int32Array, i32),
        VectorElement::U64 => drain_le_primitive!(UInt64Array, u64),
        VectorElement::I64 => drain_le_primitive!(Int64Array, i64),
        VectorElement::F16 => {
            // half::f16 is a native 2-byte type, but Arrow exposes it via
            // Float16Array. Treat the backing buffer as `u16` for the
            // LE byte-slice view — the bit pattern matches the on-disk
            // xvec f16 layout verbatim.
            let arr = values
                .as_any()
                .downcast_ref::<Float16Array>()
                .ok_or_else(|| "values not Float16Array".to_string())?;
            let raw: &[half::f16] = arr.values();
            let expected_elems = base_offset + row_count * dim;
            if raw.len() < expected_elems {
                return Err(format!(
                    "values buffer has {} f16 elements, need {}",
                    raw.len(), expected_elems,
                ));
            }
            write_rows_primitive::<half::f16>(raw, base_offset, row_count, dim, row_bytes, ordinal_start, sink);
            Ok(row_count)
        }
    }
}

/// Write `row_count` rows of `dim` elements of type `T` from the flat
/// primitive buffer `raw[base_offset..]` into `sink`.
///
/// On little-endian targets the T-slice byte layout equals the xvec wire
/// format, so each row is one raw-byte slice handed to `write_record`. On
/// big-endian targets we fall back to per-element byteswap.
fn write_rows_primitive<T: Copy>(
    raw: &[T],
    base_offset: usize,
    row_count: usize,
    dim: usize,
    row_bytes: usize,
    ordinal_start: i64,
    sink: &mut dyn VecSink,
) {
    let elem_size = std::mem::size_of::<T>();
    debug_assert_eq!(row_bytes, dim * elem_size);

    #[cfg(target_endian = "little")]
    {
        // SAFETY: T is a plain-old-data numeric primitive (f32/f64/i*/u*/f16),
        // so the backing buffer of `raw` is a contiguous allocation of
        // row_count * dim * size_of::<T>() bytes starting at
        // `raw.as_ptr().add(base_offset)`. We are only reading those bytes
        // — no writes, no aliasing, no lifetime extension — and T has no
        // padding (`size_of::<T>()` exactly equals its storage).
        let byte_base = unsafe {
            std::slice::from_raw_parts(
                raw.as_ptr().add(base_offset) as *const u8,
                row_count * row_bytes,
            )
        };
        // Use the batch write path on the sink. For xvec outputs this
        // amortizes the per-record virtual dispatch across the whole batch
        // and lets the writer keep its buffered state warm.
        sink.write_records_fixed_dim(ordinal_start, byte_base, row_count, row_bytes);
    }

    #[cfg(target_endian = "big")]
    {
        // Big-endian fallback: per-element byteswap. Slow but correct.
        // T is a POD with no padding, so we reinterpret via bytemuck-style
        // copy + byteswap. Avoided adding a bytemuck dep by copying into
        // a small stack buffer per row.
        let mut row_buf = vec![0u8; row_bytes];
        let mut ordinal = ordinal_start;
        for row in 0..row_count {
            for col in 0..dim {
                let src_idx = base_offset + row * dim + col;
                let dst_off = col * elem_size;
                // SAFETY: reading `elem_size` bytes from one T element.
                let t_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        (&raw[src_idx] as *const T) as *const u8,
                        elem_size,
                    )
                };
                // Reverse into row_buf (primitives are natively BE on BE target).
                for b in 0..elem_size {
                    row_buf[dst_off + b] = t_bytes[elem_size - 1 - b];
                }
            }
            sink.write_record(ordinal, &row_buf);
            ordinal += 1;
        }
    }
}

// Satisfy the unused `Arc` import on some feature configurations.
#[allow(dead_code)]
fn _unused_arc(_a: Arc<()>) {}

// ────────────────────────────────────────────────────────────────────────
// Orchestration: parquet-file(s) → xvec output
// ────────────────────────────────────────────────────────────────────────

/// Progress callback invoked at completion milestones.
///
/// Arguments: `(units_this_tick, cumulative_units)`. The unit is decided
/// by the caller (in the parallel pwrite path it's "shards processed");
/// the orchestrator just owns the counter and the calling cadence.
///
/// `Sync` is required because the parallel paths invoke the callback
/// from a dedicated ticker thread.
pub type BatchProgress<'a> = &'a (dyn Fn(usize, u64) + Sync);

/// Rich per-tick progress info for the parallel parquet→xvec extract
/// orchestrator. Includes everything needed to compute an accurate
/// average rate and ETA on a *resumed* run, where some shards complete
/// near-instantly via the cache-skip path and would otherwise pollute
/// a naïve "cumulative records / total elapsed" calculation.
///
/// Use `decoded_records / decode_elapsed_secs` for the
/// actually-meaningful "decode rate" — that's the one that predicts
/// remaining work. The shard counters are still useful for the
/// progress bar's fill (so users see honest progress including the
/// cached prefix), but the *rate* shouldn't be derived from them.
#[derive(Debug, Clone, Copy)]
pub struct ExtractProgressTick {
    /// Shards completed since the last tick (delta).
    pub shards_delta: usize,
    /// Total shards completed so far (cached + decoded).
    pub shards_done: u64,
    /// Total shards in the job.
    pub shards_total: u64,
    /// Records corresponding to all completed shards (cached + decoded).
    pub records_done: u64,
    /// Total records in the job (sum of per-file row counts).
    pub records_total: u64,
    /// Records decoded from parquet on this run only — excludes
    /// records contributed by shards that were already cached and
    /// skipped at startup. This is the numerator of the honest rate.
    pub decoded_records: u64,
    /// Wall-clock seconds since the *first* actually-decoded shard
    /// began. `0.0` if no decode has started yet (every completed
    /// shard so far was cached). This is the denominator of the
    /// honest rate.
    pub decode_elapsed_secs: f64,
}

/// Rich progress callback for the parquet→xvec extract path.
///
/// Invoked from a dedicated ticker thread, so `Sync` is required.
pub type ExtractProgress<'a> = &'a (dyn Fn(&ExtractProgressTick) + Sync);

/// Free-form logging callback that worker threads call at meaningful
/// transitions (shard start/finish/skip, phase boundaries). Lets the
/// caller route messages into the pipeline UI's log pane.
///
/// `Sync` is required because every worker thread invokes the callback
/// independently.
pub type LogCallback<'a> = &'a (dyn Fn(&str) + Sync);

/// Read a parquet file (or directory of parquet files), extract a vector
/// column, and write it to `output` as an xvec file. Single-threaded.
///
/// Prefer [`extract_parquet_to_xvec_threaded`] for directories with many
/// files — this variant is kept as a simple entry point and for tests.
pub fn extract_parquet_to_xvec(
    source: &Path,
    output: &Path,
    target_format: VecFormat,
    column_hint: Option<&str>,
) -> Result<u64, String> {
    extract_parquet_to_xvec_threaded(
        source, output, target_format, column_hint, None, None, 1,
    )
}

/// Read a parquet file (or directory of parquet files), extract a vector
/// column, and write it to `output` as an xvec file.
///
/// This is the entry point that `transform convert` dispatches to when the
/// source is parquet and the target is a uniform xvec format. Per-batch work
/// hoists all Arrow downcasts out of the per-row loop; per-row work is one
/// bounds-checked slice and one write.
///
/// - `source` — single `.parquet` file, or a directory containing them.
/// - `output` — destination xvec file. Parent directory is created.
/// - `target_format` — the xvec variant to write. Must match the element
///   type inferred from the parquet column (no element conversion here; use
///   the generic convert path if you need conversion).
/// - `column_hint` — optional column name in the parquet schema. If `None`,
///   picks the first list-of-numeric column.
/// - `progress` — optional per-batch callback for UI updates.
/// - `threads` — `≤ 1` is sequential; `≥ 2` spawns that many loader threads
///   that each claim the next file from a shared atomic counter.
///
/// ## Ordinal stability
///
/// The output stream is byte-for-byte independent of `threads`. Loader
/// threads may finish files out of order, but the consumer reassembles in
/// strict file-sorted order via a reorder buffer, so ordinals count up as
/// `file_0_rows → file_1_rows → file_2_rows → …` for any thread count.
/// This is the property that makes the parallelism safe for downstream
/// pipeline steps that assume row indices are stable (shuffle, partitioning,
/// ground-truth lookup, …).
///
/// Returns the total number of rows written on success.
pub fn extract_parquet_to_xvec_threaded(
    source: &Path,
    output: &Path,
    target_format: VecFormat,
    column_hint: Option<&str>,
    progress: Option<ExtractProgress<'_>>,
    log: Option<LogCallback<'_>>,
    threads: usize,
) -> Result<u64, String> {
    let files = collect_parquet_files(source)?;
    if files.is_empty() {
        return Err(format!("no parquet files under {}", source.display()));
    }

    // Peek the first file's schema to compile the extractor.
    let first_file = File::open(&files[0])
        .map_err(|e| format!("open {}: {}", files[0].display(), e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(first_file)
        .map_err(|e| format!("parquet open {}: {}", files[0].display(), e))?;
    let schema = builder.schema().clone();
    let mut extractor = CompiledVectorExtractor::compile(&schema, column_hint)?;

    // Enforce that the target format matches what the parquet column
    // natively contains — the fast path is identity-conversion only.
    let natural = extractor.preferred_xvec_format();
    if natural != target_format {
        return Err(format!(
            "parquet element type {:?} produces {}; target is {} — use the generic convert \
             path for element conversion",
            extractor.element(),
            natural.name(),
            target_format.name(),
        ));
    }

    // For variable-length lists, `dimension` is not yet known — we need to
    // open the first file's first batch to discover it before opening the
    // sink (which requires a fixed dim for uniform xvec output).
    if extractor.list_kind() == ListKind::Variable {
        let reader = builder.build().map_err(|e| format!("parquet build: {}", e))?;
        let first_batch = reader
            .into_iter()
            .next()
            .ok_or("parquet file has no record batches")?
            .map_err(|e| format!("parquet first batch: {}", e))?;
        // drain_batch on a Variable list populates `dimension` from the
        // first row. Use a discard sink so we don't double-write.
        let mut discard = DiscardSink::new();
        extractor.drain_batch(&first_batch, 0, &mut discard)?;
        // We need to re-read the file from the start below, so reopen.
    }

    let dim = extractor.dimension();
    if dim == 0 {
        return Err("could not determine vector dimension from parquet source".into());
    }

    // Create output directory if needed.
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create dir {}: {}", parent.display(), e))?;
        }
    }

    // Single-thread path keeps the existing veks-io sink for simplicity.
    if threads <= 1 {
        let io_format =
            veks_io::VecFormat::from_extension(target_format.name()).ok_or_else(|| {
                format!("target format {} is not writable by veks-io", target_format.name())
            })?;
        let mut sink = veks_io::create_format(output, io_format, dim)?;
        // Single-threaded sink path doesn't surface the rich tick info
        // (it's only used for tests / threads<=1); drop the progress
        // callback rather than mapping types.
        let _ = progress;
        let cumulative = drain_sequential(&files, &mut extractor, sink.as_mut(), None)?;
        sink.finish()?;
        return Ok(cumulative);
    }

    // Per-file row counts (from parquet metadata only — no decode).
    let mut per_file_rows = Vec::with_capacity(files.len());
    for f in &files {
        let file = File::open(f).map_err(|e| format!("open {}: {}", f.display(), e))?;
        let pq = SerializedFileReader::new(file)
            .map_err(|e| format!("parquet metadata {}: {}", f.display(), e))?;
        per_file_rows.push(pq.metadata().file_metadata().num_rows() as u64);
    }
    let total_rows: u64 = per_file_rows.iter().sum();
    let element_size = extractor.element().element_size();
    let stride = 4 + dim as usize * element_size;
    let total_bytes = total_rows * stride as u64;
    let output_ext = target_format.preferred_extension();

    // Where to keep per-source intermediate xvec shards. One file per
    // input parquet, congruent with the source name (file_stem + the
    // output extension). Lives next to the final output so it shares the
    // same cache directory.
    let shards_dir = shards_dir_for(output);
    let output_tmp = with_tmp_suffix(output);

    // Idempotent fast-exit: with phase 2 writing to `<output>.tmp` and
    // atomically renaming into place, the existence of `output` at
    // exactly `total_bytes` is sufficient proof that an earlier run
    // completed successfully — the rename is the success boundary. The
    // shards dir is always around afterwards (we keep it as the
    // resumable intermediate for stage-1 → stage-2), so we deliberately
    // do NOT consult its presence here.
    if let Ok(meta) = fs::metadata(output) {
        if meta.len() == total_bytes {
            if let Some(cb) = progress {
                cb(&ExtractProgressTick {
                    shards_delta: files.len(),
                    shards_done: files.len() as u64,
                    shards_total: files.len() as u64,
                    records_done: total_rows,
                    records_total: total_rows,
                    decoded_records: 0,
                    decode_elapsed_secs: 0.0,
                });
            }
            return Ok(total_rows);
        }
    }

    // No trustworthy output. Wipe any prior in-progress artifacts so the
    // phases below start from a known-clean slate (besides resumable
    // shards, which are size-checked individually).
    if output.exists() {
        fs::remove_file(output).map_err(|e| {
            format!("remove stale {}: {}", output.display(), e)
        })?;
    }
    if output_tmp.exists() {
        let _ = fs::remove_file(&output_tmp);
    }

    fs::create_dir_all(&shards_dir)
        .map_err(|e| format!("create shards dir {}: {}", shards_dir.display(), e))?;

    // Phase 1: each worker decodes one parquet file and writes its
    // xvec output to an intermediate shard. Skip-if-correct-size makes
    // the phase resumable file by file.
    if let Some(cb) = log {
        cb(&format!(
            "phase 1: decoding {} parquet shards into {} ({} threads, {} in-flight)",
            files.len(),
            shards_dir.display(),
            threads.min(MAX_LOADER_THREADS).min(files.len()).max(1),
            MAX_IN_FLIGHT.min(files.len()),
        ));
    }
    write_parquet_shards(
        &files,
        &per_file_rows,
        &shards_dir,
        output_ext,
        &extractor,
        stride,
        progress,
        log,
        threads,
    )?;
    if let Some(cb) = log {
        cb(&format!(
            "phase 2: concatenating {} shards ({} bytes) into {}",
            files.len(),
            total_bytes,
            output.display(),
        ));
    }

    // Phase 2: stream every shard into a `<output>.tmp` file via
    // copy_file_range, in source-file order, at the pre-computed
    // per-shard byte offsets. Shards are deleted after a successful
    // copy to reclaim disk space.
    concat_shards_into_output(
        &files,
        &per_file_rows,
        &shards_dir,
        output_ext,
        &output_tmp,
        stride,
        total_bytes,
    )?;

    // Atomic publish: rename <output>.tmp → <output>. Until this rename
    // succeeds, the final output path does not exist — an observer can
    // never see a partially-written output. The mere existence of
    // `output` after this call is the success contract.
    fs::rename(&output_tmp, output).map_err(|e| {
        format!("rename {} → {}: {}", output_tmp.display(), output.display(), e)
    })?;

    Ok(total_rows)
}

/// Compute the directory where per-source intermediate shards live for
/// a given final output path: `<output_filename>.shards/` in the same
/// parent directory. So `${cache}/all_vectors.fvecs` →
/// `${cache}/all_vectors.fvecs.shards/`.
pub fn shards_dir_for(output: &Path) -> PathBuf {
    let parent = output.parent().unwrap_or(Path::new(""));
    let filename = output.file_name().and_then(|s| s.to_str()).unwrap_or("output");
    parent.join(format!("{}.shards", filename))
}


/// Compute the path of the intermediate shard for a given source
/// parquet file. Shard name is the source's file stem plus the output
/// extension — congruent with the source so one can map between them
/// at a glance.
fn shard_path_for(source: &Path, shards_dir: &Path, output_ext: &str) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("shard");
    shards_dir.join(format!("{}.{}", stem, output_ext))
}

/// Sequential drain: open each file in sorted order, iterate its record
/// batches, drain each batch into the sink.
fn drain_sequential(
    files: &[PathBuf],
    extractor: &mut CompiledVectorExtractor,
    sink: &mut dyn VecSink,
    progress: Option<BatchProgress<'_>>,
) -> Result<u64, String> {
    let mut ordinal: i64 = 0;
    let mut cumulative: u64 = 0;
    for file_path in files {
        let file = File::open(file_path)
            .map_err(|e| format!("open {}: {}", file_path.display(), e))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| format!("parquet open {}: {}", file_path.display(), e))?;
        let reader = builder
            .build()
            .map_err(|e| format!("parquet build {}: {}", file_path.display(), e))?;
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| format!("parquet batch {}: {}", file_path.display(), e))?;
            let rows = extractor.drain_batch(&batch, ordinal, sink)?;
            ordinal += rows as i64;
            cumulative += rows as u64;
            if let Some(cb) = progress {
                cb(rows, cumulative);
            }
        }
    }
    Ok(cumulative)
}

/// Phase 1 of the shard-then-concat design: write each input parquet's
/// xvec output to a per-source intermediate file in `shards_dir`. Workers
/// run in parallel, each handling one parquet file end-to-end (read,
/// decode, write its shard). The phase is **resumable**: if a shard file
/// already exists with the right size (`per_file_rows[i] * stride`), the
/// worker skips it and counts those rows toward progress.
///
/// Shard files are full xvec format on their own — concatenating them in
/// source order yields a valid xvec file (each record carries its own
/// `[dim:i32 LE]` prefix), which is what phase 2 does.
///
/// Same in-flight memory bound applies; same ordinal-congruence guarantee
/// applies (a shard's bytes go to the same logical position they would in
/// a single-file pwrite).
fn write_parquet_shards(
    files: &[PathBuf],
    per_file_rows: &[u64],
    shards_dir: &Path,
    output_ext: &str,
    extractor: &CompiledVectorExtractor,
    stride: usize,
    progress: Option<ExtractProgress<'_>>,
    log: Option<LogCallback<'_>>,
    threads: usize,
) -> Result<(), String> {
    let total_files = files.len();
    let threads = threads.min(MAX_LOADER_THREADS).min(total_files).max(1);
    let max_in_flight = MAX_IN_FLIGHT.min(total_files);

    let dim = extractor.dimension();
    // Pre-computed dim header for the records.
    let dim_prefix: [u8; 4] = (dim as i32).to_le_bytes();

    let total_records: u64 = per_file_rows.iter().sum();

    let next_idx = Arc::new(AtomicUsize::new(0));
    let next_to_prefetch = Arc::new(AtomicUsize::new(0));
    // Count of completed shards (decoded + written). Drives backpressure
    // for the prefetch pool — prefetchers stall if they get more than
    // MAX_LOOKAHEAD ahead of the most-recently-completed shard.
    let writer_advance = Arc::new(AtomicUsize::new(0));
    let files_shared: Arc<Vec<PathBuf>> = Arc::new(files.to_vec());
    let rows_shared: Arc<Vec<u64>> = Arc::new(per_file_rows.to_vec());
    let shards_dir_shared: Arc<PathBuf> = Arc::new(shards_dir.to_path_buf());
    let output_ext_shared: Arc<String> = Arc::new(output_ext.to_string());
    let in_flight = Arc::new((Mutex::new(0usize), Condvar::new()));
    // Counters tracked separately so we can compute an honest rate on
    // a resumed run. `shards_done` and `records_done` include the
    // cache-skip path (so the bar fills accurately); `decoded_records`
    // counts only actually-decoded records (the rate numerator), and
    // `decode_start_offset_nanos` is the wall-clock offset (from
    // `phase_start`) at which the *first* real decode began (the rate
    // denominator anchor). Initial sentinel `u64::MAX` means "no real
    // decode has started yet."
    let shards_done = Arc::new(AtomicU64::new(0));
    let records_done = Arc::new(AtomicU64::new(0));
    let decoded_records = Arc::new(AtomicU64::new(0));
    let decode_start_offset_nanos = Arc::new(AtomicU64::new(u64::MAX));
    let phase_start = std::time::Instant::now();
    let progress_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let prefetch_threads = PREFETCH_THREADS.min(total_files.saturating_sub(1).max(1));

    // Stop signal is split: a lock-free `AtomicBool` for the per-iteration
    // hot-path check (workers consult it without contention), and a
    // `Mutex<Option<String>>` for the actual error message (only touched
    // when an error fires). Without this split, every worker iteration
    // contended on the same mutex just to read `is_some()`, which can
    // serialize 32+ workers and explode the wall-clock time.
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let first_err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    thread::scope(|scope| {
        // Ticker thread: translates atomic counters → rich progress
        // callback at PROGRESS_TICK cadence. Decouples UI from worker
        // hot path. Computes `decode_elapsed_secs` from the wall-clock
        // offset captured by the *first* worker that actually decoded
        // — so a resumed run's rate denominator excludes the time
        // spent skipping cached shards at startup.
        let ticker_shards = Arc::clone(&shards_done);
        let ticker_records = Arc::clone(&records_done);
        let ticker_decoded = Arc::clone(&decoded_records);
        let ticker_decode_start = Arc::clone(&decode_start_offset_nanos);
        let ticker_done = Arc::clone(&progress_done);
        let ticker_progress = progress;
        let total_files_u64 = total_files as u64;
        let ticker_handle = scope.spawn(move || {
            let emit = |last_seen: &mut u64| {
                let s = ticker_shards.load(Ordering::Relaxed);
                if s <= *last_seen { return; }
                let delta = (s - *last_seen) as usize;
                *last_seen = s;
                let r = ticker_records.load(Ordering::Relaxed);
                let d = ticker_decoded.load(Ordering::Relaxed);
                let decode_elapsed_secs = match ticker_decode_start.load(Ordering::Relaxed) {
                    u64::MAX => 0.0,
                    start_off => {
                        let now_off = phase_start.elapsed().as_nanos() as u64;
                        now_off.saturating_sub(start_off) as f64 / 1_000_000_000.0
                    }
                };
                if let Some(cb) = ticker_progress {
                    cb(&ExtractProgressTick {
                        shards_delta: delta,
                        shards_done: s,
                        shards_total: total_files_u64,
                        records_done: r,
                        records_total: total_records,
                        decoded_records: d,
                        decode_elapsed_secs,
                    });
                }
            };
            let mut last_seen: u64 = 0;
            loop {
                emit(&mut last_seen);
                if ticker_done.load(Ordering::Relaxed) {
                    emit(&mut last_seen);
                    break;
                }
                thread::sleep(PROGRESS_TICK);
            }
        });

        // Prefetch pool: dedicated read drivers that pull upcoming
        // parquets into the page cache via real read() syscalls so
        // decoder workers' `read_to_end` hits warm pages and is
        // CPU-bound (decode), not EBS-read-bound. Backpressured via
        // `writer_advance` so cached working set stays bounded.
        let mut prefetch_handles = Vec::with_capacity(prefetch_threads);
        for _ in 0..prefetch_threads {
            let next_to_prefetch = Arc::clone(&next_to_prefetch);
            let writer_advance = Arc::clone(&writer_advance);
            let files_shared = Arc::clone(&files_shared);
            let rows_shared = Arc::clone(&rows_shared);
            let shards_dir_shared = Arc::clone(&shards_dir_shared);
            let output_ext_shared = Arc::clone(&output_ext_shared);
            let stop_flag = Arc::clone(&stop_flag);
            prefetch_handles.push(scope.spawn(move || {
                parquet_prefetch_loop(
                    next_to_prefetch,
                    writer_advance,
                    files_shared,
                    rows_shared,
                    shards_dir_shared,
                    output_ext_shared,
                    stop_flag,
                    stride,
                    total_files,
                );
            }));
        }

        // Decoder pool: each worker owns its own scratch buffer + extractor
        // clone, picks files atomically, writes a per-source shard file.
        // Source pages are pre-warmed by the prefetch pool above.
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let next_idx = Arc::clone(&next_idx);
            let files_shared = Arc::clone(&files_shared);
            let rows_shared = Arc::clone(&rows_shared);
            let shards_dir_shared = Arc::clone(&shards_dir_shared);
            let output_ext_shared = Arc::clone(&output_ext_shared);
            let in_flight = Arc::clone(&in_flight);
            let shards_done_w = Arc::clone(&shards_done);
            let records_done_w = Arc::clone(&records_done);
            let decoded_records_w = Arc::clone(&decoded_records);
            let decode_start_w = Arc::clone(&decode_start_offset_nanos);
            let writer_advance = Arc::clone(&writer_advance);
            let stop_flag = Arc::clone(&stop_flag);
            let first_err = Arc::clone(&first_err);
            let local_extractor = extractor.clone();

            handles.push(scope.spawn(move || {
                shard_worker_loop(
                    next_idx,
                    files_shared,
                    rows_shared,
                    shards_dir_shared,
                    output_ext_shared,
                    in_flight,
                    shards_done_w,
                    records_done_w,
                    decoded_records_w,
                    decode_start_w,
                    phase_start,
                    writer_advance,
                    stop_flag,
                    first_err,
                    local_extractor,
                    dim_prefix,
                    stride,
                    max_in_flight,
                    total_files,
                    log,
                );
            }));
        }

        // Wait for all decoder workers.
        for h in handles {
            let _ = h.join();
        }
        // Decoders are done; signal stop so prefetchers exit even if
        // they're parked on backpressure waiting for work that's no
        // longer needed.
        stop_flag.store(true, Ordering::Relaxed);
        for h in prefetch_handles { let _ = h.join(); }

        // Tell the ticker we're done; let it flush a final tick.
        progress_done.store(true, Ordering::Relaxed);
        let _ = ticker_handle.join();
    });

    if let Some(e) = first_err.lock().unwrap().take() {
        return Err(e);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn shard_worker_loop(
    next_idx: Arc<AtomicUsize>,
    files: Arc<Vec<PathBuf>>,
    rows: Arc<Vec<u64>>,
    shards_dir: Arc<PathBuf>,
    output_ext: Arc<String>,
    in_flight: Arc<(Mutex<usize>, Condvar)>,
    shards_done: Arc<AtomicU64>,
    records_done: Arc<AtomicU64>,
    decoded_records: Arc<AtomicU64>,
    decode_start_offset_nanos: Arc<AtomicU64>,
    phase_start: std::time::Instant,
    writer_advance: Arc<AtomicUsize>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    first_err: Arc<Mutex<Option<String>>>,
    mut extractor: CompiledVectorExtractor,
    dim_prefix: [u8; 4],
    stride: usize,
    max_in_flight: usize,
    total_files: usize,
    log: Option<LogCallback<'_>>,
) {
    let mut scratch: Vec<u8> = Vec::new();

    loop {
        // Lock-free hot-path check — avoids the per-iteration mutex
        // contention that was throttling worker throughput at scale.
        if stop_flag.load(Ordering::Relaxed) {
            let (_, cvar) = &*in_flight;
            cvar.notify_all();
            break;
        }

        let idx = next_idx.fetch_add(1, Ordering::Relaxed);
        if idx >= total_files { break; }

        let path = &files[idx];
        let expected_rows = rows[idx];
        let expected_shard_size = expected_rows * stride as u64;
        let shard_path = shard_path_for(path, &shards_dir, &output_ext);

        // Resumability: skip the entire file if a shard at this path
        // already exists with the exact expected size (rows × stride).
        // The shard's content is fully determined by the source's
        // record contents and our pure-data extraction — same source
        // file → same shard bytes — so size congruence is sufficient.
        if let Ok(meta) = fs::metadata(&shard_path) {
            if meta.len() == expected_shard_size {
                shards_done.fetch_add(1, Ordering::Relaxed);
                records_done.fetch_add(expected_rows, Ordering::Relaxed);
                writer_advance.fetch_max(idx + 1, Ordering::Relaxed);
                if let Some(cb) = log {
                    cb(&format!(
                        "[shard {}/{}] skip {} ({} bytes, already cached)",
                        idx + 1, total_files,
                        path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                        expected_shard_size,
                    ));
                }
                continue;
            }
        }

        // Acquire in-flight slot only when we actually have work to do
        // (skipped shards don't consume a slot — they're zero-cost).
        {
            let (lock, cvar) = &*in_flight;
            let mut count = lock.lock().unwrap();
            while *count >= max_in_flight {
                if stop_flag.load(Ordering::Relaxed) { return; }
                count = cvar.wait(count).unwrap();
            }
            *count += 1;
        }

        let shard_start = std::time::Instant::now();
        // Capture the wall-clock offset of the *first* actual decode so
        // the progress ticker can compute an honest decode rate that
        // excludes the (near-zero) time spent skipping cached shards
        // earlier in this run. CAS from sentinel u64::MAX → only the
        // very first decoder wins; everyone else is a no-op.
        let now_off = phase_start.elapsed().as_nanos() as u64;
        let _ = decode_start_offset_nanos.compare_exchange(
            u64::MAX, now_off, Ordering::Relaxed, Ordering::Relaxed,
        );
        if let Some(cb) = log {
            cb(&format!(
                "[shard {}/{}] reading {}",
                idx + 1, total_files,
                path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            ));
        }
        let result: Result<u64, String> = (|| {
            // Open the source. The prefetch pool has already pulled
            // this file into the page cache and issued the SEQUENTIAL
            // + WILLNEED hints on its own fd; re-issuing them here
            // would just kick off another round of readahead on
            // already-cached pages and add to EBS read concurrency
            // for no benefit. We do still want to be able to issue
            // DONTNEED on this fd once we're done with the data, so
            // the open + the read stays — but the fadvise calls don't.
            let src = File::open(path)
                .map_err(|e| format!("open {}: {}", path.display(), e))?;
            let src_len = src.metadata()
                .map_err(|e| format!("stat {}: {}", path.display(), e))?
                .len();
            let mut raw = Vec::with_capacity(src_len as usize);
            (&src).read_to_end(&mut raw)
                .map_err(|e| format!("read {}: {}", path.display(), e))?;

            let parquet_bytes = bytes::Bytes::from(raw);
            let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes)
                .map_err(|e| format!("parquet open {}: {}", path.display(), e))?;
            let reader = builder.build()
                .map_err(|e| format!("parquet build {}: {}", path.display(), e))?;
            let mut file_rows: u64 = 0;

            // Build the entire shard in `scratch` so we can write it
            // out as chunked sequential pwrites with sync_file_range —
            // the EBS-friendly write pattern.
            scratch.clear();
            for batch_result in reader {
                let batch = batch_result
                    .map_err(|e| format!("parquet batch {}: {}", path.display(), e))?;
                let row_count = batch.num_rows();
                if row_count == 0 { continue; }
                append_batch_to_buf(&mut extractor, &batch, &dim_prefix, &mut scratch)?;
                file_rows += row_count as u64;
            }

            if file_rows != expected_rows {
                return Err(format!(
                    "row-count mismatch for {}: parquet metadata claims {}, decoded {}",
                    path.display(), expected_rows, file_rows,
                ));
            }
            debug_assert_eq!(scratch.len() as u64, expected_shard_size);

            // Tell the kernel we won't re-read the source — frees the
            // page cache for the next prefetched parquet.
            advise_dontneed(&src, 0, src_len);
            drop(src);

            // Write atomically via a `.tmp` sibling + rename, so a
            // crash mid-write never leaves a wrong-sized shard that
            // a re-run would mistake for valid via the size check.
            // The write itself is chunked into WRITE_CHUNK_BYTES
            // pieces, each kicked into writeback immediately via
            // sync_file_range(WRITE) so the EBS write queue is
            // continuously fed instead of bursting at the end.
            let tmp_path = with_tmp_suffix(&shard_path);
            {
                use std::os::unix::fs::FileExt;
                let shard_file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&tmp_path)
                    .map_err(|e| format!("create {}: {}", tmp_path.display(), e))?;
                shard_file.set_len(scratch.len() as u64)
                    .map_err(|e| format!("set_len {}: {}", tmp_path.display(), e))?;
                let mut written: u64 = 0;
                let total = scratch.len() as u64;
                while written < total {
                    let chunk = WRITE_CHUNK_BYTES.min(total - written);
                    let slice = &scratch[written as usize..(written + chunk) as usize];
                    shard_file.write_all_at(slice, written)
                        .map_err(|e| format!(
                            "pwrite {} bytes @ {}: {}", chunk, written, e,
                        ))?;
                    kick_writeback(&shard_file, written, chunk);
                    written += chunk;
                }
                // Crash-safety boundary: wait until every byte we wrote
                // is durably on disk before the rename publishes the
                // shard at its final name. Without this, a power-loss
                // window exists where the rename is journal-committed
                // but data writeback hasn't reached EBS yet — leaving
                // a final-path file at correct size with stale blocks
                // that the resumability size-check would mistake for
                // completed work.
                shard_file.sync_data().map_err(|e| {
                    format!("fdatasync {}: {}", tmp_path.display(), e)
                })?;
                advise_dontneed(&shard_file, 0, total);
            }
            fs::rename(&tmp_path, &shard_path).map_err(|e| {
                format!("rename {} → {}: {}", tmp_path.display(), shard_path.display(), e)
            })?;

            shards_done.fetch_add(1, Ordering::Relaxed);
            records_done.fetch_add(expected_rows, Ordering::Relaxed);
            decoded_records.fetch_add(expected_rows, Ordering::Relaxed);
            writer_advance.fetch_max(idx + 1, Ordering::Relaxed);
            if let Some(cb) = log {
                let elapsed = shard_start.elapsed();
                let secs = elapsed.as_secs_f64().max(0.001);
                let mb = expected_shard_size as f64 / (1024.0 * 1024.0);
                cb(&format!(
                    "[shard {}/{}] done {} ({:.1} MiB in {:.2}s, {:.1} MiB/s)",
                    idx + 1, total_files,
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                    mb, secs, mb / secs,
                ));
            }
            Ok(file_rows)
        })();

        // Release the slot regardless of success/failure.
        {
            let (lock, cvar) = &*in_flight;
            let mut count = lock.lock().unwrap();
            *count = count.saturating_sub(1);
            cvar.notify_one();
        }

        if let Err(e) = result {
            // Stash the message and flip the lock-free stop flag so other
            // workers exit on their next iteration without going through
            // the mutex.
            let mut slot = first_err.lock().unwrap();
            if slot.is_none() { *slot = Some(e); }
            stop_flag.store(true, Ordering::Relaxed);
            let (_, cvar) = &*in_flight;
            cvar.notify_all();
            break;
        }
    }
}

/// Prefetch worker for the parquet → xvec extract path. Runs ahead
/// of the decoder workers and pulls upcoming source parquets into the
/// page cache via real `read()`s (advisory `WILLNEED` alone doesn't
/// reliably populate large files). Backpressures off `writer_advance`
/// so the cached working set stays bounded to ~`MAX_LOOKAHEAD` files.
///
/// Skips files whose target shard already exists at the right size —
/// no point pulling source pages we won't decode.
fn parquet_prefetch_loop(
    next_to_prefetch: Arc<AtomicUsize>,
    writer_advance: Arc<AtomicUsize>,
    files: Arc<Vec<PathBuf>>,
    rows: Arc<Vec<u64>>,
    shards_dir: Arc<PathBuf>,
    output_ext: Arc<String>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    stride: usize,
    total_files: usize,
) {
    let mut buf = vec![0u8; PREFETCH_CHUNK];
    loop {
        if stop_flag.load(Ordering::Relaxed) { break; }
        let idx = next_to_prefetch.fetch_add(1, Ordering::Relaxed);
        if idx >= total_files { break; }

        // Backpressure: don't run more than MAX_LOOKAHEAD ahead of the
        // most-recently-completed shard. Sleep-poll is fine — events
        // are large (one shard takes hundreds of ms to decode + write),
        // so 5 ms granularity is invisible.
        while !stop_flag.load(Ordering::Relaxed) {
            let advanced = writer_advance.load(Ordering::Relaxed);
            if idx < advanced + MAX_LOOKAHEAD { break; }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        if stop_flag.load(Ordering::Relaxed) { break; }

        let path = &files[idx];
        let expected_shard_size = rows[idx] * stride as u64;
        let shard_path = shard_path_for(path, &shards_dir, &output_ext);

        // Don't prefetch if the shard is already cached — the decoder
        // will skip it anyway.
        if let Ok(meta) = fs::metadata(&shard_path) {
            if meta.len() == expected_shard_size { continue; }
        }

        // Force the file's pages into the page cache. Errors here are
        // non-fatal — the decoder will hit a cache miss and pay the
        // EBS read cost, but the whole job still completes.
        if let Ok(f) = File::open(path) {
            unsafe {
                libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
                libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED);
            }
            let mut reader = f;
            loop {
                match std::io::Read::read(&mut reader, &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => continue,
                }
            }
        }
    }
}

/// Trigger asynchronous writeback for a range of `file` without
/// waiting. The kernel marks the range for immediate writeback
/// instead of waiting on its periodic timer or the
/// `dirty_background_bytes` threshold — keeps the EBS write queue
/// continuously fed when chained over a large sequential write.
///
/// Errors are non-fatal; the syscall is advisory.
pub(super) fn kick_writeback(file: &File, offset: u64, len: u64) {
    if len == 0 { return; }
    unsafe {
        libc::sync_file_range(
            file.as_raw_fd(),
            offset as i64,
            len as i64,
            libc::SYNC_FILE_RANGE_WRITE,
        );
    }
}

pub(super) fn with_tmp_suffix(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Phase 2: walk shards in source-file order and concatenate them into
/// the final output file via `copy_file_range` (kernel-side, no
/// user-space buffer). Pre-sizes the output with `set_len` so per-shard
/// copies land at deterministic offsets.
///
/// **Shards are kept on disk after concat — as a `.cache`-level
/// intermediate, not as part of the published dataset.** They live in
/// `.cache/<output>.shards/` and exist purely to make the extraction
/// resumable and per-stem verifiable: a re-run skips any
/// `<shards_dir>/<stem>.<ext>` already at exactly `rows × stride`
/// bytes, and the consolidated output can be rebuilt from them at any
/// time without re-reading the source parquets. They are subject to
/// the same eventual cache cleanup as everything else under `.cache/`
/// — not promoted, not catalogued, not Merkle-signed. Disk cost is
/// roughly 2× the consolidated output size for as long as the cache
/// retains them; that's the price of resumability for an extraction
/// whose total cost dwarfs the shard disk space.
fn concat_shards_into_output(
    files: &[PathBuf],
    per_file_rows: &[u64],
    shards_dir: &Path,
    output_ext: &str,
    output: &Path,
    stride: usize,
    total_bytes: u64,
) -> Result<(), String> {
    let out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output)
        .map_err(|e| format!("create {}: {}", output.display(), e))?;
    out_file
        .set_len(total_bytes)
        .map_err(|e| format!("set_len {}: {}", output.display(), e))?;

    let mut output_offset: u64 = 0;
    for (idx, src) in files.iter().enumerate() {
        let shard_path = shard_path_for(src, shards_dir, output_ext);
        let expected_size = per_file_rows[idx] * stride as u64;
        let shard_meta = fs::metadata(&shard_path)
            .map_err(|e| format!("stat {}: {}", shard_path.display(), e))?;
        if shard_meta.len() != expected_size {
            return Err(format!(
                "shard {} has size {} but expected {} (rows {} × stride {})",
                shard_path.display(), shard_meta.len(), expected_size,
                per_file_rows[idx], stride,
            ));
        }
        let shard_file = File::open(&shard_path)
            .map_err(|e| format!("open {}: {}", shard_path.display(), e))?;
        copy_file_range_full(&shard_file, &out_file, output_offset, expected_size)
            .map_err(|e| format!(
                "copy {} → {} @ {}: {}",
                shard_path.display(), output.display(), output_offset, e,
            ))?;
        advise_dontneed(&out_file, output_offset, expected_size);
        output_offset += expected_size;
        // NB: shard intentionally NOT unlinked. See the function-level
        // doc for the resumability contract this preserves.
    }
    debug_assert_eq!(output_offset, total_bytes);
    Ok(())
}

/// Copy `len` bytes from `src` (starting at offset 0) to `dst` (starting
/// at `dst_offset`) using the `copy_file_range` syscall. The kernel
/// performs the copy inside the kernel — no user-space buffer, no extra
/// page-cache fill on the source side. May yield fewer bytes than
/// requested per call, so we loop until done.
pub(super) fn copy_file_range_full(
    src: &File,
    dst: &File,
    dst_offset: u64,
    len: u64,
) -> std::io::Result<()> {
    let mut src_off: i64 = 0;
    let mut dst_off: i64 = dst_offset as i64;
    let mut remaining: usize = len as usize;
    while remaining > 0 {
        let n = unsafe {
            libc::copy_file_range(
                src.as_raw_fd(),
                &mut src_off as *mut i64,
                dst.as_raw_fd(),
                &mut dst_off as *mut i64,
                remaining,
                0,
            )
        };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "copy_file_range returned 0 before completing copy",
            ));
        }
        remaining = remaining.saturating_sub(n as usize);
    }
    Ok(())
}

/// Hint the kernel that the just-written byte range of the output file is
/// safe to evict from the page cache. The kernel will let writeback
/// complete (the data is already in the dirty queue from the preceding
/// `pwrite`), then drop the page-cache entries — which prevents the
/// growing output file from monopolizing the page cache and starving
/// inbound parquet readahead.
///
/// This is the per-pwrite companion to host-level dirty-page tuning
/// (`vm.dirty_bytes` / `vm.dirty_background_bytes`): the sysctls cap how
/// much dirty data can stack up; this hint lets the kernel reclaim those
/// pages immediately once they're written out, rather than holding them
/// in case we re-read them (we won't).
///
/// Errors are non-fatal — the syscall is advisory; if it fails, the only
/// consequence is slightly less efficient cache management.
pub(super) fn advise_dontneed(file: &File, offset: u64, len: u64) {
    if len == 0 { return; }
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as i64,
            len as i64,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

/// Read a file in one shot with `posix_fadvise(SEQUENTIAL | WILLNEED)`
/// hints to the kernel before the read. The hints tell the kernel to
/// double its readahead window and start prefetching the whole file
/// immediately — meaningful on storage with high latency-to-bandwidth
/// ratios like EBS, where the default conservative readahead leaves
/// throughput on the table.
pub(super) fn read_file_with_advise(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    // POSIX_FADV_SEQUENTIAL: ask kernel to double readahead window.
    // POSIX_FADV_WILLNEED: ask kernel to start prefetching the whole
    // range into the page cache asynchronously. Errors are non-fatal —
    // the syscalls are advisory and the read below works either way.
    let fd = file.as_raw_fd();
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_WILLNEED);
    }
    let mut buf = Vec::with_capacity(len);
    let mut reader = file;
    reader.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Append `batch`'s interleaved xvec payload `[dim:i32 LE][record_bytes]`
/// for every row to `scratch`. Caller is responsible for clearing `scratch`
/// when starting a new write batch — this lets the parallel pwrite worker
/// accumulate every parquet batch of a file into one contiguous buffer
/// before flushing in a single large `pwrite`.
fn append_batch_to_buf(
    extractor: &mut CompiledVectorExtractor,
    batch: &RecordBatch,
    dim_prefix: &[u8; 4],
    scratch: &mut Vec<u8>,
) -> Result<usize, String> {
    let mut sink = BufSink { buf: scratch, dim_prefix: *dim_prefix };
    extractor.drain_batch(batch, 0, &mut sink)
}

/// `VecSink` that appends interleaved `[dim][record]` payloads to a
/// caller-owned `Vec<u8>`. Used by the parallel pwrite worker to stage one
/// batch's bytes before issuing a single kernel write.
struct BufSink<'a> {
    buf: &'a mut Vec<u8>,
    dim_prefix: [u8; 4],
}

impl VecSink for BufSink<'_> {
    fn write_record(&mut self, _ordinal: i64, data: &[u8]) {
        self.buf.extend_from_slice(&self.dim_prefix);
        self.buf.extend_from_slice(data);
    }

    fn write_records_fixed_dim(
        &mut self,
        _ordinal_start: i64,
        packed: &[u8],
        count: usize,
        record_size: usize,
    ) {
        debug_assert_eq!(packed.len(), count * record_size);
        let stride = 4 + record_size;
        let initial = self.buf.len();
        self.buf.resize(initial + count * stride, 0);
        let mut dst = initial;
        let mut src = 0;
        for _ in 0..count {
            self.buf[dst..dst + 4].copy_from_slice(&self.dim_prefix);
            self.buf[dst + 4..dst + stride]
                .copy_from_slice(&packed[src..src + record_size]);
            dst += stride;
            src += record_size;
        }
    }

    fn finish(self: Box<Self>) -> Result<(), String> { Ok(()) }
}


/// Probe a parquet source cheaply: returns the compiled extractor (which
/// carries element type, list kind, and dimension for FixedSizeList) plus
/// the total row count. Used by the convert command to decide whether the
/// fast path applies without opening the full reader.
pub struct ParquetVectorProbe {
    pub extractor: CompiledVectorExtractor,
    pub file_count: usize,
    pub row_count: u64,
    /// Row count per file, in the same sort order as `collect_parquet_files`.
    /// Used by the parallel pwrite path to compute per-file byte offsets so
    /// workers can write directly to disjoint regions of the output without
    /// a centralized reorder + writer thread.
    pub per_file_rows: Vec<u64>,
}

pub fn probe_parquet_vectors(
    source: &Path,
    column_hint: Option<&str>,
) -> Result<ParquetVectorProbe, String> {
    let files = collect_parquet_files(source)?;
    let first = File::open(&files[0])
        .map_err(|e| format!("open {}: {}", files[0].display(), e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(first)
        .map_err(|e| format!("parquet open {}: {}", files[0].display(), e))?;
    let schema = builder.schema().clone();
    let extractor = CompiledVectorExtractor::compile(&schema, column_hint)?;

    // Row count from parquet metadata — no batch decode needed.
    let mut per_file_rows = Vec::with_capacity(files.len());
    for f in &files {
        let file = File::open(f)
            .map_err(|e| format!("open {}: {}", f.display(), e))?;
        let pq = SerializedFileReader::new(file)
            .map_err(|e| format!("parquet metadata {}: {}", f.display(), e))?;
        per_file_rows.push(pq.metadata().file_metadata().num_rows() as u64);
    }
    let total: u64 = per_file_rows.iter().sum();

    Ok(ParquetVectorProbe {
        extractor,
        file_count: files.len(),
        row_count: total,
        per_file_rows,
    })
}

/// Collect `.parquet` files under `path` (or the file itself), sorted.
fn collect_parquet_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(path)
            .map_err(|e| format!("read dir {}: {}", path.display(), e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".parquet"))
            .map(|e| e.path())
            .collect();
        entries.sort_by(|a, b| natural_cmp(a, b));
        if entries.is_empty() {
            return Err(format!("no .parquet files under {}", path.display()));
        }
        Ok(entries)
    } else if path.is_file() {
        Ok(vec![path.to_path_buf()])
    } else {
        Err(format!("{} does not exist", path.display()))
    }
}

/// Natural-order path comparison: treats embedded digit runs as integers so
/// `1, 2, …, 9, 10, 11, …, 100` sorts numerically rather than as `1, 10,
/// 100, 11, 12, …`. Lexicographic order is wrong for the common case where
/// parquet shards are named without zero-padding (e.g., `0.parquet`,
/// `1.parquet`, …, `9999.parquet`), and the per-file offset arithmetic in
/// the parallel pwrite path depends on the ordering being stable AND
/// matching what a human reading the directory expects.
///
/// Comparison rule, applied to the file basenames as byte strings:
/// 1. Walk both names in lockstep.
/// 2. If both current bytes are ASCII digits, slurp the maximal digit run
///    on each side, parse to `u128`, and compare numerically. Tie → keep
///    walking; if numeric values are equal but textual lengths differ
///    (e.g., `1` vs `01`), the shorter run sorts first.
/// 3. Otherwise compare bytes directly.
///
/// Falls back to byte-wise lexicographic ordering on the parent path so
/// that nested directories are compared per-component naturally too.
fn natural_cmp(a: &Path, b: &Path) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let ap = a.parent().unwrap_or(Path::new(""));
    let bp = b.parent().unwrap_or(Path::new(""));
    match ap.cmp(bp) {
        Ordering::Equal => {}
        other => return other,
    }
    let an = a.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let bn = b.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    natural_str_cmp(&an, &bn)
}

fn natural_str_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let mut i = 0;
    let mut j = 0;
    while i < ab.len() && j < bb.len() {
        let ac = ab[i];
        let bc = bb[j];
        if ac.is_ascii_digit() && bc.is_ascii_digit() {
            // Slurp digit runs on both sides.
            let i_start = i;
            let j_start = j;
            while i < ab.len() && ab[i].is_ascii_digit() { i += 1; }
            while j < bb.len() && bb[j].is_ascii_digit() { j += 1; }
            let a_digits = &ab[i_start..i];
            let b_digits = &bb[j_start..j];
            // Numeric compare via u128 (parquet shard counts can be
            // large but are nowhere near 2^128).
            let av: u128 = std::str::from_utf8(a_digits).unwrap_or("0").parse().unwrap_or(0);
            let bv: u128 = std::str::from_utf8(b_digits).unwrap_or("0").parse().unwrap_or(0);
            match av.cmp(&bv) {
                Ordering::Equal => {
                    // Same value; shorter (no padding) sorts first to be
                    // deterministic when both `1` and `01` exist.
                    match a_digits.len().cmp(&b_digits.len()) {
                        Ordering::Equal => continue,
                        other => return other,
                    }
                }
                other => return other,
            }
        } else {
            match ac.cmp(&bc) {
                Ordering::Equal => { i += 1; j += 1; }
                other => return other,
            }
        }
    }
    // One ran out first → shorter sorts first.
    ab.len().cmp(&bb.len())
}

#[cfg(test)]
mod natural_cmp_tests {
    use super::natural_str_cmp;
    use std::cmp::Ordering;

    #[test]
    fn numeric_runs_compare_numerically() {
        // Without natural sort, "10" < "2" lexicographically.
        assert_eq!(natural_str_cmp("file2.parquet", "file10.parquet"), Ordering::Less);
        assert_eq!(natural_str_cmp("file10.parquet", "file2.parquet"), Ordering::Greater);
    }

    #[test]
    fn deeply_nested_numbers_sort_correctly() {
        let names = vec![
            "shard-1.parquet",
            "shard-2.parquet",
            "shard-9.parquet",
            "shard-10.parquet",
            "shard-100.parquet",
        ];
        let mut shuffled = names.clone();
        shuffled.sort();  // lexicographic — wrong order
        assert_ne!(shuffled, names, "sanity: lexicographic should differ");

        let mut natural = names.clone();
        natural.sort_by(|a, b| natural_str_cmp(a, b));
        assert_eq!(natural, names);
    }

    #[test]
    fn unpadded_and_padded_coexist_deterministically() {
        // When both `1.parquet` and `01.parquet` are present (unusual but
        // possible), shorter representation sorts first for stability.
        assert_eq!(natural_str_cmp("1.parquet", "01.parquet"), Ordering::Less);
    }

    #[test]
    fn pure_text_falls_back_to_lex() {
        assert_eq!(natural_str_cmp("apple", "banana"), Ordering::Less);
        assert_eq!(natural_str_cmp("apple", "apple"), Ordering::Equal);
    }

    #[test]
    fn unpadded_numeric_part_pattern() {
        // Reproduce a real-world unpadded sequential file naming: 0..7433 unpadded.
        let names: Vec<String> = (0..15).map(|i| format!("{}.parquet", i)).collect();
        let mut shuffled = names.clone();
        shuffled.sort();  // lexicographic puts 10 before 2
        assert_ne!(shuffled, names);

        let mut natural = names.clone();
        natural.sort_by(|a, b| natural_str_cmp(a, b));
        assert_eq!(natural, names);
    }
}

/// A `VecSink` that discards everything. Used during probe-phase dimension
/// discovery for variable-length ListArrays — `CompiledVectorExtractor::drain_batch`
/// populates its `dimension` field as a side effect of the first drain call.
struct DiscardSink {
    count: usize,
}
impl DiscardSink {
    fn new() -> Self { Self { count: 0 } }
}
impl VecSink for DiscardSink {
    fn write_record(&mut self, _ordinal: i64, _data: &[u8]) { self.count += 1; }
    fn finish(self: Box<Self>) -> Result<(), String> { Ok(()) }
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, FixedSizeListArray, Float32Array, Int32Array, ListArray, UInt8Array,
    };
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// In-memory `VecSink` that records every call for inspection.
    struct MemSink {
        records: Vec<(i64, Vec<u8>)>,
    }
    impl MemSink {
        fn new() -> Self { Self { records: Vec::new() } }
    }
    impl VecSink for MemSink {
        fn write_record(&mut self, ordinal: i64, data: &[u8]) {
            self.records.push((ordinal, data.to_vec()));
        }
        fn finish(self: Box<Self>) -> Result<(), String> { Ok(()) }
    }

    fn schema_fixed(dim: i32, element: DataType) -> Schema {
        Schema::new(vec![
            Field::new("embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", element, false)),
                    dim,
                ),
                false,
            ),
        ])
    }

    #[test]
    fn compile_fixed_size_f32() {
        let schema = schema_fixed(4, DataType::Float32);
        let ext = CompiledVectorExtractor::compile(&schema, None).unwrap();
        assert_eq!(ext.list_kind(), ListKind::FixedSize);
        assert_eq!(ext.element(), VectorElement::F32);
        assert_eq!(ext.dimension(), 4);
        assert_eq!(ext.preferred_xvec_format(), VecFormat::Fvec);
    }

    #[test]
    fn compile_rejects_non_list_column() {
        let schema = Schema::new(vec![
            Field::new("not_a_vector", DataType::Utf8, false),
        ]);
        let err = CompiledVectorExtractor::compile(&schema, None).unwrap_err();
        assert!(err.contains("no list-of-numeric column"));
    }

    #[test]
    fn compile_column_hint_by_name() {
        let schema = Schema::new(vec![
            Field::new("junk", DataType::Int32, false),
            Field::new("embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    3,
                ),
                false,
            ),
        ]);
        let ext = CompiledVectorExtractor::compile(&schema, Some("embedding")).unwrap();
        assert_eq!(ext.column_index(), 1);
        assert_eq!(ext.dimension(), 3);
    }

    #[test]
    fn compile_column_hint_unknown_name_errors() {
        let schema = schema_fixed(3, DataType::Float32);
        let err = CompiledVectorExtractor::compile(&schema, Some("does_not_exist"))
            .unwrap_err();
        assert!(err.contains("does_not_exist"));
    }

    /// Build a FixedSizeListArray of f32 vectors for testing.
    fn fixed_size_f32_batch(rows: &[[f32; 3]]) -> RecordBatch {
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
        let arr = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            3,
            values,
            None,
        );
        let schema = Arc::new(schema_fixed(3, DataType::Float32));
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    #[test]
    fn drain_fixed_size_f32_emits_le_bytes() {
        let rows = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let batch = fixed_size_f32_batch(&rows);
        let schema = batch.schema();
        let mut ext = CompiledVectorExtractor::compile(&schema, None).unwrap();

        let mut sink = MemSink::new();
        let written = ext.drain_batch(&batch, 10, &mut sink).unwrap();

        assert_eq!(written, 2);
        assert_eq!(sink.records.len(), 2);
        assert_eq!(sink.records[0].0, 10);
        assert_eq!(sink.records[1].0, 11);

        // Expected LE byte layout: 3 × f32 little-endian per row.
        let expected_row0: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        let expected_row1: Vec<u8> = [4.0f32, 5.0, 6.0]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(sink.records[0].1, expected_row0);
        assert_eq!(sink.records[1].1, expected_row1);
    }

    #[test]
    fn drain_fixed_size_u8_emits_raw_bytes() {
        // u8 row bytes equal the raw payload (no byte-order conversion).
        let values = Arc::new(UInt8Array::from(vec![1u8, 2, 3, 4, 5, 6])) as ArrayRef;
        let arr = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            3,
            values,
            None,
        );
        let schema = Arc::new(schema_fixed(3, DataType::UInt8));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let mut ext = CompiledVectorExtractor::compile(&schema, None).unwrap();
        assert_eq!(ext.element(), VectorElement::U8);
        assert_eq!(ext.preferred_xvec_format(), VecFormat::Bvec);

        let mut sink = MemSink::new();
        ext.drain_batch(&batch, 0, &mut sink).unwrap();
        assert_eq!(sink.records[0].1, vec![1u8, 2, 3]);
        assert_eq!(sink.records[1].1, vec![4u8, 5, 6]);
    }

    #[test]
    fn drain_fixed_size_i32_emits_le_bytes() {
        let values = Arc::new(Int32Array::from(vec![1_i32, 2, 3, 4])) as ArrayRef;
        let arr = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            2,
            values,
            None,
        );
        let schema = Arc::new(schema_fixed(2, DataType::Int32));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let mut ext = CompiledVectorExtractor::compile(&schema, None).unwrap();
        assert_eq!(ext.preferred_xvec_format(), VecFormat::Ivec);

        let mut sink = MemSink::new();
        ext.drain_batch(&batch, 0, &mut sink).unwrap();
        assert_eq!(sink.records[0].1, [1_i32.to_le_bytes(), 2_i32.to_le_bytes()].concat());
        assert_eq!(sink.records[1].1, [3_i32.to_le_bytes(), 4_i32.to_le_bytes()].concat());
    }

    #[test]
    fn drain_variable_list_with_uniform_rows_succeeds() {
        // List<Float32> where every row has the same length should take the
        // same fast path (after validation) and round-trip cleanly.
        let values = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0])) as ArrayRef;
        let offsets = OffsetBuffer::new(vec![0_i32, 2, 4].into());
        let arr = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            offsets,
            values,
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("v",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let mut ext = CompiledVectorExtractor::compile(&schema, None).unwrap();
        assert_eq!(ext.list_kind(), ListKind::Variable);

        let mut sink = MemSink::new();
        ext.drain_batch(&batch, 0, &mut sink).unwrap();
        assert_eq!(sink.records.len(), 2);
        assert_eq!(ext.dimension(), 2);
        assert_eq!(sink.records[0].1, [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat());
        assert_eq!(sink.records[1].1, [3.0f32.to_le_bytes(), 4.0f32.to_le_bytes()].concat());
    }

    /// End-to-end: write a parquet file with a `FixedSizeList<Float32>`
    /// column, run the orchestrator, read the resulting `.fvecs` file, and
    /// confirm every record matches the source bytes exactly.
    #[test]
    fn extract_parquet_to_xvec_fixed_size_f32_roundtrip() {
        use parquet::arrow::ArrowWriter;
        use std::io::Read;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();

        // Build a parquet file with FixedSizeList<Float32, 3> × 4 rows.
        let rows: Vec<[f32; 3]> = vec![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
        let list_field = Arc::new(Field::new("item", DataType::Float32, false));
        let arr = FixedSizeListArray::new(Arc::clone(&list_field), 3, values, None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(list_field, 3),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let parquet_path = tmp.path().join("vectors.parquet");
        let pq_file = fs::File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(pq_file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Run the orchestrator.
        let output_path = tmp.path().join("vectors.fvecs");
        let rows_written = extract_parquet_to_xvec(
            &parquet_path,
            &output_path,
            VecFormat::Fvec,
            None,
        )
        .unwrap();
        assert_eq!(rows_written, 4);

        // xvec format: each record is `[dim:i32 LE][dim × f32 LE]`.
        let mut bytes = Vec::new();
        fs::File::open(&output_path).unwrap().read_to_end(&mut bytes).unwrap();
        let record_size = 4 + 3 * 4;
        assert_eq!(bytes.len(), 4 * record_size);
        for (row_idx, expected) in rows.iter().enumerate() {
            let off = row_idx * record_size;
            let dim = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            assert_eq!(dim, 3);
            for (i, &expected_v) in expected.iter().enumerate() {
                let vo = off + 4 + i * 4;
                let got = f32::from_le_bytes(bytes[vo..vo + 4].try_into().unwrap());
                assert_eq!(got, expected_v, "row {} element {}", row_idx, i);
            }
        }
    }

    #[test]
    fn probe_parquet_vectors_reports_schema_and_row_count() {
        use parquet::arrow::ArrowWriter;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let values = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0])) as ArrayRef;
        let list_field = Arc::new(Field::new("item", DataType::Float32, false));
        let arr = FixedSizeListArray::new(Arc::clone(&list_field), 2, values, None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(list_field, 2),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let parquet_path = tmp.path().join("shard-0.parquet");
        let pq = fs::File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(pq, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let probe = probe_parquet_vectors(&parquet_path, None).unwrap();
        assert_eq!(probe.file_count, 1);
        assert_eq!(probe.row_count, 2);
        assert_eq!(probe.extractor.dimension(), 2);
        assert_eq!(probe.extractor.element(), VectorElement::F32);
        assert_eq!(probe.extractor.preferred_xvec_format(), VecFormat::Fvec);
    }

    /// After a successful run, intermediate shards at
    /// `<output>.shards/<source_stem>.<ext>` are kept on disk so that a
    /// re-run can resume per-stem and verify the output against them.
    /// They live under `.cache/` and are subject to the usual cache
    /// cleanup, but `concat_shards_into_output` itself does not unlink
    /// them.
    #[test]
    fn shards_directory_is_kept_after_concat() {
        use parquet::arrow::ArrowWriter;
        use std::io::Read;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        // Build two parquet files so we exercise multi-shard concat.
        for shard in 0..2 {
            let flat: Vec<f32> = (0..3 * 4)
                .map(|i| (shard * 100 + i) as f32)
                .collect();
            let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
            let list_field = Arc::new(Field::new("item", DataType::Float32, false));
            let arr = FixedSizeListArray::new(Arc::clone(&list_field), 3, values, None);
            let schema = Arc::new(Schema::new(vec![Field::new(
                "embedding",
                DataType::FixedSizeList(list_field, 3),
                false,
            )]));
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();
            let path = tmp.path().join(format!("part-{}.parquet", shard));
            let f = std::fs::File::create(&path).unwrap();
            let mut writer = ArrowWriter::try_new(f, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let output = tmp.path().join("out.fvecs");
        let n = extract_parquet_to_xvec_threaded(
            tmp.path(),
            &output,
            VecFormat::Fvec,
            None,
            None,
            None,
            4,
        ).unwrap();
        assert_eq!(n, 8);

        // Output exists with expected size.
        let stride = 4 + 3 * 4;
        let expected_bytes = 8 * stride as u64;
        assert_eq!(std::fs::metadata(&output).unwrap().len(), expected_bytes);

        // Shards directory is kept (durable resumable intermediate).
        // Each shard's size must match its parquet's row count × stride.
        let shards_dir = shards_dir_for(&output);
        assert!(shards_dir.exists(),
            "shards dir should be kept after concat: {}", shards_dir.display());
        for shard in 0..2 {
            let shard_path = shards_dir.join(format!("part-{}.fvecs", shard));
            assert!(shard_path.exists(),
                "shard file should be kept: {}", shard_path.display());
            // 4 rows × stride bytes
            assert_eq!(
                std::fs::metadata(&shard_path).unwrap().len(),
                4 * stride as u64,
                "shard {} size mismatch", shard,
            );
        }

        // Output bytes match expected sequence.
        let mut bytes = Vec::new();
        std::fs::File::open(&output).unwrap().read_to_end(&mut bytes).unwrap();
        // First record's first value should be 0 (shard 0, row 0, col 0).
        let first_val = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(first_val, 0.0);
    }

    /// A pre-existing shard with the right size MUST be skipped (not
    /// re-decoded). We simulate this by writing a shard file by hand
    /// with content that is provably not what the parquet would produce
    /// (all 0xFF bytes), then run the conversion. If the worker
    /// re-decoded the parquet, our sentinel bytes would be overwritten;
    /// since it should skip, the sentinel survives into the final output.
    #[test]
    fn pre_existing_shard_with_correct_size_is_skipped() {
        use parquet::arrow::ArrowWriter;
        use std::io::{Read, Write};
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let dim = 3usize;
        let rows = 4usize;
        let stride = 4 + dim * 4;
        let shard_bytes = rows * stride;

        // Write a parquet file whose decode would produce non-0xFF data.
        let flat: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
        let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
        let list_field = Arc::new(Field::new("item", DataType::Float32, false));
        let arr = FixedSizeListArray::new(Arc::clone(&list_field), dim as i32, values, None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(list_field, dim as i32),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();
        let parquet_path = tmp.path().join("only.parquet");
        let f = std::fs::File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(f, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Pre-create the shards directory and put a sentinel shard there
        // with the correct size for skip detection.
        let output = tmp.path().join("out.fvecs");
        let shards_dir = shards_dir_for(&output);
        std::fs::create_dir_all(&shards_dir).unwrap();
        let shard_path = shard_path_for(&parquet_path, &shards_dir, "fvecs");
        let sentinel = vec![0xFFu8; shard_bytes];
        std::fs::File::create(&shard_path).unwrap().write_all(&sentinel).unwrap();

        // Run conversion — should detect the existing shard and skip
        // decoding the parquet entirely. Output will contain the
        // sentinel bytes.
        let n = extract_parquet_to_xvec_threaded(
            tmp.path(),
            &output,
            VecFormat::Fvec,
            None,
            None,
            None,
            2, // parallel path so the shard worker runs
        ).unwrap();
        assert_eq!(n, rows as u64);

        let mut got = Vec::new();
        std::fs::File::open(&output).unwrap().read_to_end(&mut got).unwrap();
        assert_eq!(got, sentinel,
            "skipped shard's bytes (0xFF sentinel) must reach the output verbatim");
    }

    /// A pre-existing shard with the WRONG size must be ignored — the
    /// worker re-decodes the parquet and overwrites it. Otherwise a
    /// crash mid-write could leave a corrupt-but-wrong-sized shard that
    /// would be silently accepted on retry.
    #[test]
    fn pre_existing_shard_with_wrong_size_is_overwritten() {
        use parquet::arrow::ArrowWriter;
        use std::io::Write;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let dim = 3usize;
        let rows = 4usize;
        let stride = 4 + dim * 4;
        let _correct_size = rows * stride;

        let flat: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
        let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
        let list_field = Arc::new(Field::new("item", DataType::Float32, false));
        let arr = FixedSizeListArray::new(Arc::clone(&list_field), dim as i32, values, None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(list_field, dim as i32),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();
        let parquet_path = tmp.path().join("only.parquet");
        let f = std::fs::File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(f, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let output = tmp.path().join("out.fvecs");
        let shards_dir = shards_dir_for(&output);
        std::fs::create_dir_all(&shards_dir).unwrap();
        let shard_path = shard_path_for(&parquet_path, &shards_dir, "fvecs");
        // Wrong size: too short by one record.
        std::fs::File::create(&shard_path)
            .unwrap()
            .write_all(&vec![0xFFu8; (rows - 1) * stride])
            .unwrap();

        extract_parquet_to_xvec_threaded(
            tmp.path(), &output, VecFormat::Fvec, None, None, None, 2,
        ).unwrap();

        // Output's first record must be the real decoded value (0.0),
        // not the sentinel.
        use std::io::Read;
        let mut got = Vec::new();
        std::fs::File::open(&output).unwrap().read_to_end(&mut got).unwrap();
        let first_val = f32::from_le_bytes(got[4..8].try_into().unwrap());
        assert_eq!(first_val, 0.0,
            "wrong-sized shard must be re-decoded, not trusted");
    }

    /// Re-running a successful conversion must be a no-op (return the
    /// expected row count without re-writing the output). Idempotency
    /// matters for pipeline runners that might retry steps.
    #[test]
    fn rerun_after_success_is_a_noop() {
        use parquet::arrow::ArrowWriter;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let flat: Vec<f32> = (0..3 * 4).map(|i| i as f32).collect();
        let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
        let list_field = Arc::new(Field::new("item", DataType::Float32, false));
        let arr = FixedSizeListArray::new(Arc::clone(&list_field), 3, values, None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(list_field, 3),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();
        let p = tmp.path().join("only.parquet");
        let f = std::fs::File::create(&p).unwrap();
        let mut writer = ArrowWriter::try_new(f, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let output = tmp.path().join("out.fvecs");
        let n1 = extract_parquet_to_xvec_threaded(
            tmp.path(), &output, VecFormat::Fvec, None, None, None, 2,
        ).unwrap();
        let mtime1 = std::fs::metadata(&output).unwrap().modified().unwrap();

        // Sleep just enough to make a different mtime detectable on
        // filesystems that time stamps at second granularity.
        std::thread::sleep(std::time::Duration::from_millis(1100));

        let n2 = extract_parquet_to_xvec_threaded(
            tmp.path(), &output, VecFormat::Fvec, None, None, None, 2,
        ).unwrap();
        let mtime2 = std::fs::metadata(&output).unwrap().modified().unwrap();

        assert_eq!(n1, n2);
        assert_eq!(n1, 4);
        assert_eq!(mtime1, mtime2,
            "rerun should not have rewritten the output file");
    }

    /// Ordinal-stability guarantee: no matter how many loader threads are
    /// used, the `.fvecs` output must be byte-identical. Builds a directory
    /// of 8 parquet shards with distinctive content (each row's first value
    /// encodes the shard id × 1000 + row index) and confirms the output is
    /// the same across thread counts 1, 2, 4, and 8.
    #[test]
    fn extract_parquet_to_xvec_is_ordinal_stable_across_thread_counts() {
        use parquet::arrow::ArrowWriter;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let shard_count = 8usize;
        let rows_per_shard = 7usize; // odd count so batches don't divide evenly
        let dim = 5usize;

        for shard in 0..shard_count {
            let flat: Vec<f32> = (0..rows_per_shard)
                .flat_map(|r| {
                    let base = (shard * 1000 + r) as f32;
                    (0..dim).map(move |c| base + 0.001 * c as f32)
                })
                .collect();
            let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
            let list_field = Arc::new(Field::new("item", DataType::Float32, false));
            let arr = FixedSizeListArray::new(Arc::clone(&list_field), dim as i32, values, None);
            let schema = Arc::new(Schema::new(vec![Field::new(
                "embedding",
                DataType::FixedSizeList(list_field, dim as i32),
                false,
            )]));
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();
            // Pad shard id with leading zeros so lexical sort == numeric sort.
            let path = tmp.path().join(format!("shard-{:02}.parquet", shard));
            let pq = fs::File::create(&path).unwrap();
            let mut writer = ArrowWriter::try_new(pq, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let read_output = |path: &Path| -> Vec<u8> {
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(
                &mut fs::File::open(path).unwrap(),
                &mut bytes,
            ).unwrap();
            bytes
        };

        let mut baseline: Option<Vec<u8>> = None;
        for threads in [1usize, 2, 4, 8, 16] {
            let out = tmp.path().join(format!("out-{}.fvecs", threads));
            let rows = extract_parquet_to_xvec_threaded(
                tmp.path(),
                &out,
                VecFormat::Fvec,
                None,
                None,
                None,
                threads,
            ).unwrap();
            assert_eq!(rows, (shard_count * rows_per_shard) as u64);
            let bytes = read_output(&out);

            // Sanity: decode row 0 — should match shard 0, row 0.
            let record_size = 4 + dim * 4;
            assert_eq!(bytes.len(), shard_count * rows_per_shard * record_size);
            let first_val = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
            assert_eq!(first_val, 0.0, "threads={}: first row first value", threads);
            // And the last record should be shard_count-1, row rows_per_shard-1.
            let last_off = bytes.len() - record_size + 4;
            let last_val = f32::from_le_bytes(
                bytes[last_off..last_off + 4].try_into().unwrap(),
            );
            let expected_last = ((shard_count - 1) * 1000 + (rows_per_shard - 1)) as f32;
            assert_eq!(last_val, expected_last, "threads={}: last row first value", threads);

            match &baseline {
                None => baseline = Some(bytes),
                Some(b) => assert_eq!(
                    &bytes, b,
                    "threads={}: output differs from baseline — ordinal stability violated",
                    threads,
                ),
            }
        }
    }

    #[test]
    fn extract_rejects_element_type_mismatch() {
        use parquet::arrow::ArrowWriter;
        use tempfile::TempDir;

        // Source is Int32, target is Fvec (f32) — should error.
        let tmp = TempDir::new().unwrap();
        let values = Arc::new(Int32Array::from(vec![1_i32, 2, 3, 4])) as ArrayRef;
        let list_field = Arc::new(Field::new("item", DataType::Int32, false));
        let arr = FixedSizeListArray::new(Arc::clone(&list_field), 2, values, None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "indices",
            DataType::FixedSizeList(list_field, 2),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let parquet_path = tmp.path().join("ints.parquet");
        let pq = fs::File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(pq, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let err = extract_parquet_to_xvec(
            &parquet_path,
            &tmp.path().join("out.fvecs"),
            VecFormat::Fvec,
            None,
        )
        .unwrap_err();
        assert!(err.contains("element"), "err was: {}", err);
    }

    #[test]
    fn drain_variable_list_with_nonuniform_rows_errors() {
        let values = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0])) as ArrayRef;
        let offsets = OffsetBuffer::new(vec![0_i32, 2, 3].into());
        let arr = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            offsets,
            values,
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("v",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let mut ext = CompiledVectorExtractor::compile(&schema, None).unwrap();
        let mut sink = MemSink::new();
        let err = ext.drain_batch(&batch, 0, &mut sink).unwrap_err();
        assert!(err.contains("dimension"));
    }
}
