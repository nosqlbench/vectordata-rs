// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Sequential xvec writer with buffered I/O.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::traits::VecSink;

/// Size of the `BufWriter`'s internal buffer.
///
/// 16 MiB is a balance between two concerns:
/// - Larger buffers reduce the number of `write` syscalls.
/// - Each flush is a blocking `write` of the whole buffer; making it too
///   large (e.g. 64 MiB) causes long stalls under writeback throttling
///   (Linux's `dirty_ratio`) on cold-disk targets like EBS, because the
///   producer side serializes on the flush.
const WRITE_BUF_SIZE: usize = 16 << 20; // 16 MiB

/// Open an xvec file for sequential writing.
///
/// The `dimension` is written as the header of every record.
/// Element size is inferred from the output format (determined by
/// the file extension at the call site).
pub fn open(path: &Path, dimension: u32) -> Result<Box<dyn VecSink>, String> {
    let file = File::create(path)
        .map_err(|e| format!("create {}: {}", path.display(), e))?;
    let writer = BufWriter::with_capacity(WRITE_BUF_SIZE, file);

    Ok(Box::new(XvecWriter {
        writer,
        // `dim_prefix` is the 4-byte little-endian dim header emitted in
        // front of every record. Pre-computed once at open time so the hot
        // loop is a plain copy of a small stack slice.
        dim_prefix: (dimension as i32).to_le_bytes(),
        // Reusable per-batch scratch buffer for the coalesced batch-write
        // path. Grows on demand and is reused across batches so the steady
        // state has zero allocations in the hot loop.
        scratch: Vec::new(),
    }))
}

struct XvecWriter {
    writer: BufWriter<File>,
    dim_prefix: [u8; 4],
    scratch: Vec<u8>,
}

impl VecSink for XvecWriter {
    fn write_record(&mut self, _ordinal: i64, data: &[u8]) {
        let _ = self.writer.write_all(&self.dim_prefix);
        let _ = self.writer.write_all(data);
    }

    /// Write `count` fixed-dim records from a contiguous packed buffer of
    /// raw element bytes. The hot path on the parquet → xvec extractor.
    ///
    /// Implementation: build a single batch-sized interleaved buffer
    /// `[dim][rec0][dim][rec1]...` in `scratch`, then issue ONE
    /// `write_all` to the underlying `BufWriter`. When the scratch slice
    /// is larger than the BufWriter's internal buffer, std's `BufWriter`
    /// bypasses its own buffer and writes the slice directly to the
    /// inner `File` in a single `write()` syscall — turning what was
    /// previously `2 * count` per-record `write_all` calls into one
    /// kernel write per batch.
    ///
    /// Tradeoff: one extra memcpy per record (Arrow → scratch, then
    /// scratch → kernel), but each memcpy is a tight `copy_from_slice`
    /// at memory bandwidth (~5 GiB/s) — cheaper than the per-record
    /// virtual + bounds-check overhead it replaces.
    fn write_records_fixed_dim(
        &mut self,
        _ordinal_start: i64,
        packed: &[u8],
        count: usize,
        record_size: usize,
    ) {
        debug_assert_eq!(packed.len(), count * record_size);
        let stride = 4 + record_size;
        let total = count * stride;
        // resize fills with zeros; the loop overwrites every byte. Could
        // be replaced with `set_len` after `reserve` for an unsafe fast
        // path, but the zero-fill is well-vectorized by the compiler and
        // not the bottleneck (tested under perf).
        self.scratch.resize(total, 0);
        let mut dst = 0;
        let mut src = 0;
        for _ in 0..count {
            self.scratch[dst..dst + 4].copy_from_slice(&self.dim_prefix);
            self.scratch[dst + 4..dst + stride]
                .copy_from_slice(&packed[src..src + record_size]);
            dst += stride;
            src += record_size;
        }
        let _ = self.writer.write_all(&self.scratch);
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        self.writer.flush().map_err(|e| format!("flush: {}", e))
    }
}

#[allow(dead_code)]
fn _unused_byteorder(_: u32) -> std::io::Result<()> {
    // Keep byteorder import live even though the hot path uses LE bytes directly.
    let mut w = std::io::sink();
    w.write_i32::<LittleEndian>(0)
}
