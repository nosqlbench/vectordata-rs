// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A facet that spans several files on disk: finding a series' shards
//! ([`discover_shards`]) and measuring what they hold
//! ([`observe_series`]). The naming rule (`<stem>__NNNN.<ext>`) is the
//! API core's (`veks_apicore::dataset::shards`); this is the format
//! layer's reading of it against a directory.

use veks_apicore::dataset::shards::shard_name;

/// The shard files of the series whose unsharded name is `path`.
///
/// Uniform naming only (SH-2): `<stem>__NNNN.<ext>`, four digits,
/// consecutive from zero. Stops at the first gap rather than skipping
/// it — a series missing shard 3 is a broken facet, and silently
/// reading shards 0–2 as the whole thing would hand back a prefix that
/// looks complete.
///
/// Empty when there is no shard zero, which is the ordinary case for
/// every unsharded file.
pub fn discover_shards(path: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Some(dir) = path.parent() else {
        return Vec::new();
    };
    let Some(file) = path.file_name().and_then(|n| n.to_str()) else {
        return Vec::new();
    };
    let (basename, ext) = match file.rsplit_once('.') {
        Some((b, e)) => (b, e),
        None => (file, ""),
    };
    let mut out = Vec::new();
    for i in 0u32.. {
        let candidate = dir.join(shard_name(basename, ext, i));
        if !candidate.is_file() {
            break;
        }
        out.push(candidate);
    }
    out
}

/// What a set of shard files on disk actually holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservedSeries {
    /// Records per shard, taken from the first — every shard but the
    /// last is full (SH-35).
    pub stride: u64,
    /// How many shards there are.
    pub shards: u32,
    /// Records across all of them.
    pub records: u64,
}

/// Measure a series from its files.
///
/// Reads each shard's dim header and size, which is two syscalls per
/// shard and no data. `None` when the shards do not agree on a
/// dimension, or when one is not a whole number of records — either
/// means these files are not one uniform facet, and declaring them as
/// one would produce a `record_count` nothing could satisfy.
pub fn observe_series(shards: &[std::path::PathBuf]) -> Option<ObservedSeries> {
    use std::io::Read;
    let mut counts: Vec<u64> = Vec::with_capacity(shards.len());
    let mut stride_bytes: Option<u64> = None;
    for path in shards {
        let len = std::fs::metadata(path).ok()?.len();
        let mut header = [0u8; 4];
        let mut f = std::fs::File::open(path).ok()?;
        f.read_exact(&mut header).ok()?;
        let dim = i32::from_le_bytes(header);
        if dim <= 0 {
            return None;
        }
        // The element width is not knowable from the header alone, but
        // it does not need to be: every shard shares one record size,
        // so the first shard's size divided by its record count gives
        // it — and the first shard is full by construction.
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let elem = crate::reader::infer_elem_size(ext) as u64;
        if elem == 0 {
            return None;
        }
        let record = 4 + dim as u64 * elem;
        match stride_bytes {
            None => stride_bytes = Some(record),
            Some(r) if r == record => {}
            // Shards disagreeing on record size are not one facet.
            Some(_) => return None,
        }
        if !len.is_multiple_of(record) {
            return None;
        }
        counts.push(len / record);
    }
    let stride = *counts.first()?;
    if stride == 0 {
        return None;
    }
    Some(ObservedSeries {
        stride,
        shards: counts.len() as u32,
        records: counts.iter().sum(),
    })
}
