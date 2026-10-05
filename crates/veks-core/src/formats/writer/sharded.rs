// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Writing a facet across a series of capped shard files.
//!
//! A [`VecSink`] decorator: it takes the same `write_record(ordinal,
//! data)` calls any sink takes and rolls over to a new file every
//! `stride` ordinals. What it wraps is an ordinary sink, opened once
//! per shard, so every format that can be written at all can be
//! written as a series without knowing that it is one.
//!
//! Three properties it exists to hold, matching the ones
//! `vectordata`'s shard writer holds for derived output:
//!
//! - **A shard is an ordinary file of its format** (SH-96). Ordinals
//!   are translated to shard-local before the inner sink sees them, so
//!   shard 3 of a slab is a slab based at zero — openable on its own,
//!   by anything, with no knowledge of the series.
//! - **Collapsing** (SH-83). A run that fits in one shard is left as
//!   the plain single file, so a facet that happened to fit does not
//!   leave behind a declaration older readers cannot open.
//! - **The declaration describes what was written** (SH-37), so
//!   [`ShardedSink::outcome`] is produced by finishing, never before.
//!
//! See `docs/design/srd-multifile-facet-shards.md`.

use std::path::{Path, PathBuf};

use vectordata::dataset::{shard_name, shard_source_spec};

use super::VecSink;

/// Opens the sink for one shard file.
///
/// Taken as a closure rather than a format, so this decorator works
/// for anything writable — including sinks a caller configures itself
/// — without a second dispatch table beside [`super::open_sink`].
pub type ShardOpener = Box<dyn Fn(&Path) -> Result<Box<dyn VecSink>, String> + Send>;

/// What a completed sharded write produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedOutcome {
    /// Files written, in ordinal order.
    pub files: Vec<PathBuf>,
    /// The `source:` value for the declaration — the `NNNN` pattern
    /// for a series, a plain filename when the run collapsed.
    pub source_spec: String,
    /// Ordinals per shard. `None` when the output collapsed.
    pub shard_stride: Option<u64>,
    /// Shards written. `None` when the output collapsed.
    pub shard_count: Option<u32>,
    /// Records written across every shard.
    pub record_count: u64,
}

impl ShardedOutcome {
    /// Whether the declaration needs `shard_stride`/`shard_count`.
    pub fn is_series(&self) -> bool {
        self.shard_stride.is_some()
    }
}

/// A sink that writes its records across capped shard files.
pub struct ShardedSink {
    dir: PathBuf,
    basename: String,
    ext: String,
    stride: u64,
    opener: ShardOpener,
    /// The shard currently open, and its index.
    open: Option<(u32, Box<dyn VecSink>)>,
    files: Vec<PathBuf>,
    records: u64,
    /// Highest shard index written so far, to catch a caller whose
    /// ordinals walk backwards across a seam.
    highest_shard: Option<u32>,
    /// The path a collapsed single-shard run ends up at.
    collapsed_path: PathBuf,
    /// The first failure seen. Held rather than raised, because
    /// `write_record` cannot report one, and surfaced by `finish`.
    failed: Option<String>,
}

impl ShardedSink {
    /// Open a sink writing `stride` records per shard file.
    ///
    /// `path` is the single-file path the facet would otherwise have
    /// had; the shards are named from its stem and extension, and a
    /// run that fits in one shard is renamed back to it.
    pub fn open(path: &Path, stride: u64, opener: ShardOpener) -> Result<Self, String> {
        if stride == 0 {
            return Err("shard stride must be greater than zero".to_string());
        }
        let file = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("{} has no filename to shard", path.display()))?;
        let (basename, ext) = match file.rsplit_once('.') {
            Some((b, e)) => (b.to_string(), e.to_string()),
            None => (file.to_string(), String::new()),
        };
        Ok(Self {
            dir: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            collapsed_path: path.to_path_buf(),
            basename,
            ext,
            stride,
            opener,
            open: None,
            files: Vec::new(),
            records: 0,
            highest_shard: None,
            failed: None,
        })
    }

    /// The final path of shard `i`.
    fn shard_path(&self, i: u32) -> PathBuf {
        self.dir.join(shard_name(&self.basename, &self.ext, i))
    }

    /// Close whichever shard is open, if any.
    fn close_current(&mut self) {
        if let Some((_, sink)) = self.open.take()
            && let Err(e) = sink.finish()
            && self.failed.is_none()
        {
            self.failed = Some(e);
        }
    }

    /// Make shard `index` the open one, closing any predecessor.
    fn ensure_shard(&mut self, index: u32) {
        if matches!(self.open, Some((i, _)) if i == index) {
            return;
        }
        // A sink writes one shard at a time: reopening a finished
        // shard would truncate it. Ordinals that walk backwards across
        // a seam are a caller error, and saying so beats writing a
        // file that is quietly missing its earlier records.
        if let Some(highest) = self.highest_shard
            && index <= highest
        {
            if self.failed.is_none() {
                self.failed = Some(format!(
                    "ordinals moved back into shard {index} after shard {highest} was \
                     closed; a sharded sink writes one shard at a time, so records must \
                     arrive in ordinal order"
                ));
            }
            return;
        }
        self.close_current();
        if self.failed.is_some() {
            return;
        }
        let path = self.shard_path(index);
        match (self.opener)(&path) {
            Ok(sink) => {
                self.files.push(path);
                self.highest_shard = Some(index);
                self.open = Some((index, sink));
            }
            Err(e) => self.failed = Some(e),
        }
    }

    /// Finish every shard and report what was written.
    ///
    /// Collapses a single-shard run back to the unsharded filename
    /// (SH-83), which is why this returns the outcome rather than the
    /// caller deriving one from a stride it asked for.
    pub fn outcome(mut self: Box<Self>) -> Result<ShardedOutcome, String> {
        self.close_current();
        if let Some(e) = self.failed.take() {
            return Err(e);
        }
        match self.files.len() {
            // Nothing was written: no files, and no series to declare.
            0 => Ok(ShardedOutcome {
                files: Vec::new(),
                source_spec: file_name_of(&self.collapsed_path),
                shard_stride: None,
                shard_count: None,
                record_count: 0,
            }),
            // One shard is not a series (SH-83): rename it back to the
            // plain filename so the declaration stays the simple form.
            1 => {
                let only = &self.files[0];
                std::fs::rename(only, &self.collapsed_path).map_err(|e| {
                    format!(
                        "collapse {} to {}: {e}",
                        only.display(),
                        self.collapsed_path.display()
                    )
                })?;
                Ok(ShardedOutcome {
                    files: vec![self.collapsed_path.clone()],
                    source_spec: file_name_of(&self.collapsed_path),
                    shard_stride: None,
                    shard_count: None,
                    record_count: self.records,
                })
            }
            n => Ok(ShardedOutcome {
                source_spec: shard_source_spec(&self.basename, &self.ext),
                shard_stride: Some(self.stride),
                shard_count: Some(n as u32),
                record_count: self.records,
                files: self.files.clone(),
            }),
        }
    }
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

impl VecSink for ShardedSink {
    fn write_record(&mut self, ordinal: i64, data: &[u8]) {
        if self.failed.is_some() {
            return;
        }
        let global = ordinal.max(0) as u64;
        let index = global / self.stride;
        // The shard index is a u32 in the filename (SH-2, four
        // digits), so a stride small enough to need more is a
        // misconfiguration rather than a layout.
        let Ok(index) = u32::try_from(index) else {
            self.failed = Some(format!(
                "ordinal {global} needs shard {index}, past what a shard index holds; \
                 raise the shard size"
            ));
            return;
        };
        self.ensure_shard(index);
        let Some((_, sink)) = self.open.as_mut() else {
            return;
        };
        // Shard-local, so each shard is an ordinary file of its format
        // based at zero (SH-96).
        sink.write_record((global % self.stride) as i64, data);
        self.records += 1;
    }

    fn finish(self: Box<Self>) -> Result<(), String> {
        self.outcome().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::VecFormat;
    use crate::formats::writer::{SinkConfig, open_sink};

    fn config(dim: u32) -> SinkConfig {
        SinkConfig {
            dimension: dim,
            source_format: VecFormat::Fvec,
            slab_page_size: None,
            slab_namespace: 0,
            schema_sidecar: None,
            // The decorator is given an explicit stride here, so the
            // per-shard config carries no cap of its own.
            max_shard_bytes: None,
        }
    }

    /// An fvec sink for each shard, at whatever path the decorator
    /// hands it.
    fn fvec_opener(dim: u32) -> ShardOpener {
        Box::new(move |path: &Path| open_sink(path, VecFormat::Fvec, &config(dim)))
    }

    fn write(sink: &mut Box<ShardedSink>, from: i64, count: i64, dim: u32) {
        for o in from..from + count {
            let data: Vec<u8> = (0..dim)
                .flat_map(|d| ((o as f32) + (d as f32) / 1000.0).to_le_bytes())
                .collect();
            sink.write_record(o, &data);
        }
    }

    /// Read an fvec back as `(dim, records)`.
    fn read_fvec(path: &Path) -> (u32, u64) {
        let bytes = std::fs::read(path).unwrap();
        let dim = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let stride = 4 + dim as u64 * 4;
        (dim, bytes.len() as u64 / stride)
    }

    /// **A capped facet becomes a series**, named by the same
    /// convention the declaration and the readers use.
    #[test]
    fn records_roll_over_into_named_shards() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let mut sink = Box::new(ShardedSink::open(&path, 10, fvec_opener(4)).unwrap());
        write(&mut sink, 0, 25, 4);
        let out = sink.outcome().unwrap();

        assert_eq!(out.record_count, 25);
        assert_eq!(out.shard_stride, Some(10));
        assert_eq!(out.shard_count, Some(3));
        assert_eq!(out.source_spec, "base__NNNN.fvec");
        assert!(out.is_series());

        let names: Vec<String> = out
            .files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["base__0000.fvec", "base__0001.fvec", "base__0002.fvec"]);
        // Full, full, remainder.
        assert_eq!(read_fvec(&out.files[0]), (4, 10));
        assert_eq!(read_fvec(&out.files[1]), (4, 10));
        assert_eq!(read_fvec(&out.files[2]), (4, 5));
    }

    /// **Each shard is an ordinary file of its format** (SH-96): the
    /// ordinals the inner sink sees are shard-local, so a shard opens
    /// on its own with no knowledge of the series.
    #[test]
    fn a_shard_is_an_ordinary_file_based_at_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("q.fvec");
        let mut sink = Box::new(ShardedSink::open(&path, 8, fvec_opener(3)).unwrap());
        write(&mut sink, 0, 20, 3);
        let out = sink.outcome().unwrap();

        // Every shard reads as a complete fvec of its own.
        for (i, f) in out.files.iter().enumerate() {
            let (dim, records) = read_fvec(f);
            assert_eq!(dim, 3, "shard {i} carries its own dim header");
            assert_eq!(records, if i < 2 { 8 } else { 4 }, "shard {i}");
        }
        // And the concatenation is the whole facet.
        let total: u64 = out.files.iter().map(|f| read_fvec(f).1).sum();
        assert_eq!(total, 20);
    }

    /// **A run that fits in one shard collapses** (SH-83) back to the
    /// plain filename, so a facet that happened to fit does not leave
    /// a declaration older readers cannot open.
    #[test]
    fn a_single_shard_run_collapses_to_one_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let mut sink = Box::new(ShardedSink::open(&path, 1000, fvec_opener(4)).unwrap());
        write(&mut sink, 0, 40, 4);
        let out = sink.outcome().unwrap();

        assert_eq!(out.source_spec, "base.fvec");
        assert_eq!(out.shard_stride, None);
        assert_eq!(out.shard_count, None);
        assert!(!out.is_series());
        assert_eq!(out.files, vec![path.clone()]);
        assert!(path.exists(), "the collapsed name is the one on disk");
        assert!(!tmp.path().join("base__0000.fvec").exists());
        assert_eq!(read_fvec(&path), (4, 40));
    }

    /// An exactly-full single shard still collapses — the boundary
    /// where "fits" and "rolls over" are one record apart.
    #[test]
    fn an_exactly_full_single_shard_still_collapses() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let mut sink = Box::new(ShardedSink::open(&path, 10, fvec_opener(2)).unwrap());
        write(&mut sink, 0, 10, 2);
        assert_eq!(sink.outcome().unwrap().shard_count, None);

        // One more record is a series.
        let path2 = tmp.path().join("other.fvec");
        let mut sink = Box::new(ShardedSink::open(&path2, 10, fvec_opener(2)).unwrap());
        write(&mut sink, 0, 11, 2);
        let out = sink.outcome().unwrap();
        assert_eq!(out.shard_count, Some(2));
        assert_eq!(read_fvec(&out.files[1]), (2, 1));
    }

    /// Writing nothing produces no files and no series, rather than an
    /// empty shard.
    #[test]
    fn writing_no_records_produces_no_files() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let sink = Box::new(ShardedSink::open(&path, 10, fvec_opener(4)).unwrap());
        let out = sink.outcome().unwrap();

        assert_eq!(out.record_count, 0);
        assert!(out.files.is_empty());
        assert!(!out.is_series());
        assert!(!path.exists());
    }

    /// **Ordinals that walk backwards across a seam are refused.** A
    /// finished shard cannot be reopened without truncating it, so the
    /// alternative to this error is a file quietly missing records.
    #[test]
    fn ordinals_that_move_back_across_a_seam_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let mut sink = Box::new(ShardedSink::open(&path, 10, fvec_opener(2)).unwrap());
        write(&mut sink, 0, 15, 2); // into shard 1
        write(&mut sink, 3, 1, 2); // back into shard 0

        let err = sink.outcome().unwrap_err();
        assert!(err.contains("ordinal order"), "{err}");
        assert!(err.contains("shard 0"), "{err}");
    }

    /// Ordinals within one shard may still arrive in any order — the
    /// constraint is on crossing a seam, not on the records inside a
    /// shard, which is what an mmap-style sink needs.
    #[test]
    fn ordinals_within_a_shard_are_unconstrained() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let mut sink = Box::new(ShardedSink::open(&path, 100, fvec_opener(2)).unwrap());
        for o in [5i64, 2, 9, 0, 7] {
            write(&mut sink, o, 1, 2);
        }
        let out = sink.outcome().unwrap();
        assert_eq!(out.record_count, 5);
    }

    /// A zero stride names no layout and is refused when the sink is
    /// opened, not discovered at the first record.
    #[test]
    fn a_zero_stride_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(ShardedSink::open(&tmp.path().join("b.fvec"), 0, fvec_opener(2)).is_err());
    }

    /// A failure opening a shard surfaces from `finish` rather than
    /// leaving a partial series behind silently.
    #[test]
    fn a_failure_to_open_a_shard_surfaces() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let opener: ShardOpener = Box::new(move |p: &Path| {
            // The second shard cannot be opened.
            if p.to_string_lossy().contains("0001") {
                Err("no space".to_string())
            } else {
                open_sink(p, VecFormat::Fvec, &config(2))
            }
        });
        let mut sink = Box::new(ShardedSink::open(&path, 4, opener).unwrap());
        write(&mut sink, 0, 9, 2);
        assert_eq!(sink.outcome().unwrap_err(), "no space");
    }

    /// A slab series shards the same way an xvec one does — the point
    /// of decorating the sink rather than teaching each format.
    #[test]
    fn a_slab_facet_shards_through_the_same_decorator() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("metadata_content.slab");
        let opener: ShardOpener = Box::new(move |p: &Path| {
            open_sink(p, VecFormat::Slab, &config(4))
        });
        let mut sink = Box::new(ShardedSink::open(&path, 10, opener).unwrap());
        write(&mut sink, 0, 25, 4);
        let out = sink.outcome().unwrap();

        assert_eq!(out.shard_count, Some(3));
        assert_eq!(out.source_spec, "metadata_content__NNNN.slab");

        // Each shard opens as a slab in its own right, based at zero.
        for (i, f) in out.files.iter().enumerate() {
            let reader = slabtastic::SlabReader::open(f).unwrap();
            assert_eq!(
                reader.total_records(),
                if i < 2 { 10 } else { 5 },
                "shard {i}"
            );
        }
    }
}
