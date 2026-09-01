// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Writing a fixed-stride facet across capped shard files.
//!
//! The second of this crate's two output seams. Record-oriented
//! commands go through
//! [`open_sink`](veks_core::formats::writer::open_sink), which shards
//! by decorating a sink; the commands here write `[dim][elements]`
//! straight into a buffered file, and this is what caps *those*.
//!
//! **Rollover is arithmetic, not a callback.** Every record of a
//! uniform xvec is exactly `record_bytes` long, so byte offset
//! `stride * record_bytes` is always a record boundary no matter how
//! the caller chunks its `write_all` calls. That is what lets the
//! existing write loops stay exactly as they are: they keep writing a
//! byte stream, and the split happens underneath at offsets that
//! cannot land mid-record.
//!
//! A write that would straddle a boundary is split across two shards
//! rather than overflowing one, so the invariant holds for a caller
//! that writes a whole record in one call, one element at a time, or
//! anything between.
//!
//! Each shard is written to a temp and renamed, and a run that fits in
//! one shard is renamed back to the unsharded name (SH-83) — so a
//! facet that happened to fit is indistinguishable from one written
//! before the cap existed.
//!
//! See `docs/design/srd-multifile-facet-shards.md`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use vectordata::dataset::{shard_name, shard_source_spec};

use super::atomic_write::{AtomicWriter, temp_path_for};

/// What a completed sharded write produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardWriteOutcome {
    /// Files written, in ordinal order.
    pub files: Vec<PathBuf>,
    /// The `source:` value the declaration needs — the `NNNN` pattern
    /// for a series, the plain filename when it collapsed.
    pub source_spec: String,
    /// Records per shard. `None` when the output collapsed to one file.
    pub shard_stride: Option<u64>,
    /// Shards written. `None` when the output collapsed.
    pub shard_count: Option<u32>,
    /// Records written across every shard.
    pub record_count: u64,
}

impl ShardWriteOutcome {
    /// Whether the declaration needs `shard_stride`/`shard_count`.
    pub fn is_series(&self) -> bool {
        self.shard_stride.is_some()
    }
}

/// A `Write` that splits its stream into capped shard files at exact
/// record boundaries.
pub struct ShardingWriter {
    dir: PathBuf,
    basename: String,
    ext: String,
    /// Bytes in one record. Every record is this long, which is what
    /// makes an arithmetic boundary a record boundary.
    record_bytes: u64,
    /// Records per shard.
    stride: u64,
    /// Bytes one full shard holds.
    shard_bytes: u64,
    /// Bytes written into the shard currently open.
    in_shard: u64,
    /// Bytes written in total.
    total: u64,
    next_index: u32,
    /// The shard being written: temp path, final path, handle.
    open: Option<(PathBuf, PathBuf, BufWriter<File>)>,
    finished: Vec<PathBuf>,
    /// The path a collapsed single-shard run ends up at.
    collapsed_path: PathBuf,
}

impl ShardingWriter {
    /// Open a writer that rolls over every `stride` records of
    /// `record_bytes` bytes each.
    pub fn new(final_path: &Path, record_bytes: u64, stride: u64) -> io::Result<Self> {
        if record_bytes == 0 || stride == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a shard needs a non-zero record size and stride",
            ));
        }
        if let Some(parent) = final_path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = final_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{} has no filename to shard", final_path.display()),
                )
            })?;
        let (basename, ext) = match file.rsplit_once('.') {
            Some((b, e)) => (b.to_string(), e.to_string()),
            None => (file.to_string(), String::new()),
        };
        Ok(Self {
            dir: final_path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            collapsed_path: final_path.to_path_buf(),
            basename,
            ext,
            record_bytes,
            stride,
            shard_bytes: stride * record_bytes,
            in_shard: 0,
            total: 0,
            next_index: 0,
            open: None,
            finished: Vec::new(),
        })
    }

    fn open_next(&mut self) -> io::Result<()> {
        let final_path = self.dir.join(shard_name(&self.basename, &self.ext, self.next_index));
        let temp_path = temp_path_for(&final_path);
        let file = File::create(&temp_path)?;
        self.open = Some((temp_path, final_path, BufWriter::with_capacity(1 << 20, file)));
        self.next_index += 1;
        self.in_shard = 0;
        Ok(())
    }

    /// Flush and rename whichever shard is open.
    fn close_current(&mut self) -> io::Result<()> {
        if let Some((temp, final_path, mut w)) = self.open.take() {
            w.flush()?;
            drop(w);
            if final_path.is_symlink() {
                std::fs::remove_file(&final_path)?;
            }
            std::fs::rename(&temp, &final_path)?;
            self.finished.push(final_path);
        }
        Ok(())
    }

    /// Finish every shard and report what was written.
    pub fn finish(mut self) -> io::Result<ShardWriteOutcome> {
        self.close_current()?;
        let records = self.total / self.record_bytes;
        let name_of = |p: &Path| {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string()
        };
        match self.finished.len() {
            0 => Ok(ShardWriteOutcome {
                files: Vec::new(),
                source_spec: name_of(&self.collapsed_path),
                shard_stride: None,
                shard_count: None,
                record_count: 0,
            }),
            // One shard is not a series (SH-83).
            1 => {
                let only = &self.finished[0];
                if self.collapsed_path.is_symlink() {
                    std::fs::remove_file(&self.collapsed_path)?;
                }
                std::fs::rename(only, &self.collapsed_path)?;
                Ok(ShardWriteOutcome {
                    files: vec![self.collapsed_path.clone()],
                    source_spec: name_of(&self.collapsed_path),
                    shard_stride: None,
                    shard_count: None,
                    record_count: records,
                })
            }
            n => Ok(ShardWriteOutcome {
                source_spec: shard_source_spec(&self.basename, &self.ext),
                shard_stride: Some(self.stride),
                shard_count: Some(n as u32),
                record_count: records,
                files: self.finished.clone(),
            }),
        }
    }
}

impl Write for ShardingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.open.is_none() {
            self.open_next()?;
        }
        // Never write past this shard's extent: the split point is the
        // whole point, so a write that would straddle it is truncated
        // here and the remainder lands in the next shard on the
        // caller's next call (which `write_all` makes for us).
        let room = self.shard_bytes - self.in_shard;
        let take = (buf.len() as u64).min(room) as usize;
        let (_, _, w) = self.open.as_mut().expect("a shard is open");
        let wrote = w.write(&buf[..take])?;
        self.in_shard += wrote as u64;
        self.total += wrote as u64;
        if self.in_shard == self.shard_bytes {
            // Exactly full, and exactly on a record boundary because
            // shard_bytes is a whole multiple of record_bytes.
            self.close_current()?;
        }
        Ok(wrote)
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some((_, _, w)) = self.open.as_mut() {
            w.flush()?;
        }
        Ok(())
    }
}

impl Drop for ShardingWriter {
    /// An abandoned write leaves no half-shard behind: the temp of the
    /// shard still open is removed, and the renamed ones stay as the
    /// complete files they are.
    fn drop(&mut self) {
        if let Some((temp, _, _)) = self.open.take() {
            let _ = std::fs::remove_file(&temp);
        }
    }
}

/// A facet output that is one file or a series, decided by the cap.
///
/// Both arms are `Write`, so a command's existing write loop does not
/// change: it opens one of these instead of a `BufWriter`, writes the
/// same bytes, and finishes with an outcome describing whichever shape
/// it got. A command that passes no cap gets exactly the single file
/// it always got, through the same buffered path.
pub enum FacetWriter {
    /// One file, written to a temp and renamed on finish.
    ///
    /// [`AtomicWriter`] rather than a bare `BufWriter`, so an
    /// abandoned single-file write cleans up after itself exactly as
    /// an abandoned sharded one does.
    One {
        writer: AtomicWriter,
        bytes: u64,
        record_bytes: u64,
    },
    /// A capped series.
    Series(ShardingWriter),
}

impl FacetWriter {
    /// Open an output for records of `record_bytes`, capped at
    /// `max_shard_bytes` if given.
    ///
    /// Falls back to a single file when there is no cap, when the
    /// record size is unknown (`0`), or when the cap cannot hold a
    /// meaningful run — in that last case the alternative is a file
    /// per handful of records, which is worse than one large file and
    /// is not what a cap is asking for.
    pub fn open(
        final_path: &Path,
        record_bytes: u64,
        max_shard_bytes: Option<u64>,
    ) -> io::Result<Self> {
        let stride = match (max_shard_bytes, record_bytes) {
            (Some(cap), rb) if rb > 0 => {
                vectordata::dataset::shard_sizing::plan_fixed(cap, rb).map(|p| p.stride)
            }
            _ => None,
        };
        match stride {
            Some(stride) => Ok(Self::Series(ShardingWriter::new(
                final_path,
                record_bytes,
                stride,
            )?)),
            None => Ok(Self::One {
                writer: AtomicWriter::new(final_path)?,
                bytes: 0,
                record_bytes,
            }),
        }
    }

    /// Whether this output is being written as a series.
    pub fn is_sharded(&self) -> bool {
        matches!(self, Self::Series(_))
    }

    /// Finish the output and report what was written.
    ///
    /// A run that wrote nothing still leaves the file, empty. That is
    /// what the single-file writer this replaces always did, and steps
    /// downstream check their inputs by existence — an extract over an
    /// empty range produces an empty facet, not a missing one.
    pub fn finish(self) -> io::Result<ShardWriteOutcome> {
        match self {
            Self::Series(w) => {
                let target = w.collapsed_path.clone();
                let mut outcome = w.finish()?;
                if outcome.files.is_empty() {
                    File::create(&target)?;
                    outcome.files.push(target);
                }
                Ok(outcome)
            }
            Self::One {
                writer,
                bytes,
                record_bytes,
            } => {
                let final_path = writer.final_path().to_path_buf();
                writer.finish()?;
                Ok(ShardWriteOutcome {
                    source_spec: final_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or_default()
                        .to_string(),
                    record_count: if record_bytes > 0 { bytes / record_bytes } else { 0 },
                    files: vec![final_path],
                    shard_stride: None,
                    shard_count: None,
                })
            }
        }
    }
}

impl Write for FacetWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Series(w) => w.write(buf),
            Self::One { writer, bytes, .. } => {
                let n = writer.write(buf)?;
                *bytes += n as u64;
                Ok(n)
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Series(w) => w.flush(),
            Self::One { writer, .. } => writer.flush(),
        }
    }
}

/// The shard cap that applies to a step's output, if any.
///
/// **Only a facet is capped.** A cache artifact — a shuffle
/// permutation, a sorted-ordinal run, a sketch — is consumed by a
/// later step that maps one file, and it is not something any
/// declaration describes. Splitting one would break the next step to
/// no benefit, since nothing outside the run ever stores or moves it.
///
/// Returns the governor's `shardsize` for anything written outside the
/// workspace cache, and `None` for anything inside it.
pub fn cap_for_output(governor: &super::resource::ResourceGovernor, workspace: &Path, output: &Path) -> Option<u64> {
    let cache = workspace.join(".cache");
    // Compare lexically rather than canonicalizing: the output does
    // not exist yet, so `canonicalize` would fail on the very path
    // being asked about.
    if output.starts_with(&cache) {
        return None;
    }
    governor.current("shardsize")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/test-tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// Write `count` fvec records of `dim`, one `write_all` per field —
    /// the shape the extract loops actually use.
    fn write_records(w: &mut ShardingWriter, dim: usize, count: usize, first: usize) {
        for i in first..first + count {
            w.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for d in 0..dim {
                w.write_all(&((i * dim + d) as f32).to_le_bytes()).unwrap();
            }
        }
    }

    fn read_records(path: &Path, dim: usize) -> Vec<Vec<f32>> {
        let data = std::fs::read(path).unwrap();
        let stride = 4 + dim * 4;
        assert_eq!(data.len() % stride, 0, "{} is not whole records", path.display());
        (0..data.len() / stride)
            .map(|r| {
                let at = r * stride;
                assert_eq!(
                    i32::from_le_bytes(data[at..at + 4].try_into().unwrap()),
                    dim as i32,
                    "record {r} of {} has a wrong dim header",
                    path.display()
                );
                (0..dim)
                    .map(|d| {
                        let e = at + 4 + d * 4;
                        f32::from_le_bytes(data[e..e + 4].try_into().unwrap())
                    })
                    .collect()
            })
            .collect()
    }

    /// **The stream splits at record boundaries.** Every shard is a
    /// whole number of records with an intact dim header — which is
    /// the property that lets the write loops stay byte-oriented.
    #[test]
    fn shards_split_on_whole_records() {
        let tmp = tmpdir();
        let path = tmp.path().join("base.fvecs");
        let mut w = ShardingWriter::new(&path, 4 + 4 * 4, 10).unwrap();
        write_records(&mut w, 4, 25, 0);
        let out = w.finish().unwrap();

        assert_eq!(out.shard_count, Some(3));
        assert_eq!(out.shard_stride, Some(10));
        assert_eq!(out.record_count, 25);
        assert_eq!(out.source_spec, "base__NNNN.fvecs");

        let counts: Vec<usize> = out.files.iter().map(|f| read_records(f, 4).len()).collect();
        assert_eq!(counts, [10, 10, 5], "full, full, remainder");
    }

    /// **The concatenation is the original stream**, value for value.
    /// A split that lost or duplicated a record would still produce
    /// whole records, so the content is checked and not just the shape.
    #[test]
    fn the_shards_concatenate_back_to_what_was_written() {
        let tmp = tmpdir();
        let sharded = tmp.path().join("a.fvecs");
        let mut w = ShardingWriter::new(&sharded, 4 + 3 * 4, 7).unwrap();
        write_records(&mut w, 3, 30, 0);
        let out = w.finish().unwrap();

        let got: Vec<Vec<f32>> = out.files.iter().flat_map(|f| read_records(f, 3)).collect();
        let want: Vec<Vec<f32>> = (0..30)
            .map(|i| (0..3).map(|d| (i * 3 + d) as f32).collect())
            .collect();
        assert_eq!(got, want);
    }

    /// **However the caller chunks its writes, the split lands in the
    /// same place.** One call per record, one per element, and one for
    /// everything at once must all produce identical shards.
    #[test]
    fn the_split_is_independent_of_how_the_caller_chunks_writes() {
        let tmp = tmpdir();
        let dim = 4usize;
        let record = 4 + dim * 4;

        // One write_all per field.
        let a = tmp.path().join("a.fvecs");
        let mut w = ShardingWriter::new(&a, record as u64, 6).unwrap();
        write_records(&mut w, dim, 20, 0);
        let out_a = w.finish().unwrap();

        // One write_all for the entire stream.
        let b = tmp.path().join("b.fvecs");
        let mut whole = Vec::new();
        for i in 0..20usize {
            whole.extend_from_slice(&(dim as i32).to_le_bytes());
            for d in 0..dim {
                whole.extend_from_slice(&((i * dim + d) as f32).to_le_bytes());
            }
        }
        let mut w = ShardingWriter::new(&b, record as u64, 6).unwrap();
        w.write_all(&whole).unwrap();
        let out_b = w.finish().unwrap();

        // Byte-identical shards, both times.
        assert_eq!(out_a.shard_count, out_b.shard_count);
        for (fa, fb) in out_a.files.iter().zip(out_b.files.iter()) {
            assert_eq!(
                std::fs::read(fa).unwrap(),
                std::fs::read(fb).unwrap(),
                "{} vs {}",
                fa.display(),
                fb.display()
            );
        }
    }

    /// A write that straddles a boundary is split rather than
    /// overflowing the shard — the case a single large `write_all`
    /// hits on every seam.
    #[test]
    fn a_straddling_write_is_split_not_overflowed() {
        let tmp = tmpdir();
        let path = tmp.path().join("a.fvecs");
        // 8-byte records, 2 per shard: a 40-byte write crosses two
        // seams in one call.
        let mut w = ShardingWriter::new(&path, 8, 2).unwrap();
        w.write_all(&[7u8; 40]).unwrap();
        let out = w.finish().unwrap();

        assert_eq!(out.shard_count, Some(3), "40 bytes at 16 per shard");
        let sizes: Vec<u64> = out.files.iter().map(|f| std::fs::metadata(f).unwrap().len()).collect();
        assert_eq!(sizes, [16, 16, 8]);
        assert!(sizes.iter().all(|s| *s <= 16), "no shard exceeds its cap");
    }

    /// A run that fits in one shard collapses to the unsharded name
    /// (SH-83), so a facet that happened to fit reads like one written
    /// before the cap existed.
    #[test]
    fn a_single_shard_run_collapses() {
        let tmp = tmpdir();
        let path = tmp.path().join("base.fvecs");
        let mut w = ShardingWriter::new(&path, 20, 1000).unwrap();
        write_records(&mut w, 4, 25, 0);
        let out = w.finish().unwrap();

        assert!(!out.is_series());
        assert_eq!(out.source_spec, "base.fvecs");
        assert_eq!(out.record_count, 25);
        assert!(path.exists());
        assert!(!tmp.path().join("base__0000.fvecs").exists());
        assert_eq!(read_records(&path, 4).len(), 25);
    }

    /// An exactly-full single shard still collapses; one record more
    /// is a series. The boundary either side.
    #[test]
    fn the_collapse_boundary_is_exact() {
        let tmp = tmpdir();
        let a = tmp.path().join("a.fvecs");
        let mut w = ShardingWriter::new(&a, 20, 10).unwrap();
        write_records(&mut w, 4, 10, 0);
        assert!(!w.finish().unwrap().is_series(), "exactly full is one file");

        let b = tmp.path().join("b.fvecs");
        let mut w = ShardingWriter::new(&b, 20, 10).unwrap();
        write_records(&mut w, 4, 11, 0);
        let out = w.finish().unwrap();
        assert_eq!(out.shard_count, Some(2));
        assert_eq!(read_records(&out.files[1], 4).len(), 1);
    }

    /// Nothing written is no files at all, rather than an empty shard.
    #[test]
    fn writing_nothing_produces_no_files() {
        let tmp = tmpdir();
        let path = tmp.path().join("base.fvecs");
        let out = ShardingWriter::new(&path, 20, 10).unwrap().finish().unwrap();
        assert_eq!(out.record_count, 0);
        assert!(out.files.is_empty());
        assert!(!path.exists());
    }

    /// Shards are invisible until they are complete: a reader
    /// listing the directory mid-write sees finished shards and no
    /// partial one under its final name.
    #[test]
    fn shards_are_invisible_until_complete() {
        let tmp = tmpdir();
        let path = tmp.path().join("base.fvecs");
        let mut w = ShardingWriter::new(&path, 20, 10).unwrap();
        write_records(&mut w, 4, 15, 0);
        w.flush().unwrap();

        // Shard 0 is done and renamed; shard 1 is still a temp.
        assert!(tmp.path().join("base__0000.fvecs").exists());
        assert!(!tmp.path().join("base__0001.fvecs").exists());
        let _ = w.finish().unwrap();
        assert!(tmp.path().join("base__0001.fvecs").exists());
    }

    /// An abandoned write leaves no partial shard behind under a name
    /// anything would try to read.
    #[test]
    fn an_abandoned_write_leaves_no_partial_shard() {
        let tmp = tmpdir();
        let path = tmp.path().join("base.fvecs");
        {
            let mut w = ShardingWriter::new(&path, 20, 10).unwrap();
            write_records(&mut w, 4, 15, 0);
            w.flush().unwrap();
            // dropped without finish()
        }
        assert!(tmp.path().join("base__0000.fvecs").exists(), "complete shards stay");
        let leftovers: Vec<String> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .filter(|n| n.contains("0001"))
            .collect();
        assert!(leftovers.is_empty(), "no partial shard 1: {leftovers:?}");
    }

    /// A zero record size or stride names no layout and is refused
    /// when the writer opens.
    #[test]
    fn a_degenerate_shape_is_refused() {
        let tmp = tmpdir();
        assert!(ShardingWriter::new(&tmp.path().join("a.fvecs"), 0, 10).is_err());
        assert!(ShardingWriter::new(&tmp.path().join("a.fvecs"), 20, 0).is_err());
    }
}


#[cfg(test)]
mod facet_writer_tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/test-tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// **Writing nothing still leaves the file.** The single-file
    /// writer this replaces always created one, and steps downstream
    /// check their inputs by existence — an extract over an empty
    /// range produces an empty facet, not a missing one.
    #[test]
    fn an_empty_output_is_still_a_file() {
        for cap in [None, Some(1_000_000_000_000u64), Some(200)] {
            let tmp = tmpdir();
            let path = tmp.path().join("base.fvecs");
            let out = FacetWriter::open(&path, 20, cap).unwrap().finish().unwrap();

            assert!(path.exists(), "cap {cap:?}: the file must exist");
            assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
            assert_eq!(out.record_count, 0);
            assert_eq!(out.files, vec![path], "cap {cap:?}");
            assert!(!out.is_series());
        }
    }

    /// No cap means the single-file path, byte-identical to what the
    /// plain writer produced.
    #[test]
    fn without_a_cap_the_output_is_one_file() {
        let tmp = tmpdir();
        let path = tmp.path().join("base.fvecs");
        let mut w = FacetWriter::open(&path, 20, None).unwrap();
        assert!(!w.is_sharded());
        w.write_all(&[1u8; 100]).unwrap();
        let out = w.finish().unwrap();

        assert_eq!(out.record_count, 5, "100 bytes of 20-byte records");
        assert!(!out.is_series());
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 100);
    }

    /// A cap the output exceeds produces a series; one it does not
    /// produces the plain file. Same call, decided by the data.
    #[test]
    fn the_cap_decides_the_layout_not_the_call_site() {
        let tmp = tmpdir();

        let big = tmp.path().join("big.fvecs");
        let mut w = FacetWriter::open(&big, 20, Some(200)).unwrap();
        w.write_all(&[1u8; 500]).unwrap();
        let out = w.finish().unwrap();
        assert_eq!(out.shard_count, Some(3), "500 bytes at 200 per shard");
        assert_eq!(out.source_spec, "big__NNNN.fvecs");

        let small = tmp.path().join("small.fvecs");
        let mut w = FacetWriter::open(&small, 20, Some(200)).unwrap();
        w.write_all(&[1u8; 100]).unwrap();
        let out = w.finish().unwrap();
        assert!(!out.is_series());
        assert_eq!(out.source_spec, "small.fvecs");
        assert!(small.exists());
    }

    /// An unusable record size falls back to one file rather than
    /// refusing: a caller that cannot state a record size still needs
    /// its output written.
    #[test]
    fn an_unknown_record_size_falls_back_to_one_file() {
        let tmp = tmpdir();
        let path = tmp.path().join("base.fvecs");
        let w = FacetWriter::open(&path, 0, Some(200)).unwrap();
        assert!(!w.is_sharded());
        w.finish().unwrap();
        assert!(path.exists());
    }
}
