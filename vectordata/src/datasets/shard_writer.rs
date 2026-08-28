// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Writing a facet across a series of shard files.
//!
//! Takes a stream of whole records and rolls over to the next file every
//! `stride` records, producing the **uniform** layout — the only form
//! this project generates (SH-35). The explicit form describes data
//! written elsewhere; nothing here emits it.
//!
//! Three properties the writer exists to guarantee:
//!
//! - **Atomic per file** (SH-37). Each shard is written to a temp and
//!   renamed, and the caller writes the declaration only after
//!   [`ShardWriter::finish`] returns — so a reader never meets a
//!   declaration promising shards whose files are not there.
//! - **Collapsing** (SH-83). A run that fits in one shard emits the
//!   *single-file* form. A pipeline that happened to fit must not leave
//!   behind a declaration older readers cannot open.
//! - **Deterministic** (SH-36). The same records and the same stride
//!   produce byte-identical shards; nothing here depends on timing,
//!   memory pressure, or output size.
//!
//! See `docs/design/srd-multifile-facet-shards.md`.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// What a completed write produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShardOutcome {
    /// The files written, in ordinal order.
    pub files: Vec<PathBuf>,
    /// Records written across all of them.
    pub records: u64,
    /// Ordinals per shard, for every shard but the last.
    pub stride: u64,
    /// Whether the output collapsed to the single-file form (SH-83).
    pub collapsed: bool,
}

impl ShardOutcome {
    /// The `source` string for the declaration this output needs.
    ///
    /// The `NNNN` pattern for a series; the plain filename when the
    /// output collapsed.
    pub fn source_spec(&self) -> String {
        let name = |p: &PathBuf| {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string()
        };
        match (self.collapsed, self.files.first()) {
            (true, Some(f)) => name(f),
            (false, Some(f)) => {
                // Replace the shard field with the literal token, so the
                // declaration says what it means rather than naming one
                // file (SH-47).
                let n = name(f);
                match n.rfind("__") {
                    Some(i) => format!("{}__{}{}", &n[..i], "NNNN", &n[i + 6..]),
                    None => n,
                }
            }
            _ => String::new(),
        }
    }

    /// Whether the declaration needs `shard_stride`/`shard_count`.
    pub fn is_series(&self) -> bool {
        !self.collapsed
    }

    /// Shard count, for the declaration.
    pub fn shard_count(&self) -> u32 {
        self.files.len() as u32
    }
}

/// Writes a record stream across shard files.
pub(crate) struct ShardWriter {
    dir: PathBuf,
    basename: String,
    ext: String,
    stride: u64,
    /// Records in the shard currently open.
    in_shard: u64,
    /// Records across all shards.
    total: u64,
    /// Next shard index to open.
    next_index: u32,
    /// The shard being written: its temp path, final path, and handle.
    open: Option<(PathBuf, PathBuf, fs::File)>,
    finished: Vec<(PathBuf, PathBuf)>,
}

impl ShardWriter {
    /// Open a writer that rolls over every `stride` records.
    ///
    /// `stride` must be non-zero: a zero stride names no layout, and
    /// silently treating it as "one file" would make the declaration
    /// disagree with the request.
    pub(crate) fn new(dir: &Path, basename: &str, ext: &str, stride: u64) -> io::Result<Self> {
        if stride == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shard stride must be greater than zero",
            ));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            basename: basename.to_string(),
            ext: ext.to_string(),
            stride,
            in_shard: 0,
            total: 0,
            next_index: 0,
            open: None,
            finished: Vec::new(),
        })
    }

    /// The final path of shard `i` — four digits, always (SH-2).
    fn shard_path(&self, i: u32) -> PathBuf {
        self.dir
            .join(format!("{}__{:04}.{}", self.basename, i, self.ext))
    }

    /// The single-file path this output collapses to (SH-83).
    fn collapsed_path(&self) -> PathBuf {
        self.dir.join(format!("{}.{}", self.basename, self.ext))
    }

    fn open_next(&mut self) -> io::Result<()> {
        let final_path = self.shard_path(self.next_index);
        let mut temp = final_path.clone().into_os_string();
        // `.partial` rather than an invented suffix: it is already on
        // the publish-exclusion list, so a temp surviving a hard kill
        // cannot be shipped as if it were a shard.
        temp.push(".partial");
        let temp = PathBuf::from(temp);
        let file = fs::File::create(&temp)?;
        self.open = Some((temp, final_path, file));
        self.next_index += 1;
        self.in_shard = 0;
        Ok(())
    }

    /// Close the shard in progress, leaving its temp in place for
    /// [`Self::finish`] to rename.
    fn close_current(&mut self) -> io::Result<()> {
        if let Some((temp, final_path, mut file)) = self.open.take() {
            file.flush()?;
            file.sync_all()?;
            drop(file);
            self.finished.push((temp, final_path));
        }
        Ok(())
    }

    /// Append one whole record.
    ///
    /// Rollover happens **between** records, never inside one: a record
    /// spans no shard boundary (SH-13), and because the caller hands
    /// whole records that is true by construction rather than by a
    /// check.
    pub(crate) fn write_record(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.open.is_none() || self.in_shard == self.stride {
            self.close_current()?;
            self.open_next()?;
        }
        let (_, _, file) = self.open.as_mut().expect("a shard is open");
        file.write_all(bytes)?;
        self.in_shard += 1;
        self.total += 1;
        Ok(())
    }

    /// Finish, renaming every shard into place.
    ///
    /// Renames happen only here, so a partially-written series is never
    /// visible under a real shard name (SH-37, SH-40).
    pub(crate) fn finish(mut self) -> io::Result<ShardOutcome> {
        self.close_current()?;

        // A run that fits in one shard is spelled as a single file —
        // otherwise a pipeline that happened to fit leaves behind a
        // declaration older readers cannot open (SH-83, SH-4).
        let collapsed = self.finished.len() <= 1;
        let mut files = Vec::with_capacity(self.finished.len());
        for (i, (temp, final_path)) in self.finished.iter().enumerate() {
            let dest = if collapsed && i == 0 {
                self.collapsed_path()
            } else {
                final_path.clone()
            };
            fs::rename(temp, &dest)?;
            files.push(dest);
        }
        Ok(ShardOutcome {
            files,
            records: self.total,
            stride: self.stride,
            collapsed,
        })
    }
}

impl Drop for ShardWriter {
    /// Remove any temp left behind when a write is abandoned.
    ///
    /// Only temps: a shard renamed into place by `finish` is durable
    /// output, and a panic partway through must not take it with it.
    fn drop(&mut self) {
        if let Some((temp, _, _)) = self.open.take() {
            let _ = fs::remove_file(temp);
        }
        for (temp, _) in &self.finished {
            let _ = fs::remove_file(temp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// Write `n` records of `size` bytes, each filled with its index.
    fn write_n(w: &mut ShardWriter, n: usize, size: usize) {
        for i in 0..n {
            w.write_record(&vec![i as u8; size]).unwrap();
        }
    }

    fn names(o: &ShardOutcome) -> Vec<String> {
        o.files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect()
    }

    /// Rollover happens at exactly the stride, and the last shard holds
    /// the remainder (SH-35, SH-12).
    #[test]
    fn records_roll_over_at_exactly_the_stride() {
        let d = tmpdir();
        let mut w = ShardWriter::new(d.path(), "base_vectors", "fvec", 4).unwrap();
        write_n(&mut w, 10, 8);
        let out = w.finish().unwrap();

        assert_eq!(out.records, 10);
        assert_eq!(
            names(&out),
            vec![
                "base_vectors__0000.fvec",
                "base_vectors__0001.fvec",
                "base_vectors__0002.fvec"
            ]
        );
        let len = |p: &PathBuf| std::fs::metadata(p).unwrap().len();
        assert_eq!(len(&out.files[0]), 4 * 8, "full shard");
        assert_eq!(len(&out.files[1]), 4 * 8, "full shard");
        assert_eq!(len(&out.files[2]), 2 * 8, "the remainder");
    }

    /// **A run that fits in one shard emits the single-file form**
    /// (SH-83). Otherwise a pipeline that happened to fit leaves behind
    /// a declaration older readers cannot open.
    #[test]
    fn a_single_shard_output_collapses_to_one_file() {
        let d = tmpdir();
        let mut w = ShardWriter::new(d.path(), "base_vectors", "fvec", 100).unwrap();
        write_n(&mut w, 30, 8);
        let out = w.finish().unwrap();

        assert!(out.collapsed);
        assert_eq!(names(&out), vec!["base_vectors.fvec"]);
        assert!(
            !d.path().join("base_vectors__0000.fvec").exists(),
            "the sharded spelling must not be left behind"
        );
        assert_eq!(out.source_spec(), "base_vectors.fvec");
        assert!(!out.is_series());
    }

    /// Exactly one stride's worth still collapses — one shard is one
    /// shard however precisely it filled.
    #[test]
    fn an_exactly_full_single_shard_still_collapses() {
        let d = tmpdir();
        let mut w = ShardWriter::new(d.path(), "q", "fvec", 10).unwrap();
        write_n(&mut w, 10, 4);
        let out = w.finish().unwrap();
        assert!(out.collapsed);
        assert_eq!(names(&out), vec!["q.fvec"]);
    }

    /// One record past the stride is two shards, and the declaration
    /// says so.
    #[test]
    fn one_record_past_the_stride_is_a_series() {
        let d = tmpdir();
        let mut w = ShardWriter::new(d.path(), "q", "fvec", 10).unwrap();
        write_n(&mut w, 11, 4);
        let out = w.finish().unwrap();

        assert!(!out.collapsed);
        assert_eq!(out.shard_count(), 2);
        assert_eq!(out.source_spec(), "q__NNNN.fvec");
        assert_eq!(names(&out), vec!["q__0000.fvec", "q__0001.fvec"]);
    }

    /// The declaration names the **pattern**, not a file (SH-47).
    #[test]
    fn the_source_spec_is_the_pattern_not_a_filename() {
        let d = tmpdir();
        let mut w = ShardWriter::new(d.path(), "metadata_results", "ivvec", 2).unwrap();
        write_n(&mut w, 5, 4);
        let out = w.finish().unwrap();
        assert_eq!(out.source_spec(), "metadata_results__NNNN.ivvec");
        assert_eq!(out.shard_count(), 3);
    }

    /// **Shard indices are four digits** (SH-2), so lexicographic order
    /// is numeric order past nine.
    #[test]
    fn shard_indices_are_four_digits() {
        let d = tmpdir();
        let mut w = ShardWriter::new(d.path(), "x", "u8", 1).unwrap();
        write_n(&mut w, 12, 1);
        let out = w.finish().unwrap();
        assert_eq!(names(&out)[9], "x__0009.u8");
        assert_eq!(names(&out)[10], "x__0010.u8");
        let mut sorted = names(&out);
        sorted.sort();
        assert_eq!(sorted, names(&out), "lexicographic order is numeric order");
    }

    /// **Nothing appears under a real shard name until `finish`**
    /// (SH-37, SH-40): a reader must never meet a half-written series.
    #[test]
    fn shards_are_invisible_until_the_write_completes() {
        let d = tmpdir();
        let mut w = ShardWriter::new(d.path(), "base", "fvec", 2).unwrap();
        write_n(&mut w, 5, 4);
        // Mid-write: no final name exists yet.
        for i in 0..3 {
            assert!(
                !d.path().join(format!("base__{i:04}.fvec")).exists(),
                "shard {i} visible before finish"
            );
        }
        let out = w.finish().unwrap();
        assert_eq!(out.files.len(), 3);
        for f in &out.files {
            assert!(f.exists());
        }
    }

    /// An abandoned write leaves no temps behind.
    #[test]
    fn an_abandoned_write_cleans_up_after_itself() {
        let d = tmpdir();
        {
            let mut w = ShardWriter::new(d.path(), "base", "fvec", 2).unwrap();
            write_n(&mut w, 5, 4);
            // dropped without finish
        }
        let left: Vec<_> = std::fs::read_dir(d.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(left.is_empty(), "temps left behind: {left:?}");
    }

    /// **Deterministic** (SH-36): the same records and stride produce
    /// byte-identical shards.
    #[test]
    fn the_same_records_and_stride_produce_identical_shards() {
        let run = || {
            let d = tmpdir();
            let mut w = ShardWriter::new(d.path(), "base", "fvec", 3).unwrap();
            write_n(&mut w, 10, 6);
            let out = w.finish().unwrap();
            let bytes: Vec<Vec<u8>> = out
                .files
                .iter()
                .map(|f| std::fs::read(f).unwrap())
                .collect();
            (names(&out), bytes, d)
        };
        let (n1, b1, _d1) = run();
        let (n2, b2, _d2) = run();
        assert_eq!(n1, n2);
        assert_eq!(b1, b2);
    }

    /// A zero stride names no layout and is refused rather than
    /// silently treated as "one file".
    #[test]
    fn a_zero_stride_is_refused() {
        let d = tmpdir();
        assert!(ShardWriter::new(d.path(), "x", "fvec", 0).is_err());
    }

    /// Writing nothing produces nothing — not an empty shard.
    #[test]
    fn writing_no_records_produces_no_files() {
        let d = tmpdir();
        let w = ShardWriter::new(d.path(), "x", "fvec", 4).unwrap();
        let out = w.finish().unwrap();
        assert_eq!(out.records, 0);
        assert!(out.files.is_empty());
    }

    /// The concatenated shards are exactly the record stream — no
    /// record split across a boundary, none dropped, none doubled
    /// (SH-13).
    #[test]
    fn the_shards_concatenate_back_to_the_record_stream() {
        let d = tmpdir();
        let mut w = ShardWriter::new(d.path(), "x", "fvec", 3).unwrap();
        let records: Vec<Vec<u8>> = (0..10u8).map(|i| vec![i; 5]).collect();
        for r in &records {
            w.write_record(r).unwrap();
        }
        let out = w.finish().unwrap();
        let joined: Vec<u8> = out
            .files
            .iter()
            .flat_map(|f| std::fs::read(f).unwrap())
            .collect();
        let expected: Vec<u8> = records.concat();
        assert_eq!(joined, expected);
    }
}
