// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The `IDXFOR__` offset index each published
//! variable-length file ships with, and its merkle reference where the
//! dataset is merkled.
//!
//! A remote reader opens a variable-length file by ordinal through its
//! index; without one, finding its records means downloading the file
//! whole, which an open refuses to do behind the caller
//! ([`IoError::OffsetIndexUnavailable`](crate::io::IoError::OffsetIndexUnavailable)).
//! So every publishing path — `vectordata push`, `veks publish`,
//! [`CatalogTree::write`](crate::catalog::generate::CatalogTree::write),
//! `vectordata datasets derive` and the `generate vvec-index` pipeline
//! step — brings the index current here, before it ships.
//!
//! A file is indexed when its extension names the variable-length
//! container (`.i32vvec`, `.fvvecs`, …). The bare `.ivec` is not: it is
//! also the uniform ground-truth format.

use std::path::{Path, PathBuf};

use veks_datatypes::{Container, ElementType};

/// What [`prepare`] found for one variable-length file, and did (or, in
/// a dry run, would do).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedIndex {
    /// The variable-length file.
    pub data: PathBuf,
    /// Its `IDXFOR__` sidecar.
    pub index: PathBuf,
    /// Whether the sidecar was written (would be, in a dry run).
    pub index_written: bool,
    /// The sidecar's merkle reference, when the file has one: a dataset
    /// that merkles its files merkles their indexes too.
    pub mref: Option<PathBuf>,
    /// Whether the reference was written (would be, in a dry run).
    pub mref_written: bool,
}

/// Whether `name` is a file [`prepare`] indexes: its extension names the
/// variable-length container, and it is not a sidecar or a name the
/// publish filters exclude.
pub fn is_indexed_file(name: &str) -> bool {
    if name.starts_with("IDXFOR__") || crate::filters::is_derived_sidecar(name) || crate::filters::is_excluded_file(name)
    {
        return false;
    }
    let ext = name.rsplit('.').next().unwrap_or("");
    matches!(ElementType::parse_extension(ext), Some((_, Container::Vvec)))
}

/// Every file under `root` that [`is_indexed_file`], skipping the
/// directories the publish filters exclude, in path order.
pub fn indexed_files_under(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            if path.is_dir() {
                if !crate::filters::is_excluded_dir(&name) {
                    dirs.push(path);
                }
            } else if path.is_file() && is_indexed_file(&name) {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Bring the `IDXFOR__` index of the local variable-length file `data`
/// current, and, when `data` has a `.mref` beside it, the index's own
/// `.mref`, at the same chunk size. Idempotent: what is current is left
/// as it is. With `dry_run`, nothing is written and the result says what
/// would be.
///
/// `progress` hears the bytes walked or hashed so far, for a large file.
/// A file that is not a whole variable-length file — empty, or not
/// ending at a record boundary — is an error, not something to publish.
///
/// ```
/// use vectordata::vvec_index::prepare;
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let data = dir.path().join("metadata_results.ivvec");
/// # let mut f = std::fs::File::create(&data).unwrap();
/// # for v in [vec![4i32, 7], vec![], vec![1]] { f.write_all(&(v.len() as i32).to_le_bytes()).unwrap(); for x in v { f.write_all(&x.to_le_bytes()).unwrap(); } }
/// # drop(f);
/// let planned = prepare(&data, true, &mut |_| {}).unwrap();   // a dry run says what it would write
/// assert!(planned.index_written && !planned.index.exists());
/// let done = prepare(&data, false, &mut |_| {}).unwrap();
/// assert!(done.index.ends_with("IDXFOR__metadata_results.ivvec.i32") && done.index.is_file());
/// assert!(!prepare(&data, false, &mut |_| {}).unwrap().index_written); // current: left as it is
/// ```
pub fn prepare(data: &Path, dry_run: bool, progress: &mut dyn FnMut(u64)) -> Result<PublishedIndex, String> {
    let shown = data.display();
    let index = veks_io::reader::vvec_index_path(data).map_err(|e| format!("{shown}: {e}"))?;
    let index_written = if dry_run {
        !veks_io::reader::vvec_index_is_current(data).map_err(|e| format!("{shown}: {e}"))?
    } else {
        veks_io::reader::index_vvec_file(data, progress).map_err(|e| format!("indexing {shown}: {e}"))?.written
    };

    let data_mref = with_suffix(data, ".mref");
    if !data_mref.is_file() {
        return Ok(PublishedIndex { data: data.to_path_buf(), index, index_written, mref: None, mref_written: false });
    }
    let mref = with_suffix(&index, ".mref");
    let mref_current = !index_written && is_no_older(&mref, &index);
    if mref_current || dry_run {
        return Ok(PublishedIndex { data: data.to_path_buf(), index, index_written, mref: Some(mref), mref_written: !mref_current });
    }
    let chunk_size = crate::merkle::MerkleRef::load(&data_mref)
        .map_err(|e| format!("reading {}: {e}", data_mref.display()))?
        .shape()
        .chunk_size;
    let tree = crate::merkle::MerkleRef::from_path_with_progress(&index, chunk_size, progress)
        .map_err(|e| format!("hashing {}: {e}", index.display()))?;
    crate::atomic_file::replace(&mref, &tree.to_bytes()).map_err(|e| format!("writing {}: {e}", mref.display()))?;
    Ok(PublishedIndex { data: data.to_path_buf(), index, index_written, mref: Some(mref), mref_written: true })
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// Whether `path` exists and is no older than `than`.
fn is_no_older(path: &Path, than: &Path) -> bool {
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    matches!((mtime(path), mtime(than)), (Some(a), Some(b)) if a >= b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_vvec(path: &Path, records: &[&[i32]]) {
        let mut bytes = Vec::new();
        for r in records {
            bytes.extend_from_slice(&(r.len() as i32).to_le_bytes());
            for x in *r {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// **Only the variable-length container is indexed**: not the bare
    /// `.ivec`, which is also uniform ground truth, nor a sidecar.
    #[test]
    fn only_variable_length_files_are_indexed() {
        for name in ["metadata_results.ivvec", "m.i32vvecs", "labels.u8vvec", "f.fvvec"] {
            assert!(is_indexed_file(name), "{name}");
        }
        for name in ["neighbor_indices.ivec", "base.fvec", "IDXFOR__m.ivvec.i32", "m.ivvec.mref", ".m.ivvec", "dataset.yaml"] {
            assert!(!is_indexed_file(name), "{name}");
        }
    }

    /// **A merkled file's index is merkled too**, at the file's chunk
    /// size, and both are left alone once current.
    #[test]
    fn a_merkled_files_index_gets_a_reference() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("m.ivvec");
        write_vvec(&data, &[&[1, 2], &[], &[3]]);
        let data_tree = crate::merkle::MerkleRef::from_content(&std::fs::read(&data).unwrap(), 4096);
        std::fs::write(with_suffix(&data, ".mref"), data_tree.to_bytes()).unwrap();

        let done = prepare(&data, false, &mut |_| {}).unwrap();
        let mref = done.mref.clone().unwrap();
        assert!(done.index_written && done.mref_written);
        let tree = crate::merkle::MerkleRef::load(&mref).unwrap();
        assert_eq!(tree.shape().chunk_size, 4096);
        assert_eq!(tree.root_hash(), crate::merkle::MerkleRef::from_content(&std::fs::read(&done.index).unwrap(), 4096).root_hash());

        let again = prepare(&data, false, &mut |_| {}).unwrap();
        assert!(!again.index_written && !again.mref_written, "{again:?}");
        let planned = prepare(&data, true, &mut |_| {}).unwrap();
        assert!(!planned.index_written && !planned.mref_written, "{planned:?}");
    }

    /// **A file with no reference gets none**: an unmerkled dataset's
    /// index ships without one.
    #[test]
    fn an_unmerkled_files_index_gets_no_reference() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("m.ivvec");
        write_vvec(&data, &[&[5]]);
        let done = prepare(&data, false, &mut |_| {}).unwrap();
        assert_eq!(done.mref, None);
        assert!(!with_suffix(&done.index, ".mref").exists());
    }

    /// **The walk finds every indexed file, and skips excluded
    /// directories.**
    #[test]
    fn indexed_files_are_found_under_a_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("profiles/default")).unwrap();
        std::fs::create_dir_all(root.join(".cache")).unwrap();
        for p in ["a.ivvec", "profiles/default/b.u8vvec", ".cache/c.ivvec", "gt.ivec"] {
            write_vvec(&root.join(p), &[&[1]]);
        }
        assert_eq!(indexed_files_under(root), vec![root.join("a.ivvec"), root.join("profiles/default/b.u8vvec")]);
    }
}
