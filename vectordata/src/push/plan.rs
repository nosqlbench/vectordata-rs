// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Source-mode detection, known-good validation, and content scanning
//! for `vectordata push`.
//!
//! Push transfers a directory subtree (the publish root) — it does not
//! re-resolve the profile/facet graph. The source mode picks how
//! strictly we validate before shipping; what gets shipped is always
//! "every content file under the root", with one `SHA256SUMS` per
//! directory level.
//!
//! See `docs/design/push-command.md` — *The three source modes*.

use std::path::Path;

use super::checksums::{is_sentinel, CHECKSUMS_FILE};

/// Which shape of source we are pushing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceMode {
    /// A structured dataset directory (`dataset.yaml` / `dataset.json`).
    Structured,
    /// A legacy `knn_entries.yaml` catalog map.
    Catalog,
    /// An ad-hoc directory of arbitrary files (`--raw`).
    Raw,
    /// A whole publish hierarchy whose file set was selected by a
    /// producer (e.g. `veks publish`) and injected directly. The caller
    /// owns selection and validation; push owns transfer/provenance.
    Hierarchy,
}

impl SourceMode {
    pub fn label(self) -> &'static str {
        match self {
            SourceMode::Structured => "structured dataset (dataset.yaml)",
            SourceMode::Catalog => "catalog map (knn_entries.yaml)",
            SourceMode::Raw => "ad-hoc directory (--raw)",
            SourceMode::Hierarchy => "publish hierarchy (delegated)",
        }
    }
}

/// Detect the source mode from the directory contents. `--raw` is
/// required to push an unstructured directory — we refuse to silently
/// ship bytes that match no known shape.
pub fn detect_mode(root: &Path, raw: bool) -> Result<SourceMode, String> {
    let has_dataset = root.join("dataset.yaml").is_file() || root.join("dataset.json").is_file();
    let has_knn = root.join("knn_entries.yaml").is_file();
    if raw {
        return Ok(SourceMode::Raw);
    }
    if has_dataset {
        Ok(SourceMode::Structured)
    } else if has_knn {
        Ok(SourceMode::Catalog)
    } else {
        Err(format!(
            "no dataset.yaml or knn_entries.yaml in {}; pass --raw to push an ad-hoc directory",
            root.display()
        ))
    }
}

/// Known-good validation for the detected mode. Skippable with
/// `--no-check`, but binding/checksum/provenance rules in the
/// orchestrator are never skipped.
pub fn validate(mode: SourceMode, root: &Path) -> Result<(), String> {
    match mode {
        SourceMode::Structured => {
            let path = if root.join("dataset.yaml").is_file() {
                root.join("dataset.yaml")
            } else {
                root.join("dataset.json")
            };
            let cfg = crate::dataset::config::DatasetConfig::load(&path)
                .map_err(|e| format!("dataset config does not parse: {e}"))?;
            let mut missing = Vec::new();
            if cfg.is_zero_vector_free() != Some(true) {
                missing.push("is_zero_vector_free: true");
            }
            if cfg.is_duplicate_vector_free() != Some(true) {
                missing.push("is_duplicate_vector_free: true");
            }
            if !missing.is_empty() {
                return Err(format!(
                    "dataset is not in a publishable known-good state — missing/!true attributes: {}",
                    missing.join(", ")
                ));
            }
            Ok(())
        }
        SourceMode::Catalog => {
            let path = root.join("knn_entries.yaml");
            let entries = crate::knn_entries::KnnEntries::load(&path)?;
            // Best-effort: every referenced facet path must exist on disk.
            for (key, e) in &entries.entries {
                for (facet, rel) in [("base", &e.base), ("query", &e.query), ("gt", &e.gt)] {
                    if !root.join(rel).is_file() {
                        return Err(format!(
                            "knn_entries entry '{key}' {facet} file is missing: {rel}"
                        ));
                    }
                }
            }
            Ok(())
        }
        SourceMode::Raw => Ok(()),
        // A producer-injected hierarchy was validated by the caller
        // (e.g. veks publish ran its check suite) before delegation.
        SourceMode::Hierarchy => Ok(()),
    }
}

/// The result of scanning the publish root for content.
#[derive(Debug, Clone, Default)]
pub struct Scan {
    /// Directories (relative, forward-slashed; `""` is the root) that
    /// hold at least one content file — each needs a `SHA256SUMS`.
    pub content_dirs: Vec<String>,
    /// Every content file, relative to the root, forward-slashed.
    pub files: Vec<String>,
}

impl Scan {
    /// Build a scan from an externally-selected set of files (relative,
    /// forward-slashed) — the producer-driven path where a caller like
    /// `veks` has already decided exactly what is publishable. Content
    /// directories are derived from the files. Any sentinel paths in the
    /// input are dropped so the caller can't accidentally enqueue them.
    pub fn from_files(mut files: Vec<String>) -> Scan {
        files.retain(|f| {
            let name = f.rsplit('/').next().unwrap_or(f);
            !is_sentinel(name)
        });
        files.sort();
        files.dedup();
        let mut content_dirs: Vec<String> = files
            .iter()
            .map(|f| match f.rfind('/') {
                Some(i) => f[..i].to_string(),
                None => String::new(),
            })
            .collect();
        content_dirs.sort();
        content_dirs.dedup();
        Scan { content_dirs, files }
    }
}

/// Recursively collect content files and the directories that hold
/// them. Sentinels (`SHA256SUMS`, `.publish_url`, `pushlog.jsonl`) are
/// excluded — they are managed by the orchestrator, not treated as
/// payload.
pub fn scan(root: &Path) -> Result<Scan, String> {
    let mut files = Vec::new();
    let mut content_dirs = Vec::new();
    walk(root, root, &mut files, &mut content_dirs)
        .map_err(|e| format!("scanning {}: {e}", root.display()))?;
    files.sort();
    content_dirs.sort();
    content_dirs.dedup();
    Ok(Scan { content_dirs, files })
}

fn walk(
    root: &Path,
    dir: &Path,
    files: &mut Vec<String>,
    content_dirs: &mut Vec<String>,
) -> std::io::Result<()> {
    let mut had_content = false;
    let mut subdirs = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if ft.is_dir() {
            subdirs.push(entry.path());
        } else if ft.is_file() && !is_sentinel(&name) {
            had_content = true;
            files.push(rel_of(root, &entry.path()));
        }
    }
    if had_content {
        content_dirs.push(rel_of(root, dir));
    }
    subdirs.sort();
    for sd in subdirs {
        walk(root, &sd, files, content_dirs)?;
    }
    Ok(())
}

/// Forward-slashed path of `path` relative to `root` (`""` for root).
fn rel_of(root: &Path, path: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(r) => r.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/"),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

/// The `SHA256SUMS` key for a content directory (`SHA256SUMS` at root,
/// `<dir>/SHA256SUMS` otherwise).
pub fn sums_key(dir_rel: &str) -> String {
    if dir_rel.is_empty() {
        CHECKSUMS_FILE.to_string()
    } else {
        format!("{dir_rel}/{CHECKSUMS_FILE}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("vd-plan-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn detect_modes() {
        let d = tmpdir("detect");
        assert!(detect_mode(&d, false).is_err());
        assert_eq!(detect_mode(&d, true).unwrap(), SourceMode::Raw);
        std::fs::write(d.join("knn_entries.yaml"), "x: {}").unwrap();
        assert_eq!(detect_mode(&d, false).unwrap(), SourceMode::Catalog);
        std::fs::write(d.join("dataset.yaml"), "name: x").unwrap();
        assert_eq!(detect_mode(&d, false).unwrap(), SourceMode::Structured);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn scan_collects_content_and_dirs_excluding_sentinels() {
        let d = tmpdir("scan");
        std::fs::write(d.join("dataset.yaml"), "name: x").unwrap();
        std::fs::write(d.join(".publish_url"), "file:///x/").unwrap();
        std::fs::write(d.join("pushlog.jsonl"), "").unwrap();
        std::fs::create_dir_all(d.join("profiles/1m")).unwrap();
        std::fs::write(d.join("profiles/1m/base.fvec"), b"v").unwrap();
        // an empty dir should not show up as a content dir
        std::fs::create_dir_all(d.join("profiles/empty")).unwrap();

        let scan = scan(&d).unwrap();
        assert!(scan.files.contains(&"dataset.yaml".to_string()));
        assert!(scan.files.contains(&"profiles/1m/base.fvec".to_string()));
        assert!(!scan.files.iter().any(|f| f.contains(".publish_url")));
        assert!(!scan.files.iter().any(|f| f.contains("pushlog.jsonl")));
        assert!(scan.content_dirs.contains(&"".to_string()));
        assert!(scan.content_dirs.contains(&"profiles/1m".to_string()));
        assert!(!scan.content_dirs.contains(&"profiles/empty".to_string()));
        std::fs::remove_dir_all(&d).ok();
    }
}
