// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The catalog generator: find the publishable datasets
//! under a directory and write the catalog files that list them.
//!
//! Both `veks prepare catalog generate` and the pipeline's
//! `catalog generate` step (each dataset's `gen-catalog`) use it, so a
//! tree gets the same catalogs whichever wrote them. The rules
//! (SRD runtime catalogs §11, item 3):
//!
//! - **Discovery.** A directory holding `dataset.yaml` is a dataset, and
//!   nothing below it is searched. A directory marked `.do_not_catalog`
//!   is not searched at all, and gets no catalog file. Hidden,
//!   underscore and build directories are skipped. Only datasets marked
//!   `.publish` are listed.
//! - **Content.** Each dataset is read with
//!   [`DatasetConfig::load_and_resolve`], so sized profiles that wait on
//!   `variables.yaml` are listed, and every profile gains the views its
//!   per-profile pipeline steps produce.
//! - **Files.** Every directory from a dataset up to the scan root gets
//!   `<basename>.json` and `<basename>.yaml` listing the datasets below
//!   it, and a `knn_entries.yaml` for jvector, written first and deepest
//!   first so each catalog is newer than what it covers.
//! - **Paths.** Paths are logical: symlinks are not resolved, so a tree
//!   reached through one keeps the `.publish_url` above it.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use crate::dataset::{CatalogEntry, CatalogLayout, DatasetConfig};

/// Marks the top of a catalog hierarchy below a publish root.
pub const CATALOG_ROOT_FILE: &str = ".catalog_root";

/// Marks a directory whose datasets are listed in no catalog.
pub const DO_NOT_CATALOG_FILE: &str = ".do_not_catalog";

/// Marks a dataset directory as publishable, and so listed.
pub const PUBLISH_SENTINEL: &str = ".publish";

/// A publishable dataset the generator found.
#[derive(Debug, Clone)]
pub struct FoundDataset {
    yaml_path: PathBuf,
    config: DatasetConfig,
}

impl FoundDataset {
    /// Its `dataset.yaml`.
    pub fn yaml_path(&self) -> &Path {
        &self.yaml_path
    }

    /// Its description, resolved as the catalog lists it.
    pub fn config(&self) -> &DatasetConfig {
        &self.config
    }

    /// Its directory.
    pub fn dir(&self) -> &Path {
        self.yaml_path.parent().unwrap_or(Path::new("."))
    }

    /// The entry a catalog in `catalog_dir` lists for it: its
    /// `dataset.yaml` relative to that directory, and its layout.
    pub fn entry_in(&self, catalog_dir: &Path) -> CatalogEntry {
        let relative: PathBuf = self
            .yaml_path
            .strip_prefix(catalog_dir)
            .unwrap_or(&self.yaml_path)
            .components()
            .filter(|c| !matches!(c, Component::CurDir))
            .collect();
        let layout = CatalogLayout {
            // The dataset's own version travels into the catalog, so a
            // consumer too old for it can say so from the listing
            // rather than after fetching (V-13).
            format_version: self.config.format_version,
            profile_tags: self.config.profile_tags.clone(),
            attributes: self.config.attributes.clone(),
            profiles: self.config.profiles.clone(),
        };
        CatalogEntry::builder(self.config.name.clone())
            .descriptor(&crate::Location::from_path(relative), layout)
            .build()
            .expect("an entry with a descriptor always builds")
    }
}

/// What the generator is doing, for a progress display.
#[derive(Debug)]
pub enum GenerateEvent<'a> {
    /// Searching for datasets under this directory.
    Scanning(&'a Path),
    /// A publishable dataset was found.
    Found(&'a FoundDataset),
    /// A `dataset.yaml` could not be read; its dataset is not listed.
    Unreadable {
        /// The file.
        path: &'a Path,
        /// Why.
        error: &'a str,
    },
    /// A directory marked `.do_not_catalog` gets no catalog file.
    Marked(&'a Path),
    /// A directory's catalog files were written.
    Wrote {
        /// The directory.
        dir: &'a Path,
        /// How many datasets its catalog lists.
        entries: usize,
        /// The files written.
        files: &'a [PathBuf],
    },
}

/// The publishable datasets under a scan root, ready to be written into
/// catalogs.
#[derive(Debug, Clone)]
pub struct CatalogTree {
    scan_root: PathBuf,
    datasets: Vec<FoundDataset>,
}

impl CatalogTree {
    /// Find the publishable datasets under `scan_root`, which is taken
    /// as a logical path ([`logical_path`]).
    pub fn scan(scan_root: &Path, progress: &mut dyn FnMut(GenerateEvent)) -> Self {
        let scan_root = logical_path(scan_root);
        progress(GenerateEvent::Scanning(&scan_root));
        let mut datasets = Vec::new();
        walk(&scan_root, &mut datasets, progress);
        datasets.retain(|d| d.dir().join(PUBLISH_SENTINEL).exists());
        datasets.sort_by(|a, b| a.yaml_path.cmp(&b.yaml_path));
        for dataset in &datasets {
            progress(GenerateEvent::Found(dataset));
        }
        CatalogTree { scan_root, datasets }
    }

    /// The directory scanned.
    pub fn scan_root(&self) -> &Path {
        &self.scan_root
    }

    /// The publishable datasets found, ordered by path.
    pub fn datasets(&self) -> &[FoundDataset] {
        &self.datasets
    }

    /// Every directory a catalog belongs in: the scan root, and each
    /// directory from a dataset up to it.
    pub fn catalog_dirs(&self) -> BTreeSet<PathBuf> {
        let mut dirs = BTreeSet::from([self.scan_root.clone()]);
        for dataset in &self.datasets {
            let mut dir = dataset.dir().to_path_buf();
            while dir.starts_with(&self.scan_root) {
                dirs.insert(dir.clone());
                if !dir.pop() {
                    break;
                }
            }
        }
        dirs
    }

    /// The entries a catalog in `dir` lists: every dataset below it.
    pub fn entries_in(&self, dir: &Path) -> Vec<CatalogEntry> {
        self.datasets
            .iter()
            .filter(|d| d.yaml_path.starts_with(dir))
            .map(|d| d.entry_in(dir))
            .collect()
    }

    /// Write the catalog files of each of `dirs` that lists anything and
    /// is not marked `.do_not_catalog`, deepest first, so a parent's
    /// catalog is always newer than its children's. Returns the files
    /// written.
    pub fn write(
        &self,
        dirs: &BTreeSet<PathBuf>,
        basename: &str,
        progress: &mut dyn FnMut(GenerateEvent),
    ) -> Result<Vec<PathBuf>, String> {
        let base_url = crate::push::binding::read_binding(&self.scan_root)
            .ok()
            .flatten()
            .map(|(_, parsed)| parsed.url);
        let mut ordered: Vec<&PathBuf> = dirs.iter().collect();
        ordered.sort_by(|a, b| b.components().count().cmp(&a.components().count()).then_with(|| a.cmp(b)));
        let mut written = Vec::new();
        for dir in ordered {
            if dir.join(DO_NOT_CATALOG_FILE).exists() {
                progress(GenerateEvent::Marked(dir));
                continue;
            }
            let entries = self.entries_in(dir);
            if entries.is_empty() {
                continue;
            }
            let mut files = Vec::new();
            // knn_entries.yaml first: the catalog files must be newer
            // for the publish pre-flight's mtime checks.
            let knn = self.knn_entries_in(dir, base_url.as_deref());
            if !knn.is_empty() {
                files.push(write_file(&dir.join("knn_entries.yaml"), &knn)?);
            }
            let json = serde_json::to_string_pretty(&entries).map_err(|e| format!("JSON serialization failed: {e}"))?;
            files.push(write_file(&dir.join(format!("{basename}.json")), &json)?);
            let yaml = serde_yaml::to_string(&entries).map_err(|e| format!("YAML serialization failed: {e}"))?;
            files.push(write_file(&dir.join(format!("{basename}.yaml")), &yaml)?);
            progress(GenerateEvent::Wrote { dir, entries: entries.len(), files: &files });
            written.extend(files);
        }
        Ok(written)
    }

    /// The `knn_entries.yaml` (jvector's format) for `dir`: one
    /// `<dataset>:<profile>` entry for each profile below it that has
    /// base vectors, query vectors and neighbor indices, with paths
    /// relative to `dir`, and the publish URL as `_defaults.base_url`.
    /// Empty when no profile qualifies.
    pub fn knn_entries_in(&self, dir: &Path, base_url: Option<&str>) -> String {
        let mut out = String::new();
        for dataset in self.datasets.iter().filter(|d| d.yaml_path.starts_with(dir)) {
            let relative = dataset.dir().strip_prefix(dir).map(|r| r.to_string_lossy().into_owned()).unwrap_or_default();
            let prefix = if relative.is_empty() { String::new() } else { format!("{relative}/") };
            for (profile_name, profile) in &dataset.config.profiles.profiles {
                let path = |facet: &str| profile.view(facet).map(|v| v.path().to_string());
                let (Some(base), Some(query), Some(gt)) =
                    (path("base_vectors"), path("query_vectors"), path("neighbor_indices"))
                else {
                    continue;
                };
                // The key holds a ':', so it is quoted.
                out.push_str(&format!("\"{}:{profile_name}\":\n", dataset.config.name));
                out.push_str(&format!("  base: {prefix}{base}\n"));
                out.push_str(&format!("  query: {prefix}{query}\n"));
                out.push_str(&format!("  gt: {prefix}{gt}\n\n"));
            }
        }
        match (out.is_empty(), base_url) {
            (false, Some(url)) => format!("_defaults:\n  base_url: {}\n\n{out}", url.trim_end_matches('/')),
            _ => out,
        }
    }
}

fn write_file(path: &Path, content: &str) -> Result<PathBuf, String> {
    std::fs::write(path, content).map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    Ok(path.to_path_buf())
}

fn walk(dir: &Path, datasets: &mut Vec<FoundDataset>, progress: &mut dyn FnMut(GenerateEvent)) {
    if dir.join(DO_NOT_CATALOG_FILE).exists() {
        return;
    }
    let yaml_path = dir.join("dataset.yaml");
    if yaml_path.exists() {
        match DatasetConfig::load_and_resolve(&yaml_path) {
            Ok(mut config) => {
                let templates: Vec<_> = crate::dataset::collect_all_steps(&config)
                    .into_iter()
                    .filter(|s| s.per_profile)
                    .collect();
                config.profiles.derive_views_from_templates(&templates);
                datasets.push(FoundDataset { yaml_path, config });
            }
            Err(error) => progress(GenerateEvent::Unreadable { path: &yaml_path, error: &error }),
        }
        return;
    }
    let Ok(read_dir) = std::fs::read_dir(dir) else { return };
    let mut subdirs: Vec<PathBuf> = read_dir.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subdirs.sort();
    for subdir in subdirs {
        let name = subdir.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !crate::filters::is_excluded_dir(name) {
            walk(&subdir, datasets, progress);
        }
    }
}

/// Whether a dataset in `dataset_dir` is listed in no catalog under
/// `scan_root`: it, or a directory between it and the scan root, is
/// marked `.do_not_catalog`. The rule [`CatalogTree::scan`] follows, for
/// a check of which catalogs a tree should hold.
pub fn is_withheld(dataset_dir: &Path, scan_root: &Path) -> bool {
    let scan_root = logical_path(scan_root);
    logical_path(dataset_dir)
        .ancestors()
        .take_while(|d| d.starts_with(&scan_root))
        .any(|d| d.join(DO_NOT_CATALOG_FILE).exists())
}

/// The closest directory at or above `dir` marked `.catalog_root` or
/// bound by `.publish_url`: the top of the catalog hierarchy `dir` is in.
pub fn find_catalog_boundary(dir: &Path) -> Option<PathBuf> {
    ancestors_marked(dir, |d| d.join(CATALOG_ROOT_FILE).is_file() || d.join(crate::push::binding::PUBLISH_FILE).is_file())
}

/// The closest directory at or above `dir` marked `.catalog_root`.
pub fn find_catalog_root(dir: &Path) -> Option<PathBuf> {
    ancestors_marked(dir, |d| d.join(CATALOG_ROOT_FILE).is_file())
}

fn ancestors_marked(dir: &Path, marked: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    logical_path(dir).ancestors().find(|d| marked(d)).map(Path::to_path_buf)
}

/// `path` made absolute against the working directory, with `.` and
/// `..` resolved by name and symlinks left alone, so a tree reached
/// through a symlink stays the tree the user sees.
pub fn logical_path(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A publishable dataset at `dir` named `name`, with base, query and
    /// ground-truth facets.
    fn dataset(dir: &Path, name: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("dataset.yaml"),
            format!(
                "name: {name}\nattributes:\n  distance_function: L2\nprofiles:\n  default:\n    \
                 base_vectors: base.fvec\n    query_vectors: query.fvec\n    neighbor_indices: gt.ivec\n"
            ),
        )
        .unwrap();
        std::fs::write(dir.join(PUBLISH_SENTINEL), "").unwrap();
    }

    fn names(tree: &CatalogTree) -> Vec<&str> {
        tree.datasets().iter().map(|d| d.config().name.as_str()).collect()
    }

    /// **Only publishable datasets are listed, and hidden, underscore and
    /// build directories are not searched.**
    #[test]
    fn the_scan_finds_publishable_datasets_only() {
        let tmp = tempfile::tempdir().unwrap();
        dataset(&tmp.path().join("real"), "real");
        dataset(&tmp.path().join(".hidden"), "hidden");
        dataset(&tmp.path().join("target"), "target");
        dataset(&tmp.path().join("local"), "local");
        std::fs::remove_file(tmp.path().join("local").join(PUBLISH_SENTINEL)).unwrap();
        let tree = CatalogTree::scan(tmp.path(), &mut |_| {});
        assert_eq!(names(&tree), ["real"]);
    }

    /// **A dataset below a `.do_not_catalog` directory is in no catalog**,
    /// not even those above the marker: the rule the published catalogs
    /// were made with (SRD §11, item 3). The manual command used to list
    /// such datasets in the catalogs above.
    #[test]
    fn nothing_below_a_do_not_catalog_marker_is_listed() {
        let tmp = tempfile::tempdir().unwrap();
        dataset(&tmp.path().join("shown"), "shown");
        dataset(&tmp.path().join("private").join("inner"), "inner");
        std::fs::write(tmp.path().join("private").join(DO_NOT_CATALOG_FILE), "").unwrap();
        let tree = CatalogTree::scan(tmp.path(), &mut |_| {});
        assert_eq!(names(&tree), ["shown"]);
        assert!(is_withheld(&tmp.path().join("private").join("inner"), tmp.path()));
        assert!(!is_withheld(&tmp.path().join("shown"), tmp.path()));
        let dirs = tree.catalog_dirs();
        tree.write(&dirs, "catalog", &mut |_| {}).unwrap();
        assert!(!tmp.path().join("private").join("catalog.json").exists());
        assert!(!tmp.path().join("private").join("inner").join("catalog.json").exists());
        let top = std::fs::read_to_string(tmp.path().join("catalog.json")).unwrap();
        assert!(!top.contains("inner"), "{top}");
    }

    /// **Every directory from a dataset up to the scan root gets a
    /// catalog listing what is below it**, each entry pointing at its
    /// `dataset.yaml` relative to that catalog and carrying its layout.
    #[test]
    fn each_level_lists_the_datasets_below_it() {
        let tmp = tempfile::tempdir().unwrap();
        dataset(&tmp.path().join("group").join("a"), "a");
        dataset(&tmp.path().join("b"), "b");
        let tree = CatalogTree::scan(tmp.path(), &mut |_| {});
        let dirs = tree.catalog_dirs();
        let mut wrote = Vec::new();
        tree.write(&dirs, "catalog", &mut |e| {
            if let GenerateEvent::Wrote { dir, entries, .. } = e {
                wrote.push((dir.to_path_buf(), entries));
            }
        })
        .unwrap();
        let root = logical_path(tmp.path());
        let depth = |p: &Path| p.components().count();
        assert!(wrote.windows(2).all(|w| depth(&w[0].0) >= depth(&w[1].0)), "deepest first: {wrote:?}");
        assert_eq!(wrote.last().unwrap(), &(root.clone(), 2));

        let entries = crate::dataset::load_catalog(&root.join("group").join("catalog.json")).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].location(), "a/dataset.yaml");
        assert_eq!(entries[0].attributes().unwrap().distance_function.as_deref(), Some("L2"));
        assert_eq!(entries[0].dataset_type(), "dataset.yaml");
        let top = crate::dataset::load_catalog(&root.join("catalog.yaml")).unwrap();
        let mut listed: Vec<&str> = top.iter().map(|e| e.location()).collect();
        listed.sort();
        assert_eq!(listed, ["b/dataset.yaml", "group/a/dataset.yaml"]);
    }

    /// **`knn_entries.yaml` lists each profile with all three facets,
    /// relative to its directory, under the publish URL as its base**, and
    /// reads back through the catalog's own parser to the same files.
    #[test]
    fn knn_entries_name_the_files_under_the_publish_url() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".publish_url"), "s3://bucket/datasets/\n").unwrap();
        dataset(&tmp.path().join("ds"), "ds");
        let tree = CatalogTree::scan(tmp.path(), &mut |_| {});
        tree.write(&tree.catalog_dirs(), "catalog", &mut |_| {}).unwrap();
        let text = std::fs::read_to_string(tmp.path().join("knn_entries.yaml")).unwrap();
        assert_eq!(
            text,
            "_defaults:\n  base_url: s3://bucket/datasets\n\n\"ds:default\":\n  base: ds/base.fvec\n  \
             query: ds/query.fvec\n  gt: ds/gt.ivec\n\n"
        );
        let entries = crate::catalog::knn_entries::parse_knn_entries_yaml(&text, "/elsewhere").unwrap();
        let default = entries[0].profiles().profile("default").unwrap();
        assert_eq!(default.views["base_vectors"].path(), "s3://bucket/datasets/ds/base.fvec");
    }

    /// **Paths stay logical**: a tree reached through a symlink keeps the
    /// symlink in its paths, so the `.publish_url` above the link is the
    /// one found, not one above the link's target.
    #[cfg(unix)]
    #[test]
    fn a_tree_through_a_symlink_keeps_its_logical_path() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real");
        dataset(&real.join("ds"), "ds");
        let view = tmp.path().join("view");
        std::fs::create_dir_all(&view).unwrap();
        std::fs::write(view.join(".publish_url"), "https://h/view/\n").unwrap();
        std::os::unix::fs::symlink(&real, view.join("link")).unwrap();
        let through = view.join("link").join("ds");
        assert_eq!(find_catalog_boundary(&through), Some(logical_path(&view)));
        assert!(logical_path(&through.join("..")).ends_with("view/link"));
    }
}
