// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! `veks catalog generate` — scan directory trees for `dataset.yaml` files
//! and write `catalog.json` / `catalog.yaml` index files at each directory
//! level in the hierarchy.
//!
//! The output uses layout-embedded catalog entries compatible with the
//! upstream Java catalog specification. Each entry contains:
//!
//! - `name` — the dataset name (from `dataset.yaml`)
//! - `path` — relative path from the catalog file to `dataset.yaml`
//! - `dataset_type` — always `"dataset.yaml"`
//! - `layout` — embedded dataset configuration containing `attributes`
//!   and `profiles`

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vectordata::dataset::{CatalogEntry, CatalogLayout, DatasetConfig};

/// Sentinel file that marks the top of the catalog hierarchy.
///
/// When present in a parent directory, `catalog generate` automatically
/// uses that directory as the scan root and generates catalogs at every
/// level from there down. This allows the user to keep an uncataloged
/// leading path in a remote URL while still getting automatic updates.
const CATALOG_ROOT_FILE: &str = ".catalog_root";

/// Sentinel file that prevents catalog generation in this directory and below.
const DO_NOT_CATALOG_FILE: &str = ".do_not_catalog";

/// Walk up from `dir` looking for a `.catalog_root` file.
/// Returns the directory containing it, or `None`.
fn find_catalog_root(dir: &Path) -> Option<PathBuf> {
    let mut current = dir.to_path_buf();

    loop {
        let candidate = current.join(CATALOG_ROOT_FILE);
        if candidate.is_file() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Internal representation of a discovered dataset before path relativization.
struct DiscoveredDataset {
    /// Absolute (canonical) path to the `dataset.yaml` file.
    yaml_path: PathBuf,
    /// The loaded dataset configuration.
    config: DatasetConfig,
}

impl DiscoveredDataset {
    /// Create a catalog entry with the path relativized to the given directory.
    ///
    /// Both `self.yaml_path` and `catalog_dir` are expected to be canonical
    /// (absolute, symlink-resolved) paths so that `strip_prefix` works
    /// reliably.
    fn to_entry(&self, catalog_dir: &Path) -> CatalogEntry {
        let rel_path = self
            .yaml_path
            .strip_prefix(catalog_dir)
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| self.yaml_path.clone());

        // Elide "." components (e.g., "laion400b/./img-search" → "laion400b/img-search")
        let cleaned: std::path::PathBuf = rel_path.components()
            .filter(|c| !matches!(c, std::path::Component::CurDir))
            .collect();

        CatalogEntry {
            name: self.config.name.clone(),
            path: cleaned.to_string_lossy().to_string(),
            dataset_type: "dataset.yaml".to_string(),
            layout: CatalogLayout {
                attributes: self.config.attributes.clone(),
                profiles: self.config.profiles.clone(),
            },
        }
    }
}

/// Run `veks catalog generate`.
///
/// Modes:
/// - `for_publish_url`: walk up to `.publish_url` root and generate catalogs
///   for the entire publish hierarchy.
/// - `update` (default true): only update catalog files that already exist in
///   the hierarchy. Directories with no existing catalog are skipped.
/// - Neither: generate catalogs at all hierarchy levels from `input` down.
///   If `.publish_url` is detected above `input` and `update` is false, warn
///   that partial catalogs may leave the publish hierarchy out of sync.
pub fn run(input: &Path, basename: &str, for_publish_url: bool, update: bool) {
    if basename.contains('.') {
        eprintln!("error: basename must not contain a dot");
        std::process::exit(1);
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Display a path relative to cwd for user-facing output.
    let rel = |p: &Path| -> String {
        p.strip_prefix(&cwd)
            .map(|r| if r.as_os_str().is_empty() { ".".to_string() } else { r.to_string_lossy().to_string() })
            .unwrap_or_else(|_| p.to_string_lossy().to_string())
    };

    let input_path = if input.is_absolute() {
        input.to_path_buf()
    } else {
        cwd.join(input)
    };

    if !input_path.is_dir() {
        eprintln!("error: {} is not a directory", rel(&input_path));
        std::process::exit(1);
    }

    // Resolve the effective scan root.
    //
    // Priority: always prefer the outermost enclosing publish root so that
    // catalogs are generated for the full publish hierarchy. Inner publish
    // roots are naturally included because we scan downward from the root.
    //
    // 1. Walk up from input_path (including itself) to find .publish_url
    // 2. If not found above, scan immediate children for .publish_url
    // 3. No .publish_url found → refuse to run
    let scan_root = {
        // Walk up from input_path to find enclosing .publish_url
        match crate::check::publish_url::find_publish_file(&input_path) {
            Some(pf) => {
                let root = pf.parent().unwrap().to_path_buf();
                let root = if root.is_absolute() { root } else { cwd.join(&root) };
                // Normalize without resolving symlinks (canonicalize breaks
                // symlink-contained views by crossing into the physical path)
                let root = normalize_path(&root);
                eprintln!("Publish root: {} (from {})", rel(&root), rel(&pf));
                root
            }
            None => {
                // No .publish_url at or above input_path — scan children
                let mut child_roots: Vec<PathBuf> = Vec::new();
                if let Ok(entries) = std::fs::read_dir(&input_path) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir() && path.join(".publish_url").exists() {
                            child_roots.push(path);
                        }
                    }
                }

                if !child_roots.is_empty() {
                    // Found publish roots in children — generate catalogs for each
                    child_roots.sort();
                    for child_root in &child_roots {
                        eprintln!("Found child publish root: {}", rel(child_root));
                        run(child_root, basename, for_publish_url, update);
                    }
                    // Also generate a top-level catalog covering all children
                    eprintln!("Generating top-level catalog at {}", rel(&input_path));
                    input_path.clone()
                } else {
                    eprintln!("error: no .publish_url found at, within, or above {}", rel(&input_path));
                    eprintln!("  Catalog generation requires a publish tree.");
                    eprintln!("  Create one with: echo 's3://bucket/prefix/' > .publish_url");
                    std::process::exit(1);
                }
            }
        }
    };

    let effective_update = update;

    // Scan input_path (not scan_root) for datasets — the user pointed at
    // this directory. scan_root is only the upper bound for catalog placement.
    eprintln!("Scanning {} for datasets...", rel(&input_path));

    let mut all_datasets: Vec<DiscoveredDataset> = Vec::new();
    walk_for_datasets(&input_path, &mut all_datasets, &cwd);
    // Only include publishable datasets (those with .publish sentinel)
    let datasets: Vec<DiscoveredDataset> = all_datasets.into_iter()
        .filter(|ds| {
            let ds_dir = ds.yaml_path.parent().unwrap_or(std::path::Path::new("."));
            ds_dir.join(".publish").exists()
        })
        .collect();
    let mut datasets = datasets;
    datasets.sort_by(|a, b| a.yaml_path.cmp(&b.yaml_path));

    eprintln!("Found {} publishable dataset(s)", datasets.len());

    if datasets.is_empty() {
        eprintln!("No publishable datasets found (no .publish sentinels).");
        return;
    }

    for ds in &datasets {
        let profile_count = ds.config.profile_names().len();
        eprintln!(
            "  {} — {} profile(s) [{}]",
            ds.config.name,
            profile_count,
            rel(&ds.yaml_path)
        );
    }

    // Determine all candidate catalog directories (hierarchical): every
    // directory between a dataset root and the scan root gets a catalog.
    let mut catalog_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    catalog_dirs.insert(scan_root.clone());
    for ds in &datasets {
        let ds_dir = ds.yaml_path.parent().unwrap_or(&ds.yaml_path);
        let mut dir = ds_dir.to_path_buf();
        while dir.starts_with(&scan_root) {
            catalog_dirs.insert(dir.clone());
            match dir.parent() {
                Some(parent) => dir = parent.to_path_buf(),
                None => break,
            }
        }
    }

    // In --update mode (and no .catalog_root override), filter to only
    // directories that already have a catalog
    if effective_update && !for_publish_url {
        let before = catalog_dirs.len();
        catalog_dirs.retain(|dir| {
            let json_path = dir.join(format!("{}.json", basename));
            let yaml_path = dir.join(format!("{}.yaml", basename));
            json_path.exists() || yaml_path.exists()
        });
        let skipped = before - catalog_dirs.len();
        if skipped > 0 {
            eprintln!("  --update: skipping {} directories with no existing catalog", skipped);
        }
        if catalog_dirs.is_empty() {
            eprintln!("No existing catalog files found to update. Use --no-update to create new ones,");
            eprintln!("or --for-publish-url to generate the full publish hierarchy.");
            return;
        }
    }

    let mut total_files = 0usize;

    // Write catalogs bottom-up (deepest directories first) so that parent
    // catalog mtimes are always newer than their children. The staleness
    // check compares parent vs child mtimes.
    let mut sorted_dirs: Vec<&PathBuf> = catalog_dirs.iter().collect();
    sorted_dirs.sort_by(|a, b| b.components().count().cmp(&a.components().count())
        .then_with(|| a.cmp(b)));

    for catalog_dir in &sorted_dirs {
        // Skip directories with .do_not_catalog sentinel
        if catalog_dir.join(DO_NOT_CATALOG_FILE).exists() {
            eprintln!("  Skipping {} (.do_not_catalog)", rel(catalog_dir));
            continue;
        }

        // Collect entries whose dataset.yaml is under this directory
        let entries: Vec<CatalogEntry> = datasets
            .iter()
            .filter(|ds| {
                ds.yaml_path.starts_with(catalog_dir)
            })
            .map(|ds| ds.to_entry(catalog_dir))
            .collect();

        if entries.is_empty() {
            continue;
        }

        // Write catalog.json
        let json_path = catalog_dir.join(format!("{}.json", basename));
        let json = serde_json::to_string_pretty(&entries).unwrap_or_else(|e| {
            eprintln!("error: JSON serialization failed: {}", e);
            std::process::exit(1);
        });
        if let Err(e) = std::fs::write(&json_path, &json) {
            eprintln!("error: failed to write {}: {}", rel(&json_path), e);
            std::process::exit(1);
        }

        // Write catalog.yaml
        let yaml_path = catalog_dir.join(format!("{}.yaml", basename));
        let yaml = serde_yaml::to_string(&entries).unwrap_or_else(|e| {
            eprintln!("error: YAML serialization failed: {}", e);
            std::process::exit(1);
        });
        if let Err(e) = std::fs::write(&yaml_path, &yaml) {
            eprintln!("error: failed to write {}: {}", rel(&yaml_path), e);
            std::process::exit(1);
        }

        eprintln!(
            "Wrote {} entries to {} and {}",
            entries.len(),
            rel(&json_path),
            rel(&yaml_path)
        );
        total_files += 2;
    }

    eprintln!(
        "{} datasets cataloged across {} directories ({} files written)",
        datasets.len(),
        catalog_dirs.len(),
        total_files
    );
}

/// Recursively walk directories to find dataset.yaml files.
///
/// When a directory contains `dataset.yaml`, it is treated as a dataset
/// root: the config is loaded and no further descent occurs.
fn walk_for_datasets(dir: &Path, datasets: &mut Vec<DiscoveredDataset>, cwd: &Path) {
    // Note: .do_not_catalog prevents catalog *placement* at a directory,
    // but does NOT prevent descending through it to find datasets below.
    // The filtering happens in the catalog-writing loop, not here.

    let yaml_path = dir.join("dataset.yaml");
    if yaml_path.exists() {
        match DatasetConfig::load(&yaml_path) {
            Ok(config) => {
                datasets.push(DiscoveredDataset {
                    yaml_path,
                    config,
                });
            }
            Err(e) => {
                let rel = yaml_path.strip_prefix(cwd)
                    .map(|r| r.to_string_lossy().to_string())
                    .unwrap_or_else(|_| yaml_path.to_string_lossy().to_string());
                eprintln!("WARNING: failed to load {}: {}", rel, e);
            }
        }
        // Don't descend into dataset directories
        return;
    }

    if let Ok(read_dir) = std::fs::read_dir(dir) {
        let mut subdirs: Vec<PathBuf> = read_dir
            .flatten()
            .filter(|e| e.path().is_dir())
            .map(|e| e.path())
            .collect();
        subdirs.sort();

        for subdir in subdirs {
            let name = subdir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if crate::filters::is_excluded_dir(name) {
                continue;
            }
            walk_for_datasets(&subdir, datasets, cwd);
        }
    }
}

/// Normalize a path by resolving `.` and `..` components without following
/// symlinks. Unlike `canonicalize`, this preserves the logical path through
/// symlinks so that relative path computations stay within the user's view.
fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => { result.pop(); }
            Component::CurDir => {}
            other => result.push(other),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mut datasets = Vec::new();
        walk_for_datasets(tmp.path(), &mut datasets, tmp.path());
        assert!(datasets.is_empty());
    }

    #[test]
    fn test_walk_finds_datasets() {
        let tmp = tempfile::tempdir().unwrap();

        let ds1 = tmp.path().join("ds-a");
        std::fs::create_dir(&ds1).unwrap();
        std::fs::write(
            ds1.join("dataset.yaml"),
            "name: alpha\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
        )
        .unwrap();

        let ds2 = tmp.path().join("ds-b");
        std::fs::create_dir(&ds2).unwrap();
        std::fs::write(
            ds2.join("dataset.yaml"),
            "name: beta\nprofiles:\n  default:\n    base_vectors: base.fvec\n    query_vectors: query.fvec\n",
        )
        .unwrap();

        let mut datasets = Vec::new();
        walk_for_datasets(tmp.path(), &mut datasets, tmp.path());
        datasets.sort_by(|a, b| a.config.name.cmp(&b.config.name));

        assert_eq!(datasets.len(), 2);
        assert_eq!(datasets[0].config.name, "alpha");
        assert_eq!(datasets[1].config.name, "beta");
    }

    #[test]
    fn test_entry_layout_format() {
        let tmp = tempfile::tempdir().unwrap();

        let ds = tmp.path().join("my-dataset");
        std::fs::create_dir(&ds).unwrap();
        std::fs::write(
            ds.join("dataset.yaml"),
            r#"name: sift-128
attributes:
  distance_function: L2
  license: MIT
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
"#,
        )
        .unwrap();

        let mut datasets = Vec::new();
        walk_for_datasets(tmp.path(), &mut datasets, tmp.path());
        assert_eq!(datasets.len(), 1);

        let root = tmp.path().canonicalize().unwrap();
        let entry = datasets[0].to_entry(&root);

        assert_eq!(entry.name, "sift-128");
        assert_eq!(entry.dataset_type, "dataset.yaml");
        assert!(entry.path.contains("dataset.yaml"));

        // Verify layout has attributes and profiles
        let attrs = entry.layout.attributes.as_ref().unwrap();
        assert_eq!(attrs.distance_function.as_deref(), Some("L2"));
        assert_eq!(attrs.license.as_deref(), Some("MIT"));
        assert!(!entry.layout.profiles.is_empty());

        // Verify JSON serialization is layout-embedded
        let json = serde_json::to_value(&entry).unwrap();
        assert!(json["layout"].is_object());
        assert!(json["layout"]["profiles"]["default"].is_object());
        assert_eq!(json["layout"]["attributes"]["distance_function"], "L2");
        assert_eq!(json["dataset_type"], "dataset.yaml");
    }

    #[test]
    fn test_entry_no_attributes() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("dataset.yaml"),
            "name: simple\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
        )
        .unwrap();

        let mut datasets = Vec::new();
        walk_for_datasets(tmp.path(), &mut datasets, tmp.path());
        assert_eq!(datasets.len(), 1);

        let root = tmp.path().canonicalize().unwrap();
        let entry = datasets[0].to_entry(&root);

        // JSON should omit attributes when None
        let json = serde_json::to_value(&entry).unwrap();
        assert!(json.get("layout").unwrap().get("attributes").is_none());
        assert!(json["layout"]["profiles"]["default"].is_object());
    }

    #[test]
    fn test_skips_hidden_and_target_dirs() {
        let tmp = tempfile::tempdir().unwrap();

        // Hidden dir with a dataset — should be skipped
        let hidden = tmp.path().join(".hidden");
        std::fs::create_dir(&hidden).unwrap();
        std::fs::write(
            hidden.join("dataset.yaml"),
            "name: hidden\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
        )
        .unwrap();

        // target dir with a dataset — should be skipped
        let target = tmp.path().join("target");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(
            target.join("dataset.yaml"),
            "name: target\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
        )
        .unwrap();

        // Real dataset
        let real = tmp.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(
            real.join("dataset.yaml"),
            "name: real\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
        )
        .unwrap();

        let mut datasets = Vec::new();
        walk_for_datasets(tmp.path(), &mut datasets, tmp.path());
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].config.name, "real");
    }
}
