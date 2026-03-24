// Copyright (c) DataStax, Inc.
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

        CatalogEntry {
            name: self.config.name.clone(),
            path: rel_path.to_string_lossy().to_string(),
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

    // Resolve the effective scan root based on mode.
    //
    // Priority:
    // 1. --for-publish-url: walk up to .publish_url
    // 2. .catalog_root in parent path: use that directory automatically
    // 3. fall back to input directory
    let scan_root = if for_publish_url {
        // Walk up to find .publish_url, use its parent as root
        match crate::check::publish_url::find_publish_file(&input_path) {
            Some(publish_file) => {
                let root = publish_file.parent().unwrap().to_path_buf();
                eprintln!(
                    "Found {} — scanning from publish root: {}",
                    rel(&publish_file),
                    rel(&root)
                );
                root
            }
            None => {
                eprintln!("error: --for-publish-url specified but no .publish_url found above {}", rel(&input_path));
                std::process::exit(1);
            }
        }
    } else if let Some(catalog_root) = find_catalog_root(&input_path) {
        // .catalog_root found — use that directory as the scan root and
        // generate catalogs at every level from there down.
        eprintln!(
            "Found {} — catalog root: {}",
            rel(&catalog_root.join(CATALOG_ROOT_FILE)),
            rel(&catalog_root)
        );
        catalog_root
    } else {
        input_path.clone()
    };

    // When .catalog_root was detected, we always generate the full hierarchy
    // from that root (not just existing catalogs), so override update mode.
    let catalog_root_detected = !for_publish_url && find_catalog_root(&input_path).is_some();
    let effective_update = if catalog_root_detected { false } else { update };

    // Detect .publish_url for warning purposes.
    // When --for-publish-url is active or .catalog_root was detected,
    // we're already scanning from an appropriate root — no warning needed.
    // Otherwise, check if the publish root is above our scan root.
    if !for_publish_url && !catalog_root_detected {
        if let Some(publish_file) = crate::check::publish_url::find_publish_file(&input_path) {
            let publish_root = publish_file.parent().unwrap();
            if publish_root != scan_root {
                eprintln!("WARNING: .publish_url found at {}", rel(&publish_file));
                eprintln!("  The publish root is above the scan directory.");
                if effective_update {
                    eprintln!("  With --update (default), only existing catalog files will be refreshed.");
                    eprintln!("  Parent directories between {} and {} that lack catalogs will remain uncovered.", rel(&scan_root), rel(publish_root));
                } else {
                    eprintln!("  Generating catalogs only from {} will leave parent catalogs stale.", rel(&scan_root));
                }
                eprintln!("  Use --for-publish-url to regenerate the full publish hierarchy,");
                eprintln!("  or place a .catalog_root file at the desired catalog top level.");
                eprintln!();
            }
        }
    }

    eprintln!("Scanning {} for datasets...", rel(&scan_root));

    let mut datasets: Vec<DiscoveredDataset> = Vec::new();
    walk_for_datasets(&scan_root, &mut datasets, &cwd);
    datasets.sort_by(|a, b| a.yaml_path.cmp(&b.yaml_path));

    eprintln!("Found {} dataset(s)", datasets.len());

    if datasets.is_empty() {
        eprintln!("No datasets found.");
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

    for catalog_dir in &catalog_dirs {
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
    // Skip directories with .do_not_catalog sentinel
    if dir.join(DO_NOT_CATALOG_FILE).exists() {
        return;
    }

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
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }
            walk_for_datasets(&subdir, datasets, cwd);
        }
    }
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
