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
pub fn run(input: &Path, basename: &str) {
    if basename.contains('.') {
        eprintln!("error: basename must not contain a dot");
        std::process::exit(1);
    }

    let input_path = if input.is_absolute() {
        input.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(input)
    };

    if !input_path.is_dir() {
        eprintln!("error: {} is not a directory", input_path.display());
        std::process::exit(1);
    }

    eprintln!("Scanning {} for datasets...", input_path.display());

    let mut datasets: Vec<DiscoveredDataset> = Vec::new();
    walk_for_datasets(&input_path, &mut datasets);
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
            ds.yaml_path.display()
        );
    }

    // Determine all catalog directories (hierarchical): every directory
    // between a dataset root and the input root gets its own catalog.
    let input_canonical = input_path
        .canonicalize()
        .unwrap_or_else(|_| input_path.clone());
    let mut catalog_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    catalog_dirs.insert(input_canonical.clone());
    for ds in &datasets {
        let ds_dir = ds.yaml_path.parent().unwrap_or(&ds.yaml_path);
        let ds_canonical = ds_dir
            .canonicalize()
            .unwrap_or_else(|_| ds_dir.to_path_buf());
        let mut dir = ds_canonical;
        while dir.starts_with(&input_canonical) {
            catalog_dirs.insert(dir.clone());
            match dir.parent() {
                Some(parent) => dir = parent.to_path_buf(),
                None => break,
            }
        }
    }

    let mut total_files = 0usize;

    for catalog_dir in &catalog_dirs {
        // Collect entries whose dataset.yaml is under this directory
        let entries: Vec<CatalogEntry> = datasets
            .iter()
            .filter(|ds| {
                let ds_canonical = ds
                    .yaml_path
                    .canonicalize()
                    .unwrap_or_else(|_| ds.yaml_path.clone());
                ds_canonical.starts_with(catalog_dir)
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
            eprintln!("error: failed to write {}: {}", json_path.display(), e);
            std::process::exit(1);
        }

        // Write catalog.yaml
        let yaml_path = catalog_dir.join(format!("{}.yaml", basename));
        let yaml = serde_yaml::to_string(&entries).unwrap_or_else(|e| {
            eprintln!("error: YAML serialization failed: {}", e);
            std::process::exit(1);
        });
        if let Err(e) = std::fs::write(&yaml_path, &yaml) {
            eprintln!("error: failed to write {}: {}", yaml_path.display(), e);
            std::process::exit(1);
        }

        eprintln!(
            "Wrote {} entries to {} and {}",
            entries.len(),
            json_path.display(),
            yaml_path.display()
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
fn walk_for_datasets(dir: &Path, datasets: &mut Vec<DiscoveredDataset>) {
    let yaml_path = dir.join("dataset.yaml");
    if yaml_path.exists() {
        match DatasetConfig::load(&yaml_path) {
            Ok(config) => {
                let abs_yaml = yaml_path.canonicalize().unwrap_or(yaml_path);
                datasets.push(DiscoveredDataset {
                    yaml_path: abs_yaml,
                    config,
                });
            }
            Err(e) => {
                eprintln!(
                    "WARNING: failed to load {}: {}",
                    yaml_path.display(),
                    e
                );
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
            walk_for_datasets(&subdir, datasets);
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
        walk_for_datasets(tmp.path(), &mut datasets);
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
        walk_for_datasets(tmp.path(), &mut datasets);
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
        walk_for_datasets(tmp.path(), &mut datasets);
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
        walk_for_datasets(tmp.path(), &mut datasets);
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
        walk_for_datasets(tmp.path(), &mut datasets);
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].config.name, "real");
    }
}
