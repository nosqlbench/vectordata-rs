// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Catalog freshness check.
//!
//! Verifies that `catalog.json` (or `catalog.yaml`) exists at every directory
//! level from each dataset directory up to the bucket-bound root, and that
//! each catalog's timestamp is the same or newer than all catalogs nested
//! below it.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::CheckResult;

/// Check that catalog files are present and current at every directory level.
///
/// For each dataset directory, walks upward to the nearest `.s3-bucket` (or
/// filesystem root if none). At each level, checks that `catalog.json` or
/// `catalog.yaml` exists and is at least as recent as any child catalog.
pub fn check(root: &Path, dataset_files: &[PathBuf]) -> CheckResult {
    if dataset_files.is_empty() {
        return CheckResult::fail("catalogs", vec![
            "no dataset.yaml files found — nothing to check".to_string(),
        ]);
    }

    // Find the bucket-bound root (or use the scan root)
    let bound_root = super::publish_url::find_publish_file(root)
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| root.to_path_buf());

    // Collect all directories that need catalog files: every directory on the
    // path from each dataset dir up to (and including) bound_root.
    let mut dirs_needing_catalog: Vec<PathBuf> = Vec::new();

    for ds_path in dataset_files {
        let ds_dir = ds_path.parent().unwrap_or(Path::new("."));
        let mut current = if ds_dir.is_absolute() {
            ds_dir.to_path_buf()
        } else {
            std::fs::canonicalize(ds_dir).unwrap_or_else(|_| ds_dir.to_path_buf())
        };
        let bound = std::fs::canonicalize(&bound_root)
            .unwrap_or_else(|_| bound_root.clone());

        loop {
            if !dirs_needing_catalog.contains(&current) {
                dirs_needing_catalog.push(current.clone());
            }
            if current == bound || !current.starts_with(&bound) {
                break;
            }
            if !current.pop() {
                break;
            }
        }
    }

    dirs_needing_catalog.sort();
    dirs_needing_catalog.dedup();

    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;

    // Check each directory has a catalog file
    let mut catalog_mtimes: Vec<(PathBuf, SystemTime)> = Vec::new();

    for dir in &dirs_needing_catalog {
        let catalog_json = dir.join("catalog.json");
        let catalog_yaml = dir.join("catalog.yaml");

        let catalog_path = if catalog_json.exists() {
            Some(catalog_json)
        } else if catalog_yaml.exists() {
            Some(catalog_yaml)
        } else {
            None
        };

        let rel = dir.strip_prefix(root).unwrap_or(dir);
        let rel_display = {
            let s = rel.to_string_lossy();
            if s.is_empty() { ".".to_string() } else { s.to_string() }
        };

        match catalog_path {
            None => {
                failures.push(format!(
                    "{}: no catalog.json or catalog.yaml",
                    rel_display,
                ));
            }
            Some(path) => {
                checked += 1;
                if let Ok(meta) = std::fs::metadata(&path) {
                    if let Ok(mtime) = meta.modified() {
                        catalog_mtimes.push((dir.clone(), mtime));
                    }
                }
            }
        }
    }

    // Check timestamp ordering: parent catalogs must be >= all child catalogs
    for i in 0..catalog_mtimes.len() {
        let (ref parent_dir, parent_mtime) = catalog_mtimes[i];
        for j in 0..catalog_mtimes.len() {
            if i == j { continue; }
            let (ref child_dir, child_mtime) = catalog_mtimes[j];
            // child_dir is under parent_dir?
            if child_dir.starts_with(parent_dir) && child_dir != parent_dir {
                if parent_mtime < child_mtime {
                    let parent_rel = parent_dir.strip_prefix(root).unwrap_or(parent_dir);
                    let child_rel = child_dir.strip_prefix(root).unwrap_or(child_dir);
                    failures.push(format!(
                        "{}: catalog is older than child {}",
                        parent_rel.display(),
                        child_rel.display(),
                    ));
                }
            }
        }
    }

    if failures.is_empty() {
        let mut result = CheckResult::ok("catalogs");
        result.messages.push(format!(
            "{} directory level(s) checked, all have current catalogs",
            checked,
        ));
        result
    } else {
        CheckResult::fail("catalogs", failures)
    }
}
