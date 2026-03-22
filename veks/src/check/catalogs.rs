// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Catalog freshness check.
//!
//! Verifies that `catalog.json` (or `catalog.yaml`) exists at every directory
//! level from each dataset directory up to the catalog bound root, and that
//! each catalog's timestamp is the same or newer than all catalogs nested
//! below it.
//!
//! The catalog bound root is determined by:
//! 1. `.catalog_root` — if present, the directory containing it
//! 2. `.publish_url` — if present and no `.catalog_root`, the publish root
//! 3. The scan root directory — fallback
//!
//! If both `.catalog_root` and `.publish_url` exist, `.catalog_root` must be
//! at or interior to the `.publish_url` directory. Violation is an error.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::CheckResult;

/// Sentinel file marking the top of the catalog hierarchy.
const CATALOG_ROOT_FILE: &str = ".catalog_root";

/// Walk up from `dir` looking for `.catalog_root`.
fn find_catalog_root(dir: &Path) -> Option<PathBuf> {
    let mut current = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    loop {
        if current.join(CATALOG_ROOT_FILE).is_file() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Check that catalog files are present and current at every directory level.
///
/// The verification boundary is the innermost of `.catalog_root` or
/// `.publish_url`. Catalogs are required at every directory level from each
/// dataset up to that boundary.
pub fn check(root: &Path, dataset_files: &[PathBuf]) -> CheckResult {
    if dataset_files.is_empty() {
        return CheckResult::fail("catalogs", vec![
            "no dataset.yaml files found — nothing to check".to_string(),
        ]);
    }

    // Find .catalog_root and .publish_url boundaries
    let catalog_root = find_catalog_root(root);
    let publish_root = super::publish_url::find_publish_file(root)
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));

    // Validate: if both exist, .catalog_root must be at or below .publish_url
    if let (Some(cr), Some(pr)) = (&catalog_root, &publish_root) {
        let cr_canon = std::fs::canonicalize(cr).unwrap_or_else(|_| cr.clone());
        let pr_canon = std::fs::canonicalize(pr).unwrap_or_else(|_| pr.clone());
        if !cr_canon.starts_with(&pr_canon) {
            return CheckResult::fail("catalogs", vec![
                format!(
                    ".catalog_root at {} is outside the .publish_url root at {} — \
                     .catalog_root must be at or interior to the publish root",
                    cr.display(), pr.display(),
                ),
            ]);
        }
    }

    // Use .catalog_root if present, else .publish_url, else scan root
    let bound_root = catalog_root
        .or(publish_root)
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

    // Check that each catalog is at least as recent as all publishable files
    // in its directory tree. A stale catalog means data changed without
    // regenerating the index.
    let publishable = crate::publish::enumerate_publishable_files(root);
    for (catalog_dir, catalog_mtime) in &catalog_mtimes {
        for file in &publishable {
            if !file.starts_with(catalog_dir) { continue; }
            // Skip .mref and catalog files themselves
            let fname = file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if fname == "catalog.json" || fname == "catalog.yaml" || fname.ends_with(".mref") {
                continue;
            }
            if let Ok(file_mtime) = std::fs::metadata(file).and_then(|m| m.modified()) {
                if file_mtime > *catalog_mtime {
                    let rel = file.strip_prefix(root).unwrap_or(file);
                    let cat_rel = catalog_dir.strip_prefix(root).unwrap_or(catalog_dir);
                    let cat_display = if cat_rel.to_string_lossy().is_empty() { ".".to_string() } else { cat_rel.display().to_string() };
                    failures.push(format!(
                        "{}: catalog is older than {}",
                        cat_display,
                        rel.display(),
                    ));
                    break; // one failure per catalog is enough
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
