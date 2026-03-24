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

/// Sentinel file that prevents catalog generation in this directory and below.
const DO_NOT_CATALOG_FILE: &str = ".do_not_catalog";

/// Walk up from `dir` looking for `.catalog_root`.
/// Returns a relative path from `dir`.
fn find_catalog_root(dir: &Path) -> Option<PathBuf> {
    let abs = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    let mut current = abs.clone();
    let mut levels_up: usize = 0;
    loop {
        if current.join(CATALOG_ROOT_FILE).is_file() {
            let mut rel = dir.to_path_buf();
            for _ in 0..levels_up { rel = rel.join(".."); }
            return Some(rel);
        }
        if !current.pop() {
            return None;
        }
        levels_up += 1;
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
                    super::rel_display(cr), super::rel_display(pr),
                ),
            ]);
        }
    }

    // Use .catalog_root if present, else .publish_url, else scan root
    let bound_root = catalog_root
        .or(publish_root)
        .unwrap_or_else(|| root.to_path_buf());
    // Canonicalize bound root for path comparison, keep for display relativization
    let bound_canon = std::fs::canonicalize(&bound_root).unwrap_or_else(|_| bound_root.clone());
    // Helper: display a path relative to the bound root
    let rel = |p: &Path| -> String {
        if let Ok(r) = p.strip_prefix(&bound_canon) {
            let s = r.to_string_lossy();
            if s.is_empty() { ".".to_string() } else { s.to_string() }
        } else {
            super::rel_display(p)
        }
    };

    // Only publishable datasets (those with a .publish sentinel file)
    // require catalog coverage. Non-publishable datasets are local-only
    // and don't need to be in the catalog hierarchy.
    let publishable_datasets: Vec<&PathBuf> = dataset_files.iter()
        .filter(|ds| {
            let ds_dir = ds.parent().unwrap_or(Path::new("."));
            ds_dir.join(".publish").exists()
        })
        .collect();

    if publishable_datasets.is_empty() {
        let mut result = CheckResult::ok("catalogs");
        result.messages.push("no publishable datasets (no .publish sentinels) — nothing to check".to_string());
        return result;
    }

    // Collect all directories that need catalog files: every directory on the
    // path from each publishable dataset dir up to (and including) bound_root.
    // Use canonicalized paths for comparison but store relative paths.
    let mut dirs_needing_catalog: Vec<PathBuf> = Vec::new();
    let bound_abs = std::fs::canonicalize(&bound_root).unwrap_or_else(|_| bound_root.clone());

    for ds_path in &publishable_datasets {
        let ds_dir = ds_path.parent().unwrap_or(Path::new("."));
        let mut current = std::fs::canonicalize(ds_dir).unwrap_or_else(|_| ds_dir.to_path_buf());
        let bound = &bound_abs;

        loop {
            if !dirs_needing_catalog.contains(&current) {
                dirs_needing_catalog.push(current.clone());
            }
            if current == *bound || !current.starts_with(bound) {
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

    // Check each directory: catalogs required unless .do_not_catalog is present.
    // Directories with .do_not_catalog must NOT have catalog files.
    let mut catalog_mtimes: Vec<(PathBuf, SystemTime)> = Vec::new();

    for dir in &dirs_needing_catalog {
        let rel = dir.strip_prefix(root).unwrap_or(dir);
        let rel_display = {
            let s = rel.to_string_lossy();
            if s.is_empty() { ".".to_string() } else { s.to_string() }
        };

        let has_do_not_catalog = dir.join(DO_NOT_CATALOG_FILE).exists();
        let catalog_json = dir.join("catalog.json");
        let catalog_yaml = dir.join("catalog.yaml");
        let has_json = catalog_json.exists();
        let has_yaml = catalog_yaml.exists();

        if has_do_not_catalog {
            // .do_not_catalog present — catalog files must NOT exist here
            if has_json || has_yaml {
                failures.push(format!(
                    "{}: has .do_not_catalog but catalog files exist (remove them)",
                    rel_display,
                ));
            }
            continue; // no catalog required
        }

        let catalog_path = if has_json {
            Some(catalog_json)
        } else if has_yaml {
            Some(catalog_yaml)
        } else {
            None
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
            let fname = file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if fname == "catalog.json" || fname == "catalog.yaml" || fname.ends_with(".mref") {
                continue;
            }
            if let Ok(file_mtime) = std::fs::metadata(file).and_then(|m| m.modified()) {
                if file_mtime > *catalog_mtime {
                    failures.push(format!(
                        "{}: catalog is older than {}",
                        rel(catalog_dir),
                        rel(file),
                    ));
                    break;
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
            if child_dir.starts_with(parent_dir) && child_dir != parent_dir {
                if parent_mtime < child_mtime {
                    failures.push(format!(
                        "{}: catalog is older than child {}",
                        rel(parent_dir),
                        rel(child_dir),
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
