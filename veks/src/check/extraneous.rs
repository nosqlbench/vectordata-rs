// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Extraneous file detection using projected artifact manifests.
//!
//! Each pipeline command's `project_artifacts` method declares its inputs
//! and outputs without executing. This module collects all manifests,
//! combines them with profile view paths and known infrastructure files,
//! and identifies publishable files that aren't accounted for.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use vectordata::dataset::DatasetConfig;

use crate::pipeline::manifest;
use crate::pipeline::registry::CommandRegistry;

use super::CheckResult;

/// Known infrastructure files that are always expected.
const KNOWN_INFRA: &[&str] = &[
    "dataset.yaml",
    "dataset.yml",
    "dataset.json",
    "dataset.jsonl",
    "dataset.log",
    "catalog.json",
    "catalog.yaml",
    "variables.json",
    "variables.yaml",
    "runlog.jsonl",
];

/// Check for extraneous publishable files not accounted for by the pipeline.
///
/// Uses `project_artifacts` on every pipeline command to build a complete
/// manifest, then compares against the actual publishable files on disk.
pub fn check(
    _root: &Path,
    dataset_files: &[PathBuf],
    publishable: &[PathBuf],
) -> CheckResult {
    if dataset_files.is_empty() {
        return CheckResult::ok("extraneous-files");
    }

    let registry = CommandRegistry::with_builtins();
    let mut all_extraneous: Vec<String> = Vec::new();

    for dataset_path in dataset_files {
        let workspace = dataset_path.parent().unwrap_or(Path::new("."));

        let mut config = match DatasetConfig::load(dataset_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Resolve all steps (including deferred profile expansion) using
        // the same logic as veks run — single source of truth.
        let _ = veks_pipeline::pipeline::resolve_all_steps(&mut config, workspace);

        let wm = match manifest::project_workspace(dataset_path, &config, &registry) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Build the complete set of accounted-for paths
        let mut accounted: HashSet<String> = HashSet::new();
        for p in &wm.final_artifacts {
            accounted.insert(p.clone());
        }
        for p in &wm.intermediates {
            accounted.insert(p.clone());
        }
        // Inputs that are also outputs of other steps are already covered;
        // inputs that are external sources are not in the workspace.

        // Check each publishable file under this workspace
        for file in publishable {
            if !file.starts_with(workspace) {
                continue;
            }

            let rel = file.strip_prefix(workspace)
                .unwrap_or(file)
                .to_string_lossy()
                .to_string();

            // Skip known infrastructure
            let filename = file.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if KNOWN_INFRA.contains(&filename.as_str()) {
                continue;
            }

            // Skip .mref if base file is accounted for
            if rel.ends_with(".mref") {
                let base = &rel[..rel.len() - 5];
                if accounted.contains(base) {
                    continue;
                }
            }

            // Skip .mrkl files (local merkle state)
            if rel.ends_with(".mrkl") {
                continue;
            }

            // Skip IDXFOR__ index files if their data file is accounted for
            if filename.starts_with("IDXFOR__") {
                // IDXFOR__metadata_indices.ivvec.i32 → metadata_indices.ivvec
                let data_name = filename.strip_prefix("IDXFOR__")
                    .and_then(|s| s.rsplit_once('.'))  // strip .i32/.i64
                    .map(|(base, _)| base);
                if let Some(dn) = data_name {
                    let data_rel = rel.rsplit_once('/')
                        .map(|(dir, _)| format!("{}/{}", dir, dn))
                        .unwrap_or_else(|| dn.to_string());
                    if accounted.contains(&data_rel) {
                        continue;
                    }
                }
            }

            if !accounted.contains(&rel) {
                let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
                all_extraneous.push(format!("{} ({})", rel, format_size(size)));
            }
        }
    }

    if all_extraneous.is_empty() {
        let mut result = CheckResult::ok("extraneous-files");
        result.messages.push("all publishable files are accounted for by the pipeline".to_string());
        result
    } else {
        let mut messages = vec![
            format!("{} extraneous file(s) not in any pipeline manifest:", all_extraneous.len()),
        ];
        for f in &all_extraneous {
            messages.push(format!("  {}", f));
        }
        CheckResult::fail("extraneous-files", messages)
    }
}

/// Find extraneous files and return their paths (for --clean-files).
pub fn find_extraneous(
    dataset_path: &Path,
    publishable: &[PathBuf],
) -> Vec<PathBuf> {
    let workspace = dataset_path.parent().unwrap_or(Path::new("."));
    let registry = CommandRegistry::with_builtins();

    let config = match DatasetConfig::load(dataset_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let wm = match manifest::project_workspace(dataset_path, &config, &registry) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };

    let mut accounted: HashSet<String> = HashSet::new();
    let workspace_canonical = workspace.canonicalize().unwrap_or(workspace.to_path_buf());

    // Normalize all manifest paths to relative form for consistent comparison.
    // Absolute paths are stripped of the workspace prefix; relative paths pass through.
    let normalize = |p: &str| -> String {
        let path = std::path::Path::new(p);
        if path.is_absolute() {
            // Try stripping workspace prefix (both original and canonical)
            if let Ok(rel) = path.strip_prefix(workspace) {
                return rel.to_string_lossy().to_string();
            }
            if let Ok(rel) = path.strip_prefix(&workspace_canonical) {
                return rel.to_string_lossy().to_string();
            }
            // Try canonicalizing the path and stripping
            if let Ok(canon) = path.canonicalize() {
                if let Ok(rel) = canon.strip_prefix(&workspace_canonical) {
                    return rel.to_string_lossy().to_string();
                }
            }
        }
        p.to_string()
    };

    for p in &wm.final_artifacts {
        accounted.insert(normalize(p));
    }
    for p in &wm.intermediates {
        accounted.insert(normalize(p));
    }
    for p in &wm.inputs {
        accounted.insert(normalize(p));
    }

    let mut result = Vec::new();
    for file in publishable {
        if !file.starts_with(workspace) {
            continue;
        }

        let rel = file.strip_prefix(workspace)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();

        let filename = file.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if KNOWN_INFRA.contains(&filename.as_str()) {
            continue;
        }

        if rel.ends_with(".mref") {
            let base = &rel[..rel.len() - 5];
            if accounted.contains(base) {
                continue;
            }
        }

        if rel.ends_with(".mrkl") {
            continue;
        }

        // Skip IDXFOR__ index files if their data file is accounted for
        if filename.starts_with("IDXFOR__") {
            let data_name = filename.strip_prefix("IDXFOR__")
                .and_then(|s| s.rsplit_once('.'))
                .map(|(base, _)| base);
            if let Some(dn) = data_name {
                let data_rel = rel.rsplit_once('/')
                    .map(|(dir, _)| format!("{}/{}", dir, dn))
                    .unwrap_or_else(|| dn.to_string());
                if accounted.contains(&data_rel) {
                    continue;
                }
            }
        }

        if !accounted.contains(&rel) {
            result.push(file.clone());
        }
    }

    result
}

/// List all cache paths that must be retained by the pipeline.
///
/// Includes intermediates, inputs that live in cache (e.g., downloaded
/// source data), and any outputs stored in cache. Anything in `.cache/`
/// not in this set is safe to delete.
pub fn retained_cache_paths(
    dataset_path: &Path,
) -> Vec<String> {
    let registry = CommandRegistry::with_builtins();
    let config = match DatasetConfig::load(dataset_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    match manifest::project_workspace(dataset_path, &config, &registry) {
        Ok(m) => m.retained_cache_paths().into_iter().collect(),
        Err(_) => Vec::new(),
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
