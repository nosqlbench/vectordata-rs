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

        // Add all profile view paths — these are dataset artifacts even
        // if they're symlinks (e.g., partition profile query_vectors).
        for (_name, profile) in &config.profiles.profiles {
            for (_facet, view) in profile.views() {
                let path = view.path();
                if !path.is_empty() {
                    accounted.insert(path.to_string());
                }
            }
        }

        // Account for partition profile artifacts that are produced by
        // Phase 3 re-expansion (per_profile steps for partition profiles).
        // These aren't in the initial step resolution but ARE legitimate
        // pipeline outputs.
        for (name, _profile) in &config.profiles.profiles {
            if name == "default" { continue; }
            let profile_dir = format!("profiles/{}", name);
            let profile_path = workspace.join(&profile_dir);
            if profile_path.is_dir()
                && let Ok(entries) = std::fs::read_dir(&profile_path) {
                    for entry in entries.flatten() {
                        let fname = entry.file_name().to_string_lossy().to_string();
                        // Skip IDXFOR files (handled separately below)
                        if fname.starts_with("IDXFOR__") { continue; }
                        accounted.insert(format!("{}/{}", profile_dir, fname));
                    }
                }
        }

        // knn_entries.yaml is produced by catalog generate
        accounted.insert("knn_entries.yaml".to_string());

        // docs/ directory is a standard dataset artifact directory —
        // all files within it are part of the dataset.
        let docs_dir = workspace.join("docs");
        if docs_dir.is_dir()
            && let Ok(entries) = std::fs::read_dir(&docs_dir) {
                for entry in entries.flatten() {
                    if entry.path().is_file() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        accounted.insert(format!("docs/{}", name));
                    }
                }
            }

        // Check each publishable file under this workspace
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
            if !is_accounted(&rel, &filename, &accounted) {
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
            if let Ok(canon) = path.canonicalize()
                && let Ok(rel) = canon.strip_prefix(&workspace_canonical) {
                    return rel.to_string_lossy().to_string();
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

    // Add all profile view paths
    for (_name, profile) in &config.profiles.profiles {
        for (_facet, view) in profile.views() {
            let path = view.path();
            if !path.is_empty() {
                accounted.insert(path.to_string());
            }
        }
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
        if !is_accounted(&rel, &filename, &accounted) {
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

/// Whether a publishable file at workspace-relative `rel` (whose last
/// component is `filename`) is accounted for by the pipeline — the one
/// rule both the check and the clean path apply.
///
/// A file is accounted for when the manifest names it, when it is
/// known infrastructure, when it is local merkle state, or when it is a
/// derivative of something the manifest names: a `.mref` of it, an
/// `IDXFOR__` index of it, or a **shard of it** — `base_vectors__0003.fvecs`
/// belongs to the series the manifest knows as `base_vectors.fvecs`
/// or declares as `base_vectors__NNNN.fvecs`.
fn is_accounted(rel: &str, filename: &str, accounted: &HashSet<String>) -> bool {
    if accounted.contains(rel) || KNOWN_INFRA.contains(&filename) || rel.ends_with(".mrkl") {
        return true;
    }
    // Static payload — a license, a notice, a readme placed by hand —
    // and its merkle reference.
    if vectordata::filters::is_static_payload(filename)
        || filename
            .strip_suffix(".mref")
            .is_some_and(vectordata::filters::is_static_payload)
    {
        return true;
    }
    let dir = rel.rsplit_once('/').map(|(d, _)| d);
    let in_dir = |name: &str| match dir {
        Some(d) => format!("{}/{}", d, name),
        None => name.to_string(),
    };
    // A .mref of an accounted file.
    if let Some(base) = rel.strip_suffix(".mref")
        && accounted.contains(base)
    {
        return true;
    }
    // IDXFOR__metadata_results.ivvec.i32 → metadata_results.ivvec
    if let Some(data_name) = filename
        .strip_prefix("IDXFOR__")
        .and_then(|s| s.rsplit_once('.'))
        .map(|(base, _)| base)
        && accounted.contains(&in_dir(data_name))
    {
        return true;
    }
    // A shard of a series the manifest names by its unsharded name or
    // by its declared pattern.
    if let Some(series) = vectordata::dataset::shards::series_of_shard(filename) {
        if accounted.contains(&in_dir(&series)) {
            return true;
        }
        let (stem, ext) = match series.rsplit_once('.') {
            Some((s, e)) => (s.to_string(), e.to_string()),
            None => (series.clone(), String::new()),
        };
        if accounted.contains(&in_dir(&vectordata::dataset::shards::shard_source_spec(&stem, &ext))) {
            return true;
        }
    }
    false
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

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// The shards of a series the manifest names are accounted for;
    /// shards of a series it does not name are not.
    #[test]
    fn shards_are_accounted_for_by_their_series() {
        let accounted = set(&["profiles/base/base_vectors.fvecs"]);
        assert!(is_accounted("profiles/base/base_vectors__0000.fvecs", "base_vectors__0000.fvecs", &accounted));
        assert!(is_accounted("profiles/base/base_vectors__0004.fvecs", "base_vectors__0004.fvecs", &accounted));
        assert!(!is_accounted("profiles/base/other__0000.fvecs", "other__0000.fvecs", &accounted));
        assert!(!is_accounted("profiles/base/base_vectors.mvecs", "base_vectors.mvecs", &accounted));
        // A declared pattern accounts for its shards too.
        let declared = set(&["profiles/base/base_vectors__NNNN.fvecs"]);
        assert!(is_accounted("profiles/base/base_vectors__0001.fvecs", "base_vectors__0001.fvecs", &declared));
    }

    /// The other derivative rules are unchanged by the refactor.
    #[test]
    fn static_payload_is_accounted_for() {
        let accounted: HashSet<String> = HashSet::new();
        assert!(is_accounted("LICENSE.md", "LICENSE.md", &accounted));
        assert!(is_accounted("LICENSE.md.mref", "LICENSE.md.mref", &accounted));
        assert!(is_accounted("docs/NOTICE.md", "NOTICE.md", &accounted));
        assert!(!is_accounted("license-draft.md", "license-draft.md", &accounted));
    }

    #[test]
    fn derivatives_and_infrastructure_are_accounted_for() {
        let accounted = set(&["profiles/default/metadata_results.ivvec", "a.fvecs"]);
        assert!(is_accounted("profiles/default/IDXFOR__metadata_results.ivvec.i32", "IDXFOR__metadata_results.ivvec.i32", &accounted));
        assert!(!is_accounted("profiles/default/IDXFOR__other.ivvec.i32", "IDXFOR__other.ivvec.i32", &accounted));
        assert!(is_accounted("a.fvecs.mref", "a.fvecs.mref", &accounted));
        assert!(!is_accounted("b.fvecs.mref", "b.fvecs.mref", &accounted));
        assert!(is_accounted("state.mrkl", "state.mrkl", &accounted));
        assert!(is_accounted("dataset.yaml", "dataset.yaml", &accounted));
        assert!(!is_accounted("stray.bin", "stray.bin", &accounted));
    }
}

