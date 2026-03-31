// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pre-flight checks for dataset readiness (`veks check`).
//!
//! Validates pipeline completeness, publish URL binding, merkle coverage,
//! and file integrity across a dataset directory tree.

use std::path::{Path, PathBuf};

use clap::Args;

pub mod publish_url;
pub mod catalogs;
pub mod extraneous;
pub mod fix;
pub mod integrity;
pub mod merkle;
pub mod pipelines;

/// Arguments for `veks check`.
#[derive(Args)]
pub struct CheckArgs {
    /// Target directory to check (default: current directory)
    #[arg(default_value = ".")]
    pub directory: PathBuf,

    /// Check all categories (default when no --check-* flags given)
    #[arg(long)]
    pub check_all: bool,

    /// Check pipeline completeness
    #[arg(long)]
    pub check_pipelines: bool,

    /// Check .publish_url binding and transport support
    #[arg(long)]
    pub check_publish: bool,

    /// Check merkle (.mref) coverage for large files
    #[arg(long)]
    pub check_merkle: bool,

    /// Check file format integrity (geometry, record structure)
    #[arg(long)]
    pub check_integrity: bool,

    /// Check catalog files are present and current at every directory level
    #[arg(long)]
    pub check_catalogs: bool,

    /// Check for extraneous files not accounted for by pipeline or profiles
    #[arg(long)]
    pub check_extraneous: bool,

    /// List extraneous files to stdout (one per line), suitable for piping
    #[arg(long)]
    pub clean: bool,

    /// Remove extraneous publishable files not accounted for by the pipeline
    #[arg(long)]
    pub clean_files: bool,

    /// Minimum file size for merkle coverage check
    #[arg(long, default_value = "0", value_name = "SIZE")]
    pub merkle_min_size: String,

    /// Auto-update dataset.yaml with missing pipeline steps (e.g., merkle)
    #[arg(long)]
    pub update_pipeline: bool,

    /// Emit results as JSON
    #[arg(long)]
    pub json: bool,

    /// Suppress detail; exit code only
    #[arg(long)]
    pub quiet: bool,
}

/// Result of a single check category.
pub struct CheckResult {
    pub name: &'static str,
    pub passed: bool,
    pub messages: Vec<String>,
}

impl CheckResult {
    fn ok(name: &'static str) -> Self {
        CheckResult { name, passed: true, messages: vec![] }
    }

    fn fail(name: &'static str, messages: Vec<String>) -> Self {
        CheckResult { name, passed: false, messages }
    }
}

/// Entry point for `veks check`.
pub fn run(args: CheckArgs) {
    // Keep directory as-is for file operations. The find_publish_file and
    // find_catalog_root helpers internally canonicalize when needed for
    // upward traversal, but all user-facing output uses relative paths.
    let directory = args.directory.clone();

    if !directory.is_dir() {
        eprintln!("Error: '{}' is not a directory", rel_display(&directory));
        std::process::exit(2);
    }

    // Detect context: dataset directory, publish path, or unrecognized.
    let has_dataset_yaml = directory.join("dataset.yaml").exists();
    let has_publish_url = publish_url::find_publish_file(&directory).is_some();
    let has_catalog_root = find_catalog_root_file(&directory).is_some();
    let is_publish_path = has_publish_url || has_catalog_root;

    if !has_dataset_yaml && !is_publish_path {
        eprintln!("Error: '{}' has no dataset.yaml and is not part of a publish path", rel_display(&directory));
        eprintln!("  (no .publish_url or .catalog_root found in any parent directory)");
        std::process::exit(2);
    }

    // Determine which checks to run based on context:
    // - Dataset directory: pipeline, integrity, extraneous, merkle
    // - Publish path (non-dataset): add publish, catalogs
    let any_specific = args.check_pipelines || args.check_publish
        || args.check_merkle || args.check_integrity || args.check_catalogs
        || args.check_extraneous;
    // --clean only needs the extraneous check, not the full suite
    let run_all = args.check_all || (!any_specific && !args.clean);

    // A directory can be both dataset context AND publish context when
    // .publish_url is at or above a dataset directory. Both check modes
    // run when applicable.
    let is_dataset_context = has_dataset_yaml;
    let is_publish_context = is_publish_path;

    let run_pipelines = is_dataset_context && (run_all || args.check_pipelines);
    let run_publish = is_publish_context && (run_all || args.check_publish);
    let run_merkle = is_dataset_context && (run_all || args.check_merkle);
    let run_integrity = is_dataset_context && (run_all || args.check_integrity);
    let run_catalogs = is_publish_context && (run_all || args.check_catalogs);
    let run_extraneous = is_dataset_context && (run_all || args.check_extraneous || args.clean || args.clean_files);

    let merkle_threshold = parse_size(&args.merkle_min_size).unwrap_or_else(|| {
        eprintln!("Error: invalid size '{}'", args.merkle_min_size);
        std::process::exit(2);
    });

    // Discover all dataset.yaml files under the target directory.
    let dataset_files = discover_datasets(&directory);

    // Collect files for merkle/integrity/extraneous checks.
    // In dataset context, enumerate all non-excluded files under the workspace
    // (no .publish sentinel required — merkle coverage is always checked).
    // In publish context, use the publish-aware enumerator which respects .publish.
    let publishable = if run_merkle || run_integrity || run_extraneous {
        if is_dataset_context {
            enumerate_workspace_files(&directory)
        } else {
            crate::publish::enumerate_publishable_files(&directory)
        }
    } else {
        vec![]
    };

    let mut results: Vec<CheckResult> = Vec::new();

    if run_pipelines {
        results.push(pipelines::check(&directory, &dataset_files));
        if run_merkle {
            results.push(pipelines::check_coverage(
                &directory, &dataset_files, &publishable, merkle_threshold,
            ));
        }
    }
    // Check required dataset attributes
    if is_dataset_context && (run_all || args.check_pipelines) {
        results.push(check_dataset_attributes(&dataset_files));
    }
    if run_publish {
        results.push(publish_url::check(&directory, &dataset_files));
    }
    if run_merkle {
        results.push(merkle::check(&directory, &publishable, merkle_threshold));
    }
    if run_integrity {
        results.push(integrity::check(&directory, &publishable));
    }
    if run_catalogs {
        results.push(catalogs::check(&directory, &dataset_files));
    }
    if run_extraneous {
        results.push(extraneous::check(&directory, &dataset_files, &publishable));
    }

    // Handle --clean: list extraneous files to stdout and exit.
    // No check output, no fix plans — just bare paths for piping.
    // Exit code 1 if extraneous files exist, 0 if clean.
    if args.clean {
        let mut count = 0;
        for ds_path in &dataset_files {
            let extra = extraneous::find_extraneous(ds_path, &publishable);
            for file in &extra {
                let rel = file.strip_prefix(&directory).unwrap_or(file);
                println!("{}", rel.display());
                count += 1;
            }
        }
        std::process::exit(if count > 0 { 1 } else { 0 });
    }

    // Output results.
    if args.json {
        print_json(&directory, &results);
    } else if !args.quiet {
        print_human(&results);
    }

    let all_passed = results.iter().all(|r| r.passed);

    // Build fix plans for each dataset.yaml when there are failures
    if !all_passed && !args.quiet {
        let missing_merkle: Vec<PathBuf> = if run_merkle {
            merkle::missing_mref_files(&publishable, merkle_threshold)
        } else {
            vec![]
        };
        let missing_publish = results.iter()
            .any(|r| r.name == "publish" && !r.passed);
        let stale_catalogs = results.iter()
            .any(|r| r.name == "catalogs" && !r.passed);
        let stale_steps: Vec<String> = results.iter()
            .filter(|r| r.name == "pipeline-execution" && !r.passed)
            .flat_map(|r| r.messages.iter().cloned())
            .collect();

        // Per-dataset fix plans
        for ds_path in &dataset_files {
            let plan = fix::plan_fixes(
                ds_path,
                &missing_merkle,
                missing_publish,
                stale_catalogs,
                &stale_steps,
            );

            // Print advisories
            if !plan.advisories.is_empty() {
                println!();
                println!("To resolve ({}):", rel_display(ds_path));
                for advice in &plan.advisories {
                    println!("  {}", advice);
                }
            }

            if args.update_pipeline {
                if !plan.steps_to_add.is_empty() {
                    println!();
                    println!(
                        "Updating {} ({} step(s) to add)...",
                        rel_display(ds_path),
                        plan.steps_to_add.len(),
                    );
                    println!("A backup will be created in .backup/ before any changes.");
                    match fix::apply_fix(&plan) {
                        Ok(()) => {
                            println!("Pipeline updated. To execute new steps:\n  veks run {}", rel_display(ds_path));
                        }
                        Err(e) => {
                            eprintln!("Error applying fix: {}", e);
                        }
                    }
                }
            } else if !plan.steps_to_add.is_empty() {
                println!();
                println!(
                    "{} is missing {} pipeline step(s) (e.g., merkle tree generation).",
                    rel_display(ds_path),
                    plan.steps_to_add.len(),
                );
                println!(
                    "To auto-add them, re-run with:\n  veks check --update-pipeline {}",
                    rel_display(&args.directory),
                );
                println!("(A backup of dataset.yaml will be created in .backup/ before changes.)");
            }
        }
    }

    // Handle --clean-files: remove extraneous publishable files
    if args.clean_files {
        for ds_path in &dataset_files {
            let extra = extraneous::find_extraneous(ds_path, &publishable);
            if extra.is_empty() {
                println!("No extraneous files to clean.");
            } else {
                println!();
                println!("Removing {} extraneous file(s):", extra.len());
                for file in &extra {
                    let rel = file.strip_prefix(&directory).unwrap_or(file);
                    match std::fs::remove_file(file) {
                        Ok(()) => println!("  removed: {}", rel.display()),
                        Err(e) => eprintln!("  failed:  {} — {}", rel.display(), e),
                    }
                }
                // Clean up empty directories left behind
                clean_empty_dirs(&directory);
            }
        }
    }

    // Dataset readout for publish paths: show all discovered datasets
    // color-coded by publish status (.publish sentinel file).
    if is_publish_context && !args.quiet && !args.json {
        println!();
        println!("{}", crate::term::bold("Datasets in publish tree:"));
        let mut ds_entries: Vec<(String, bool, String)> = Vec::new(); // (name, publishable, rel_path)
        for ds_path in &dataset_files {
            if let Ok(config) = vectordata::dataset::DatasetConfig::load(ds_path) {
                let ds_dir = ds_path.parent().unwrap_or(std::path::Path::new("."));
                let publishable = ds_dir.join(".publish").exists();
                let rel = ds_path.strip_prefix(&directory)
                    .map(|r| r.parent().unwrap_or(r).to_string_lossy().to_string())
                    .unwrap_or_else(|_| rel_display(ds_path).to_string());
                ds_entries.push((config.name, publishable, rel));
            }
        }
        ds_entries.sort_by(|a, b| a.0.cmp(&b.0));
        let max_name = ds_entries.iter().map(|(n, _, _)| n.len()).max().unwrap_or(10);
        for (name, publishable, rel_path) in &ds_entries {
            let status = if *publishable { "publish" } else { "local" };
            let colored = if *publishable {
                crate::term::green(&format!("{:<width$}  {:<8}  {}", name, status, rel_path, width = max_name))
            } else {
                crate::term::dim(&format!("{:<width$}  {:<8}  {}", name, status, rel_path, width = max_name))
            };
            println!("  {}", colored);
        }
        let pub_count = ds_entries.iter().filter(|(_, p, _)| *p).count();
        let local_count = ds_entries.iter().filter(|(_, p, _)| !*p).count();
        println!();
        println!("  {} total: {} {} {}",
            ds_entries.len(),
            crate::term::green(&format!("{} publishable", pub_count)),
            crate::term::dim(&format!("{} local", local_count)),
            "(legend: publish=has .publish file, local=no .publish file)",
        );
    }

    std::process::exit(if all_passed { 0 } else { 1 });
}

/// Remove empty directories recursively (bottom-up), excluding hidden dirs.
/// Check that each dataset.yaml has the required attributes set.
fn check_dataset_attributes(dataset_files: &[PathBuf]) -> CheckResult {
    let mut missing_msgs: Vec<String> = Vec::new();

    for ds_path in dataset_files {
        let config = match vectordata::dataset::DatasetConfig::load(ds_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let ds_dir = ds_path.parent().unwrap_or(std::path::Path::new("."));
        let ds_rel = rel_display(ds_dir);

        match &config.attributes {
            None => {
                missing_msgs.push(format!(
                    "{}: attributes section missing (need is_zero_vector_free, is_duplicate_vector_free)",
                    ds_rel,
                ));
            }
            Some(attrs) => {
                let missing = attrs.missing_required();
                if !missing.is_empty() {
                    missing_msgs.push(format!(
                        "{}: missing required attributes: {}",
                        ds_rel, missing.join(", "),
                    ));
                }
            }
        }
    }

    if missing_msgs.is_empty() {
        let mut r = CheckResult::ok("dataset-attributes");
        r.messages.push("all required attributes present".to_string());
        r
    } else {
        CheckResult::fail("dataset-attributes", missing_msgs)
    }
}

fn clean_empty_dirs(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            clean_empty_dirs(&path);
            // Try to remove — will only succeed if empty
            let _ = std::fs::remove_dir(&path);
        }
    }
}

/// Discover all `dataset.yaml` files under a directory tree.
pub fn discover_datasets(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    discover_datasets_recursive(root, &mut found);
    found.sort();
    found
}

fn discover_datasets_recursive(dir: &Path, found: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if crate::filters::is_excluded_dir(&name_str) {
                continue;
            }
            discover_datasets_recursive(&path, found);
        } else if name_str == "dataset.yaml" || name_str == "dataset.yml" {
            found.push(path);
        }
    }
}

/// Display a path relative to the current working directory.
/// Falls back to the full path if stripping fails.
pub(crate) fn rel_display(path: &Path) -> String {
    if let Ok(cwd) = std::env::current_dir() {
        path.strip_prefix(&cwd)
            .map(|r| {
                let s = r.to_string_lossy().to_string();
                if s.is_empty() { ".".to_string() } else { s }
            })
            .unwrap_or_else(|_| path.to_string_lossy().to_string())
    } else {
        path.to_string_lossy().to_string()
    }
}

/// Enumerate all non-excluded files in a dataset workspace.
///
/// This is the dataset-context equivalent of `enumerate_publishable_files`.
/// It doesn't require `.publish` sentinels — it just walks the workspace
/// and applies the standard exclusion rules (hidden files, underscore prefix,
/// tmp/partial/pyc).
fn enumerate_workspace_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    enumerate_workspace_recursive(dir, &mut files);
    files.sort();
    files
}

fn enumerate_workspace_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if crate::filters::is_excluded_dir(&name_str) {
                continue;
            }
            enumerate_workspace_recursive(&path, files);
        } else {
            if crate::filters::is_excluded_file(&name_str) {
                continue;
            }
            files.push(path);
        }
    }
}

/// Walk up from `dir` looking for `.catalog_root`.
/// Returns a relative path from `dir` to the directory containing `.catalog_root`.
fn find_catalog_root_file(dir: &Path) -> Option<PathBuf> {
    let abs = std::fs::canonicalize(dir).unwrap_or(dir.to_path_buf());
    let mut current = abs.clone();
    let mut levels_up: usize = 0;

    loop {
        if current.join(".catalog_root").is_file() {
            let mut rel = dir.to_path_buf();
            for _ in 0..levels_up {
                rel = rel.join("..");
            }
            return Some(rel);
        }
        if !current.pop() {
            return None;
        }
        levels_up += 1;
    }
}

/// Parse a human-readable size string like "100M", "50MiB", "1G" into bytes.
pub fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Find where digits end and suffix begins
    let digit_end = s.find(|c: char| !c.is_ascii_digit() && c != '_' && c != '.')
        .unwrap_or(s.len());

    let num_str: String = s[..digit_end].chars().filter(|c| *c != '_').collect();
    let suffix = &s[digit_end..];

    let num: f64 = num_str.parse().ok()?;

    let multiplier: u64 = match suffix.to_uppercase().as_str() {
        "" => 1,
        "K" => 1_000,
        "M" => 1_000_000,
        "G" | "B" => 1_000_000_000,
        "T" => 1_000_000_000_000,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        "TB" => 1_000_000_000_000,
        "KIB" => 1_024,
        "MIB" => 1_048_576,
        "GIB" => 1_073_741_824,
        "TIB" => 1_099_511_627_776,
        _ => return None,
    };

    Some((num * multiplier as f64) as u64)
}

fn print_human(results: &[CheckResult]) {
    use crate::term;

    for result in results {
        if result.passed {
            println!("{} check {}: {}",
                term::ok("\u{2713}"),
                term::bold(result.name),
                term::ok("ok"),
            );
            for msg in &result.messages {
                println!("    {}", term::dim(msg));
            }
        } else {
            println!("{} check {}: {}",
                term::fail("\u{2717}"),
                term::bold(result.name),
                term::fail("FAILED"),
            );
            for msg in &result.messages {
                println!("    {}", msg);
            }
        }
    }
}

fn print_json(directory: &Path, results: &[CheckResult]) {
    let overall = if results.iter().all(|r| r.passed) { "ok" } else { "fail" };

    print!("{{\"directory\":");
    print_json_string(&directory.to_string_lossy());
    print!(",\"checks\":{{");
    for (i, r) in results.iter().enumerate() {
        if i > 0 { print!(","); }
        print_json_string(r.name);
        print!(":{{\"status\":");
        print_json_string(if r.passed { "ok" } else { "fail" });
        if !r.messages.is_empty() {
            print!(",\"messages\":[");
            for (j, m) in r.messages.iter().enumerate() {
                if j > 0 { print!(","); }
                print_json_string(m);
            }
            print!("]");
        }
        print!("}}");
    }
    print!("}},\"overall\":");
    print_json_string(overall);
    println!("}}");
}

fn print_json_string(s: &str) {
    print!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("100M"), Some(100_000_000));
        assert_eq!(parse_size("50MiB"), Some(52_428_800));
        assert_eq!(parse_size("1G"), Some(1_000_000_000));
        assert_eq!(parse_size("1GiB"), Some(1_073_741_824));
        assert_eq!(parse_size("500"), Some(500));
        assert_eq!(parse_size("10K"), Some(10_000));
    }

    #[test]
    fn test_parse_size_invalid() {
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("abc"), None);
    }
}
