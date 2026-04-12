// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Dataset publishing to S3 (`veks publish`).
//!
//! Synchronizes the publishable content of a local dataset directory to
//! a remote destination using the transport specified by the `.publish_url`
//! file. Currently supports S3 (`s3://`) via the AWS CLI.

mod transport;

use std::path::{Path, PathBuf};

use clap::Args;

use transport::{SyncOptions, transport_for_scheme};

/// Arguments for `veks publish`.
#[derive(Args)]
pub struct PublishArgs {
    /// Target directory to publish (default: current directory)
    #[arg(default_value = ".")]
    pub directory: PathBuf,

    /// Skip confirmation prompt
    #[arg(short = 'y')]
    pub yes: bool,

    /// Show what would be uploaded without transferring
    #[arg(long)]
    pub dry_run: bool,

    /// Remove remote objects that no longer exist locally
    #[arg(long)]
    pub delete: bool,

    /// Number of parallel upload streams
    #[arg(long, default_value = "4")]
    pub concurrency: u32,

    /// Additional glob patterns to exclude
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Additional glob patterns to force-include (overrides excludes)
    #[arg(long)]
    pub include: Vec<String>,

    /// Skip based on size only, ignoring timestamps
    #[arg(long)]
    pub size_only: bool,

    /// AWS profile name for credentials
    #[arg(long)]
    pub profile: Option<String>,

    /// Custom S3 endpoint (for S3-compatible stores)
    #[arg(long)]
    pub endpoint_url: Option<String>,

    /// Skip pre-flight checks
    #[arg(long)]
    pub no_check: bool,
}

use crate::filters;

/// Entry point for `veks publish`.
pub fn run(args: PublishArgs) {
    let directory = resolve_directory(&args.directory);

    if !directory.is_dir() {
        eprintln!("Error: '{}' is not a directory", crate::check::rel_display(&directory));
        std::process::exit(2);
    }

    // Locate .publish_url file
    let publish_file = crate::check::publish_url::find_publish_file(&directory);
    let publish_file = match publish_file {
        Some(p) => p,
        None => {
            eprintln!("Error: no .publish_url file found in '{}' or parent directories", crate::check::rel_display(&directory));
            eprintln!();
            eprintln!("Create one with:");
            eprintln!("  echo 's3://bucket-name/prefix/' > {}/.publish_url", crate::check::rel_display(&directory));
            std::process::exit(2);
        }
    };

    let content = std::fs::read_to_string(&publish_file).unwrap_or_else(|e| {
        eprintln!("Error: failed to read {}: {}", crate::check::rel_display(&publish_file), e);
        std::process::exit(2);
    });

    let parsed = match crate::check::publish_url::parse_publish_url(&content) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: invalid .publish_url at {}: {}", crate::check::rel_display(&publish_file), e);
            std::process::exit(2);
        }
    };
    // The publish root is the directory containing .publish_url.
    // The sync source should be the publish root, and the S3 destination
    // preserves the directory hierarchy from publish root down.
    let publish_root = publish_file.parent().unwrap().to_path_buf();
    let s3_url = parsed.url;

    // Check for parent publish roots — warn if this publish root is interior
    // to another (the user may intend to publish from the outer root instead)
    {
        let abs_root = std::fs::canonicalize(&publish_root).unwrap_or_else(|_| publish_root.clone());
        let mut walk = abs_root.clone();
        if walk.pop() { // skip self
            loop {
                let candidate = walk.join(".publish_url");
                if candidate.is_file() {
                    if let Ok(outer_content) = std::fs::read_to_string(&candidate) {
                        if let Ok(outer) = crate::check::publish_url::parse_publish_url(&outer_content) {
                            println!("{}", crate::term::warn("Note: this publish root is interior to another:"));
                            println!("  {} {} → {}",
                                crate::term::info("outer:"),
                                crate::check::rel_display(&walk),
                                crate::term::info(&outer.url));
                            println!("  {} {} → {}",
                                crate::term::info("local:"),
                                crate::check::rel_display(&publish_root),
                                crate::term::info(&s3_url));
                            println!("  Publishing from the {} publish root.",
                                crate::term::bold("local"));
                            println!();
                        }
                    }
                    break; // only report the nearest outer root
                }
                if !walk.pop() { break; }
            }
        }
    }

    // Run pre-flight checks unless --no-check
    if !args.no_check && !args.dry_run {
        let check_ok = run_preflight_checks(&directory);
        if !check_ok {
            eprintln!();
            eprintln!("Pre-flight checks failed. Run 'veks check' for details,");
            eprintln!("or use '--no-check' to override.");
            std::process::exit(1);
        }
    }

    // Enumerate publishable datasets and files from the publish root
    let mut dataset_dirs: Vec<PathBuf> = Vec::new();
    find_dataset_dirs(&publish_root, &mut dataset_dirs);
    dataset_dirs.sort();

    let publishable = enumerate_publishable_files(&publish_root);
    let total_size: u64 = publishable.iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();

    // Present summary and get confirmation
    println!("{}", crate::term::bold("Publish summary:"));
    println!("  Source:      {}", crate::check::rel_display(&publish_root));
    println!("  Destination: {}", crate::term::info(&s3_url));
    println!();
    println!("  Datasets ({}):", dataset_dirs.len());
    for ds_dir in &dataset_dirs {
        let rel = ds_dir.strip_prefix(&publish_root)
            .map(|r| r.to_string_lossy().to_string())
            .unwrap_or_else(|_| crate::check::rel_display(ds_dir));
        // Count files within this dataset
        let ds_file_count = publishable.iter()
            .filter(|f| f.starts_with(ds_dir))
            .count();
        let ds_size: u64 = publishable.iter()
            .filter(|f| f.starts_with(ds_dir))
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len())
            .sum();
        println!("    {} ({} files, {})", rel, ds_file_count, format_size(ds_size));
    }
    println!();
    println!("  Files:       {} to sync, {} total",
        publishable.len(),
        format_size(total_size),
    );
    if args.delete {
        println!("  Delete:      enabled (remote-only objects will be removed)");
    } else {
        println!("  Delete:      disabled");
    }
    println!();

    if args.dry_run {
        println!("(dry run — no changes will be made)");
        println!();
    } else if !args.yes {
        // Interactive confirmation
        if !crate::term::is_interactive() {
            eprintln!("Error: stdin is not a TTY. Use -y to skip confirmation.");
            std::process::exit(3);
        }

        eprint!("Proceed? Type YES to confirm: ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap_or(0);
        if input.trim().to_uppercase() != "YES" {
            eprintln!("Aborted.");
            std::process::exit(3);
        }
        println!();
    }

    // Select transport based on URL scheme
    let transport = match transport_for_scheme(&parsed.scheme) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(2);
        }
    };

    // Build the include list from enumerate_publishable_files — the single
    // source of truth for what gets published. The sync uses exclude-all +
    // include-only, so no separate exclude logic is needed.
    let include_files: Vec<String> = publishable.iter()
        .filter_map(|p| p.strip_prefix(&publish_root).ok())
        .map(|r| r.to_string_lossy().to_string())
        .collect();

    // Log infrastructure files for debugging publish completeness
    let infra_count = include_files.iter()
        .filter(|f| {
            let name = f.rsplit('/').next().unwrap_or(f);
            filters::is_infrastructure_file(name)
        })
        .count();
    println!("  Includes:    {} files ({} infrastructure: dataset.yaml, catalog.json, etc.)",
        include_files.len(), infra_count);
    if infra_count == 0 {
        eprintln!("  WARNING: no infrastructure files (dataset.yaml, catalog.json) in publish set!");
        eprintln!("           Check that .publish sentinel exists in the dataset directory.");
    }

    let sync_opts = SyncOptions {
        dry_run: args.dry_run,
        delete: args.delete,
        size_only: args.size_only,
        include_files: &include_files,
        profile: args.profile.as_deref(),
        endpoint_url: args.endpoint_url.as_deref(),
    };

    let exit_code = transport.sync(&publish_root, &s3_url, &sync_opts);
    std::process::exit(exit_code);
}

/// Run all pre-flight checks before publishing.
///
/// Runs the same checks as `veks check --check-all`: pipelines, publish,
/// merkle, integrity, and catalogs. Returns `true` only if every check passes.
fn run_preflight_checks(directory: &Path) -> bool {
    // Only check publishable datasets (those with .publish sentinel)
    let all_datasets = crate::check::discover_datasets(directory);
    let dataset_files: Vec<std::path::PathBuf> = all_datasets.into_iter()
        .filter(|ds| {
            let ds_dir = ds.parent().unwrap_or(std::path::Path::new("."));
            ds_dir.join(".publish").exists()
        })
        .collect();
    let publishable = crate::publish::enumerate_publishable_files(directory);
    let merkle_threshold = 0u64; // all files get merkle trees

    let checks: Vec<crate::check::CheckResult> = vec![
        crate::check::pipelines::check(directory, &dataset_files),
        crate::check::pipelines::check_coverage(directory, &dataset_files, &publishable, merkle_threshold),
        crate::check::publish_url::check(directory, &dataset_files),
        crate::check::merkle::check(directory, &publishable, merkle_threshold),
        crate::check::integrity::check(directory, &publishable),
        crate::check::catalogs::check(directory, &dataset_files),
    ];

    let mut all_ok = true;
    for result in &checks {
        if !result.passed {
            eprintln!("{} {}",
                veks_core::term::fail("Pre-flight check failed:"),
                veks_core::term::bold(&result.name));
            for msg in &result.messages {
                eprintln!("  {}", veks_core::term::red(msg));
            }
            all_ok = false;
        }
    }

    all_ok
}

/// Enumerate all publishable files under a directory (applying default exclusions).
pub fn enumerate_publishable_files(root: &Path) -> Vec<PathBuf> {
    // Step 1: discover all dataset directories (containing dataset.yaml)
    let mut dataset_dirs: Vec<PathBuf> = Vec::new();
    find_dataset_dirs(root, &mut dataset_dirs);

    // Step 2: collect all directories on the path from root to each dataset dir.
    // These are the "on-path" directories whose files participate in publishing.
    let mut on_path_dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    for ds_dir in &dataset_dirs {
        let mut current = ds_dir.clone();
        loop {
            on_path_dirs.insert(current.clone());
            if current == root { break; }
            if !current.pop() { break; }
        }
    }

    // Step 3: enumerate files within on-path directories and dataset subtrees
    let mut files = Vec::new();
    enumerate_recursive(root, root, &mut files, &on_path_dirs, &dataset_dirs);
    files.sort();
    files
}

/// Sentinel file that marks a dataset directory for publishing.
const PUBLISH_SENTINEL: &str = ".publish";

/// Recursively find directories containing dataset.yaml that are marked
/// for publishing (have a `.publish` sentinel file).
fn find_dataset_dirs(dir: &Path, result: &mut Vec<PathBuf>) {
    let yaml_path = dir.join("dataset.yaml");
    if yaml_path.exists() {
        // Only include datasets that have a .publish sentinel file
        if dir.join(PUBLISH_SENTINEL).exists() {
            result.push(dir.to_path_buf());
        }
        return; // don't descend into dataset directories for nested datasets
    }

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if path.is_dir() && !filters::is_excluded_dir(&name_str) {
                find_dataset_dirs(&path, result);
            }
        }
    }
}

fn enumerate_recursive(
    root: &Path, dir: &Path, files: &mut Vec<PathBuf>,
    on_path_dirs: &std::collections::HashSet<PathBuf>,
    dataset_dirs: &[PathBuf],
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Is this directory inside a dataset directory? If so, include all
    // non-excluded content (it's part of the dataset).
    let inside_dataset = dataset_dirs.iter().any(|ds| dir.starts_with(ds));

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if filters::is_excluded_dir(&name_str) {
                continue;
            }
            if inside_dataset {
                // Inside a dataset — descend into all non-excluded subdirs
                enumerate_recursive(root, &path, files, on_path_dirs, dataset_dirs);
            } else if on_path_dirs.contains(&path) || dataset_dirs.iter().any(|ds| ds.starts_with(&path)) {
                // On the path to a dataset — descend
                enumerate_recursive(root, &path, files, on_path_dirs, dataset_dirs);
            }
            // else: not on any dataset path — skip silently
        } else if !filters::is_excluded_file(&name_str) {
            if inside_dataset {
                // Inside a dataset — include all non-excluded files
                files.push(path);
            } else if on_path_dirs.contains(dir) && filters::is_intermediate_publishable(&name_str) {
                // On-path intermediate directory — only infrastructure files
                // (catalogs and their merkle companions)
                files.push(path);
            }
        }
    }
}

fn resolve_directory(dir: &Path) -> PathBuf {
    if dir.is_absolute() {
        dir.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(dir)
    }
}

/// Format a byte count as a human-readable string.
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
    use std::fs;
    use std::path::Path;

    /// Create a file (and parent dirs) with dummy content.
    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, "data").unwrap();
    }

    /// Create a valid dataset.yaml, optionally marked for publishing.
    fn write_dataset_yaml(dir: &Path, publishable: bool) {
        fs::create_dir_all(dir).unwrap();
        let name = dir.file_name().unwrap().to_string_lossy();
        fs::write(dir.join("dataset.yaml"), format!("name: {}\nprofiles: {{}}\n", name)).unwrap();
        if publishable {
            fs::write(dir.join(".publish"), "").unwrap();
        }
    }

    /// Get relative paths of publishable files from a root directory.
    fn publishable_rel(root: &Path) -> Vec<String> {
        enumerate_publishable_files(root)
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().to_string())
            .collect()
    }

    // ── File exclusion rules ──────────────────────────────────────

    #[test]
    fn excludes_hidden_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("ds"), true);
        touch(&root.join("ds/.hidden_file"));
        touch(&root.join("ds/visible.fvec"));
        let files = publishable_rel(root);
        assert!(files.contains(&"ds/visible.fvec".to_string()));
        assert!(files.contains(&"ds/dataset.yaml".to_string()));
        assert!(!files.iter().any(|f| f.contains(".hidden")));
    }

    #[test]
    fn excludes_underscore_prefixed_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("ds"), true);
        touch(&root.join("ds/_source.mvec"));
        touch(&root.join("ds/profiles/base/base.fvec"));
        let files = publishable_rel(root);
        assert!(!files.iter().any(|f| f.contains("_source")));
        assert!(files.iter().any(|f| f.contains("base.fvec")));
    }

    #[test]
    fn excludes_tmp_and_partial_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("ds"), true);
        touch(&root.join("ds/data.fvec"));
        touch(&root.join("ds/data.fvec.tmp"));
        touch(&root.join("ds/data.partial"));
        let files = publishable_rel(root);
        assert!(files.contains(&"ds/data.fvec".to_string()));
        assert!(!files.iter().any(|f| f.ends_with(".tmp")));
        assert!(!files.iter().any(|f| f.ends_with(".partial")));
    }

    #[test]
    fn excludes_pyc_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("ds"), true);
        touch(&root.join("ds/tool.pyc"));
        let files = publishable_rel(root);
        assert!(!files.iter().any(|f| f.ends_with(".pyc")));
    }

    // ── Directory exclusion rules ─────────────────────────────────

    #[test]
    fn excludes_hidden_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("ds"), true);
        touch(&root.join("ds/.cache/state.yaml"));
        touch(&root.join("ds/.scratch/tmp.bin"));
        touch(&root.join("ds/data.fvec"));
        let files = publishable_rel(root);
        assert!(files.contains(&"ds/data.fvec".to_string()));
        assert!(!files.iter().any(|f| f.contains(".cache")));
        assert!(!files.iter().any(|f| f.contains(".scratch")));
    }

    #[test]
    fn excludes_pycache_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("ds"), true);
        touch(&root.join("ds/__pycache__/module.pyc"));
        touch(&root.join("ds/data.fvec"));
        let files = publishable_rel(root);
        assert!(!files.iter().any(|f| f.contains("__pycache__")));
    }

    // ── Dataset path structure rules ──────────────────────────────

    #[test]
    fn includes_files_inside_dataset_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("myds"), true);
        touch(&root.join("myds/base.fvec"));
        touch(&root.join("myds/profiles/default/indices.ivec"));
        touch(&root.join("myds/profiles/1m/indices.ivec"));
        let files = publishable_rel(root);
        assert!(files.contains(&"myds/dataset.yaml".to_string()));
        assert!(files.contains(&"myds/base.fvec".to_string()));
        assert!(files.contains(&"myds/profiles/default/indices.ivec".to_string()));
        assert!(files.contains(&"myds/profiles/1m/indices.ivec".to_string()));
    }

    #[test]
    fn includes_catalog_files_on_intermediate_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("catalog.json"));
        touch(&root.join("group/catalog.json"));
        write_dataset_yaml(&root.join("group/ds"), true);
        touch(&root.join("group/ds/data.fvec"));
        let files = publishable_rel(root);
        assert!(files.contains(&"catalog.json".to_string()));
        assert!(files.contains(&"group/catalog.json".to_string()));
        assert!(files.contains(&"group/ds/data.fvec".to_string()));
    }

    #[test]
    fn excludes_data_files_at_intermediate_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Infrastructure at intermediate level — included
        touch(&root.join("group/catalog.json"));
        touch(&root.join("group/catalog.yaml"));
        touch(&root.join("group/catalog.json.mref"));
        // Data file at intermediate level — NOT a dataset, must be excluded
        touch(&root.join("group/base_test.mvec"));
        touch(&root.join("group/stray_data.fvec"));
        // Actual dataset content — included
        write_dataset_yaml(&root.join("group/ds"), true);
        touch(&root.join("group/ds/base.fvec"));
        let files = publishable_rel(root);
        // Infrastructure files at intermediate level
        assert!(files.contains(&"group/catalog.json".to_string()));
        assert!(files.contains(&"group/catalog.yaml".to_string()));
        assert!(files.contains(&"group/catalog.json.mref".to_string()));
        // Dataset content
        assert!(files.contains(&"group/ds/base.fvec".to_string()));
        // Data files at intermediate level must NOT be published
        assert!(!files.iter().any(|f| f.contains("base_test")));
        assert!(!files.iter().any(|f| f.contains("stray_data")));
    }

    #[test]
    fn excludes_directories_not_on_dataset_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("datasets/ds1"), true);
        touch(&root.join("datasets/ds1/data.fvec"));
        touch(&root.join("sourcedata/raw/big_file.npy"));
        touch(&root.join("scripts/convert.sh"));
        touch(&root.join("tools/helper.py"));
        let files = publishable_rel(root);
        assert!(files.contains(&"datasets/ds1/data.fvec".to_string()));
        assert!(!files.iter().any(|f| f.contains("sourcedata")));
        assert!(!files.iter().any(|f| f.contains("scripts")));
        assert!(!files.iter().any(|f| f.contains("tools")));
    }

    #[test]
    fn excludes_sibling_non_dataset_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("group/ds1"), true);
        touch(&root.join("group/ds1/data.fvec"));
        touch(&root.join("group/scratch/temp.bin"));
        touch(&root.join("group/raw_data/source.npy"));
        let files = publishable_rel(root);
        assert!(files.contains(&"group/ds1/data.fvec".to_string()));
        assert!(!files.iter().any(|f| f.contains("scratch")));
        assert!(!files.iter().any(|f| f.contains("raw_data")));
    }

    // ── Multiple dataset directories ──────────────────────────────

    #[test]
    fn includes_multiple_datasets_under_same_parent() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("group/ds1"), true);
        touch(&root.join("group/ds1/base.fvec"));
        write_dataset_yaml(&root.join("group/ds2"), true);
        touch(&root.join("group/ds2/base.fvec"));
        touch(&root.join("group/not_a_dataset/random.bin"));
        let files = publishable_rel(root);
        assert!(files.contains(&"group/ds1/base.fvec".to_string()));
        assert!(files.contains(&"group/ds2/base.fvec".to_string()));
        assert!(!files.iter().any(|f| f.contains("not_a_dataset")));
    }

    #[test]
    fn includes_datasets_at_different_depths() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("shallow"), true);
        touch(&root.join("shallow/data.fvec"));
        write_dataset_yaml(&root.join("deep/nested/dir"), true);
        touch(&root.join("deep/nested/dir/data.fvec"));
        touch(&root.join("deep/other/stuff.bin"));
        let files = publishable_rel(root);
        assert!(files.contains(&"shallow/data.fvec".to_string()));
        assert!(files.contains(&"deep/nested/dir/data.fvec".to_string()));
        assert!(!files.iter().any(|f| f.contains("other")));
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn dataset_at_root_includes_everything() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(root, true);
        touch(&root.join("base.fvec"));
        touch(&root.join("profiles/default/indices.ivec"));
        let files = publishable_rel(root);
        assert!(files.contains(&"dataset.yaml".to_string()));
        assert!(files.contains(&"base.fvec".to_string()));
        assert!(files.contains(&"profiles/default/indices.ivec".to_string()));
    }

    #[test]
    fn empty_root_returns_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let files = enumerate_publishable_files(tmp.path());
        assert!(files.is_empty());
    }

    #[test]
    fn no_datasets_returns_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("random_file.bin"));
        touch(&root.join("subdir/another.bin"));
        let files = enumerate_publishable_files(root);
        assert!(files.is_empty());
    }

    // ── Access level rules ────────────────────────────────────────

    #[test]
    fn excludes_datasets_without_publish_sentinel() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("published_ds"), true);
        touch(&root.join("published_ds/data.fvec"));
        write_dataset_yaml(&root.join("local_ds"), false);
        touch(&root.join("local_ds/data.fvec"));
        let files = publishable_rel(root);
        assert!(files.iter().any(|f| f.contains("published_ds")));
        assert!(!files.iter().any(|f| f.contains("local_ds")));
    }

    #[test]
    fn excludes_datasets_without_publish_file_default() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("with_publish"), true);
        touch(&root.join("with_publish/data.fvec"));
        // Dataset without .publish file — not publishable
        write_dataset_yaml(&root.join("no_publish"), false);
        touch(&root.join("no_publish/data.fvec"));
        let files = publishable_rel(root);
        assert!(files.iter().any(|f| f.contains("with_publish")));
        assert!(!files.iter().any(|f| f.contains("no_publish")));
    }

    #[test]
    fn mixed_publish_sentinels() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("group/pub1"), true);
        touch(&root.join("group/pub1/data.fvec"));
        write_dataset_yaml(&root.join("group/pub2"), true);
        touch(&root.join("group/pub2/data.fvec"));
        write_dataset_yaml(&root.join("group/local1"), false);
        touch(&root.join("group/local1/data.fvec"));
        let files = publishable_rel(root);
        assert!(files.iter().any(|f| f.contains("pub1")));
        assert!(files.iter().any(|f| f.contains("pub2")));
        assert!(!files.iter().any(|f| f.contains("local1")));
    }

    #[test]
    fn no_publishable_datasets_returns_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_dataset_yaml(&root.join("ds1"), false);
        touch(&root.join("ds1/data.fvec"));
        write_dataset_yaml(&root.join("ds2"), false);
        touch(&root.join("ds2/data.fvec"));
        let files = enumerate_publishable_files(root);
        assert!(files.is_empty());
    }
}
