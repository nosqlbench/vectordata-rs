// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! `datasets stratify` — add sized profiles to an existing dataset.
//!
//! Reads an existing `dataset.yaml`, determines the base vector count,
//! builds sized profile views from the default profile facets, and writes
//! back the updated config. A backup of the original file is created
//! before any modification.
//!
//! This produces the same sized profile structure that the import wizard
//! would have generated if the user had opted for sized profiles during
//! initial import — but without requiring a full re-import.

use std::path::{Path, PathBuf};
use std::io::{self, Write};

use indexmap::IndexMap;
use vectordata::dataset::config::DatasetConfig;
use vectordata::dataset::profile::{DSProfile, DSView};
use vectordata::dataset::source::{DSSource, DSWindow, DSInterval};

use crate::formats::VecFormat;

/// Classification of how a facet participates in sized profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FacetRole {
    /// Windowed by base_count range (e.g., base_vectors, metadata_content)
    Windowed,
    /// Each sized profile gets its own directory (e.g., neighbor_indices)
    PerProfile,
    /// Shared across all profiles (e.g., query_vectors, predicates)
    Shared,
}

/// Classify a facet by its role in sized profiles.
pub(crate) fn classify_facet(name: &str) -> FacetRole {
    match name {
        "base_vectors" | "metadata_content" => FacetRole::Windowed,
        "neighbor_indices" | "neighbor_distances"
        | "filtered_neighbor_indices" | "filtered_neighbor_distances"
        | "metadata_indices" => FacetRole::PerProfile,
        _ => FacetRole::Shared,
    }
}

/// Run the stratify command.
pub fn run(path: &Path, spec: Option<&str>, force: bool, yes: bool) {
    let (dataset_dir, dataset_path) = resolve_dataset_path(path);

    // Load existing config
    let mut config = DatasetConfig::load(&dataset_path).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    // Verify default profile exists with base_vectors
    let default = match config.profiles.default_profile() {
        Some(p) => p.clone(),
        None => {
            eprintln!("Error: dataset.yaml has no 'default' profile");
            std::process::exit(1);
        }
    };

    if default.view("base_vectors").is_none() {
        eprintln!("Error: default profile has no base_vectors facet");
        std::process::exit(1);
    }

    // Check for existing sized profiles (any profile with base_count set)
    let existing_sized: Vec<&str> = config.profiles.profiles.iter()
        .filter(|(name, p)| name.as_str() != "default" && p.base_count.is_some())
        .map(|(name, _)| name.as_str())
        .collect();

    if !existing_sized.is_empty() && !force {
        eprintln!("Error: dataset already has sized profiles: {:?}", existing_sized);
        eprintln!("Use --force to overwrite them.");
        std::process::exit(1);
    }

    // Remove existing sized profiles if --force
    if !existing_sized.is_empty() && force {
        let to_remove: Vec<String> = existing_sized.iter().map(|s| s.to_string()).collect();
        for name in &to_remove {
            config.profiles.profiles.swap_remove(name);
        }
        println!("Removed {} existing sized profile(s)", to_remove.len());
    }

    // Determine base vector count
    let base_vector_count = determine_base_count(&dataset_dir, &config);
    if base_vector_count == 0 {
        eprintln!("Error: could not determine base vector count");
        eprintln!("Run the pipeline first to compute variables, or provide a count in variables.yaml");
        std::process::exit(1);
    }

    // Detect self-search mode
    let is_self_search = detect_self_search(&config);
    let query_count = config.upstream.as_ref()
        .and_then(|u| u.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let effective_max = if is_self_search {
        base_vector_count.saturating_sub(query_count)
    } else {
        base_vector_count
    };

    let max_label = format_count_label(effective_max);

    let spec_str = if let Some(s) = spec {
        s.to_string()
    } else if yes {
        standard_spec(effective_max)
    } else {
        // Simple prompt with standard spec as default
        let default_spec = standard_spec(effective_max);
        println!("--- Sized profiles ---");
        println!("  Base vector count: ~{} ({})", max_label, effective_max);
        if is_self_search {
            println!("  (after subtracting {} query vectors)", query_count);
        }
        println!();
        println!("  Standard spec: {}", default_spec);
        println!("  (Use --interactive for guided setup)");
        eprint!("  Spec (Enter for standard): ");
        io::stderr().flush().unwrap_or(());
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap_or(0);
        let trimmed = input.trim();
        if trimmed.is_empty() {
            default_spec
        } else {
            trimmed.to_string()
        }
    };

    println!("  Spec: {}", spec_str);

    // Parse the spec into (name, base_count) pairs
    let pairs = parse_spec_to_pairs(&spec_str);
    if pairs.is_empty() {
        eprintln!("Error: spec produced no profiles");
        std::process::exit(1);
    }

    // Filter out profiles larger than effective_max
    let pairs: Vec<(String, u64)> = pairs.into_iter()
        .filter(|(_, count)| *count <= effective_max)
        .collect();

    println!("  Generating {} sized profiles:", pairs.len());
    for (name, count) in &pairs {
        println!("    {} — {} base vectors", name, count);
    }

    // Build sized profile views from the default profile
    for (prof_name, count) in &pairs {
        let mut views = IndexMap::new();

        for (facet_name, view) in &default.views {
            let role = classify_facet(facet_name);
            let path_str = &view.source.path;

            match role {
                FacetRole::Windowed => {
                    // Window the source path with the profile's range
                    let windowed_source = DSSource {
                        path: path_str.clone(),
                        namespace: view.source.namespace.clone(),
                        window: DSWindow(vec![DSInterval {
                            min_incl: 0,
                            max_excl: *count,
                        }]),
                    };
                    views.insert(facet_name.clone(), DSView {
                        source: windowed_source,
                        window: None,
                    });
                }
                FacetRole::PerProfile => {
                    // Replace "profiles/default/" with "profiles/{name}/"
                    let new_path = path_str.replace(
                        "profiles/default/",
                        &format!("profiles/{}/", prof_name),
                    );
                    let source = DSSource {
                        path: new_path,
                        namespace: view.source.namespace.clone(),
                        window: DSWindow::default(),
                    };
                    views.insert(facet_name.clone(), DSView {
                        source,
                        window: None,
                    });
                }
                FacetRole::Shared => {
                    views.insert(facet_name.clone(), view.clone());
                }
            }
        }

        let profile = DSProfile {
            maxk: default.maxk,
            base_count: Some(*count),
            views,
        };
        config.profiles.profiles.insert(prof_name.clone(), profile);
    }

    // Backup the existing file
    let backup = crate::check::fix::create_backup(&dataset_path);
    match backup {
        Ok(ref bp) => println!("  Backed up {} → {}", crate::check::rel_display(&dataset_path), crate::check::rel_display(bp)),
        Err(ref e) => eprintln!("  Warning: backup failed: {}", e),
    }

    // Serialize and write
    let yaml = serde_yaml::to_string(&config).unwrap_or_else(|e| {
        eprintln!("Error: failed to serialize config: {}", e);
        std::process::exit(1);
    });

    let tmp_path = dataset_path.with_extension("yaml.tmp");
    if let Err(e) = std::fs::write(&tmp_path, &yaml) {
        eprintln!("Error: failed to write {}: {}", crate::check::rel_display(&tmp_path), e);
        std::process::exit(1);
    }
    if let Err(e) = std::fs::rename(&tmp_path, &dataset_path) {
        eprintln!("Error: failed to rename: {}", e);
        std::process::exit(1);
    }

    // Touch the progress log so that the pipeline runner doesn't
    // invalidate all existing step records just because dataset.yaml is
    // newer. Stratification adds profiles but doesn't change existing
    // pipeline steps — previously completed steps remain valid.
    let progress_path = dataset_dir.join(".cache/.upstream.progress.yaml");
    if progress_path.exists() {
        let _ = filetime::set_file_mtime(
            &progress_path,
            filetime::FileTime::now(),
        );
    }

    println!();
    println!("Updated {} with {} sized profiles", crate::check::rel_display(&dataset_path), pairs.len());
    println!();
    println!("To compute ground truth for the new profiles, run:");
    println!("  veks run");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the dataset directory and dataset.yaml path.
fn resolve_dataset_path(path: &Path) -> (PathBuf, PathBuf) {
    if path.is_file() {
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        (dir, path.to_path_buf())
    } else {
        let yaml = path.join("dataset.yaml");
        if !yaml.exists() {
            eprintln!("Error: no dataset.yaml found in {}", crate::check::rel_display(&path.to_path_buf()));
            std::process::exit(1);
        }
        (path.to_path_buf(), yaml)
    }
}

/// Determine the base vector count from variables.yaml or by probing the file.
fn determine_base_count(dataset_dir: &Path, config: &DatasetConfig) -> u64 {
    // Try variables.yaml first
    if let Ok(vars) = crate::pipeline::variables::load(dataset_dir) {
        // Prefer clean_count (post-dedup), then vector_count (pre-dedup)
        if let Some(v) = vars.get("clean_count").or_else(|| vars.get("vector_count")) {
            if let Ok(n) = v.parse::<u64>() {
                return n;
            }
        }
    }

    // Fallback: probe the base_vectors file
    let default = match config.profiles.default_profile() {
        Some(p) => p,
        None => return 0,
    };
    let bv = match default.view("base_vectors") {
        Some(v) => &v.source.path,
        None => return 0,
    };

    let bv_path = if Path::new(bv).is_absolute() {
        PathBuf::from(bv)
    } else {
        dataset_dir.join(bv)
    };

    probe_vector_count(&bv_path).unwrap_or(0)
}

/// Probe a vector file for its record count.
fn probe_vector_count(path: &Path) -> Option<u64> {
    let format = VecFormat::detect(path)?;
    if path.is_dir() { return None; }
    let meta = crate::formats::reader::probe_source(path, format).ok()?;
    meta.record_count
}

/// Detect if the dataset uses self-search mode.
fn detect_self_search(config: &DatasetConfig) -> bool {
    if let Some(ref upstream) = config.upstream {
        if let Some(ref steps) = upstream.steps {
            for step in steps {
                let id = step.effective_id();
                if id.contains("shuffle") || id.contains("extract-query") || id.contains("extract-base") {
                    return true;
                }
            }
        }
    }
    false
}

/// Parse a spec string into (name, base_count) pairs.
fn parse_spec_to_pairs(spec: &str) -> Vec<(String, u64)> {
    let specs: Vec<&str> = spec.split(',').map(|s| s.trim()).collect();
    let mut all_pairs = Vec::new();
    for entry in specs {
        if let Ok(pairs) = vectordata::dataset::profile::parse_sized_entry(entry) {
            all_pairs.extend(pairs);
        } else {
            eprintln!("Warning: failed to parse spec '{}'", entry);
        }
    }
    all_pairs.sort_by_key(|(_, count)| *count);
    all_pairs.dedup_by(|a, b| a.0 == b.0);
    all_pairs
}

/// Format a count as a human-readable label.
/// Build the standard spec: fib + mul/2 + linear(10m), deduped.
///
/// This is the default for `--auto` and option 1 in the wizard.
/// Build a standard sized profile spec. The upper bound is implicit —
/// profiles are only valid for sizes ≤ the default profile's base count.
pub(crate) fn standard_spec(effective_max: u64) -> String {
    if effective_max >= 2_000_000 {
        // Use binary units so the doubling series produces clean IEC names:
        // 1mi, 2mi, 4mi, ..., 512mi, 1gi, 2gi, ...
        "mul:1mi/2".to_string()
    } else if effective_max >= 100_000 {
        let start = format_count_label((effective_max / 10).max(1000));
        format!("mul:{}/2", start)
    } else if effective_max >= 10_000 {
        let half = format_count_label(effective_max / 2);
        half
    } else {
        format_count_label(effective_max / 2)
    }
}

/// Interactive wizard for choosing sized profile specs.
pub fn interactive_spec_wizard(path: &Path) -> String {
    let (dataset_dir, dataset_path) = resolve_dataset_path(path);
    let config = DatasetConfig::load(&dataset_path).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let base_count = determine_base_count(&dataset_dir, &config);
    if base_count == 0 {
        eprintln!("Error: could not determine base vector count");
        std::process::exit(1);
    }

    let is_self_search = detect_self_search(&config);
    let query_count = config.upstream.as_ref()
        .and_then(|u| u.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let effective_max = if is_self_search {
        base_count.saturating_sub(query_count)
    } else {
        base_count
    };

    let max_label = format_count_label(effective_max);
    let std_spec = standard_spec(effective_max);

    println!();
    println!("--- Sized Profile Wizard ---");
    println!();
    println!("  Dataset: {}", config.name);
    println!("  Base vectors: ~{} ({})", max_label, effective_max);
    if is_self_search {
        println!("  Self-search mode: {} query vectors subtracted", query_count);
    }

    // Preview standard spec
    let std_pairs = parse_spec_to_pairs(&std_spec);
    let std_filtered: Vec<_> = std_pairs.iter().filter(|(_, c)| *c <= effective_max).collect();
    println!();
    println!("  Standard spec: {}", std_spec);
    println!("  ({} profiles: {})",
        std_filtered.len(),
        std_filtered.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));

    println!();
    println!("  Choose a strategy:");
    println!();
    println!("    1) Standard — fib + mul/2 + linear/10m (recommended)");
    println!("    2) Multiplicative only (powers of 2)");
    println!("    3) Fibonacci only");
    println!("    4) Linear only (fixed step)");
    println!("    5) Custom spec");
    println!();

    let choice = prompt_with_default("  Strategy [1-5]", "1");

    let spec = match choice.trim() {
        "1" => std_spec,
        "2" => {
            let start = prompt_with_default(
                &format!("  Start size [max={}]", max_label),
                "1mi",
            );
            format!("mul:{}..{}/2", start.trim(), max_label)
        }
        "3" => {
            let start = prompt_with_default(
                &format!("  Start size [max={}]", max_label),
                "1m",
            );
            format!("fib:{}..{}", start.trim(), max_label)
        }
        "4" => {
            let step = prompt_with_default(
                &format!("  Step size [max={}]", max_label),
                &format_count_label((effective_max / 10).max(1_000_000)),
            );
            format!("0m..{}/{}", max_label, step.trim())
        }
        "5" => {
            let custom = prompt_with_default("  Spec string", &std_spec);
            custom.trim().to_string()
        }
        _ => {
            eprintln!("  Unknown choice, using standard.");
            std_spec
        }
    };

    // Preview
    let pairs = parse_spec_to_pairs(&spec);
    let filtered: Vec<_> = pairs.iter().filter(|(_, c)| *c <= effective_max).collect();
    println!();
    println!("  Preview: {} profiles from spec '{}'", filtered.len(), spec);
    for (name, count) in &filtered {
        println!("    {:<10} {:>12} vectors", name, count);
    }

    if filtered.is_empty() {
        eprintln!("  Warning: spec produced no valid profiles within the size limit.");
    }

    println!();
    spec
}

fn prompt_with_default(prompt: &str, default: &str) -> String {
    use std::io::Write;
    if default.is_empty() {
        eprint!("{}: ", prompt);
    } else {
        eprint!("{} [{}]: ", prompt, default);
    }
    io::stderr().flush().unwrap_or(());
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    let trimmed = input.trim();
    if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    }
}

fn format_count_label(n: u64) -> String {
    if n >= 1_000_000_000 && n % 1_000_000_000 == 0 {
        format!("{}b", n / 1_000_000_000)
    } else if n >= 1_000_000 && n % 1_000_000 == 0 {
        format!("{}m", n / 1_000_000)
    } else if n >= 1_000 && n % 1_000 == 0 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}
