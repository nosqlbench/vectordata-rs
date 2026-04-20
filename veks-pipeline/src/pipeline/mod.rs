// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Command stream pipeline framework for veks.
//!
//! This module provides a declarative YAML-driven pipeline system that can
//! describe multi-step transformation pipelines embedded in `dataset.yaml`.
//! Features include:
//!
//! - DAG-based step ordering with automatic dependency detection
//! - Skip-if-fresh semantics via a persistent progress log
//! - Variable interpolation (`${name}`, `${name:-fallback}`, `${env:VAR}`)
//! - Dry-run mode for validation
//! - Extensible command registry
//!
//! ## Usage
//!
//! The primary entry point is [`run_pipeline`], invoked by `veks run`:
//!
//! ```text
//! veks run dataset.yaml              # execute pipeline
//! veks run dataset.yaml --dry-run    # validate only
//! veks run dataset.yaml --clean      # delete progress + intermediates
//! ```

pub mod atomic_write;
pub mod bound;
pub mod cli;
pub mod command;
pub mod commands;
pub mod dag;
pub mod element_type;
pub mod gz_cache;
pub mod interpolate;
pub mod manifest;
pub mod resource;
pub mod rng;
pub mod simd_distance;
pub mod progress;
pub mod registry;
pub mod runner;
pub mod schema;
pub mod variables;

/// Return the number of physical CPU cores (not hyperthreads).
///
/// Falls back to `available_parallelism` (logical CPUs) if the physical
/// count cannot be determined.
pub(crate) fn physical_core_count() -> usize {
    // Linux: parse /sys/devices/system/cpu/cpu*/topology/core_id to count
    // unique (socket, core) pairs.
    if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") {
        use std::collections::HashSet;
        let mut cores = HashSet::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with("cpu")
                || !name_str[3..]
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
            {
                continue;
            }
            let topo = entry.path().join("topology");
            let socket = std::fs::read_to_string(topo.join("physical_package_id"))
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok());
            let core = std::fs::read_to_string(topo.join("core_id"))
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok());
            if let (Some(s), Some(c)) = (socket, core) {
                cores.insert((s, c));
            }
        }
        if !cores.is_empty() {
            return cores.len();
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

use std::path::{Path, PathBuf};

use clap::Args;
use clap_complete::engine::ArgValueCompleter;
use indexmap::IndexMap;

use vectordata::dataset::DatasetConfig;
use vectordata::dataset::pipeline::StepDef;
use command::StreamContext;
use progress::ProgressLog;
use registry::CommandRegistry;

/// Resolve the full expanded step list for a dataset, including deferred
/// profile expansion from variables.yaml. This is the single source of
/// truth for step resolution — used by `run`, `check`, `publish`, and
/// `dry-run`.
///
/// Returns the expanded steps and the mutated config (with profiles resolved).
pub fn resolve_all_steps(
    config: &mut DatasetConfig,
    workspace: &Path,
) -> Vec<StepDef> {
    let query_count: u64 = config.upstream.as_ref()
        .and_then(|p| p.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    // Resolve deferred sized profiles. base_count is required — profile
    // expansion must know the actual dataset size so it never generates
    // invalid profiles. Check both variables.yaml and config.variables
    // (the dataset.yaml variables: section).
    if config.profiles.has_deferred() {
        let mut var_map: indexmap::IndexMap<String, String> = config.variables.clone();
        if let Ok(file_vars) = variables::load(workspace) {
            for (k, v) in file_vars {
                var_map.insert(k, v);
            }
        }
        let base_count: u64 = var_map.get("base_count")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if base_count > 0 {
            let added = config.profiles.expand_deferred_sized(&var_map, base_count);
            if added > 0 {
                log::info!("Resolved {} deferred sized profiles (base_count={})", added, base_count);
            }
        } else {
            log::warn!("Cannot expand sized profiles: base_count not yet known");
        }
    }

    // Detect oracle sub-facet scope from the pipeline steps.
    // If a partition-profiles step exists, determine the scope from the
    // facet spec stored in the oracle_scope attribute, or default to BQG.
    let raw_steps = vectordata::dataset::collect_all_steps(config);
    let has_partition_step = raw_steps.iter().any(|s| s.effective_id() == "partition-profiles");
    let oracle_scope = if has_partition_step {
        config.attributes.as_ref()
            .and_then(|a| a.tags.get("oracle_scope"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "BQG".to_string())
    } else {
        String::new()
    };
    let oracle_ref = if has_partition_step { Some(oracle_scope.as_str()) } else { None };

    let expanded = vectordata::dataset::expand_per_profile_steps_scoped(
        raw_steps, &config.profiles, query_count, oracle_ref,
    );

    // Auto-derive profile views for sized profiles from template outputs
    let template_steps: Vec<_> = vectordata::dataset::collect_all_steps(config)
        .into_iter()
        .filter(|s| s.per_profile)
        .collect();
    config.profiles.derive_views_from_templates(&template_steps);

    expanded
}

/// CLI arguments for `veks run`.
#[derive(Args)]
pub struct RunArgs {
    /// Path to dataset.yaml, or a directory when --recursive is given
    /// (default: dataset.yaml in current directory)
    pub dataset: Option<PathBuf>,

    /// Recursively find and run all dataset.yaml files under the target
    /// directory. The positional argument is treated as a root directory
    /// instead of a single dataset.yaml path.
    #[arg(long, short = 'r')]
    pub recursive: bool,

    /// Run steps for a specific profile, or `all` to run every profile
    /// with barriers between them. Steps with a `profiles` field are gated:
    /// they run only when the active profile is listed. Steps without a
    /// `profiles` field are shared and always run.
    #[arg(long, default_value = "all")]
    pub profile: String,

    /// Validate the pipeline without executing any steps.
    #[arg(long)]
    pub dry_run: bool,

    /// Delete progress log and intermediate files, then exit.
    #[arg(long)]
    pub clean: bool,

    /// Rewind progress by one step: remove only the artifacts of the
    /// most recently completed (or partially completed) step and its
    /// progress entry, then exit. Lets you re-run just the last step
    /// — typically because its inputs changed, the run aborted, or
    /// you want to retry it with different options — without
    /// invalidating any of the earlier steps' work.
    #[arg(long)]
    pub clean_last: bool,

    /// Full reset: remove progress log, .cache/, generated profile data,
    /// and pipeline-produced facet files (neighbors, distances, ordinals,
    /// etc.), then re-run the pipeline from scratch. Preserves
    /// dataset.yaml, variables.yaml, identity symlinks, and source data.
    #[arg(long)]
    pub reset: bool,

    /// Override default variables (format: key=value). Can be specified
    /// multiple times.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub overrides: Vec<String>,

    /// Number of threads for parallel operations (0 = auto).
    #[arg(long, default_value = "0")]
    pub threads: usize,

    /// Emit the fully resolved pipeline as YAML instead of executing it.
    ///
    /// Loads the dataset config, merges defaults, validates the DAG, and
    /// interpolates all step options, then prints the resulting
    /// `PipelineConfig` to stdout.
    #[arg(long)]
    pub emit_yaml: bool,


    /// Resource configuration for the governor (e.g., "mem:25%-50%,threads:4-8").
    ///
    /// Controls how much of the system's resources the pipeline may use.
    /// Supports absolute values, percentages, and ranges that the governor
    /// adjusts dynamically during execution.
    #[arg(long, num_args = 1, add = ArgValueCompleter::new(cli::resource_completer))]
    pub resources: Option<String>,

    /// Governor strategy for resource adjustment.
    ///
    /// - `maximize` (default): aggressively use resources up to the configured ceiling
    /// - `conservative`: start at floor values, only increase after sustained low utilization
    /// - `fixed`: use midpoint values, never adjust (useful for benchmarking)
    #[arg(long, default_value = "maximize", value_parser = ["maximize", "conservative", "fixed"])]
    pub governor: String,

    /// Interval in milliseconds between UI status updates and TUI redraws.
    #[arg(long, default_value = "250", value_name = "MS")]
    pub status_interval: u64,

    /// Output mode: tui (rich terminal, default when TTY), basic (plain
    /// text progress on stderr), batch (log-only, no console output).
    /// Auto-detected when not specified: tui if TTY, basic otherwise.
    #[arg(long, default_value = "auto", value_parser = ["auto", "tui", "basic", "batch"])]
    pub output: String,
}

/// CLI arguments for `veks script`.
#[derive(Args)]
pub struct ScriptArgs {
    /// Path to dataset.yaml (default: dataset.yaml in current directory)
    pub dataset: Option<PathBuf>,

    /// Emit steps for a specific profile, or `all` for every profile.
    #[arg(long, default_value = "all")]
    pub profile: String,

    /// Comment out steps that are already recorded as completed in the
    /// progress log, prefixed with `# DONE:`.
    #[arg(long)]
    pub show_progress: bool,

    /// Emit all file paths as absolute paths. Without this flag, paths
    /// are relative to the dataset workspace directory.
    #[arg(long)]
    pub absolute: bool,
}

/// Entry point for `veks script` — emit a pipeline as CLI commands.
pub fn run_script(args: ScriptArgs) {
    let dataset_path = args.dataset.unwrap_or_else(|| {
        let default = PathBuf::from("dataset.yaml");
        if default.exists() {
            println!("Using dataset.yaml in current directory");
            default
        } else {
            eprintln!("Error: no dataset.yaml specified and none found in current directory");
            std::process::exit(1);
        }
    });
    let dataset_path = &dataset_path;

    let config = DatasetConfig::load(dataset_path).unwrap_or_else(|e| {
        println!("{}", e);
        std::process::exit(1);
    });

    let raw_workspace = dataset_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf();

    // Resolve workspace: --absolute makes all paths absolute; otherwise
    // keep as-is (may be relative to CWD).
    let workspace = if args.absolute {
        std::fs::canonicalize(&raw_workspace).unwrap_or_else(|e| {
            println!("Failed to resolve absolute path for {}: {}", veks_core::paths::rel_display(&raw_workspace), e);
            std::process::exit(1);
        })
    } else {
        raw_workspace
    };

    // Determine if workspace is the current directory — if so, --workspace
    // can be elided from the generated commands.
    let cwd = std::env::current_dir().unwrap_or_default();
    let workspace_is_cwd = std::fs::canonicalize(&workspace)
        .ok()
        .map(|w| std::fs::canonicalize(&cwd).ok().map(|c| w == c).unwrap_or(false))
        .unwrap_or(false);

    let profile_name = &args.profile;

    let mut config = config; // make mutable for resolve_all_steps
    let expanded_steps = resolve_all_steps(&mut config, &workspace);

    let steps = if profile_name == "all" {
        expanded_steps
    } else {
        vectordata::dataset::filter_steps_for_profile(expanded_steps, profile_name)
    };

    if steps.is_empty() {
        println!(
            "No pipeline steps found for profile '{}' in {}",
            profile_name,
            veks_core::paths::rel_display(&dataset_path)
        );
        std::process::exit(1);
    }

    let pipeline_dag = dag::build_dag(&steps).unwrap_or_else(|e| {
        println!("Pipeline DAG error: {}", e);
        std::process::exit(1);
    });

    // Build defaults from upstream config + variables.yaml (no CLI overrides
    // for script mode — the script itself is the reproducible artifact).
    let mut defaults = IndexMap::new();
    if let Some(ref pipe) = config.upstream {
        if let Some(ref defs) = pipe.defaults {
            defaults.extend(defs.clone());
        }
    }
    match variables::load(&workspace) {
        Ok(vars) => defaults.extend(vars),
        Err(_) => {}
    }

    // Optionally load progress log for --show-progress
    let progress = if args.show_progress {
        let progress_path = ProgressLog::path_for_dataset(dataset_path);
        ProgressLog::load(&progress_path).ok().map(|(log, _)| log)
    } else {
        None
    };

    emit_cli_commands(
        &pipeline_dag,
        &defaults,
        &workspace,
        progress.as_ref(),
        workspace_is_cwd,
        args.absolute,
    );
}

/// Entry point for `veks run` — execute a command stream pipeline.
///
/// Returns `Ok(())` on success, `Err(message)` on failure. Callers in
/// single-dataset mode should `process::exit(1)` on error; callers in
/// recursive mode should collect the error and continue.
pub fn run_pipeline(args: RunArgs) -> Result<(), String> {
    let dataset_path = args.dataset.unwrap_or_else(|| {
        let default = PathBuf::from("dataset.yaml");
        if default.exists() {
            println!("Using dataset.yaml in current directory");
            default
        } else {
            eprintln!("Error: no dataset.yaml specified and none found in current directory");
            std::process::exit(1);
        }
    });
    let dataset_path = &dataset_path;

    // Load dataset config
    let mut config = DatasetConfig::load(dataset_path).unwrap_or_else(|e| {
        println!("{}", e);
        std::process::exit(1);
    });

    let workspace = dataset_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    // Handle --clean / --reset: full reset (remove all generated artifacts),
    // then continue to run the pipeline from scratch.
    if args.clean || args.reset {
        reset_pipeline(&workspace, dataset_path, &config);
    }

    // Handle --clean-last: rewind one step and exit. Doesn't continue
    // into the run — the user can re-invoke `veks run` (without
    // --clean-last) to actually re-execute the rewound step.
    if args.clean_last {
        clean_last_step(&workspace, dataset_path);
        return Ok(());
    }

    // Create managed cache directory.
    let cache_dir = workspace.join(".cache");
    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        println!("Failed to create cache directory: {}", e);
        std::process::exit(1);
    }

    // Ensure .gitignore covers managed directories
    ensure_gitignore(&workspace);

    // Recover identity symlinks if they were removed (e.g., by a
    // previous --clean with different logic, or manual deletion).
    // Profile views that point to non-existent files are checked against
    // their original source targets.
    recover_identity_symlinks(&workspace, &config);

    // Resolve all steps including deferred profile expansion
    let expanded_steps = resolve_all_steps(&mut config, &workspace);

    // Extract oracle sub-facet scope for partition profile filtering.
    // Used by all expansion phases to restrict per_profile templates.
    let oracle_scope: Option<String> = {
        let raw = vectordata::dataset::collect_all_steps(&config);
        let has_partition = raw.iter().any(|s| s.effective_id() == "partition-profiles");
        if has_partition {
            Some(config.attributes.as_ref()
                .and_then(|a| a.tags.get("oracle_scope"))
                .map(|s| s.to_string())
                .unwrap_or_else(|| "BQG".to_string()))
        } else {
            None
        }
    };

    let query_count: u64 = config.upstream.as_ref()
        .and_then(|p| p.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    // Optionally filter to a single profile
    let profile_name = &args.profile;
    let steps = if profile_name == "all" {
        expanded_steps
    } else {
        // Validate profile exists
        if !config.profiles.is_empty() && config.profiles.profile(profile_name).is_none() {
            println!(
                "Profile '{}' not found. Available profiles: {}",
                profile_name,
                config.profile_names().join(", ")
            );
            std::process::exit(1);
        }
        vectordata::dataset::filter_steps_for_profile(expanded_steps, profile_name)
    };

    if steps.is_empty() {
        println!(
            "No pipeline steps found for profile '{}' in {}",
            profile_name,
            veks_core::paths::rel_display(&dataset_path)
        );
        println!("Add an 'upstream' section with 'steps' to define a pipeline.");
        std::process::exit(1);
    }

    // Partition steps: compute steps run in Phases 1/2/3, finalize steps
    // run once at the end after all expansion phases complete. This ensures
    // finalization (merkle, catalog, dataset-json) sees the full set of
    // profiles and artifacts without needing progress invalidation hacks.
    let compute_steps: Vec<_> = steps.iter().filter(|s| !s.finalize).cloned().collect();
    let has_finalize_steps = steps.iter().any(|s| s.finalize);

    // Build DAG from compute steps only — finalize steps run in a
    // separate pass after all expansion phases.
    let pipeline_dag = dag::build_dag(&compute_steps).unwrap_or_else(|e| {
        println!("Pipeline DAG error: {}", e);
        std::process::exit(1);
    });

    // Build defaults from pipeline config + dataset variables + variables.yaml + CLI overrides
    let mut defaults = IndexMap::new();
    if let Some(ref pipe) = config.upstream {
        if let Some(ref defs) = pipe.defaults {
            defaults.extend(defs.clone());
        }
    }
    // Layer in dataset.yaml variables: section
    defaults.extend(config.variables.clone());
    // Layer in variables.yaml (overrides upstream.defaults and dataset variables, but CLI wins)
    match variables::load(&workspace) {
        Ok(vars) => {
            if !vars.is_empty() {
                println!("Loaded {} variable(s) from variables.yaml", vars.len());
                defaults.extend(vars);
            }
        }
        Err(e) => {
            println!("Warning: failed to load variables.yaml: {}", e);
        }
    }

    for ov in &args.overrides {
        if let Some((key, value)) = ov.split_once('=') {
            defaults.insert(key.to_string(), value.to_string());
        } else {
            println!("Invalid --set format: '{}' (expected key=value)", ov);
            std::process::exit(1);
        }
    }

    // Handle --emit-yaml: resolve and print the full pipeline (compute + finalize)
    if args.emit_yaml {
        let full_dag = dag::build_dag(&steps).unwrap_or_else(|e| {
            println!("Pipeline DAG error: {}", e);
            std::process::exit(1);
        });
        emit_resolved_yaml(&full_dag, &defaults, &workspace);
        return Ok(());
    }

    let finalize_step_count = steps.len() - compute_steps.len();
    println!(
        "Pipeline: {} compute + {} finalize steps in topological order (profile: {})",
        pipeline_dag.steps.len(),
        finalize_step_count,
        profile_name,
    );
    for (i, step) in pipeline_dag.steps.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, step.id, step.def.run);
    }
    // Show partition expansion note if partition-profiles step exists
    let has_partition_step = compute_steps.iter()
        .any(|s| s.effective_id() == "partition-profiles");
    if has_partition_step {
        let scope = oracle_scope.as_deref().unwrap_or("BQG");
        println!("  --- after partition-profiles: Phase 3 re-expansion ---");
        println!("  · per-partition steps for sub-facets: {}", scope);
    }
    if finalize_step_count > 0 {
        println!("  --- finalization (runs after all phases) ---");
        for step in &steps {
            if step.finalize {
                let id = step.effective_id();
                println!("  · {} ({})", id, step.run);
            }
        }
    }
    println!();

    // Load or create progress log
    let progress_path = ProgressLog::path_for_dataset(dataset_path);
    let (progress, schema_msg) = ProgressLog::load(&progress_path).unwrap_or_else(|e| {
        println!("Warning: failed to load progress log: {}", e);
        (ProgressLog::new(), None)
    });
    if let Some(msg) = schema_msg {
        println!("{}", msg);
    }

    // Invalidation is now per-step via fingerprint chains (computed in
    // the runner). No whole-log invalidation needed — changing a step's
    // options only invalidates that step and its dependents.

    let threads = if args.threads == 0 {
        physical_core_count()
    } else {
        args.threads
    };

    // Parse --resources if provided
    let resource_budget = if let Some(ref res_str) = args.resources {
        resource::ResourceBudget::parse(res_str).unwrap_or_else(|e| {
            println!("Invalid --resources: {}", e);
            std::process::exit(1);
        })
    } else {
        resource::ResourceBudget::new()
    };

    let governor = resource::ResourceGovernor::new(resource_budget, Some(&workspace));

    // Apply governor strategy from --governor flag
    match args.governor.as_str() {
        "maximize" => {} // default, already set
        "conservative" => {
            governor.set_strategy(Box::new(resource::ConservativeStrategy::default()));
        }
        "fixed" => {
            governor.set_strategy(Box::new(resource::FixedStrategy));
        }
        other => {
            println!("Unknown governor strategy: '{}'. Use maximize, conservative, or fixed.", other);
            std::process::exit(1);
        }
    }

    governor.print_summary();

    // Persistent run log: plain text in .cache/run.log, JSONL in workspace/runlog.jsonl
    let run_log_path = cache_dir.join("run.log");
    let run_jsonl_path = workspace.join("runlog.jsonl");

    // Build execution context
    let dataset_name = config.name.clone();
    // Profile names ordered by size: sized ascending, then default last
    let all_profile_names: Vec<String> = {
        let profiles = &config.profiles;
        let mut sized: Vec<(&str, u64)> = profiles.profiles.iter()
            .filter(|(name, p)| name.as_str() != "default" && p.base_count.is_some())
            .map(|(name, p)| (name.as_str(), p.base_count.unwrap()))
            .collect();
        sized.sort_by_key(|(_, bc)| *bc);
        let mut names: Vec<String> = sized.into_iter().map(|(n, _)| n.to_string()).collect();
        if profiles.profiles.contains_key("default") {
            names.push("default".to_string());
        }
        names
    };
    // Estimate total steps including deferred per-profile expansions.
    // Deferred profiles will each get one copy of every per_profile template.
    let per_profile_template_count = vectordata::dataset::collect_all_steps(&config)
        .iter()
        .filter(|s| s.per_profile)
        .count();
    let deferred_profile_estimate = config.profiles.deferred_sized.len() * 3; // rough: each spec generates ~3 profiles
    let estimated_total = pipeline_dag.steps.len()
        + deferred_profile_estimate * per_profile_template_count
        + finalize_step_count;

    let cache_dir_for_guidance = cache_dir.clone();
    let mut ctx = StreamContext {
        dataset_name,
        profile: profile_name.clone(),
        profile_names: all_profile_names,
        workspace,
        cache: cache_dir,
        defaults,
        dry_run: args.dry_run,
        progress,
        threads,
        step_id: String::new(),
        governor,
        ui: {
            let mut handle = match args.output.as_str() {
                "batch" => veks_core::ui::UiHandle::new(
                    std::sync::Arc::new(veks_core::ui::HeadlessSink::new())),
                "basic" => veks_core::ui::UiHandle::new(
                    std::sync::Arc::new(veks_core::ui::PlainSink::new())),
                "tui" => veks_core::ui::auto_ui_handle_with_interval(
                    std::time::Duration::from_millis(args.status_interval)),
                _ /* auto */ => veks_core::ui::auto_ui_handle_with_interval(
                    std::time::Duration::from_millis(args.status_interval)),
            };
            // Install global logger and attach log file to UiHandle for
            // lock-step file writes from ctx.ui.log().
            let log_writers = veks_core::ui::install_logger(handle.sink_arc(), Some(&run_log_path), Some(&run_jsonl_path));
            handle.set_log_writers(log_writers);
            handle
        },
        status_interval: std::time::Duration::from_millis(args.status_interval),
        estimated_total_steps: estimated_total,
    };

    ctx.ui.log(&format!("pipeline initialized: {} steps, profile={}, build={}",
        pipeline_dag.steps.len(), profile_name,
        concat!(env!("CARGO_PKG_VERSION"), "+", env!("VEKS_BUILD_HASH"), ".", env!("VEKS_BUILD_NUMBER"))));

    // Persist dataset-level metadata as variables so stats.csv can use them
    let _ = variables::set_and_save(&ctx.workspace, "dataset_name", &ctx.dataset_name);
    ctx.defaults.insert("dataset_name".into(), ctx.dataset_name.clone());
    if let Some(df) = config.distance_function() {
        let _ = variables::set_and_save(&ctx.workspace, "distance_function", df);
        ctx.defaults.insert("distance_function".into(), df.to_string());
    }

    // Build command registry
    let registry = CommandRegistry::with_builtins();

    // Install a panic hook that restores the terminal before printing
    // the panic message. Without this, a panic during TUI mode leaves
    // the terminal in raw/alternate-screen mode with no visible error.
    {
        let sink = ctx.ui.sink_arc();
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // Shut down the TUI to restore the terminal
            sink.shutdown();
            // Then show the panic
            prev_hook(info);
        }));
    }

    // Phase 1: Run steps (core + any already-resolved per-profile steps)
    let result = runner::run_steps(&pipeline_dag.steps, &registry, &mut ctx);

    // Phase 2: If there are deferred sized entries (containing ${variable}
    // references), expand them now that core steps have produced the actual
    // counts in variables.yaml, then run the newly generated per-profile steps.
    let result = if result.is_ok() && config.profiles.has_deferred() {
        // Reload variables — core steps may have produced base_count etc.
        match variables::load(&ctx.workspace) {
            Ok(vars) => {
                for (k, v) in &vars {
                    ctx.defaults.insert(k.clone(), v.clone());
                }
            }
            Err(e) => {
                log::warn!("failed to reload variables for deferred profiles: {}", e);
            }
        }

        let base_count: u64 = ctx.defaults.get("base_count")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let added = if base_count > 0 {
            config.profiles.expand_deferred_sized(&ctx.defaults, base_count)
        } else {
            ctx.ui.log("WARNING: base_count not available — cannot expand sized profiles");
            0
        };
        if added > 0 {
            ctx.ui.log(&format!(
                "Expanded {} deferred sized profiles (base_count={})", added, base_count
            ));

            // Re-derive views for the new profiles
            let template_steps: Vec<_> = vectordata::dataset::collect_all_steps(&config)
                .into_iter()
                .filter(|s| s.per_profile)
                .collect();
            config.profiles.derive_views_from_templates(&template_steps);

            // Re-expand per-profile steps with the new profiles
            let raw_steps = vectordata::dataset::collect_all_steps(&config);
            let new_expanded = vectordata::dataset::expand_per_profile_steps_scoped(
                raw_steps, &config.profiles, query_count, oracle_scope.as_deref(),
            );

            // Filter to only compute steps (finalize runs in a separate final pass)
            let new_steps: Vec<_> = if profile_name == "all" {
                new_expanded
            } else {
                vectordata::dataset::filter_steps_for_profile(new_expanded, profile_name)
            }.into_iter().filter(|s| !s.finalize).collect();

            match dag::build_dag(&new_steps) {
                Ok(new_dag) => {
                    // Run phase 2 steps — already-completed steps will be
                    // skipped via freshness checks
                    runner::run_steps(&new_dag.steps, &registry, &mut ctx)
                }
                Err(e) => {
                    Err(format!("DAG error for deferred profiles: {}", e))
                }
            }
        } else {
            result
        }
    } else {
        result
    };

    // Phase 3: If partition profiles were created by prepare-partitions,
    // reload dataset.yaml, pick up the new profiles, and re-expand
    // per_profile templates (compute-knn, verify-knn) for each partition.
    let result = if result.is_ok() {
        // Check if partition profiles were generated
        let partition_count: usize = ctx.defaults.get("partition_count")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        if partition_count > 0 {
            ctx.ui.log(&format!(
                "Re-expanding pipeline for {} partition profiles...", partition_count));

            // Reload dataset.yaml to pick up the new profile entries
            match vectordata::dataset::DatasetConfig::load(&dataset_path) {
                Ok(mut updated_config) => {
                    // Derive views for partition profiles so compute-knn
                    // outputs are registered as profile views.
                    let template_steps: Vec<_> = vectordata::dataset::collect_all_steps(&updated_config)
                        .into_iter()
                        .filter(|s| s.per_profile)
                        .collect();
                    updated_config.profiles.derive_views_from_templates(&template_steps);

                    let raw_steps = vectordata::dataset::collect_all_steps(&updated_config);
                    let new_expanded = vectordata::dataset::expand_per_profile_steps_scoped(
                        raw_steps, &updated_config.profiles, query_count, oracle_scope.as_deref(),
                    );

                    // Compute steps only — finalize runs in the final pass
                    let new_steps: Vec<_> = if profile_name == "all" {
                        new_expanded
                    } else {
                        vectordata::dataset::filter_steps_for_profile(new_expanded, profile_name)
                    }.into_iter().filter(|s| !s.finalize).collect();

                    match dag::build_dag(&new_steps) {
                        Ok(new_dag) => {
                            let result = runner::run_steps(&new_dag.steps, &registry, &mut ctx);
                            // After partition compute steps complete, save the
                            // updated config with derived views (neighbor_indices,
                            // neighbor_distances) so finalization and consumers see them.
                            if result.is_ok() {
                                if let Err(e) = updated_config.save(&dataset_path) {
                                    ctx.ui.log(&format!("  warning: failed to save updated views: {}", e));
                                }
                            }
                            result
                        }
                        Err(e) => Err(format!("DAG error for partition profiles: {}", e)),
                    }
                }
                Err(e) => {
                    ctx.ui.log(&format!("WARNING: could not reload dataset.yaml for partition expansion: {}", e));
                    result
                }
            }
        } else {
            result
        }
    } else {
        result
    };

    // Finalization pass: run finalize steps once, after all compute phases
    // (core, deferred sized, partition expansion) have completed. This
    // ensures merkle, catalog, dataset-json, and variables-json see the
    // full set of profiles and artifacts.
    let result = if result.is_ok() && has_finalize_steps {
        // Reload dataset.yaml to get the final state (may include
        // Sync variables and derived attributes to dataset.yaml BEFORE
        // finalization, so generate-dataset-json and generate-catalog see
        // the updated is_normalized, is_zero_vector_free, etc.
        update_dataset_attributes(dataset_path, dataset_path.parent().unwrap_or(Path::new(".")));

        // partition profiles added during Phase 3).
        let finalize_config = vectordata::dataset::DatasetConfig::load(&dataset_path)
            .unwrap_or_else(|_| config.clone());
        let raw_steps = vectordata::dataset::collect_all_steps(&finalize_config);
        let all_expanded = vectordata::dataset::expand_per_profile_steps_scoped(
            raw_steps, &finalize_config.profiles, query_count, oracle_scope.as_deref(),
        );

        let finalize_steps: Vec<_> = all_expanded.into_iter()
            .filter(|s| s.finalize)
            .collect();

        if finalize_steps.is_empty() {
            result
        } else {
            ctx.ui.log(&format!(
                "Finalization: {} step(s) to run", finalize_steps.len()));

            match dag::build_dag_partial(&finalize_steps) {
                Ok(finalize_dag) => {
                    runner::run_steps(&finalize_dag.steps, &registry, &mut ctx)
                }
                Err(e) => Err(format!("DAG error for finalization steps: {}", e)),
            }
        }
    } else {
        result
    };

    // Retrieve buffered log messages before shutting down the TUI.
    let console_log = ctx.ui.take_console_log();

    // Explicitly shut down the TUI sink to restore the terminal.
    // The global logger holds a leaked Arc<dyn UiSink>, so Drop alone
    // never fires — shutdown() sends the Shutdown signal and joins the
    // render thread, restoring raw mode and the alternate screen.
    ctx.ui.shutdown();
    drop(ctx);

    // Replay buffered log messages to stdout so they persist in scrollback.
    // Each message gets its own line — the TUI log pane stores them without newlines.
    if !console_log.is_empty() {
        // Ensure terminal is in a clean state before replaying log
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let _ = std::io::Write::flush(&mut std::io::stderr());
        eprintln!(); // start on a fresh line

        for msg in &console_log {
            eprintln!("{}", msg);
        }
        eprintln!();
    }

    match result {
        Err(e) => {
            // Sync variables even on failure — earlier steps (dedup, zeros)
            // may have completed and their variables should be persisted.
            update_dataset_attributes(dataset_path, dataset_path.parent().unwrap_or(Path::new(".")));
            // Ensure we're on a clean line before printing the error.
            eprintln!();
            eprintln!("Pipeline failed: {}", e);
            eprintln!("Run with --clean to reset.");
            return Err(e);
        }
        Ok(summary) => {
            // Sync variables from variables.yaml into dataset.yaml.
            // Always attempt — even if all steps were fresh, the variables
            // may not have been synced yet (e.g., first run after upgrade).
            update_dataset_attributes(dataset_path, dataset_path.parent().unwrap_or(Path::new(".")));
            print_run_summary(&summary);
            // Post-completion guidance about the cache directory
            if summary.executed > 0 {
                print_cache_guidance(&cache_dir_for_guidance, &pipeline_dag.steps);
            }
            // Suggest stratification if no sized profiles exist yet
            let has_sized = config.profiles.profiles.iter()
                .any(|(name, _)| name != "default")
                || config.profiles.has_deferred();
            if !has_sized && summary.executed > 0 {
                println!();
                println!("{}",
                    veks_core::term::info("Tip: add sized profiles for multi-scale benchmarking:"));
                println!("  veks prepare stratify");
            }
        }
    }

    Ok(())
}

/// Sync variables from `variables.yaml` into `dataset.yaml` so consumers
/// can see pipeline-produced properties (counts, flags) without a separate
/// file. Also updates derived attributes (is_normalized, is_zero_vector_free,
/// is_duplicate_vector_free).
///
/// Uses the `DatasetConfig::save()` API for canonical serialization.
fn update_dataset_attributes(dataset_path: &Path, workspace: &Path) {
    let vars = match variables::load(workspace) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  note: could not load variables.yaml: {}", e);
            return;
        }
    };
    if vars.is_empty() {
        return;
    }

    let mut config = match DatasetConfig::load(dataset_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  note: could not load dataset.yaml: {}", e);
            return;
        }
    };

    // Sync all variables
    for (k, v) in &vars {
        config.set_variable(k, v);
    }

    // Derive attributes from pipeline results.
    // The attribute reflects the actual output state:
    // - If the pipeline removed zeros/duplicates, the output is free of them.
    // - If the pipeline only scanned (--no-dedup), the attribute reflects
    //   whether any were found (count == 0 means free).
    if let Some(zc) = vars.get("zero_count") {
        let is_free = zc == "0";
        config.set_attribute("is_zero_vector_free", if is_free { "true" } else { "false" });
    }
    if let Some(dc) = vars.get("duplicate_count") {
        let is_free = dc == "0";
        config.set_attribute("is_duplicate_vector_free", if is_free { "true" } else { "false" });
    }
    // Normalization: check is_normalized first, fall back to source_was_normalized
    if let Some(v) = vars.get("is_normalized") {
        config.set_attribute("is_normalized", v);
    } else if let Some(v) = vars.get("source_was_normalized") {
        config.set_attribute("is_normalized", v);
    }

    if let Err(e) = config.save(dataset_path) {
        eprintln!("  warning: failed to save dataset.yaml: {}", e);
    } else {
        eprintln!("  synced {} variables to dataset.yaml", vars.len());
    }
}

/// Print a post-TUI run summary to stdout.
///
/// This runs after the ratatui alternate screen is gone, so the output
/// persists in the user's terminal scrollback.
fn print_run_summary(summary: &runner::RunSummary) {
    use veks_core::term;
    println!();

    if summary.executed == 0 && summary.skipped == summary.total {
        println!("{} all {} steps verified fresh — nothing to do.",
            term::ok("Pipeline:"),
            summary.total,
        );
        let show = summary.step_outcomes.len().min(6);
        for (id, _, reason) in &summary.step_outcomes[..show] {
            println!("  {} — {}", term::dim(id), term::dim(reason));
        }
        if summary.step_outcomes.len() > show {
            println!("  {}", term::dim(&format!("... and {} more", summary.step_outcomes.len() - show)));
        }
    } else if summary.skipped > 0 {
        println!(
            "{} {} executed ({:.1}s), {} skipped (fresh), {} total.",
            term::ok("Pipeline complete:"),
            summary.executed,
            summary.total_elapsed.as_secs_f64(),
            summary.skipped,
            summary.total,
        );
        for (id, executed, msg) in &summary.step_outcomes {
            if *executed {
                println!("  {} {} — {}", term::ok("\u{2713}"), id, msg);
            }
        }
    } else {
        println!(
            "{} {} steps executed in {:.1}s.",
            term::ok("Pipeline complete:"),
            summary.executed,
            summary.total_elapsed.as_secs_f64(),
        );
    }
}

/// Print post-completion guidance about the cache directory.
///
/// Informs the user that `.cache/` contains intermediate artifacts that are
/// expensive to recompute, and suggests `veks prepare cache-compress` if
/// the pipeline was not already configured with `compress_cache: true`.
fn print_cache_guidance(cache_dir: &std::path::Path, steps: &[dag::ResolvedStep]) {
    use veks_core::term;

    // Check if any step has compress_cache enabled
    let has_compress = steps.iter().any(|s| {
        s.def.options.get("compress_cache")
            .and_then(|v| v.as_str())
            .map(|v| v == "true")
            .unwrap_or(false)
    });

    // Calculate cache size
    let cache_size = dir_size_recursive(cache_dir);
    let size_str = if cache_size > 0 {
        format!(" ({})", format_bytes(cache_size))
    } else {
        String::new()
    };

    println!();
    println!("{}", term::bold("Next steps:"));
    println!("  The {} directory{} contains intermediate artifacts.",
        cache_dir.file_name().unwrap_or_default().to_string_lossy(),
        size_str,
    );
    println!("  Removing it is safe but may require expensive recomputation,");
    println!("  especially for large datasets (KNN ground truth, sorted runs).");
    if !has_compress {
        println!();
        println!("  To reduce cache size, run:");
        println!("    {}", term::ok("veks prepare cache-compress"));
    }
}

/// Recursively calculate the total size of files in a directory.
fn dir_size_recursive(dir: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += dir_size_recursive(&path);
            } else if let Ok(meta) = std::fs::metadata(&path) {
                total += meta.len();
            }
        }
    }
    total
}

/// Format a byte count as human-readable.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

// collect_all_steps, filter_steps_for_profile, and expand_per_profile_steps
// are now in vectordata::dataset::expansion (shared code used by all modules
// that read dataset.yaml).

// The following local function was part of the old expand_per_profile_steps.
// It has been moved to vectordata::dataset::expansion.
/// Auto-derive profile views from per_profile step outputs.
///
/// For each profile that has per_profile template expansions (including default
/// when no explicit default-gated steps exist), scans templates for `output`
/// options and creates profile view entries using the filename stem as the view
/// key and `profiles/{name}/{filename}` as the path. Existing explicit views
/// are not overridden.
// `derive_sized_profile_views` logic is now in
// `DSProfileGroup::derive_views_from_templates` in the dataset crate.

/// Emit the pipeline as a sequence of `veks pipeline` CLI commands to stdout.
///
/// Each step is rendered as a standalone invocation with `--workspace` and
/// all resolved options as `--key=value` flags. Variable references that
/// contain `${...}` patterns are emitted in the shell-safe qualified form
/// `${variables:name}` and single-quoted.
///
/// The output is a valid shell script that reproduces the pipeline when
/// executed top-to-bottom.
fn emit_cli_commands(
    pipeline_dag: &dag::PipelineDag,
    _defaults: &IndexMap<String, String>,
    workspace: &Path,
    progress: Option<&ProgressLog>,
    workspace_is_cwd: bool,
    absolute: bool,
) {
    let workspace_str = workspace.to_string_lossy();

    // Load variable names for qualified-form rewriting
    let var_names: std::collections::HashSet<String> = variables::load(workspace)
        .unwrap_or_default()
        .keys()
        .cloned()
        .collect();

    println!("#!/usr/bin/env bash");
    println!("# Generated by: veks script");
    println!("# Workspace: {}", workspace_str);
    if !workspace_is_cwd {
        println!("#");
        println!("# WARNING: This script references a workspace outside the current directory.");
        println!("# Each command includes --workspace to resolve relative paths correctly.");
        println!("# The script is valid when run from: .");
    }
    println!("set -euo pipefail");
    println!();

    for step in &pipeline_dag.steps {
        let parts: Vec<&str> = step.def.run.splitn(2, ' ').collect();
        let (group, subcommand) = if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            (parts[0], "")
        };

        // Check if step is completed in progress log
        let is_done = progress
            .map(|p| p.is_step_fresh(&step.id, None))
            .unwrap_or(false);

        let prefix = if is_done { "# DONE: " } else { "" };

        println!("{}# step: {}", prefix, step.id);

        let mut cmd = format!("{}veks pipeline {} {}", prefix, group, subcommand);

        // Only include --workspace when the workspace is not the CWD
        if !workspace_is_cwd {
            cmd.push_str(&format!(" \\\n{}  --workspace='{}'", prefix, workspace_str));
        }

        for (key, value) in &step.def.options {
            let val_str = match value {
                serde_yaml::Value::String(s) => s.clone(),
                serde_yaml::Value::Number(n) => n.to_string(),
                serde_yaml::Value::Bool(b) => b.to_string(),
                serde_yaml::Value::Null => continue,
                other => format!("{:?}", other),
            };

            // Relativize absolute paths when --absolute is not set.
            let val_str = if !absolute && val_str.starts_with('/') {
                relativize_path(&val_str, workspace)
            } else {
                val_str
            };

            // Rewrite unqualified ${name} references to ${variables:name}
            // when the name is a known variables.yaml key.
            let cli_val = rewrite_var_refs(&val_str, &var_names);

            // Single-quote values that contain variable references or shell metacharacters
            if cli_val.contains("${") || cli_val.contains(' ') || cli_val.contains('\'') {
                let escaped = cli_val.replace('\'', "'\\''");
                cmd.push_str(&format!(" \\\n{}  --{}='{}'", prefix, key, escaped));
            } else {
                cmd.push_str(&format!(" \\\n{}  --{}={}", prefix, key, cli_val));
            }
        }

        println!("{}", cmd);
        println!();
    }
}

/// Make a path relative to the workspace directory.
///
/// If the path starts with the workspace prefix, strips it.
/// Handles window suffixes like `file.mvec[0..100)`.
fn relativize_path(path_str: &str, workspace: &Path) -> String {
    // Handle paths with window suffixes
    let (path_part, suffix) = if let Some(bracket) = path_str.find('[') {
        (&path_str[..bracket], &path_str[bracket..])
    } else if let Some(paren) = path_str.find('(') {
        (&path_str[..paren], &path_str[paren..])
    } else {
        (path_str, "")
    };

    let path = Path::new(path_part);
    if let Ok(rel) = path.strip_prefix(workspace) {
        format!("{}{}", rel.display(), suffix)
    } else {
        // Already relative or external — return as-is
        path_str.to_string()
    }
}

/// Rewrite unqualified `${name}` variable references to the qualified
/// `${variables:name}` form when `name` is a known variables.yaml key.
///
/// This makes the output shell-safe: `${variables:name}` is not a valid
/// shell variable name (the colon prevents expansion), so it passes
/// through bash verbatim even without quoting.
fn rewrite_var_refs(input: &str, var_names: &std::collections::HashSet<String>) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_expr = String::new();
            let mut depth = 1;
            while let Some(c) = chars.next() {
                if c == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                } else if c == '{' {
                    depth += 1;
                }
                var_expr.push(c);
            }

            // Extract the variable name (before :- fallback)
            let name = if let Some(idx) = var_expr.find(":-") {
                &var_expr[..idx]
            } else {
                &var_expr
            };

            // Qualify if it's a known variables.yaml key and not already qualified
            if var_names.contains(name)
                && !name.starts_with("env:")
                && !name.starts_with("variables:")
                && !name.starts_with("variables.yaml:")
            {
                result.push_str("${variables:");
                result.push_str(&var_expr);
                result.push('}');
            } else {
                result.push_str("${");
                result.push_str(&var_expr);
                result.push('}');
            }
        } else {
            result.push(ch);
        }
    }

    result
}

/// Emit the fully resolved pipeline as YAML to stdout.
///
/// Delegates to [`resolve_pipeline_yaml`] and prints the result.
fn emit_resolved_yaml(
    pipeline_dag: &dag::PipelineDag,
    defaults: &IndexMap<String, String>,
    workspace: &Path,
) {
    let yaml = resolve_pipeline_yaml(pipeline_dag, defaults, workspace).unwrap_or_else(|e| {
        println!("{}", e);
        std::process::exit(1);
    });
    print!("{}", yaml);
}

/// Resolve all pipeline steps and produce the complete `PipelineConfig` YAML.
///
/// Walks the DAG in topological order, interpolates every step's options
/// against the merged defaults, and serializes a complete `PipelineConfig`.
/// Returns the YAML string or an error message.
fn resolve_pipeline_yaml(
    pipeline_dag: &dag::PipelineDag,
    defaults: &IndexMap<String, String>,
    workspace: &Path,
) -> Result<String, String> {
    let mut resolved_steps: Vec<schema::StepDef> = Vec::new();

    for step in &pipeline_dag.steps {
        let resolved_opts = interpolate::interpolate_options(
            &step.def.options,
            defaults,
            workspace,
        )
        .map_err(|e| format!("step '{}': {}", step.id, e))?;

        let mut options = IndexMap::new();
        for (k, v) in &resolved_opts {
            options.insert(k.clone(), serde_yaml::Value::String(v.clone()));
        }

        resolved_steps.push(schema::StepDef {
            id: step.def.id.clone(),
            run: step.def.run.clone(),
            description: step.def.description.clone(),
            after: step.def.after.clone(),
            profiles: step.def.profiles.clone(),
            per_profile: step.def.per_profile,
            phase: step.def.phase,
            finalize: step.def.finalize,
            on_partial: step.def.on_partial.clone(),
            options,
        });
    }

    let resolved_defaults = if defaults.is_empty() {
        None
    } else {
        Some(defaults.clone())
    };

    let config = schema::PipelineConfig {
        defaults: resolved_defaults,
        steps: Some(resolved_steps),
    };

    serde_yaml::to_string(&config).map_err(|e| format!("Failed to serialize pipeline: {}", e))
}

/// Clean up pipeline artifacts: remove the progress log.
///
/// The `.cache/` directory is intentionally preserved — users can delete it
/// manually if they want to discard cached intermediates.
/// Rewind progress by one step: identify the most recently
/// completed (or partially completed) step from the progress log,
/// delete its recorded output artifacts, and remove its progress
/// entry. Subsequent `veks run` invocations re-execute that step
/// (and only that step — earlier steps stay marked fresh).
///
/// "Most recent" is determined by `StepRecord::completed_at`; both
/// `Status::Ok` and `Status::Error` records count, so this also
/// rewinds a step that aborted partway through. Steps that were
/// killed before any record was written naturally don't appear in
/// the log; in that case `--clean-last` rewinds the previous
/// successfully-finished step instead, which is the right behavior
/// (the killed step has no recorded artifacts to clean up).
fn clean_last_step(_workspace: &Path, dataset_path: &Path) {
    let progress_path = ProgressLog::path_for_dataset(dataset_path);
    if !progress_path.exists() {
        println!("No progress log at {} — nothing to rewind.",
            veks_core::paths::rel_display(&progress_path));
        return;
    }

    let (mut log, version_msg) = match ProgressLog::load(&progress_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Failed to load progress log: {}", e);
            return;
        }
    };
    if let Some(msg) = version_msg {
        println!("{}", msg);
    }

    // Find the step with the latest completed_at timestamp.
    let last = log.steps.iter()
        .max_by(|(_, a), (_, b)| a.completed_at.cmp(&b.completed_at))
        .map(|(id, rec)| (id.clone(), rec.clone()));

    let (step_id, record) = match last {
        Some(t) => t,
        None => {
            println!("Progress log has no recorded steps — nothing to rewind.");
            return;
        }
    };

    println!(
        "Rewinding last step: {} (status={}, completed_at={})",
        step_id, record.status, record.completed_at,
    );

    // Delete each recorded output file. A step's recorded outputs
    // are the artifacts it produced; downstream steps' inputs may
    // reference these by path. Removing them forces this step to
    // re-execute on the next run, and the freshness check on every
    // downstream step will see its input is gone (or its
    // fingerprint changed once this step re-runs) and cascade
    // re-execution.
    let mut removed = 0usize;
    let mut missing = 0usize;
    let mut failed = 0usize;
    for out in &record.outputs {
        let p = std::path::Path::new(&out.path);
        match std::fs::metadata(p) {
            Ok(_) => match std::fs::remove_file(p) {
                Ok(()) => {
                    println!("  removed {}", veks_core::paths::rel_display(&p.to_path_buf()));
                    removed += 1;
                }
                Err(e) => {
                    println!("  failed to remove {}: {}", veks_core::paths::rel_display(&p.to_path_buf()), e);
                    failed += 1;
                }
            },
            Err(_) => {
                println!("  skipped {} (already gone)", veks_core::paths::rel_display(&p.to_path_buf()));
                missing += 1;
            }
        }
    }
    if record.outputs.is_empty() {
        println!("  (no recorded output artifacts for this step)");
    }

    // Drop the step's progress entry so the freshness check on the
    // next run treats it as never-executed.
    log.steps.remove(&step_id);
    if let Err(e) = log.save() {
        println!("Warning: failed to save progress log after rewind: {}", e);
    } else {
        println!("Removed progress entry for step '{}'.", step_id);
    }

    println!(
        "Rewound 1 step: {} artifact(s) removed, {} already missing, {} failed.",
        removed, missing, failed,
    );
    println!("Re-run `veks run` to re-execute this step (and any downstream steps).");
}

fn clean_pipeline(workspace: &Path, dataset_path: &Path) {
    let progress_path = ProgressLog::path_for_dataset(dataset_path);
    if progress_path.exists() {
        match std::fs::remove_file(&progress_path) {
            Ok(()) => println!("Removed {}", veks_core::paths::rel_display(&progress_path)),
            Err(e) => println!("Failed to remove {}: {}", veks_core::paths::rel_display(&progress_path), e),
        }
    } else {
        println!("No progress log found at {}", veks_core::paths::rel_display(&progress_path));
    }

    println!("Clean complete. Cache and output files are preserved.");
    println!(
        "To remove outputs, delete them manually from {}",
        veks_core::paths::rel_display(&workspace.to_path_buf())
    );
}

/// Full pipeline reset: remove progress, cache, and all generated artifacts.
///
/// Preserves: dataset.yaml, variables.yaml, identity symlinks
/// (those pointing outside the workspace), and source data files.
/// Preserves source files and configuration for re-bootstrap.
pub fn reset_pipeline(workspace: &Path, dataset_path: &Path, config: &DatasetConfig) {
    println!("Resetting pipeline — removing all generated artifacts...");

    // 1. Remove progress log
    let progress_path = ProgressLog::path_for_dataset(dataset_path);
    if progress_path.exists() {
        let _ = std::fs::remove_file(&progress_path);
        println!("  removed {}", veks_core::paths::rel_display(&progress_path));
    }

    // 2. Remove .cache/ directory entirely
    let cache_dir = workspace.join(".cache");
    if cache_dir.exists() {
        match std::fs::remove_dir_all(&cache_dir) {
            Ok(()) => println!("  removed {}", veks_core::paths::rel_display(&cache_dir)),
            Err(e) => println!("  failed to remove {}: {}", veks_core::paths::rel_display(&cache_dir), e),
        }
    }

    // 3. Remove generated files in profiles/ but preserve Identity symlinks
    //    (symlinks to source data created during bootstrap). These point to
    //    files outside the workspace and must survive --clean.
    let profiles_dir = workspace.join("profiles");
    if profiles_dir.exists() {
        clean_profiles_preserving_symlinks(&profiles_dir);
    }

    // 5. Remove generated facet files in the workspace root (classic layout).
    //    These are pipeline outputs like neighbors.ivec, distances.fvec, etc.
    //    We preserve: dataset.yaml, dataset.json, dataset.log, dataset.jsonl,
    //    variables.yaml, variables.json, catalog.*, .publish, .publish_url,
    //    .catalog_root, .gitignore, and identity symlinks pointing to source data.
    // Remove generated log/variable/json files
    for name in &["dataset.json", "dataset.jsonl", "dataset.log",
                   "runlog.jsonl", "variables.yaml", "variables.json"] {
        let p = workspace.join(name);
        if p.exists() {
            let _ = std::fs::remove_file(&p);
            println!("  removed {}", veks_core::paths::rel_display(&p));
        }
    }

    let preserve = |name: &str| -> bool {
        name == "dataset.yaml" || name == "dataset.yml"
            || name == ".publish" || name == ".publish_url" || name == ".catalog_root"
            || name == ".do_not_catalog" || name == ".gitignore"
            || name == "catalog.json" || name == "catalog.yaml"
    };

    // Collect pipeline output filenames from profile views (classic layout)
    let mut pipeline_outputs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_name, profile) in &config.profiles.profiles {
        for (key, view) in profile.views() {
            let path = std::path::Path::new(&view.source.path);
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                pipeline_outputs.insert(name.to_string());
            }
            // Also the key-based default names
            pipeline_outputs.insert(format!("{}.ivec", key));
            pipeline_outputs.insert(format!("{}.fvec", key));
            pipeline_outputs.insert(format!("{}.mvec", key));
            pipeline_outputs.insert(format!("{}.slab", key));
        }
    }

    if let Ok(entries) = std::fs::read_dir(workspace) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() { continue; }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            // Preserve infrastructure files
            if preserve(&name) { continue; }
            // Preserve hidden files (other than those we explicitly handle)
            if name.starts_with('.') { continue; }
            // Preserve source data files (underscore-prefixed originals)
            if name.starts_with('_') { continue; }

            // Only remove files that are known pipeline outputs or their
            // artifacts (.mref, .mrkl). Symlinks are only removed if they
            // match a pipeline output name — user-created symlinks for
            // other purposes are preserved.
            let is_pipeline_output = pipeline_outputs.contains(&name)
                || name.ends_with(".mref")
                || name.ends_with(".mrkl");

            if path.is_symlink() {
                if is_pipeline_output {
                    let _ = std::fs::remove_file(&path);
                    println!("  removed symlink {}", name);
                }
                continue;
            }

            // Remove pipeline-generated regular files
            let is_generated = is_pipeline_output
                || name.ends_with(".ivec") || name.ends_with(".fvec")
                || name.ends_with(".mvec") || name.ends_with(".slab");
            if is_generated {
                let _ = std::fs::remove_file(&path);
                println!("  removed {}", name);
            }
        }
    }

    println!("Reset complete. Pipeline will re-run from scratch.");
}

/// Recover identity symlinks that may have been removed by --clean or
/// manual deletion. Scans profile views for files that don't exist on
/// disk but whose original source (underscore-prefixed file in the
/// workspace root) does exist, and recreates the symlink.
fn recover_identity_symlinks(workspace: &Path, config: &DatasetConfig) {
    for (_name, profile) in &config.profiles.profiles {
        // Skip partition profiles — their files are pipeline-generated,
        // not identity symlinks to source data.
        if profile.partition {
            continue;
        }
        for (_facet, view) in profile.views() {
            let view_path = workspace.join(view.path());
            if view_path.exists() || view_path.symlink_metadata().is_ok() {
                continue; // file or symlink exists
            }
            // Check if this looks like a profile path (profiles/<name>/<file>)
            let components: Vec<_> = std::path::Path::new(view.path()).components().collect();
            if components.len() < 3 { continue; }
            let filename = components.last()
                .and_then(|c| c.as_os_str().to_str())
                .unwrap_or("");

            // Look for underscore-prefixed source files in workspace root
            // that match the view's filename pattern
            if let Ok(entries) = std::fs::read_dir(workspace) {
                for entry in entries.flatten() {
                    let source_name = entry.file_name().to_string_lossy().to_string();
                    if !source_name.starts_with('_') { continue; }
                    // Match by extension and approximate name
                    let view_ext = std::path::Path::new(filename).extension()
                        .and_then(|e| e.to_str()).unwrap_or("");
                    let source_ext = entry.path().extension()
                        .and_then(|e| e.to_str().map(|s| s.to_string())).unwrap_or_default();
                    if view_ext != source_ext { continue; }

                    // Check if this source was the original symlink target
                    // by matching the canonical facet name in the filename
                    let facet_in_source = source_name.to_lowercase();
                    let facet_in_view = filename.to_lowercase();
                    let is_match = (facet_in_view.contains("base") && facet_in_source.contains("base"))
                        || (facet_in_view.contains("query") && facet_in_source.contains("query"))
                        || (facet_in_view.contains("neighbor") && facet_in_source.contains("groundtruth"))
                        || (facet_in_view.contains("neighbor") && facet_in_source.contains("gt"))
                        || (facet_in_view.contains("distance") && facet_in_source.contains("dist"));

                    if is_match {
                        // Compute relative symlink target
                        if let Some(parent) = view_path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                            // Compute relative path from the view directory to the source
                            let rel = {
                                let from = std::fs::canonicalize(parent).unwrap_or(parent.to_path_buf());
                                let to = std::fs::canonicalize(entry.path()).unwrap_or(entry.path());
                                let from_parts: Vec<_> = from.components().collect();
                                let to_parts: Vec<_> = to.components().collect();
                                let common = from_parts.iter().zip(to_parts.iter())
                                    .take_while(|(a, b)| a == b).count();
                                let mut rel = PathBuf::new();
                                for _ in common..from_parts.len() { rel.push(".."); }
                                for part in &to_parts[common..] { rel.push(part); }
                                rel
                            };
                            match std::os::unix::fs::symlink(&rel, &view_path) {
                                Ok(()) => println!("  recovered symlink: {} → {}",
                                    veks_core::paths::rel_display(&view_path),
                                    rel.display()),
                                Err(e) => println!("  warning: failed to recover symlink {}: {}",
                                    view_path.display(), e),
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
}

/// Remove generated files in profiles/ but preserve symlinks to source data.
///
/// Identity symlinks (created by bootstrap's `create_identity_symlinks`) point
/// to source files outside the pipeline's control. Removing them during --clean
/// would break subsequent runs that depend on the symlinked data.
fn clean_profiles_preserving_symlinks(profiles_dir: &Path) {
    let entries = match std::fs::read_dir(profiles_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dir_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            // Only preserve symlinks in base/ and default/ directories
            // (identity symlinks to source data). Partition profile
            // directories (label-*, or anything else) are entirely
            // pipeline-generated and should be removed completely.
            if dir_name == "base" || dir_name == "default" {
                clean_profile_dir(&path);
                // Remove dir if empty after cleaning
                let _ = std::fs::remove_dir(&path);
            } else {
                // Partition profile — remove entirely
                match std::fs::remove_dir_all(&path) {
                    Ok(()) => println!("  removed partition profile {}",
                        veks_core::paths::rel_display(&path)),
                    Err(e) => println!("  warning: failed to remove {}: {}",
                        veks_core::paths::rel_display(&path), e),
                }
            }
        } else if !path.is_symlink() {
            let _ = std::fs::remove_file(&path);
        }
    }
    // Remove profiles/ itself if empty
    let _ = std::fs::remove_dir(profiles_dir);
}

/// Remove non-symlink files from a profile directory, preserving symlinks.
fn clean_profile_dir(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut removed = 0;
    let mut kept = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_symlink() {
            kept += 1; // preserve Identity symlinks
        } else if path.is_dir() {
            let _ = std::fs::remove_dir_all(&path);
            removed += 1;
        } else {
            let _ = std::fs::remove_file(&path);
            removed += 1;
        }
    }
    if removed > 0 || kept > 0 {
        println!("  cleaned {} ({} files removed, {} symlinks preserved)",
            veks_core::paths::rel_display(dir), removed, kept);
    }
}

/// Ensure the workspace `.gitignore` excludes everything except `dataset.yaml`.
///
/// The dataset workspace contains large generated artifacts (vectors, indices,
/// metadata slabs) that should never be committed. Only the `dataset.yaml`
/// descriptor itself belongs in version control.
///
/// Reads the existing `.gitignore` (if any), appends any missing entries, and
/// writes back. Creates the file if it does not exist.
fn ensure_gitignore(workspace: &Path) {
    let gitignore_path = workspace.join(".gitignore");
    // Exclude everything, then whitelist only dataset.yaml
    let required_entries = ["*", "!dataset.yaml", "!.gitignore"];

    let existing = std::fs::read_to_string(&gitignore_path).unwrap_or_default();
    let existing_lines: Vec<&str> = existing.lines().collect();

    let mut missing: Vec<&str> = Vec::new();
    for entry in &required_entries {
        if !existing_lines.iter().any(|line| line.trim() == *entry) {
            missing.push(entry);
        }
    }

    if missing.is_empty() {
        return;
    }

    let mut content = existing;
    if !content.is_empty() && !content.ends_with('\n') {
        content.push('\n');
    }
    for entry in missing {
        content.push_str(entry);
        content.push('\n');
    }

    if let Err(e) = std::fs::write(&gitignore_path, &content) {
        println!(
            "Warning: failed to update {}: {}",
            gitignore_path.display(),
            e
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schema::OnPartial;

    fn make_step(
        id: &str,
        run: &str,
        after: Vec<&str>,
        opts: Vec<(&str, &str)>,
    ) -> schema::StepDef {
        make_step_with_profiles(id, run, after, opts, vec![])
    }

    fn make_step_with_profiles(
        id: &str,
        run: &str,
        after: Vec<&str>,
        opts: Vec<(&str, &str)>,
        profiles: Vec<&str>,
    ) -> schema::StepDef {
        let mut options = IndexMap::new();
        for (k, v) in opts {
            options.insert(k.to_string(), serde_yaml::Value::String(v.to_string()));
        }
        schema::StepDef {
            id: Some(id.to_string()),
            run: run.to_string(),
            description: None,
            after: after.into_iter().map(String::from).collect(),
            profiles: profiles.into_iter().map(String::from).collect(),
            per_profile: false,
            phase: 0,
            finalize: false,
            on_partial: OnPartial::default(),
            options,
        }
    }

    #[test]
    fn test_resolve_pipeline_yaml_basic() {
        let steps = vec![
            make_step("gen", "generate vectors", vec![], vec![
                ("dim", "128"),
                ("output", "base.fvec"),
            ]),
            make_step("knn", "compute knn", vec!["gen"], vec![
                ("source", "base.fvec"),
                ("k", "100"),
            ]),
        ];
        let pipeline_dag = dag::build_dag(&steps).unwrap();
        let defaults = IndexMap::new();

        let yaml = resolve_pipeline_yaml(
            &pipeline_dag,
            &defaults,
            Path::new("/workspace"),
        )
        .unwrap();

        assert!(yaml.contains("run: generate vectors"));
        assert!(yaml.contains("run: compute knn"));
        assert!(yaml.contains("dim: '128'"));
        assert!(yaml.contains("source: base.fvec"));
        assert!(yaml.contains("id: gen"));
        assert!(yaml.contains("id: knn"));
        assert!(yaml.contains("- gen"));

        // Should not have defaults section when no defaults
        assert!(!yaml.contains("defaults:"));

        // Verify it round-trips
        let config: schema::PipelineConfig = serde_yaml::from_str(&yaml).unwrap();
        let resolved_steps = config.steps.unwrap();
        assert_eq!(resolved_steps.len(), 2);
        assert_eq!(resolved_steps[0].run, "generate vectors");
        assert_eq!(resolved_steps[1].run, "compute knn");
    }

    #[test]
    fn test_resolve_pipeline_yaml_with_defaults() {
        let steps = vec![make_step("s1", "analyze stats", vec![], vec![
            ("source", "${input_file}"),
        ])];
        let pipeline_dag = dag::build_dag(&steps).unwrap();

        let mut defaults = IndexMap::new();
        defaults.insert("input_file".to_string(), "data.fvec".to_string());
        defaults.insert("seed".to_string(), "42".to_string());

        let yaml = resolve_pipeline_yaml(
            &pipeline_dag,
            &defaults,
            Path::new("/workspace"),
        )
        .unwrap();

        // Interpolation should resolve ${input_file}
        assert!(yaml.contains("source: data.fvec"));
        assert!(!yaml.contains("${input_file}"));

        // Defaults should be present
        assert!(yaml.contains("defaults:"));
        assert!(yaml.contains("seed: '42'"));
    }

    #[test]
    fn test_resolve_pipeline_yaml_workspace_interpolation() {
        let steps = vec![make_step("s1", "analyze stats", vec![], vec![
            ("source", "${workspace}/data.fvec"),
        ])];
        let pipeline_dag = dag::build_dag(&steps).unwrap();
        let defaults = IndexMap::new();

        let yaml = resolve_pipeline_yaml(
            &pipeline_dag,
            &defaults,
            Path::new("/my/data"),
        )
        .unwrap();

        assert!(yaml.contains("source: /my/data/data.fvec"));
    }

    #[test]
    fn test_resolve_pipeline_yaml_preserves_dag_order() {
        // B depends on A; define them in reverse order
        let steps = vec![
            make_step("b", "compute knn", vec!["a"], vec![("source", "base.fvec")]),
            make_step("a", "generate vectors", vec![], vec![("output", "base.fvec")]),
        ];
        let pipeline_dag = dag::build_dag(&steps).unwrap();
        let defaults = IndexMap::new();

        let yaml = resolve_pipeline_yaml(
            &pipeline_dag,
            &defaults,
            Path::new("/workspace"),
        )
        .unwrap();

        // A should appear before B in the output (topological order)
        let a_pos = yaml.find("run: generate vectors").unwrap();
        let b_pos = yaml.find("run: compute knn").unwrap();
        assert!(a_pos < b_pos, "step A should appear before step B");
    }

    #[test]
    fn test_resolve_pipeline_yaml_unresolved_var_error() {
        let steps = vec![make_step("s1", "analyze stats", vec![], vec![
            ("source", "${undefined_var}"),
        ])];
        let pipeline_dag = dag::build_dag(&steps).unwrap();
        let defaults = IndexMap::new();

        let result = resolve_pipeline_yaml(
            &pipeline_dag,
            &defaults,
            Path::new("/workspace"),
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not defined"));
    }

    #[test]
    fn test_ensure_gitignore_creates_file() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        ensure_gitignore(workspace);

        let content = std::fs::read_to_string(workspace.join(".gitignore")).unwrap();
        assert!(content.contains("*"));
        assert!(content.contains("!dataset.yaml"));
        assert!(content.contains("!.gitignore"));
    }

    #[test]
    fn test_ensure_gitignore_does_not_duplicate() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        // Run twice
        ensure_gitignore(workspace);
        ensure_gitignore(workspace);

        let content = std::fs::read_to_string(workspace.join(".gitignore")).unwrap();
        // Each entry should appear exactly once
        assert_eq!(content.matches("!dataset.yaml").count(), 1);
        assert_eq!(content.matches("!.gitignore").count(), 1);
    }

    #[test]
    fn test_ensure_gitignore_preserves_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        // Write an existing gitignore with custom content
        std::fs::write(workspace.join(".gitignore"), "*.log\nbuild/\n").unwrap();

        ensure_gitignore(workspace);

        let content = std::fs::read_to_string(workspace.join(".gitignore")).unwrap();
        assert!(content.contains("*.log"));
        assert!(content.contains("build/"));
        assert!(content.contains("!dataset.yaml"));
    }

    #[test]
    fn test_filter_steps_shared_always_included() {
        let steps = vec![
            make_step("shared", "import", vec![], vec![("source", "x")]),
            make_step_with_profiles("default-only", "compute knn", vec![], vec![], vec!["default"]),
            make_step_with_profiles("ten-m", "compute knn", vec![], vec![], vec!["10M"]),
        ];
        let filtered = vectordata::dataset::filter_steps_for_profile(steps, "default");
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].effective_id(), "shared");
        assert_eq!(filtered[1].effective_id(), "default-only");
    }

    #[test]
    fn test_filter_steps_profile_10m() {
        let steps = vec![
            make_step("shared", "import", vec![], vec![]),
            make_step_with_profiles("default-only", "compute knn", vec![], vec![], vec!["default"]),
            make_step_with_profiles("ten-m", "compute knn", vec![], vec![], vec!["10M"]),
        ];
        let filtered = vectordata::dataset::filter_steps_for_profile(steps, "10M");
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].effective_id(), "shared");
        assert_eq!(filtered[1].effective_id(), "ten-m");
    }

    #[test]
    fn test_filter_steps_multi_profile_gate() {
        let steps = vec![
            make_step_with_profiles("multi", "compute knn", vec![], vec![], vec!["10M", "100M"]),
        ];
        assert_eq!(vectordata::dataset::filter_steps_for_profile(steps.clone(), "10M").len(), 1);
        assert_eq!(vectordata::dataset::filter_steps_for_profile(steps.clone(), "100M").len(), 1);
        assert_eq!(vectordata::dataset::filter_steps_for_profile(steps, "default").len(), 0);
    }

    #[test]
    fn test_filter_steps_no_profiles_means_shared() {
        let steps = vec![
            make_step("a", "import", vec![], vec![]),
            make_step("b", "compute knn", vec![], vec![]),
        ];
        // All steps are shared — any profile gets all of them
        assert_eq!(vectordata::dataset::filter_steps_for_profile(steps.clone(), "anything").len(), 2);
    }

    fn make_per_profile_step(
        id: &str,
        run: &str,
        after: Vec<&str>,
        opts: Vec<(&str, &str)>,
    ) -> schema::StepDef {
        let mut step = make_step(id, run, after, opts);
        step.per_profile = true;
        step
    }

    fn test_profile_group(yaml: &str) -> vectordata::dataset::DSProfileGroup {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn test_expand_per_profile_basic() {
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.mvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_step("shared", "import", vec![], vec![("output", "all.mvec")]),
            make_per_profile_step(
                "extract",
                "generate mvec-extract",
                vec!["shared"],
                vec![("output", "${profile_dir}base.mvec"), ("range", "[${query_count},${base_end})")],
            ),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps, &profiles, 10_000);
        // Should have: shared + extract-10M (sized first) + extract (default last)
        assert_eq!(expanded.len(), 3);
        assert_eq!(expanded[0].effective_id(), "shared");
        assert_eq!(expanded[1].effective_id(), "extract-10M");
        // Check variable resolution in options
        let output = expanded[1].output_path().unwrap();
        assert_eq!(output, "profiles/10M/base.mvec");
        let range_val = expanded[1].options.get("range").unwrap().as_str().unwrap();
        assert_eq!(range_val, "[10000,10010000)");
        // Should be gated to 10M
        assert_eq!(expanded[1].profiles, vec!["10M"]);
        assert_eq!(expanded[2].effective_id(), "extract");
        assert_eq!(expanded[2].profiles, vec!["default"]);
        // Classic layout: base_vectors has no profiles/ prefix → default outputs
        // go in the dataset root, not profiles/default/.
        assert_eq!(expanded[2].output_path().unwrap(), "base.mvec");
    }

    #[test]
    fn test_expand_per_profile_auto_prefix() {
        // Test that bare filenames get auto-prefixed without ${profile_dir}
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.mvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_per_profile_step("extract", "generate mvec-extract", vec![], vec![
                ("output", "base.mvec"),
            ]),
            make_per_profile_step("knn", "compute knn", vec!["extract"], vec![
                ("base", "base.mvec"),          // cross-ref to template output → prefixed
                ("query", "query.mvec"),         // not a template output → unchanged
                ("output", "gnd.ivec"),
            ]),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps, &profiles, 10_000);
        // default (2 steps) + 10M (2 steps) = 4
        assert_eq!(expanded.len(), 4);

        // Check 10M knn step
        let knn_10m = expanded.iter().find(|s| s.effective_id() == "knn-10M").unwrap();
        let base_val = knn_10m.options.get("base").unwrap().as_str().unwrap();
        assert_eq!(base_val, "profiles/10M/base.mvec"); // auto-prefixed
        let query_val = knn_10m.options.get("query").unwrap().as_str().unwrap();
        assert_eq!(query_val, "query.mvec"); // not a template output, unchanged
        assert_eq!(knn_10m.output_path().unwrap(), "profiles/10M/gnd.ivec");
    }

    #[test]
    fn test_expand_per_profile_multiple_profiles() {
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.mvec
10M:
  base_count: 10000000
20M:
  base_count: 20000000
"#);
        let steps = vec![
            make_per_profile_step("knn", "compute knn", vec![], vec![
                ("output", "${profile_dir}gnd.ivec"),
            ]),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps, &profiles, 10_000);
        // 10M + 20M + default = 3 (sized ascending, default last)
        assert_eq!(expanded.len(), 3);
        assert_eq!(expanded[0].effective_id(), "knn-10M");
        assert_eq!(expanded[0].output_path().unwrap(), "profiles/10M/gnd.ivec");
        assert_eq!(expanded[1].effective_id(), "knn-20M");
        assert_eq!(expanded[1].output_path().unwrap(), "profiles/20M/gnd.ivec");
        assert_eq!(expanded[2].effective_id(), "knn");
        assert_eq!(expanded[2].profiles, vec!["default"]);
        // Classic layout: default outputs go in dataset root.
        assert_eq!(expanded[2].output_path().unwrap(), "gnd.ivec");
    }

    #[test]
    fn test_expand_per_profile_after_rewriting() {
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.mvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_per_profile_step("extract", "generate mvec-extract", vec![], vec![
                ("output", "${profile_dir}base.mvec"),
            ]),
            make_per_profile_step("knn", "compute knn", vec!["extract", "shared-query"], vec![
                ("base", "${profile_dir}base.mvec"),
            ]),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps, &profiles, 10_000);
        // default (2) + 10M (2) = 4
        assert_eq!(expanded.len(), 4);

        // Check 10M knn step
        let knn = expanded.iter().find(|s| s.effective_id() == "knn-10M").unwrap();
        // "extract" is a per_profile template → suffixed
        assert!(knn.after.contains(&"extract-10M".to_string()));
        // "shared-query" is NOT a template → kept as-is
        assert!(knn.after.contains(&"shared-query".to_string()));

        // Check default knn step — template refs NOT suffixed for default
        let knn_default = expanded.iter().find(|s| s.effective_id() == "knn" && s.profiles.contains(&"default".to_string())).unwrap();
        assert!(knn_default.after.contains(&"extract".to_string()));
        assert!(knn_default.after.contains(&"shared-query".to_string()));
    }

    #[test]
    fn test_expand_no_per_profile_returns_unchanged() {
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.mvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_step("a", "import", vec![], vec![]),
            make_step("b", "compute knn", vec!["a"], vec![]),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps.clone(), &profiles, 10_000);
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn test_expand_default_profile_from_templates() {
        // Default profile gets per_profile expansion when no explicit steps exist
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.mvec
"#);
        let steps = vec![
            make_per_profile_step("extract", "generate mvec-extract", vec![], vec![
                ("output", "base.mvec"),
            ]),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps, &profiles, 10_000);
        // Template expanded for default (no suffix)
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].effective_id(), "extract");
        assert_eq!(expanded[0].profiles, vec!["default"]);
        // Classic layout: base_vectors has no profiles/ prefix → default outputs
        // go in the dataset root.
        assert_eq!(expanded[0].output_path().unwrap(), "base.mvec");
    }

    #[test]
    fn test_expand_default_skipped_when_explicit_steps_exist() {
        // When explicit default-gated steps exist, templates don't expand for default
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.mvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_step_with_profiles("extract", "generate mvec-extract", vec![], vec![
                ("output", "profiles/default/base.mvec"),
            ], vec!["default"]),
            make_per_profile_step("extract", "generate mvec-extract", vec![], vec![
                ("output", "base.mvec"),
            ]),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps, &profiles, 10_000);
        // explicit default step + 10M expansion = 2
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].effective_id(), "extract");
        assert_eq!(expanded[0].profiles, vec!["default"]);
        assert_eq!(expanded[1].effective_id(), "extract-10M");
        assert_eq!(expanded[1].profiles, vec!["10M"]);
    }

    #[test]
    fn test_expand_per_profile_indices_distances_prefixed() {
        // Verify that indices and distances keys (used by compute knn)
        // are auto-prefixed with the profile directory, not just "output".
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.mvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_per_profile_step("knn", "compute knn", vec![], vec![
                ("base", "base_vectors.mvec"),
                ("query", "query_vectors.mvec"),
                ("indices", "neighbor_indices.ivecs"),
                ("distances", "neighbor_distances.fvecs"),
            ]),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps, &profiles, 10_000);

        // 10M + default = 2
        assert_eq!(expanded.len(), 2);

        let knn_10m = expanded.iter().find(|s| s.effective_id() == "knn-10M").unwrap();
        let indices = knn_10m.options.get("indices").unwrap().as_str().unwrap();
        assert_eq!(indices, "profiles/10M/neighbor_indices.ivecs");
        let distances = knn_10m.options.get("distances").unwrap().as_str().unwrap();
        assert_eq!(distances, "profiles/10M/neighbor_distances.fvecs");
        // Non-output options should NOT be prefixed
        let query = knn_10m.options.get("query").unwrap().as_str().unwrap();
        assert_eq!(query, "query_vectors.mvec");
    }


    #[test]
    fn test_derive_sized_profile_views() {
        let mut profiles = test_profile_group(r#"
default:
  query_vectors: query.mvec
10M:
  base_count: 10000000
"#);
        let templates = vec![
            make_per_profile_step("extract", "generate mvec-extract", vec![], vec![
                ("output", "base_vectors.mvec"),
            ]),
            make_per_profile_step("knn", "compute knn", vec![], vec![
                ("output", "neighbor_indices.ivecs"),
            ]),
        ];
        profiles.derive_views_from_templates(&templates);

        // Default gets auto-derived views (no explicit base_vectors view)
        let pdef = profiles.profile("default").unwrap();
        assert_eq!(pdef.view("base_vectors").unwrap().path(), "profiles/default/base_vectors.mvec");
        assert_eq!(pdef.view("neighbor_indices").unwrap().path(), "profiles/default/neighbor_indices.ivecs");
        // Explicit view preserved
        assert_eq!(pdef.view("query_vectors").unwrap().path(), "query.mvec");

        // Sized profile gets window-based views referencing default's files
        let p10 = profiles.profile("10M").unwrap();
        assert_eq!(p10.view("base_vectors").unwrap().path(), "profiles/default/base_vectors.mvec");
        assert!(!p10.view("base_vectors").unwrap().source.window.is_empty());
        assert_eq!(p10.view("base_vectors").unwrap().source.window.0[0].max_excl, 10000000);
        assert_eq!(p10.view("neighbor_indices").unwrap().path(), "profiles/default/neighbor_indices.ivecs");
        assert!(!p10.view("neighbor_indices").unwrap().source.window.is_empty());
        // Inherited shared view unchanged
        assert_eq!(p10.view("query_vectors").unwrap().path(), "query.mvec");
    }

    // ─── --clean-last (rewind by one step) tests ────────────────────

    use std::collections::HashMap;
    use chrono::{TimeZone, Utc};
    use progress::{StepRecord, OutputRecord};
    use crate::pipeline::command::Status;

    /// Build a minimal `dataset.yaml` so `clean_last_step` can locate
    /// the progress file (it derives `<dir>/.cache/.upstream.progress.yaml`
    /// from the dataset path).
    fn make_dataset(dir: &Path) -> PathBuf {
        std::fs::create_dir_all(dir.join(".cache")).unwrap();
        let dsy = dir.join("dataset.yaml");
        std::fs::write(&dsy, "name: test\n").unwrap();
        dsy
    }

    /// Append a step record with a controllable timestamp + outputs.
    fn record(
        log: &mut ProgressLog,
        id: &str,
        completed_at: chrono::DateTime<Utc>,
        outputs: Vec<&str>,
        status: Status,
    ) {
        let outs = outputs.into_iter().map(|p| OutputRecord {
            path: p.to_string(),
            size: 0,
            mtime: None,
        }).collect();
        log.record_step(id, StepRecord {
            status,
            message: format!("{} done", id),
            completed_at,
            elapsed_secs: 1.0,
            outputs: outs,
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            fingerprint: None,
            build_version: None,
        });
    }

    /// `--clean-last` should remove the artifacts of the
    /// most-recently-completed step and drop its progress entry,
    /// while leaving older steps' artifacts and entries intact.
    #[test]
    fn clean_last_removes_only_most_recent_step() {
        let tmp = tempfile::tempdir().unwrap();
        let dsy = make_dataset(tmp.path());

        // Three step artifacts on disk
        let a = tmp.path().join("a.bin");
        let b = tmp.path().join("b.bin");
        let c = tmp.path().join("c.bin");
        std::fs::write(&a, b"a").unwrap();
        std::fs::write(&b, b"b").unwrap();
        std::fs::write(&c, b"c").unwrap();

        // Build progress log with strictly increasing completion times.
        let progress_path = ProgressLog::path_for_dataset(&dsy);
        let (mut log, _) = ProgressLog::load(&progress_path).unwrap();
        record(&mut log, "step-a", Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            vec![a.to_str().unwrap()], Status::Ok);
        record(&mut log, "step-b", Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 1).unwrap(),
            vec![b.to_str().unwrap()], Status::Ok);
        record(&mut log, "step-c", Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 2).unwrap(),
            vec![c.to_str().unwrap()], Status::Ok);
        log.save().unwrap();

        clean_last_step(tmp.path(), &dsy);

        // Most recent (c) is gone. Earlier ones survive.
        assert!(!c.exists(), "c.bin should have been removed");
        assert!(a.exists(), "a.bin must remain");
        assert!(b.exists(), "b.bin must remain");

        // Progress entry for step-c is gone; a and b remain.
        let (reloaded, _) = ProgressLog::load(&progress_path).unwrap();
        assert!(reloaded.steps.contains_key("step-a"));
        assert!(reloaded.steps.contains_key("step-b"));
        assert!(!reloaded.steps.contains_key("step-c"),
            "step-c's progress entry must be removed");
    }

    /// A failed (partially-completed) step is the most recent → it's
    /// what `--clean-last` rewinds. This is the "abort happened mid-
    /// step, want to retry" case.
    #[test]
    fn clean_last_rewinds_a_failed_step() {
        let tmp = tempfile::tempdir().unwrap();
        let dsy = make_dataset(tmp.path());
        let good = tmp.path().join("good.bin");
        let partial = tmp.path().join("partial.bin");
        std::fs::write(&good, b"complete").unwrap();
        std::fs::write(&partial, b"half").unwrap();

        let progress_path = ProgressLog::path_for_dataset(&dsy);
        let (mut log, _) = ProgressLog::load(&progress_path).unwrap();
        record(&mut log, "step-good", Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            vec![good.to_str().unwrap()], Status::Ok);
        record(&mut log, "step-aborted", Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 1).unwrap(),
            vec![partial.to_str().unwrap()], Status::Error);
        log.save().unwrap();

        clean_last_step(tmp.path(), &dsy);

        assert!(good.exists(), "earlier successful step's artifact must remain");
        assert!(!partial.exists(), "aborted step's partial artifact must be removed");

        let (reloaded, _) = ProgressLog::load(&progress_path).unwrap();
        assert!(reloaded.steps.contains_key("step-good"));
        assert!(!reloaded.steps.contains_key("step-aborted"));
    }

    /// Missing artifact paths are tolerated (logged + skipped) — the
    /// progress entry should still be removed so the next run
    /// re-executes the step.
    #[test]
    fn clean_last_tolerates_already_missing_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let dsy = make_dataset(tmp.path());

        let progress_path = ProgressLog::path_for_dataset(&dsy);
        let (mut log, _) = ProgressLog::load(&progress_path).unwrap();
        // Output path doesn't exist on disk.
        record(&mut log, "step-only",
            Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            vec![tmp.path().join("never-existed.bin").to_str().unwrap()],
            Status::Ok);
        log.save().unwrap();

        clean_last_step(tmp.path(), &dsy);

        let (reloaded, _) = ProgressLog::load(&progress_path).unwrap();
        assert!(!reloaded.steps.contains_key("step-only"),
            "progress entry must be removed even when artifacts were already gone");
    }

    /// Missing progress log: no-op (reports no work to do, doesn't
    /// crash). Equivalent to a fresh dataset that's never been run.
    #[test]
    fn clean_last_with_no_progress_log_is_safe() {
        let tmp = tempfile::tempdir().unwrap();
        let dsy = make_dataset(tmp.path());
        // Don't write any progress file.

        // Just shouldn't panic.
        clean_last_step(tmp.path(), &dsy);

        let progress_path = ProgressLog::path_for_dataset(&dsy);
        assert!(!progress_path.exists(),
            "no progress file should be created by --clean-last on a fresh tree");
    }

    /// An empty progress log (file exists but has no step records):
    /// no-op, doesn't crash, doesn't break the file.
    #[test]
    fn clean_last_with_empty_progress_log_is_safe() {
        let tmp = tempfile::tempdir().unwrap();
        let dsy = make_dataset(tmp.path());
        let progress_path = ProgressLog::path_for_dataset(&dsy);
        let (log, _) = ProgressLog::load(&progress_path).unwrap();
        log.save().unwrap();

        clean_last_step(tmp.path(), &dsy);

        // File still loadable, still empty.
        let (reloaded, _) = ProgressLog::load(&progress_path).unwrap();
        assert!(reloaded.steps.is_empty());
    }

}
