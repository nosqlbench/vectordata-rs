// Copyright (c) DataStax, Inc.
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
pub mod predicate;
pub mod resource;
pub mod rng;
pub mod simd_distance;
pub mod progress;
pub mod registry;
pub mod runner;
pub mod schema;
pub mod variables;

use std::path::{Path, PathBuf};

use clap::Args;
use clap_complete::engine::ArgValueCompleter;
use indexmap::IndexMap;

use vectordata::dataset::DatasetConfig;
use command::StreamContext;
use progress::ProgressLog;
use registry::CommandRegistry;

/// CLI arguments for `veks run`.
#[derive(Args)]
pub struct RunArgs {
    /// Path to dataset.yaml (default: dataset.yaml in current directory)
    pub dataset: Option<PathBuf>,

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
    #[arg(long, default_value = "maximize", add = ArgValueCompleter::new(cli::governor_completer))]
    pub governor: String,

    /// Interval in milliseconds between UI status updates and TUI redraws.
    #[arg(long, default_value = "250", value_name = "MS")]
    pub status_interval: u64,
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
        .unwrap_or(Path::new("."))
        .to_path_buf();

    // Resolve workspace: --absolute makes all paths absolute; otherwise
    // keep as-is (may be relative to CWD).
    let workspace = if args.absolute {
        std::fs::canonicalize(&raw_workspace).unwrap_or_else(|e| {
            println!("Failed to resolve absolute path for {}: {}", raw_workspace.display(), e);
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

    let raw_steps = vectordata::dataset::collect_all_steps(&config);

    let query_count: u64 = config
        .upstream
        .as_ref()
        .and_then(|p| p.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let expanded_steps = vectordata::dataset::expand_per_profile_steps(raw_steps, &config.profiles, query_count);
    // Profile ordering is handled by the sequential executor — steps are
    // emitted in profile order (sized ascending, then default) by
    // expand_per_profile_steps, and the sequential runner processes them
    // in topological order. No synthetic barrier steps needed.

    let steps = if profile_name == "all" {
        expanded_steps
    } else {
        vectordata::dataset::filter_steps_for_profile(expanded_steps, profile_name)
    };

    if steps.is_empty() {
        println!(
            "No pipeline steps found for profile '{}' in {}",
            profile_name,
            dataset_path.display()
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
    );
}

/// Entry point for `veks run` — execute a command stream pipeline.
pub fn run_pipeline(args: RunArgs) {
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

    // Handle --clean
    if args.clean {
        clean_pipeline(&workspace, dataset_path);
        return;
    }

    // Create managed scratch and cache directories
    let scratch_dir = workspace.join(".scratch");
    let cache_dir = workspace.join(".cache");
    if let Err(e) = std::fs::create_dir_all(&scratch_dir) {
        println!("Failed to create scratch directory: {}", e);
        std::process::exit(1);
    }
    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        println!("Failed to create cache directory: {}", e);
        std::process::exit(1);
    }

    // Ensure .gitignore covers managed directories
    ensure_gitignore(&workspace);

    // Collect steps and expand per_profile templates
    let pipeline = config.upstream.as_ref();

    let raw_steps = vectordata::dataset::collect_all_steps(&config);

    // Read query_count from upstream defaults (used by per_profile expansion)
    let query_count: u64 = pipeline
        .and_then(|p| p.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    // Expand per_profile template steps into concrete profile-gated steps
    let expanded_steps = vectordata::dataset::expand_per_profile_steps(raw_steps, &config.profiles, query_count);

    // Auto-derive profile views for sized profiles from template outputs
    let template_steps: Vec<_> = vectordata::dataset::collect_all_steps(&config)
        .into_iter()
        .filter(|s| s.per_profile)
        .collect();
    config.profiles.derive_views_from_templates(&template_steps);

    // Insert barriers between profile groups (runtime injection, not in YAML)
    // Profile ordering is handled by the sequential executor — steps are
    // emitted in profile order (sized ascending, then default) by
    // expand_per_profile_steps, and the sequential runner processes them
    // in topological order. No synthetic barrier steps needed.

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
            dataset_path.display()
        );
        println!("Add an 'upstream' section with 'steps' to define a pipeline.");
        std::process::exit(1);
    }

    // Build DAG
    let pipeline_dag = dag::build_dag(&steps).unwrap_or_else(|e| {
        println!("Pipeline DAG error: {}", e);
        std::process::exit(1);
    });

    // Build defaults from pipeline config + variables.yaml + CLI overrides
    let mut defaults = IndexMap::new();
    if let Some(pipe) = pipeline {
        if let Some(ref defs) = pipe.defaults {
            defaults.extend(defs.clone());
        }
    }
    // Layer in variables.yaml (overrides upstream.defaults, but CLI wins)
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

    // Handle --emit-yaml: resolve and print the pipeline instead of executing
    if args.emit_yaml {
        emit_resolved_yaml(&pipeline_dag, &defaults, &workspace);
        return;
    }

    println!(
        "Pipeline: {} steps in topological order (profile: {})",
        pipeline_dag.steps.len(),
        profile_name,
    );
    for (i, step) in pipeline_dag.steps.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, step.id, step.def.run);
    }
    println!();

    // Load or create progress log
    let progress_path = ProgressLog::path_for_dataset(dataset_path);
    let (mut progress, schema_msg) = ProgressLog::load(&progress_path).unwrap_or_else(|e| {
        println!("Warning: failed to load progress log: {}", e);
        (ProgressLog::new(), None)
    });
    if let Some(msg) = schema_msg {
        println!("{}", msg);
    }

    // Invalidate progress log if dataset.yaml is newer
    if let Some(msg) = progress.invalidate_if_stale(dataset_path) {
        println!("{}", msg);
    }

    let threads = if args.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
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

    // Open run log in .cache/ for persistent pipeline execution history
    let run_log_path = cache_dir.join("run.log");
    let run_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&run_log_path)
        .ok()
        .map(|f| std::sync::Arc::new(std::sync::Mutex::new(std::io::BufWriter::new(f))));
    if let Some(ref log) = run_log {
        use std::io::Write;
        let mut w = log.lock().unwrap();
        let _ = writeln!(w, "\n--- veks run {} (profile: {}) ---",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            profile_name,
        );
        let _ = writeln!(w, "dataset: {}", dataset_path.display());
        let _ = writeln!(w, "steps: {}", pipeline_dag.steps.len());
        let _ = w.flush();
    }

    // Build execution context
    let dataset_name = config.name.clone();
    // Profile names ordered by size: sized ascending, then default last
    let all_profile_names: Vec<String> = {
        let profiles = &config.profiles;
        let mut sized: Vec<(&str, u64)> = profiles.0.iter()
            .filter(|(name, p)| name.as_str() != "default" && p.base_count.is_some())
            .map(|(name, p)| (name.as_str(), p.base_count.unwrap()))
            .collect();
        sized.sort_by_key(|(_, bc)| *bc);
        let mut names: Vec<String> = sized.into_iter().map(|(n, _)| n.to_string()).collect();
        if profiles.0.contains_key("default") {
            names.push("default".to_string());
        }
        names
    };
    let mut ctx = StreamContext {
        dataset_name,
        profile: profile_name.clone(),
        profile_names: all_profile_names,
        workspace,
        scratch: scratch_dir.clone(),
        cache: cache_dir,
        defaults,
        dry_run: args.dry_run,
        progress,
        threads,
        step_id: String::new(),
        governor,
        ui: crate::ui::auto_ui_handle_with_interval(std::time::Duration::from_millis(args.status_interval)),
        status_interval: std::time::Duration::from_millis(args.status_interval),
    };

    // Build command registry
    let registry = CommandRegistry::with_builtins();

    // Run
    let result = runner::run_steps(&pipeline_dag.steps, &registry, &mut ctx);

    // Retrieve buffered log messages before dropping the TUI.
    let console_log = ctx.ui.take_console_log();

    // Drop ctx (including the UI handle) to restore the terminal before
    // printing any messages. Without this, error output goes to the
    // alternate screen and is lost when the ratatui sink is dropped.
    drop(ctx);

    // Replay buffered log messages to stdout so they persist in scrollback.
    if !console_log.is_empty() {
        println!();
        for msg in &console_log {
            println!("{}", msg);
        }
    }

    match result {
        Err(e) => {
            eprintln!("\nPipeline failed: {}", e);
            eprintln!(
                "Scratch directory preserved for debugging: {}",
                scratch_dir.display()
            );
            eprintln!("Run with --clean to reset.");
            std::process::exit(1);
        }
        Ok(summary) => {
            print_run_summary(&summary);
        }
    }

    // On success, clean scratch directory contents
    clean_scratch_contents(&scratch_dir);
}

/// Print a post-TUI run summary to stdout.
///
/// This runs after the ratatui alternate screen is gone, so the output
/// persists in the user's terminal scrollback.
fn print_run_summary(summary: &runner::RunSummary) {
    use crate::term;
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
        println!("# The script is valid when run from: {}", std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "(unknown)".into()));
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

/// Clean up pipeline artifacts: remove progress log and scratch directory.
///
/// The `.cache/` directory is intentionally preserved — users can delete it
/// manually if they want to discard cached intermediates.
fn clean_pipeline(workspace: &Path, dataset_path: &Path) {
    let progress_path = ProgressLog::path_for_dataset(dataset_path);
    if progress_path.exists() {
        match std::fs::remove_file(&progress_path) {
            Ok(()) => println!("Removed {}", progress_path.display()),
            Err(e) => println!("Failed to remove {}: {}", progress_path.display(), e),
        }
    } else {
        println!("No progress log found at {}", progress_path.display());
    }

    // Delete scratch directory entirely
    let scratch_dir = workspace.join(".scratch");
    if scratch_dir.exists() {
        match std::fs::remove_dir_all(&scratch_dir) {
            Ok(()) => println!("Removed {}", scratch_dir.display()),
            Err(e) => println!("Failed to remove {}: {}", scratch_dir.display(), e),
        }
    }

    println!("Clean complete. Cache and output files are preserved.");
    println!(
        "To remove outputs, delete them manually from {}",
        workspace.display()
    );
}

/// Remove all files inside the scratch directory, leaving the directory itself.
fn clean_scratch_contents(scratch_dir: &Path) {
    if !scratch_dir.exists() {
        return;
    }
    let entries = match std::fs::read_dir(scratch_dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let _ = std::fs::remove_dir_all(&path);
        } else {
            let _ = std::fs::remove_file(&path);
        }
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
        assert_eq!(expanded[2].output_path().unwrap(), "profiles/default/base.mvec");
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
        assert_eq!(expanded[2].output_path().unwrap(), "profiles/default/gnd.ivec");
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
        // Output auto-prefixed with profiles/default/
        assert_eq!(expanded[0].output_path().unwrap(), "profiles/default/base.mvec");
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
                ("indices", "neighbor_indices.ivec"),
                ("distances", "neighbor_distances.fvec"),
            ]),
        ];
        let expanded = vectordata::dataset::expand_per_profile_steps(steps, &profiles, 10_000);

        // 10M + default = 2
        assert_eq!(expanded.len(), 2);

        let knn_10m = expanded.iter().find(|s| s.effective_id() == "knn-10M").unwrap();
        let indices = knn_10m.options.get("indices").unwrap().as_str().unwrap();
        assert_eq!(indices, "profiles/10M/neighbor_indices.ivec");
        let distances = knn_10m.options.get("distances").unwrap().as_str().unwrap();
        assert_eq!(distances, "profiles/10M/neighbor_distances.fvec");
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
                ("output", "neighbor_indices.ivec"),
            ]),
        ];
        profiles.derive_views_from_templates(&templates);

        // Default gets auto-derived views (no explicit base_vectors view)
        let pdef = profiles.profile("default").unwrap();
        assert_eq!(pdef.view("base_vectors").unwrap().path(), "profiles/default/base_vectors.mvec");
        assert_eq!(pdef.view("neighbor_indices").unwrap().path(), "profiles/default/neighbor_indices.ivec");
        // Explicit view preserved
        assert_eq!(pdef.view("query_vectors").unwrap().path(), "query.mvec");

        // Sized profile gets window-based views referencing default's files
        let p10 = profiles.profile("10M").unwrap();
        assert_eq!(p10.view("base_vectors").unwrap().path(), "profiles/default/base_vectors.mvec");
        assert!(!p10.view("base_vectors").unwrap().source.window.is_empty());
        assert_eq!(p10.view("base_vectors").unwrap().source.window.0[0].max_excl, 10000000);
        assert_eq!(p10.view("neighbor_indices").unwrap().path(), "profiles/default/neighbor_indices.ivec");
        assert!(!p10.view("neighbor_indices").unwrap().source.window.is_empty());
        // Inherited shared view unchanged
        assert_eq!(p10.view("query_vectors").unwrap().path(), "query.mvec");
    }

    #[test]
    fn test_clean_scratch_contents() {
        let tmp = tempfile::tempdir().unwrap();
        let scratch = tmp.path().join(".scratch");
        std::fs::create_dir_all(&scratch).unwrap();
        std::fs::write(scratch.join("tmp1.fvec"), b"data").unwrap();
        std::fs::write(scratch.join("tmp2.fvec"), b"data").unwrap();
        let subdir = scratch.join("subdir");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("nested.fvec"), b"data").unwrap();

        clean_scratch_contents(&scratch);

        // Directory itself should still exist
        assert!(scratch.exists());
        // But contents should be gone
        assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 0);
    }
}
