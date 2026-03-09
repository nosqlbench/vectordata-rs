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

pub mod bound;
pub mod cli;
pub mod command;
pub mod commands;
pub mod dag;
pub mod interpolate;
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
use clap_complete::engine::{ArgValueCandidates, ArgValueCompleter};
use indexmap::IndexMap;

use dataset::DatasetConfig;
use command::StreamContext;
use progress::ProgressLog;
use registry::CommandRegistry;

/// CLI arguments for `veks run`.
#[derive(Args)]
pub struct RunArgs {
    /// Path to dataset.yaml containing the pipeline definition.
    pub dataset: PathBuf,

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
    #[arg(long, default_value = "maximize", add = ArgValueCandidates::new(cli::governor_candidates))]
    pub governor: String,

    /// Interval in milliseconds between UI status updates and TUI redraws.
    #[arg(long, default_value = "1000", value_name = "MS")]
    pub status_interval: u64,
}

/// CLI arguments for `veks script`.
#[derive(Args)]
pub struct ScriptArgs {
    /// Path to dataset.yaml containing the pipeline definition.
    pub dataset: PathBuf,

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
    let dataset_path = &args.dataset;

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

    let raw_steps = collect_all_steps(&config);

    let query_count: u64 = config
        .upstream
        .as_ref()
        .and_then(|p| p.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let mut expanded_steps = expand_per_profile_steps(raw_steps, &config.profiles, query_count);
    insert_profile_barriers(&mut expanded_steps, &config.profiles);

    let steps = if profile_name == "all" {
        expanded_steps
    } else {
        filter_steps_for_profile(expanded_steps, profile_name)
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
    let dataset_path = &args.dataset;

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

    let raw_steps = collect_all_steps(&config);

    // Read query_count from upstream defaults (used by per_profile expansion)
    let query_count: u64 = pipeline
        .and_then(|p| p.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    // Expand per_profile template steps into concrete profile-gated steps
    let mut expanded_steps = expand_per_profile_steps(raw_steps, &config.profiles, query_count);

    // Auto-derive profile views for sized profiles from template outputs
    let template_steps: Vec<_> = collect_all_steps(&config)
        .into_iter()
        .filter(|s| s.per_profile)
        .collect();
    config.profiles.derive_views_from_templates(&template_steps);

    // Insert barriers between profile groups (runtime injection, not in YAML)
    insert_profile_barriers(&mut expanded_steps, &config.profiles);

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
        filter_steps_for_profile(expanded_steps, profile_name)
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

    // Drop ctx (including the UI handle) to restore the terminal before
    // printing any messages. Without this, error output goes to the
    // alternate screen and is lost when the ratatui sink is dropped.
    drop(ctx);

    if let Err(e) = result {
        eprintln!("\nPipeline failed: {}", e);
        eprintln!(
            "Scratch directory preserved for debugging: {}",
            scratch_dir.display()
        );
        eprintln!("Run with --clean to reset.");
        std::process::exit(1);
    }

    // On success, clean scratch directory contents
    clean_scratch_contents(&scratch_dir);
}

/// Collect all pipeline steps from the config's top-level upstream.
fn collect_all_steps(config: &DatasetConfig) -> Vec<schema::StepDef> {
    let mut steps = Vec::new();

    if let Some(ref pipeline) = config.upstream {
        if let Some(ref shared_steps) = pipeline.steps {
            steps.extend(shared_steps.clone());
        }
    }

    steps
}

/// Filter steps to only those that should run for the given profile.
///
/// A step runs if:
/// - It has no `profiles` field (shared step, always runs), OR
/// - Its `profiles` list contains the active profile name.
fn filter_steps_for_profile(steps: Vec<schema::StepDef>, profile: &str) -> Vec<schema::StepDef> {
    steps
        .into_iter()
        .filter(|step| step.profiles.is_empty() || step.profiles.iter().any(|p| p == profile))
        .collect()
}

/// Expand `per_profile` template steps into concrete steps for each profile.
///
/// For each profile (including "default" when no explicit default-gated steps
/// exist), template steps are cloned with:
/// - ID suffixed with `-{profile_name}` (no suffix for default)
/// - `profiles: [profile_name]` set
/// - `after` references to other template steps similarly suffixed
/// - Option values with `${profile_dir}`, `${base_count}`, `${base_end}`,
///   `${query_count}`, `${profile_name}` resolved
/// - Output paths and cross-references to template outputs auto-prefixed
///   with `profiles/{name}/` (so `${profile_dir}` is optional in YAML)
///
/// Template steps (per_profile=true) are removed from the output; their
/// expansions replace them. Non-template steps pass through unchanged.
fn expand_per_profile_steps(
    steps: Vec<schema::StepDef>,
    profiles: &dataset::DSProfileGroup,
    query_count: u64,
) -> Vec<schema::StepDef> {
    use std::collections::HashSet;

    let (templates, regular): (Vec<_>, Vec<_>) = steps
        .into_iter()
        .partition(|s| s.per_profile);

    if templates.is_empty() {
        return regular;
    }

    let template_ids: HashSet<String> = templates
        .iter()
        .map(|s| s.effective_id())
        .collect();

    // Collect bare output filenames from templates for auto-prefix matching.
    // When a non-output option value matches one of these filenames, it is
    // treated as a cross-reference and auto-prefixed with the profile dir.
    // Uses output_paths() to capture all output-like options (output,
    // indices, distances) — not just the "output" key.
    let template_output_names: HashSet<String> = templates
        .iter()
        .flat_map(|s| s.output_paths())
        .map(|p| p.strip_prefix("${profile_dir}").unwrap_or(&p).to_string())
        .collect();

    // Check if there are already explicit default-gated steps — if so, skip
    // expanding templates for default (backwards compatibility).
    let has_explicit_default_steps = regular
        .iter()
        .any(|s| s.profiles.iter().any(|p| p == "default"));

    let mut result = regular;

    // Collect all profiles to expand: sized ascending by base_count, then
    // default (full dataset) last.
    let mut sized: Vec<(&str, u64)> = profiles.0.iter()
        .filter(|(name, p)| name.as_str() != "default" && p.base_count.is_some())
        .map(|(name, p)| (name.as_str(), p.base_count.unwrap()))
        .collect();
    sized.sort_by_key(|(_, bc)| *bc);

    let mut all_profiles: Vec<(&str, Option<u64>)> = sized.into_iter()
        .map(|(name, bc)| (name, Some(bc)))
        .collect();
    if profiles.0.contains_key("default") && !has_explicit_default_steps {
        all_profiles.push(("default", None));
    }

    for (profile_name, base_count_opt) in &all_profiles {
        let profile_dir = format!("profiles/{}/", profile_name);
        let suffix = if *profile_name == "default" {
            String::new()
        } else {
            format!("-{}", profile_name)
        };

        let base_end_str = match base_count_opt {
            Some(bc) => (query_count + bc).to_string(),
            // Default profile: base_end resolves to ${vector_count} at runtime
            None => "${vector_count}".to_string(),
        };

        for template in &templates {
            let template_id = template.effective_id();
            let expanded_id = if suffix.is_empty() {
                template_id.clone()
            } else {
                format!("{}{}", template_id, suffix)
            };

            // Rewrite after references: template refs get suffixed, others stay
            let expanded_after: Vec<String> = template.after.iter().map(|dep| {
                if template_ids.contains(dep.as_str()) {
                    if suffix.is_empty() {
                        dep.clone()
                    } else {
                        format!("{}{}", dep, suffix)
                    }
                } else {
                    dep.clone()
                }
            }).collect();

            // Rewrite option values with profile variables and auto-prefix
            let mut expanded_options = template.options.clone();
            for (key, v) in expanded_options.iter_mut() {
                if let serde_yaml::Value::String(s) = v {
                    // If the value already uses ${profile_dir}, do explicit replacement
                    if s.contains("${profile_dir}") {
                        *s = s.replace("${profile_dir}", &profile_dir);
                    } else {
                        // Auto-prefix bare filenames:
                        // - output-like keys (output, indices, distances)
                        //   always get prefixed
                        // - other values get prefixed if they match a
                        //   template output filename (cross-references)
                        let bare = s.as_str();
                        let is_output_key = key == "output"
                            || key == "indices"
                            || key == "distances";
                        let should_prefix = !bare.contains('/')
                            && !bare.contains("${")
                            && (is_output_key || template_output_names.contains(bare));
                        if should_prefix {
                            *s = format!("{}{}", profile_dir, s);
                        }
                    }

                    // Substitute remaining profile variables
                    *s = s
                        .replace("${profile_name}", profile_name)
                        .replace("${base_end}", &base_end_str)
                        .replace("${query_count}", &query_count.to_string());
                    if let Some(bc) = base_count_opt {
                        *s = s.replace("${base_count}", &bc.to_string());
                    }
                }
            }

            // Auto-inject range for sized profile steps that don't already
            // have one.  This ensures per-profile steps only operate on the
            // profile's data window (ordinals [0, base_end)).
            if base_count_opt.is_some() && !expanded_options.contains_key("range") {
                let base_end = query_count + base_count_opt.unwrap();
                expanded_options.insert(
                    "range".to_string(),
                    serde_yaml::Value::String(format!("[0,{})", base_end)),
                );
            }

            result.push(schema::StepDef {
                id: Some(expanded_id),
                run: template.run.clone(),
                description: template.description.clone(),
                after: expanded_after,
                profiles: vec![profile_name.to_string()],
                per_profile: false,
                on_partial: template.on_partial.clone(),
                options: expanded_options,
            });
        }
    }

    result
}

/// Insert barrier steps between profile groups so that all steps for one
/// profile complete before the next profile's steps begin.
///
/// For each profile transition (shared → default → 10M → 20M ...), a synthetic
/// barrier step is inserted that depends on all steps from the previous profile.
fn insert_profile_barriers(
    steps: &mut Vec<schema::StepDef>,
    profiles: &dataset::DSProfileGroup,
) {
    // Barrier order: sized profiles ascending by base_count, then default last.
    // Shared steps (no profiles) are not in the barrier chain — they run
    // when their explicit deps are satisfied.
    let mut sized_profiles: Vec<(String, u64)> = profiles.0.iter()
        .filter(|(name, p)| name.as_str() != "default" && p.base_count.is_some())
        .map(|(name, p)| (name.clone(), p.base_count.unwrap()))
        .collect();
    sized_profiles.sort_by_key(|(_, bc)| *bc);

    let mut profile_order: Vec<String> = sized_profiles.into_iter().map(|(name, _)| name).collect();
    if profiles.0.contains_key("default") {
        profile_order.push("default".to_string());
    }

    if profile_order.len() <= 1 {
        return; // No barriers needed with only one profile
    }

    // Collect step IDs per profile group.
    // Shared steps (empty profiles) are NOT included in the barrier chain.
    // They run whenever their explicit `after` dependencies are satisfied.
    // Profile steps that need shared outputs already declare those deps directly.
    let step_ids_per_profile: Vec<(String, Vec<String>)> = profile_order
        .iter()
        .map(|pname| {
            let ids: Vec<String> = steps
                .iter()
                .filter(|s| s.profiles.iter().any(|p| p == pname))
                .map(|s| s.effective_id())
                .collect();
            (pname.clone(), ids)
        })
        .collect();

    // Insert barriers: for each profile after the first, add a barrier step
    // that depends on all steps from the previous profile
    for window in step_ids_per_profile.windows(2) {
        let prev_profile = &window[0];
        let next_profile = &window[1];

        if prev_profile.1.is_empty() || next_profile.1.is_empty() {
            continue;
        }

        let barrier_id = format!("barrier-{}", next_profile.0);

        // Make every step in the next profile depend on the barrier
        for step in steps.iter_mut() {
            if step.profiles.iter().any(|p| p == &next_profile.0) {
                if !step.after.contains(&barrier_id) {
                    step.after.push(barrier_id.clone());
                }
            }
        }

        // Insert the barrier step itself
        steps.push(schema::StepDef {
            id: Some(barrier_id),
            run: "barrier".to_string(),
            description: Some(format!(
                "Wait for all {} steps to complete before starting {}",
                prev_profile.0, next_profile.0
            )),
            after: prev_profile.1.clone(),
            profiles: vec![next_profile.0.clone()],
            per_profile: false,
            on_partial: schema::OnPartial::default(),
            options: IndexMap::new(),
        });
    }
}

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
        let filtered = filter_steps_for_profile(steps, "default");
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
        let filtered = filter_steps_for_profile(steps, "10M");
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].effective_id(), "shared");
        assert_eq!(filtered[1].effective_id(), "ten-m");
    }

    #[test]
    fn test_filter_steps_multi_profile_gate() {
        let steps = vec![
            make_step_with_profiles("multi", "compute knn", vec![], vec![], vec!["10M", "100M"]),
        ];
        assert_eq!(filter_steps_for_profile(steps.clone(), "10M").len(), 1);
        assert_eq!(filter_steps_for_profile(steps.clone(), "100M").len(), 1);
        assert_eq!(filter_steps_for_profile(steps, "default").len(), 0);
    }

    #[test]
    fn test_filter_steps_no_profiles_means_shared() {
        let steps = vec![
            make_step("a", "import", vec![], vec![]),
            make_step("b", "compute knn", vec![], vec![]),
        ];
        // All steps are shared — any profile gets all of them
        assert_eq!(filter_steps_for_profile(steps.clone(), "anything").len(), 2);
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

    fn test_profile_group(yaml: &str) -> dataset::DSProfileGroup {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn test_expand_per_profile_basic() {
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.hvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_step("shared", "import", vec![], vec![("output", "all.hvec")]),
            make_per_profile_step(
                "extract",
                "generate hvec-extract",
                vec!["shared"],
                vec![("output", "${profile_dir}base.hvec"), ("range", "[${query_count},${base_end})")],
            ),
        ];
        let expanded = expand_per_profile_steps(steps, &profiles, 10_000);
        // Should have: shared + extract-10M (sized first) + extract (default last)
        assert_eq!(expanded.len(), 3);
        assert_eq!(expanded[0].effective_id(), "shared");
        assert_eq!(expanded[1].effective_id(), "extract-10M");
        // Check variable resolution in options
        let output = expanded[1].output_path().unwrap();
        assert_eq!(output, "profiles/10M/base.hvec");
        let range_val = expanded[1].options.get("range").unwrap().as_str().unwrap();
        assert_eq!(range_val, "[10000,10010000)");
        // Should be gated to 10M
        assert_eq!(expanded[1].profiles, vec!["10M"]);
        assert_eq!(expanded[2].effective_id(), "extract");
        assert_eq!(expanded[2].profiles, vec!["default"]);
        assert_eq!(expanded[2].output_path().unwrap(), "profiles/default/base.hvec");
    }

    #[test]
    fn test_expand_per_profile_auto_prefix() {
        // Test that bare filenames get auto-prefixed without ${profile_dir}
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.hvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_per_profile_step("extract", "generate hvec-extract", vec![], vec![
                ("output", "base.hvec"),
            ]),
            make_per_profile_step("knn", "compute knn", vec!["extract"], vec![
                ("base", "base.hvec"),          // cross-ref to template output → prefixed
                ("query", "query.hvec"),         // not a template output → unchanged
                ("output", "gnd.ivec"),
            ]),
        ];
        let expanded = expand_per_profile_steps(steps, &profiles, 10_000);
        // default (2 steps) + 10M (2 steps) = 4
        assert_eq!(expanded.len(), 4);

        // Check 10M knn step
        let knn_10m = expanded.iter().find(|s| s.effective_id() == "knn-10M").unwrap();
        let base_val = knn_10m.options.get("base").unwrap().as_str().unwrap();
        assert_eq!(base_val, "profiles/10M/base.hvec"); // auto-prefixed
        let query_val = knn_10m.options.get("query").unwrap().as_str().unwrap();
        assert_eq!(query_val, "query.hvec"); // not a template output, unchanged
        assert_eq!(knn_10m.output_path().unwrap(), "profiles/10M/gnd.ivec");
    }

    #[test]
    fn test_expand_per_profile_multiple_profiles() {
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.hvec
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
        let expanded = expand_per_profile_steps(steps, &profiles, 10_000);
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
  base_vectors: base.hvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_per_profile_step("extract", "generate hvec-extract", vec![], vec![
                ("output", "${profile_dir}base.hvec"),
            ]),
            make_per_profile_step("knn", "compute knn", vec!["extract", "shared-query"], vec![
                ("base", "${profile_dir}base.hvec"),
            ]),
        ];
        let expanded = expand_per_profile_steps(steps, &profiles, 10_000);
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
  base_vectors: base.hvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_step("a", "import", vec![], vec![]),
            make_step("b", "compute knn", vec!["a"], vec![]),
        ];
        let expanded = expand_per_profile_steps(steps.clone(), &profiles, 10_000);
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn test_expand_default_profile_from_templates() {
        // Default profile gets per_profile expansion when no explicit steps exist
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.hvec
"#);
        let steps = vec![
            make_per_profile_step("extract", "generate hvec-extract", vec![], vec![
                ("output", "base.hvec"),
            ]),
        ];
        let expanded = expand_per_profile_steps(steps, &profiles, 10_000);
        // Template expanded for default (no suffix)
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].effective_id(), "extract");
        assert_eq!(expanded[0].profiles, vec!["default"]);
        // Output auto-prefixed with profiles/default/
        assert_eq!(expanded[0].output_path().unwrap(), "profiles/default/base.hvec");
    }

    #[test]
    fn test_expand_default_skipped_when_explicit_steps_exist() {
        // When explicit default-gated steps exist, templates don't expand for default
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.hvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_step_with_profiles("extract", "generate hvec-extract", vec![], vec![
                ("output", "profiles/default/base.hvec"),
            ], vec!["default"]),
            make_per_profile_step("extract", "generate hvec-extract", vec![], vec![
                ("output", "base.hvec"),
            ]),
        ];
        let expanded = expand_per_profile_steps(steps, &profiles, 10_000);
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
  base_vectors: base.hvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            make_per_profile_step("knn", "compute knn", vec![], vec![
                ("base", "base_vectors.hvec"),
                ("query", "query_vectors.hvec"),
                ("indices", "neighbor_indices.ivec"),
                ("distances", "neighbor_distances.fvec"),
            ]),
        ];
        let expanded = expand_per_profile_steps(steps, &profiles, 10_000);

        // 10M + default = 2
        assert_eq!(expanded.len(), 2);

        let knn_10m = expanded.iter().find(|s| s.effective_id() == "knn-10M").unwrap();
        let indices = knn_10m.options.get("indices").unwrap().as_str().unwrap();
        assert_eq!(indices, "profiles/10M/neighbor_indices.ivec");
        let distances = knn_10m.options.get("distances").unwrap().as_str().unwrap();
        assert_eq!(distances, "profiles/10M/neighbor_distances.fvec");
        // Non-output options should NOT be prefixed
        let query = knn_10m.options.get("query").unwrap().as_str().unwrap();
        assert_eq!(query, "query_vectors.hvec");
    }

    #[test]
    fn test_insert_profile_barriers() {
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.hvec
10M:
  base_count: 10000000
20M:
  base_count: 20000000
"#);
        let mut steps = vec![
            make_step("shared", "import", vec![], vec![]),
            make_step_with_profiles("def-knn", "compute knn", vec![], vec![], vec!["default"]),
            make_step_with_profiles("knn-10M", "compute knn", vec![], vec![], vec!["10M"]),
            make_step_with_profiles("knn-20M", "compute knn", vec![], vec![], vec!["20M"]),
        ];
        insert_profile_barriers(&mut steps, &profiles);

        // Order: 10M → 20M → default (sized ascending, default last).
        // Shared steps are NOT in the barrier chain.
        // Should have 2 barrier steps: barrier-20M, barrier-default.
        let barrier_steps: Vec<_> = steps.iter()
            .filter(|s| s.run == "barrier")
            .collect();
        assert_eq!(barrier_steps.len(), 2);

        // barrier-20M should depend on 10M's steps
        let b20 = barrier_steps.iter().find(|s| s.effective_id() == "barrier-20M").unwrap();
        assert!(b20.after.contains(&"knn-10M".to_string()));

        // barrier-default should depend on 20M's steps
        let bdef = barrier_steps.iter().find(|s| s.effective_id() == "barrier-default").unwrap();
        assert!(bdef.after.contains(&"knn-20M".to_string()));

        // def-knn should depend on barrier-default
        let def_knn = steps.iter().find(|s| s.effective_id() == "def-knn").unwrap();
        assert!(def_knn.after.contains(&"barrier-default".to_string()));

        // shared step should NOT have any barrier dependency
        let shared_step = steps.iter().find(|s| s.effective_id() == "shared").unwrap();
        assert!(shared_step.after.is_empty());
    }

    #[test]
    fn test_barriers_no_cycle_with_shared_deps_on_shared() {
        // Regression: shared steps must depend on other shared steps, NOT on
        // per-profile template outputs. This mirrors the real dataset.yaml
        // pattern where survey/synthesize run against the full imported data.
        // Pattern: import (shared) → survey (shared) → extract (per_profile) → evaluate (per_profile)
        let profiles = test_profile_group(r#"
default:
  base_vectors: base.hvec
10M:
  base_count: 10000000
"#);
        let steps = vec![
            // Shared import — no profiles, no per_profile
            make_step("import-metadata", "import", vec![], vec![
                ("output", "metadata_all.slab"),
            ]),
            // Shared survey depends on shared import (NOT per-profile extract)
            make_step("survey-metadata", "analyze survey", vec!["import-metadata"], vec![]),
            // Per-profile extract depends on shared import
            make_per_profile_step("extract-metadata", "transform slab-extract", vec!["import-metadata"], vec![
                ("output", "${profile_dir}meta.slab"),
            ]),
            // Per-profile evaluate depends on its own extract + shared survey
            make_per_profile_step("evaluate", "transform evaluate", vec!["extract-metadata", "survey-metadata"], vec![
                ("output", "${profile_dir}eval.slab"),
            ]),
        ];
        let expanded = expand_per_profile_steps(steps, &profiles, 10_000);
        let mut all_steps = expanded;
        insert_profile_barriers(&mut all_steps, &profiles);

        // Must not panic — if barriers created a cycle, build_dag would fail
        let result = dag::build_dag(&all_steps);
        assert!(result.is_ok(), "DAG cycle detected: {}", result.unwrap_err());
    }

    #[test]
    fn test_derive_sized_profile_views() {
        let mut profiles = test_profile_group(r#"
default:
  query_vectors: query.hvec
10M:
  base_count: 10000000
"#);
        let templates = vec![
            make_per_profile_step("extract", "generate hvec-extract", vec![], vec![
                ("output", "base_vectors.hvec"),
            ]),
            make_per_profile_step("knn", "compute knn", vec![], vec![
                ("output", "neighbor_indices.ivec"),
            ]),
        ];
        profiles.derive_views_from_templates(&templates);

        // Default gets auto-derived views (no explicit base_vectors view)
        let pdef = profiles.profile("default").unwrap();
        assert_eq!(pdef.view("base_vectors").unwrap().path(), "profiles/default/base_vectors.hvec");
        assert_eq!(pdef.view("neighbor_indices").unwrap().path(), "profiles/default/neighbor_indices.ivec");
        // Explicit view preserved
        assert_eq!(pdef.view("query_vectors").unwrap().path(), "query.hvec");

        // Sized profile gets window-based views referencing default's files
        let p10 = profiles.profile("10M").unwrap();
        assert_eq!(p10.view("base_vectors").unwrap().path(), "profiles/default/base_vectors.hvec");
        assert!(!p10.view("base_vectors").unwrap().source.window.is_empty());
        assert_eq!(p10.view("base_vectors").unwrap().source.window.0[0].max_excl, 10000000);
        assert_eq!(p10.view("neighbor_indices").unwrap().path(), "profiles/default/neighbor_indices.ivec");
        assert!(!p10.view("neighbor_indices").unwrap().source.window.is_empty());
        // Inherited shared view unchanged
        assert_eq!(p10.view("query_vectors").unwrap().path(), "query.hvec");
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
