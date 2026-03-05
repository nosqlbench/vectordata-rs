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
pub mod rng;
pub mod simd_distance;
pub mod progress;
pub mod registry;
pub mod runner;
pub mod schema;

use std::path::{Path, PathBuf};

use clap::Args;
use indexmap::IndexMap;

use crate::import::dataset::DatasetConfig;
use command::StreamContext;
use progress::ProgressLog;
use registry::CommandRegistry;

/// CLI arguments for `veks run`.
#[derive(Args)]
pub struct RunArgs {
    /// Path to dataset.yaml containing the pipeline definition.
    pub dataset: PathBuf,

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
}

/// Entry point for `veks run` — execute a command stream pipeline.
pub fn run_pipeline(args: RunArgs) {
    let dataset_path = &args.dataset;

    // Load dataset config
    let config = DatasetConfig::load(dataset_path).unwrap_or_else(|e| {
        eprintln!("{}", e);
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
        eprintln!("Failed to create scratch directory: {}", e);
        std::process::exit(1);
    }
    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        eprintln!("Failed to create cache directory: {}", e);
        std::process::exit(1);
    }

    // Ensure .gitignore covers managed directories
    ensure_gitignore(&workspace);

    // Collect steps from top-level upstream and per-facet upstreams
    let pipeline = config.upstream.as_ref();

    let steps: Vec<schema::StepDef> = collect_all_steps(&config);

    if steps.is_empty() {
        eprintln!("No pipeline steps found in {}", dataset_path.display());
        eprintln!("Add an 'upstream' section with 'steps' to define a pipeline.");
        std::process::exit(1);
    }

    // Build DAG
    let pipeline_dag = dag::build_dag(&steps).unwrap_or_else(|e| {
        eprintln!("Pipeline DAG error: {}", e);
        std::process::exit(1);
    });

    // Build defaults from pipeline config + CLI overrides
    let mut defaults = IndexMap::new();
    if let Some(pipe) = pipeline {
        if let Some(ref defs) = pipe.defaults {
            defaults.extend(defs.clone());
        }
    }
    for ov in &args.overrides {
        if let Some((key, value)) = ov.split_once('=') {
            defaults.insert(key.to_string(), value.to_string());
        } else {
            eprintln!("Invalid --set format: '{}' (expected key=value)", ov);
            std::process::exit(1);
        }
    }

    // Handle --emit-yaml: resolve and print the pipeline instead of executing
    if args.emit_yaml {
        emit_resolved_yaml(&pipeline_dag, &defaults, &workspace);
        return;
    }

    eprintln!(
        "Pipeline: {} steps in topological order",
        pipeline_dag.steps.len()
    );
    for (i, step) in pipeline_dag.steps.iter().enumerate() {
        eprintln!("  {}. {} ({})", i + 1, step.id, step.def.run);
    }
    eprintln!();

    // Load or create progress log
    let progress_path = ProgressLog::path_for_dataset(dataset_path);
    let progress = ProgressLog::load(&progress_path).unwrap_or_else(|e| {
        eprintln!("Warning: failed to load progress log: {}", e);
        ProgressLog::new()
    });

    let threads = if args.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        args.threads
    };

    // Build execution context
    let mut ctx = StreamContext {
        workspace,
        scratch: scratch_dir.clone(),
        cache: cache_dir,
        defaults,
        dry_run: args.dry_run,
        progress,
        threads,
        step_id: String::new(),
    };

    // Build command registry
    let registry = CommandRegistry::with_builtins();

    // Run
    if let Err(e) = runner::run_steps(&pipeline_dag.steps, &registry, &mut ctx) {
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

/// Emit the fully resolved pipeline as YAML to stdout.
///
/// Delegates to [`resolve_pipeline_yaml`] and prints the result.
fn emit_resolved_yaml(
    pipeline_dag: &dag::PipelineDag,
    defaults: &IndexMap<String, String>,
    workspace: &Path,
) {
    let yaml = resolve_pipeline_yaml(pipeline_dag, defaults, workspace).unwrap_or_else(|e| {
        eprintln!("{}", e);
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
            after: step.def.after.clone(),
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
            Ok(()) => eprintln!("Removed {}", progress_path.display()),
            Err(e) => eprintln!("Failed to remove {}: {}", progress_path.display(), e),
        }
    } else {
        eprintln!("No progress log found at {}", progress_path.display());
    }

    // Delete scratch directory entirely
    let scratch_dir = workspace.join(".scratch");
    if scratch_dir.exists() {
        match std::fs::remove_dir_all(&scratch_dir) {
            Ok(()) => eprintln!("Removed {}", scratch_dir.display()),
            Err(e) => eprintln!("Failed to remove {}: {}", scratch_dir.display(), e),
        }
    }

    eprintln!("Clean complete. Cache and output files are preserved.");
    eprintln!(
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
        eprintln!(
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
        let mut options = IndexMap::new();
        for (k, v) in opts {
            options.insert(k.to_string(), serde_yaml::Value::String(v.to_string()));
        }
        schema::StepDef {
            id: Some(id.to_string()),
            run: run.to_string(),
            after: after.into_iter().map(String::from).collect(),
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
