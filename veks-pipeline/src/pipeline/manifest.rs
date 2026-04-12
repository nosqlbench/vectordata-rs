// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Workspace artifact manifest generation.
//!
//! Projects the complete set of input, output, and intermediate artifacts
//! for a pipeline by dry-running every step's `project_artifacts` method.
//! Used by the check system for extraneous file detection and cache cleanup.

use std::collections::HashSet;
use std::path::Path;

use indexmap::IndexMap;
use vectordata::dataset::DatasetConfig;

use super::command::{ArtifactManifest, Options};
use super::interpolate;
use super::registry::CommandRegistry;

/// The complete projected manifest for a workspace.
#[derive(Debug, serde::Serialize)]
pub struct WorkspaceManifest {
    /// All final output artifacts (publishable).
    pub final_artifacts: HashSet<String>,
    /// All intermediate artifacts (cache, not publishable).
    pub intermediates: HashSet<String>,
    /// All input artifacts consumed (may overlap with outputs of earlier steps).
    pub inputs: HashSet<String>,
    /// Per-step manifests.
    pub steps: Vec<ArtifactManifest>,
}

impl WorkspaceManifest {
    /// All cache paths that must be retained — the union of inputs,
    /// outputs, and intermediates that live under `.cache/` or `${cache}`.
    ///
    /// This is the authoritative set for cache cleanup: anything in
    /// `.cache/` not in this set is safe to delete. Anything in it is
    /// needed by the pipeline (as source data, intermediate computation,
    /// or cached state).
    pub fn retained_cache_paths(&self) -> HashSet<String> {
        let is_cache = |p: &str| {
            p.starts_with("${cache}") || p.starts_with(".cache/") || p.contains("/.cache/")
        };
        let mut retained = HashSet::new();
        for p in &self.inputs {
            if is_cache(p) { retained.insert(p.clone()); }
        }
        for p in &self.intermediates {
            if is_cache(p) { retained.insert(p.clone()); }
        }
        for p in &self.final_artifacts {
            if is_cache(p) { retained.insert(p.clone()); }
        }
        retained
    }
}

/// Build a workspace manifest by projecting all pipeline steps.
pub fn project_workspace(
    dataset_path: &Path,
    config: &DatasetConfig,
    registry: &CommandRegistry,
) -> Result<WorkspaceManifest, String> {
    let workspace = dataset_path.parent().unwrap_or(Path::new("."));

    // Use the fully expanded step set (including per-profile expansions)
    // so the manifest accounts for all per-profile output paths.
    let steps = vectordata::dataset::resolve_steps(config);
    if steps.is_empty() {
        return Ok(WorkspaceManifest {
            final_artifacts: HashSet::new(),
            intermediates: HashSet::new(),
            inputs: HashSet::new(),
            steps: Vec::new(),
        });
    }

    // Build defaults from upstream config
    let mut defaults = IndexMap::new();
    if let Some(ref pipe) = config.upstream {
        if let Some(ref defs) = pipe.defaults {
            defaults.extend(defs.clone());
        }
    }

    let mut all_final: HashSet<String> = HashSet::new();
    let mut all_intermediate: HashSet<String> = HashSet::new();
    let mut all_inputs: HashSet<String> = HashSet::new();
    let mut step_manifests: Vec<ArtifactManifest> = Vec::new();

    for step in &steps {
        let step_id = step.effective_id();

        // Resolve command
        let factory = match registry.get(&step.run) {
            Some(f) => f,
            None => continue, // Unknown command, skip
        };
        let cmd = factory();

        // Interpolate options
        let resolved_opts = match interpolate::interpolate_options(
            &step.options, &defaults, workspace,
        ) {
            Ok(opts) => opts,
            Err(_) => continue, // Can't resolve variables yet
        };

        let mut options = Options::new();
        for (k, v) in &resolved_opts {
            options.set(k, v);
        }

        let manifest = cmd.project_artifacts(&step_id, &options);

        for f in &manifest.outputs {
            all_final.insert(f.clone());
        }
        for f in &manifest.intermediates {
            all_intermediate.insert(f.clone());
        }
        for f in &manifest.inputs {
            all_inputs.insert(f.clone());
        }

        step_manifests.push(manifest);
    }

    // Also add profile view paths as final artifacts
    for (_, profile) in &config.profiles.profiles {
        for (_, view) in &profile.views {
            let source = &view.source.path;
            if !source.is_empty() {
                // Strip window notation
                let path = source.split('[').next().unwrap_or(source);
                all_final.insert(path.to_string());
            }
        }
    }

    Ok(WorkspaceManifest {
        final_artifacts: all_final,
        intermediates: all_intermediate,
        inputs: all_inputs,
        steps: step_manifests,
    })
}

// collect_all_steps, expand_per_profile_steps, and filter_steps_for_profile
// are now in vectordata::dataset::expansion (shared code).
