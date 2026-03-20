// Copyright (c) DataStax, Inc.
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
use super::schema::StepDef;

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

/// Build a workspace manifest by projecting all pipeline steps.
pub fn project_workspace(
    dataset_path: &Path,
    config: &DatasetConfig,
    registry: &CommandRegistry,
) -> Result<WorkspaceManifest, String> {
    let workspace = dataset_path.parent().unwrap_or(Path::new("."));

    let steps = collect_all_steps(config);
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
    for (_, profile) in &config.profiles.0 {
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

fn collect_all_steps(config: &DatasetConfig) -> Vec<StepDef> {
    let mut steps = Vec::new();
    if let Some(ref pipeline) = config.upstream {
        if let Some(ref shared_steps) = pipeline.steps {
            steps.extend(shared_steps.clone());
        }
    }
    steps
}
