// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Auto-repair for dataset pipelines.
//!
//! When `veks check --fix` detects missing pipeline steps (e.g., merkle
//! tree generation), this module augments the `dataset.yaml` with the
//! required steps. A timestamped backup is created before any modification.

use std::path::{Path, PathBuf};

use vectordata::dataset::DatasetConfig;

/// Examine a dataset.yaml and propose fixes for missing pipeline steps.
///
/// Returns a list of YAML step blocks to append, and advisory messages
/// for issues that can't be auto-fixed.
pub struct FixPlan {
    /// YAML step blocks to append to upstream.steps.
    pub steps_to_add: Vec<String>,
    /// Advisory messages (user must act manually).
    pub advisories: Vec<String>,
    /// Path to the dataset.yaml being fixed.
    pub dataset_path: PathBuf,
}

/// Analyze a dataset.yaml and build a fix plan.
pub fn plan_fixes(
    dataset_path: &Path,
    missing_merkle_files: &[PathBuf],
    missing_publish: bool,
    stale_catalogs: bool,
    stale_pipeline_steps: &[String],
) -> FixPlan {
    let mut plan = FixPlan {
        steps_to_add: Vec::new(),
        advisories: Vec::new(),
        dataset_path: dataset_path.to_path_buf(),
    };

    let workspace = dataset_path.parent().unwrap_or(Path::new("."));

    // Load existing step IDs to avoid duplicates
    let existing_ids = load_existing_step_ids(dataset_path);

    // ── Missing merkle coverage ─────────────────────────────────────
    // If there are files without .mref and no "merkle create" step that
    // covers the workspace, add a single directory-walking step.
    if !missing_merkle_files.is_empty() {
        let has_merkle_step = existing_ids.iter().any(|id|
            id == "merkle-all" || id == "generate-merkle"
            || id.starts_with("merkle") || id.contains("merkle")
        ) || has_merkle_create_step(dataset_path);
        if !has_merkle_step {
            let step = concat!(
                "    - id: merkle-all\n",
                "      description: Create merkle hash trees for all publishable data files\n",
                "      run: merkle create\n",
                "      source: .\n",
            ).to_string();
            plan.steps_to_add.push(step);
        }
    }

    let rel = super::rel_display;

    // ── Stale pipeline (needs re-run) ────────────────────────────────
    if !stale_pipeline_steps.is_empty() {
        plan.advisories.push(format!(
            "Execute pipeline to complete {} stale step(s):\n  veks run {}",
            stale_pipeline_steps.len(),
            rel(dataset_path),
        ));
    }

    // ── Missing .publish_url ──────────────────────────────────────────
    if missing_publish {
        plan.advisories.push(format!(
            "Create a publish URL file:\n  echo 's3://your-bucket/prefix/' > {}/.publish_url",
            rel(workspace),
        ));
    }

    // ── Stale/missing catalogs ───────────────────────────────────────
    if stale_catalogs {
        plan.advisories.push(format!(
            "Regenerate catalog index:\n  veks prepare catalog generate {}",
            rel(workspace),
        ));
    }

    plan
}

/// Apply a fix plan: back up dataset.yaml and append new steps.
pub fn apply_fix(plan: &FixPlan) -> Result<(), String> {
    if plan.steps_to_add.is_empty() {
        return Ok(());
    }

    let dataset_path = &plan.dataset_path;

    // Create backup
    let backup_path = create_backup(dataset_path)?;
    println!("Backed up {} -> {}", super::rel_display(dataset_path), super::rel_display(&backup_path));

    // Read existing content
    let content = std::fs::read_to_string(dataset_path)
        .map_err(|e| format!("failed to read {}: {}", super::rel_display(dataset_path), e))?;

    // Find the insertion point: just before the "profiles:" line
    let insert_before = find_profiles_line(&content);

    let mut new_content = String::with_capacity(content.len() + plan.steps_to_add.len() * 200);

    match insert_before {
        Some(pos) => {
            new_content.push_str(&content[..pos]);
            // Add a section comment
            new_content.push_str(
                "    # ── Auto-added by veks check --fix ─────────────────────────\n\n"
            );
            for step in &plan.steps_to_add {
                new_content.push_str(step);
                new_content.push('\n');
            }
            new_content.push_str(&content[pos..]);
        }
        None => {
            // No profiles: line found — append steps section
            new_content.push_str(&content);
            if !content.ends_with('\n') {
                new_content.push('\n');
            }
            new_content.push_str(
                "\n    # ── Auto-added by veks check --fix ─────────────────────────\n\n"
            );
            for step in &plan.steps_to_add {
                new_content.push_str(step);
                new_content.push('\n');
            }
        }
    }

    std::fs::write(dataset_path, &new_content)
        .map_err(|e| format!("failed to write {}: {}", super::rel_display(dataset_path), e))?;

    println!(
        "Added {} step(s) to {}",
        plan.steps_to_add.len(),
        super::rel_display(dataset_path),
    );

    Ok(())
}

/// Create a timestamped backup of a file.
///
/// Writes to `.backup/dataset.yaml_<timestamp>` in the same directory.
pub(crate) fn create_backup(path: &Path) -> Result<PathBuf, String> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let backup_dir = dir.join(".backup");
    std::fs::create_dir_all(&backup_dir)
        .map_err(|e| format!("failed to create backup dir: {}", e))?;

    let filename = path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "dataset.yaml".to_string());

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let backup_name = format!("{}_{}", filename, timestamp);
    let backup_path = backup_dir.join(&backup_name);

    std::fs::copy(path, &backup_path)
        .map_err(|e| format!("failed to create backup: {}", e))?;

    Ok(backup_path)
}

/// Check if a dataset.yaml has a `merkle create` step (by command name).
fn has_merkle_create_step(dataset_path: &Path) -> bool {
    if let Ok(config) = DatasetConfig::load(dataset_path) {
        if let Some(ref pipeline) = config.upstream {
            if let Some(ref steps) = pipeline.steps {
                return steps.iter().any(|s| s.run == "merkle create");
            }
        }
    }
    false
}

/// Load the set of existing step IDs from a dataset.yaml.
fn load_existing_step_ids(dataset_path: &Path) -> std::collections::HashSet<String> {
    let mut ids = std::collections::HashSet::new();
    if let Ok(config) = DatasetConfig::load(dataset_path) {
        if let Some(ref pipeline) = config.upstream {
            if let Some(ref steps) = pipeline.steps {
                for step in steps {
                    ids.insert(step.effective_id());
                }
            }
        }
    }
    ids
}

/// Find the step ID that produces a given output path.
fn find_producing_step(dataset_path: &Path, output_rel: &str) -> Option<String> {
    let config = DatasetConfig::load(dataset_path).ok()?;
    let pipeline = config.upstream.as_ref()?;
    let steps = pipeline.steps.as_ref()?;

    for step in steps {
        // Check if this step's output matches
        if let Some(output) = step.options.get("output") {
            if let Some(s) = output.as_str() {
                if s == output_rel {
                    return Some(step.effective_id());
                }
            }
        }
        // Also check multi-output steps (indices, distances, etc.)
        for key in &["indices", "distances", "source"] {
            if let Some(val) = step.options.get(*key) {
                if let Some(s) = val.as_str() {
                    if s == output_rel {
                        return Some(step.effective_id());
                    }
                }
            }
        }
    }
    None
}

/// Find the byte offset of the "profiles:" line in a YAML string.
fn find_profiles_line(content: &str) -> Option<usize> {
    // Look for a line that starts with "profiles:" at the top level
    for (offset, line) in content.lines().scan(0usize, |pos, line| {
        let start = *pos;
        *pos += line.len() + 1; // +1 for newline
        Some((start, line))
    }) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("profiles:") && !line.starts_with(' ') {
            return Some(offset);
        }
    }
    None
}
