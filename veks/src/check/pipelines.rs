// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline checks: coverage and execution.
//!
//! **Coverage** — does the pipeline define all the steps needed for the
//! data files present? For example, are merkle steps defined for every
//! publishable file above the threshold?
//!
//! **Execution** — have all defined pipeline steps been run to completion
//! with fresh outputs?

use std::path::{Path, PathBuf};

use vectordata::dataset::DatasetConfig;

use crate::pipeline::progress::ProgressLog;
use crate::pipeline::schema::StepDef;

use super::CheckResult;

/// Check pipeline execution: all defined steps completed and fresh.
pub fn check(root: &Path, dataset_files: &[PathBuf]) -> CheckResult {
    if dataset_files.is_empty() {
        return CheckResult::fail("pipeline-execution", vec![
            "no dataset.yaml files found".to_string(),
        ]);
    }

    let mut failures: Vec<String> = Vec::new();
    let mut ok_count = 0;
    let mut total_steps = 0usize;
    let mut fresh_steps = 0usize;

    for dataset_path in dataset_files {
        let rel = dataset_path
            .strip_prefix(root)
            .unwrap_or(dataset_path)
            .to_string_lossy();

        match check_execution(dataset_path) {
            Ok(counts) => {
                ok_count += 1;
                total_steps += counts.total;
                fresh_steps += counts.fresh;
            }
            Err(msgs) => {
                for msg in msgs {
                    failures.push(format!("{}: {}", rel, msg));
                }
            }
        }
    }

    if failures.is_empty() {
        let mut result = CheckResult::ok("pipeline-execution");
        result.messages.push(format!(
            "{} dataset(s), {}/{} steps fresh",
            ok_count, fresh_steps, total_steps,
        ));
        result
    } else {
        CheckResult::fail("pipeline-execution", failures)
    }
}

/// Check pipeline coverage: are required steps (like merkle) defined?
///
/// Compares the publishable data files against pipeline step definitions
/// to find gaps — files that exist but have no corresponding pipeline step.
pub fn check_coverage(
    root: &Path,
    dataset_files: &[PathBuf],
    publishable: &[PathBuf],
    merkle_threshold: u64,
) -> CheckResult {
    if dataset_files.is_empty() {
        return CheckResult::fail("pipeline-coverage", vec![
            "no dataset.yaml files found".to_string(),
        ]);
    }

    let mut gaps: Vec<String> = Vec::new();

    for dataset_path in dataset_files {
        let rel = dataset_path
            .strip_prefix(root)
            .unwrap_or(dataset_path)
            .to_string_lossy();

        let _workspace = dataset_path.parent().unwrap_or(Path::new("."));

        // Load pipeline step outputs
        let defined_outputs = load_step_outputs(dataset_path);

        // Check if there's a blanket merkle create step (source: ".")
        // that covers all files in the workspace.
        let has_blanket_merkle = defined_outputs.iter()
            .any(|(cmd, output)| cmd == "merkle create" && (output == "." || output == "./"));

        // If a blanket merkle step exists, individual file coverage is
        // satisfied — the step walks the directory and creates .mref for everything.
        if has_blanket_merkle {
            continue;
        }

        // Check merkle coverage: every publishable file above the threshold
        // should have a corresponding merkle step or existing .mref
        for file in publishable {
            let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
            if size < merkle_threshold {
                continue;
            }

            if file.extension().map(|e| e == "mref").unwrap_or(false) {
                continue;
            }

            let file_rel = super::rel_display(file);

            let has_merkle_step = defined_outputs.iter()
                .any(|(cmd, output)| {
                    cmd == "merkle create" && output == &file_rel
                });

            let mref_path = file.with_extension(
                format!("{}.mref", file.extension().and_then(|e| e.to_str()).unwrap_or(""))
            );
            let has_mref = mref_path.exists();

            if !has_merkle_step && !has_mref {
                gaps.push(format!(
                    "{}: no merkle step for '{}' ({:.1} MiB)",
                    rel,
                    file_rel,
                    size as f64 / (1024.0 * 1024.0),
                ));
            }
        }
    }

    if gaps.is_empty() {
        let mut result = CheckResult::ok("pipeline-coverage");
        result.messages.push("all publishable files have pipeline coverage".to_string());
        result
    } else {
        CheckResult::fail("pipeline-coverage", gaps)
    }
}

// ── Execution check internals ────────────────────────────────────────────

struct StepCounts {
    total: usize,
    fresh: usize,
}

/// Check execution state for a single dataset.yaml.
fn check_execution(dataset_path: &Path) -> Result<StepCounts, Vec<String>> {
    let mut config = DatasetConfig::load(dataset_path)
        .map_err(|e| vec![format!("failed to parse: {}", e)])?;

    let workspace = dataset_path.parent().unwrap_or(Path::new("."));
    let steps = veks_pipeline::pipeline::resolve_all_steps(&mut config, workspace);
    if steps.is_empty() {
        return Ok(StepCounts { total: 0, fresh: 0 });
    }

    let progress_path = ProgressLog::path_for_dataset(dataset_path);
    if !progress_path.exists() {
        return Err(vec![format!(
            "no progress log — pipeline never run ({} steps defined)",
            steps.len(),
        )]);
    }

    let (progress, schema_msg) = ProgressLog::load(&progress_path)
        .map_err(|e| vec![format!("failed to load progress log: {}", e)])?;

    let mut errors: Vec<String> = Vec::new();

    if let Some(msg) = schema_msg {
        errors.push(msg);
    }

    // Per-step provenance checking (v5 ProvenanceMap under the active
    // selector) handles invalidation — no whole-log invalidation needed
    // here.

    let workspace = dataset_path.parent().unwrap_or(Path::new("."));
    let mut fresh = 0usize;

    for step in &steps {
        let step_id = step.effective_id();

        match progress.get_step(&step_id) {
            None => {
                errors.push(format!("step '{}': not executed", step_id));
            }
            Some(record) => {
                use crate::pipeline::command::Status;
                if record.status != Status::Ok {
                    errors.push(format!(
                        "step '{}': status {:?} — {}",
                        step_id, record.status, record.message,
                    ));
                    continue;
                }
                if let Some(reason) = progress.check_step_freshness(
                    &step_id, None, Some(workspace),
                ) {
                    errors.push(format!("step '{}': stale — {}", step_id, reason));
                } else {
                    fresh += 1;
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(StepCounts { total: steps.len(), fresh })
    } else {
        Err(errors)
    }
}

// ── Coverage check internals ─────────────────────────────────────────────

/// Load all (command, output_path) pairs from a dataset's pipeline steps.
fn load_step_outputs(dataset_path: &Path) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let config = match DatasetConfig::load(dataset_path) {
        Ok(c) => c,
        Err(_) => return results,
    };

    if let Some(ref pipeline) = config.upstream {
        if let Some(ref steps) = pipeline.steps {
            for step in steps {
                let cmd = step.run.clone();
                // Collect the "source" option for merkle create (that's the input file)
                if let Some(source) = step.options.get("source") {
                    if let Some(s) = source.as_str() {
                        results.push((cmd.clone(), s.to_string()));
                    }
                }
                if let Some(output) = step.options.get("output") {
                    if let Some(s) = output.as_str() {
                        results.push((cmd.clone(), s.to_string()));
                    }
                }
            }
        }
    }
    results
}

/// Collect pipeline steps from a DatasetConfig (shared steps only).
fn collect_steps(config: &DatasetConfig) -> Vec<StepDef> {
    let mut steps = Vec::new();
    if let Some(ref pipeline) = config.upstream {
        if let Some(ref shared_steps) = pipeline.steps {
            steps.extend(shared_steps.clone());
        }
    }
    steps
}
