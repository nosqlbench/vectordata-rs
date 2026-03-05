// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline step execution engine.
//!
//! Executes steps in topological order, implementing:
//! - Skip-if-fresh: steps recorded as OK in the progress log with unchanged
//!   outputs are skipped
//! - Artifact state checks: partial outputs are deleted (or resumed) before
//!   re-execution
//! - Dry-run mode: prints the execution plan without running anything
//! - Progress tracking: records each step's result in the progress log

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use chrono::Utc;
use log::info;

use super::command::{ArtifactState, CommandResult, Options, Status, StreamContext};
use super::dag::ResolvedStep;
use super::interpolate;
use super::progress::{OutputRecord, StepRecord};
use super::registry::CommandRegistry;
use super::schema::OnPartial;

/// Execute a list of resolved steps in order.
///
/// Steps are expected to already be in topological order (from `build_dag`).
pub fn run_steps(
    steps: &[ResolvedStep],
    registry: &CommandRegistry,
    ctx: &mut StreamContext,
) -> Result<(), String> {
    let total = steps.len();
    eprintln!("Pipeline: {} step(s) to evaluate\n", total);

    for (i, step) in steps.iter().enumerate() {
        let step_num = i + 1;
        let prefix = format!("[{}/{}]", step_num, total);

        // 1. Check progress log → skip if recorded OK and outputs match
        if ctx.progress.is_step_fresh(&step.id) {
            eprintln!("{} {} — SKIP (fresh)", prefix, step.id);
            continue;
        }

        // 2. Resolve command from registry
        let factory = registry.get(&step.def.run).ok_or_else(|| {
            format!(
                "step '{}': unknown command '{}'. Available: {:?}",
                step.id,
                step.def.run,
                registry.command_paths()
            )
        })?;

        let mut cmd = factory();

        // 3. Interpolate options
        let resolved_opts = interpolate::interpolate_options(
            &step.def.options,
            &ctx.defaults,
            &ctx.workspace,
        )
        .map_err(|e| format!("step '{}': {}", step.id, e))?;

        let mut options = Options::new();
        for (k, v) in &resolved_opts {
            options.set(k, v);
        }

        // 4. Check artifact state
        if let Some(output_path) = step.def.output_path() {
            let full_path = if std::path::Path::new(&output_path).is_absolute() {
                PathBuf::from(&output_path)
            } else {
                ctx.workspace.join(&output_path)
            };

            let state = cmd.check_artifact(&full_path, &options);
            match state {
                ArtifactState::Complete => {
                    // Not in progress log but artifact is complete — record and skip
                    eprintln!("{} {} — SKIP (artifact complete)", prefix, step.id);
                    ctx.progress.record_step(
                        &step.id,
                        StepRecord {
                            status: Status::Ok,
                            message: "artifact already complete".to_string(),
                            completed_at: Utc::now(),
                            elapsed_secs: 0.0,
                            outputs: vec![OutputRecord {
                                path: output_path.clone(),
                                size: std::fs::metadata(&full_path)
                                    .map(|m| m.len())
                                    .unwrap_or(0),
                                mtime: None,
                            }],
                            resolved_options: resolved_opts.iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect(),
                            error: None,
                        },
                    );
                    if let Err(e) = ctx.progress.save() {
                        eprintln!("  warning: failed to save progress: {}", e);
                    }
                    continue;
                }
                ArtifactState::Partial => {
                    match step.def.on_partial {
                        OnPartial::Restart => {
                            eprintln!(
                                "{} {} — removing partial output '{}'",
                                prefix, step.id, output_path
                            );
                            if !ctx.dry_run {
                                let _ = std::fs::remove_file(&full_path);
                            }
                        }
                        OnPartial::Resume => {
                            eprintln!(
                                "{} {} — resuming from partial output '{}'",
                                prefix, step.id, output_path
                            );
                        }
                    }
                }
                ArtifactState::Absent => {
                    // Expected — will be created by the step
                }
            }
        }

        // 5. Validate required options from describe_options
        let option_descs = cmd.describe_options();
        for desc in &option_descs {
            if desc.required && !options.has(&desc.name) {
                // Use default if available
                if let Some(ref default_val) = desc.default {
                    options.set(&desc.name, default_val);
                } else {
                    let msg = format!(
                        "step '{}' ({}): missing required {} option '{}' — {}",
                        step.id,
                        cmd.command_path(),
                        desc.type_name,
                        desc.name,
                        desc.description,
                    );
                    if ctx.dry_run {
                        eprintln!("{} {} — ERROR: {}", prefix, step.id, msg);
                    } else {
                        return Err(msg);
                    }
                }
            }
        }

        // 6. Dry-run: print plan line and continue
        if ctx.dry_run {
            let opts_summary: Vec<String> = resolved_opts
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            eprintln!(
                "{} {} — WOULD RUN: {} [{}]",
                prefix,
                step.id,
                cmd.command_path(),
                opts_summary.join(", ")
            );
            continue;
        }

        // 6. Execute
        eprintln!("{} {} — running: {}", prefix, step.id, step.def.run);
        ctx.step_id = step.id.clone();
        let start = Instant::now();
        let result = cmd.execute(&options, ctx);
        let elapsed = start.elapsed();

        // 7. Record result
        let record = step_record_from_result(&result, &resolved_opts);
        ctx.progress.record_step(&step.id, record);

        match result.status {
            Status::Ok => {
                eprintln!(
                    "{} {} — OK ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                );
            }
            Status::Warning => {
                eprintln!(
                    "{} {} — WARNING ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                );
            }
            Status::Error => {
                eprintln!(
                    "{} {} — ERROR ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                );
                // Save progress so we can resume later
                if let Err(e) = ctx.progress.save() {
                    eprintln!("  warning: failed to save progress: {}", e);
                }
                return Err(format!("step '{}' failed: {}", step.id, result.message));
            }
        }

        // 8. Post-execution bound check
        if let Some(output_path) = step.def.output_path() {
            let full_path = if std::path::Path::new(&output_path).is_absolute() {
                PathBuf::from(&output_path)
            } else {
                ctx.workspace.join(&output_path)
            };
            let state = cmd.check_artifact(&full_path, &options);
            if state != ArtifactState::Complete {
                eprintln!(
                    "{} {} — WARNING: post-execution check: artifact '{}' is {:?}",
                    prefix, step.id, output_path, state
                );
            }
        }

        // 9. Save progress incrementally
        if let Err(e) = ctx.progress.save() {
            eprintln!("  warning: failed to save progress: {}", e);
        }

        info!(
            "Step {} completed in {:.1}s",
            step.id,
            elapsed.as_secs_f64()
        );
    }

    eprintln!("\nPipeline complete.");
    Ok(())
}

/// Build a `StepRecord` from a `CommandResult`.
fn step_record_from_result(
    result: &CommandResult,
    resolved_opts: &indexmap::IndexMap<String, String>,
) -> StepRecord {
    let outputs: Vec<OutputRecord> = result
        .produced
        .iter()
        .map(|p| {
            let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            OutputRecord {
                path: p.to_string_lossy().into_owned(),
                size,
                mtime: None,
            }
        })
        .collect();

    let error = if result.status == Status::Error {
        Some(result.message.clone())
    } else {
        None
    };

    StepRecord {
        status: result.status.clone(),
        message: result.message.clone(),
        completed_at: Utc::now(),
        elapsed_secs: result.elapsed.as_secs_f64(),
        outputs,
        resolved_options: resolved_opts
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<HashMap<_, _>>(),
        error,
    }
}
