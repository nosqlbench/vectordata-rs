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
use super::progress::{OutputRecord, ResourceSummary, StepRecord};
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
    ctx.display.log(&format!("Pipeline: {} step(s) to evaluate\n", total));

    for (i, step) in steps.iter().enumerate() {
        let step_num = i + 1;
        let prefix = format!("[{}/{}]", step_num, total);

        // 1. Interpolate options early so we can compare against the progress log
        let resolved_opts = interpolate::interpolate_options(
            &step.def.options,
            &ctx.defaults,
            &ctx.workspace,
        )
        .map_err(|e| format!("step '{}': {}", step.id, e))?;

        let resolved_map: HashMap<String, String> = resolved_opts
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // 2. Check progress log → skip if recorded OK, outputs match, and options unchanged
        if ctx.progress.is_step_fresh(&step.id, Some(&resolved_map)) {
            ctx.display.log(&format!("{} {} — SKIP (fresh)", prefix, step.id));
            continue;
        }

        // 3. Resolve command from registry
        let factory = registry.get(&step.def.run).ok_or_else(|| {
            format!(
                "step '{}': unknown command '{}'. Available: {:?}",
                step.id,
                step.def.run,
                registry.command_paths()
            )
        })?;

        let mut cmd = factory();

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
                    ctx.display.log(&format!("{} {} — SKIP (artifact complete)", prefix, step.id));
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
                            resource_summary: None,
                        },
                    );
                    if let Err(e) = ctx.progress.save() {
                        ctx.display.log(&format!("  warning: failed to save progress: {}", e));
                    }
                    continue;
                }
                ArtifactState::Partial => {
                    match step.def.on_partial {
                        OnPartial::Restart => {
                            ctx.display.log(&format!(
                                "{} {} — removing partial output '{}'",
                                prefix, step.id, output_path
                            ));
                            if !ctx.dry_run {
                                let _ = std::fs::remove_file(&full_path);
                            }
                        }
                        OnPartial::Resume => {
                            ctx.display.log(&format!(
                                "{} {} — resuming from partial output '{}'",
                                prefix, step.id, output_path
                            ));
                        }
                    }
                }
                ArtifactState::Absent => {
                    // Expected — will be created by the step
                }
                ArtifactState::Unknown(ref reason) => {
                    let msg = format!(
                        "step '{}': artifact check failed: {}",
                        step.id, reason,
                    );
                    if ctx.dry_run {
                        ctx.display.log(&format!("{} {} — ERROR: {}", prefix, step.id, msg));
                    } else {
                        return Err(msg);
                    }
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
                        ctx.display.log(&format!("{} {} — ERROR: {}", prefix, step.id, msg));
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
            ctx.display.log(&format!(
                "{} {} — WOULD RUN: {} [{}]",
                prefix,
                step.id,
                cmd.command_path(),
                opts_summary.join(", ")
            ));
            continue;
        }

        // 5b. Check resource governance compliance (REQ-RM-11)
        {
            let resources = cmd.describe_resources();
            let declared_names: std::collections::HashSet<&str> =
                resources.iter().map(|r| r.name.as_str()).collect();

            if resources.is_empty() {
                // Check if any file option points to a large file
                for (k, v) in &resolved_opts {
                    let path = if std::path::Path::new(v).is_absolute() {
                        std::path::PathBuf::from(v)
                    } else {
                        ctx.workspace.join(v)
                    };
                    if path.is_file() {
                        if let Ok(meta) = std::fs::metadata(&path) {
                            if meta.len() > 1_073_741_824 {
                                // > 1 GiB
                                ctx.display.log(&format!(
                                    "{} {} — WARNING: option '{}' references {} ({:.1} GB) \
                                     but command declares no resource requirements. \
                                     Resource governance is disabled for this step.",
                                    prefix,
                                    step.id,
                                    k,
                                    v,
                                    meta.len() as f64 / 1_000_000_000.0,
                                ));
                                break;
                            }
                        }
                    }
                }
            }

            // Log any budget resources not declared by this command (§6.4.4a)
            for budget_resource in ctx.governor.budget_resource_names() {
                if !declared_names.contains(budget_resource.as_str()) {
                    ctx.governor.log_ignored(cmd.command_path(), &budget_resource);
                }
            }
        }

        // 6. Execute
        ctx.display.clear();
        if let Some(ref desc) = step.def.description {
            ctx.display.log(&format!("{} {} — {}", prefix, step.id, desc));
        }
        ctx.display.log(&format!("{} {} — running: {}", prefix, step.id, step.def.run));
        ctx.step_id = step.id.clone();
        ctx.governor.set_step_id(&step.id);

        // Start resource status bar
        let resource_bar = ctx.display.resource_bar();
        let resource_source = ctx.governor.status_source();
        resource_bar.set_message(resource_source.status_line());
        let resource_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let resource_stop2 = resource_stop.clone();
        let resource_handle = std::thread::Builder::new()
            .name("resource-status".into())
            .spawn(move || {
                while !resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    if resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    resource_bar.set_message(resource_source.status_line());
                }
                resource_bar.finish_and_clear();
            })
            .ok();

        // Capture baseline snapshot for resource delta
        let baseline_snapshot = super::resource::SystemSnapshot::sample();
        let start = Instant::now();
        let result = cmd.execute(&options, ctx);
        let elapsed = start.elapsed();

        // Stop resource status bar
        resource_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = resource_handle {
            let _ = h.join();
        }

        // Capture end snapshot and compute resource summary
        let end_snapshot = super::resource::SystemSnapshot::sample();
        let tps = if end_snapshot.ticks_per_sec > 0.0 {
            end_snapshot.ticks_per_sec
        } else {
            1.0
        };
        let resource_summary = Some(ResourceSummary {
            peak_rss_bytes: end_snapshot.rss_bytes.max(baseline_snapshot.rss_bytes),
            cpu_user_secs: (end_snapshot.cpu_user_ticks.saturating_sub(baseline_snapshot.cpu_user_ticks)) as f64 / tps,
            cpu_system_secs: (end_snapshot.cpu_system_ticks.saturating_sub(baseline_snapshot.cpu_system_ticks)) as f64 / tps,
            io_read_bytes: end_snapshot.io_read_bps.saturating_sub(baseline_snapshot.io_read_bps),
            io_write_bytes: end_snapshot.io_write_bps.saturating_sub(baseline_snapshot.io_write_bps),
        });

        // 7. Record result
        ctx.display.clear();
        let record = step_record_from_result(&result, &resolved_opts, resource_summary);
        ctx.progress.record_step(&step.id, record);

        match result.status {
            Status::Ok => {
                ctx.display.log(&format!(
                    "{} {} — OK ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                ));
            }
            Status::Warning => {
                ctx.display.log(&format!(
                    "{} {} — WARNING ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                ));
            }
            Status::Error => {
                ctx.display.log(&format!(
                    "{} {} — ERROR ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                ));
                // Save progress so we can resume later
                if let Err(e) = ctx.progress.save() {
                    ctx.display.log(&format!("  warning: failed to save progress: {}", e));
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
                ctx.display.log(&format!(
                    "{} {} — WARNING: post-execution check: artifact '{}' is {:?}",
                    prefix, step.id, output_path, state
                ));
            }
        }

        // 9. Save progress incrementally
        if let Err(e) = ctx.progress.save() {
            ctx.display.log(&format!("  warning: failed to save progress: {}", e));
        }

        info!(
            "Step {} completed in {:.1}s",
            step.id,
            elapsed.as_secs_f64()
        );
    }

    ctx.display.log("\nPipeline complete.");
    Ok(())
}

/// Build a `StepRecord` from a `CommandResult`.
fn step_record_from_result(
    result: &CommandResult,
    resolved_opts: &indexmap::IndexMap<String, String>,
    resource_summary: Option<ResourceSummary>,
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
        resource_summary,
    }
}
