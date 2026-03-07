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

use std::collections::{HashMap, HashSet};
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

    // Build a display prefix with dataset name and profile context
    let ctx_label = match (ctx.dataset_name.is_empty(), ctx.profile.is_empty()) {
        (false, false) => format!("{} [{}]", ctx.dataset_name, ctx.profile),
        (false, true) => ctx.dataset_name.clone(),
        (true, false) => format!("[{}]", ctx.profile),
        (true, true) => String::new(),
    };

    if ctx_label.is_empty() {
        ctx.ui.log(&format!("Pipeline: {} step(s) to evaluate\n", total));
    } else {
        ctx.ui.log(&format!("{} — {} step(s) to evaluate\n", ctx_label, total));
    }

    // Track which steps actually executed (not skipped) for cascade invalidation.
    // If an upstream step ran, all downstream dependents must also run.
    let mut executed_steps: HashSet<String> = HashSet::new();

    let profile_count = ctx.profile_names.len();

    for (i, step) in steps.iter().enumerate() {
        let step_num = i + 1;

        // Determine which profile this step belongs to.
        let step_profile = step.def.profiles.first().map(|s| s.as_str()).unwrap_or("shared");
        let profile_idx = ctx.profile_names.iter().position(|p| p == step_profile);
        let profile_tag = if profile_count > 1 {
            if let Some(idx) = profile_idx {
                format!(" profile {}/{} ({})", idx + 1, profile_count, step_profile)
            } else {
                format!(" profile ({})", step_profile)
            }
        } else {
            String::new()
        };

        let prefix = if ctx_label.is_empty() {
            format!("[{}/{}]", step_num, total)
        } else {
            format!("{} [{}/{}]", ctx_label, step_num, total)
        };

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

        // 2. Cascade invalidation: if any dependency executed this run, force re-run
        let upstream_ran = step.def.after.iter().any(|dep| executed_steps.contains(dep));

        // 3. Check progress log → skip if recorded OK, outputs match, options unchanged,
        //    AND no upstream dependency ran this session
        if !upstream_ran && ctx.progress.is_step_fresh(&step.id, Some(&resolved_map)) {
            ctx.ui.log(&format!("{} {} — SKIP (fresh)", prefix, step.id));
            continue;
        }
        if upstream_ran {
            ctx.ui.log(&format!("{} {} — invalidated (upstream dependency re-ran)", prefix, step.id));
        }

        // 4. Resolve command from registry
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

        // 5. Check artifact state
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
                    ctx.ui.log(&format!("{} {} — SKIP (artifact complete)", prefix, step.id));
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
                        ctx.ui.log(&format!("  warning: failed to save progress: {}", e));
                    }
                    continue;
                }
                ArtifactState::Partial => {
                    match step.def.on_partial {
                        OnPartial::Restart => {
                            ctx.ui.log(&format!(
                                "{} {} — removing partial output '{}'",
                                prefix, step.id, output_path
                            ));
                            if !ctx.dry_run {
                                let _ = std::fs::remove_file(&full_path);
                            }
                        }
                        OnPartial::Resume => {
                            ctx.ui.log(&format!(
                                "{} {} — resuming from partial output '{}'",
                                prefix, step.id, output_path
                            ));
                        }
                    }
                }
                ArtifactState::Absent => {
                    // Ensure parent directories exist (e.g. profiles/default/)
                    if let Some(parent) = full_path.parent() {
                        if !parent.exists() && !ctx.dry_run {
                            let _ = std::fs::create_dir_all(parent);
                        }
                    }
                }
                ArtifactState::Unknown(ref reason) => {
                    let msg = format!(
                        "step '{}': artifact check failed: {}",
                        step.id, reason,
                    );
                    if ctx.dry_run {
                        ctx.ui.log(&format!("{} {} — ERROR: {}", prefix, step.id, msg));
                    } else {
                        return Err(msg);
                    }
                }
            }
        }

        // 6. Validate required options from describe_options
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
                        ctx.ui.log(&format!("{} {} — ERROR: {}", prefix, step.id, msg));
                    } else {
                        return Err(msg);
                    }
                }
            }
        }

        // 6b. Dry-run: print plan line and continue
        if ctx.dry_run {
            let opts_summary: Vec<String> = resolved_opts
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            ctx.ui.log(&format!(
                "{} {} — WOULD RUN: {} [{}]",
                prefix,
                step.id,
                cmd.command_path(),
                opts_summary.join(", ")
            ));
            continue;
        }

        // 6c. Check resource governance compliance (REQ-RM-11)
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
                                ctx.ui.log(&format!(
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

        // 7. Execute
        ctx.ui.clear();

        // Update UI context header with dataset/profile/step info
        if ctx_label.is_empty() {
            ctx.ui.set_context(format!("[{}/{}] {}{}", step_num, total, step.id, profile_tag));
        } else {
            ctx.ui.set_context(format!("{} [{}/{}] {}{}", ctx_label, step_num, total, step.id, profile_tag));
        }

        if let Some(ref desc) = step.def.description {
            ctx.ui.log(&format!("{} {} — {}", prefix, step.id, desc));
        }
        ctx.ui.log(&format!("{} {} — running: {}", prefix, step.id, step.def.run));
        ctx.step_id = step.id.clone();
        ctx.governor.set_step_id(&step.id);

        // Resource status — updated by background thread via ui.resource_status()
        // and emergency logging.
        let resource_source = ctx.governor.status_source();
        let resource_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let resource_stop2 = resource_stop.clone();
        let resource_ui = ctx.ui.clone();
        let resource_handle = std::thread::Builder::new()
            .name("resource-status".into())
            .spawn(move || {
                let mut emergency_ticks = 0u32;
                // Immediate first sample so the status line appears right away.
                resource_ui.resource_status(resource_source.status_line());
                while !resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    if resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }

                    // Push resource status to the UI for display and charting.
                    resource_ui.resource_status(resource_source.status_line());

                    // If EMERGENCY persists, abort the process to prevent system
                    // lockup.  Base grace period is 10s (20 ticks at 500ms), but
                    // shrinks by 1s (2 ticks) for every 1% RSS is over ceiling.
                    // Going back under ceiling resets the counter.
                    if resource_source.is_emergency() {
                        emergency_ticks += 1;
                        let overage = resource_source.rss_overage_pct();
                        let reduction = (overage.floor() as u32) * 2; // 2 ticks = 1s per 1%
                        let grace_ticks = 20u32.saturating_sub(reduction).max(2); // floor at 1s
                        if emergency_ticks >= grace_ticks {
                            resource_ui.log(&format!(
                                "FATAL: resource emergency for {:.1}s, RSS {:.0}% over ceiling — aborting to prevent system lockup",
                                emergency_ticks as f64 * 0.5,
                                overage,
                            ));
                            resource_ui.log(&format!(
                                "  last status: {}", resource_source.status_line()
                            ));
                            std::process::exit(1);
                        }
                    } else {
                        emergency_ticks = 0;
                    }
                }
            })
            .ok();

        // Capture baseline snapshot for resource delta
        let baseline_snapshot = super::resource::SystemSnapshot::sample();
        let start = Instant::now();
        let result = cmd.execute(&options, ctx);
        let elapsed = start.elapsed();

        // Stop resource monitoring thread
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

        // 8. Record result
        ctx.ui.clear();
        let record = step_record_from_result(&result, &resolved_opts, resource_summary);
        ctx.progress.record_step(&step.id, record);

        match result.status {
            Status::Ok => {
                executed_steps.insert(step.id.clone());
                ctx.ui.log(&format!(
                    "{} {} — OK ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                ));
            }
            Status::Warning => {
                executed_steps.insert(step.id.clone());
                ctx.ui.log(&format!(
                    "{} {} — WARNING ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                ));
            }
            Status::Error => {
                ctx.ui.log(&format!(
                    "{} {} — ERROR ({:.1}s): {}",
                    prefix,
                    step.id,
                    elapsed.as_secs_f64(),
                    result.message
                ));
                // Save progress so we can resume later
                if let Err(e) = ctx.progress.save() {
                    ctx.ui.log(&format!("  warning: failed to save progress: {}", e));
                }
                return Err(format!("step '{}' failed: {}", step.id, result.message));
            }
        }

        // 9. Post-execution bound check — non-Complete artifacts are errors
        if let Some(output_path) = step.def.output_path() {
            let full_path = if std::path::Path::new(&output_path).is_absolute() {
                PathBuf::from(&output_path)
            } else {
                ctx.workspace.join(&output_path)
            };
            let state = cmd.check_artifact(&full_path, &options);
            if state != ArtifactState::Complete {
                let msg = format!(
                    "step '{}': post-execution check failed — artifact '{}' is {:?}",
                    step.id, output_path, state
                );
                ctx.ui.log(&format!("{} {} — ERROR: {}", prefix, step.id, msg));
                // Re-record as error so progress log reflects the failure
                ctx.progress.record_step(
                    &step.id,
                    StepRecord {
                        status: Status::Error,
                        message: msg.clone(),
                        completed_at: Utc::now(),
                        elapsed_secs: elapsed.as_secs_f64(),
                        outputs: vec![],
                        resolved_options: resolved_opts
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                        error: Some(msg.clone()),
                        resource_summary: None,
                    },
                );
                if let Err(e) = ctx.progress.save() {
                    ctx.ui.log(&format!("  warning: failed to save progress: {}", e));
                }
                return Err(msg);
            }
        }

        // 9b. Zero-length produced file check — any produced file with size 0 is an error
        for produced_path in &result.produced {
            if let Ok(meta) = std::fs::metadata(produced_path) {
                if meta.len() == 0 {
                    let msg = format!(
                        "step '{}': produced file '{}' is empty (zero bytes)",
                        step.id,
                        produced_path.display(),
                    );
                    ctx.ui.log(&format!("{} {} — ERROR: {}", prefix, step.id, msg));
                    ctx.progress.record_step(
                        &step.id,
                        StepRecord {
                            status: Status::Error,
                            message: msg.clone(),
                            completed_at: Utc::now(),
                            elapsed_secs: elapsed.as_secs_f64(),
                            outputs: vec![],
                            resolved_options: resolved_opts
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect(),
                            error: Some(msg.clone()),
                            resource_summary: None,
                        },
                    );
                    if let Err(e) = ctx.progress.save() {
                        ctx.ui.log(&format!("  warning: failed to save progress: {}", e));
                    }
                    return Err(msg);
                }
            }
        }

        // 10. Save progress incrementally
        if let Err(e) = ctx.progress.save() {
            ctx.ui.log(&format!("  warning: failed to save progress: {}", e));
        }

        info!(
            "Step {} completed in {:.1}s",
            step.id,
            elapsed.as_secs_f64()
        );
    }

    if ctx_label.is_empty() {
        ctx.ui.log("\nPipeline complete.");
    } else {
        ctx.ui.log(&format!("\n{} — pipeline complete.", ctx_label));
    }
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
