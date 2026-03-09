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

use chrono::{DateTime, Utc};
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
        if upstream_ran {
            ctx.ui.log(&format!("{} {} — invalidated (upstream dependency re-ran)", prefix, step.id));
        } else {
            match ctx.progress.check_step_freshness(&step.id, Some(&resolved_map), Some(&ctx.workspace)) {
                None => {
                    ctx.ui.log(&format!("{} {} — SKIP (fresh)", prefix, step.id));
                    continue;
                }
                Some(reason) => {
                    // Only log the reason for non-trivial cases (not "not recorded")
                    if !reason.starts_with("not recorded") {
                        ctx.ui.log(&format!("{} {} — stale: {}", prefix, step.id, reason));
                    }
                }
            }
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

        // 5. Check artifact state (use interpolated output path, not raw)
        let resolved_output = resolved_opts.get("output").cloned();
        let mut use_buffer_write = false;
        if let Some(ref output_path) = resolved_output {
            let full_path = if std::path::Path::new(output_path.as_str()).is_absolute() {
                PathBuf::from(output_path)
            } else {
                ctx.workspace.join(output_path)
            };
            let buffer_path = buffer_path_for(&full_path);

            let state = cmd.check_artifact(&full_path, &options);
            match state {
                ArtifactState::Complete => {
                    // Not in progress log but artifact is complete — record and skip
                    ctx.ui.log(&format!("{} {} — SKIP (artifact complete)", prefix, step.id));
                    // Clean up any stale buffer file
                    let _ = std::fs::remove_file(&buffer_path);
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
                                let _ = std::fs::remove_file(&buffer_path);
                            }
                            // Restart writes to buffer
                            use_buffer_write = true;
                        }
                        OnPartial::Resume => {
                            ctx.ui.log(&format!(
                                "{} {} — resuming from partial output '{}'",
                                prefix, step.id, output_path
                            ));
                            // Resume writes directly — command manages its own state
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
                    // Clean up any stale buffer from a prior crash
                    let _ = std::fs::remove_file(&buffer_path);
                    // New writes go to buffer
                    use_buffer_write = true;
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

        // 5b. Buffer-write: redirect the output option to a _buffer path.
        // The command writes to the buffer; on success, the runner renames
        // it to the final path. This prevents partially-written files from
        // being mistaken for valid artifacts.
        let buffer_final_rename: Option<(PathBuf, PathBuf)> = if use_buffer_write {
            if let Some(ref output_path) = resolved_output {
                let full_path = if std::path::Path::new(output_path.as_str()).is_absolute() {
                    PathBuf::from(output_path)
                } else {
                    ctx.workspace.join(output_path)
                };
                let buf_path = buffer_path_for(&full_path);
                // Rewrite the output option to the buffer path
                let buf_str = buf_path.to_string_lossy().to_string();
                options.set("output", &buf_str);
                Some((buf_path, full_path))
            } else {
                None
            }
        } else {
            None
        };

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

        // Send the step definition as YAML for the TUI step panel.
        match serde_yaml::to_string(&step.def) {
            Ok(yaml) => ctx.ui.set_step_yaml(yaml),
            Err(_) => ctx.ui.set_step_yaml(""),
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
        let poll_interval = ctx.status_interval;
        let resource_handle = std::thread::Builder::new()
            .name("resource-status".into())
            .spawn(move || {
                let mut emergency_ticks = 0u32;
                let tick_secs = poll_interval.as_secs_f64();
                // Immediate first sample so the status line appears right away.
                {
                    let (line, metrics) = resource_source.status_line_with_metrics();
                    resource_ui.resource_status_with_metrics(line, metrics);
                }
                while !resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(poll_interval);
                    if resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }

                    // Push resource status to the UI for display and charting.
                    {
                        let (line, metrics) = resource_source.status_line_with_metrics();
                        resource_ui.resource_status_with_metrics(line, metrics);
                    }

                    // If EMERGENCY persists, abort the process to prevent system
                    // lockup.  Base grace period is ~10s worth of ticks, but
                    // shrinks for every 1% RSS is over ceiling.
                    // Going back under ceiling resets the counter.
                    if resource_source.is_emergency() {
                        emergency_ticks += 1;
                        let overage = resource_source.rss_overage_pct();
                        let base_ticks = (10.0 / tick_secs).ceil() as u32;
                        let reduction = overage.floor() as u32;
                        let grace_ticks = base_ticks.saturating_sub(reduction).max(1);
                        if emergency_ticks >= grace_ticks {
                            resource_ui.log(&format!(
                                "FATAL: resource emergency for {:.1}s, RSS {:.0}% over ceiling — aborting to prevent system lockup",
                                emergency_ticks as f64 * tick_secs,
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
            Status::Ok | Status::Warning => {
                // 8a. Buffer-write: rename buffer → final path
                if let Some((ref buf_path, ref final_path)) = buffer_final_rename {
                    if buf_path.exists() {
                        if let Err(e) = std::fs::rename(buf_path, final_path) {
                            let msg = format!(
                                "step '{}': failed to rename buffer '{}' → '{}': {}",
                                step.id,
                                buf_path.display(),
                                final_path.display(),
                                e,
                            );
                            ctx.ui.log(&format!("{} {} — ERROR: {}", prefix, step.id, msg));
                            if let Err(e) = ctx.progress.save() {
                                ctx.ui.log(&format!("  warning: failed to save progress: {}", e));
                            }
                            return Err(msg);
                        }
                    }
                }

                executed_steps.insert(step.id.clone());
                let level = if result.status == Status::Ok { "OK" } else { "WARNING" };
                ctx.ui.log(&format!(
                    "{} {} — {} ({:.1}s): {}",
                    prefix,
                    step.id,
                    level,
                    elapsed.as_secs_f64(),
                    result.message
                ));
            }
            Status::Error => {
                // Leave buffer file in place for diagnostics; clean up on next run
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
        //    Check the final path (buffer has been renamed at this point).
        if let Some(ref output_path) = resolved_output {
            let full_path = if std::path::Path::new(output_path.as_str()).is_absolute() {
                PathBuf::from(output_path)
            } else {
                ctx.workspace.join(output_path)
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

/// Compute the buffer path for a given output path.
///
/// Appends `_buffer` before the file extension (or at the end if no extension):
/// - `foo.slab` → `foo_buffer.slab`
/// - `data.hvec` → `data_buffer.hvec`
/// - `noext` → `noext_buffer`
fn buffer_path_for(path: &std::path::Path) -> PathBuf {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path.extension().map(|e| e.to_string_lossy());
    let buffered_name = match ext {
        Some(e) => format!("{}_buffer.{}", stem, e),
        None => format!("{}_buffer", stem),
    };
    path.with_file_name(buffered_name)
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
            let meta = std::fs::metadata(p).ok();
            let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let mtime = meta
                .and_then(|m| m.modified().ok())
                .and_then(|t| {
                    let duration = t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok()?;
                    let dt = DateTime::<Utc>::from(std::time::UNIX_EPOCH + duration);
                    Some(dt.to_rfc3339())
                });
            OutputRecord {
                path: p.to_string_lossy().into_owned(),
                size,
                mtime,
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
