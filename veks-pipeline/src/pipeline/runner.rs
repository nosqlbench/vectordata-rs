// Copyright (c) Jonathan Shook
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
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, Utc};
use log::info;

/// Append-only provenance log for dataset preparation.
///
/// Records every pipeline run and step outcome in human-readable form
/// in `dataset.log` at the workspace root. This is a formal record of
/// how the data was prepared — not hidden in `.cache/`.
struct DatasetLog {
    writer: Option<std::io::BufWriter<std::fs::File>>,
}

impl DatasetLog {
    fn open(workspace: &std::path::Path) -> Self {
        let path = workspace.join("dataset.log");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok();
        DatasetLog {
            writer: file.map(std::io::BufWriter::new),
        }
    }

    fn log(&mut self, msg: &str) {
        if let Some(ref mut w) = self.writer {
            let _ = writeln!(w, "{}", msg);
        }
    }

    fn flush(&mut self) {
        if let Some(ref mut w) = self.writer {
            let _ = w.flush();
        }
    }
}

use super::command::{ArtifactState, CommandResult, Options, Status, StreamContext};
use super::dag::ResolvedStep;
use super::interpolate;
use super::progress::{OutputRecord, ResourceSummary, StepRecord};
use super::provenance::Address;
use super::registry::CommandRegistry;
use super::schema::OnPartial;

/// Summary of a pipeline run, returned to the caller for post-TUI display.
pub struct RunSummary {
    pub total: usize,
    pub skipped: usize,
    pub executed: usize,
    pub total_elapsed: std::time::Duration,
    /// Per-step outcomes: (step_id, was_executed, message)
    pub step_outcomes: Vec<(String, bool, String)>,
}

/// Execute a list of resolved steps in order.
///
/// Steps are expected to already be in topological order (from `build_dag`).
pub fn run_steps(
    steps: &[ResolvedStep],
    registry: &CommandRegistry,
    ctx: &mut StreamContext,
) -> Result<RunSummary, String> {
    let total = steps.len();

    // Build a display prefix with dataset name and profile context
    // The run's own profile selection (`all`, a name, or nothing); a
    // per-profile step narrows `ctx.profile` to its own while it runs.
    let run_profile = ctx.profile.clone();
    let ctx_label = match (ctx.dataset_name.is_empty(), ctx.profile.is_empty()) {
        (false, false) => format!("{} [{}]", ctx.dataset_name, ctx.profile),
        (false, true) => ctx.dataset_name.clone(),
        (true, false) => format!("[{}]", ctx.profile),
        (true, true) => String::new(),
    };

    let est_total = ctx.estimated_total_steps;
    let show_estimate = est_total > total;

    // Open the provenance log
    let mut dlog = DatasetLog::open(&ctx.workspace);
    dlog.log(&format!("\n=== veks run {} ===", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    dlog.log(&format!("pipeline: {} steps, profile: {}", total, ctx.profile));

    if ctx_label.is_empty() {
        if show_estimate {
            ctx.ui.log(&format!("Pipeline: {} step(s) ready, ~{} total with deferred profiles\n", total, est_total));
        } else {
            ctx.ui.log(&format!("Pipeline: {} step(s) to evaluate\n", total));
        }
    } else {
        if show_estimate {
            ctx.ui.log(&format!("{} — {} step(s) ready, ~{} total with deferred profiles\n", ctx_label, total, est_total));
        } else {
            ctx.ui.log(&format!("{} — {} step(s) to evaluate\n", ctx_label, total));
        }
    }

    // Track which steps actually executed (not skipped) for cascade invalidation.
    // Dry run: the outputs the plan would produce, by absolute path,
    // with the step producing each. At run time a step whose input is
    // newer than its output re-executes (the Make rule below); a dry
    // run has no new timestamps, so it plays the same rule against
    // the plan — a step reading what an earlier planned step writes
    // would run too. Without this a dry run shows only the first hop
    // of a cascade.
    let mut planned_outputs: HashMap<PathBuf, String> = HashMap::new();
    // Dry run: the planned steps that have outputs to write.
    let mut planned_with_outputs: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Summary counters for the final report.
    let mut skipped_count: usize = 0;
    let mut executed_count: usize = 0;
    let mut total_elapsed = std::time::Duration::ZERO;
    let mut step_outcomes: Vec<(String, bool, String)> = Vec::with_capacity(total);

    let overall_pb = ctx.ui.bar_with_unit(total as u64, "pipeline", "steps");

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

        let step_counter = if show_estimate {
            format!("[{}/{} of ~{}]", step_num, total, est_total)
        } else {
            format!("[{}/{}]", step_num, total)
        };
        let prefix = if ctx_label.is_empty() {
            step_counter.clone()
        } else {
            format!("{} {}", ctx_label, step_counter)
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

        // 2. Resolve command from registry (needed for build_version in fingerprint)
        let factory = registry.get(&step.def.run).ok_or_else(|| {
            format!(
                "step '{}': unknown command '{}'. Available: {:?}",
                step.id,
                step.def.run,
                registry.command_paths()
            )
        })?;

        let mut cmd = factory();
        let cmd_build_version = cmd.build_version().to_string();

        // 3. Build the structured provenance for this step.
        //    Captures identity, decomposed binary version, resolved
        //    options, and the address of every upstream step's node.
        //    Staleness is decided by hashing the node under the
        //    runtime-selected `ProvenanceFlags` selector; storing the
        //    full node lets us compare under a different selector on a
        //    later run without re-execution.
        let upstream_ids: Vec<&str> = step.def.after.iter()
            .map(|s| s.as_str())
            .collect();
        let provenance = ctx.progress.build_provenance(
            &step.id,
            &step.def.run,
            &resolved_map,
            &upstream_ids,
            &cmd_build_version,
        );
        let selector = ctx.provenance_selector;

        // 4. Check freshness: progress log + provenance under selector.
        //    The provenance check subsumes the v4 fingerprint cascade —
        //    if an upstream's recorded provenance differs (under the
        //    same selector), this step's hash differs.
        let provenance_reason = ctx.progress.check_provenance(&step.id, &provenance, selector);
        let planned_input: Option<(String, String)> = if ctx.dry_run && !planned_outputs.is_empty() {
            let descs = cmd.describe_options();
            resolved_opts.iter().find_map(|(key, val)| {
                let is_input = descs.iter().any(|d| d.name == *key && d.role == super::command::OptionRole::Input);
                if !is_input {
                    return None;
                }
                let path = resolve_in(val, &ctx.workspace);
                planned_outputs.get(&path).map(|producer| (val.clone(), producer.clone()))
            })
        } else {
            None
        };
        // A declared upstream that executed in this session makes the
        // step stale whatever its record says: `after` is the data
        // dependency (sequencing edges live apart), so what the
        // upstream just wrote is what this step reads. In a dry run
        // the steps that would run count the same way.
        // In a dry run nothing is written, so a planned upstream with
        // outputs stands for outputs newer than this step's record.
        // An upstream without outputs — a variable set — cascades
        // through the dependent's own resolved options instead.
        let upstream_planned: Option<String> = if ctx.dry_run {
            step.def.after.iter().find(|u| planned_with_outputs.contains(u.as_str())).cloned()
        } else {
            None
        };
        // An upstream that wrote a recorded output after this step's
        // record — it ran again, in this session or another, or was
        // declared since and ran later — makes the step stale: what it
        // read is not what the upstream now holds.
        let upstream_wrote: Option<(String, String)> = if ctx.dry_run {
            None
        } else {
            ctx.progress.upstream_wrote_after(&step.id, &step.def.after, Some(&ctx.workspace))
        };
        let progress_fresh = match ctx.progress.check_step_freshness(&step.id, Some(&resolved_map), Some(&ctx.workspace)) {
            None if provenance_reason.is_none() && upstream_planned.is_none() && upstream_wrote.is_none() && planned_input.is_none() => true,
            None if provenance_reason.is_none() => {
                let reason = match (&upstream_planned, &upstream_wrote, &planned_input) {
                    (Some(u), _, _) => format!("upstream '{}' runs in this plan", u),
                    (None, Some((u, path)), _) => format!("upstream '{}' wrote '{}' after this step's record", u, path),
                    (None, None, Some((input, producer))) => format!("input '{}' is produced by '{}' in this plan", input, producer),
                    (None, None, None) => unreachable!("one of the three made the step stale"),
                };
                ctx.ui.log(&format!("{} {} — stale: {}", prefix, step.id, reason));
                false
            }
            None => {
                // Outputs/options match but provenance changed — upstream config or build changed
                ctx.ui.log(&format!("{} {} — stale: {}", prefix, step.id,
                    provenance_reason.as_deref().unwrap_or("provenance changed")));
                false
            }
            Some(reason) => {
                if !reason.starts_with("not recorded") {
                    ctx.ui.log(&format!("{} {} — stale: {}", prefix, step.id, reason));
                }
                false
            }
        };

        let mut options = Options::new();
        for (k, v) in &resolved_opts {
            options.set(k, v);
        }

        // 5. Check artifact state — only when progress log confirmed the step
        //    succeeded previously. Without provenance, artifact existence alone
        //    is not trustworthy (could be stale from a different configuration).
        //    A fresh step with no output to check is simply fresh.
        let resolved_output = resolved_opts.get("output").cloned();
        if progress_fresh && resolved_output.is_none() {
            skipped_count += 1;
            step_outcomes.push((step.id.clone(), false, "fresh".into()));
            ctx.ui.log(&format!("{} {} — fresh, skipping", prefix, step.id));
            dlog.log(&format!("  [skip] {} — fresh", step.id));
            overall_pb.inc(1);
            continue;
        }
        if progress_fresh && let Some(ref output_path) = resolved_output {
            let full_path = resolve_in(output_path, &ctx.workspace);
            // The output's time: the file's, or the newest shard's when
            // the option names a sharded series.
            let output_mtime = std::fs::metadata(&full_path)
                .ok()
                .and_then(|m| m.modified().ok())
                .or_else(|| {
                    vectordata::dataset::shards::discover_shards(&full_path)
                        .iter()
                        .filter_map(|s| std::fs::metadata(s).ok().and_then(|m| m.modified().ok()))
                        .max()
                });
            // Make-style freshness: an input newer than the output makes
            // the step stale even when the record and the artifact agree.
            let newer_input: Option<String> = output_mtime.and_then(|out_t| {
                let descs = cmd.describe_options();
                resolved_opts.iter().find_map(|(key, val)| {
                    let is_input = descs.iter().any(|d| d.name == *key && d.role == super::command::OptionRole::Input);
                    if !is_input {
                        return None;
                    }
                    let in_t = std::fs::metadata(resolve_in(val, &ctx.workspace))
                        .ok()
                        .and_then(|m| m.modified().ok())?;
                    (in_t > out_t).then(|| val.clone())
                })
            });
            match cmd.check_artifact(&full_path, &options) {
                ArtifactState::Partial => {
                    match step.def.on_partial {
                        OnPartial::Restart => {
                            ctx.ui.log(&format!(
                                "{} {} — removing partial output '{}'",
                                prefix, step.id, output_path
                            ));
                            if !ctx.dry_run {
                                let _ = std::fs::remove_file(&full_path);
                                // Also try removing as directory for directory-based outputs
                                let _ = std::fs::remove_dir_all(&full_path);
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
                ArtifactState::PartialResumable => {
                    ctx.ui.log(&format!(
                        "{} {} — resuming incomplete output '{}'",
                        prefix, step.id, output_path
                    ));
                }
                _ => {
                    // Complete — or nothing the command can judge at
                    // the option path: a sharded series, an output
                    // named another way. The record verified every
                    // recorded output by size, so it stands, unless an
                    // input is newer.
                    if let Some(input) = newer_input {
                        ctx.ui.log(&format!(
                            "{} {} — stale: input '{}' is newer than the output",
                            prefix, step.id, input
                        ));
                    } else {
                        skipped_count += 1;
                        step_outcomes.push((step.id.clone(), false, "fresh".into()));
                        ctx.ui.log(&format!("{} {} — fresh, skipping", prefix, step.id));
                        dlog.log(&format!("  [skip] {} — fresh", step.id));
                        overall_pb.inc(1);
                        continue;
                    }
                }
            }
        }

        // Ensure output directory exists even when skipping artifact check
        if let Some(ref output_path) = resolved_output {
            let full_path = if std::path::Path::new(output_path.as_str()).is_absolute() {
                PathBuf::from(output_path)
            } else {
                ctx.workspace.join(output_path)
            };
            if let Some(parent) = full_path.parent()
                && !parent.exists() && !ctx.dry_run {
                    let _ = std::fs::create_dir_all(parent);
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
            ctx.progress.executed.insert(step.id.clone());
            if let Some(ref output_path) = resolved_output {
                planned_outputs.insert(resolve_in(output_path, &ctx.workspace), step.id.clone());
                planned_with_outputs.insert(step.id.clone());
            }
            for produced in cmd.project_artifacts(&step.id, &options).outputs {
                planned_outputs.insert(resolve_in(&produced, &ctx.workspace), step.id.clone());
                planned_with_outputs.insert(step.id.clone());
            }
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
                    if path.is_file()
                        && let Ok(meta) = std::fs::metadata(&path)
                            && meta.len() > 1_073_741_824 {
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

            // Log any budget resources not declared by this command (§6.4.4a)
            for budget_resource in ctx.governor.budget_resource_names() {
                if !declared_names.contains(budget_resource.as_str()) {
                    ctx.governor.log_ignored(cmd.command_path(), &budget_resource);
                }
            }
        }

        // 7. Execute
        ctx.ui.clear();

        // Update UI context header with dataset/profile/step info.
        // Append the step's description (when set) so the running-step
        // banner conveys what the step actually does, not just its id —
        // e.g. "prepare-vectors — External merge-sort + duplicate
        // detection" is meaningful, "prepare-vectors" alone isn't.
        let step_label = match step.def.description.as_deref() {
            Some(desc) if !desc.is_empty() => format!("{} — {}", step.id, desc),
            _ => step.id.clone(),
        };
        if ctx_label.is_empty() {
            ctx.ui.set_context(format!("[{}/{}] {}{}", step_num, total, step_label, profile_tag));
        } else {
            ctx.ui.set_context(format!("{} [{}/{}] {}{}", ctx_label, step_num, total, step_label, profile_tag));
        }

        // Send a structured step summary for the TUI step panel.
        // Groups options by role (inputs, outputs, config) using OptionDesc metadata.
        {
            let option_descs = cmd.describe_options();
            let mut panel = format!("run: {}\n", step.def.run);

            let mut inputs: Vec<String> = Vec::new();
            let mut outputs: Vec<String> = Vec::new();
            let mut config: Vec<String> = Vec::new();

            for desc in &option_descs {
                if let Some(val) = resolved_opts.get(&desc.name) {
                    let line = format!("  {}: {}", desc.name, val);
                    match desc.role {
                        super::command::OptionRole::Input => inputs.push(line),
                        super::command::OptionRole::Output => outputs.push(line),
                        super::command::OptionRole::Config => config.push(line),
                    }
                }
            }
            // Include any options not in describe_options (extra YAML keys)
            for (k, v) in &resolved_opts {
                if !option_descs.iter().any(|d| d.name == *k) {
                    config.push(format!("  {}: {}", k, v));
                }
            }

            if !inputs.is_empty() {
                panel.push_str("inputs:\n");
                for line in &inputs { panel.push_str(line); panel.push('\n'); }
            }
            if !outputs.is_empty() {
                panel.push_str("outputs:\n");
                for line in &outputs { panel.push_str(line); panel.push('\n'); }
            }
            if !config.is_empty() {
                panel.push_str("config:\n");
                for line in &config { panel.push_str(line); panel.push('\n'); }
            }

            ctx.ui.set_step_yaml(panel);
        }

        if let Some(ref desc) = step.def.description {
            ctx.ui.log(&format!("{} {} — {}", prefix, step.id, desc));
        }
        ctx.ui.clear_step_log();
        ctx.ui.log(&format!("{} {} — running: {}", prefix, step.id, step.def.run));
        dlog.log(&format!("  {} [start] {} — {}",
            chrono::Local::now().format("%H:%M:%S"), step.id, step.def.run));
        dlog.flush();
        ctx.step_id = step.id.clone();
        // A per-profile step is scoped to its profile: every facet it
        // resolves by default comes from that profile, not from
        // `default`. A shared step keeps the run's own selection.
        ctx.profile = step
            .def
            .profiles
            .first()
            .cloned()
            .unwrap_or_else(|| run_profile.clone());
        ctx.governor.set_step_id(&step.id);

        // Resource status — updated by background thread via ui.resource_status()
        // and emergency logging.
        let resource_source = ctx.governor.status_source();
        let resource_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let resource_stop2 = resource_stop.clone();
        let resource_ui = ctx.ui.clone();
        let poll_interval = ctx.status_interval;
        // Capture the step id so the emergency-abort path can name
        // which step blew the budget. Without this the user gets a
        // FATAL with no context about where in the pipeline it died.
        let resource_step_id = step.id.clone();
        let resource_step_num = step_num;
        let resource_step_total = total;
        let resource_step_run = step.def.run.clone();
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
                            // Build the message lines once. They go to:
                            //   1. The TUI log pane (so they appear in
                            //      runlog.jsonl alongside other ticks).
                            //   2. stderr after the TUI tears down (so
                            //      the user actually sees them at the
                            //      shell prompt — without this stderr
                            //      print, the alt-screen swallows them
                            //      and the user gets a silent exit 1).
                            // Include the step identity so the user
                            // knows exactly where in the pipeline we
                            // blew the budget — without it the FATAL
                            // is context-free.
                            let fatal = format!(
                                "FATAL: step [{}/{}] {} ({}) — resource emergency for {:.1}s, RSS {:.0}% over ceiling — aborting to prevent system lockup",
                                resource_step_num, resource_step_total,
                                resource_step_id, resource_step_run,
                                emergency_ticks as f64 * tick_secs,
                                overage,
                            );
                            let status = format!(
                                "  last status: {}", resource_source.status_line(),
                            );
                            resource_ui.log(&fatal);
                            resource_ui.log(&status);
                            // Shut down the TUI to restore the terminal,
                            // then print to stderr so the message is
                            // visible after the alt-screen exits.
                            resource_ui.shutdown();
                            eprintln!();
                            eprintln!("{}", fatal);
                            eprintln!("{}", status);
                            eprintln!("Hint: lower the in-flight count, narrow the workload, or pass `--mem <smaller>` and re-run.");
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

        // 8. Record result with the original (unmunged) options
        ctx.ui.clear();
        let record = step_record_from_result(&result, &resolved_opts, resource_summary, Some(provenance.clone()), &ctx.workspace);
        ctx.progress.record_step(&step.id, record);

        // If this step modified files that were outputs of previous steps
        // (e.g., overlap removal rewrites query_vectors.fvec), update the
        // stored sizes in those earlier step records so freshness checks
        // don't flag them as stale.
        for produced_path in &result.produced {
            let rel = workspace_relative(produced_path, &ctx.workspace);
            if let Ok(meta) = std::fs::metadata(produced_path) {
                ctx.progress.update_output_size(&rel, meta.len());
            }
        }

        match result.status {
            Status::Ok | Status::Warning => {
                ctx.progress.executed.insert(step.id.clone());
                executed_count += 1;
                overall_pb.inc(1);
                total_elapsed += elapsed;
                step_outcomes.push((step.id.clone(), true, result.message.clone()));
                let level = if result.status == Status::Ok { "OK" } else { "WARNING" };
                ctx.ui.log(&format!(
                    "{} {} — {} ({:.1}s): {}",
                    prefix, step.id, level, elapsed.as_secs_f64(), result.message
                ));
                dlog.log(&format!("  {} [{}] {} ({:.1}s): {}",
                    chrono::Local::now().format("%H:%M:%S"),
                    level.to_lowercase(), step.id, elapsed.as_secs_f64(), result.message));
                // Flush all captured step log lines with timestamps
                let ts = chrono::Local::now().format("%H:%M:%S");
                for line in ctx.ui.drain_step_log() {
                    dlog.log(&format!("  {} | {}", ts, line));
                }
                dlog.flush();
            }
            Status::Error => {
                // Leave buffer file in place for diagnostics; clean up on next run
                ctx.ui.log(&format!(
                    "{} {} — ERROR ({:.1}s): {}",
                    prefix, step.id, elapsed.as_secs_f64(), result.message
                ));
                dlog.log(&format!("  {} [error] {} ({:.1}s): {}",
                    chrono::Local::now().format("%H:%M:%S"),
                    step.id, elapsed.as_secs_f64(), result.message));
                // Flush step log lines for error diagnostics
                let ts = chrono::Local::now().format("%H:%M:%S");
                for line in ctx.ui.drain_step_log() {
                    dlog.log(&format!("  {} | {}", ts, line));
                }
                dlog.flush();
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
            if state == ArtifactState::Absent {
                // Only fail post-execution if the output is genuinely missing.
                // The step reported Ok and wrote the file — trust it for
                // structural checks (Partial may just mean verified_count
                // wasn't propagated yet).
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
                        provenance: Some(provenance.clone()),
                    },
                );
                if let Err(e) = ctx.progress.save() {
                    ctx.ui.log(&format!("  warning: failed to save progress: {}", e));
                }
                return Err(msg);
            }
        }

        // 9b. Zero-length produced file check — warn but don't fail.
        // Some commands legitimately produce empty output (e.g., find-zeros
        // with no zero vectors, find-duplicates with no duplicates).
        // The command returned Ok, so it knows whether empty is valid.
        for produced_path in &result.produced {
            if let Ok(meta) = std::fs::metadata(produced_path)
                && meta.len() == 0 {
                    ctx.ui.log(&format!(
                        "{} {} — warning: produced file '{}' is empty (zero bytes)",
                        prefix, step.id, produced_path.display(),
                    ));
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

    overall_pb.finish();

    // Write summary to provenance log
    dlog.log(&format!("summary: {} executed ({:.1}s), {} skipped, {} total",
        executed_count, total_elapsed.as_secs_f64(), skipped_count, total));

    // Write data flow accounting from variables.yaml
    if let Ok(vars) = super::variables::load(&ctx.workspace)
        && !vars.is_empty() {
            dlog.log("\ndata flow:");
            let get = |k: &str| vars.get(k).and_then(|v| v.parse::<u64>().ok());
            if let Some(vc) = get("vector_count") {
                dlog.log(&format!("  source vectors:      {:>12}", vc));
            }
            if let Some(dc) = get("duplicate_count") {
                dlog.log(&format!("  duplicates removed:  {:>12}", dc));
            }
            if let Some(zc) = get("zero_count") {
                dlog.log(&format!("  zero vectors:        {:>12}", zc));
            }
            if let Some(cc) = get("clean_count") {
                dlog.log(&format!("  after cleaning:      {:>12}", cc));
                if let (Some(vc), Some(dc), Some(zc)) = (get("vector_count"), get("duplicate_count"), get("zero_count")) {
                    let expected = vc.saturating_sub(dc).saturating_sub(zc);
                    if expected != cc {
                        dlog.log(&format!("  WARNING: expected {} - {} - {} = {}, got {}",
                            vc, dc, zc, expected, cc));
                    }
                }
            }
            if let Some(qc) = vars.get("query_count").and_then(|v| v.parse::<u64>().ok()) {
                dlog.log(&format!("  query vectors:       {:>12}", qc));
            }
            if let Some(bc) = get("base_count") {
                dlog.log(&format!("  base vectors:        {:>12}", bc));
                if let (Some(cc), Some(qc)) = (get("clean_count"), vars.get("query_count").and_then(|v| v.parse::<u64>().ok())) {
                    let expected = cc.saturating_sub(qc);
                    if expected != bc {
                        dlog.log(&format!("  WARNING: expected {} - {} = {}, got {}",
                            cc, qc, expected, bc));
                    }
                }
            }
        }

    dlog.flush();

    Ok(RunSummary {
        total,
        skipped: skipped_count,
        executed: executed_count,
        total_elapsed,
        step_outcomes,
    })
}


/// Build a `StepRecord` from a `CommandResult`.
/// A produced path as the progress log records it: relative to the
/// workspace when it lies inside it, so a record reads the same from
/// any working directory and `veks check` can find the file; absolute
/// otherwise.
fn workspace_relative(path: &Path, workspace: &Path) -> String {
    match path.strip_prefix(workspace) {
        Ok(rel) if !rel.as_os_str().is_empty() => rel.to_string_lossy().into_owned(),
        _ => path.to_string_lossy().into_owned(),
    }
}

fn step_record_from_result(
    result: &CommandResult,
    resolved_opts: &indexmap::IndexMap<String, String>,
    resource_summary: Option<ResourceSummary>,
    provenance: Option<Address>,
    workspace: &Path,
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
                path: workspace_relative(p, workspace),
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
        provenance,
    }
}

/// A path as an option names it — relative to the workspace or
/// absolute, with any `[window]` stripped — as an absolute path.
fn resolve_in(value: &str, workspace: &Path) -> PathBuf {
    let clean = PathBuf::from(value.split('[').next().unwrap_or(value));
    if clean.is_absolute() { clean } else { workspace.join(clean) }
}
