// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: display compute environment capabilities.
//!
//! Reports system information relevant to vector computation: CPU features,
//! available memory, thread count, and SIMD capabilities.
//!
//! Equivalent to the Java `CMD_info_compute` command (adapted for Rust —
//! reports Rust-specific capabilities instead of JVM/Panama info).

use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: display compute environment info.
pub struct InfoComputeOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(InfoComputeOp)
}

impl CommandOp for InfoComputeOp {
    fn command_path(&self) -> &str {
        "analyze compute-info"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Display compute capability information".into(),
            body: format!(
                r#"# analyze compute-info

Display compute capability information.

## Description

Reports system information relevant to vector computation: CPU
architecture, operating system, available CPU count, Rust compiler
target, SIMD instruction set support, and available memory. On Linux,
memory information is read from `/proc/meminfo`.

## How It Works

The command queries compile-time and runtime system information. CPU
count comes from `std::thread::available_parallelism`. SIMD feature
detection uses Rust `cfg(target_feature)` attributes compiled into
the binary, so the reported features reflect what was available at
compile time (which may differ from runtime capabilities if
cross-compiling). Detected features include AVX-512F, AVX2, AVX,
SSE4.2, SSE4.1, SSE2, and NEON. The `short` option produces a
single-line summary suitable for logging.

## Data Preparation Role

`info compute` helps you understand the hardware acceleration
available for vector distance computations and other SIMD-intensive
pipeline operations. The SIMD features directly affect the performance
of KNN search, vector normalization, and distance matrix computation.
If key features like AVX2 are missing, the output includes a
recommendation to recompile with `RUSTFLAGS="-C target-cpu=native"`
for optimal performance. This command is typically run at the
beginning of a pipeline to log the execution environment.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let short = options.get("short") == Some("true");

        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let arch = std::env::consts::ARCH;
        let os = std::env::consts::OS;

        // The level the vector kernels dispatch to on this CPU, and every
        // level it can run — detected at runtime, not read from the
        // compile-time target baseline (which is SSE2 for any portable
        // x86-64 build, whatever the CPU).
        let dispatched = veks_simd::SimdLevel::detected().name();
        let supported: Vec<&str> = veks_simd::SimdLevel::supported().iter().map(|l| l.name()).collect();

        if short {
            ctx.ui.log(&format!(
                "{} {} | {} CPUs | SIMD: {}",
                os, arch, cpus, dispatched,
            ));
        } else {
            ctx.ui.log("Compute Environment");
            ctx.ui.log(&format!("  OS:           {} {}", os, arch));
            ctx.ui.log(&format!("  CPUs:         {}", cpus));
            ctx.ui.log(&format!("  Rust version: {}", env!("CARGO_PKG_VERSION")));
            ctx.ui.log(&format!(
                "  Target:       {}",
                std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
            ));

            ctx.ui.log(&format!("  SIMD:         {} (kernels dispatch here)", dispatched));
            ctx.ui.log(&format!("  SIMD levels:  {} (all this CPU can run)", supported.join(", ")));

            // Memory info (best effort)
            #[cfg(target_os = "linux")]
            {
                if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                    for line in meminfo.lines().take(3) {
                        ctx.ui.log(&format!("  {}", line.trim()));
                    }
                }
            }

            ctx.ui.log("");
            ctx.ui.log("  Distance kernels: veks-simd, selected at runtime — no target-cpu");
            ctx.ui.log("  build flags are needed to reach the dispatched level.");
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "compute env: {} {} ({} CPUs, SIMD: {})",
                os, arch, cpus, dispatched,
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![OptionDesc {
            name: "short".to_string(),
            type_name: "bool".to_string(),
            required: false,
            default: Some("false".to_string()),
            description: "Show one-line summary only".to_string(),
            extended_description: None,
                role: OptionRole::Config,
    }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            attributes: Vec::new(),
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    #[test]
    fn test_info_compute() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = test_ctx(tmp.path());

        let opts = Options::new();
        let mut op = InfoComputeOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_info_compute_short() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = test_ctx(tmp.path());

        let mut opts = Options::new();
        opts.set("short", "true");
        let mut op = InfoComputeOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn reports_the_dispatched_level() {
        // SK-16: the reported level is the one the kernels run at,
        // which on any x86-64 CPU from the last decade is above the
        // SSE2 compile-time baseline a portable build carries.
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = test_ctx(tmp.path());
        let mut opts = Options::new();
        opts.set("short", "true");
        let result = InfoComputeOp.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        let level = veks_simd::SimdLevel::detected().name();
        assert!(result.message.ends_with(&format!("SIMD: {level})")), "{}", result.message);
    }
}
