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

        let short = options.get("short").map_or(false, |s| s == "true");

        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let arch = std::env::consts::ARCH;
        let os = std::env::consts::OS;

        // Detect SIMD capabilities via cfg attributes
        let simd_features = detect_simd_features();

        if short {
            ctx.ui.log(&format!(
                "{} {} | {} CPUs | SIMD: {}",
                os,
                arch,
                cpus,
                if simd_features.is_empty() {
                    "none detected".to_string()
                } else {
                    simd_features.join(", ")
                }
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

            if simd_features.is_empty() {
                ctx.ui.log("  SIMD:         none detected at compile time");
            } else {
                ctx.ui.log("  SIMD features:");
                for feat in &simd_features {
                    ctx.ui.log(&format!("    - {}", feat));
                }
            }

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
            ctx.ui.log("  Vectorized distance computation: Rust auto-vectorization");
            ctx.ui.log("  For best performance, compile with:");
            ctx.ui.log("    RUSTFLAGS=\"-C target-cpu=native\" cargo build --release");
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "compute env: {} {} ({} CPUs, SIMD: {})",
                os,
                arch,
                cpus,
                if simd_features.is_empty() {
                    "none"
                } else {
                    "available"
                }
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
                role: OptionRole::Config,
    }]
    }
}

/// Detect available SIMD features at compile time.
fn detect_simd_features() -> Vec<&'static str> {
    let mut features = Vec::new();

    #[cfg(target_feature = "avx512f")]
    features.push("AVX-512F");

    #[cfg(target_feature = "avx2")]
    features.push("AVX2");

    #[cfg(target_feature = "avx")]
    features.push("AVX");

    #[cfg(target_feature = "sse4.2")]
    features.push("SSE4.2");

    #[cfg(target_feature = "sse4.1")]
    features.push("SSE4.1");

    #[cfg(target_feature = "sse2")]
    features.push("SSE2");

    #[cfg(target_feature = "neon")]
    features.push("NEON");

    features
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
    fn test_detect_simd() {
        // Just verify it doesn't panic
        let features = detect_simd_features();
        // On x86_64, SSE2 should always be present
        #[cfg(target_arch = "x86_64")]
        assert!(features.contains(&"SSE2"));
    }
}
