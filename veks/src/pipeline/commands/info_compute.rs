// Copyright (c) DataStax, Inc.
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
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

/// Pipeline command: display compute environment info.
pub struct InfoComputeOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(InfoComputeOp)
}

impl CommandOp for InfoComputeOp {
    fn command_path(&self) -> &str {
        "info compute"
    }

    fn execute(&mut self, options: &Options, _ctx: &mut StreamContext) -> CommandResult {
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
            eprintln!(
                "{} {} | {} CPUs | SIMD: {}",
                os,
                arch,
                cpus,
                if simd_features.is_empty() {
                    "none detected".to_string()
                } else {
                    simd_features.join(", ")
                }
            );
        } else {
            eprintln!("Compute Environment");
            eprintln!("  OS:           {} {}", os, arch);
            eprintln!("  CPUs:         {}", cpus);
            eprintln!("  Rust version: {}", env!("CARGO_PKG_VERSION"));
            eprintln!(
                "  Target:       {}",
                std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
            );

            if simd_features.is_empty() {
                eprintln!("  SIMD:         none detected at compile time");
            } else {
                eprintln!("  SIMD features:");
                for feat in &simd_features {
                    eprintln!("    - {}", feat);
                }
            }

            // Memory info (best effort)
            #[cfg(target_os = "linux")]
            {
                if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                    for line in meminfo.lines().take(3) {
                        eprintln!("  {}", line.trim());
                    }
                }
            }

            eprintln!();
            eprintln!("  Vectorized distance computation: Rust auto-vectorization");
            eprintln!("  For best performance, compile with:");
            eprintln!("    RUSTFLAGS=\"-C target-cpu=native\" cargo build --release");
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
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
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
