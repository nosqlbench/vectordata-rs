// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: normalize vectors (knn\_utils personality).
//!
//! Replicates knn\_utils normalization exactly by calling numpy directly:
//!
//! ```python
//! norms = np.linalg.norm(arr, axis=1, keepdims=True)
//! norms[norms == 0] = 1
//! arr = arr / norms
//! ```
//!
//! numpy's internal pairwise reduction algorithm (compiled C code with
//! f64 accumulators and an unspecified block structure) cannot be reliably
//! replicated in Rust. Calling numpy via subprocess guarantees
//! byte-identical floating-point results with knn\_utils.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};

fn error_result(message: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: message.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

/// Pipeline command: L2-normalize vectors using numpy (knn\_utils compatible).
pub struct TransformNormalizeKnnUtilsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(TransformNormalizeKnnUtilsOp)
}

impl CommandOp for TransformNormalizeKnnUtilsOp {
    fn command_path(&self) -> &str {
        "transform normalize-knnutils"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "L2-normalize vectors using numpy (knn_utils compatible)".into(),
            body: format!(r#"# transform normalize-knnutils

L2-normalize all vectors in an fvec file by calling numpy directly,
guaranteeing byte-identical results with knn\_utils:

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr = arr / norms

Requires `python3` with `numpy` installed.

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        ctx.ui.log(&format!(
            "  normalize-knnutils: {} -> {} (via numpy)",
            source_path.display(), output_path.display(),
        ));

        let py_script = format!(
            r#"
import numpy as np, struct
data = np.fromfile('{src}', dtype=np.float32)
dim = struct.unpack('<I', data[:1].tobytes())[0]
arr = data.reshape(-1, dim + 1)[:, 1:]
norms = np.linalg.norm(arr, axis=1, keepdims=True)
zero_count = int((norms.flatten() == 0).sum())
norms[norms == 0] = 1
normed = arr / norms
n, d = normed.shape
d_repr = struct.unpack('<f', np.uint32(d))[0]
formatted = np.concatenate((np.full((n, 1), d_repr, dtype=np.float32), normed.astype(np.float32)), axis=1)
formatted.tofile('{dst}')
print(f'{{n}} {{zero_count}}')
"#,
            src = source_path.display(),
            dst = output_path.display(),
        );

        let result = std::process::Command::new("python3")
            .arg("-c")
            .arg(&py_script)
            .output();

        match result {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return error_result(format!("numpy normalization failed: {}", stderr), start);
                }
                let stdout = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = stdout.trim().split_whitespace().collect();
                let count: usize = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                let zero_count: u64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

                ctx.ui.log(&format!(
                    "  normalized {} vectors ({} zeros passed through)",
                    count, zero_count,
                ));

                let var_name = format!("verified_count:{}",
                    output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
                let _ = crate::pipeline::variables::set_and_save(
                    &ctx.workspace, &var_name, &count.to_string());
                ctx.defaults.insert(var_name, count.to_string());

                CommandResult {
                    status: Status::Ok,
                    message: format!(
                        "normalized {} vectors ({} zeros) to {}",
                        count, zero_count, output_path.display(),
                    ),
                    produced: vec![output_path],
                    elapsed: start.elapsed(),
                }
            }
            Err(e) => error_result(format!("failed to run python3: {}", e), start),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source fvec file".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output normalized fvec file".to_string(),
                role: OptionRole::Output,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output"],
        )
    }
}
