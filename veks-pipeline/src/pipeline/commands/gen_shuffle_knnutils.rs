// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate shuffle permutation (knn\_utils personality).
//!
//! Replicates the exact shuffle algorithm from knn\_utils:
//!
//! ```python
//! np.random.seed(42)
//! np.random.shuffle(base)
//! ```
//!
//! This uses:
//! - **MT19937** (Mersenne Twister) PRNG, matching numpy's legacy
//!   `RandomState` seeded with a 32-bit integer
//! - **rk\_interval** bounded random generation (bit-mask rejection
//!   sampling from numpy's `randomkit.c`)
//! - **Fisher-Yates** shuffle iterating `i` from `n-1` down to `1`
//!
//! The native veks shuffle uses Xoshiro256++ with a different bounded
//! random method, producing a different permutation for the same seed.
//! This command exists so that `--personality knn_utils` pipelines
//! produce the same vector ordering as knn\_utils.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand_mt::Mt;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::atomic_write::AtomicWriter;
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

/// Replicate numpy's `rk_interval` from `randomkit.c`.
///
/// Returns a uniform random integer in `[0, max]` inclusive, using
/// bit-mask rejection sampling identical to numpy's legacy RandomState.
fn rk_interval(max: u32, rng: &mut Mt) -> u32 {
    if max == 0 {
        return 0;
    }
    // Compute smallest bit mask >= max
    let mut mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;

    // Rejection sampling
    loop {
        let value = rng.next_u32() & mask;
        if value <= max {
            return value;
        }
    }
}

/// Pipeline command: numpy-compatible shuffle permutation.
pub struct GenerateShuffleKnnUtilsOp;

/// Creates a boxed `GenerateShuffleKnnUtilsOp` for command registration.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateShuffleKnnUtilsOp)
}

impl CommandOp for GenerateShuffleKnnUtilsOp {
    fn command_path(&self) -> &str {
        "generate shuffle-knnutils"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate a shuffle permutation using numpy-compatible MT19937".into(),
            body: format!(r#"# generate shuffle-knnutils

Generate a random ordinal shuffle mapping using the same PRNG and
algorithm as numpy's `np.random.shuffle()`.

## Algorithm

Uses MT19937 (Mersenne Twister) with numpy's `rk_interval` bit-mask
rejection sampling, producing a permutation identical to:

```python
np.random.seed(seed)
arr = list(range(interval))
np.random.shuffle(arr)
```

The native `generate shuffle` command uses Xoshiro256++ instead, which
produces a different permutation for the same seed. Use this command
when the `--personality knn_utils` pipeline needs to match knn\_utils
vector ordering exactly.

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let interval: usize = match options.require("interval") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid interval: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let seed: u32 = match options.get("seed") {
            Some(s) => s.parse().unwrap_or(42),
            None => 42,
        };

        let ordinals_path = options.get("ordinals").map(|s| resolve_path(s, &ctx.workspace));
        let prng_state_in = options.get("prng-state-in").map(|s| resolve_path(s, &ctx.workspace));
        let prng_state_out = options.get("prng-state-out").map(|s| resolve_path(s, &ctx.workspace));
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("create directory: {}", e), start);
                }
            }
        }

        // Build value sequence to shuffle
        let mut values: Vec<i32> = if let Some(ref ord_path) = ordinals_path {
            ctx.ui.log(&format!("  loading ordinals from {}", ord_path.display()));
            let data = match std::fs::read(ord_path) {
                Ok(d) => d,
                Err(e) => return error_result(format!("read ordinals: {}", e), start),
            };
            let mut ords = Vec::with_capacity(interval);
            let mut pos = 0;
            while pos + 8 <= data.len() && ords.len() < interval {
                pos += 4; // skip dim header
                let val = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                ords.push(val);
                pos += 4;
            }
            if ords.len() != interval {
                ctx.ui.log(&format!("  warning: ordinals has {} records, expected {}",
                    ords.len(), interval));
            }
            ords
        } else {
            (0..interval as i32).collect()
        };

        ctx.ui.log(&format!(
            "  shuffle-knnutils: {} elements, seed={}, PRNG=MT19937 (numpy-compatible)",
            interval, seed,
        ));

        // Fisher-Yates shuffle with MT19937 + rk_interval
        // Matches: np.random.seed(seed); np.random.shuffle(arr)
        //
        // When prng-state-in is provided, continue from a previously saved
        // PRNG state. This is needed when knn_utils shuffles base and query
        // with a single np.random.seed(42) call — the query shuffle must
        // continue from the PRNG state left after the base shuffle.
        let pb = ctx.ui.bar(interval as u64, "shuffle");
        let mut rng = if let Some(ref state_path) = prng_state_in {
            let data = match std::fs::read(state_path) {
                Ok(d) => d,
                Err(e) => return error_result(format!("read prng state: {}", e), start),
            };
            ctx.ui.log(&format!("  loaded PRNG state from {}", state_path.display()));
            // Deserialize: binary format is the Mt struct bytes
            if data.len() != std::mem::size_of::<Mt>() {
                return error_result(format!(
                    "PRNG state file has wrong size: {} (expected {})",
                    data.len(), std::mem::size_of::<Mt>()
                ), start);
            }
            unsafe { std::ptr::read(data.as_ptr() as *const Mt) }
        } else {
            Mt::new(seed)
        };
        let batch = 10_000.max(interval / 1000);
        for i in (1..interval).rev() {
            let j = rk_interval(i as u32, &mut rng) as usize;
            values.swap(i, j);
            if i % batch == 0 {
                pb.set_position((interval - i) as u64);
            }
        }
        pb.set_position(interval as u64);
        pb.finish();

        // Save PRNG state if requested (for chaining base → query shuffles)
        if let Some(ref state_path) = prng_state_out {
            if let Some(parent) = state_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let state_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &rng as *const Mt as *const u8,
                    std::mem::size_of::<Mt>(),
                )
            };
            match std::fs::write(state_path, state_bytes) {
                Ok(()) => ctx.ui.log(&format!("  saved PRNG state to {}", state_path.display())),
                Err(e) => ctx.ui.log(&format!("  warning: failed to save PRNG state: {}", e)),
            }
        }

        // Write as 1-dimensional ivec records
        let write_pb = ctx.ui.bar(values.len() as u64, "write");
        let mut writer = match AtomicWriter::new(&output_path) {
            Ok(w) => w,
            Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
        };
        let dim: i32 = 1;
        for (i, &value) in values.iter().enumerate() {
            if let Err(e) = writer.write_all(&dim.to_le_bytes()) {
                return error_result(format!("write: {}", e), start);
            }
            if let Err(e) = writer.write_all(&value.to_le_bytes()) {
                return error_result(format!("write: {}", e), start);
            }
            if (i + 1) % batch == 0 {
                write_pb.inc(batch as u64);
            }
        }
        write_pb.set_position(values.len() as u64);
        if let Err(e) = writer.finish() {
            return error_result(format!("finalize: {}", e), start);
        }
        write_pb.finish();

        // Write verified count
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &interval.to_string());
        ctx.defaults.insert(var_name, interval.to_string());

        CommandResult {
            status: Status::Ok,
            message: format!(
                "generated numpy-compatible shuffle of {} elements (seed={}) to {}",
                interval, seed, output_path.display(),
            ),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output ivec file path".to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "interval".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of elements to shuffle (0 to interval-1)".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("42".to_string()),
                description: "Random seed (numpy-compatible MT19937, default 42)".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "ordinals".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Ordinals ivec to shuffle (if omitted, shuffles [0, interval))".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "prng-state-in".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Load PRNG state from file (continue from previous shuffle)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "prng-state-out".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Save PRNG state after shuffle (for chaining to next shuffle)".to_string(),
                role: OptionRole::Output,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["ordinals"],
            &["output"],
        )
    }
}
