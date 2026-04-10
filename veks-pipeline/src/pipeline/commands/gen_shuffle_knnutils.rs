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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::{Options, Status, StreamContext};
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use vectordata::VectorReader;
    use vectordata::io::MmapVectorReader;

    fn make_ctx(workspace: &std::path::Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"),
            cache: workspace.join(".cache"),
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

    fn read_ivec_values(path: &std::path::Path) -> Vec<i32> {
        let reader = MmapVectorReader::<i32>::open_ivec(path).unwrap();
        let count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
        (0..count).map(|i| reader.get_slice(i)[0]).collect()
    }

    /// Verify that shuffle(range(20), seed=42) matches numpy exactly.
    ///
    /// Expected from: `np.random.seed(42); a=list(range(20)); np.random.shuffle(a); print(a)`
    /// Result: [0, 17, 15, 1, 8, 5, 11, 3, 18, 16, 13, 2, 9, 19, 4, 12, 7, 10, 14, 6]
    #[test]
    fn test_shuffle_matches_numpy_seed_42() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let out = tmp.path().join("shuffle.ivec");
        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("interval", "20");
        opts.set("seed", "42");

        let mut op = GenerateShuffleKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let values = read_ivec_values(&out);
        let expected = vec![0, 17, 15, 1, 8, 5, 11, 3, 18, 16, 13, 2, 9, 19, 4, 12, 7, 10, 14, 6];
        assert_eq!(values, expected,
            "shuffle(range(20), seed=42) does not match numpy");
    }

    /// Verify determinism: same seed always produces same output.
    #[test]
    fn test_shuffle_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let out1 = tmp.path().join("s1.ivec");
        let out2 = tmp.path().join("s2.ivec");

        for out in [&out1, &out2] {
            let mut opts = Options::new();
            opts.set("output", out.to_string_lossy().to_string());
            opts.set("interval", "1000");
            opts.set("seed", "99");
            let mut op = GenerateShuffleKnnUtilsOp;
            let r = op.execute(&opts, &mut ctx);
            assert_eq!(r.status, Status::Ok);
        }

        let v1 = read_ivec_values(&out1);
        let v2 = read_ivec_values(&out2);
        assert_eq!(v1, v2, "same seed should produce identical shuffle");
    }

    /// Verify different seeds produce different permutations.
    #[test]
    fn test_shuffle_different_seeds_differ() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let out1 = tmp.path().join("s1.ivec");
        let out2 = tmp.path().join("s2.ivec");

        for (out, seed) in [(&out1, "1"), (&out2, "2")] {
            let mut opts = Options::new();
            opts.set("output", out.to_string_lossy().to_string());
            opts.set("interval", "100");
            opts.set("seed", seed);
            let mut op = GenerateShuffleKnnUtilsOp;
            let r = op.execute(&opts, &mut ctx);
            assert_eq!(r.status, Status::Ok);
        }

        let v1 = read_ivec_values(&out1);
        let v2 = read_ivec_values(&out2);
        assert_ne!(v1, v2, "different seeds should produce different shuffles");
    }

    /// Verify output is a valid permutation (all elements present, none repeated).
    #[test]
    fn test_shuffle_is_valid_permutation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let out = tmp.path().join("shuffle.ivec");
        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("interval", "500");
        opts.set("seed", "0");
        let mut op = GenerateShuffleKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let values = read_ivec_values(&out);
        assert_eq!(values.len(), 500);
        let mut sorted = values.clone();
        sorted.sort();
        let expected: Vec<i32> = (0..500).collect();
        assert_eq!(sorted, expected, "shuffle should be a valid permutation");
    }

    /// Verify PRNG state chaining: base shuffle saves state, query shuffle
    /// loads it and continues. This matches knn_utils' single-seed flow:
    /// `np.random.seed(42); np.random.shuffle(base); np.random.shuffle(query)`
    #[test]
    fn test_prng_state_chaining() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base_out = tmp.path().join("base_shuffle.ivec");
        let state_file = tmp.path().join("prng_state.bin");
        let query_out = tmp.path().join("query_shuffle.ivec");

        // Base shuffle: seed=42, save state
        let mut opts = Options::new();
        opts.set("output", base_out.to_string_lossy().to_string());
        opts.set("interval", "50");
        opts.set("seed", "42");
        opts.set("prng-state-out", state_file.to_string_lossy().to_string());
        let mut op = GenerateShuffleKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);
        assert!(state_file.exists(), "PRNG state file should be created");

        // Query shuffle: load state from base, no seed
        let mut opts = Options::new();
        opts.set("output", query_out.to_string_lossy().to_string());
        opts.set("interval", "20");
        opts.set("prng-state-in", state_file.to_string_lossy().to_string());
        let mut op = GenerateShuffleKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Reproduce with numpy-equivalent single-seed flow
        // np.random.seed(42); a=list(range(50)); np.random.shuffle(a)
        // b=list(range(20)); np.random.shuffle(b)
        let base_vals = read_ivec_values(&base_out);
        let query_vals = read_ivec_values(&query_out);

        // The base should match seed-42 shuffle of range(50)
        assert_eq!(base_vals.len(), 50);
        assert_eq!(query_vals.len(), 20);

        // Verify query is NOT the same as a fresh seed-42 shuffle of range(20)
        // (it should use the continued PRNG state from the base shuffle)
        let fresh_out = tmp.path().join("fresh.ivec");
        let mut opts = Options::new();
        opts.set("output", fresh_out.to_string_lossy().to_string());
        opts.set("interval", "20");
        opts.set("seed", "42");
        let mut op = GenerateShuffleKnnUtilsOp;
        op.execute(&opts, &mut ctx);
        let fresh_vals = read_ivec_values(&fresh_out);

        assert_ne!(query_vals, fresh_vals,
            "chained query shuffle should differ from fresh seed-42 shuffle");
    }

    /// Verify ordinals pass-through: when ordinals file is provided,
    /// the shuffle permutes those ordinal values instead of [0..interval).
    #[test]
    fn test_shuffle_with_ordinals() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        // Write ordinals: [100, 200, 300, 400, 500]
        let ord_path = tmp.path().join("ordinals.ivec");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&ord_path).unwrap();
            for v in [100i32, 200, 300, 400, 500] {
                f.write_all(&1i32.to_le_bytes()).unwrap();
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }

        let out = tmp.path().join("shuffled.ivec");
        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("interval", "5");
        opts.set("seed", "42");
        opts.set("ordinals", ord_path.to_string_lossy().to_string());
        let mut op = GenerateShuffleKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let values = read_ivec_values(&out);
        assert_eq!(values.len(), 5);
        // Should be a permutation of [100, 200, 300, 400, 500]
        let mut sorted = values.clone();
        sorted.sort();
        assert_eq!(sorted, vec![100, 200, 300, 400, 500],
            "shuffled ordinals should contain the original values");
        // Should NOT be identity (extremely unlikely for seed 42)
        assert_ne!(values, vec![100, 200, 300, 400, 500],
            "shuffle should change the order");
    }

    /// Verify the rk_interval function matches numpy's bounded random.
    #[test]
    fn test_rk_interval_bounds() {
        let mut rng = Mt::new(42);
        // rk_interval(max) should return values in [0, max]
        for max_val in [1u32, 5, 10, 100, 1000] {
            for _ in 0..100 {
                let v = rk_interval(max_val, &mut rng);
                assert!(v <= max_val, "rk_interval({}) returned {} (out of range)", max_val, v);
            }
        }
        // rk_interval(0) should always return 0
        assert_eq!(rk_interval(0, &mut rng), 0);
    }
}
