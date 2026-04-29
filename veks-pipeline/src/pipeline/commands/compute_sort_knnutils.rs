// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: sort and deduplicate vectors (knn\_utils personality).
//!
//! Replicates the sort+dedup algorithm from knn\_utils' `fvecs_deduplicator.py`:
//!
//! - **Full lexicographic sort** on all vector components (matching Python's
//!   tuple comparison), not the prefix-based approximate sort used by the
//!   native `compute sort`.
//! - **Consecutive duplicate removal**: after sorting, adjacent identical
//!   vectors are collapsed to one, keeping the first occurrence.
//!
//! The native `compute sort` uses a prefix of N leading components for the
//! primary sort and only reads full vectors when prefixes collide. This is
//! faster but can produce a different sorted order when two vectors share
//! the same N-component prefix but differ in later components. For the
//! knn\_utils personality, exact sort-order parity is required so that
//! downstream shuffle produces the same permutation.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

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

/// Pipeline command: full-lexicographic sort + dedup (knn\_utils compatible).
pub struct ComputeSortKnnUtilsOp;

/// Creates a boxed `ComputeSortKnnUtilsOp` for command registration.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeSortKnnUtilsOp)
}

impl CommandOp for ComputeSortKnnUtilsOp {
    fn command_path(&self) -> &str {
        "compute sort-knnutils"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_COMPUTE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Sort and deduplicate vectors using knn_utils full lexicographic order".into(),
            body: format!(r#"# compute sort-knnutils

Sort and deduplicate vectors using full lexicographic comparison on all
components, matching knn\_utils `fvecs_deduplicator.py`.

Unlike the native `compute sort` which uses a prefix-based sort, this
command compares all vector components during sorting. This ensures the
sorted order is identical to Python's `sorted(vectors, key=lambda v: tuple(v))`,
which is necessary for the knn\_utils personality pipeline to produce
byte-identical shuffle results.

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
        let duplicates_str = match options.get("duplicates") {
            Some(s) => s.to_string(),
            None => {
                let source_path = resolve_path(source_str, &ctx.workspace);
                let stem = source_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("vectors");
                let dir = source_path.parent().unwrap_or(Path::new("."));
                dir.join(format!("{}_duplicates.ivec", stem)).to_string_lossy().to_string()
            }
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let duplicates_path = resolve_path(&duplicates_str, &ctx.workspace);

        // Create output directories
        for p in [&output_path, &duplicates_path] {
            if let Some(parent) = p.parent() {
                if !parent.exists() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
        }

        // Open source vectors
        let reader = match MmapVectorReader::<f32>::open_fvec(&source_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
        };
        let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);

        if count == 0 {
            return error_result("empty source file", start);
        }

        ctx.ui.log(&format!(
            "sort-knnutils: {} vectors, dim={}, full lexicographic sort + dedup",
            count, dim,
        ));

        // Phase 1: Build ordinal array and sort by full vector content.
        // This matches knn_utils' `chunk.sort(key=lambda x: x[0])` where
        // x[0] is the float tuple — Python tuple comparison is lexicographic
        // on all components.
        let sort_start = Instant::now();
        let sort_pb = ctx.ui.spinner(&format!("sorting {} vectors (full lexicographic)", count));

        let mut ordinals: Vec<u32> = (0..count as u32).collect();

        // Parallel sort using rayon
        use rayon::prelude::*;
        let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        let sort_fn = |ords: &mut Vec<u32>| {
            ords.par_sort_unstable_by(|&a, &b| {
                let va = reader.get_slice(a as usize);
                let vb = reader.get_slice(b as usize);
                // Full lexicographic comparison on all components
                for i in 0..dim {
                    match va[i].partial_cmp(&vb[i]) {
                        Some(std::cmp::Ordering::Equal) => continue,
                        Some(ord) => return ord,
                        None => {
                            // Handle NaN: treat NaN as greater than all values
                            let a_nan = va[i].is_nan();
                            let b_nan = vb[i].is_nan();
                            if a_nan && b_nan { continue; }
                            if a_nan { return std::cmp::Ordering::Greater; }
                            return std::cmp::Ordering::Less;
                        }
                    }
                }
                // Fully equal vectors: stable tiebreak by ordinal
                a.cmp(&b)
            });
        };

        if let Some(ref p) = pool {
            p.install(|| sort_fn(&mut ordinals));
        } else {
            sort_fn(&mut ordinals);
        }
        sort_pb.finish();

        let sort_secs = sort_start.elapsed().as_secs_f64();
        ctx.ui.log(&format!("  sorted in {:.1}s ({} threads)", sort_secs, threads));

        // Phase 2: Deduplicate by scanning consecutive pairs.
        // Matches knn_utils: keep first occurrence, mark rest as duplicates.
        let dedup_start = Instant::now();
        let dedup_pb = ctx.ui.bar_with_unit(count as u64, "dedup", "vectors");

        let mut unique_ordinals: Vec<u32> = Vec::with_capacity(count);
        let mut dup_ordinals: Vec<u32> = Vec::new();

        unique_ordinals.push(ordinals[0]);
        dedup_pb.inc(1);

        for i in 1..ordinals.len() {
            let prev = reader.get_slice(ordinals[i - 1] as usize);
            let curr = reader.get_slice(ordinals[i] as usize);

            if prev == curr {
                dup_ordinals.push(ordinals[i]);
            } else {
                unique_ordinals.push(ordinals[i]);
            }

            if (i + 1) % 100_000 == 0 {
                dedup_pb.set_position((i + 1) as u64);
            }
        }
        dedup_pb.finish();

        let dedup_secs = dedup_start.elapsed().as_secs_f64();
        let unique_count = unique_ordinals.len();
        let dup_count = dup_ordinals.len();
        ctx.ui.log(&format!(
            "  dedup: {:.1}s, {} unique, {} duplicates ({:.2}%)",
            dedup_secs, unique_count, dup_count,
            if count > 0 { 100.0 * dup_count as f64 / count as f64 } else { 0.0 },
        ));

        // Phase 3: Write output ivec (sorted unique ordinals)
        let write_start = Instant::now();
        let write_pb = ctx.ui.bar(unique_count as u64, "write sorted");

        let mut writer = match AtomicWriter::new(&output_path) {
            Ok(w) => w,
            Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
        };
        let dim_1: i32 = 1;
        let batch = 10_000.max(unique_count / 1000);
        for (i, &ord) in unique_ordinals.iter().enumerate() {
            writer.write_all(&dim_1.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            writer.write_all(&(ord as i32).to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            if (i + 1) % batch == 0 {
                write_pb.inc(batch as u64);
            }
        }
        write_pb.set_position(unique_count as u64);
        writer.finish().map_err(|e| e.to_string()).unwrap();
        write_pb.finish();

        // Write duplicates ivec
        let dup_pb = ctx.ui.bar(dup_count as u64, "write dups");
        let mut dup_writer = match AtomicWriter::new(&duplicates_path) {
            Ok(w) => w,
            Err(e) => return error_result(format!("create {}: {}", duplicates_path.display(), e), start),
        };
        for (i, &ord) in dup_ordinals.iter().enumerate() {
            dup_writer.write_all(&dim_1.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            dup_writer.write_all(&(ord as i32).to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            if (i + 1) % batch == 0 {
                dup_pb.inc(batch as u64);
            }
        }
        dup_pb.set_position(dup_count as u64);
        dup_writer.finish().map_err(|e| e.to_string()).unwrap();
        dup_pb.finish();

        let write_secs = write_start.elapsed().as_secs_f64();
        ctx.ui.log(&format!("  wrote in {:.1}s", write_secs));

        // Write verified counts
        let set_var = |ctx: &mut StreamContext, name: &str, value: &str| {
            let _ = crate::pipeline::variables::set_and_save(&ctx.workspace, name, value);
            ctx.defaults.insert(name.to_string(), value.to_string());
        };
        let out_var = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        set_var(ctx, &out_var, &unique_count.to_string());
        let dup_var = format!("verified_count:{}",
            duplicates_path.file_name().and_then(|n| n.to_str()).unwrap_or("dups"));
        set_var(ctx, &dup_var, &dup_count.to_string());

        let total_secs = start.elapsed().as_secs_f64();
        let produced = vec![output_path, duplicates_path];

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} vectors -> {} unique, {} dup ({:.2}%) elided, {:.1}s (full lexicographic sort)",
                count, unique_count, dup_count,
                if count > 0 { 100.0 * dup_count as f64 / count as f64 } else { 0.0 },
                total_secs,
            ),
            produced,
            elapsed: start.elapsed(),
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
                description: "Output sorted unique ordinals (ivec)".to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "duplicates".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output duplicate ordinals (ivec)".to_string(),
                role: OptionRole::Output,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output", "duplicates"],
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
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 2,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
        }
    }

    fn write_fvec(path: &std::path::Path, vectors: &[Vec<f32>]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    fn read_ivec_values(path: &std::path::Path) -> Vec<i32> {
        let reader = MmapVectorReader::<i32>::open_ivec(path).unwrap();
        let count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
        (0..count).map(|i| reader.get_slice(i)[0]).collect()
    }

    /// Basic sort: verifies vectors are output in lexicographic order.
    #[test]
    fn test_sort_lexicographic_order() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let vectors = vec![
            vec![0.5, 0.1, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.1, 0.2, 0.0],
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.0, 0.1],
        ];
        let fvec = tmp.path().join("source.fvec");
        write_fvec(&fvec, &vectors);

        let out = tmp.path().join("sorted.ivec");
        let dups = tmp.path().join("dups.ivec");
        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("duplicates", dups.to_string_lossy().to_string());

        let mut op = ComputeSortKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let ordinals = read_ivec_values(&out);
        // Expected lexicographic order:
        // [0.0, 0.0, 0.1] < [0.1, 0.2, 0.0] < [0.1, 0.9, 0.0] < [0.5, 0.0, 0.0] < [0.5, 0.1, 0.0]
        // Indices: 4, 2, 1, 3, 0
        assert_eq!(ordinals, vec![4, 2, 1, 3, 0],
            "ordinals should be in lexicographic order of vectors");

        let dup_ordinals = read_ivec_values(&dups);
        assert_eq!(dup_ordinals.len(), 0, "no duplicates expected");
    }

    /// Dedup: exact duplicate vectors are removed.
    #[test]
    fn test_sort_removes_duplicates() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let vectors = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![1.0, 2.0],  // dup of 0
            vec![5.0, 6.0],
            vec![3.0, 4.0],  // dup of 1
        ];
        let fvec = tmp.path().join("source.fvec");
        write_fvec(&fvec, &vectors);

        let out = tmp.path().join("sorted.ivec");
        let dups = tmp.path().join("dups.ivec");
        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("duplicates", dups.to_string_lossy().to_string());

        let mut op = ComputeSortKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let ordinals = read_ivec_values(&out);
        assert_eq!(ordinals.len(), 3, "should have 3 unique vectors");

        let dup_ordinals = read_ivec_values(&dups);
        assert_eq!(dup_ordinals.len(), 2, "should have 2 duplicate vectors");

        // Verify the sorted unique ordinals reference distinct vectors
        let reader = MmapVectorReader::<f32>::open_fvec(&fvec).unwrap();
        for i in 0..ordinals.len() - 1 {
            let a = reader.get_slice(ordinals[i] as usize);
            let b = reader.get_slice(ordinals[i + 1] as usize);
            assert_ne!(a, b, "adjacent sorted vectors should be different");
            // Verify lexicographic order
            for d in 0..a.len() {
                if a[d] < b[d] { break; }
                if a[d] > b[d] { panic!("sort order violated at position {}", i); }
            }
        }
    }

    /// Adversarial: vectors that share long prefixes but differ in late
    /// components. This is the case that the native prefix-based sort
    /// may not order correctly within prefix groups.
    #[test]
    fn test_sort_long_shared_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        // Vectors share first 20 components, differ only at position 20
        let mut vectors = Vec::new();
        for i in (0..5).rev() {
            let mut v = vec![0.0f32; 25];
            v[20] = i as f32 * 0.1;
            vectors.push(v);
        }
        // vectors[0] has v[20]=0.4, vectors[1] has v[20]=0.3, ..., vectors[4] has v[20]=0.0

        let fvec = tmp.path().join("source.fvec");
        write_fvec(&fvec, &vectors);

        let out = tmp.path().join("sorted.ivec");
        let dups = tmp.path().join("dups.ivec");
        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("duplicates", dups.to_string_lossy().to_string());

        let mut op = ComputeSortKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let ordinals = read_ivec_values(&out);
        // Lexicographic order: component 20 determines order.
        // 0.0 < 0.1 < 0.2 < 0.3 < 0.4 → indices 4, 3, 2, 1, 0
        assert_eq!(ordinals, vec![4, 3, 2, 1, 0],
            "should sort correctly on late-differing component");
    }

    /// Adversarial: single vector, no dedup needed.
    #[test]
    fn test_sort_single_vector() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let fvec = tmp.path().join("source.fvec");
        write_fvec(&fvec, &vectors);

        let out = tmp.path().join("sorted.ivec");
        let dups = tmp.path().join("dups.ivec");
        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("duplicates", dups.to_string_lossy().to_string());

        let mut op = ComputeSortKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        assert_eq!(read_ivec_values(&out), vec![0]);
        assert_eq!(read_ivec_values(&dups).len(), 0);
    }

    /// Adversarial: all vectors identical → all but one are duplicates.
    #[test]
    fn test_sort_all_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let vectors: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0, 2.0]).collect();
        let fvec = tmp.path().join("source.fvec");
        write_fvec(&fvec, &vectors);

        let out = tmp.path().join("sorted.ivec");
        let dups = tmp.path().join("dups.ivec");
        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("duplicates", dups.to_string_lossy().to_string());

        let mut op = ComputeSortKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let ordinals = read_ivec_values(&out);
        assert_eq!(ordinals.len(), 1, "all identical → 1 unique");
        let dup_ordinals = read_ivec_values(&dups);
        assert_eq!(dup_ordinals.len(), 9, "9 duplicates");
    }
}
