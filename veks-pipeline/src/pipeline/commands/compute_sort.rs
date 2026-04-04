// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: sort vectors by a specified criterion.
//!
//! Reads an fvec file and writes a sorted copy. Sorting criteria include:
//! - `norm`: L2 norm of each vector (ascending)
//! - `dimension:N`: value of dimension N (ascending)
//!
//! For large files, uses an index-sort approach: compute sort keys, sort
//! indices, then write output in sorted order via mmap random access.
//!
//! Equivalent to the Java `CMD_compute_sort` command.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};
use std::time::Instant;

use rayon::prelude::*;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

// ---------------------------------------------------------------------------
// Format-agnostic vector reader (f32 and f16 → f32 upcast)
// ---------------------------------------------------------------------------

/// Uniform read interface that always returns `Vec<f32>`, upcasting f16
/// when the source is an mvec file.
enum VecReader {
    F32(MmapVectorReader<f32>),
    F16(MmapVectorReader<half::f16>),
}

impl VecReader {
    /// Open the appropriate reader based on file extension.
    fn open(path: &Path) -> Result<Self, String> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("mvec") | Some("mvecs") => {
                MmapVectorReader::<half::f16>::open_mvec(path)
                    .map(VecReader::F16)
                    .map_err(|e| format!("failed to open {}: {}", path.display(), e))
            }
            _ => {
                MmapVectorReader::<f32>::open_fvec(path)
                    .map(VecReader::F32)
                    .map_err(|e| format!("failed to open {}: {}", path.display(), e))
            }
        }
    }

    fn count(&self) -> usize {
        match self {
            VecReader::F32(r) => <MmapVectorReader<f32> as VectorReader<f32>>::count(r),
            VecReader::F16(r) => <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(r),
        }
    }

    fn dim(&self) -> usize {
        match self {
            VecReader::F32(r) => <MmapVectorReader<f32> as VectorReader<f32>>::dim(r),
            VecReader::F16(r) => <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(r),
        }
    }

    /// Get vector at index as f32 (upcasting f16 if needed).
    fn get_f32(&self, index: usize) -> Result<Vec<f32>, String> {
        match self {
            VecReader::F32(r) => r.get(index)
                .map_err(|e| format!("failed to read vector {}: {}", index, e)),
            VecReader::F16(r) => {
                let v = r.get(index)
                    .map_err(|e| format!("failed to read vector {}: {}", index, e))?;
                Ok(v.iter().map(|x| x.to_f32()).collect())
            }
        }
    }

    /// Get a raw byte view of the vector at index directly from the mmap.
    ///
    /// Zero allocation, zero copy — returns a slice into the memory-mapped
    /// file. The caller can write this directly to the output.
    fn raw_bytes(&self, index: usize) -> &[u8] {
        match self {
            VecReader::F32(r) => {
                let slice = r.get_slice(index);
                unsafe {
                    std::slice::from_raw_parts(
                        slice.as_ptr() as *const u8,
                        slice.len() * 4,
                    )
                }
            }
            VecReader::F16(r) => {
                let slice = r.get_slice(index);
                unsafe {
                    std::slice::from_raw_parts(
                        slice.as_ptr() as *const u8,
                        slice.len() * 2,
                    )
                }
            }
        }
    }

    /// Element size in bytes for the native format.
    fn element_size(&self) -> usize {
        match self {
            VecReader::F32(_) => 4,
            VecReader::F16(_) => 2,
        }
    }
}

/// Pipeline command: sort vectors.
pub struct ComputeSortOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeSortOp)
}

/// Sort criterion.
#[derive(Debug, Clone)]
enum SortCriterion {
    /// Sort by L2 norm ascending.
    Norm,
    /// Sort by value of a specific dimension ascending.
    Dimension(usize),
}

impl SortCriterion {
    fn from_str(s: &str, max_dim: usize) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "norm" | "l2norm" | "l2" => Ok(SortCriterion::Norm),
            other => {
                if let Some(dim_str) = other.strip_prefix("dimension:") {
                    let dim: usize = dim_str
                        .parse()
                        .map_err(|_| format!("invalid dimension index: '{}'", dim_str))?;
                    if dim >= max_dim {
                        return Err(format!(
                            "dimension index {} out of range (max {})",
                            dim,
                            max_dim - 1
                        ));
                    }
                    Ok(SortCriterion::Dimension(dim))
                } else if let Some(dim_str) = other.strip_prefix("dim:") {
                    let dim: usize = dim_str
                        .parse()
                        .map_err(|_| format!("invalid dimension index: '{}'", dim_str))?;
                    if dim >= max_dim {
                        return Err(format!(
                            "dimension index {} out of range (max {})",
                            dim,
                            max_dim - 1
                        ));
                    }
                    Ok(SortCriterion::Dimension(dim))
                } else {
                    Err(format!(
                        "unknown sort criterion: '{}'. Use 'norm' or 'dimension:N'",
                        s
                    ))
                }
            }
        }
    }

    /// Compute the sort key for a vector.
    fn key(&self, vec: &[f32]) -> f64 {
        match self {
            SortCriterion::Norm => {
                let mut sum = 0.0f64;
                for &v in vec {
                    let vf = v as f64;
                    sum += vf * vf;
                }
                sum.sqrt()
            }
            SortCriterion::Dimension(d) => vec[*d] as f64,
        }
    }
}

impl CommandOp for ComputeSortOp {
    fn command_path(&self) -> &str {
        "compute sort"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Sort vectors by ordinal mapping".into(),
            body: format!(r#"# compute sort

Sort vectors by ordinal mapping.

## Description

Reads an fvec file and writes a new copy with vectors reordered according
to the chosen sort criterion. The original file is not modified.

### Sort Criteria

The following criteria are supported (all sort ascending):

- **`norm`** / `l2norm` / `l2` — sort by the L2 (Euclidean) norm of each
  vector. Vectors with smaller magnitudes come first.
- **`dimension:N`** / `dim:N` — sort by the value of a single dimension N.
  Useful for ordering vectors by a particular feature axis.

### How It Works

The command uses an **index-sort** approach for efficiency:

1. Compute a scalar sort key for every vector (e.g., its L2 norm).
2. Build an array of (key, original_index) pairs and sort it. The sort is
   stable, so vectors with equal keys preserve their original relative
   order.
3. Write the output file by reading vectors from the source in the new
   sorted order via mmap random access.

This strategy avoids shuffling full vector payloads during the sort itself;
only lightweight index pairs are moved until the final sequential write
pass.

### Role in the Pipeline

Sorting vectors by ordinal mapping is a preprocessing step that can improve
data locality for downstream operations. For example, sorting base vectors
by norm groups similarly-scaled vectors together, which can benefit
partitioned KNN computation by producing more uniform distance
distributions within each partition. It can also be used to reorder vectors
to match an externally defined permutation or clustering assignment.

## Options

{}

## Notes

- Sort is stable: vectors with equal keys preserve their original order.
- Accepts `norm`, `l2norm`, `l2`, `dimension:N`, or `dim:N` as sort criteria.
- Output file size matches the input file size exactly.
"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Sort buffers for vector reordering".into(), adjustable: true },
            ResourceDesc { name: "threads".into(), description: "Parallel sort/copy".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let sort_by_str = options.get("sort-by").unwrap_or("norm");

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Open source (format-agnostic: fvec or mvec)
        let reader = match VecReader::open(&source_path) {
            Ok(r) => r,
            Err(e) => return error_result(e, start),
        };

        let count = reader.count();
        let dim = reader.dim();

        let criterion = match SortCriterion::from_str(sort_by_str, dim) {
            Ok(c) => c,
            Err(e) => return error_result(e, start),
        };

        ctx.ui.log(&format!(
            "Sort: {} vectors (dim={}) by {:?}",
            count, dim, criterion
        ));

        // Governor checkpoint before sort processing
        if ctx.governor.checkpoint() {
            ctx.ui.log("  governor: throttle active");
        }

        // Build a scoped rayon thread pool governed by the resource governor.
        let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;
        ctx.ui.log(&format!("  {} threads for parallel sort-key computation", threads));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        // Phase 1: Compute sort keys (parallel)
        let phase1_start = Instant::now();
        ctx.ui.log(&format!("  phase 1/3: computing sort keys for {} vectors...", count));
        let pb = ctx.ui.bar_with_unit(count as u64, "computing sort keys", "vec");
        let mut indexed_keys: Vec<(usize, f64)>;
        {
            let progress = AtomicU64::new(0);
            // Update progress every ~10K records or 1% of total, whichever is larger
            let update_interval = (count / 100).max(10_000);

            let build_fn = || {
                (0..count)
                    .into_par_iter()
                    .map(|i| {
                        let vec = reader.get_f32(i)?;
                        let key = criterion.key(&vec);
                        let done = progress.fetch_add(1, AtomicOrd::Relaxed) + 1;
                        if done % update_interval as u64 == 0 {
                            pb.set_position(done);
                        }
                        Ok((i, key))
                    })
                    .collect::<Result<Vec<_>, String>>()
            };

            let result = if let Some(ref p) = pool {
                p.install(build_fn)
            } else {
                build_fn()
            };

            indexed_keys = match result {
                Ok(v) => v,
                Err(e) => return error_result(e, start),
            };
        }
        pb.finish();
        ctx.ui.log(&format!("  phase 1/3 done ({:.1}s)", phase1_start.elapsed().as_secs_f64()));

        // Phase 2: Sort by key (parallel, stable)
        let phase2_start = Instant::now();
        ctx.ui.log(&format!("  phase 2/3: sorting {} index pairs...", count));
        let sort_sp = ctx.ui.spinner("sorting index pairs");
        {
            let mut sort_fn = || {
                indexed_keys.par_sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            };
            if let Some(ref p) = pool {
                p.install(sort_fn);
            } else {
                sort_fn();
            }
        }
        sort_sp.finish();
        ctx.ui.log(&format!("  phase 2/3 done ({:.1}s)", phase2_start.elapsed().as_secs_f64()));

        // Governor checkpoint after sort phase
        if ctx.governor.checkpoint() {
            ctx.ui.log("  governor: throttle active");
        }

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        // Phase 3: Write sorted vectors (preserves native format)
        let phase3_start = Instant::now();
        ctx.ui.log(&format!("  phase 3/3: writing {} sorted vectors...", count));
        let write_pb = ctx.ui.bar_with_unit(count as u64, "writing sorted vectors", "vec");
        if let Err(e) = write_sorted_xvec(&output_path, &reader, dim, &indexed_keys, &write_pb) {
            return error_result(e, start);
        }
        write_pb.finish();
        ctx.ui.log(&format!("  phase 3/3 done ({:.1}s)", phase3_start.elapsed().as_secs_f64()));

        // Write verified count for the bound checker
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &count.to_string());
        ctx.defaults.insert(var_name, count.to_string());

        CommandResult {
            status: Status::Ok,
            message: format!(
                "sorted {} vectors (dim={}) by {:?} to {}",
                count,
                dim,
                criterion,
                output_path.display()
            ),
            produced: vec![output_path],
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
                description: "Input fvec file".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output sorted fvec file".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "sort-by".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("norm".to_string()),
                description: "Sort criterion: 'norm' or 'dimension:N'".to_string(),
                role: OptionRole::Config,
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

/// Write vectors in the order specified by sorted indices.
///
/// Preserves the native element size of the source format — fvec output
/// for fvec input, mvec output for mvec input.
fn write_sorted_xvec(
    output: &Path,
    reader: &VecReader,
    dim: usize,
    sorted_indices: &[(usize, f64)],
    pb: &veks_core::ui::ProgressHandle,
) -> Result<(), String> {
    use std::io::Write;

    let mut writer = AtomicWriter::new(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;

    let update_interval = (sorted_indices.len() / 200).max(1_000);
    let dim_i32 = dim as i32;
    let dim_header = dim_i32.to_le_bytes();

    for (i, &(idx, _)) in sorted_indices.iter().enumerate() {
        // Zero-alloc: raw_bytes is a direct slice into the mmap
        let raw = reader.raw_bytes(idx);

        writer
            .write_all(&dim_header)
            .map_err(|e| e.to_string())?;
        writer
            .write_all(raw)
            .map_err(|e| e.to_string())?;

        if (i + 1) % update_interval == 0 {
            pb.set_position((i + 1) as u64);
        }
    }

    writer.finish().map_err(|e| e.to_string())?;
    Ok(())
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
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

    /// Write a small fvec file with known vectors for testing.
    fn write_test_fvec(path: &Path, vectors: &[Vec<f32>]) {
        use std::io::Write;
        let file = std::fs::File::create(path).unwrap();
        let mut w = std::io::BufWriter::new(file);
        for vec in vectors {
            let dim = vec.len() as i32;
            w.write_all(&dim.to_le_bytes()).unwrap();
            for &v in vec {
                w.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    /// Read an fvec file back into vectors.
    fn read_fvec(path: &Path) -> Vec<Vec<f32>> {
        let data = std::fs::read(path).unwrap();
        let mut result = Vec::new();
        let mut offset = 0;
        while offset < data.len() {
            let dim = i32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                let v = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                vec.push(v);
                offset += 4;
            }
            result.push(vec);
        }
        result
    }

    #[test]
    fn test_sort_by_norm() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        // Vectors with known norms: [3,0]→3, [1,0]→1, [2,0]→2
        let source = workspace.join("input.fvec");
        write_test_fvec(
            &source,
            &[vec![3.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]],
        );

        let output = workspace.join("sorted.fvec");
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("sort-by", "norm");

        let mut op = ComputeSortOp;
        let mut ctx = test_ctx(workspace);
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let sorted = read_fvec(&output);
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0], vec![1.0, 0.0]); // norm 1
        assert_eq!(sorted[1], vec![2.0, 0.0]); // norm 2
        assert_eq!(sorted[2], vec![3.0, 0.0]); // norm 3
    }

    #[test]
    fn test_sort_by_dimension() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        // Sort by dimension 1 (second component)
        let source = workspace.join("input.fvec");
        write_test_fvec(
            &source,
            &[
                vec![0.0, 5.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 3.0, 0.0],
            ],
        );

        let output = workspace.join("sorted.fvec");
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("sort-by", "dimension:1");

        let mut op = ComputeSortOp;
        let mut ctx = test_ctx(workspace);
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let sorted = read_fvec(&output);
        assert_eq!(sorted[0][1], 1.0);
        assert_eq!(sorted[1][1], 3.0);
        assert_eq!(sorted[2][1], 5.0);
    }

    #[test]
    fn test_sort_preserves_size() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        let source = workspace.join("input.fvec");
        write_test_fvec(
            &source,
            &[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        );

        let output = workspace.join("sorted.fvec");
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());

        let mut op = ComputeSortOp;
        let mut ctx = test_ctx(workspace);
        op.execute(&opts, &mut ctx);

        let src_size = std::fs::metadata(&source).unwrap().len();
        let out_size = std::fs::metadata(&output).unwrap().len();
        assert_eq!(src_size, out_size);
    }

    #[test]
    fn test_sort_criterion_parsing() {
        assert!(SortCriterion::from_str("norm", 10).is_ok());
        assert!(SortCriterion::from_str("l2norm", 10).is_ok());
        assert!(SortCriterion::from_str("dimension:0", 10).is_ok());
        assert!(SortCriterion::from_str("dim:5", 10).is_ok());
        assert!(SortCriterion::from_str("dimension:10", 10).is_err()); // out of range
        assert!(SortCriterion::from_str("invalid", 10).is_err());
    }

    #[test]
    fn test_sort_invalid_dimension() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        let source = workspace.join("input.fvec");
        write_test_fvec(&source, &[vec![1.0, 2.0]]);

        let output = workspace.join("sorted.fvec");
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("sort-by", "dimension:5");

        let mut op = ComputeSortOp;
        let mut ctx = test_ctx(workspace);
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("out of range"));
    }
}
