// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate a predicated dataset.
//!
//! Starting from an existing base dataset (base vectors, queries, ground truth),
//! adds structured metadata attributes, query predicates, and recomputed
//! predicated ground truth (KNN over predicate-filtered base vectors).
//!
//! Produces:
//! - `attribute_schema.yaml` — field definitions
//! - `attributes_<name>.dat` — per-field binary column files
//! - `predicates.pnodes` — binary predicate tree per query
//! - `predicated_indices.ivec` — predicated KNN indices
//! - `predicated_distances.fvec` — predicated KNN distances

use std::collections::BinaryHeap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::predicate::attribute::{AttributeColumn, FieldType};
use crate::pipeline::predicate::codec;
use crate::pipeline::predicate::generator;
use crate::pipeline::predicate::predicate::{PNode, evaluate};
use crate::pipeline::rng;
use crate::pipeline::simd_distance;

/// Pipeline command: generate predicated dataset.
pub struct GeneratePredicatedOp;

/// Create a boxed `GeneratePredicatedOp`.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GeneratePredicatedOp)
}

impl CommandOp for GeneratePredicatedOp {
    fn command_path(&self) -> &str {
        "generate predicated"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate predicated dataset with filtered ground truth".into(),
            body: format!(r#"# generate predicated

Generate predicated dataset with filtered ground truth.

## Description

Starting from an existing base dataset (containing `base.fvec`,
`query.fvec`, and `dataset.yaml`), this command augments the dataset with
structured metadata attributes, per-query predicate filters, and
recomputed ground truth that accounts for those filters. The result is a
dataset suitable for evaluating **filtered (predicated) ANN search**
implementations.

## What it produces

- `attribute_schema.yaml` -- field definitions (name, type, cardinality)
- `attributes_<name>.dat` -- per-field binary column files with synthetic
  attribute values for each base vector
- `predicates.pnodes` -- one binary PNode predicate tree per query
- `predicated_indices.ivec` -- predicated KNN indices (only neighbors
  that satisfy the query's predicate)
- `predicated_distances.fvec` -- predicated KNN distances

The existing `dataset.yaml` is updated with references to the new
predicated facets.

## How it works

1. **Attribute generation** -- for each configured field, a column of
   synthetic attribute values is generated for every base vector. Field
   types include `int`, `long`, `enum`, and `enum_set`, each with a
   configurable cardinality.
2. **Predicate generation** -- for each query, a PNode predicate tree is
   synthesized targeting the configured selectivity (fraction of base
   vectors that satisfy the predicate). Predicate complexity can be
   `simple` (AND-only) or `compound` (AND/OR).
3. **Predicated KNN** -- for each query, the base vectors are filtered by
   the query's predicate, and exact KNN is computed over the filtered
   subset using SIMD-accelerated distance functions. This produces the
   ground truth for filtered search evaluation.

## Role in dataset pipelines

Filtered search is a key capability for production vector databases, where
queries often include metadata constraints (e.g. "find similar images
where category = 'landscape' AND year >= 2020"). This command creates
the test harness for measuring filtered-search recall and performance.

The selectivity parameter controls the difficulty of the benchmark: lower
selectivity means fewer eligible neighbors, which is harder for
approximate algorithms. The `predicate-complexity` parameter controls
whether predicates are simple single-clause filters or compound boolean
trees.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Dataset buffers during predicated output".into(), adjustable: true },
            ResourceDesc { name: "threads".into(), description: "Parallel processing".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential prefetch for input files".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_dir_str = match options.require("input-dir") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let input_dir = resolve_path(input_dir_str, &ctx.workspace);
        let output_dir = options
            .get("output-dir")
            .map(|s| resolve_path(s, &ctx.workspace))
            .unwrap_or_else(|| input_dir.clone());

        let field_count: usize = options
            .get("fields")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let field_types_str = options.get("field-types").unwrap_or("int,enum,enum_set");
        let cardinalities_str = options.get("cardinalities").unwrap_or("12,30,28");
        let selectivity: f64 = options
            .get("selectivity")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.1);
        let complexity = options.get("predicate-complexity").unwrap_or("simple");
        let seed = rng::parse_seed(options.get("seed"));

        // Parse dataset.yaml to get distance metric and neighbors
        let dataset_yaml_path = input_dir.join("dataset.yaml");
        let dataset_yaml = match std::fs::read_to_string(&dataset_yaml_path) {
            Ok(s) => s,
            Err(e) => {
                return error_result(
                    format!("failed to read dataset.yaml: {}", e),
                    start,
                );
            }
        };

        let distance_name = extract_yaml_value(&dataset_yaml, "distance_function")
            .unwrap_or("L2".to_string());
        let neighbors_from_yaml: usize = extract_yaml_value(&dataset_yaml, "neighbors")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        let neighbors: usize = options
            .get("neighbors")
            .and_then(|s| s.parse().ok())
            .unwrap_or(neighbors_from_yaml);

        // Open base and query vectors
        let base_path = input_dir.join("base.fvec");
        let query_path = input_dir.join("query.fvec");

        let base = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(format!("failed to open base {}: {}", base_path.display(), e), start);
            }
        };
        let query = match MmapVectorReader::<f32>::open_fvec(&query_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(format!("failed to open query {}: {}", query_path.display(), e), start);
            }
        };

        let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&base);
        let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query);

        // Parse field types and cardinalities
        let type_strs: Vec<&str> = field_types_str.split(',').map(|s| s.trim()).collect();
        let cardinalities: Vec<u32> = cardinalities_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        // Generate schema
        let schema = generator::generate_schema(field_count, &type_strs, &cardinalities);

        // Generate attribute columns
        ctx.ui.log(&format!(
            "  Generating {} attribute columns for {} base vectors...",
            field_count, base_count
        ));
        let mut rng_inst = rng::seeded_rng(seed);
        let columns = generator::generate_columns(&schema, base_count, &mut rng_inst);

        // Generate predicates for each query
        ctx.ui.log(&format!(
            "  Generating {} predicates (selectivity={}, complexity={})...",
            query_count, selectivity, complexity
        ));
        let predicates: Vec<PNode> = (0..query_count)
            .map(|_| match complexity {
                "compound" => {
                    generator::generate_predicate_compound(&schema, selectivity, &mut rng_inst)
                }
                _ => {
                    generator::generate_predicate_simple(&schema, selectivity, &mut rng_inst)
                }
            })
            .collect();

        // Create output directory
        if let Err(e) = std::fs::create_dir_all(&output_dir) {
            return error_result(format!("failed to create output dir: {}", e), start);
        }

        // Write attribute schema
        let schema_path = output_dir.join("attribute_schema.yaml");
        match schema.to_yaml() {
            Ok(yaml) => {
                if let Err(e) = std::fs::write(&schema_path, &yaml) {
                    return error_result(format!("failed to write schema: {}", e), start);
                }
            }
            Err(e) => return error_result(e, start),
        }

        // Write attribute columns
        let mut produced = vec![schema_path.clone()];
        for (fi, (field, col)) in schema.fields.iter().zip(columns.iter()).enumerate() {
            let col_path = output_dir.join(format!("attributes_{}.dat", field.name));
            if let Err(e) = write_column(&col_path, col, &field.field_type) {
                return error_result(
                    format!("failed to write column {}: {}", fi, e),
                    start,
                );
            }
            produced.push(col_path);
        }

        // Write predicates
        let predicates_path = output_dir.join("predicates.pnodes");
        match std::fs::File::create(&predicates_path) {
            Ok(f) => {
                let mut bw = std::io::BufWriter::new(f);
                if let Err(e) = codec::write_predicates(&mut bw, &predicates) {
                    return error_result(format!("failed to write predicates: {}", e), start);
                }
                if let Err(e) = bw.flush() {
                    return error_result(format!("flush error: {}", e), start);
                }
            }
            Err(e) => {
                return error_result(
                    format!("failed to create predicates file: {}", e),
                    start,
                );
            }
        }
        produced.push(predicates_path);

        // Compute predicated ground truth
        let metric = match simd_distance::Metric::from_str(&distance_name) {
            Some(m) => m,
            None => {
                return error_result(
                    format!("unknown distance metric: {}", distance_name),
                    start,
                );
            }
        };
        let dist_fn = simd_distance::select_distance_fn(metric);
        let effective_k = neighbors.min(base_count);

        let indices_path = output_dir.join("predicated_indices.ivec");
        let distances_path = output_dir.join("predicated_distances.fvec");

        ctx.ui.log(&format!(
            "  Computing predicated {}-NN ({} queries × {} base)...",
            effective_k, query_count, base_count
        ));

        // Governor checkpoint before predicated KNN computation
        if ctx.governor.checkpoint() {
            ctx.ui.log("  governor: throttle active");
        }

        match compute_predicated_knn(
            &base,
            &query,
            &columns,
            &predicates,
            &indices_path,
            &distances_path,
            effective_k,
            dist_fn,
        ) {
            Ok(_) => {}
            Err(e) => return error_result(e, start),
        }

        produced.push(indices_path);
        produced.push(distances_path);

        // Update dataset.yaml with predicated facets
        let updated_yaml = format!(
            "{}\n\
             \x20 predicated_attributes:\n\
             \x20   path: attribute_schema.yaml\n\
             \x20 predicated_predicates:\n\
             \x20   path: predicates.pnodes\n\
             \x20 predicated_indices:\n\
             \x20   path: predicated_indices.ivec\n\
             \x20 predicated_distances:\n\
             \x20   path: predicated_distances.fvec\n",
            dataset_yaml.trim_end()
        );
        let updated_yaml_path = output_dir.join("dataset.yaml");
        if let Err(e) = std::fs::write(&updated_yaml_path, &updated_yaml) {
            return error_result(format!("failed to update dataset.yaml: {}", e), start);
        }

        // Report selectivity stats
        let avg_sel: f64 = predicates
            .iter()
            .map(|p| generator::measure_selectivity(p, &columns, base_count))
            .sum::<f64>()
            / query_count as f64;
        ctx.ui.log(&format!(
            "  Average selectivity: {:.4} (target: {:.4})",
            avg_sel, selectivity
        ));

        ctx.ui.log(&format!(
            "  Predicated dataset created: {}",
            output_dir.display()
        ));

        // Write verified counts for the bound checker
        for xvec_path in &produced {
            let ext = xvec_path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "fvec" | "ivec" | "bvec" | "dvec" | "mvec" | "svec") {
                let var_name = format!("verified_count:{}",
                    xvec_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
                let _ = crate::pipeline::variables::set_and_save(
                    &ctx.workspace, &var_name, &query_count.to_string());
                ctx.defaults.insert(var_name, query_count.to_string());
            }
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "generated predicated dataset: {} fields, {} queries, k={}, avg_sel={:.4}",
                field_count, query_count, effective_k, avg_sel
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "input-dir".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Existing dataset directory with base.fvec, query.fvec, dataset.yaml"
                    .to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output-dir".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output directory (defaults to input-dir)".to_string(),
                        role: OptionRole::Output,
        },
            OptionDesc {
                name: "fields".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("3".to_string()),
                description: "Number of attribute fields".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "field-types".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("int,enum,enum_set".to_string()),
                description: "Comma-separated field types (int, long, enum, enum_set)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "cardinalities".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("12,30,28".to_string()),
                description: "Comma-separated cardinalities per field".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "selectivity".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.1".to_string()),
                description: "Target fraction of base vectors satisfying predicates".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "predicate-complexity".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("simple".to_string()),
                description: "Predicate complexity: simple (AND-only) or compound (AND/OR)"
                    .to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "neighbors".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "K for predicated KNN (default: from dataset.yaml)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("42".to_string()),
                description: "PRNG seed for reproducibility".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Compute predicated KNN: for each query, filter base vectors by predicate,
/// then compute exact KNN over the filtered set.
fn compute_predicated_knn(
    base: &MmapVectorReader<f32>,
    query: &MmapVectorReader<f32>,
    columns: &[AttributeColumn],
    predicates: &[PNode],
    indices_path: &Path,
    distances_path: &Path,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> Result<(), String> {
    let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(base);
    let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(query);

    let mut idx_file = std::io::BufWriter::new(
        std::fs::File::create(indices_path)
            .map_err(|e| format!("failed to create indices: {}", e))?,
    );
    let mut dist_file = std::io::BufWriter::new(
        std::fs::File::create(distances_path)
            .map_err(|e| format!("failed to create distances: {}", e))?,
    );

    let k_bytes = (k as i32).to_le_bytes();

    for qi in 0..query_count {
        let qvec = query
            .get(qi)
            .map_err(|e| format!("read query {}: {}", qi, e))?;

        let pred = &predicates[qi % predicates.len()];

        // Max-heap for top-k smallest distances
        let mut heap: BinaryHeap<DistEntry> = BinaryHeap::new();

        for bi in 0..base_count {
            // Filter by predicate
            if !evaluate(pred, columns, bi) {
                continue;
            }

            let bvec = base
                .get(bi)
                .map_err(|e| format!("read base {}: {}", bi, e))?;
            let d = dist_fn(&qvec, &bvec);

            if heap.len() < k {
                heap.push(DistEntry { dist: d, idx: bi });
            } else if let Some(top) = heap.peek() {
                if d < top.dist {
                    heap.pop();
                    heap.push(DistEntry { dist: d, idx: bi });
                }
            }
        }

        // Sort by distance ascending
        let mut results: Vec<DistEntry> = heap.into_sorted_vec();
        results.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Write indices
        idx_file
            .write_all(&k_bytes)
            .map_err(|e| format!("write error: {}", e))?;
        for r in &results {
            idx_file
                .write_all(&(r.idx as i32).to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }
        for _ in results.len()..k {
            idx_file
                .write_all(&(-1i32).to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }

        // Write distances
        dist_file
            .write_all(&k_bytes)
            .map_err(|e| format!("write error: {}", e))?;
        for r in &results {
            dist_file
                .write_all(&r.dist.to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }
        for _ in results.len()..k {
            dist_file
                .write_all(&f32::MAX.to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }
    }

    idx_file
        .flush()
        .map_err(|e| format!("flush error: {}", e))?;
    dist_file
        .flush()
        .map_err(|e| format!("flush error: {}", e))?;

    Ok(())
}

/// Write an attribute column to a binary file.
fn write_column(path: &Path, col: &AttributeColumn, _ft: &FieldType) -> Result<(), String> {
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    let mut buf = std::io::BufWriter::new(&mut file);

    match col {
        AttributeColumn::Int(vals) => {
            for &v in vals {
                buf.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write error: {}", e))?;
            }
        }
        AttributeColumn::Long(vals) => {
            for &v in vals {
                buf.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write error: {}", e))?;
            }
        }
        AttributeColumn::Enum(vals) => {
            for &v in vals {
                buf.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write error: {}", e))?;
            }
        }
        AttributeColumn::EnumSet(vals) => {
            // Write each bitmask with a length prefix
            for bitmask in vals {
                let len = bitmask.len() as u32;
                buf.write_all(&len.to_le_bytes())
                    .map_err(|e| format!("write error: {}", e))?;
                buf.write_all(bitmask)
                    .map_err(|e| format!("write error: {}", e))?;
            }
        }
    }

    buf.flush().map_err(|e| format!("flush error: {}", e))?;
    Ok(())
}

/// Extract a simple value from YAML by key (basic line-based parsing).
fn extract_yaml_value(yaml: &str, key: &str) -> Option<String> {
    for line in yaml.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(key) {
            if let Some(value) = rest.strip_prefix(':') {
                return Some(value.trim().to_string());
            }
        }
    }
    None
}

#[derive(Debug, Clone)]
struct DistEntry {
    dist: f32,
    idx: usize,
}

impl PartialEq for DistEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for DistEntry {}

impl PartialOrd for DistEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for max-heap (smallest distances at top)
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
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
    use crate::pipeline::commands::gen_dataset::GenerateDatasetOp;
    use crate::pipeline::predicate::attribute::AttributeSchema;
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
        }
    }

    #[test]
    fn test_generate_predicated_small() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // First generate a base dataset
        let dataset_dir = ws.join("dataset");
        let mut opts = Options::new();
        opts.set("output-dir", dataset_dir.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("base-count", "50");
        opts.set("query-count", "5");
        opts.set("neighbors", "3");
        opts.set("seed", "42");

        let mut dataset_op = GenerateDatasetOp;
        let result = dataset_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "dataset gen failed: {}", result.message);

        // Now generate predicated dataset
        let mut opts = Options::new();
        opts.set("input-dir", dataset_dir.to_string_lossy().to_string());
        opts.set("fields", "2");
        opts.set("field-types", "int,enum");
        opts.set("cardinalities", "10,5");
        opts.set("selectivity", "0.5");
        opts.set("neighbors", "3");
        opts.set("seed", "42");

        let mut op = GeneratePredicatedOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "predicated gen failed: {}", result.message);

        // Verify output files exist
        assert!(dataset_dir.join("attribute_schema.yaml").exists());
        assert!(dataset_dir.join("predicates.pnodes").exists());
        assert!(dataset_dir.join("predicated_indices.ivec").exists());
        assert!(dataset_dir.join("predicated_distances.fvec").exists());

        // Verify predicated indices file is valid
        let idx_reader =
            MmapVectorReader::<i32>::open_ivec(&dataset_dir.join("predicated_indices.ivec"))
                .unwrap();
        assert_eq!(
            <MmapVectorReader<i32> as VectorReader<i32>>::count(&idx_reader),
            5
        );
    }

    #[test]
    fn test_predicated_neighbors_satisfy_predicate() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate base dataset
        let dataset_dir = ws.join("dataset");
        let mut opts = Options::new();
        opts.set("output-dir", dataset_dir.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("base-count", "100");
        opts.set("query-count", "3");
        opts.set("neighbors", "5");
        opts.set("seed", "42");

        let mut dataset_op = GenerateDatasetOp;
        dataset_op.execute(&opts, &mut ctx);

        // Generate predicated with simple int field
        let mut opts = Options::new();
        opts.set("input-dir", dataset_dir.to_string_lossy().to_string());
        opts.set("fields", "1");
        opts.set("field-types", "int");
        opts.set("cardinalities", "10");
        opts.set("selectivity", "0.3");
        opts.set("neighbors", "5");
        opts.set("seed", "42");

        let mut op = GeneratePredicatedOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        // Read the schema and regenerate columns to verify
        let schema_yaml =
            std::fs::read_to_string(dataset_dir.join("attribute_schema.yaml")).unwrap();
        let schema = AttributeSchema::from_yaml(&schema_yaml).unwrap();

        let mut rng_inst = rng::seeded_rng(42);
        let columns = generator::generate_columns(&schema, 100, &mut rng_inst);

        // Read predicates
        let pred_data = std::fs::read(dataset_dir.join("predicates.pnodes")).unwrap();
        let mut cursor = std::io::Cursor::new(&pred_data);
        let predicates = codec::read_predicates(&mut cursor).unwrap();
        assert_eq!(predicates.len(), 3);

        // Read predicated indices
        let idx_reader =
            MmapVectorReader::<i32>::open_ivec(&dataset_dir.join("predicated_indices.ivec"))
                .unwrap();

        // Verify all non-(-1) indices satisfy the predicate
        for qi in 0..3 {
            let indices = idx_reader.get(qi).unwrap();
            let pred = &predicates[qi];
            for &idx in &*indices {
                if idx >= 0 {
                    assert!(
                        evaluate(pred, &columns, idx as usize),
                        "query {} neighbor {} does not satisfy predicate",
                        qi,
                        idx
                    );
                }
            }
        }
    }

    #[test]
    fn test_extract_yaml_value() {
        let yaml = "name: test\n  distance_function: COSINE\n  dimension: 8\n";
        assert_eq!(
            extract_yaml_value(yaml, "distance_function"),
            Some("COSINE".to_string())
        );
        assert_eq!(
            extract_yaml_value(yaml, "dimension"),
            Some("8".to_string())
        );
        assert_eq!(extract_yaml_value(yaml, "missing"), None);
    }

    #[test]
    fn test_predicated_missing_dataset() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let mut opts = Options::new();
        opts.set("input-dir", ws.join("nonexistent").to_string_lossy().to_string());

        let mut op = GeneratePredicatedOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }
}
