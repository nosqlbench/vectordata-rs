// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate aggregated stats CSV across all datasets.
//!
//! Scans the catalog root for dataset directories containing
//! `variables.yaml`, collects their variables, and writes a single
//! CSV file with one row per dataset. Key columns appear first in
//! a canonical order.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};

/// Columns that appear first in the CSV, in this exact order.
const PRIORITY_COLUMNS: &[&str] = &[
    "dataset_name",
    "dim",
    "query_count",
    "vector_count",
    "duplicate_count",
    "zero_count",
    "base_count",
    "distance_function",
    "source_was_normalized",
    "source_mean_normal_epsilon",
    "source_max_normal_epsilon",
    "source_min_normal_epsilon",
    "is_normalized",
    "output_mean_normal_epsilon",
    "output_max_normal_epsilon",
    "output_min_normal_epsilon",
    "normal_threshold",
    "clean_count",
    "extract_input_count",
    "extract_output_count",
    "knn_queries_with_ties",
    "knn_tied_neighbors",
    // Legacy names (kept for backward compatibility)
    "mean_normal_epsilon",
    "max_normal_epsilon",
    "min_normal_epsilon",
];

pub struct CatalogStatsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(CatalogStatsOp)
}

impl CommandOp for CatalogStatsOp {
    fn command_path(&self) -> &str {
        "catalog stats"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate aggregated stats.csv across all cataloged datasets".into(),
            body: format!(
                "# catalog stats\n\n\
                 Scan the catalog root for datasets and aggregate their variables \
                 into a single CSV file.\n\n\
                 ## Description\n\n\
                 Walks the directory tree from the catalog root (found via \
                 `.catalog_root` sentinel), discovers all directories containing \
                 `variables.yaml`, and writes a multi-row CSV to `stats.csv` at \
                 the catalog root. Each row contains one dataset's variables.\n\n\
                 Key columns (dataset name, dimensions, counts, norm quality) \
                 appear first in a canonical order. Additional variables follow \
                 alphabetically. Datasets missing a variable get an empty cell.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".into(),
                type_name: "Path".into(),
                required: false,
                default: Some(".".into()),
                description: "Starting directory (walks up to find .catalog_root)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".into(),
                type_name: "Path".into(),
                required: false,
                default: Some("stats.csv".into()),
                description: "Output CSV filename (relative to catalog root)".into(),
                role: OptionRole::Output,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = options.get("source").unwrap_or(".");
        let output_name = options.get("output").unwrap_or("stats.csv");
        let input_path = if Path::new(input_str).is_absolute() {
            PathBuf::from(input_str)
        } else {
            ctx.workspace.join(input_str)
        };

        // Find catalog root
        let catalog_root = match find_catalog_root(&input_path) {
            Some(r) => r,
            None => {
                // No .catalog_root — use workspace
                ctx.ui.log("  no .catalog_root found — using workspace as root");
                ctx.workspace.clone()
            }
        };

        let output_path = catalog_root.join(output_name);
        ctx.ui.log(&format!("  catalog root: {}", catalog_root.display()));

        // Discover all dataset directories with variables.yaml
        let mut datasets: Vec<(String, indexmap::IndexMap<String, String>)> = Vec::new();
        discover_datasets(&catalog_root, &catalog_root, &mut datasets, &ctx.ui);

        if datasets.is_empty() {
            return CommandResult {
                status: Status::Ok,
                message: "no datasets with variables.yaml found".into(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Sort datasets by name
        datasets.sort_by(|a, b| a.0.cmp(&b.0));

        // Build column set: priority columns first, then all others sorted
        let mut all_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (_, vars) in &datasets {
            for k in vars.keys() {
                if !k.starts_with("verified_count:") {
                    all_keys.insert(k.clone());
                }
            }
        }

        let mut columns: Vec<String> = Vec::new();
        for &col in PRIORITY_COLUMNS {
            if all_keys.contains(col) {
                columns.push(col.to_string());
                all_keys.remove(col);
            }
        }
        for k in &all_keys {
            columns.push(k.clone());
        }

        // Write CSV
        let mut csv = String::new();
        csv.push_str(&columns.join(","));
        csv.push('\n');

        for (_name, vars) in &datasets {
            let values: Vec<String> = columns.iter()
                .map(|col| {
                    let val = vars.get(col).map(|s| s.as_str()).unwrap_or("");
                    if val.contains(',') || val.contains('"') || val.contains('\n') {
                        format!("\"{}\"", val.replace('"', "\"\""))
                    } else {
                        val.to_string()
                    }
                })
                .collect();
            csv.push_str(&values.join(","));
            csv.push('\n');
        }

        if let Err(e) = std::fs::write(&output_path, &csv) {
            return CommandResult {
                status: Status::Error,
                message: format!("failed to write {}: {}", output_path.display(), e),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Write the variable dictionary in both markdown and CSV formats
        let dict_path = output_path.with_file_name("stats-dictionary.md");
        let dict_content = build_dictionary_md(&columns);
        if let Err(e) = std::fs::write(&dict_path, &dict_content) {
            ctx.ui.log(&format!("  warning: failed to write dictionary md: {}", e));
        } else {
            ctx.ui.log(&format!("  wrote {}", dict_path.display()));
        }

        let dict_csv_path = output_path.with_file_name("stats-dictionary.csv");
        let dict_csv_content = build_dictionary_csv(&columns);
        if let Err(e) = std::fs::write(&dict_csv_path, &dict_csv_content) {
            ctx.ui.log(&format!("  warning: failed to write dictionary csv: {}", e));
        } else {
            ctx.ui.log(&format!("  wrote {}", dict_csv_path.display()));
        }

        let msg = format!(
            "{} datasets, {} columns → {}",
            datasets.len(), columns.len(), output_path.display(),
        );
        ctx.ui.log(&format!("  {}", msg));

        CommandResult {
            status: Status::Ok,
            message: msg,
            produced: vec![output_path, dict_path, dict_csv_path],
            elapsed: start.elapsed(),
        }
    }
}

/// Walk up from `dir` looking for a `.catalog_root` file.
fn find_catalog_root(dir: &Path) -> Option<PathBuf> {
    let mut current = dir.to_path_buf();
    loop {
        if current.join(".catalog_root").is_file() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Recursively discover datasets with variables.yaml under `dir`.
fn discover_datasets(
    dir: &Path,
    catalog_root: &Path,
    datasets: &mut Vec<(String, indexmap::IndexMap<String, String>)>,
    ui: &veks_core::ui::UiHandle,
) {
    let vars_path = dir.join("variables.yaml");
    if vars_path.is_file() {
        match crate::pipeline::variables::load(dir) {
            Ok(vars) if !vars.is_empty() => {
                // Skip datasets without dataset_name — incomplete or
                // pre-date the current variable schema.
                if !vars.contains_key("dataset_name") {
                    let rel = dir.strip_prefix(catalog_root)
                        .unwrap_or(dir)
                        .to_string_lossy();
                    ui.log(&format!("  skipped: {} (no dataset_name — incomplete)", rel));
                    return;
                }
                let name = vars.get("dataset_name")
                    .cloned()
                    .unwrap_or_else(|| {
                        dir.file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string()
                    });
                let rel = dir.strip_prefix(catalog_root)
                    .unwrap_or(dir)
                    .to_string_lossy();
                ui.log(&format!("  found: {} ({})", name, rel));
                datasets.push((name, vars));
            }
            Ok(_) => {}
            Err(e) => {
                ui.log(&format!("  warning: failed to load {}: {}", vars_path.display(), e));
            }
        }
        // Don't recurse into dataset directories
        return;
    }

    // Recurse into subdirectories (skip hidden dirs and common non-dataset dirs)
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut subdirs: Vec<PathBuf> = entries
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().unwrap_or_default().to_string_lossy();
                !name.starts_with('.') && name != "profiles" && name != ".cache"
            })
            .collect();
        subdirs.sort();
        for subdir in subdirs {
            discover_datasets(&subdir, catalog_root, datasets, ui);
        }
    }
}

/// Build a CSV dictionary with name,description columns.
fn build_dictionary_csv(columns: &[String]) -> String {
    let mut csv = String::from("name,description\n");
    for col in columns {
        let desc = variable_description(col).unwrap_or("No description available.");
        // Collapse to single line and CSV-quote
        let one_line = desc.replace('\n', " ").replace("  ", " ");
        csv.push_str(&format!("{},\"{}\"\n", col, one_line.replace('"', "\"\"")));
    }
    csv
}

/// Build a markdown dictionary explaining every variable in the CSV.
fn build_dictionary_md(columns: &[String]) -> String {
    let mut md = String::new();
    md.push_str("# Dataset Statistics Dictionary\n\n");
    md.push_str("This document describes each variable in `stats.csv`. Variables are listed\n");
    md.push_str("in the order they appear as CSV columns.\n\n");
    md.push_str("---\n\n");

    for col in columns {
        if let Some(desc) = variable_description(col) {
            md.push_str(&format!("## `{}`\n\n", col));
            md.push_str(desc);
            md.push_str("\n\n");
        } else {
            md.push_str(&format!("## `{}`\n\n", col));
            md.push_str("No description available for this variable.\n\n");
        }
    }

    md.push_str("---\n\n");
    md.push_str("*Generated by `veks catalog stats`. See SRD §20 for pipeline design details.*\n");
    md
}

/// Return a multi-sentence description for a known variable name.
fn variable_description(name: &str) -> Option<&'static str> {
    Some(match name {
        "dataset_name" =>
            "The human-readable name of the dataset, taken from the `name` field in `dataset.yaml`. \
             This is set during bootstrap and persisted as a pipeline variable at the start of each run. \
             It serves as the primary key when aggregating statistics across multiple datasets.",

        "dim" =>
            "The dimensionality of each vector (number of floating-point components per vector). \
             Measured during the extract-base step by reading the vector file header. \
             Common values are 128, 256, 768, 1024, and 1536 depending on the embedding model. \
             All vectors in a dataset share the same dimensionality.",

        "query_count" =>
            "The number of query vectors used for KNN ground truth computation and verification. \
             For self-search datasets, this is the number of vectors extracted from the shuffled \
             combined source as the test split. For HDF5 datasets with a separate test set, this \
             is the record count of the test dataset. Set as an upstream default during bootstrap \
             (typically 10,000) and persisted to variables during extraction.",

        "vector_count" =>
            "The total number of vectors in the working source file before any processing \
             (deduplication, zero removal, or train/test split). For self-search datasets where \
             base and query vectors are combined before processing, this is the combined count. \
             For HDF5 datasets with separate train/test sets, this is the train set count. \
             Set by the `count-vectors` pipeline step.",

        "source_base_count" =>
            "The record count of the original base vectors source file, measured before any \
             processing. For native vector files (fvec, mvec), this is the file's record count. \
             For HDF5 files, this is the record count of the selected dataset (e.g., `#train`). \
             This count predates deduplication, zero removal, and any base/query combining.",

        "source_query_count" =>
            "The record count of the original query vectors source file, measured before any \
             processing. For self-search datasets (no separate query file), this is 0. \
             For datasets with separate base and query files, this is the query file's record \
             count before zero removal or normalization.",

        "duplicate_count" =>
            "The number of exact-duplicate vectors detected and removed during the prepare-vectors \
             (sort + dedup) step. Duplicates are identified by external merge sort with prefix-key \
             comparison: vectors with matching 10-component prefixes are compared bitwise for exact \
             equality. When `elide=true` (default), duplicates are removed from the sorted ordinal \
             index so they are excluded from extraction.",

        "zero_count" =>
            "The number of near-zero base vectors filtered during extraction. A vector is classified \
             as near-zero when its L2 norm (computed in f64 precision) is below the zero threshold \
             (default 1e-6). Near-zero vectors produce degenerate unit vectors after normalization \
             and are excluded from the output. The zero ordinals are recorded in `zero_ordinals.ivecs` \
             for audit purposes.",

        "base_count" =>
            "The final number of base vectors in the prepared dataset, after all processing: \
             deduplication, zero removal, and optional fraction-based subsetting. This is the count \
             of vectors in `base_vectors.fvec` as written by the extract-base step. The provenance \
             chain is: `vector_count - duplicate_count - zero_count = base_count` (for self-search \
             datasets, `query_count` vectors are also subtracted from the available pool).",

        "distance_function" =>
            "The distance metric used for KNN ground truth computation, as detected from the source \
             filename or specified during bootstrap. Common values are COSINE, L2 (Euclidean), and \
             DOT_PRODUCT. Detection checks the symlink chain for keywords like 'cosine', 'dot', \
             'euclidean', or 'l2'. This determines which SIMD kernel is used for distance computation \
             and whether normalized dot-product optimization is applied.",

        "is_normalized" =>
            "Whether the output vectors are L2-normalized. Always `true` after pipeline processing, \
             since normalization is applied during extraction (or verified as already present). \
             Normalized vectors have unit L2 norm (within f32 precision), enabling the use of \
             dot-product kernels for cosine similarity computation.",

        "source_was_normalized" =>
            "Whether the source vectors were already L2-normalized before pipeline processing. \
             Determined by sampling up to 10,000 vectors from the source file and checking if \
             the mean |norm - 1.0| is below the normalization threshold (default 1e-5 for f32). \
             When true, the extraction step skips the per-component normalization multiply \
             (just copies raw data) but still performs zero detection and records norm statistics.",

        "source_mean_normal_epsilon" =>
            "The mean deviation of source vector L2 norms from 1.0, computed in f64 precision \
             during the extraction step before normalization is applied. Calculated as \
             mean(|sqrt(sum(x_i^2)) - 1.0|) across all non-zero vectors. A value near 0 indicates \
             the source was already well-normalized; larger values indicate how far the source \
             vectors deviated from unit length on average.",

        "source_stddev_normal_epsilon" =>
            "The standard deviation of source vector norm deviations from 1.0, computed using the \
             single-pass sum-of-squares method during extraction. Combined with the mean, this \
             characterizes the distribution of norm quality in the source data. A small stddev \
             with a small mean indicates uniformly well-normalized source vectors.",

        "source_median_normal_epsilon" =>
            "The approximate median of source vector norm deviations from 1.0, estimated via \
             reservoir sampling (approximately 1 in every 1000 vectors sampled per thread chunk). \
             The median is less sensitive to outliers than the mean and provides a robust \
             characterization of typical source norm quality.",

        "source_min_normal_epsilon" =>
            "The minimum deviation of any source vector's L2 norm from 1.0, computed in f64 \
             precision during extraction. This represents the best-normalized vector in the source \
             dataset. For already-normalized sources, this is typically on the order of 1e-12 \
             (f32 precision limit).",

        "source_max_normal_epsilon" =>
            "The maximum deviation of any source vector's L2 norm from 1.0, computed in f64 \
             precision during extraction. This represents the worst-normalized vector in the source \
             dataset. Large values (> 0.01) indicate vectors that were far from unit length and \
             benefited significantly from normalization.",

        "output_mean_normal_epsilon" =>
            "The mean deviation of output vector L2 norms from 1.0, measured from the actual bytes \
             written to the output file after normalization. Computed by re-reading each normalized \
             vector from the output buffer and measuring its f32-precision norm. For vectors that \
             were normalized during extraction, this reflects the f32 rounding error (typically \
             ~1e-7). For already-normalized sources that were copied without modification, this \
             matches the source statistics.",

        "output_stddev_normal_epsilon" =>
            "The standard deviation of output vector norm deviations from 1.0, measured from the \
             actual written output. Characterizes the uniformity of normalization precision in \
             the final dataset. Should be very small (~1e-8) for f32-normalized vectors.",

        "output_median_normal_epsilon" =>
            "The approximate median of output vector norm deviations from 1.0, estimated via \
             reservoir sampling from the written output buffer. Provides a robust central tendency \
             measure for the normalization precision of the output dataset.",

        "output_min_normal_epsilon" =>
            "The minimum norm deviation in the output dataset. Represents the best-normalized \
             vector after processing. Typically on the order of 1e-12 for f32 normalization.",

        "output_max_normal_epsilon" =>
            "The maximum norm deviation in the output dataset. Represents the worst-normalized \
             vector after processing. For f32-normalized vectors, this is typically ~1e-7 \
             (limited by IEEE 754 single-precision arithmetic).",

        "normal_threshold" =>
            "The element-type-specific threshold used to determine whether a vector set is \
             'normalized'. For f32 vectors, the default is 1e-5; for f16, 1e-1; for f64, 1e-14. \
             If the mean norm deviation is below this threshold, the source is considered already \
             normalized and the per-component multiply is skipped during extraction. The threshold \
             accounts for the precision limits of each floating-point format.",

        "clean_count" =>
            "The number of vectors remaining after deduplication but before zero removal or \
             train/test splitting. Set by the prepare-vectors (compute sort) step as the count \
             of unique vectors in the sorted ordinal index. For datasets without deduplication, \
             this equals `vector_count`. The shuffle permutation operates over `clean_count` \
             ordinals to produce the randomized train/test split.",

        "extract_input_count" =>
            "The number of base vectors read during the extraction step (input to extract-base). \
             For self-search datasets, this is `clean_count - query_count` (the base portion of \
             the shuffled split). For HDF5 datasets, this is the number of vectors selected by \
             the sorted ordinal index after deduplication.",

        "extract_output_count" =>
            "The number of base vectors actually written to the output file by extract-base. \
             This may be less than `extract_input_count` if near-zero vectors were filtered \
             during extraction. Equal to `base_count` when the count-base step runs after extraction.",

        "query_input_count" =>
            "The number of query vectors read during query extraction or conversion. For \
             self-search datasets, this is the `query_count` portion of the shuffle permutation. \
             For HDF5 datasets, this is the total record count of the test/query dataset before \
             zero removal.",

        "query_output_count" =>
            "The number of query vectors actually written to the output file after normalization \
             and zero removal. May be less than `query_input_count` if near-zero query vectors \
             were filtered. This is the effective number of queries available for KNN computation.",

        "query_zero_count" =>
            "The number of query vectors filtered as near-zero during query extraction or \
             conversion. These vectors had L2 norms below the zero threshold (1e-6) and would \
             produce degenerate results after normalization.",

        "knn_queries_with_ties" =>
            "The number of queries where the k-th and (k+1)-th nearest neighbors have exactly \
             the same distance, creating a boundary tie at rank k. With deterministic tie-breaking \
             (lower vector index wins), the result is still unique and reproducible. If any tie \
             involves bitwise-identical vectors, the pipeline reports an implementation bug \
             (duplicates survived dedup). Measured during the partition merge phase of compute-knn.",

        "knn_tied_neighbors" =>
            "The total number of extra tied neighbors beyond rank k across all queries. Each \
             boundary tie contributes one or more extra neighbors that share the exact distance \
             of the k-th neighbor. These are resolved deterministically by the tie-breaking rule \
             (lower index wins) but their presence is recorded for data quality assessment.",

        "is_shuffled" =>
            "Whether the base and query vectors were produced via a Fisher-Yates shuffle \
             permutation of the combined or deduplicated source. True for self-search datasets \
             (B-only or combined B+Q) where the shuffle provides the train/test split. False \
             for HDF5 datasets with pre-defined train/test datasets that are processed independently.",

        "is_self_search" =>
            "Whether the dataset uses self-search mode, where query vectors are drawn from the \
             same pool as base vectors via a shuffle-based split. True when no separate query \
             file is provided, or when separate base and query files are combined before \
             processing (Strategy 1: combined B+Q). False for HDF5 datasets with independent \
             train/test datasets (Strategy 2).",

        "combined_bq" =>
            "Whether separate base and query vector files were combined into a single source \
             before processing. When true, the pipeline concatenated the base and query files, \
             then deduplicated the combined set and used a shuffle to produce a clean train/test \
             split with no overlap by construction. This is Strategy 1 in the pipeline design \
             (SRD §20.7) and applies to non-HDF5 datasets that provide both base and query files.",

        "k" =>
            "The number of nearest neighbors computed per query vector (the 'k' in KNN). \
             Typically 100. Set during bootstrap and used by compute-knn to determine the heap \
             size for top-k selection. The ground truth files (neighbor_indices.ivecs and \
             neighbor_distances.fvecs) contain exactly k neighbors per query, ordered by distance \
             ascending with deterministic tie-breaking (lower index wins).",

        // Legacy aliases (source stats with old names)
        "mean_normal_epsilon" =>
            "Legacy alias for `source_mean_normal_epsilon`. The mean deviation of source vector \
             L2 norms from 1.0. See `source_mean_normal_epsilon` for full description.",

        "stddev_normal_epsilon" =>
            "Legacy alias for `source_stddev_normal_epsilon`. The standard deviation of source \
             vector norm deviations. See `source_stddev_normal_epsilon` for full description.",

        "median_normal_epsilon" =>
            "Legacy alias for `source_median_normal_epsilon`. The approximate median of source \
             vector norm deviations. See `source_median_normal_epsilon` for full description.",

        "min_normal_epsilon" =>
            "Legacy alias for `source_min_normal_epsilon`. The minimum source vector norm \
             deviation. See `source_min_normal_epsilon` for full description.",

        "max_normal_epsilon" =>
            "Legacy alias for `source_max_normal_epsilon`. The maximum source vector norm \
             deviation. See `source_max_normal_epsilon` for full description.",

        _ => return None,
    })
}
