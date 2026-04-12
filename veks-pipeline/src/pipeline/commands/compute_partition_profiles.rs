// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: prepare per-label partition profiles.
//!
//! Given an existing predicated dataset (M), extracts base vectors for
//! each unique label value into separate partition profiles and registers
//! them in dataset.yaml. KNN computation for each partition is handled
//! by the existing `per_profile` compute-knn template via pipeline
//! re-expansion.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

pub struct ComputePartitionProfilesOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputePartitionProfilesOp)
}

impl CommandOp for ComputePartitionProfilesOp {
    fn command_path(&self) -> &str {
        "compute partition-profiles"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Prepare per-label partition profiles with extracted base vectors".into(),
            body: format!(
                r#"# compute partition-profiles

Prepare per-label partition profiles from a predicated dataset.

## Description

For each unique metadata label value, extracts the matching base
vectors into a partition profile directory and registers the profile
in dataset.yaml. KNN computation within each partition is then
handled by the existing `per_profile` compute-knn template via
pipeline re-expansion — the same code path used for sized profiles.

Each partition profile contains:
- Base vectors (subset matching the label, in partition ordinal space)
- Query vectors (symlinked from shared source)

After this step completes, the pipeline re-expands `per_profile`
templates (compute-knn, verify-knn) for the new partition profiles.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Base vector buffers".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let metadata_str = match options.require("metadata") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let k: usize = match options.require("neighbors") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid neighbors: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let prefix = options.get("prefix").unwrap_or("label");
        let labels_filter: Option<Vec<u64>> = options.get("labels").map(|s| {
            s.split(',').filter_map(|v| v.trim().parse().ok()).collect()
        });
        let max_partitions: usize = options.get("allowed-partitions")
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);
        let on_undersized = options.get("on-undersized").unwrap_or("error");

        let base_path = resolve_path(base_str, &ctx.workspace);
        let query_path = resolve_path(query_str, &ctx.workspace);
        let metadata_path = resolve_path(metadata_str, &ctx.workspace);

        // Open base vectors
        let base_reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open base: {}", e), start),
        };
        let base_count = VectorReader::<f32>::count(&base_reader);
        let dim = VectorReader::<f32>::dim(&base_reader);

        // Read metadata labels — supports scalar formats (u8, i32, etc.)
        // and slab files (reads first field from each record).
        let labels = match read_metadata_labels(&metadata_path, base_count) {
            Ok(l) => l,
            Err(e) => return error_result(e, start),
        };

        // Build partitions: label → [ordinals]
        let mut partitions: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if let Some(ref filter) = labels_filter {
                if !filter.contains(&label) { continue; }
            }
            partitions.entry(label).or_default().push(i);
        }

        // Check partition count against limit
        if partitions.len() > max_partitions {
            return error_result(format!(
                "{} distinct labels would create {} partitions, exceeding limit of {}. \
                 Use --allowed-partitions={} to override, or --labels to select specific values.",
                partitions.len(), partitions.len(), max_partitions, partitions.len()), start);
        }

        ctx.ui.log(&format!(
            "partition-profiles: {} base vectors, dim={}, k={}",
            base_count, dim, k));
        ctx.ui.log(&format!(
            "  {} unique labels → {} partition profiles (limit: {})",
            partitions.len(), partitions.len(), max_partitions));

        let pb = ctx.ui.bar(partitions.len() as u64, "extracting partitions");
        let mut profiles_created = 0u32;
        let mut profile_names: Vec<(String, usize)> = Vec::new(); // (name, base_count)

        for (&label, ordinals) in &partitions {
            let profile_name = format!("{}-{}", prefix, label);
            let profile_dir = ctx.workspace.join("profiles").join(&profile_name);
            let _ = std::fs::create_dir_all(&profile_dir);

            let partition_size = ordinals.len();
            if partition_size < k {
                match on_undersized {
                    "error" => {
                        return error_result(format!(
                            "label {} has {} matching vectors, fewer than k={}. \
                             Use --on-undersized=warn to skip, or --on-undersized=include to pad.",
                            label, partition_size, k), start);
                    }
                    "warn" => {
                        ctx.ui.log(&format!("  {} ({} vectors) — SKIPPED: fewer than k={}",
                            profile_name, partition_size, k));
                        pb.inc(1);
                        continue;
                    }
                    _ => {} // "include" — proceed
                }
            }
            ctx.ui.log(&format!("  {} ({} vectors)", profile_name, partition_size));

            // Extract partition base vectors
            let part_base_path = profile_dir.join("base_vectors.fvec");
            {
                let mut f = match std::fs::File::create(&part_base_path) {
                    Ok(f) => std::io::BufWriter::new(f),
                    Err(e) => {
                        ctx.ui.log(&format!("    ERROR creating {}: {}", part_base_path.display(), e));
                        continue;
                    }
                };
                let dim_bytes = (dim as i32).to_le_bytes();
                for &ord in ordinals {
                    let slice = base_reader.get_slice(ord);
                    f.write_all(&dim_bytes).unwrap_or(());
                    for &val in slice {
                        f.write_all(&val.to_le_bytes()).unwrap_or(());
                    }
                }
            }

            // Symlink query vectors
            let part_query_path = profile_dir.join("query_vectors.fvec");
            if !part_query_path.exists() {
                let rel = relative_path(&profile_dir, &query_path);
                let _ = std::os::unix::fs::symlink(&rel, &part_query_path);
            }

            profile_names.push((profile_name, partition_size));
            profiles_created += 1;
            pb.inc(1);
        }
        pb.finish();

        // Set variable so pipeline runner knows partitions were created
        let names_csv = profile_names.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(",");
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, "partition_profiles", &names_csv);
        ctx.defaults.insert("partition_profiles".into(), names_csv.clone());
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, "partition_count", &profiles_created.to_string());
        ctx.defaults.insert("partition_count".into(), profiles_created.to_string());

        // Register partition profiles in dataset.yaml so the pipeline runner
        // can expand per_profile templates for them in the re-expansion phase.
        //
        // Uses the DatasetConfig API: load → add profiles → save with
        // canonical ordering. Idempotent — existing profiles with the same
        // name are replaced.
        if profiles_created > 0 {
            let ds_path = ctx.workspace.join("dataset.yaml");
            match vectordata::dataset::DatasetConfig::load(&ds_path) {
                Ok(mut config) => {
                    for (name, count) in &profile_names {
                        use vectordata::dataset::profile::{DSProfile, DSView};
                        use vectordata::dataset::source::{DSSource, DSWindow};

                        let mut views = indexmap::IndexMap::new();
                        views.insert("base_vectors".to_string(), DSView {
                            source: DSSource {
                                path: format!("profiles/{}/base_vectors.fvec", name),
                                namespace: None,
                                window: DSWindow::default(),
                            },
                            window: None,
                        });
                        views.insert("query_vectors".to_string(), DSView {
                            source: DSSource {
                                path: format!("profiles/{}/query_vectors.fvec", name),
                                namespace: None,
                                window: DSWindow::default(),
                            },
                            window: None,
                        });

                        // Inherit maxk from the default profile
                        let maxk = config.default_profile()
                            .and_then(|p| p.maxk)
                            .unwrap_or(k as u32);

                        config.set_profile(name, DSProfile {
                            maxk: Some(maxk),
                            base_count: Some(*count as u64),
                            views,
                        });
                    }

                    if let Err(e) = config.save(&ds_path) {
                        ctx.ui.log(&format!("  WARNING: failed to save dataset.yaml: {}", e));
                    } else {
                        ctx.ui.log(&format!("  registered {} partition profiles in dataset.yaml",
                            profile_names.len()));
                    }
                }
                Err(e) => {
                    ctx.ui.log(&format!("  WARNING: could not load dataset.yaml: {}", e));
                }
            }
        }

        let message = format!(
            "{} partition profiles prepared ({} base vectors extracted)",
            profiles_created, base_count);

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("base", "Path", true, None, "Base vectors file (full dataset)"),
            opt("query", "Path", true, None, "Query vectors file"),
            opt("metadata", "Path", true, None, "Metadata labels file (scalar u8/i32/etc. or slab)"),
            opt("neighbors", "int", true, None, "k — partitions smaller than k are rejected or skipped"),
            opt("metric", "string", true, None, "Distance metric (passed to per-profile compute-knn)"),
            opt("prefix", "string", false, Some("label"), "Profile name prefix"),
            opt("labels", "string", false, None, "Comma-separated label values to partition (default: all)"),
            opt("allowed-partitions", "int", false, Some("50"), "Maximum partition profiles to create"),
            opt("on-undersized", "string", false, Some("error"), "When partition < k vectors: error, warn (skip), include"),
        ]
    }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        role: if type_name == "Path" { OptionRole::Input } else { OptionRole::Config },
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

/// Read metadata label values from a scalar file or slab file.
///
/// Supported formats:
/// - Scalar: `.u8`, `.i8`, `.u16`, `.i16`, `.u32`, `.i32`, `.u64`, `.i64`
///   (flat packed, one element per record)
/// - Slab: `.slab` (reads first i32 field from each record)
///
/// Returns one label value per record. Validates that the record count
/// matches `expected_count`.
fn read_metadata_labels(path: &Path, expected_count: usize) -> Result<Vec<u64>, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    if ext == "slab" {
        return read_slab_labels(path, expected_count);
    }

    // Scalar format
    let elem_size: usize = match ext {
        "u8" | "i8" => 1, "u16" | "i16" => 2,
        "u32" | "i32" => 4, "u64" | "i64" => 8,
        _ => return Err(format!("unsupported metadata format: '{}' — expected scalar (u8, i32, etc.) or slab", ext)),
    };
    let data = std::fs::read(path)
        .map_err(|e| format!("read {}: {}", path.display(), e))?;
    let count = data.len() / elem_size;
    if data.len() % elem_size != 0 {
        return Err(format!("metadata file size {} is not a multiple of element size {}", data.len(), elem_size));
    }
    if count != expected_count {
        return Err(format!("metadata count {} != base count {}", count, expected_count));
    }

    let labels: Vec<u64> = (0..count).map(|i| {
        let off = i * elem_size;
        match elem_size {
            1 => data[off] as u64,
            2 => u16::from_le_bytes([data[off], data[off+1]]) as u64,
            4 => u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]) as u64,
            8 => u64::from_le_bytes(data[off..off+8].try_into().unwrap()),
            _ => 0,
        }
    }).collect();

    Ok(labels)
}

/// Read label values from a slab file.
///
/// Each slab record is a binary-encoded row. The first field is read as
/// an integer label value. For multi-field records, only the first field
/// is used for partitioning.
fn read_slab_labels(path: &Path, expected_count: usize) -> Result<Vec<u64>, String> {
    use slabtastic::{Page, SlabReader};

    let reader = SlabReader::open(path)
        .map_err(|e| format!("open slab {}: {}", path.display(), e))?;

    let total = reader.total_records() as usize;
    if total != expected_count {
        return Err(format!("slab metadata count {} != base count {}", total, expected_count));
    }

    let mut page_entries = reader.page_entries();
    page_entries.sort_by_key(|e| e.start_ordinal);

    let mut labels = Vec::with_capacity(total);

    for entry in &page_entries {
        let buf = reader.page_buf(entry);
        let record_count = Page::record_count_from_buf(buf)
            .map_err(|e| format!("read slab page: {}", e))?;

        for i in 0..record_count {
            let data = Page::get_record_ref_from_buf(buf, i, record_count)
                .map_err(|e| format!("read slab record {}: {}", i, e))?;

            // Parse first field as integer label.
            // Slab records use a length-prefixed binary encoding:
            // [field_len: u16 LE] [field_data: field_len bytes]
            // For integer fields, field_data is the LE-encoded value.
            if data.len() < 2 {
                labels.push(0);
                continue;
            }
            let field_len = u16::from_le_bytes([data[0], data[1]]) as usize;
            if field_len == 0 || data.len() < 2 + field_len {
                labels.push(0);
                continue;
            }
            let field_data = &data[2..2 + field_len];
            let label: u64 = match field_len {
                1 => field_data[0] as u64,
                2 => u16::from_le_bytes([field_data[0], field_data[1]]) as u64,
                4 => u32::from_le_bytes(field_data[..4].try_into().unwrap()) as u64,
                8 => u64::from_le_bytes(field_data[..8].try_into().unwrap()),
                _ => {
                    // String or larger field — hash it for partitioning
                    let mut h: u64 = 0xcbf29ce484222325;
                    for &b in field_data {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    h
                }
            };
            labels.push(label);
        }
    }

    Ok(labels)
}

/// Compute relative path from `from` directory to `to` file.
fn relative_path(from: &Path, to: &Path) -> PathBuf {
    let from_abs = std::fs::canonicalize(from).unwrap_or_else(|_| from.to_path_buf());
    let to_abs = std::fs::canonicalize(to).unwrap_or_else(|_| to.to_path_buf());
    let from_parts: Vec<_> = from_abs.components().collect();
    let to_parts: Vec<_> = to_abs.components().collect();
    let common = from_parts.iter().zip(to_parts.iter())
        .take_while(|(a, b)| a == b).count();
    let mut result = PathBuf::new();
    for _ in common..from_parts.len() { result.push(".."); }
    for part in &to_parts[common..] { result.push(part); }
    result
}

