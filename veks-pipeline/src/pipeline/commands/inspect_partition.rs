// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: explain a partition profile for a single query.
//!
//! Traces how a query's predicate selects base vectors, how those vectors
//! are remapped into the partition's ordinal space, and how the partition
//! KNN relates to the global KNN.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

pub struct ExplainPartitionsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ExplainPartitionsOp)
}

impl CommandOp for ExplainPartitionsOp {
    fn command_path(&self) -> &str {
        "analyze explain-partitions"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Trace a query through partition profile creation and KNN".into(),
            body: format!(
                r#"# analyze explain-partitions

Trace how a query's predicate creates a partition and how KNN results
map between global and partition ordinal spaces.

## Description

For a given query ordinal and label, narrates:

1. **Predicate** — what filter this query applies
2. **Partition membership** — which global base ordinals have this label
3. **Ordinal remapping** — global ordinal → partition ordinal mapping
4. **Global KNN** — unfiltered nearest neighbors (from default profile)
5. **Partition KNN** — nearest neighbors within the partition
6. **Ordinal translation** — partition KNN ordinals mapped back to global
7. **Comparison** — how partition KNN differs from filtered KNN

When run in a directory with `dataset.yaml`, file paths are
auto-resolved from the profile.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    #[allow(unused_assignments)]
    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let ordinal: usize = match options.require("ordinal") {
            Ok(s) => match s.parse() {
                Ok(n) => n,
                _ => return error_result(format!("invalid ordinal: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let label_str = options.get("label");
        let profile_name = options.get("profile").unwrap_or("default");
        let limit: usize = options.get("limit").and_then(|s| s.parse().ok()).unwrap_or(20);
        let prefix = options.get("prefix").unwrap_or("label");

        // Load dataset.yaml for profile resolution
        let ds_path = ctx.workspace.join("dataset.yaml");
        let config = match vectordata::dataset::DatasetConfig::load(&ds_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("load dataset.yaml: {}", e), start),
        };
        let profile = match config.profiles.profiles.get(profile_name) {
            Some(p) => p,
            None => return error_result(format!("profile '{}' not found", profile_name), start),
        };

        // Resolve paths from profile views
        let resolve = |facet: &str| -> Option<PathBuf> {
            profile.view(facet).map(|v| ctx.workspace.join(v.path()))
        };

        let metadata_path = match resolve("metadata_content") {
            Some(p) => p,
            None => return error_result("metadata_content facet not found in profile".into(), start),
        };
        let predicates_path = match resolve("metadata_predicates") {
            Some(p) => p,
            None => return error_result("metadata_predicates facet not found in profile".into(), start),
        };

        // Read metadata labels
        let meta_ext = metadata_path.extension().and_then(|e| e.to_str()).unwrap_or("u8");
        let elem_size: usize = match meta_ext {
            "u8" | "i8" => 1, "u16" | "i16" => 2,
            "u32" | "i32" => 4, "u64" | "i64" => 8,
            _ => return error_result(format!("unsupported metadata format: {}", meta_ext), start),
        };
        let meta_bytes = match std::fs::read(&metadata_path) {
            Ok(b) => b, Err(e) => return error_result(format!("read metadata: {}", e), start),
        };
        let base_count = meta_bytes.len() / elem_size;

        let read_value = |bytes: &[u8], idx: usize, es: usize| -> u64 {
            let off = idx * es;
            match es {
                1 => bytes[off] as u64,
                2 => u16::from_le_bytes([bytes[off], bytes[off+1]]) as u64,
                4 => u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as u64,
                8 => u64::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3],
                    bytes[off+4], bytes[off+5], bytes[off+6], bytes[off+7]]),
                _ => 0,
            }
        };

        // Read predicate value for this query
        let pred_bytes = match std::fs::read(&predicates_path) {
            Ok(b) => b, Err(e) => return error_result(format!("read predicates: {}", e), start),
        };
        let pred_ext = predicates_path.extension().and_then(|e| e.to_str()).unwrap_or("u8");
        let pred_elem = match pred_ext {
            "u8" | "i8" => 1usize, "u16" | "i16" => 2,
            "u32" | "i32" => 4, "u64" | "i64" => 8,
            _ => 1,
        };
        let pred_val = read_value(&pred_bytes, ordinal, pred_elem);

        // Determine which label to explain
        let label = if let Some(l) = label_str {
            l.parse::<u64>().unwrap_or(pred_val)
        } else {
            pred_val
        };

        // ════════════════════════════════════════════════════════════════
        ctx.ui.log(&format!(
            "\n═══ Query {} — Partition Trace (label={}) ══════════════════",
            ordinal, label));

        // Stage 1: Predicate
        ctx.ui.log("\n┌─ Stage 1: Predicate ─────────────────────────────────────");
        ctx.ui.log(&format!("│  query[{}] predicate: field_0 == {}", ordinal, pred_val));
        if label != pred_val {
            ctx.ui.log(&format!("│  (explaining label {} instead of predicate {})", label, pred_val));
        }

        // Stage 2: Partition membership
        ctx.ui.log("│");
        ctx.ui.log("├─ Stage 2: Partition Membership ──────────────────────────");

        let mut partition_ordinals: Vec<usize> = Vec::new();
        for i in 0..base_count {
            if read_value(&meta_bytes, i, elem_size) == label {
                partition_ordinals.push(i);
            }
        }

        let selectivity = partition_ordinals.len() as f64 / base_count as f64;
        ctx.ui.log(&format!("│  {} of {} base vectors have label={} ({:.2}%)",
            partition_ordinals.len(), base_count, label, selectivity * 100.0));

        let show = partition_ordinals.len().min(limit);
        ctx.ui.log(&format!("│  global ordinals: [{}{}]",
            partition_ordinals[..show].iter()
                .map(|o| o.to_string()).collect::<Vec<_>>().join(", "),
            if partition_ordinals.len() > show {
                format!(", ... ({} more)", partition_ordinals.len() - show)
            } else { String::new() }
        ));

        // Stage 3: Ordinal remapping
        ctx.ui.log("│");
        ctx.ui.log("├─ Stage 3: Ordinal Remapping ─────────────────────────────");

        let global_to_partition: HashMap<usize, usize> = partition_ordinals.iter()
            .enumerate()
            .map(|(pi, &gi)| (gi, pi))
            .collect();

        let remap_show = partition_ordinals.len().min(8);
        for i in 0..remap_show {
            ctx.ui.log(&format!("│    global[{:>6}] → partition[{:>4}]",
                partition_ordinals[i], i));
        }
        if partition_ordinals.len() > remap_show {
            ctx.ui.log(&format!("│    ... ({} more mappings)", partition_ordinals.len() - remap_show));
        }

        // Stage 4: Global KNN (from default profile)
        ctx.ui.log("│");
        ctx.ui.log("├─ Stage 4: Global KNN (default profile) ─────────────────");

        let _global_gt: Vec<i32> = if let Some(gt_path) = resolve("neighbor_indices") {
            match MmapVectorReader::<i32>::open_ivec(&gt_path) {
                Ok(reader) if ordinal < VectorReader::<i32>::count(&reader) => {
                    let slice = reader.get_slice(ordinal);
                    let k = slice.len();
                    ctx.ui.log(&format!("│  k={} neighbors for query {}", k, ordinal));

                    let partition_set: HashSet<usize> = partition_ordinals.iter().copied().collect();
                    let show_gt = k.min(limit);
                    for i in 0..show_gt {
                        let ord = slice[i];
                        let in_partition = partition_set.contains(&(ord as usize));
                        let marker = if in_partition { "●" } else { "·" };
                        let remap = if in_partition {
                            format!(" → partition[{}]", global_to_partition[&(ord as usize)])
                        } else { String::new() };
                        ctx.ui.log(&format!("│    [{:>3}] global {:>6}  {}{}",
                            i, ord, marker, remap));
                    }
                    if k > show_gt {
                        ctx.ui.log(&format!("│    ... ({} more)", k - show_gt));
                    }

                    let pass_count = slice.iter()
                        .filter(|&&o| o >= 0 && partition_set.contains(&(o as usize)))
                        .count();
                    ctx.ui.log(&format!("│  {} of {} global neighbors are in this partition ({:.1}%)",
                        pass_count, k, 100.0 * pass_count as f64 / k as f64));

                    slice.to_vec()
                }
                _ => {
                    ctx.ui.log("│  (neighbor_indices not available)");
                    vec![]
                }
            }
        } else {
            ctx.ui.log("│  (neighbor_indices not in profile)");
            vec![]
        };

        // Stage 5: Partition KNN
        let partition_profile = format!("{}-{}", prefix, label);
        ctx.ui.log("│");
        ctx.ui.log(&format!("├─ Stage 5: Partition KNN (profile: {}) ─────────────────",
            partition_profile));

        let part_gt_path = ctx.workspace.join(format!("profiles/{}/neighbor_indices.ivec", partition_profile));
        let part_dist_path = ctx.workspace.join(format!("profiles/{}/neighbor_distances.fvec", partition_profile));

        let partition_neighbors: Vec<i32>;
        let partition_distances: Vec<f32>;

        if part_gt_path.exists() {
            match MmapVectorReader::<i32>::open_ivec(&part_gt_path) {
                Ok(reader) if ordinal < VectorReader::<i32>::count(&reader) => {
                    let slice = reader.get_slice(ordinal);
                    partition_neighbors = slice.to_vec();

                    // Read distances if available
                    partition_distances = if part_dist_path.exists() {
                        match MmapVectorReader::<f32>::open_fvec(&part_dist_path) {
                            Ok(dr) if ordinal < VectorReader::<f32>::count(&dr) => {
                                dr.get_slice(ordinal).to_vec()
                            }
                            _ => vec![],
                        }
                    } else { vec![] };

                    let real_count = partition_neighbors.iter().filter(|&&o| o >= 0).count();
                    ctx.ui.log(&format!("│  {} neighbors ({} real, {} padding)",
                        slice.len(), real_count, slice.len() - real_count));

                    let show_pn = real_count.min(limit);
                    for i in 0..show_pn {
                        let p_ord = partition_neighbors[i];
                        if p_ord < 0 { break; }
                        let g_ord = partition_ordinals[p_ord as usize];
                        let dist_str = partition_distances.get(i)
                            .map(|d| format!("  dist={:.6}", d))
                            .unwrap_or_default();
                        ctx.ui.log(&format!("│    [{:>3}] partition[{:>4}] → global[{:>6}]{}",
                            i, p_ord, g_ord, dist_str));
                    }
                    if real_count > show_pn {
                        ctx.ui.log(&format!("│    ... ({} more)", real_count - show_pn));
                    }
                }
                _ => {
                    ctx.ui.log("│  (partition KNN not available — run compute partition-profiles first)");
                    partition_neighbors = vec![];
                    partition_distances = vec![];
                }
            }
        } else {
            ctx.ui.log(&format!("│  partition profile '{}' not found at {}", partition_profile,
                part_gt_path.display()));
            ctx.ui.log("│  run: veks pipeline compute partition-profiles ...");
            partition_neighbors = vec![];
            partition_distances = vec![];
        }

        // Stage 6: Comparison with filtered KNN
        ctx.ui.log("│");
        ctx.ui.log("└─ Stage 6: Partition vs Filtered KNN ────────────────────");

        let filtered_path = resolve("filtered_neighbor_indices");
        if let Some(fp) = filtered_path {
            if fp.exists() {
                match MmapVectorReader::<i32>::open_ivec(&fp) {
                    Ok(reader) if ordinal < VectorReader::<i32>::count(&reader) => {
                        let filtered = reader.get_slice(ordinal);
                        let filtered_set: HashSet<i32> = filtered.iter()
                            .filter(|&&o| o >= 0).copied().collect();

                        // Map partition neighbors back to global ordinals
                        let partition_global: Vec<i32> = partition_neighbors.iter()
                            .filter(|&&o| o >= 0)
                            .map(|&o| partition_ordinals[o as usize] as i32)
                            .collect();
                        let partition_set: HashSet<i32> = partition_global.iter().copied().collect();

                        let intersection = filtered_set.intersection(&partition_set).count();
                        let filtered_real = filtered.iter().filter(|&&o| o >= 0).count();
                        let partition_real = partition_global.len();

                        ctx.ui.log(&format!("   filtered KNN:    {} neighbors (global ordinals)",
                            filtered_real));
                        ctx.ui.log(&format!("   partition KNN:   {} neighbors (remapped to global)",
                            partition_real));
                        ctx.ui.log(&format!("   intersection:    {} neighbors", intersection));

                        if filtered_real > 0 && partition_real > 0 {
                            let recall = intersection as f64 / filtered_real.min(partition_real) as f64;
                            ctx.ui.log(&format!("   overlap:         {:.1}%", recall * 100.0));
                        }

                        // Show the rank mapping for shared neighbors
                        if intersection > 0 {
                            ctx.ui.log("");
                            ctx.ui.log("   rank comparison (shared neighbors):");

                            let filtered_rank: HashMap<i32, usize> = filtered.iter().enumerate()
                                .filter(|(_, o)| **o >= 0)
                                .map(|(i, o)| (*o, i))
                                .collect();

                            let mut shared: Vec<(i32, usize, usize)> = Vec::new();
                            for (pi, &g_ord) in partition_global.iter().enumerate() {
                                if let Some(&fi) = filtered_rank.get(&g_ord) {
                                    shared.push((g_ord, pi, fi));
                                }
                            }
                            shared.sort_by_key(|s| s.1);

                            let show_shared = shared.len().min(limit);
                            for &(g_ord, p_rank, f_rank) in shared.iter().take(show_shared) {
                                let shift = p_rank as i32 - f_rank as i32;
                                let arrow = if shift == 0 { " =" }
                                    else if shift < 0 { " ↑" }
                                    else { " ↓" };
                                ctx.ui.log(&format!("     global {:>6}: partition[{}] vs filtered[{}] ({}{:+})",
                                    g_ord, p_rank, f_rank, arrow, shift));
                            }
                            if shared.len() > show_shared {
                                ctx.ui.log(&format!("     ... ({} more)", shared.len() - show_shared));
                            }

                            let avg_shift: f64 = shared.iter()
                                .map(|s| (s.1 as i32 - s.2 as i32).abs() as f64)
                                .sum::<f64>() / shared.len() as f64;
                            ctx.ui.log(&format!("   avg |rank shift|: {:.1}", avg_shift));
                        }

                        // Unique to each
                        let only_partition: Vec<i32> = partition_global.iter()
                            .filter(|o| !filtered_set.contains(o))
                            .copied().collect();
                        let only_filtered: Vec<i32> = filtered.iter()
                            .filter(|&&o| o >= 0 && !partition_set.contains(&o))
                            .copied().collect();

                        if !only_partition.is_empty() || !only_filtered.is_empty() {
                            ctx.ui.log("");
                            if !only_partition.is_empty() {
                                ctx.ui.log(&format!("   {} neighbors unique to partition KNN",
                                    only_partition.len()));
                            }
                            if !only_filtered.is_empty() {
                                ctx.ui.log(&format!("   {} neighbors unique to filtered KNN",
                                    only_filtered.len()));
                            }
                        }
                    }
                    _ => ctx.ui.log("   (filtered KNN not readable)"),
                }
            } else {
                ctx.ui.log("   (filtered_neighbor_indices not available)");
            }
        } else {
            ctx.ui.log("   (filtered_neighbor_indices not in profile)");
        }

        ctx.ui.log("");

        CommandResult {
            status: Status::Ok,
            message: format!("query {} partition trace for label={} ({} vectors in partition)",
                ordinal, label, partition_ordinals.len()),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("ordinal", "int", true, None, "Query ordinal to explain"),
            opt("label", "string", false, None, "Label value to trace (default: query's predicate value)"),
            opt("profile", "string", false, Some("default"), "Profile for global KNN/metadata resolution"),
            opt("prefix", "string", false, Some("label"), "Partition profile name prefix"),
            opt("limit", "int", false, Some("20"), "Max entries to display per stage"),
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
        role: OptionRole::Config,
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
