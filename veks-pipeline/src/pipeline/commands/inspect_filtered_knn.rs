// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: explain a filtered KNN result for a single query.
//!
//! Traces through every stage of the filtered KNN pipeline for a given
//! query ordinal: predicate → selectivity → unfiltered GT → filtered GT →
//! intersection analysis. Supports slab and scalar metadata formats.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use slabtastic::SlabReader;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use veks_core::formats::anode::ANode;
use veks_core::formats::anode_vernacular::{self, Vernacular};
use veks_core::formats::pnode::PNode;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::commands::compute_filtered_knn::PredicateIndices;

/// Pipeline command: explain filtered KNN for a single query.
pub struct ExplainFilteredKnnOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ExplainFilteredKnnOp)
}

impl CommandOp for ExplainFilteredKnnOp {
    fn command_path(&self) -> &str {
        "analyze explain-filtered-knn"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Trace a filtered KNN result through every pipeline stage".into(),
            body: format!(
                r#"# analyze explain-filtered-knn

Trace a filtered KNN result through every pipeline stage.

## Description

For a given query ordinal, narrates the full filtered KNN pipeline:

1. **Predicate** — what filter this query applies
2. **Selectivity** — how many base vectors pass the filter (from R)
3. **Unfiltered KNN** — global ground truth neighbors (from G)
4. **Filtered KNN** — neighbors after predicate filtering (from F)
5. **Intersection** — overlap between filtered and unfiltered top-k
6. **Rank shift** — how filtering changed neighbor rankings and distances

When run in a directory with `dataset.yaml`, all file paths are
auto-resolved from the profile. Use `--profile` to select a non-default
profile.

## Options

{}

## Notes

- The ordinal indexes into the query vectors / predicates.
- Unfiltered GT (G) is optional — when absent, the intersection analysis
  is skipped and only the filtered results are shown.
- Distance values are read from `filtered_neighbor_distances` (D facet)
  when available.

## Oracle Partitions

When oracle partition profiles exist (O facet), each partition contains
only the queries whose predicate matches that partition's label. A query
with predicate `field_0 == 5` appears only in the `label_05` partition,
not in the other partitions. This means the partition's KNN result files
are indexed by partition query ordinal, not global query ordinal.

Use `analyze explain-partitions --ordinal N` to trace the full
global → partition query mapping and see the partition-specific KNN.
"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let profile_name = options.get("profile").unwrap_or("default");
        let ordinal: usize = match options.require("ordinal") {
            Ok(s) => match s.parse() {
                Ok(n) => n,
                _ => return error_result(format!("invalid ordinal: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let limit: usize = options
            .get("limit")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);

        // Resolve paths from explicit options or dataset.yaml profile
        let ds_path = ctx.workspace.join("dataset.yaml");
        let config = match vectordata::dataset::DatasetConfig::load(&ds_path) {
            Ok(c) => c,
            Err(_) => {
                // No dataset.yaml — all paths must be explicit
                return error_result(
                    "no dataset.yaml found; provide all paths explicitly".into(), start);
            }
        };
        let profile = match config.profiles.profiles.get(profile_name) {
            Some(p) => p,
            None => return error_result(
                format!("profile '{}' not found in dataset.yaml", profile_name), start),
        };

        let resolve = |facet: &str, option: &str| -> Option<PathBuf> {
            options.get(option)
                .map(|s| resolve_path(s, &ctx.workspace))
                .or_else(|| profile.view(facet).map(|v| resolve_path(v.path(), &ctx.workspace)))
        };

        let predicates_path = match resolve("metadata_predicates", "predicates") {
            Some(p) => p,
            None => return error_result("cannot resolve predicates path".into(), start),
        };
        let metadata_path = match resolve("metadata_content", "metadata") {
            Some(p) => p,
            None => return error_result("cannot resolve metadata path".into(), start),
        };
        let indices_path = resolve("metadata_indices", "metadata-indices");
        let gt_path = resolve("neighbor_indices", "ground-truth");
        let filtered_indices_path = resolve("filtered_neighbor_indices", "filtered-indices");
        let filtered_distances_path = resolve("filtered_neighbor_distances", "filtered-distances");

        let pred_ext = file_ext(&predicates_path);
        let meta_ext = file_ext(&metadata_path);

        // ── Total metadata count (for selectivity) ──
        let total_metadata = count_records(&metadata_path, &meta_ext);

        // ════════════════════════════════════════════════════════════════
        // Stage 1: Predicate
        // ════════════════════════════════════════════════════════════════
        ctx.ui.log(&format!(
            "\n═══ Query ordinal {} ═══════════════════════════════════════",
            ordinal));

        // Check for oracle partition profiles — if they exist, note that
        // within each partition only the queries matching that partition's
        // label are evaluated.
        let has_partitions = config.profiles.profiles.iter()
            .any(|(_, p)| p.partition);
        if has_partitions {
            ctx.ui.log("");
            ctx.ui.log("  ℹ Oracle partition profiles exist for this dataset.");
            ctx.ui.log("    Within each partition, only queries whose predicate");
            ctx.ui.log("    matches the partition label are included in KNN.");
            ctx.ui.log("    Use `analyze explain-partitions` for partition details.");
        }

        ctx.ui.log("\n┌─ Stage 1: Predicate ─────────────────────────────────────");
        if pred_ext == "slab" {
            match SlabReader::open(&predicates_path) {
                Ok(reader) => match reader.get(ordinal as i64) {
                    Ok(bytes) => match PNode::from_bytes_named(&bytes) {
                        Ok(pnode) => {
                            let rendered = anode_vernacular::render(
                                &ANode::PNode(pnode), Vernacular::Readout);
                            ctx.ui.log(&format!("│  {}", rendered));
                        }
                        Err(e) => ctx.ui.log(&format!("│  decode error: {}", e)),
                    },
                    Err(e) => ctx.ui.log(&format!("│  read error: {}", e)),
                },
                Err(e) => ctx.ui.log(&format!("│  open error: {}", e)),
            }
        } else {
            match read_scalar_value(&predicates_path, ordinal) {
                Ok(val) => ctx.ui.log(&format!("│  field_0 == {}", val)),
                Err(e) => ctx.ui.log(&format!("│  read error: {}", e)),
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Stage 2: Selectivity (R facet)
        // ════════════════════════════════════════════════════════════════
        ctx.ui.log("│");
        ctx.ui.log("├─ Stage 2: Selectivity ───────────────────────────────────");

        let matching_ordinals: Vec<i32> = if let Some(ref keys_path) = indices_path {
            match PredicateIndices::open(keys_path) {
                Ok(reader) => match reader.get_ordinals(ordinal) {
                    Ok(ords) => ords,
                    Err(e) => {
                        ctx.ui.log(&format!("│  read error: {}", e));
                        vec![]
                    }
                },
                Err(e) => {
                    ctx.ui.log(&format!("│  open error: {}", e));
                    vec![]
                }
            }
        } else {
            ctx.ui.log("│  (metadata_indices not available — skipping)");
            vec![]
        };

        let selectivity = if total_metadata > 0 {
            matching_ordinals.len() as f64 / total_metadata as f64
        } else { 0.0 };

        if !matching_ordinals.is_empty() {
            ctx.ui.log(&format!("│  {} of {} base vectors pass filter",
                matching_ordinals.len(), total_metadata));
            ctx.ui.log(&format!("│  selectivity: {:.6} ({:.2}%)",
                selectivity, selectivity * 100.0));
            ctx.ui.log(&format!("│  1/selectivity: {:.1}× reduction",
                if selectivity > 0.0 { 1.0 / selectivity } else { f64::INFINITY }));
        }

        let matching_set: HashSet<i32> = matching_ordinals.iter().copied().collect();

        // ════════════════════════════════════════════════════════════════
        // Stage 3: Unfiltered Ground Truth (G facet)
        // ════════════════════════════════════════════════════════════════
        ctx.ui.log("│");
        ctx.ui.log("├─ Stage 3: Unfiltered Ground Truth (G) ──────────────────");

        let gt_neighbors: Vec<i32> = if let Some(ref gp) = gt_path {
            match MmapVectorReader::<i32>::open_ivec(gp) {
                Ok(reader) => {
                    if ordinal < reader.count() {
                        let slice = reader.get_slice(ordinal);
                        ctx.ui.log(&format!("│  k={} neighbors for query {}", slice.len(), ordinal));
                        // Show first few
                        let show = slice.len().min(limit);
                        for i in 0..show {
                            let ord = slice[i];
                            let passes = matching_set.contains(&ord);
                            let marker = if passes { "✓" } else { "·" };
                            ctx.ui.log(&format!("│    [{}] ordinal {:>8}  {}", i, ord, marker));
                        }
                        if slice.len() > show {
                            ctx.ui.log(&format!("│    ... and {} more", slice.len() - show));
                        }
                        // Count how many unfiltered GT pass the predicate
                        let pass_count = slice.iter().filter(|o| matching_set.contains(o)).count();
                        ctx.ui.log(&format!("│  {} of {} unfiltered neighbors pass predicate ({:.1}%)",
                            pass_count, slice.len(), 100.0 * pass_count as f64 / slice.len().max(1) as f64));
                        slice.to_vec()
                    } else {
                        ctx.ui.log(&format!("│  ordinal {} out of range (count={})", ordinal, reader.count()));
                        vec![]
                    }
                }
                Err(e) => {
                    ctx.ui.log(&format!("│  open error: {}", e));
                    vec![]
                }
            }
        } else {
            ctx.ui.log("│  (neighbor_indices not available — skipping)");
            vec![]
        };

        // ════════════════════════════════════════════════════════════════
        // Stage 4: Filtered KNN Results (F facet)
        // ════════════════════════════════════════════════════════════════
        ctx.ui.log("│");
        ctx.ui.log("├─ Stage 4: Filtered KNN Results (F) ─────────────────────");

        let filtered_neighbors: Vec<i32>;
        let filtered_distances: Vec<f32>;

        if let Some(ref fp) = filtered_indices_path {
            match MmapVectorReader::<i32>::open_ivec(fp) {
                Ok(reader) => {
                    if ordinal < reader.count() {
                        let slice = reader.get_slice(ordinal);
                        filtered_neighbors = slice.to_vec();
                        ctx.ui.log(&format!("│  k={} filtered neighbors for query {}", slice.len(), ordinal));
                    } else {
                        ctx.ui.log(&format!("│  ordinal {} out of range (count={})", ordinal, reader.count()));
                        filtered_neighbors = vec![];
                    }
                }
                Err(e) => {
                    ctx.ui.log(&format!("│  open error: {}", e));
                    filtered_neighbors = vec![];
                }
            }
        } else {
            ctx.ui.log("│  (filtered_neighbor_indices not available — skipping)");
            filtered_neighbors = vec![];
        }

        // Read distances if available
        if let Some(ref dp) = filtered_distances_path {
            match MmapVectorReader::<f32>::open_fvec(dp) {
                Ok(reader) => {
                    if ordinal < reader.count() {
                        filtered_distances = reader.get_slice(ordinal).to_vec();
                    } else {
                        filtered_distances = vec![];
                    }
                }
                Err(_) => { filtered_distances = vec![]; }
            }
        } else {
            filtered_distances = vec![];
        }

        // Display filtered results with distances and metadata values
        let show = filtered_neighbors.len().min(limit);
        for i in 0..show {
            let ord = filtered_neighbors[i];
            let dist_str = filtered_distances.get(i)
                .map(|d| format!("  dist={:.6}", d))
                .unwrap_or_default();
            let meta_str = read_scalar_value(&metadata_path, ord as usize)
                .map(|v| format!("  meta={}", v))
                .unwrap_or_default();
            ctx.ui.log(&format!("│    [{}] ordinal {:>8}{}{}", i, ord, dist_str, meta_str));
        }
        if filtered_neighbors.len() > show {
            ctx.ui.log(&format!("│    ... and {} more", filtered_neighbors.len() - show));
        }

        // Verify all filtered neighbors pass the predicate
        if !matching_set.is_empty() && !filtered_neighbors.is_empty() {
            let violations: Vec<i32> = filtered_neighbors.iter()
                .filter(|o| !matching_set.contains(o))
                .copied()
                .collect();
            if violations.is_empty() {
                ctx.ui.log("│  ✓ all filtered neighbors pass the predicate");
            } else {
                ctx.ui.log(&format!("│  ✗ {} filtered neighbors FAIL the predicate: {:?}",
                    violations.len(), &violations[..violations.len().min(5)]));
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Stage 5: Intersection Analysis
        // ════════════════════════════════════════════════════════════════
        if !gt_neighbors.is_empty() && !filtered_neighbors.is_empty() {
            ctx.ui.log("│");
            ctx.ui.log("└─ Stage 5: Intersection Analysis ────────────────────────");

            let gt_set: HashSet<i32> = gt_neighbors.iter().copied().collect();
            let filtered_set: HashSet<i32> = filtered_neighbors.iter().copied().collect();
            let intersection: HashSet<i32> = gt_set.intersection(&filtered_set).copied().collect();

            ctx.ui.log(&format!("   unfiltered GT:     {} neighbors", gt_neighbors.len()));
            ctx.ui.log(&format!("   filtered GT:       {} neighbors", filtered_neighbors.len()));
            ctx.ui.log(&format!("   intersection:      {} neighbors", intersection.len()));

            let recall_at_k = if !filtered_neighbors.is_empty() {
                intersection.len() as f64 / filtered_neighbors.len() as f64
            } else { 0.0 };
            ctx.ui.log(&format!("   recall@{} (filtered in unfiltered): {:.4} ({:.1}%)",
                filtered_neighbors.len().min(gt_neighbors.len()),
                recall_at_k, recall_at_k * 100.0));

            // Rank shift: for each neighbor in both sets, show its rank in
            // unfiltered vs filtered
            let gt_rank: HashMap<i32, usize> = gt_neighbors.iter().enumerate()
                .map(|(i, &o)| (o, i)).collect();

            let mut shifts: Vec<(i32, usize, usize, i32)> = Vec::new(); // (ord, filtered_rank, gt_rank, shift)
            for (fi, &ord) in filtered_neighbors.iter().enumerate() {
                if let Some(&gi) = gt_rank.get(&ord) {
                    let shift = fi as i32 - gi as i32;
                    shifts.push((ord, fi, gi, shift));
                }
            }
            if !shifts.is_empty() {
                ctx.ui.log("");
                ctx.ui.log("   rank shifts (filtered rank vs unfiltered rank):");
                let show_shifts = shifts.len().min(limit);
                for &(ord, fi, gi, shift) in shifts.iter().take(show_shifts) {
                    let arrow = if shift == 0 { "  =" }
                        else if shift < 0 { " ↑" }
                        else { " ↓" };
                    ctx.ui.log(&format!("     ordinal {:>8}:  filtered[{}] ← unfiltered[{}]  ({}{:+})",
                        ord, fi, gi, arrow, shift));
                }
                if shifts.len() > show_shifts {
                    ctx.ui.log(&format!("     ... and {} more", shifts.len() - show_shifts));
                }

                let avg_shift: f64 = shifts.iter().map(|s| s.3.abs() as f64).sum::<f64>() / shifts.len() as f64;
                ctx.ui.log(&format!("   avg |rank shift|: {:.1}", avg_shift));

                // New neighbors: in filtered but not in unfiltered GT
                let new_count = filtered_neighbors.iter()
                    .filter(|o| !gt_set.contains(o))
                    .count();
                if new_count > 0 {
                    ctx.ui.log(&format!("   {} neighbors unique to filtered results (not in unfiltered top-{})",
                        new_count, gt_neighbors.len()));
                }
            }
        } else if filtered_neighbors.is_empty() && gt_neighbors.is_empty() {
            ctx.ui.log("│");
            ctx.ui.log("└─ (no intersection analysis — neither G nor F available)");
        } else {
            ctx.ui.log("│");
            ctx.ui.log("└─ (intersection analysis requires both G and F facets)");
        }

        ctx.ui.log("");

        let message = format!(
            "query {} explained: selectivity {:.4}, {} filtered neighbors",
            ordinal, selectivity, filtered_neighbors.len(),
        );

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("ordinal", "int", true, None, "Query ordinal to explain", OptionRole::Config),
            opt("profile", "string", false, Some("default"), "Profile to resolve facets from", OptionRole::Config),
            opt("predicates", "Path", false, None, "Predicates file (auto-resolved from profile)", OptionRole::Input),
            opt("metadata", "Path", false, None, "Metadata file (auto-resolved from profile)", OptionRole::Input),
            opt("metadata-indices", "Path", false, None, "Predicate results file (auto-resolved from profile)", OptionRole::Input),
            opt("ground-truth", "Path", false, None, "Unfiltered GT indices (auto-resolved from profile)", OptionRole::Input),
            opt("filtered-indices", "Path", false, None, "Filtered KNN indices (auto-resolved from profile)", OptionRole::Input),
            opt("filtered-distances", "Path", false, None, "Filtered KNN distances (auto-resolved from profile)", OptionRole::Input),
            opt("limit", "int", false, Some("20"), "Max neighbors to display per stage", OptionRole::Config),
        ]
    }
}

/// Count records in a metadata file.
fn count_records(path: &Path, ext: &str) -> usize {
    let elem_size: usize = match ext {
        "u8" | "i8" => 1,
        "u16" | "i16" => 2,
        "u32" | "i32" => 4,
        "u64" | "i64" => 8,
        "slab" => {
            return SlabReader::open(path)
                .map(|r| r.total_records() as usize)
                .unwrap_or(0);
        }
        _ => return 0,
    };
    std::fs::metadata(path)
        .map(|m| m.len() as usize / elem_size)
        .unwrap_or(0)
}

/// Read a single scalar value at the given ordinal.
fn read_scalar_value(path: &Path, ordinal: usize) -> Result<String, String> {
    use std::io::{Read, Seek, SeekFrom};

    let ext = file_ext(path);
    let (elem_size, signed) = match ext.as_str() {
        "u8" => (1, false),
        "i8" => (1, true),
        "u16" => (2, false),
        "i16" => (2, true),
        "u32" => (4, false),
        "i32" => (4, true),
        "u64" => (8, false),
        "i64" => (8, true),
        _ => return Err(format!("unsupported scalar extension '.{}'", ext)),
    };

    let offset = ordinal as u64 * elem_size as u64;
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    f.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("seek to ordinal {}: {}", ordinal, e))?;

    let mut buf = [0u8; 8];
    f.read_exact(&mut buf[..elem_size])
        .map_err(|e| format!("read ordinal {}: {}", ordinal, e))?;

    let val = match (elem_size, signed) {
        (1, false) => format!("{}", buf[0]),
        (1, true) => format!("{}", buf[0] as i8),
        (2, false) => format!("{}", u16::from_le_bytes([buf[0], buf[1]])),
        (2, true) => format!("{}", i16::from_le_bytes([buf[0], buf[1]])),
        (4, false) => format!("{}", u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])),
        (4, true) => format!("{}", i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])),
        (8, false) => format!("{}", u64::from_le_bytes(buf)),
        (8, true) => format!("{}", i64::from_le_bytes(buf)),
        _ => unreachable!(),
    };

    Ok(val)
}

fn file_ext(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
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

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        role,
    }
}
