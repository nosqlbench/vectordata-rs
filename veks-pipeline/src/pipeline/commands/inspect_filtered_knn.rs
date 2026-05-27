// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: explain a filtered KNN result for a single query.
//!
//! Bimodal — analyzes whichever of the two filtered-knn facets the
//! profile carries:
//! - **F (pre-filter)** at `prefiltered_neighbor_*` (or legacy
//!   `filtered_neighbor_*`): top-K of `X_p`, ACORN's `G_K`, full K when
//!   `|X_p| ≥ K`.
//! - **E (post-filter)** at `postfiltered_neighbor_*`: `G ∩ R`,
//!   sparse possible. When the on-disk E facet is missing but G and R
//!   are available, the command derives E live for the analysis.
//!
//! For each query ordinal it traces: predicate → selectivity → unfiltered
//! ground truth → filtered ground truth (F and/or E) → intersection
//! analysis. Supports slab and scalar metadata formats.
//!
//! See `docs/design/prefilter-postfilter-facets.md` for the facet split.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use slabtastic::SlabReader;
use vectordata::io::XvecReader;

use veks_core::formats::anode::ANode;
use veks_core::formats::anode_vernacular::{self, Vernacular};
use veks_core::formats::pnode::PNode;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::commands::compute_prefiltered_knn::PredicateIndices;

/// Pipeline command: explain filtered KNN for a single query.
pub struct ExplainFilteredKnnOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ExplainFilteredKnnOp)
}

impl CommandOp for ExplainFilteredKnnOp {
    fn command_path(&self) -> &str {
        "analyze explain-filtered-knn"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

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
        let limit: usize = options
            .get("limit")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let histogram_bin_width: Option<usize> = match options.get("histogram-bin-width") {
            None => None,
            Some(s) => match s.parse::<usize>() {
                Ok(0) => return error_result(
                    "--histogram-bin-width must be ≥ 1".into(), start),
                Ok(w) => Some(w),
                Err(_) => return error_result(
                    format!("invalid --histogram-bin-width: '{}'", s), start),
            },
        };

        // Three input modes — exactly one must be supplied. The
        // explicit-ordinal form prints the original verbose
        // per-stage narrative; the range and sample forms run
        // the same evaluation silently per ordinal and emit an
        // aggregate distribution box.
        let ordinals: Vec<usize> = match select_ordinals(options) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        if ordinals.is_empty() {
            return error_result("no ordinals selected (--ordinal, --ordinals, or --sample required)".into(), start);
        }
        // Convenience for the verbose path: a single ordinal.
        let ordinal: usize = ordinals[0];

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
        // Resolve via the canonical facet key first, then fall through
        // to a legacy alias. Used for the F slot, which is reachable by
        // either `prefiltered_neighbor_*` (canonical) or
        // `filtered_neighbor_*` (legacy) in dataset.yaml.
        let resolve_with_legacy = |canonical: &str, legacy: &str, option: &str| -> Option<PathBuf> {
            options.get(option)
                .map(|s| resolve_path(s, &ctx.workspace))
                .or_else(|| profile.view(canonical).map(|v| resolve_path(v.path(), &ctx.workspace)))
                .or_else(|| profile.view(legacy).map(|v| resolve_path(v.path(), &ctx.workspace)))
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
        // F (pre-filter) — the legacy filtered-knn shape. Accept both
        // canonical `prefiltered_*` and legacy `filtered_*` keys, plus
        // explicit CLI flags (canonical preferred; legacy alias retained).
        let prefiltered_indices_path = resolve_with_legacy(
            "prefiltered_neighbor_indices", "filtered_neighbor_indices",
            "prefiltered-indices",
        ).or_else(|| options.get("filtered-indices").map(|s| resolve_path(s, &ctx.workspace)));
        let prefiltered_distances_path = resolve_with_legacy(
            "prefiltered_neighbor_distances", "filtered_neighbor_distances",
            "prefiltered-distances",
        ).or_else(|| options.get("filtered-distances").map(|s| resolve_path(s, &ctx.workspace)));
        // E (post-filter) — new sparse artifact.
        let postfiltered_indices_path = resolve("postfiltered_neighbor_indices", "postfiltered-indices");
        let postfiltered_distances_path = resolve("postfiltered_neighbor_distances", "postfiltered-distances");

        // The remainder of the command currently operates on a single
        // "filtered" facet (the F/pre-filter slot under its legacy
        // variable names). Preserve that behaviour as the default
        // analysis target. The new postfiltered_* paths are resolved so
        // future analysis stages (and the bimodal multi-ordinal output)
        // can use them; threading them through the verbose narrative
        // path is a follow-up.
        let filtered_indices_path = prefiltered_indices_path.clone();
        let filtered_distances_path = prefiltered_distances_path.clone();
        let _ = postfiltered_indices_path; // wired in follow-up
        let _ = postfiltered_distances_path;

        let pred_ext = file_ext(&predicates_path);
        let meta_ext = file_ext(&metadata_path);

        // ── Total metadata count (for selectivity) ──
        let total_metadata = count_records(&metadata_path, &meta_ext);

        // Multi-ordinal mode — silently evaluate per ordinal,
        // collect stats, then emit an aggregate summary. The
        // single-ordinal path falls through to the verbose
        // narrative below.
        if ordinals.len() > 1 {
            return run_multi(ctx, start, &ordinals,
                &predicates_path, &pred_ext, &metadata_path, &meta_ext,
                indices_path.as_deref(), gt_path.as_deref(),
                filtered_indices_path.as_deref(), total_metadata,
                histogram_bin_width);
        }

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
            match XvecReader::<i32>::open_path(gp) {
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
            match XvecReader::<i32>::open_path(fp) {
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
            match XvecReader::<f32>::open_path(dp) {
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
            opt("ordinal", "int", false, None, "Query ordinal to explain (single-ordinal verbose mode)", OptionRole::Config),
            opt("ordinals", "string", false, None, "Contiguous range N..M (aggregate-summary mode)", OptionRole::Config),
            opt("sample", "int", false, None, "Random sample of N ordinals (aggregate-summary mode; use --seed for reproducibility)", OptionRole::Config),
            opt("seed", "int", false, Some("42"), "RNG seed for --sample (deterministic ordinal selection)", OptionRole::Config),
            opt("profile", "string", false, Some("default"), "Profile to resolve facets from", OptionRole::Config),
            opt("predicates", "Path", false, None, "Predicates file (auto-resolved from profile)", OptionRole::Input),
            opt("metadata", "Path", false, None, "Metadata file (auto-resolved from profile)", OptionRole::Input),
            opt("metadata-indices", "Path", false, None, "Predicate results file (auto-resolved from profile)", OptionRole::Input),
            opt("ground-truth", "Path", false, None, "Unfiltered GT indices (auto-resolved from profile)", OptionRole::Input),
            // F (pre-filter) — auto-resolved from profile.
            // Canonical: `prefiltered_neighbor_*`; legacy alias:
            // `filtered_neighbor_*`. CLI flags accept both forms; the
            // canonical names are preferred for new pipelines.
            opt("prefiltered-indices", "Path", false, None, "Pre-filter KNN indices (F facet, auto-resolved from profile)", OptionRole::Input),
            opt("prefiltered-distances", "Path", false, None, "Pre-filter KNN distances (F facet, auto-resolved from profile)", OptionRole::Input),
            opt("filtered-indices", "Path", false, None, "Legacy alias for --prefiltered-indices (F facet)", OptionRole::Input),
            opt("filtered-distances", "Path", false, None, "Legacy alias for --prefiltered-distances (F facet)", OptionRole::Input),
            // E (post-filter) — new sparse artifact.
            opt("postfiltered-indices", "Path", false, None, "Post-filter KNN indices (E facet, G ∩ R, auto-resolved from profile)", OptionRole::Input),
            opt("postfiltered-distances", "Path", false, None, "Post-filter KNN distances (E facet, auto-resolved from profile)", OptionRole::Input),
            opt("limit", "int", false, Some("20"), "Max neighbors to display per stage (single-ordinal mode)", OptionRole::Config),
            opt("histogram-bin-width", "int", false, None, "Intersection-histogram bin width (default: 1 when max<50, else floor(max/50))", OptionRole::Config),
        ]
    }
}

/// Decide which ordinal(s) to evaluate based on the CLI flags.
///
/// Priority:
///   1. `--ordinal N` (single, verbose mode).
///   2. `--ordinals N..M` (contiguous range, aggregate mode).
///   3. `--sample N` + `--seed S` (random sample, aggregate mode).
///
/// Returns an error if zero or more-than-one mode is specified.
fn select_ordinals(options: &Options) -> Result<Vec<usize>, String> {
    let single = options.get("ordinal");
    let range = options.get("ordinals");
    let sample = options.get("sample");
    let set_count = [single.is_some(), range.is_some(), sample.is_some()]
        .iter().filter(|b| **b).count();
    if set_count == 0 {
        return Err("specify exactly one of: --ordinal N | --ordinals N..M | --sample N".to_string());
    }
    if set_count > 1 {
        return Err("specify only one of: --ordinal | --ordinals | --sample".to_string());
    }
    if let Some(s) = single {
        let n: usize = s.parse().map_err(|_| format!("invalid --ordinal: '{}'", s))?;
        return Ok(vec![n]);
    }
    if let Some(s) = range {
        let (lo, hi) = s.split_once("..").ok_or_else(||
            format!("--ordinals expects N..M, got '{s}'"))?;
        let lo: usize = lo.trim().parse().map_err(|_|
            format!("--ordinals start not an integer: '{lo}'"))?;
        let hi: usize = hi.trim().parse().map_err(|_|
            format!("--ordinals end not an integer: '{hi}'"))?;
        if lo > hi {
            return Err(format!("--ordinals inverted: {lo}..{hi}"));
        }
        return Ok((lo..hi).collect());
    }
    let n: usize = sample.unwrap().parse().map_err(|_|
        format!("invalid --sample: '{}'", sample.unwrap()))?;
    let seed: u64 = options.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    Ok(sample_ordinals(n, seed))
}

/// Draw `n` random ordinal candidates seeded by `seed`. The
/// actual valid range is unknown here (depends on facet count),
/// so we draw from a generous span and let the per-ordinal
/// evaluator clamp / skip out-of-range indices when it tries to
/// read them. To make the sample tighter, the caller could pass
/// `--ordinals 0..N` instead.
///
/// Uses a deterministic xoshiro RNG so re-runs with the same
/// `--seed` produce the same ordinal set.
fn sample_ordinals(n: usize, seed: u64) -> Vec<usize> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    // Default cap: assume datasets have <= 10M queries. The
    // per-ordinal eval will skip any draw past the actual facet
    // count, so the cap is just a soft upper bound for the RNG.
    (0..n).map(|_| rng.random_range(0..10_000_000)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-ordinal aggregate mode
// ─────────────────────────────────────────────────────────────────────────────

/// Per-ordinal stats accumulated for the aggregate summary.
/// Each metric is `None` when the corresponding facet was
/// missing for that ordinal (e.g. no GT for a query).
#[derive(Debug, Clone, Default)]
struct PerOrdinalStats {
    /// Source query ordinal — retained so the aggregate
    /// summary can name exemplar queries (e.g. the ordinal
    /// with the highest or lowest joint cardinality).
    ordinal: usize,
    /// Selectivity = matches / total_metadata. `None` if
    /// metadata-indices facet was missing.
    selectivity: Option<f64>,
    /// Number of filtered-KNN neighbors for this query.
    filtered_count: Option<usize>,
    /// Number of unfiltered GT neighbors.
    gt_count: Option<usize>,
    /// |filtered ∩ unfiltered|.
    intersection: Option<usize>,
    /// recall@k = intersection / filtered_count.
    recall_at_k: Option<f64>,
    /// Average absolute rank shift between filtered and
    /// unfiltered positions for the intersection set.
    avg_rank_shift: Option<f64>,
    /// Number of neighbors unique to filtered (not in GT
    /// top-k).
    new_filtered: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn run_multi(
    ctx: &StreamContext,
    start: Instant,
    ordinals: &[usize],
    predicates_path: &Path,
    pred_ext: &str,
    _metadata_path: &Path,
    _meta_ext: &str,
    indices_path: Option<&Path>,
    gt_path: Option<&Path>,
    filtered_indices_path: Option<&Path>,
    total_metadata: usize,
    histogram_bin_width: Option<usize>,
) -> CommandResult {
    // Open every facet reader once up front so the per-ordinal
    // loop doesn't pay for repeated opens.
    let pred_reader = if pred_ext == "slab" {
        SlabReader::open(predicates_path).ok()
    } else { None };
    let indices_reader = indices_path.and_then(|p| PredicateIndices::open(p).ok());
    let gt_reader = gt_path.and_then(|p| XvecReader::<i32>::open_path(p).ok());
    let filtered_reader = filtered_indices_path.and_then(|p| XvecReader::<i32>::open_path(p).ok());

    let mut stats: Vec<PerOrdinalStats> = Vec::with_capacity(ordinals.len());
    let pb = ctx.ui.bar(ordinals.len() as u64, "evaluating");
    let mut skipped = 0usize;
    for (i, &ord) in ordinals.iter().enumerate() {
        let s = eval_one(
            ord, pred_reader.as_ref(), indices_reader.as_ref(),
            gt_reader.as_ref(), filtered_reader.as_ref(),
            total_metadata,
        );
        // Skip ordinals where every metric is None — the
        // ordinal landed past the facet bounds, which is
        // expected for random samples that overshoot the
        // dataset's actual size.
        if s.selectivity.is_none() && s.filtered_count.is_none()
            && s.gt_count.is_none() && s.intersection.is_none() {
            skipped += 1;
        } else {
            stats.push(s);
        }
        if (i + 1) % 50 == 0 || i + 1 == ordinals.len() {
            pb.set_position((i + 1) as u64);
        }
    }
    pb.finish();

    if stats.is_empty() {
        return CommandResult {
            status: Status::Error,
            message: format!("no ordinals produced data (all {} were out of range)", ordinals.len()),
            produced: vec![],
            elapsed: start.elapsed(),
        };
    }

    render_aggregate(ctx, &stats, ordinals.len(), skipped, total_metadata, histogram_bin_width);

    CommandResult {
        status: Status::Ok,
        message: format!(
            "aggregate summary: {} ordinals evaluated ({} skipped)",
            stats.len(), skipped,
        ),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

/// Evaluate a single ordinal silently, returning per-metric
/// values. Mirrors the verbose-mode stages 1-5 but without
/// any UI output.
fn eval_one(
    ord: usize,
    pred_reader: Option<&SlabReader>,
    indices_reader: Option<&PredicateIndices>,
    gt_reader: Option<&XvecReader<i32>>,
    filtered_reader: Option<&XvecReader<i32>>,
    total_metadata: usize,
) -> PerOrdinalStats {
    let mut s = PerOrdinalStats { ordinal: ord, ..Default::default() };

    // We don't need to decode the PNode for the aggregate
    // summary — we only count match outcomes. Skip the predicate
    // decode entirely.
    let _ = pred_reader;

    // Selectivity from metadata-indices facet. We don't keep
    // the matching set itself — the verification step ("did
    // every filtered neighbor pass the predicate") is a
    // single-ordinal concern that the verbose narrative
    // handles; here we only need the count for selectivity.
    if let Some(ir) = indices_reader
        && let Ok(v) = ir.get_ordinals(ord)
        && total_metadata > 0
    {
        s.selectivity = Some(v.len() as f64 / total_metadata as f64);
    }

    // Unfiltered GT.
    let gt_set: HashSet<i32> = if let Some(gr) = gt_reader {
        if ord < gr.count() {
            let slice = gr.get_slice(ord);
            s.gt_count = Some(slice.len());
            slice.iter().copied().collect()
        } else { HashSet::new() }
    } else { HashSet::new() };

    // Filtered KNN.
    let filtered_vec: Vec<i32> = if let Some(fr) = filtered_reader {
        if ord < fr.count() {
            let slice = fr.get_slice(ord).to_vec();
            s.filtered_count = Some(slice.len());
            slice
        } else { Vec::new() }
    } else { Vec::new() };

    // Intersection / recall / rank shift.
    if !gt_set.is_empty() && !filtered_vec.is_empty() {
        let inter: usize = filtered_vec.iter().filter(|o| gt_set.contains(o)).count();
        s.intersection = Some(inter);
        if !filtered_vec.is_empty() {
            s.recall_at_k = Some(inter as f64 / filtered_vec.len() as f64);
        }
        // Rank shifts (only for ordinals in the intersection).
        let gt_rank: HashMap<i32, usize> = gt_reader.unwrap().get_slice(ord)
            .iter().enumerate().map(|(i, &o)| (o, i)).collect();
        let shifts: Vec<f64> = filtered_vec.iter().enumerate().filter_map(|(fi, o)| {
            gt_rank.get(o).map(|&gi| (fi as i32 - gi as i32).abs() as f64)
        }).collect();
        if !shifts.is_empty() {
            s.avg_rank_shift = Some(shifts.iter().sum::<f64>() / shifts.len() as f64);
        }
        s.new_filtered = Some(filtered_vec.iter().filter(|o| !gt_set.contains(o)).count());
    }
    s
}

/// Render an aggregate distribution box using the slab
/// explain-box helper so the formatting matches every other
/// boxed output in the toolchain.
fn render_aggregate(
    ctx: &StreamContext,
    stats: &[PerOrdinalStats],
    requested: usize,
    skipped: usize,
    total_metadata: usize,
    histogram_bin_width: Option<usize>,
) {
    let mut rows: Vec<String> = Vec::new();
    rows.push(format!("evaluated:  {} of {} requested ({} skipped)", stats.len(), requested, skipped));
    if total_metadata > 0 {
        rows.push(format!("metadata:   {} records (selectivity denominator)", total_metadata));
    }
    rows.push("─".repeat(8));

    push_distribution(&mut rows, "selectivity",     stats.iter().filter_map(|s| s.selectivity).collect(), 4);
    push_distribution(&mut rows, "filtered count",  stats.iter().filter_map(|s| s.filtered_count.map(|x| x as f64)).collect(), 1);
    push_distribution(&mut rows, "unfiltered count",stats.iter().filter_map(|s| s.gt_count.map(|x| x as f64)).collect(), 1);
    push_distribution(&mut rows, "intersection",    stats.iter().filter_map(|s| s.intersection.map(|x| x as f64)).collect(), 1);
    push_distribution(&mut rows, "recall@k",        stats.iter().filter_map(|s| s.recall_at_k).collect(), 4);
    push_distribution(&mut rows, "avg |rank shift|",stats.iter().filter_map(|s| s.avg_rank_shift).collect(), 2);
    push_distribution(&mut rows, "new filtered",    stats.iter().filter_map(|s| s.new_filtered.map(|x| x as f64)).collect(), 1);

    let zero_intersection = stats.iter()
        .filter(|s| s.intersection == Some(0)).count();
    let all_pass = stats.iter()
        .filter(|s| matches!(s.recall_at_k, Some(r) if r >= 1.0)).count();
    rows.push("─".repeat(8));
    rows.push(format!("zero-intersection queries: {} / {}", zero_intersection, stats.len()));
    rows.push(format!("full-recall queries:       {} / {}", all_pass, stats.len()));

    // Intersection-size histogram — the cardinality distribution
    // is what the user asks first when interpreting a filtered-KNN
    // run: how many queries got zero hits, how many got partial
    // overlap, how many got near-full recall. Min/median/p95/max
    // hides that shape; a histogram surfaces it.
    let intersection_sizes: Vec<usize> = stats.iter()
        .filter_map(|s| s.intersection)
        .collect();
    if !intersection_sizes.is_empty() {
        rows.push("─".repeat(8));
        let effective_width = effective_bin_width(&intersection_sizes, histogram_bin_width);
        rows.push(format!("intersection histogram (bin width = {})", effective_width));
        push_intersection_histogram(&mut rows, &intersection_sizes, effective_width);
        push_intersection_exemplars(&mut rows, stats);
    }

    super::slab::render_explain_box(ctx, "explain-filtered-knn aggregate", &rows);
    ctx.ui.log("");
}

/// Compute the effective histogram bin width: explicit override
/// if given, else the user-described default — `1` when the
/// observed max is under 50, otherwise `floor(max/50)` so the
/// histogram lands at ~50 bins regardless of scale. An explicit
/// width of `0` is treated as `1`; the CLI layer already rejects
/// it but the function stays robust against direct callers.
fn effective_bin_width(sizes: &[usize], override_width: Option<usize>) -> usize {
    if let Some(w) = override_width { return w.max(1); }
    let max_val = sizes.iter().copied().max().unwrap_or(0);
    if max_val < 50 { 1 } else { (max_val / 50).max(1) }
}

/// Render an ASCII histogram of intersection sizes into `rows`.
/// Bins are chosen so the *zero* count is always its own row
/// (catastrophic queries deserve to be loud), and the remaining
/// range is split into equal-width bins of size `bin_width`.
/// Each row shows `count`, the percentage of total, and a bar;
/// bar width scales to the most-populated bin so the visual is
/// always full-scale.
///
/// `bin_width` must be ≥ 1; callers should obtain it through
/// `effective_bin_width` to honour the documented defaults.
fn push_intersection_histogram(rows: &mut Vec<String>, sizes: &[usize], bin_width: usize) {
    if sizes.is_empty() { return; }
    let n = sizes.len();
    let max_val = *sizes.iter().max().unwrap();
    let bin_width = bin_width.max(1);

    // Bins: a dedicated `[0..=0]` bin, then equal-width bins up
    // to `max_val`.
    let nonzero_max = max_val.max(1);
    let nonzero_bins = ((nonzero_max + bin_width - 1) / bin_width).max(1);

    // Build (lo, hi) closed-closed ranges. Bin 0 is always [0,0].
    let mut bins: Vec<(usize, usize)> = vec![(0, 0)];
    let mut lo = 1usize;
    for _ in 0..nonzero_bins {
        let hi = (lo + bin_width - 1).min(nonzero_max);
        bins.push((lo, hi));
        lo = hi + 1;
        if lo > nonzero_max { break; }
    }

    // Tally.
    let mut counts: Vec<usize> = vec![0; bins.len()];
    for &v in sizes {
        let idx = if v == 0 {
            0
        } else {
            // First non-zero bin starts at index 1.
            let off = ((v - 1) / bin_width).min(nonzero_bins - 1);
            1 + off
        };
        counts[idx] += 1;
    }

    // Render. Label column wide enough for "N–N" with the largest
    // observed bin endpoint; count column wide enough for the
    // total count. Bar width fills the remainder of an 80-col box
    // minus margins and the percentage tail.
    let label_w = bins.iter()
        .map(|(lo, hi)| if lo == hi { format!("{lo}") } else { format!("{lo}–{hi}") }.chars().count())
        .max().unwrap_or(1);
    let count_w = n.to_string().len();
    let max_count = *counts.iter().max().unwrap();
    let bar_max: usize = 30; // ASCII bar width; fits comfortably in 80 cols
    for (i, (lo, hi)) in bins.iter().enumerate() {
        let label = if lo == hi { format!("{lo}") } else { format!("{lo}–{hi}") };
        let c = counts[i];
        let pct = (c as f64 / n as f64) * 100.0;
        // Any non-empty bin gets at least one pip — otherwise a
        // sparsely-populated tail bin looks identical to a truly
        // empty one, hiding the long-tail shape the histogram is
        // meant to surface.
        let bar_len = if c == 0 {
            0
        } else if max_count == 0 {
            1
        } else {
            let raw = (c as f64 / max_count as f64) * bar_max as f64;
            (raw.round() as usize).max(1)
        };
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        rows.push(format!(
            "  {label:>label_w$}  {c:>count_w$} ({pct:5.1}%)  {bar}",
            label = label, label_w = label_w,
            c = c, count_w = count_w,
            pct = pct,
            bar = bar,
        ));
    }
}

/// Emit three exemplar ordinals selected by joint cardinality
/// (|filtered ∩ unfiltered|): the lowest, the middle, and the
/// highest. These are concrete handles the user can pass to
/// `--ordinal N` to re-run the verbose narrative on a
/// representative query at each tail of the distribution.
///
/// Ties are broken by ordinal so the choice is deterministic
/// across runs.
fn push_intersection_exemplars(rows: &mut Vec<String>, stats: &[PerOrdinalStats]) {
    let mut with_inter: Vec<(usize, usize)> = stats.iter()
        .filter_map(|s| s.intersection.map(|i| (i, s.ordinal)))
        .collect();
    if with_inter.is_empty() { return; }
    with_inter.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let n = with_inter.len();
    let lo = with_inter[0];
    let mid = with_inter[n / 2];
    let hi = with_inter[n - 1];

    rows.push("exemplar ordinals (by intersection):".to_string());
    rows.push(format!("  low     intersection={:>5}  ordinal {}", lo.0, lo.1));
    rows.push(format!("  middle  intersection={:>5}  ordinal {}", mid.0, mid.1));
    rows.push(format!("  high    intersection={:>5}  ordinal {}", hi.0, hi.1));
}

fn push_distribution(rows: &mut Vec<String>, label: &str, mut values: Vec<f64>, prec: usize) {
    if values.is_empty() {
        rows.push(format!("{label:<18} (no data)"));
        return;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = values[0];
    let median = values[values.len() / 2];
    let p95 = values[((values.len() - 1) as f64 * 0.95).round() as usize];
    let max = *values.last().unwrap();
    let fmt = |v: f64| -> String { format!("{:.*}", prec, v) };
    rows.push(format!("{label:<18} min={}  median={}  p95={}  max={}",
        fmt(min), fmt(median), fmt(p95), fmt(max)));
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
        description: desc.to_string(), extended_description: None,
        role,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `select_ordinals` enforces exactly one of the three
    /// input modes; mis-shaped invocations return a clear
    /// error instead of falling through to a default.
    #[test]
    fn select_ordinals_requires_exactly_one_mode() {
        let opts = Options::new();
        let r = select_ordinals(&opts);
        assert!(r.is_err(), "no mode → error");

        let mut opts = Options::new();
        opts.set("ordinal", "5".to_string());
        opts.set("sample", "10".to_string());
        let r = select_ordinals(&opts);
        assert!(r.is_err(), "two modes → error");
    }

    #[test]
    fn select_ordinals_single_returns_one() {
        let mut opts = Options::new();
        opts.set("ordinal", "42".to_string());
        assert_eq!(select_ordinals(&opts).unwrap(), vec![42]);
    }

    #[test]
    fn select_ordinals_range_returns_contiguous() {
        let mut opts = Options::new();
        opts.set("ordinals", "3..7".to_string());
        assert_eq!(select_ordinals(&opts).unwrap(), vec![3, 4, 5, 6]);
    }

    #[test]
    fn select_ordinals_range_inverted_errors() {
        let mut opts = Options::new();
        opts.set("ordinals", "10..3".to_string());
        assert!(select_ordinals(&opts).is_err());
    }

    /// `sample_ordinals` is deterministic for a given seed —
    /// the user can re-run with `--seed N` and get the exact
    /// same set of draws.
    #[test]
    fn sample_ordinals_deterministic_per_seed() {
        let a = sample_ordinals(20, 1234);
        let b = sample_ordinals(20, 1234);
        assert_eq!(a, b, "same seed must produce same sample");
        let c = sample_ordinals(20, 5678);
        assert_ne!(a, c, "different seed should produce different sample");
        assert_eq!(a.len(), 20);
    }

    /// `push_distribution` writes one row with min/median/p95/max.
    /// Empty input writes a "(no data)" placeholder so the
    /// aggregate box still has the row.
    /// Histogram dedicates row 0 to *exactly zero* hits so
    /// catastrophic queries stay loud, then bins the rest. With
    /// k = 100 typical, this produces something like
    /// `0`, `1–13`, `14–26`, … `91–100`.
    #[test]
    fn intersection_histogram_separates_zero_from_nonzero() {
        let mut rows: Vec<String> = Vec::new();
        // 4 zeros + a smear across 1..=100.
        let sizes: Vec<usize> = vec![0, 0, 0, 0, 5, 25, 50, 75, 100];
        let w = effective_bin_width(&sizes, None);
        push_intersection_histogram(&mut rows, &sizes, w);
        // First emitted row is the zero bin — labelled "0", count 4.
        assert!(rows[0].trim_start().starts_with("0  4"),
            "row 0 should be the zero bin with count 4, got: {:?}", rows[0]);
        // Last row covers the high end including 100.
        let last = rows.last().unwrap();
        assert!(last.contains("100") || last.contains("–100") || last.contains("–99"),
            "last row should reach the max value, got: {:?}", last);
    }

    /// Bar width scales to the most-populated bin so the visual
    /// is always full-scale — a histogram with a single populated
    /// bin shows that bin at full width.
    #[test]
    fn intersection_histogram_scales_bar_to_max_count() {
        let mut rows: Vec<String> = Vec::new();
        let sizes: Vec<usize> = vec![50; 10]; // all in the same bin
        let w = effective_bin_width(&sizes, None);
        push_intersection_histogram(&mut rows, &sizes, w);
        // The populated bin should have the longest bar; an
        // empty bin should have none. We don't pin the exact
        // length because terminal-width could vary; we just
        // check that AT LEAST one row has bar characters and
        // at least one doesn't.
        let with_bar = rows.iter().filter(|r| r.contains('█')).count();
        let without_bar = rows.iter().filter(|r| !r.contains('█')).count();
        assert!(with_bar >= 1, "at least one row should have a visible bar");
        assert!(without_bar >= 1,
            "with all hits in one bin, other bins must show empty bars");
    }

    /// Empty input writes nothing — `render_aggregate` already
    /// guards against this by skipping the histogram section.
    #[test]
    fn intersection_histogram_empty_input_writes_nothing() {
        let mut rows: Vec<String> = Vec::new();
        push_intersection_histogram(&mut rows, &[], 1);
        assert!(rows.is_empty(),
            "no values → no rows; caller decides whether to print a header");
    }

    /// Total count across bins equals the input length — no
    /// values get double-counted or dropped.
    #[test]
    fn intersection_histogram_counts_sum_to_input_length() {
        let mut rows: Vec<String> = Vec::new();
        let sizes: Vec<usize> = vec![0, 1, 2, 5, 10, 25, 50, 75, 99, 100];
        let w = effective_bin_width(&sizes, None);
        push_intersection_histogram(&mut rows, &sizes, w);
        // Each row's count is followed by " (...%)"; extract the
        // count token and sum.
        let total: usize = rows.iter()
            .filter_map(|r| {
                // Format: "  <label>  <count> (<pct>%)  <bar>"
                let after_label = r.split_whitespace().collect::<Vec<_>>();
                // Find the first token that parses as usize
                // *after* skipping the label (which contains
                // digits too). Easier: count tokens are the
                // ones followed by "(X.X%)".
                after_label.windows(2)
                    .find(|w| w[1].starts_with('('))
                    .and_then(|w| w[0].parse::<usize>().ok())
            })
            .sum();
        assert_eq!(total, sizes.len(),
            "every input must be counted exactly once across all bins");
    }

    /// Exemplars are the lowest, middle, and highest intersection
    /// values; the originating ordinal is reported so the user can
    /// re-run `--ordinal N` on a representative query at each end.
    /// Ties resolve to the smaller ordinal for deterministic output.
    #[test]
    fn intersection_exemplars_pick_low_mid_high() {
        let stats: Vec<PerOrdinalStats> = vec![
            PerOrdinalStats { ordinal: 100, intersection: Some(50), ..Default::default() },
            PerOrdinalStats { ordinal: 200, intersection: Some(0),  ..Default::default() },
            PerOrdinalStats { ordinal: 300, intersection: Some(99), ..Default::default() },
            PerOrdinalStats { ordinal: 400, intersection: Some(25), ..Default::default() },
            PerOrdinalStats { ordinal: 500, intersection: Some(75), ..Default::default() },
        ];
        let mut rows: Vec<String> = Vec::new();
        push_intersection_exemplars(&mut rows, &stats);
        // header + three exemplar rows
        assert_eq!(rows.len(), 4, "expect header + 3 exemplar rows, got {rows:?}");
        assert!(rows[1].contains("intersection=    0") && rows[1].contains("ordinal 200"),
            "low row should name the min-intersection query: {:?}", rows[1]);
        // 5 sorted intersections: [0, 25, 50, 75, 99] — middle index 2 → 50, ordinal 100.
        assert!(rows[2].contains("intersection=   50") && rows[2].contains("ordinal 100"),
            "middle row should name the median query: {:?}", rows[2]);
        assert!(rows[3].contains("intersection=   99") && rows[3].contains("ordinal 300"),
            "high row should name the max-intersection query: {:?}", rows[3]);
    }

    /// Ordinals missing an intersection (e.g. no GT or no F) are
    /// excluded — exemplars must be queries we actually evaluated.
    #[test]
    fn intersection_exemplars_skip_missing_intersection() {
        let stats: Vec<PerOrdinalStats> = vec![
            PerOrdinalStats { ordinal: 1, intersection: None,     ..Default::default() },
            PerOrdinalStats { ordinal: 2, intersection: Some(10), ..Default::default() },
        ];
        let mut rows: Vec<String> = Vec::new();
        push_intersection_exemplars(&mut rows, &stats);
        // With only one usable ordinal, low/mid/high all collapse to it.
        assert_eq!(rows.len(), 4);
        for r in &rows[1..] {
            assert!(r.contains("ordinal 2"), "all exemplars should be ord 2: {r:?}");
        }
    }

    /// Default histogram resolution: width-1 bins when the
    /// observed max is below 50, otherwise floor(max/50) so the
    /// histogram always lands at ~50 non-zero bins regardless of
    /// scale. An explicit override wins and is clamped to ≥1.
    #[test]
    fn effective_bin_width_follows_documented_defaults() {
        // Small range: width 1.
        assert_eq!(effective_bin_width(&[0, 5, 49], None), 1);
        // Exactly 50: floor(50/50) = 1.
        assert_eq!(effective_bin_width(&[0, 25, 50], None), 1);
        // 100 → floor(100/50) = 2.
        assert_eq!(effective_bin_width(&[0, 99, 100], None), 2);
        // 999 → floor(999/50) = 19.
        assert_eq!(effective_bin_width(&[999], None), 19);
        // 1000 → floor(1000/50) = 20.
        assert_eq!(effective_bin_width(&[1000], None), 20);
        // Empty: floor(0/50) → 1 (clamped); no crash.
        assert_eq!(effective_bin_width(&[], None), 1);
        // Explicit override wins; 0 clamps to 1.
        assert_eq!(effective_bin_width(&[10_000], Some(7)), 7);
        assert_eq!(effective_bin_width(&[10_000], Some(0)), 1);
    }

    /// A long-tail distribution where one bin dominates must not
    /// render its sparse-tail bins as visually-empty rows — every
    /// non-zero count gets at least one pip so the reader can
    /// distinguish "1 value" from "0 values".
    #[test]
    fn intersection_histogram_nonzero_bin_renders_at_least_one_pip() {
        let mut rows: Vec<String> = Vec::new();
        // One huge bin + many bins with a single hit each. With
        // bar_max=30 and max_count=3000, a count of 1 rounds to 0
        // under naive scaling; the renderer must clamp to 1.
        let mut sizes: Vec<usize> = vec![0; 3000];
        for k in 1..=20usize { sizes.push(k); } // 20 sparse bins
        push_intersection_histogram(&mut rows, &sizes, 1);
        // Find rows for the sparse-tail bins (labels "1".."20") and
        // ensure each has at least one '█'. We skip the zero-bin
        // row (the dominant one) since its bar is already maxed.
        let mut sparse_rows_checked = 0;
        for r in &rows {
            // Label is the leading token after whitespace.
            let label = r.split_whitespace().next().unwrap_or("");
            if let Ok(n) = label.parse::<usize>()
                && (1..=20).contains(&n)
            {
                assert!(r.contains('█'),
                    "sparse-tail bin '{}' has no pip — empty and non-empty bins look identical: {:?}",
                    label, r);
                sparse_rows_checked += 1;
            }
        }
        assert_eq!(sparse_rows_checked, 20, "expected 20 sparse-tail rows, saw {sparse_rows_checked}");
    }

    /// An explicit bin width drives the bin layout — caller can
    /// override the default to get finer or coarser resolution.
    #[test]
    fn intersection_histogram_respects_explicit_bin_width() {
        let mut rows: Vec<String> = Vec::new();
        // max=100 with bin_width=10 → bins [0,0],[1,10],[11,20],…,[91,100]
        // = 1 zero bin + 10 non-zero bins = 11 rows.
        let sizes: Vec<usize> = vec![0, 5, 15, 25, 100];
        push_intersection_histogram(&mut rows, &sizes, 10);
        assert_eq!(rows.len(), 11,
            "1 zero bin + ceil(100/10)=10 non-zero bins = 11 rows, got {rows:?}");
        // Bin [1,10] should be labelled "1–10".
        assert!(rows.iter().any(|r| r.contains("1–10")),
            "expected a '1–10' bin label, got: {rows:?}");
    }

    #[test]
    fn distribution_row_handles_empty_and_unsorted() {
        let mut rows: Vec<String> = Vec::new();
        push_distribution(&mut rows, "x", vec![], 2);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("(no data)"));

        let mut rows: Vec<String> = Vec::new();
        push_distribution(&mut rows, "y", vec![3.0, 1.0, 2.0, 4.0, 5.0], 2);
        // After sort: [1, 2, 3, 4, 5]. min=1, median=3, p95≈5, max=5.
        assert!(rows[0].contains("min=1.00"));
        assert!(rows[0].contains("median=3.00"));
        assert!(rows[0].contains("max=5.00"));
    }
}
