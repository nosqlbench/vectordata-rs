// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: aggregate-statistics summary across a
//! predicate slab and its matching-ordinal indices.
//!
//! Companion to `analyze explain-predicates` (which renders a
//! single predicate) — this command renders the *fleet*: how
//! many predicates ran, what fraction of metadata they matched,
//! whether the observed selectivity lined up with the
//! generation-time target (when a `:schema` sidecar is
//! present), and what the per-field / per-op distribution
//! looks like.
//!
//! Input shape:
//!   - `--predicates predicates.slab` — PNode records.
//!   - `--metadata-indices indices.slab` — per-predicate ivec/
//!     slab of matching metadata ordinals, as produced by
//!     `compute evaluate-predicates`.
//!   - `--metadata metadata.slab` (optional) — used only for
//!     the total record count; with no metadata file we fall
//!     back to the predicate slab's `PredicateSchema.count`
//!     or the maximum ordinal observed across all indices.
//!
//! Selectivity = matches / metadata_total per predicate. The
//! summary reports min / median / p95 / max plus a target-vs-
//! observed line when the schema sidecar carries a target.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use slabtastic::SlabReader;

use veks_core::formats::pnode::{FieldRef, OpType, PNode};

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc,
    Status, StreamContext, render_options_table,
};

use super::compute_prefiltered_knn::PredicateIndices;

pub struct AnalyzePredicateSummaryOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzePredicateSummaryOp)
}

impl CommandOp for AnalyzePredicateSummaryOp {
    fn command_path(&self) -> &str {
        "analyze predicate-summary"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Aggregate stats across a predicate slab and its evaluation indices".into(),
            body: format!(
                r#"# analyze predicate-summary

Aggregate statistics across a predicate slab and the matching
metadata indices produced by `compute evaluate-predicates`.

## What it shows

- Total predicates, total metadata records (when known),
  total match pairs.
- Per-predicate match-count distribution (min / median / p95 /
  max) and the corresponding selectivity range.
- Predicates that matched zero records and predicates that
  matched every record (the two ends of "useless for
  benchmarking").
- Per-field and per-op breakdown of where matches came from.
- When the predicate slab carries a `PredicateSchema` sidecar
  in its `:schema` namespace (the standard since we started
  writing it), the summary compares the observed selectivity
  to the schema's target and flags drift.

## How it works

The command reads the predicate slab record-by-record, decodes
each PNode (named-typed format), counts matches by reading the
corresponding record from `--metadata-indices`, and aggregates
into a single summary box. Evaluation itself is NOT re-run —
this is a read-only summary of a pre-computed answer key.

## Options

{}
"#,
                render_options_table(&options),
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let predicates_path = match options.require("predicates") {
            Ok(s) => resolve(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let indices_path = match options.require("metadata-indices") {
            Ok(s) => resolve(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let metadata_path = options.get("metadata").map(|s| resolve(s, &ctx.workspace));

        // Open the predicate slab — we read PNode records from the
        // default namespace and the optional :schema sidecar.
        let pred_reader = match SlabReader::open(&predicates_path) {
            Ok(r) => r,
            Err(e) => return error_result(
                format!("open predicates {}: {}", predicates_path.display(), e),
                start,
            ),
        };
        let n_preds = pred_reader.total_records() as usize;

        // Open indices.
        let indices = match PredicateIndices::open(&indices_path) {
            Ok(r) => r,
            Err(e) => return error_result(
                format!("open metadata-indices {}: {}", indices_path.display(), e),
                start,
            ),
        };
        let n_index_records = indices.count();
        if n_index_records != n_preds {
            return error_result(format!(
                "predicate count ({n_preds}) != metadata-indices count ({n_index_records}); \
                 indices must come from the same predicate set"),
                start,
            );
        }

        // Optional :schema sidecar carrying the generation
        // template / target sel / count. Absent or unparseable →
        // skip target-comparison rows but still emit the rest.
        let schema = open_predicate_schema(&predicates_path);

        // Total metadata records — used as the denominator for
        // selectivity. Priority: explicit --metadata, then
        // schema record count, then the max ordinal observed +1
        // (rough estimate; suitable for the summary row).
        let metadata_total: u64 = if let Some(p) = metadata_path.as_ref() {
            SlabReader::open(p).map(|r| r.total_records() as u64).unwrap_or(0)
        } else if let Some(s) = schema.as_ref().and_then(|s| s.count.into()) {
            // PredicateSchema.count is the predicate count, not
            // the metadata count — but if the schema doesn't
            // carry a metadata-count field we can't use it.
            // Reserved for a future schema-side metadata_count
            // field. For now, return 0 and let the renderer
            // suppress selectivity stats when total is unknown.
            let _ = s;
            0
        } else {
            0
        };

        // Walk predicates + indices once, computing per-record
        // match counts and per-field / per-op tallies.
        let mut match_counts: Vec<usize> = Vec::with_capacity(n_preds);
        let mut by_field: BTreeMap<String, FieldStats> = BTreeMap::new();
        let mut by_op: BTreeMap<String, OpStats> = BTreeMap::new();
        let mut zero_match = 0usize;
        let mut full_match = 0usize;
        let mut total_matches: u64 = 0;

        for i in 0..n_preds {
            let pnode = match pred_reader.get(i as i64) {
                Ok(bytes) => match PNode::from_bytes_named(&bytes) {
                    Ok(p) => p,
                    Err(e) => return error_result(
                        format!("decode predicate at ordinal {i}: {e}"),
                        start,
                    ),
                },
                Err(e) => return error_result(
                    format!("read predicate at ordinal {i}: {e}"),
                    start,
                ),
            };
            let ords = match indices.get_ordinals(i) {
                Ok(v) => v,
                Err(e) => return error_result(
                    format!("read indices for predicate {i}: {e}"),
                    start,
                ),
            };
            let m = ords.len();
            match_counts.push(m);
            total_matches += m as u64;
            if m == 0 { zero_match += 1; }
            if metadata_total > 0 && m as u64 == metadata_total { full_match += 1; }
            walk_predicate(&pnode, &mut by_field, &mut by_op, m);
        }

        // Render the summary.
        render_summary(
            ctx,
            n_preds,
            metadata_total,
            total_matches,
            &mut match_counts,
            zero_match,
            full_match,
            &by_field,
            &by_op,
            schema.as_ref(),
        );

        let message = if metadata_total > 0 {
            format!(
                "summarised {n_preds} predicates: {total_matches} matches / {metadata_total} records"
            )
        } else {
            format!("summarised {n_preds} predicates: {total_matches} total matches")
        };

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("predicates", "Path", true, None,
                "Predicate slab (PNode records)",
                OptionRole::Input),
            opt("metadata-indices", "Path", true, None,
                "Per-predicate matching ordinals, as produced by `compute evaluate-predicates`",
                OptionRole::Input),
            opt("metadata", "Path", false, None,
                "Metadata slab — used only for the total record count (denominator for selectivity)",
                OptionRole::Input),
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-predicate walk: tally each leaf into the field/op tables.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct FieldStats {
    leaf_count: usize,
    total_matches: u64,
}

#[derive(Default, Clone)]
struct OpStats {
    leaf_count: usize,
    total_matches: u64,
}

fn walk_predicate(
    p: &PNode,
    by_field: &mut BTreeMap<String, FieldStats>,
    by_op: &mut BTreeMap<String, OpStats>,
    matches: usize,
) {
    match p {
        PNode::Predicate(leaf) => {
            let field_name = match &leaf.field {
                FieldRef::Named(n) => n.clone(),
                FieldRef::Index(i) => format!("field[{i}]"),
            };
            let entry = by_field.entry(field_name).or_default();
            entry.leaf_count += 1;
            entry.total_matches += matches as u64;
            let op_name = op_token(leaf.op).to_string();
            let entry = by_op.entry(op_name).or_default();
            entry.leaf_count += 1;
            entry.total_matches += matches as u64;
        }
        PNode::Conjugate(c) => {
            for child in &c.children {
                walk_predicate(child, by_field, by_op, matches);
            }
        }
    }
}

fn op_token(op: OpType) -> &'static str {
    match op {
        OpType::Eq => "eq",
        OpType::Ne => "ne",
        OpType::Lt => "lt",
        OpType::Le => "le",
        OpType::Gt => "gt",
        OpType::Ge => "ge",
        OpType::In => "in",
        OpType::Matches => "matches",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema-sidecar helper.
// ─────────────────────────────────────────────────────────────────────────────

fn open_predicate_schema(
    path: &std::path::Path,
) -> Option<vectordata::metadata_schema::PredicateSchema> {
    use vectordata::metadata_schema::{PredicateSchema, SCHEMA_NAMESPACE};
    let reader = slabtastic::SlabReader::open_namespace(path, Some(SCHEMA_NAMESPACE)).ok()?;
    if reader.total_records() == 0 { return None; }
    let bytes = reader.get(0).ok()?;
    PredicateSchema::from_json_bytes(&bytes).ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer — uses the same width-aware box helpers as slab explain.
// ─────────────────────────────────────────────────────────────────────────────

fn render_summary(
    ctx: &StreamContext,
    n_preds: usize,
    metadata_total: u64,
    total_matches: u64,
    match_counts: &mut [usize],
    zero_match: usize,
    full_match: usize,
    by_field: &BTreeMap<String, FieldStats>,
    by_op: &BTreeMap<String, OpStats>,
    schema: Option<&vectordata::metadata_schema::PredicateSchema>,
) {
    // Distribution stats — sort match_counts in-place for percentiles.
    match_counts.sort_unstable();
    let min = *match_counts.first().unwrap_or(&0);
    let max = *match_counts.last().unwrap_or(&0);
    let median = percentile(match_counts, 0.50);
    let p95 = percentile(match_counts, 0.95);

    let sel = |n: usize| -> String {
        if metadata_total == 0 { "—".into() }
        else { format!("{:.4}", n as f64 / metadata_total as f64) }
    };

    let mut rows: Vec<String> = Vec::new();
    rows.push(format!("predicates:        {n_preds}"));
    if metadata_total > 0 {
        rows.push(format!("metadata records:  {metadata_total}"));
    } else {
        rows.push("metadata records:  (unknown — pass --metadata)".into());
    }
    rows.push(format!("total matches:     {total_matches}"));
    rows.push("─".repeat(8));
    rows.push(format!("match distribution (per predicate)"));
    rows.push(format!("  min:    {:>10}    sel = {}", min, sel(min)));
    rows.push(format!("  median: {:>10}    sel = {}", median, sel(median)));
    rows.push(format!("  p95:    {:>10}    sel = {}", p95, sel(p95)));
    rows.push(format!("  max:    {:>10}    sel = {}", max, sel(max)));
    rows.push(format!("  zero-match predicates: {zero_match}"));
    if metadata_total > 0 {
        rows.push(format!("  full-match predicates: {full_match}"));
    }

    // Target vs observed (when schema present).
    if let Some(s) = schema {
        rows.push("─".repeat(8));
        rows.push(format!("schema target:     {} (seed={}, count={})",
            s.selectivity, s.seed, s.count));
        if metadata_total > 0 && n_preds > 0 {
            let mean_sel = (total_matches as f64) / (n_preds as f64 * metadata_total as f64);
            rows.push(format!("observed mean sel: {:.4}", mean_sel));
            if let Some((lo, hi)) = parse_selectivity_spec(&s.selectivity) {
                let in_range = mean_sel >= lo && mean_sel <= hi;
                let flag = if in_range { "within target" } else { "OUT OF TARGET RANGE" };
                rows.push(format!("  target [{lo:.4} .. {hi:.4}] — {flag}"));
            }
        }
    }

    crate::pipeline::commands::slab::render_explain_box(ctx, "predicate summary", &rows);
    ctx.ui.log("");

    // Per-field breakdown.
    if !by_field.is_empty() {
        let rows: Vec<String> = by_field.iter().map(|(name, st)| {
            let avg = if st.leaf_count > 0 { st.total_matches as f64 / st.leaf_count as f64 } else { 0.0 };
            let avg_sel = if metadata_total > 0 && st.leaf_count > 0 {
                format!("    sel = {:.4}", avg / metadata_total as f64)
            } else { String::new() };
            format!("  {:<24}  leaves={:>5}  avg matches/leaf={:>10.1}{}",
                name, st.leaf_count, avg, avg_sel)
        }).collect();
        crate::pipeline::commands::slab::render_explain_box(ctx, "by field", &rows);
        ctx.ui.log("");
    }

    // Per-op breakdown.
    if !by_op.is_empty() {
        let rows: Vec<String> = by_op.iter().map(|(op, st)| {
            let avg = if st.leaf_count > 0 { st.total_matches as f64 / st.leaf_count as f64 } else { 0.0 };
            let avg_sel = if metadata_total > 0 && st.leaf_count > 0 {
                format!("    sel = {:.4}", avg / metadata_total as f64)
            } else { String::new() };
            format!("  {:<10}  leaves={:>5}  avg matches/leaf={:>10.1}{}",
                op, st.leaf_count, avg, avg_sel)
        }).collect();
        crate::pipeline::commands::slab::render_explain_box(ctx, "by op", &rows);
        ctx.ui.log("");
    }
}

fn percentile(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() { return 0; }
    let idx = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Parse `PredicateSchema::selectivity` strings — scalar or
/// interval — into a `(lo, hi)` range for the "within target"
/// check. Returns `None` if the string isn't recognisable.
fn parse_selectivity_spec(s: &str) -> Option<(f64, f64)> {
    if let Some((lo, hi)) = s.split_once("..") {
        Some((lo.trim().parse().ok()?, hi.trim().parse().ok()?))
    } else {
        let v: f64 = s.trim().parse().ok()?;
        // For a scalar target, accept ±20% tolerance — same
        // band the calibration tests use.
        Some((v * 0.8, v * 1.2))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small local utilities.
// ─────────────────────────────────────────────────────────────────────────────

fn resolve(s: &str, workspace: &std::path::Path) -> PathBuf {
    let p = PathBuf::from(s);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, description: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: description.to_string(),
        extended_description: None,
        role,
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_picks_index_at_quantile() {
        let v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(percentile(&v, 0.0), 0);
        assert_eq!(percentile(&v, 0.5), 5);
        assert_eq!(percentile(&v, 0.95), 9);
        assert_eq!(percentile(&v, 1.0), 9);
    }

    #[test]
    fn parse_selectivity_scalar_expands_to_band() {
        let (lo, hi) = parse_selectivity_spec("0.10").unwrap();
        assert!((lo - 0.08).abs() < 1e-9);
        assert!((hi - 0.12).abs() < 1e-9);
    }

    #[test]
    fn parse_selectivity_interval_round_trips() {
        let (lo, hi) = parse_selectivity_spec("0.05..0.20").unwrap();
        assert!((lo - 0.05).abs() < 1e-9);
        assert!((hi - 0.20).abs() < 1e-9);
    }

    #[test]
    fn parse_selectivity_invalid_returns_none() {
        assert!(parse_selectivity_spec("nope").is_none());
        assert!(parse_selectivity_spec("..").is_none());
    }

    /// `walk_predicate` recursively tallies a tree's leaves
    /// into the field and op tables. A 2-leaf AND should bump
    /// both fields and both ops by 1, both crediting the same
    /// match count.
    #[test]
    fn walk_predicate_tallies_each_leaf() {
        use veks_core::formats::pnode::{
            Comparand, ConjugateNode, ConjugateType, PredicateNode,
        };
        let tree = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("age".into()),
                    op: OpType::Ge,
                    comparands: vec![Comparand::Int(18)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("name".into()),
                    op: OpType::Matches,
                    comparands: vec![Comparand::Text("ali".into())],
                }),
            ],
        });
        let mut by_field = BTreeMap::new();
        let mut by_op = BTreeMap::new();
        walk_predicate(&tree, &mut by_field, &mut by_op, 42);
        assert_eq!(by_field.get("age").unwrap().leaf_count, 1);
        assert_eq!(by_field.get("age").unwrap().total_matches, 42);
        assert_eq!(by_field.get("name").unwrap().leaf_count, 1);
        assert_eq!(by_field.get("name").unwrap().total_matches, 42);
        assert_eq!(by_op.get("ge").unwrap().leaf_count, 1);
        assert_eq!(by_op.get("matches").unwrap().leaf_count, 1);
    }
}
