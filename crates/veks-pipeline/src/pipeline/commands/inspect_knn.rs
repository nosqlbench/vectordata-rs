// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: explain an unfiltered KNN result for a
//! single query.
//!
//! Traces through every stage of the unfiltered KNN pipeline
//! for a given query ordinal: query info → ground-truth
//! neighbors → per-neighbor preview with optional distances.
//!
//! Companion to `analyze explain-filtered-knn`: this one stays
//! on the unfiltered (G) side; the filtered variant adds the
//! predicate + intersection stages on top.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::io::XvecReader;
use vectordata::VectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc,
    Status, StreamContext, render_options_table,
};

pub struct ExplainKnnOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ExplainKnnOp)
}

impl CommandOp for ExplainKnnOp {
    fn command_path(&self) -> &str {
        "analyze explain-knn"
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
            summary: "Trace an unfiltered KNN result through every pipeline stage".into(),
            body: format!(
                r#"# analyze explain-knn

Trace an unfiltered KNN result for a single query through
every pipeline stage.

## Description

For a given query ordinal, narrates the unfiltered KNN
pipeline:

1. **Query vector** — shape (dim) and a few leading coordinates.
2. **Ground-truth neighbors (G)** — the top-k neighbor ordinals
   recorded in the dataset's `neighbor_indices` facet.
3. **Neighbor distances (D)** — when a `neighbor_distances`
   facet is present, the per-rank distance values.
4. **Per-neighbor preview** — for each top-k neighbor: ordinal,
   distance (if available), and a few leading coordinates of
   the base vector.

When run in a directory with `dataset.yaml`, all file paths
are auto-resolved from the profile. Use `--profile` to select
a non-default profile.

## Options

{}

## Notes

- The ordinal indexes into the query vectors / G facet.
- `--limit N` caps the number of per-neighbor rows rendered
  (default 20).
- `--coords N` caps the number of vector coordinates rendered
  per neighbor preview (default 6). Set to 0 to suppress the
  per-neighbor base-vector preview entirely.

## Companion

See `analyze explain-filtered-knn` for the same trace plus the
predicate / selectivity / intersection stages.
"#,
                render_options_table(&options),
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> { vec![] }

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
            .get("limit").and_then(|s| s.parse().ok()).unwrap_or(20);
        let coords: usize = options
            .get("coords").and_then(|s| s.parse().ok()).unwrap_or(6);

        // Resolve paths from explicit options or dataset.yaml profile.
        let ds_path = ctx.workspace.join("dataset.yaml");
        let config = match vectordata::dataset::DatasetConfig::load(&ds_path) {
            Ok(c) => c,
            Err(_) => return error_result(
                "no dataset.yaml found; provide all paths explicitly".into(), start),
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

        let query_path = match resolve("query_vectors", "query-vectors") {
            Some(p) => p,
            None => return error_result("cannot resolve query_vectors path".into(), start),
        };
        let base_path = match resolve("base_vectors", "base-vectors") {
            Some(p) => p,
            None => return error_result("cannot resolve base_vectors path".into(), start),
        };
        let gt_path = match resolve("neighbor_indices", "ground-truth") {
            Some(p) => p,
            None => return error_result("cannot resolve neighbor_indices path".into(), start),
        };
        let distances_path = resolve("neighbor_distances", "distances");

        ctx.ui.log(&format!(
            "\n═══ Query ordinal {} ═══════════════════════════════════════",
            ordinal));

        // ── Stage 1: Query vector ──
        ctx.ui.log("\n┌─ Stage 1: Query vector ────────────────────────────────");
        let query_reader = match XvecReader::<f32>::open_path(&query_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open query: {}", e), start),
        };
        let qcount = VectorReader::<f32>::count(&query_reader);
        let qdim = VectorReader::<f32>::dim(&query_reader);
        if ordinal >= qcount {
            return error_result(format!(
                "ordinal {} out of range (query count = {})", ordinal, qcount), start);
        }
        ctx.ui.log(&format!("│  dim={}, total queries={}", qdim, qcount));
        let q_slice = match VectorReader::<f32>::get(&query_reader, ordinal) {
            Ok(v) => v,
            Err(e) => return error_result(format!("read query {}: {}", ordinal, e), start),
        };
        if coords > 0 {
            ctx.ui.log(&format!("│  vector preview: {}", fmt_coords(&q_slice, coords)));
        }

        // ── Stage 2: Ground truth neighbors ──
        ctx.ui.log("│");
        ctx.ui.log("├─ Stage 2: Ground-truth neighbors (G) ──────────────────");
        let gt_reader = match XvecReader::<i32>::open_path(&gt_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open ground-truth: {}", e), start),
        };
        if ordinal >= gt_reader.count() {
            return error_result(format!(
                "ordinal {} out of range (G count = {})", ordinal, gt_reader.count()), start);
        }
        let gt_slice = gt_reader.get_slice(ordinal).to_vec();
        ctx.ui.log(&format!("│  k={} neighbors", gt_slice.len()));

        // ── Stage 3: Distances ──
        let dist_slice: Option<Vec<f32>> = if let Some(ref dp) = distances_path {
            match XvecReader::<f32>::open_path(dp) {
                Ok(r) if ordinal < r.count() => Some(r.get_slice(ordinal).to_vec()),
                Ok(_) => None,
                Err(e) => {
                    ctx.ui.log(&format!("│  (distance facet open failed: {})", e));
                    None
                }
            }
        } else { None };
        if dist_slice.is_some() {
            ctx.ui.log("│  (neighbor_distances facet present — values shown below)");
        } else {
            ctx.ui.log("│  (no neighbor_distances facet — distances omitted)");
        }

        // ── Stage 4: Per-neighbor preview ──
        ctx.ui.log("│");
        ctx.ui.log("└─ Stage 3: Per-neighbor preview ────────────────────────");

        let base_reader: Option<XvecReader<f32>> = if coords > 0 {
            match XvecReader::<f32>::open_path(&base_path) {
                Ok(r) => Some(r),
                Err(e) => {
                    ctx.ui.log(&format!("   (base_vectors open failed; suppressing coord preview: {})", e));
                    None
                }
            }
        } else { None };

        let show = gt_slice.len().min(limit);
        for i in 0..show {
            let ord = gt_slice[i];
            let dist_str = dist_slice.as_ref()
                .and_then(|d| d.get(i))
                .map(|d| format!("  dist={:.6}", d))
                .unwrap_or_default();
            let coord_str = base_reader.as_ref()
                .and_then(|r| {
                    if (ord as usize) < r.count() {
                        Some(fmt_coords(&r.get(ord as usize).unwrap_or_default(), coords))
                    } else { None }
                })
                .map(|s| format!("  {}", s))
                .unwrap_or_default();
            ctx.ui.log(&format!("   [{:>3}] ordinal {:>8}{}{}", i, ord, dist_str, coord_str));
        }
        if gt_slice.len() > show {
            ctx.ui.log(&format!("   ... and {} more", gt_slice.len() - show));
        }

        ctx.ui.log("");

        CommandResult {
            status: Status::Ok,
            message: format!(
                "query {} explained: k={} neighbors",
                ordinal, gt_slice.len(),
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("ordinal", "int", true, None, "Query ordinal to explain", OptionRole::Config),
            opt("profile", "string", false, Some("default"), "Profile to resolve facets from", OptionRole::Config),
            opt("query-vectors", "Path", false, None, "Query vectors (auto-resolved from profile)", OptionRole::Input),
            opt("base-vectors", "Path", false, None, "Base vectors (auto-resolved from profile)", OptionRole::Input),
            opt("ground-truth", "Path", false, None, "Neighbor-indices file (auto-resolved from profile)", OptionRole::Input),
            opt("distances", "Path", false, None, "Neighbor-distances file (auto-resolved from profile when present)", OptionRole::Input),
            opt("limit", "int", false, Some("20"), "Max neighbors to display", OptionRole::Config),
            opt("coords", "int", false, Some("6"), "Coordinates per vector preview (0 to suppress)", OptionRole::Config),
        ]
    }
}

fn fmt_coords(v: &[f32], n: usize) -> String {
    let take = v.len().min(n);
    let mut s = String::from("[");
    for (i, x) in v.iter().take(take).enumerate() {
        if i > 0 { s.push_str(", "); }
        s.push_str(&format!("{:.4}", x));
    }
    if v.len() > take { s.push_str(&format!(", …({} more)", v.len() - take)); }
    s.push(']');
    s
}

fn resolve_path(s: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(s);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        extended_description: None,
        role,
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult { status: Status::Error, message, produced: vec![], elapsed: start.elapsed() }
}
