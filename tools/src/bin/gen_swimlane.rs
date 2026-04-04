// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Generate the facet swimlane SVG diagram.
//!
//! Produces a pixel-perfect grid layout with 8 facet lanes (BQGDMPRF),
//! pipeline steps, cross-lane edges, variable pills, and optional input
//! parallelograms.
//!
//! Usage:
//!   cargo run -p tools --bin gen_swimlane > docs/design/diagrams/15-facet-swimlane.svg

use std::collections::BTreeSet;
use std::fmt::Write;

// ═══════════════════════════════════════════════════════════════════════════
// Layout constants
// ═══════════════════════════════════════════════════════════════════════════

const COL_W: i32 = 170;
const ROW_H: i32 = 56;
const PAD_X: i32 = 12;
const PAD_Y: i32 = 6;
const BOX_W: i32 = COL_W - 2 * PAD_X;
const BOX_R: i32 = 6;
const MARGIN_LR: i32 = 90;
const MARGIN_TOP: i32 = 62;
const MARGIN_BOT: i32 = 34;
const FONT: i32 = 11;
const FONT_SM: i32 = 9;
const INPUT_H: i32 = 30;
const PILL_W: i32 = 72;
const PILL_H: i32 = 16;

// Colors that contain # — can't appear literally in r#""# strings
const CLR_EDGE: &str = "#888";
const CLR_LABEL: &str = "#888";
const CLR_TEXT: &str = "#333";
const CLR_COND: &str = "#666";
const CLR_INPUT_TEXT: &str = "#555";

// ═══════════════════════════════════════════════════════════════════════════
// Data model
// ═══════════════════════════════════════════════════════════════════════════

struct Lane {
    code: &'static str,
    name: &'static str,
    fill: &'static str,
    stroke: &'static str,
    bg: &'static str,
}

struct Step {
    row: i32,
    col: i32,
    label: &'static str,
    cond: &'static str,
    fill_override: Option<&'static str>,
    tooltip: &'static str,
}

struct Artifact {
    col: i32,
    label: &'static str,
    tooltip: &'static str,
}

struct Input {
    col: i32,
    label: &'static str,
    tooltip: &'static str,
}

struct VEdge {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
}

struct XEdge {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
    color: &'static str,
    label: &'static str,
}

struct Variable {
    set_row: i32,
    set_col: i32,
    name: &'static str,
    tooltip: &'static str,
}

/// Intermediate artifact — a cache file produced between steps.
/// Rendered as a small label between rows in the lane.
struct Intermediate {
    row: i32,       // row BELOW which it appears (between row and row+1)
    col: i32,
    label: &'static str,
    tooltip: &'static str,
}

// ═══════════════════════════════════════════════════════════════════════════
// Data
// ═══════════════════════════════════════════════════════════════════════════

const LANES: &[Lane] = &[
    Lane { code: "B", name: "Base Vectors",  fill: "#d7eed8", stroke: "#4caf50", bg: "#f0f8f0" },
    Lane { code: "Q", name: "Query Vectors", fill: "#d4e6f7", stroke: "#42a5f5", bg: "#eef5fc" },
    Lane { code: "G", name: "GT Indices",    fill: "#fce8c8", stroke: "#ffa726", bg: "#fef6ec" },
    Lane { code: "D", name: "GT Distances",  fill: "#fce8c8", stroke: "#ffa726", bg: "#fef6ec" },
    Lane { code: "M", name: "Metadata",      fill: "#e8d5f0", stroke: "#ab47bc", bg: "#f6eff9" },
    Lane { code: "P", name: "Predicates",    fill: "#e8d5f0", stroke: "#ab47bc", bg: "#f6eff9" },
    Lane { code: "R", name: "Pred Results",  fill: "#e8d5f0", stroke: "#ab47bc", bg: "#f6eff9" },
    Lane { code: "F", name: "Filtered KNN",  fill: "#f5d5de", stroke: "#ef5350", bg: "#fcf0f2" },
];

const STEPS: &[Step] = &[
    Step { row: 1,  col: 0, label: "convert\nvectors",           cond: "[foreign fmt]",      fill_override: None,
        tooltip: "transform convert: Convert source vectors (npy, parquet, dir) to native xvec format.\nActive when source is not already a native xvec file.\nIdentity (symlink) when source is native fvec/mvec." },
    Step { row: 1,  col: 4, label: "convert\nmetadata",          cond: "[foreign fmt]",      fill_override: None,
        tooltip: "transform convert: Convert metadata from parquet/dir to slab format.\nActive when source is not already a native .slab file.\nIdentity (symlink) when source is native slab." },
    Step { row: 2,  col: 0, label: "count\nvectors",             cond: "",                   fill_override: None,
        tooltip: "state set: Count records in all_vectors (or combined B+Q) and store as ${vector_count}.\nAlways active when B facet is required." },
    Step { row: 2,  col: 4, label: "survey\nmetadata",           cond: "",                   fill_override: None,
        tooltip: "analyze survey: Sample metadata to discover schema and value ranges.\nProduces metadata_survey.json used by predicate generation.\nAlways active when M facet is required." },
    Step { row: 3,  col: 0, label: "prepare\nvectors",            cond: "[!no_dedup]",        fill_override: None,
        tooltip: "prepare-vectors: External merge-sort with in-segment dedup detection,\nL2 normalization (f64), near-zero detection (L2 < 1e-6),\nand normalization quality measurement. See SRD §20.3.\nProduces: sorted_ordinals.ivec, dedup_duplicates.ivec, zero_ordinals.ivec,\nsorted+normalized run files, and norm statistics (is_normalized, mean_epsilon).\nAll cleaning happens in one pass while data is cache-hot." },
    Step { row: 3,  col: 5, label: "generate\npredicates",       cond: "",                   fill_override: None,
        tooltip: "generate predicates: Synthesize test predicates from metadata survey.\nProduces predicates.slab with configurable count and selectivity.\nAlways active when M facet is required." },
    Step { row: 4,  col: 0, label: "generate\nshuffle",          cond: "[self-search\n OR combined B+Q]", fill_override: None,
        tooltip: "generate shuffle: Fisher-Yates permutation over clean ordinals\n(sorted ordinals minus duplicates minus zeros).\nDeterministic given seed. Used by vector and metadata extraction.\nElided when HDF5 with separate query (no self-search)." },
    Step { row: 4,  col: 4, label: "extract\nmetadata",          cond: "[self-search]",      fill_override: None,
        tooltip: "transform extract: Reorder metadata records to match shuffled base vectors.\nUses same shuffle as vector extraction for ordinal congruency.\nElided when not self-search (metadata already ordinal-aligned)." },
    Step { row: 4,  col: 6, label: "evaluate\npredicates [pp]",  cond: "",                   fill_override: None,
        tooltip: "compute evaluate-predicates: Evaluate each predicate against metadata records.\nPer-profile: range [0, base_count) matches the profile's base vector window.\nAlways active when M facet is required." },
    Step { row: 5,  col: 0, label: "extract\nbase",              cond: "",                   fill_override: None,
        tooltip: "transform extract: Extract base vectors from sorted+normalized data,\neliding excluded ordinals (duplicates + zeros). See SRD §20.5.\nContiguous-run memcpy between exclusion holes — no per-vector compute.\nSelf-search: uses shuffle remainder. Separate query: full clean set." },
    Step { row: 5,  col: 1, label: "extract\nqueries",           cond: "[self-search\n OR combined B+Q]",  fill_override: Some("#d4e6f7"),
        tooltip: "transform extract: Extract query vectors from shuffle.\nFirst query_count shuffled ordinals, eliding excluded.\nElided when HDF5 with separate query file." },
    Step { row: 6,  col: 0, label: "count\nbase",                cond: "",                   fill_override: None,
        tooltip: "state set: Count records in base_vectors and store as ${base_count}.\nVerifiable: vector_count - duplicate_count - zero_count = base_count." },
    Step { row: 7,  col: 2, label: "compute\nknn [pp]",          cond: "[G + !precomputed]", fill_override: None,
        tooltip: "compute knn: Brute-force exact K-nearest-neighbor computation.\nPer-profile: uses base_vectors[0..base_count) as corpus.\nProduces neighbor_indices.ivec and neighbor_distances.fvec." },
    Step { row: 7,  col: 7, label: "compute\nfiltered-knn [pp]", cond: "[F]",                fill_override: None,
        tooltip: "compute filtered-knn: KNN with predicate pre-filtering.\nPer-profile: evaluates predicates to get eligible base vectors,\nthen computes KNN among eligible only." },
    Step { row: 8, col: 2, label: "verify\nknn [pp]",            cond: "[G]",                fill_override: None,
        tooltip: "verify knn: Sparse-sample brute-force recomputation.\nPer-profile: samples queries, recomputes KNN, compares to stored GT." },
    Step { row: 8, col: 7, label: "verify\npredicates [pp]",     cond: "[F]",                fill_override: None,
        tooltip: "verify predicate-results: SQLite-backed predicate verification.\nPer-profile: compares pre-computed metadata_indices to SQL evaluation." },
];

const ARTIFACTS: &[Artifact] = &[
    Artifact { col: 0, label: "base_vectors\n.fvec/.mvec",
        tooltip: "B: The corpus vectors to search. Located at profiles/base/base_vectors.{ext}.\nFormat matches the source (fvec for f32, mvec for f16).\nRecord count = clean_count - query_count (or capped by base_end)." },
    Artifact { col: 1, label: "query_vectors\n.fvec/.mvec",
        tooltip: "Q: The query vectors to run against the corpus. Located at profiles/base/query_vectors.{ext}.\nEither extracted from base via shuffle (self-search) or provided directly.\nRecord count = query_count." },
    Artifact { col: 2, label: "neighbor_indices\n.ivec",
        tooltip: "G: Ground-truth KNN indices. Per-profile at profiles/{profile}/neighbor_indices.ivec.\nDimension = neighbors (e.g., 100). Each record lists the K nearest base vector ordinals for one query.\nEither computed by brute-force KNN or provided by user." },
    Artifact { col: 3, label: "neighbor_distances\n.fvec",
        tooltip: "D: Ground-truth KNN distances. Per-profile at profiles/{profile}/neighbor_distances.fvec.\nPaired with neighbor_indices — same shape (query_count x neighbors).\nEach value is the distance from the query to the corresponding neighbor." },
    Artifact { col: 4, label: "metadata_content\n.slab",
        tooltip: "M: Metadata records, ordinal-aligned with base vectors. At profiles/base/metadata_content.slab.\nEach record is an MNode (structured key-value metadata for one base vector).\nOrdinal congruency: record N corresponds to base_vectors record N." },
    Artifact { col: 5, label: "metadata_predicates\n.slab",
        tooltip: "P: Test predicates for filtered search. At profiles/base/predicates.slab.\nEach record is a PNode (predicate tree) synthesized from the metadata survey.\nConfigurable count and selectivity." },
    Artifact { col: 6, label: "metadata_indices\n.slab",
        tooltip: "R: Predicate evaluation results. Per-profile at profiles/{profile}/metadata_indices.slab.\nFor each predicate, stores the set of base vector ordinals that match.\nUsed by compute-filtered-knn as the pre-filter." },
    Artifact { col: 7, label: "filtered_neighbor\n_indices+_distances",
        tooltip: "F: Filtered KNN results. Per-profile at profiles/{profile}/filtered_neighbor_{indices,distances}.\nSame shape as G/D but computed with predicate pre-filtering.\nEach query uses only base vectors that match its predicate." },
];

const INPUTS: &[Input] = &[
    Input { col: 0, label: "vector source\n(fvec/npy/dir)",
        tooltip: "--base-vectors: Source vector data. Can be fvec/mvec (native), npy dir (foreign), or parquet.\nWhen native xvec: identity (symlink), no conversion step.\nWhen foreign: convert step runs to produce all_vectors.mvec." },
    Input { col: 1, label: "query vectors\n(fvec/mvec)",
        tooltip: "--query-vectors: Separate query vector file.\nWhen provided: no shuffle or extraction needed — queries used directly.\nWhen absent + self-search: queries extracted from base via shuffle." },
    Input { col: 2, label: "GT indices\n(ivec)",
        tooltip: "--ground-truth: Pre-computed KNN ground truth indices.\nWhen provided: compute-knn step is elided (identity).\nverify-knn still runs to validate the provided GT." },
    Input { col: 3, label: "GT distances\n(fvec)",
        tooltip: "--ground-truth-distances: Pre-computed KNN distances.\nPaired with GT indices. When provided: distances artifact is identity." },
    Input { col: 4, label: "metadata\n(slab/parquet)",
        tooltip: "--metadata: Metadata source for predicate-filtered search.\nWhen slab: identity (no conversion).\nWhen parquet/dir: convert step produces metadata_all.slab." },
    Input { col: 5, label: "predicates\n(slab)",
        tooltip: "User-provided predicates slab.\nWhen provided: generate-predicates step is elided.\nRare — usually predicates are synthesized from the metadata survey." },
    Input { col: 6, label: "pred results\n(slab)",
        tooltip: "User-provided predicate evaluation results.\nWhen provided: evaluate-predicates step is elided.\nRare — usually computed by the pipeline." },
    Input { col: 7, label: "filtered GT\n(ivec+fvec)",
        tooltip: "User-provided filtered KNN ground truth.\nWhen provided: compute-filtered-knn step is elided.\nRare — usually computed by the pipeline." },
];

const V_EDGES: &[VEdge] = &[
    VEdge { from_row: 1, from_col: 0, to_row: 2, to_col: 0 },  // convert → count
    VEdge { from_row: 2, from_col: 0, to_row: 3, to_col: 0 },  // count → sort+dedup+norm+zero
    VEdge { from_row: 3, from_col: 0, to_row: 4, to_col: 0 },  // sort → shuffle
    VEdge { from_row: 4, from_col: 0, to_row: 5, to_col: 0 },  // shuffle → extract-base
    VEdge { from_row: 5, from_col: 0, to_row: 6, to_col: 0 },  // extract → count-base
    VEdge { from_row: 1, from_col: 4, to_row: 2, to_col: 4 },  // convert-meta → survey
    VEdge { from_row: 1, from_col: 4, to_row: 4, to_col: 4 },  // convert-meta → extract-meta
    VEdge { from_row: 3, from_col: 5, to_row: 9, to_col: 5 },  // gen-predicates → artifact
    VEdge { from_row: 4, from_col: 6, to_row: 9, to_col: 6 },  // eval-predicates → artifact
    VEdge { from_row: 7, from_col: 2, to_row: 8, to_col: 2 },  // compute-knn → verify
    VEdge { from_row: 7, from_col: 7, to_row: 8, to_col: 7 },  // compute-fknn → verify-pred
];

/// Extra vertical edges from last step to artifact row
const ARTIFACT_EDGES: &[(i32, i32)] = &[
    (0, 6), (4, 4), (2, 8), (1, 5), (7, 8),
];

const X_EDGES: &[XEdge] = &[
    XEdge { from_row: 2, from_col: 4, to_row: 3, to_col: 5, color: "#ab47bc", label: "schema" },
    XEdge { from_row: 4, from_col: 0, to_row: 5, to_col: 1, color: "#42a5f5", label: "shuffle" },
    XEdge { from_row: 4, from_col: 0, to_row: 4, to_col: 4, color: "#ab47bc", label: "ordinals" },
    XEdge { from_row: 4, from_col: 4, to_row: 4, to_col: 6, color: "#ab47bc", label: "" },
    XEdge { from_row: 3, from_col: 5, to_row: 4, to_col: 6, color: "#ab47bc", label: "predicates" },
    XEdge { from_row: 6, from_col: 0, to_row: 7, to_col: 2, color: "#ffa726", label: "base" },
    XEdge { from_row: 6, from_col: 0, to_row: 7, to_col: 7, color: "#ef5350", label: "base" },
    XEdge { from_row: 7, from_col: 2, to_row: 9, to_col: 3, color: "#ffa726", label: "paired" },
];

const VARIABLES: &[Variable] = &[
    Variable { set_row: 2,  set_col: 0, name: "vector_count",
        tooltip: "Total records in source (or combined B+Q).\nUsed as shuffle interval when no cleaning is active." },
    Variable { set_row: 3,  set_col: 0, name: "duplicate_count",
        tooltip: "Number of duplicate vectors found during sort.\nRecorded in variables.yaml for provenance." },
    Variable { set_row: 3,  set_col: 0, name: "zero_count",
        tooltip: "Number of near-zero vectors detected (L2 < 1e-6) during prepare-vectors.\nRecorded in variables.yaml for provenance." },
    Variable { set_row: 3,  set_col: 0, name: "is_normalized",
        tooltip: "Whether vectors pass normalization quality check.\nComputed during prepare-vectors from L2 norm epsilon distribution (f64).\nThreshold: 1e-5 for f32, 1e-14 for f64, 1e-1 for f16." },
    Variable { set_row: 6,  set_col: 0, name: "base_count",
        tooltip: "Records in the extracted base_vectors file.\nbase_count = vector_count - duplicate_count - zero_count (- query_count if self-search).\nUsed by compute-knn as the search window size." },
];

const INTERMEDIATES: &[Intermediate] = &[
    Intermediate { row: 1, col: 0, label: "all_vectors.mvec",
        tooltip: "All imported vectors in native xvec format.\nLocated at ${cache}/all_vectors.{ext}.\nMay include combined B+Q for Strategy 1." },
    Intermediate { row: 3, col: 0, label: "sorted+normalized\nrun files",
        tooltip: "Sorted, L2-normalized, dedup-detected, zero-detected run files.\nLocated in ${cache}/dedup_runs/.\nAlso produces: sorted_ordinals.ivec, dedup_duplicates.ivec, zero_ordinals.ivec.\nSee SRD §20.3." },
    Intermediate { row: 4, col: 0, label: "shuffle.ivec",
        tooltip: "PRNG permutation of clean ordinals (= sorted minus dups minus zeros).\nLocated at ${cache}/shuffle.ivec.\nDeterministic given seed. Used by vector and metadata extraction." },
    Intermediate { row: 1, col: 4, label: "metadata_all.slab",
        tooltip: "All imported metadata as MNode slab records.\nLocated at ${cache}/metadata_all.slab." },
    Intermediate { row: 2, col: 4, label: "metadata_survey.json",
        tooltip: "Schema and value range survey of the metadata.\nLocated at ${cache}/metadata_survey.json.\nUsed by generate-predicates for selectivity targeting." },
];

const TOTAL_ROWS: i32 = 10;

// ═══════════════════════════════════════════════════════════════════════════
// Geometry helpers
// ═══════════════════════════════════════════════════════════════════════════

fn cx(col: i32) -> i32 { MARGIN_LR + col * COL_W + COL_W / 2 }
fn cy(row: i32) -> i32 { MARGIN_TOP + row * ROW_H + ROW_H / 2 }
fn box_x(col: i32) -> i32 { MARGIN_LR + col * COL_W + PAD_X }
fn box_y(row: i32) -> i32 { MARGIN_TOP + row * ROW_H + PAD_Y }
fn box_h() -> i32 { ROW_H - 2 * PAD_Y }

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

// ═══════════════════════════════════════════════════════════════════════════
// SVG rendering
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let w = MARGIN_LR * 2 + LANES.len() as i32 * COL_W;
    let h = MARGIN_TOP + TOTAL_ROWS * ROW_H + MARGIN_BOT;
    let mut svg = String::with_capacity(16384);

    writeln!(svg, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="Helvetica, Arial, sans-serif">"#).unwrap();
    writeln!(svg, r#"  <rect width="{w}" height="{h}" fill="white" />"#).unwrap();

    // Defs — arrow markers
    let mut colors = BTreeSet::new();
    for e in X_EDGES { colors.insert(e.color); }
    for i in INPUTS { colors.insert(LANES[i.col as usize].stroke); }
    writeln!(svg, "  <defs>").unwrap();
    for &c in &colors {
        let id = &c[1..];
        writeln!(svg, r#"    <marker id="arrow-{id}" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">"#).unwrap();
        writeln!(svg, r#"      <path d="M0,0 L8,3 L0,6" fill="{c}" />"#).unwrap();
        writeln!(svg, "    </marker>").unwrap();
    }
    writeln!(svg, "  </defs>").unwrap();

    // Lane backgrounds
    for (i, lane) in LANES.iter().enumerate() {
        let x = MARGIN_LR + i as i32 * COL_W;
        let lh = TOTAL_ROWS * ROW_H;
        writeln!(svg, r#"  <rect x="{x}" y="{MARGIN_TOP}" width="{COL_W}" height="{lh}" fill="{}" />"#, lane.bg).unwrap();
        writeln!(svg, r#"  <line x1="{x}" y1="{MARGIN_TOP}" x2="{x}" y2="{}" stroke="{}" opacity="0.15" />"#, MARGIN_TOP + lh, lane.stroke).unwrap();
    }
    let rx = MARGIN_LR + LANES.len() as i32 * COL_W;
    writeln!(svg, r#"  <line x1="{rx}" y1="{MARGIN_TOP}" x2="{rx}" y2="{}" stroke="{}" opacity="0.15" />"#, MARGIN_TOP + TOTAL_ROWS * ROW_H, LANES.last().unwrap().stroke).unwrap();

    // Input parallelograms
    writeln!(svg, r#"  <text x="{MARGIN_LR}" y="{}" font-size="9" fill="{CLR_LABEL}" font-style="italic">optional user-provided inputs (bypass pipeline when present)</text>"#, MARGIN_TOP - INPUT_H - 10).unwrap();
    for input in INPUTS {
        let lane = &LANES[input.col as usize];
        let x = cx(input.col);
        let yt = MARGIN_TOP - INPUT_H - 4;
        let bx = box_x(input.col);
        let skew = 8;
        writeln!(svg, "  <g>").unwrap();
        writeln!(svg, "    <title>{}</title>", esc(input.tooltip)).unwrap();
        writeln!(svg, r#"    <polygon points="{},{yt} {},{yt} {},{} {},{}" fill="{}" stroke="{}" stroke-width="1.5" stroke-dasharray="4,3" />"#,
            bx + skew, bx + BOX_W, bx + BOX_W - skew, yt + INPUT_H, bx, yt + INPUT_H, lane.bg, lane.stroke).unwrap();
        for (j, line) in input.label.split('\n').enumerate() {
            let ty = yt + 12 + j as i32 * (FONT_SM + 2);
            writeln!(svg, r#"    <text x="{x}" y="{ty}" text-anchor="middle" font-size="{FONT_SM}" fill="{CLR_INPUT_TEXT}">{}</text>"#, esc(line)).unwrap();
        }
        writeln!(svg, "  </g>").unwrap();
        writeln!(svg, r#"  <line x1="{x}" y1="{}" x2="{x}" y2="{}" stroke="{}" stroke-width="1.5" stroke-dasharray="4,3" />"#, yt + INPUT_H, MARGIN_TOP + PAD_Y, lane.stroke).unwrap();
        // Bypass arrow
        let art_y = MARGIN_TOP + 12 * ROW_H + PAD_Y;
        let lane_right = MARGIN_LR + input.col * COL_W + COL_W - PAD_X - 3;
        writeln!(svg, r#"  <polyline points="{},{} {},{} {},{}" fill="none" stroke="{}" stroke-width="1" stroke-dasharray="4,3" opacity="0.5" />"#,
            bx + BOX_W - skew, yt + INPUT_H / 2, lane_right, yt + INPUT_H / 2, lane_right, art_y + box_h() / 2, lane.stroke).unwrap();
        writeln!(svg, r#"  <text x="{}" y="{}" text-anchor="end" font-size="7" fill="{}" opacity="0.6">identity</text>"#, lane_right - 3, MARGIN_TOP + 6 * ROW_H, lane.stroke).unwrap();
    }

    // Vertical edges
    for e in V_EDGES {
        let x1 = cx(e.from_col);
        let x2 = cx(e.to_col);
        let y1 = cy(e.from_row) + ROW_H / 2 - PAD_Y;
        let y2 = cy(e.to_row) - ROW_H / 2 + PAD_Y;
        writeln!(svg, r#"  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{CLR_EDGE}" stroke-width="1.5" />"#).unwrap();
    }
    for &(col, last_row) in ARTIFACT_EDGES {
        let x = cx(col);
        let y1 = cy(last_row) + ROW_H / 2 - PAD_Y;
        let y2 = cy(TOTAL_ROWS - 1) - ROW_H / 2 + PAD_Y;
        writeln!(svg, r#"  <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{CLR_EDGE}" stroke-width="1.5" />"#).unwrap();
    }

    // Cross-lane edges
    for e in X_EDGES {
        let x1 = cx(e.from_col);
        let x2 = cx(e.to_col);
        let id = &e.color[1..];
        if e.from_row == e.to_row {
            let y = cy(e.from_row);
            writeln!(svg, r#"  <line x1="{}" y1="{y}" x2="{}" y2="{y}" stroke="{}" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow-{id})" />"#,
                x1 + BOX_W / 2, x2 - BOX_W / 2, e.color).unwrap();
        } else {
            let y1 = cy(e.from_row) + ROW_H / 2 - PAD_Y;
            let y2 = cy(e.to_row) - ROW_H / 2 + PAD_Y;
            let mid = (y1 + y2) / 2;
            writeln!(svg, r#"  <polyline points="{x1},{y1} {x1},{mid} {x2},{mid} {x2},{y2}" fill="none" stroke="{}" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow-{id})" />"#, e.color).unwrap();
        }
        if !e.label.is_empty() {
            let lx = (x1 + x2) / 2;
            let ly = if e.from_row == e.to_row { cy(e.from_row) - 4 } else { (cy(e.from_row) + cy(e.to_row)) / 2 - 4 };
            writeln!(svg, r#"  <text x="{lx}" y="{ly}" text-anchor="middle" font-size="{FONT_SM}" fill="{}">{}</text>"#, e.color, esc(e.label)).unwrap();
        }
    }

    // Headers — linked to SRD §2.8 (facet codes)
    for (i, lane) in LANES.iter().enumerate() {
        let bx = box_x(i as i32);
        let by = box_y(0);
        writeln!(svg, r#"  <a href="02-data-model.md#28-dataset-facets" target="_top">"#).unwrap();
        writeln!(svg, r#"    <rect x="{bx}" y="{by}" width="{BOX_W}" height="{}" rx="{BOX_R}" fill="{}" stroke="{}" stroke-width="2" style="cursor:pointer" />"#, box_h(), lane.fill, lane.stroke).unwrap();
        writeln!(svg, r#"    <text x="{}" y="{}" text-anchor="middle" font-size="{FONT}" font-weight="bold">{}</text>"#, cx(i as i32), cy(0) - 6, lane.code).unwrap();
        writeln!(svg, r#"    <text x="{}" y="{}" text-anchor="middle" font-size="{FONT_SM}">{}</text>"#, cx(i as i32), cy(0) + 8, lane.name).unwrap();
        writeln!(svg, "  </a>").unwrap();
    }

    // Steps
    for step in STEPS {
        let lane = &LANES[step.col as usize];
        let fill = step.fill_override.unwrap_or(lane.fill);
        let bx = box_x(step.col);
        let by = box_y(step.row);
        writeln!(svg, "  <g>").unwrap();
        writeln!(svg, "    <title>{}</title>", esc(step.tooltip)).unwrap();
        writeln!(svg, r#"    <rect x="{bx}" y="{by}" width="{BOX_W}" height="{}" rx="{BOX_R}" fill="{fill}" stroke="{}" stroke-width="1.5" />"#, box_h(), lane.stroke).unwrap();
        let mut lines: Vec<&str> = step.label.split('\n').collect();
        if !step.cond.is_empty() {
            lines.push(step.cond);
        }
        let total_h = lines.len() as i32 * (FONT + 2);
        let start_y = cy(step.row) - total_h / 2 + FONT;
        for (j, line) in lines.iter().enumerate() {
            let fs = if line.starts_with('[') { FONT_SM } else { FONT };
            let clr = if line.starts_with('[') { CLR_COND } else { CLR_TEXT };
            writeln!(svg, r#"    <text x="{}" y="{}" text-anchor="middle" font-size="{fs}" fill="{clr}">{}</text>"#,
                cx(step.col), start_y + j as i32 * (FONT + 2), esc(line)).unwrap();
        }
        writeln!(svg, "  </g>").unwrap();
    }

    // Variables
    for v in VARIABLES {
        let lane = &LANES[v.set_col as usize];
        let bx = box_x(v.set_col);
        let by = box_y(v.set_row);
        let px = bx - PILL_W - 6;
        let py = by + box_h() / 2 - PILL_H / 2;
        writeln!(svg, "  <g>").unwrap();
        writeln!(svg, "    <title>{}</title>", esc(v.tooltip)).unwrap();
        writeln!(svg, r#"    <rect x="{px}" y="{py}" width="{PILL_W}" height="{PILL_H}" rx="{}" fill="white" stroke="{}" stroke-width="1" opacity="0.9" />"#, PILL_H / 2, lane.stroke).unwrap();
        writeln!(svg, r#"    <text x="{}" y="{}" text-anchor="middle" font-size="8" fill="{}" font-style="italic">{}</text>"#, px + PILL_W / 2, py + PILL_H / 2 + 4, lane.stroke, esc(v.name)).unwrap();
        writeln!(svg, r#"    <line x1="{bx}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="0.8" opacity="0.5" />"#, by + box_h() / 2, px + PILL_W, py + PILL_H / 2, lane.stroke).unwrap();
        writeln!(svg, "  </g>").unwrap();
    }

    // Intermediate artifacts — small labels between rows
    for im in INTERMEDIATES {
        let lane = &LANES[im.col as usize];
        let x = cx(im.col);
        // Position between row and row+1
        let y = cy(im.row) + ROW_H / 2 - 2;
        writeln!(svg, "  <g>").unwrap();
        writeln!(svg, "    <title>{}</title>", esc(im.tooltip)).unwrap();
        writeln!(svg, r#"    <text x="{x}" y="{y}" text-anchor="middle" font-size="7" fill="{}" font-style="italic" opacity="0.7">{}</text>"#,
            lane.stroke, esc(im.label)).unwrap();
        writeln!(svg, "  </g>").unwrap();
    }

    // Artifacts
    for art in ARTIFACTS {
        let lane = &LANES[art.col as usize];
        let bx = box_x(art.col);
        let art_row = TOTAL_ROWS - 1;
        let by = box_y(art_row);
        writeln!(svg, "  <g>").unwrap();
        writeln!(svg, "    <title>{}</title>", esc(art.tooltip)).unwrap();
        writeln!(svg, r#"    <rect x="{bx}" y="{by}" width="{BOX_W}" height="{}" fill="{}" stroke="{}" stroke-width="2" />"#, box_h(), lane.fill, lane.stroke).unwrap();
        let fold = 10;
        let fx = bx + BOX_W - fold;
        writeln!(svg, r#"    <polygon points="{fx},{by} {},{} {fx},{}" fill="white" stroke="{}" stroke-width="1" />"#, bx + BOX_W, by + fold, by + fold, lane.stroke).unwrap();
        let lines: Vec<&str> = art.label.split('\n').collect();
        let total_h = lines.len() as i32 * (FONT + 2);
        let start_y = cy(art_row) - total_h / 2 + FONT;
        for (j, line) in lines.iter().enumerate() {
            let fw = if j == 0 { "bold" } else { "normal" };
            writeln!(svg, r#"    <text x="{}" y="{}" text-anchor="middle" font-size="{FONT_SM}" font-weight="{fw}">{}</text>"#,
                cx(art.col), start_y + j as i32 * (FONT + 2), esc(line)).unwrap();
        }
        writeln!(svg, "  </g>").unwrap();
    }

    // Deferred profile expansion annotation — near the bottom, above the artifact row
    {
        let ann_x = MARGIN_LR + 2;
        let ann_y = cy(TOTAL_ROWS - 2) + ROW_H / 2 + 4;
        writeln!(svg, r#"  <text x="{ann_x}" y="{ann_y}" font-size="8" fill="{CLR_COND}" font-style="italic">⤷ deferred profile expansion: per-profile [pp] steps instantiated after base_count is known</text>"#).unwrap();
    }

    // Legend with links to SRD sections
    let legend_y = MARGIN_TOP + TOTAL_ROWS * ROW_H + 4;
    writeln!(svg, r#"  <text x="{MARGIN_LR}" y="{legend_y}" font-size="8" fill="{CLR_LABEL}">"#).unwrap();
    writeln!(svg, r#"    <a href="02-data-model.md#28-dataset-facets">Facet codes (§2.8)</a>"#).unwrap();
    writeln!(svg, r#"     · <a href="12-dataset-import-flowchart.md#124-universal-flow-graph">Pipeline spec (§12)</a>"#).unwrap();
    writeln!(svg, r#"     · <a href="14-pipeline-dag-configurations.md">DAG configs (§14)</a>"#).unwrap();
    writeln!(svg, r#"     · <a href="15-facet-swimlane.md">This diagram (§15)</a>"#).unwrap();
    writeln!(svg, r#"     · <a href="20-unified-sort-normalize-extract.md">Unified pipeline (§20)</a>"#).unwrap();
    writeln!(svg, r#"     · <a href="03-invariants.md#313-stratification-invariant">Stratification (§3.13)</a>"#).unwrap();
    writeln!(svg, "  </text>").unwrap();
    let note_y = legend_y + 12;
    writeln!(svg, r#"  <text x="{MARGIN_LR}" y="{note_y}" font-size="8" fill="{CLR_COND}" font-style="italic">Note: profiles expand after base_count is known — see §3.13 Stratification Invariant</text>"#).unwrap();

    writeln!(svg, "</svg>").unwrap();
    print!("{svg}");
}
