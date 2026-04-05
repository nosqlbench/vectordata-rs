// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Generate the prepare-vectors internal data flow SVG diagram.
//!
//! Produces a grid-oriented flowchart showing the internal phases,
//! data transformations, and artifact outputs of the prepare-vectors
//! step (sort + dedup). Normalization and zero detection happen during
//! extraction (SRD §20.4), not here.
//!
//! Usage:
//!   cargo run -p tools --bin gen_prepare_flow > docs/design/diagrams/20-prepare-vectors-flow.svg

use std::fmt::Write;

// ═══════════════════════════════════════════════════════════════════════════
// Layout constants
// ═══════════════════════════════════════════════════════════════════════════

const COL_W: i32 = 200;
const ROW_H: i32 = 64;
const PAD_X: i32 = 10;
const PAD_Y: i32 = 8;
const BOX_R: i32 = 6;
const MARGIN_LR: i32 = 40;
const MARGIN_TOP: i32 = 50;
const MARGIN_BOT: i32 = 30;
const FONT: i32 = 11;
const FONT_SM: i32 = 9;
const FONT_TITLE: i32 = 14;

const CLR_DETAIL: &str = "#666";
const CLR_NOTE: &str = "#999";
const CLR_LEGEND: &str = "#888";

// ═══════════════════════════════════════════════════════════════════════════
// Data model
// ═══════════════════════════════════════════════════════════════════════════

struct Box {
    row: i32,
    col: i32,
    colspan: i32,
    label: &'static str,
    detail: &'static str,
    fill: &'static str,
    stroke: &'static str,
    tooltip: &'static str,
}

struct Arrow {
    from_row: i32,
    from_col: i32,
    to_row: i32,
    to_col: i32,
    label: &'static str,
    color: &'static str,
}

struct Artifact {
    row: i32,
    col: i32,
    label: &'static str,
    fill: &'static str,
    stroke: &'static str,
    tooltip: &'static str,
}

struct Note {
    row: i32,
    col: i32,
    text: &'static str,
}

// ═══════════════════════════════════════════════════════════════════════════
// Data — internal flow of prepare-vectors
// ═══════════════════════════════════════════════════════════════════════════

const COLS: i32 = 4;
const ROWS: i32 = 7;

const BOXES: &[Box] = &[
    // Row 0: Input
    Box { row: 0, col: 1, colspan: 2, label: "Source Vectors", detail: "(fvec/mvec, any order)",
        fill: "#e8f5e9", stroke: "#4caf50",
        tooltip: "Raw source vector file — may contain duplicates.\nAll vectors in their original order." },

    // Row 1: Phase 1 — create sorted runs
    Box { row: 1, col: 0, colspan: 4, label: "Phase 1: Create Sorted Runs (per segment)", detail: "Read prefix components → sort by 10-dim prefix → write run file",
        fill: "#e3f2fd", stroke: "#2196f3",
        tooltip: "External merge sort: read prefix components (first 10 dims) from source,\nsort by 10-component lexicographic prefix keys, write ordinal+prefix to run files.\nFull vectors are NOT read during this phase." },

    // Row 2: In-segment operations
    Box { row: 2, col: 0, colspan: 2, label: "Sort Segment", detail: "Lexicographic order (10-dim prefix)",
        fill: "#e3f2fd", stroke: "#2196f3",
        tooltip: "Sort records within the segment by prefix component values.\n10-dim prefix keys avoid full-vector reads for sorting." },
    Box { row: 2, col: 2, colspan: 2, label: "Write Run File", detail: "(ordinal, prefix) records",
        fill: "#e3f2fd", stroke: "#2196f3",
        tooltip: "Write sorted (ordinal, prefix[0..10]) records to run file.\nNo full vector data — just ordinals and prefix keys." },

    // Row 3: Phase 2 — k-way merge + dedup
    Box { row: 3, col: 0, colspan: 4, label: "Phase 2: K-Way Merge + Dedup", detail: "Merge sorted runs → detect duplicates via prefix + full-vector comparison",
        fill: "#e3f2fd", stroke: "#2196f3",
        tooltip: "Merge all sorted run files using prefix-key comparison.\nWithin prefix collision groups, read full vectors for bitwise equality.\nOutput: sorted_ordinals.ivec (canonical order, dups elided)." },

    // Row 4: Write report
    Box { row: 4, col: 1, colspan: 2, label: "Phase 3: Write Report", detail: "Persist dedup statistics",
        fill: "#f5f5f5", stroke: "#9e9e9e",
        tooltip: "Write dedup_report.json with statistics.\nSave duplicate_count to variables.yaml.\nNormalization and zero detection happen during extraction (SRD §20.4)." },
];

const ARROWS: &[Arrow] = &[
    Arrow { from_row: 0, from_col: 2, to_row: 1, to_col: 2, label: "", color: "#4caf50" },
    Arrow { from_row: 1, from_col: 2, to_row: 2, to_col: 1, label: "segment in RAM", color: "#2196f3" },
    Arrow { from_row: 2, from_col: 1, to_row: 2, to_col: 3, label: "", color: "#2196f3" },
    Arrow { from_row: 2, from_col: 1, to_row: 3, to_col: 2, label: "run files", color: "#2196f3" },
    Arrow { from_row: 3, from_col: 2, to_row: 4, to_col: 2, label: "", color: "#9e9e9e" },
];

const ARTIFACTS: &[Artifact] = &[
    Artifact { row: 5, col: 0, label: "sorted_ordinals\n.ivec",
        fill: "#e8f5e9", stroke: "#4caf50",
        tooltip: "Canonical sorted ordinal index (dups elided).\nUsed by generate-shuffle and extract-base." },
    Artifact { row: 5, col: 1, label: "dedup_duplicates\n.ivec",
        fill: "#fce4ec", stroke: "#e91e63",
        tooltip: "Ordinals of exact duplicate vectors.\nRecorded for provenance." },
    Artifact { row: 5, col: 2, label: "dedup_report\n.json",
        fill: "#f5f5f5", stroke: "#9e9e9e",
        tooltip: "Statistics report: unique count, dup count,\nprefix width, processing time." },
    Artifact { row: 5, col: 3, label: "variables.yaml\n(duplicate_count)",
        fill: "#f5f5f5", stroke: "#9e9e9e",
        tooltip: "Saved variable: duplicate_count.\nNorm stats (zero_count, is_normalized) are\nset by extract-base during extraction." },
];

const NOTES: &[Note] = &[
    Note { row: 4, col: -1, text: "Normalization +\nzero detection\nhappen during\nextraction\n(SRD §20.4)" },
];

// ═══════════════════════════════════════════════════════════════════════════
// Geometry helpers
// ═══════════════════════════════════════════════════════════════════════════

fn cx(col: i32) -> i32 { MARGIN_LR + col * COL_W + COL_W / 2 }
fn cy(row: i32) -> i32 { MARGIN_TOP + row * ROW_H + ROW_H / 2 }
fn box_x(col: i32) -> i32 { MARGIN_LR + col * COL_W + PAD_X }
fn box_y(row: i32) -> i32 { MARGIN_TOP + row * ROW_H + PAD_Y }
fn box_w(colspan: i32) -> i32 { colspan * COL_W - 2 * PAD_X }
fn box_h() -> i32 { ROW_H - 2 * PAD_Y }

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

// ═══════════════════════════════════════════════════════════════════════════
// SVG rendering
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let w = MARGIN_LR * 2 + COLS * COL_W;
    let h = MARGIN_TOP + (ROWS - 1) * ROW_H + MARGIN_BOT;
    let mut svg = String::with_capacity(8192);

    writeln!(svg, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="Helvetica, Arial, sans-serif">"#).unwrap();
    writeln!(svg, r#"  <rect width="{w}" height="{h}" fill="white" />"#).unwrap();

    // Arrow marker defs
    writeln!(svg, "  <defs>").unwrap();
    let colors: Vec<&str> = vec!["#4caf50", "#ff9800", "#2196f3", "#9c27b0", "#f44336", "#9e9e9e"];
    for color in &colors {
        let id = color.replace('#', "c");
        writeln!(svg, r#"    <marker id="arrow_{id}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">"#).unwrap();
        writeln!(svg, r#"      <polygon points="0,0 8,3 0,6" fill="{color}" />"#).unwrap();
        writeln!(svg, "    </marker>").unwrap();
    }
    writeln!(svg, "  </defs>").unwrap();

    // Title
    writeln!(svg, r#"  <text x="{}" y="30" text-anchor="middle" font-size="{FONT_TITLE}" font-weight="bold">prepare-vectors: Internal Data Flow (SRD §20.3)</text>"#, w / 2).unwrap();

    // Boxes
    for b in BOXES {
        let bx = box_x(b.col);
        let by = box_y(b.row);
        let bw = box_w(b.colspan);
        let bh = box_h();
        writeln!(svg, "  <g>").unwrap();
        writeln!(svg, "    <title>{}</title>", esc(b.tooltip)).unwrap();
        writeln!(svg, r#"    <rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="{BOX_R}" fill="{}" stroke="{}" stroke-width="1.5" />"#, b.fill, b.stroke).unwrap();
        // Label (first line bold, second line normal)
        let lines: Vec<&str> = b.label.split('\n').collect();
        let total_lines = lines.len() + 1; // +1 for detail
        let line_h = FONT + 3;
        let start_y = cy(b.row) - (total_lines as i32 * line_h) / 2 + FONT;
        let mid_x = bx + bw / 2;
        for (j, line) in lines.iter().enumerate() {
            writeln!(svg, r#"    <text x="{mid_x}" y="{}" text-anchor="middle" font-size="{FONT}" font-weight="bold">{}</text>"#,
                start_y + j as i32 * line_h, esc(line)).unwrap();
        }
        // Detail line (smaller, gray)
        writeln!(svg, r#"    <text x="{mid_x}" y="{}" text-anchor="middle" font-size="{FONT_SM}" fill="{CLR_DETAIL}">{}</text>"#,
            start_y + lines.len() as i32 * line_h, esc(b.detail)).unwrap();
        writeln!(svg, "  </g>").unwrap();
    }

    // Arrows
    for a in ARROWS {
        let marker_id = format!("arrow_{}", a.color.replace('#', "c"));
        if a.from_row == a.to_row {
            // Horizontal arrow
            let y = cy(a.from_row);
            let x1 = box_x(a.from_col) + box_w(1) - PAD_X;
            let x2 = box_x(a.to_col) + PAD_X;
            writeln!(svg, r#"  <line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{}" stroke-width="1.5" marker-end="url(#{marker_id})" />"#, a.color).unwrap();
            if !a.label.is_empty() {
                let lx = (x1 + x2) / 2;
                writeln!(svg, r#"  <text x="{lx}" y="{}" text-anchor="middle" font-size="{FONT_SM}" fill="{}">{}</text>"#,
                    y - 5, a.color, esc(a.label)).unwrap();
            }
        } else {
            // Vertical arrow
            let x = cx(a.from_col);
            let y1 = box_y(a.from_row) + box_h();
            let y2 = box_y(a.to_row);
            let to_x = cx(a.to_col);
            if x == to_x {
                writeln!(svg, r#"  <line x1="{x}" y1="{y1}" x2="{to_x}" y2="{y2}" stroke="{}" stroke-width="1.5" marker-end="url(#{marker_id})" />"#, a.color).unwrap();
            } else {
                // L-shaped: down then across
                let mid_y = (y1 + y2) / 2;
                writeln!(svg, r#"  <polyline points="{x},{y1} {x},{mid_y} {to_x},{mid_y} {to_x},{y2}" fill="none" stroke="{}" stroke-width="1.5" marker-end="url(#{marker_id})" />"#, a.color).unwrap();
            }
            if !a.label.is_empty() {
                let lx = (x + to_x) / 2 + 5;
                let ly = (y1 + y2) / 2 - 3;
                writeln!(svg, r#"  <text x="{lx}" y="{ly}" font-size="{FONT_SM}" fill="{}">{}</text>"#,
                    a.color, esc(a.label)).unwrap();
            }
        }
    }

    // Artifacts (document-fold shape)
    for art in ARTIFACTS {
        let bx = box_x(art.col);
        let by = box_y(art.row);
        let bw = box_w(1);
        let bh = box_h();
        let fold = 8;
        writeln!(svg, "  <g>").unwrap();
        writeln!(svg, "    <title>{}</title>", esc(art.tooltip)).unwrap();
        writeln!(svg, r#"    <rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{}" stroke="{}" stroke-width="1.5" />"#, art.fill, art.stroke).unwrap();
        let fx = bx + bw - fold;
        writeln!(svg, r#"    <polygon points="{fx},{by} {},{} {fx},{}" fill="white" stroke="{}" stroke-width="1" />"#, bx + bw, by + fold, by + fold, art.stroke).unwrap();
        let lines: Vec<&str> = art.label.split('\n').collect();
        let total_h = lines.len() as i32 * (FONT + 2);
        let start_y = cy(art.row) - total_h / 2 + FONT;
        let mid_x = bx + bw / 2;
        for (j, line) in lines.iter().enumerate() {
            let fw = if j == 0 { "bold" } else { "normal" };
            writeln!(svg, r#"    <text x="{mid_x}" y="{}" text-anchor="middle" font-size="{FONT_SM}" font-weight="{fw}">{}</text>"#,
                start_y + j as i32 * (FONT + 2), esc(line)).unwrap();
        }
        writeln!(svg, "  </g>").unwrap();
    }

    // Notes (italic, left-aligned)
    for note in NOTES {
        let x = if note.col < 0 {
            MARGIN_LR - 5
        } else {
            box_x(note.col) + box_w(1) + 5
        };
        let y = cy(note.row);
        for (j, line) in note.text.split('\n').enumerate() {
            let anchor = if note.col < 0 { "end" } else { "start" };
            writeln!(svg, r#"  <text x="{x}" y="{}" text-anchor="{anchor}" font-size="{FONT_SM}" fill="{CLR_NOTE}" font-style="italic">{}</text>"#,
                y - 8 + j as i32 * (FONT_SM + 2), esc(line)).unwrap();
        }
    }

    // Arrows from Phase 3 to artifacts
    for art in ARTIFACTS {
        let x = cx(art.col);
        let y1 = box_y(5) + box_h();
        let y2 = box_y(art.row);
        let marker_id = format!("arrow_{}", art.stroke.replace('#', "c"));
        writeln!(svg, r#"  <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{}" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#{marker_id})" />"#, art.stroke).unwrap();
    }

    // Legend
    let ly = h - 15;
    writeln!(svg, r#"  <text x="{MARGIN_LR}" y="{ly}" font-size="8" fill="{CLR_LEGEND}">"#).unwrap();
    writeln!(svg, r#"    <a href="20-unified-sort-normalize-extract.md">SRD §20 — Unified Sort–Deduplicate–Extract Pipeline</a>"#).unwrap();
    writeln!(svg, r#"     · <a href="19-zero-vector-detection.md">§19 — Zero Detection</a>"#).unwrap();
    writeln!(svg, "  </text>").unwrap();

    writeln!(svg, "</svg>").unwrap();
    print!("{}", svg);
}
