// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Generate an SVG infographic that breaks down the test suite by
//! crate, layer (unit vs integration), and purpose.
//!
//! Walks the workspace, scans every `*.rs` file under each crate's
//! `src/` and `tests/` directories for `#[test]`-annotated functions,
//! classifies them via filename / module-path / test-name heuristics,
//! and emits a single self-contained SVG to stdout.
//!
//! Usage:
//!   cargo run -p tools --bin gen_test_infographic > docs/design/diagrams/test-breakdown.svg
//!
//! No dependencies beyond std — the SVG is written by string-building.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

// ═══════════════════════════════════════════════════════════════════════════
// Discovery
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct TestEntry {
    crate_name: String,
    layer: Layer,
    purpose: Purpose,
    /// Test function name; only used for de-duplication and tooltips.
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    file: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Layer {
    /// Lives under `<crate>/src/**`. Inline `#[test]` modules.
    Unit,
    /// Lives under `<crate>/tests/**`. One `#[test]` per integration binary.
    Integration,
    /// Lives under `<crate>/benches/**`. Counted as a separate flavour;
    /// most workspaces have none, but keep the slot.
    #[allow(dead_code)]
    Bench,
}

impl Layer {
    fn label(self) -> &'static str {
        match self {
            Layer::Unit => "unit",
            Layer::Integration => "integration",
            Layer::Bench => "bench",
        }
    }
    fn color(self) -> &'static str {
        match self {
            Layer::Unit => "#3b82f6",        // blue
            Layer::Integration => "#10b981", // emerald
            Layer::Bench => "#f59e0b",       // amber
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Purpose {
    key: &'static str,
    label: &'static str,
    color: &'static str,
}

const PURPOSES: &[(&str, &str, &str)] = &[
    ("e2e",        "End-to-end pipeline",   "#06b6d4"), // cyan
    ("adversarial","Adversarial / regression","#ef4444"), // red
    ("dag",        "DAG configuration",     "#a855f7"), // violet
    ("http",       "HTTP / remote access",  "#0ea5e9"), // sky
    ("typed",      "Typed access",          "#84cc16"), // lime
    ("facet",      "Facet / view access",   "#22c55e"), // green
    ("data",       "Data access",           "#14b8a6"), // teal
    ("cache",      "Cache / channel",       "#eab308"), // yellow
    ("transport",  "Transport",             "#f97316"), // orange
    ("check",      "Check / validation",    "#ec4899"), // pink
    ("import",     "Import / bootstrap",    "#8b5cf6"), // purple
    ("unit",       "In-crate unit",         "#3b82f6"), // blue (default for src/)
    ("other",      "Other integration",     "#64748b"), // slate
];

fn purpose_for(layer: Layer, file: &Path, _test_name: &str) -> Purpose {
    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let key: &'static str = match layer {
        Layer::Unit => "unit",
        Layer::Bench => "other",
        Layer::Integration => {
            // Prefix-based routing — integration test files tend to
            // group by purpose, with consistent name prefixes used by
            // this workspace (see veks/tests, vectordata/tests).
            if stem.starts_with("e2e") { "e2e" }
            else if stem.starts_with("adversarial") { "adversarial" }
            else if stem.starts_with("dag") { "dag" }
            else if stem.contains("http") { "http" }
            else if stem.starts_with("typed") { "typed" }
            else if stem.starts_with("facet") { "facet" }
            else if stem.starts_with("data") { "data" }
            else if stem.starts_with("cached") || stem.contains("cache") { "cache" }
            else if stem.starts_with("transport") { "transport" }
            else if stem.starts_with("check") { "check" }
            else if stem.contains("import") || stem.contains("bootstrap") { "import" }
            else { "other" }
        }
    };
    let (label, color) = PURPOSES.iter()
        .find(|(k, _, _)| *k == key)
        .map(|(_, l, c)| (*l, *c))
        .unwrap_or(("Other integration", "#64748b"));
    Purpose { key, label, color }
}

fn workspace_root() -> PathBuf {
    // The tools crate sits at `<root>/tools/`, so go up one level.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(&manifest).to_path_buf()
}

fn discover_crates(root: &Path) -> Vec<(String, PathBuf)> {
    let toml = std::fs::read_to_string(root.join("Cargo.toml")).unwrap_or_default();
    // Tiny hand-rolled extractor for the workspace `members = [ ... ]`
    // block — full TOML would mean adding a dep. The grammar is
    // well-controlled in this workspace, so prefix scanning suffices.
    let members_start = toml.find("members = [").map(|i| i + "members = [".len());
    let mut out = Vec::new();
    if let Some(start) = members_start {
        let tail = &toml[start..];
        let end = tail.find(']').unwrap_or(tail.len());
        for raw in tail[..end].split(',') {
            let name = raw.trim().trim_matches('"');
            if name.is_empty() { continue; }
            let path = root.join(name);
            if path.is_dir() {
                out.push((name.to_string(), path));
            }
        }
    }
    out.sort_by(|(a, _), (b, _)| a.cmp(b));
    out
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if !dir.is_dir() { return; }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for ent in entries.flatten() {
        let p = ent.path();
        if p.is_dir() {
            // Skip `target/` and hidden dirs; otherwise recurse.
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "target" || name.starts_with('.') { continue; }
            collect_rs_files(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

fn scan_tests_in_file(path: &Path) -> Vec<String> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut tests = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i].trim_start();
        // Match `#[test]`, `#[tokio::test]`, etc. Skip `#[cfg(test)]` mod
        // headers — those are containers, not tests.
        let is_test_attr = (l.starts_with("#[test]") || l.starts_with("#[test ")
            || l.starts_with("#[tokio::test")
            || l.starts_with("#[async_std::test"))
            && !l.starts_with("#[test_case");
        if is_test_attr {
            // Walk forward through attribute lines to the `fn` line.
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim_start().starts_with("#[") { j += 1; }
            if j < lines.len() {
                let fn_line = lines[j].trim_start();
                if let Some(rest) = fn_line.strip_prefix("fn ")
                    .or_else(|| fn_line.strip_prefix("pub fn "))
                    .or_else(|| fn_line.strip_prefix("async fn "))
                    .or_else(|| fn_line.strip_prefix("pub async fn "))
                {
                    let name = rest.split(|c: char| !c.is_alphanumeric() && c != '_')
                        .next().unwrap_or("");
                    if !name.is_empty() {
                        tests.push(name.to_string());
                    }
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    tests
}

fn discover_tests(root: &Path) -> Vec<TestEntry> {
    let mut all = Vec::new();
    for (crate_name, crate_dir) in discover_crates(root) {
        // Skip `tools` — its own tests would inflate the chart.
        if crate_name == "tools" { continue; }

        // Unit tests: under `<crate>/src/**/*.rs`.
        let mut src_files = Vec::new();
        collect_rs_files(&crate_dir.join("src"), &mut src_files);
        for f in &src_files {
            for t in scan_tests_in_file(f) {
                let purpose = purpose_for(Layer::Unit, f, &t);
                all.push(TestEntry {
                    crate_name: crate_name.clone(),
                    layer: Layer::Unit,
                    purpose,
                    name: t,
                    file: f.clone(),
                });
            }
        }
        // Integration tests: under `<crate>/tests/*.rs`.
        let mut t_files = Vec::new();
        collect_rs_files(&crate_dir.join("tests"), &mut t_files);
        for f in &t_files {
            // Skip helper modules under tests/support/ — not actual
            // test entry points. The `#[test]` scan would skip them
            // anyway since they don't carry test annotations, but
            // explicit is friendlier.
            if f.components().any(|c| c.as_os_str() == "support") { continue; }
            for t in scan_tests_in_file(f) {
                let purpose = purpose_for(Layer::Integration, f, &t);
                all.push(TestEntry {
                    crate_name: crate_name.clone(),
                    layer: Layer::Integration,
                    purpose,
                    name: t,
                    file: f.clone(),
                });
            }
        }
    }
    all
}

// ═══════════════════════════════════════════════════════════════════════════
// SVG rendering
// ═══════════════════════════════════════════════════════════════════════════

const W: i32 = 1400;
const H: i32 = 1000;

fn svg_header(out: &mut String) {
    let _ = write!(out,
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">
<defs>
  <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"  stop-color="#0f172a"/>
    <stop offset="100%" stop-color="#1e293b"/>
  </linearGradient>
  <linearGradient id="accentGrad" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%"  stop-color="#06b6d4"/>
    <stop offset="100%" stop-color="#3b82f6"/>
  </linearGradient>
  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
    <feGaussianBlur stdDeviation="3" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<rect x="0" y="0" width="{W}" height="{H}" fill="url(#bgGrad)"/>
"##);
}

fn svg_footer(out: &mut String) {
    out.push_str("</svg>\n");
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn render(tests: &[TestEntry]) -> String {
    let mut out = String::new();
    svg_header(&mut out);

    // ── Header band ───────────────────────────────────────────────
    let _ = write!(out,
        r##"<text x="60" y="70" fill="#f8fafc" font-size="34" font-weight="700">vectordata-rs · test suite</text>
<text x="60" y="100" fill="#94a3b8" font-size="14">breakdown by crate, layer, and purpose</text>
<text x="{W}" y="70" text-anchor="end" fill="#94a3b8" font-size="13">generated {date}</text>
"##,
        W = W - 60,
        date = chrono_now_utc(),
    );

    // ── Big number ────────────────────────────────────────────────
    let total = tests.len();
    let _ = write!(out,
        r##"<text x="60" y="200" fill="url(#accentGrad)" font-size="120" font-weight="800" filter="url(#glow)">{total}</text>
<text x="60" y="240" fill="#cbd5e1" font-size="18">test functions across the workspace</text>
"##);

    // ── Layer breakdown row (right of big number) ────────────────
    let mut by_layer: BTreeMap<Layer, usize> = BTreeMap::new();
    for t in tests { *by_layer.entry(t.layer).or_default() += 1; }
    let layer_order = [Layer::Unit, Layer::Integration, Layer::Bench];
    let mut x_off = 360i32;
    for l in &layer_order {
        let n = by_layer.get(l).copied().unwrap_or(0);
        if n == 0 { continue; }
        let pct = 100.0 * n as f32 / total.max(1) as f32;
        let _ = write!(out,
            r##"<g transform="translate({x},150)">
  <text fill="{c}" font-size="48" font-weight="700">{n}</text>
  <text y="20" fill="#cbd5e1" font-size="13">{label} tests</text>
  <text y="38" fill="#64748b" font-size="11">{pct:.0}%</text>
</g>
"##,
            x = x_off, c = l.color(), n = n, label = l.label(), pct = pct,
        );
        x_off += 180;
    }

    // ── Per-crate stacked bars ───────────────────────────────────
    out.push_str(r##"<g transform="translate(60,300)">
<text fill="#f8fafc" font-size="20" font-weight="600">Per-crate breakdown</text>
<text y="22" fill="#94a3b8" font-size="12">stacked: unit (blue) + integration (emerald)</text>
"##);

    let mut by_crate: BTreeMap<String, BTreeMap<Layer, usize>> = BTreeMap::new();
    for t in tests {
        *by_crate.entry(t.crate_name.clone()).or_default()
            .entry(t.layer).or_default() += 1;
    }
    let mut crates: Vec<(String, usize)> = by_crate.iter()
        .map(|(n, m)| (n.clone(), m.values().sum())).collect();
    crates.sort_by(|a, b| b.1.cmp(&a.1));
    let max_crate = crates.first().map(|(_, n)| *n).unwrap_or(1).max(1);

    let crate_row_h = 28i32;
    let crate_label_w = 180i32;
    let crate_bar_w_max = 800i32;
    for (i, (cname, total_c)) in crates.iter().enumerate() {
        let y = 50 + (i as i32) * crate_row_h;
        let m = &by_crate[cname];
        let unit = m.get(&Layer::Unit).copied().unwrap_or(0);
        let intg = m.get(&Layer::Integration).copied().unwrap_or(0);
        let unit_w = (unit as f32 / max_crate as f32 * crate_bar_w_max as f32) as i32;
        let intg_w = (intg as f32 / max_crate as f32 * crate_bar_w_max as f32) as i32;
        let _ = write!(out,
            r##"<text x="0" y="{ty}" fill="#e2e8f0" font-size="13" font-weight="500">{cn}</text>
<rect x="{xb}" y="{yb}" width="{uw}" height="18" fill="{cu}" rx="2"/>
<rect x="{xi}" y="{yb}" width="{iw}" height="18" fill="{ci}" rx="2"/>
<text x="{xt}" y="{ty}" fill="#94a3b8" font-size="12">{total_c}</text>
"##,
            ty = y + 14,
            cn = esc(cname),
            xb = crate_label_w,
            yb = y,
            uw = unit_w,
            cu = Layer::Unit.color(),
            xi = crate_label_w + unit_w,
            iw = intg_w,
            ci = Layer::Integration.color(),
            xt = crate_label_w + crate_bar_w_max + 12,
            total_c = total_c,
        );
    }
    out.push_str("</g>\n");

    // ── Donut: tests by purpose ──────────────────────────────────
    let mut by_purpose: BTreeMap<&'static str, (usize, &'static str, &'static str)> = BTreeMap::new();
    for t in tests {
        let entry = by_purpose.entry(t.purpose.key)
            .or_insert((0, t.purpose.label, t.purpose.color));
        entry.0 += 1;
    }
    let mut purposes: Vec<_> = by_purpose.into_iter().collect();
    purposes.sort_by(|a, b| b.1.0.cmp(&a.1.0));

    let cx = 1130i32;
    let cy = 480i32;
    let r_outer = 160.0f32;
    let r_inner = 95.0f32;
    let total_f = total.max(1) as f32;
    let mut start_angle = -std::f32::consts::FRAC_PI_2;
    let _ = write!(out,
        r##"<g transform="translate(0,0)">
<text x="{cx}" y="290" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="600">Tests by purpose</text>
<text x="{cx}" y="310" text-anchor="middle" fill="#94a3b8" font-size="12">grouped by file/name heuristic</text>
"##, cx = cx);
    for (_key, (n, _label, color)) in &purposes {
        let frac = *n as f32 / total_f;
        let end_angle = start_angle + frac * 2.0 * std::f32::consts::PI;
        let large = if frac > 0.5 { 1 } else { 0 };
        let (x1, y1) = (cx as f32 + r_outer * start_angle.cos(),
                        cy as f32 + r_outer * start_angle.sin());
        let (x2, y2) = (cx as f32 + r_outer * end_angle.cos(),
                        cy as f32 + r_outer * end_angle.sin());
        let (x3, y3) = (cx as f32 + r_inner * end_angle.cos(),
                        cy as f32 + r_inner * end_angle.sin());
        let (x4, y4) = (cx as f32 + r_inner * start_angle.cos(),
                        cy as f32 + r_inner * start_angle.sin());
        let path = format!(
            "M {x1:.1} {y1:.1} A {ro} {ro} 0 {l} 1 {x2:.1} {y2:.1} \
             L {x3:.1} {y3:.1} A {ri} {ri} 0 {l} 0 {x4:.1} {y4:.1} Z",
            ro = r_outer as i32, ri = r_inner as i32, l = large,
        );
        let _ = write!(out,
            r##"<path d="{p}" fill="{c}" stroke="#0f172a" stroke-width="1.5"/>"##,
            p = path, c = color,
        );
        start_angle = end_angle;
    }
    let _ = write!(out,
        r##"<text x="{cx}" y="{cy}" text-anchor="middle" fill="#f8fafc" font-size="36" font-weight="700">{total}</text>
<text x="{cx}" y="{ly}" text-anchor="middle" fill="#94a3b8" font-size="13">total</text>
</g>
"##, cx = cx, cy = cy + 8, ly = cy + 30);

    // ── Purpose legend ───────────────────────────────────────────
    let lx = 920i32;
    let mut ly = 680i32;
    out.push_str(r#"<g>"#);
    for (_key, (n, label, color)) in &purposes {
        let pct = 100.0 * *n as f32 / total_f;
        let _ = write!(out,
            r##"<rect x="{lx}" y="{y}" width="14" height="14" rx="3" fill="{c}"/>
<text x="{lx2}" y="{ty}" fill="#e2e8f0" font-size="13">{label}</text>
<text x="{lx3}" y="{ty}" fill="#94a3b8" font-size="13" text-anchor="end">{n} · {pct:.0}%</text>
"##,
            lx = lx, lx2 = lx + 24, lx3 = lx + 380,
            y = ly, ty = ly + 12,
            c = color, label = esc(label), n = n, pct = pct,
        );
        ly += 22;
    }
    out.push_str("</g>\n");

    // ── Footer ───────────────────────────────────────────────────
    let _ = write!(out,
        r##"<text x="60" y="{y}" fill="#475569" font-size="11">scan: walk &lt;crate&gt;/src &amp; &lt;crate&gt;/tests for #[test] · purpose inferred from file prefix · regenerate with `cargo run -p tools --bin gen_test_infographic`</text>
"##,
        y = H - 30,
    );

    svg_footer(&mut out);
    out
}

/// Cheap UTC date stamp without bringing in `chrono`. Format: YYYY-MM-DD.
fn chrono_now_utc() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // Days from epoch (1970-01-01).
    let days = secs / 86_400;
    // Convert days→YMD using the standard civil-from-days algorithm
    // (Howard Hinnant). Self-contained, no leap-year drama.
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let root = workspace_root();
    let tests = discover_tests(&root);
    eprintln!("scanned {} test functions across the workspace", tests.len());
    let svg = render(&tests);
    println!("{}", svg);
}
