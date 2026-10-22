// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The workspace's crate boundaries, checked over its source tree
//! (`docs/design/srd-crate-boundaries.md`, CB-2).
//!
//! It lives in `tools` because `tools` is a default member with no
//! dependencies: a plain `cargo test` runs it, and it builds nothing.
//!
//! - **The format layer is veks-io** (FL-9): outside it, no production
//!   source reads an xvec dimension header with `i32::from_le_bytes`
//!   beside a record stride, or matches a vector-file extension literal
//!   (`"fvec"`, `"ivecs"`, `".bf16vecs"`, …) in a comparison or a
//!   `match` arm. Naming a file (`format!("{name}.ivec")`) is not
//!   matching one. `veks-datatypes`, whose extension table is the one
//!   table, is exempt, and so is `analyze_checkendian.rs`, which reads
//!   headers in both byte orders on purpose.
//! - **The streaming model is defined once**: the `VecSource`,
//!   `VecSink`, `VarlenSource` and `VarlenSink` traits only in
//!   veks-apicore, and `VecFormat` only in veks-io. Any crate may
//!   implement the traits.
//! - **The graph points down** (principle 2, AC-1, TR-8): every crate
//!   is on a layer of [`LAYERS`] and its `[dependencies]` name only
//!   workspace crates of lower layers, so veks-io and veks-transport —
//!   siblings — never reach each other and neither reaches vectordata;
//!   veks-apicore's dependencies are its vocabulary's ([`APICORE_DEPENDENCIES`]).
//!   Dev-dependencies are free: a test may stand up anything.
//! - **Only veks-transport talks HTTP** (CB-1, CB-2): no other crate
//!   declares reqwest outside `[dev-dependencies]`, and no production
//!   source outside veks-transport names `reqwest::`.
//!
//! The checks are textual and deliberately small: a line is code when
//! it is not a `//` comment, a file's tests (from a `#[cfg(test)]`
//! that opens a `mod`) are not production source, and "beside" is
//! within six code lines. Manifests are read line by line: a section
//! header, then `name = …` keys at the top level of a dependency table.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Files the format checks skip, by file name, with the reason.
const FORMAT_ALLOWED: &[(&str, &str)] =
    &[("analyze_checkendian.rs", "reads dimension headers in both byte orders to tell them apart")];

/// Crates whose sources the format checks skip: the format layer, and
/// the element model that owns the extension table.
const FORMAT_CRATES: &[&str] = &["veks-io", "veks-datatypes"];

/// Stems a vector-file extension is made of (`f` in `fvecs`, `bf16`
/// in `bf16vvec`).
const EXTENSION_STEMS: &[&str] = &[
    "f", "i", "b", "d", "m", "s", "h", "iv", "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f16", "bf16",
    "f32", "f64",
];

/// What a stride looks like near a header read.
const STRIDE_MARKS: &[&str] = &["xvec_record_bytes", "record_bytes", "entry_size", "stride"];

/// The workspace's layers, lowest first: a crate's `[dependencies]` name
/// only workspace crates of lower layers. One line of principle 2's
/// chain (`veks-datatypes` ← `veks-apicore` ← {`veks-io`,
/// `veks-transport`} ← `vectordata` ← `veks-core` ← `veks-pipeline` ←
/// `veks`; `vecd` → `vectordata`) per layer, with the leaf utilities
/// below it and the umbrella package on top.
const LAYERS: &[&[&str]] = &[
    &["veks-datatypes", "veks-completion-derive", "veks-anode", "veks-studies", "tools"],
    &["veks-simd", "veks-completion"],
    &["veks-apicore", "slabtastic"],
    &["veks-io", "veks-transport"],
    &["vectordata"],
    &["veks-core", "vecd"],
    &["veks-pipeline"],
    &["veks"],
    &["vectordata-rs"],
];

/// What veks-apicore may depend on (AC-1, as settled in the SRD's §12.1):
/// its vocabulary, and no implementation crate.
const APICORE_DEPENDENCIES: &[&str] =
    &["veks-datatypes", "thiserror", "serde", "serde_json", "serde_yaml", "indexmap", "url", "regex", "log"];

/// The one crate that may depend on reqwest outside its tests.
const HTTP_CRATE: &str = "veks-transport";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Every `.rs` file under `dir`, sorted.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = entries.map(|e| e.unwrap().path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The production source of every crate (and of the root package), as
/// `(crate, path)`.
fn production_sources(root: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let mut crates: Vec<_> = std::fs::read_dir(root.join("crates")).unwrap().map(|e| e.unwrap().path()).collect();
    crates.sort();
    for dir in crates {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let mut files = Vec::new();
        rust_files(&dir.join("src"), &mut files);
        out.extend(files.into_iter().map(|f| (name.clone(), f)));
    }
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    out.extend(files.into_iter().map(|f| ("vectordata-rs".to_string(), f)));
    out
}

/// The lines of `text` before its tests module, numbered from 1, with
/// comment lines blanked so they never match.
fn code_lines(text: &str) -> Vec<(usize, &str)> {
    let lines: Vec<&str> = text.lines().collect();
    let end = lines
        .windows(2)
        .position(|w| w[0].trim() == "#[cfg(test)]" && w[1].trim_start().starts_with("mod "))
        .unwrap_or(lines.len());
    lines[..end]
        .iter()
        .enumerate()
        .map(|(i, l)| (i + 1, if l.trim_start().starts_with("//") { "" } else { *l }))
        .collect()
}

/// Whether `lit` (a string literal's contents) is a vector-file
/// extension: an optional `.`, a stem, then `vec`, `vecs`, `vvec` or
/// `vvecs`.
fn is_vector_extension(lit: &str) -> bool {
    let lit = lit.strip_prefix('.').unwrap_or(lit);
    let Some(stem) = lit.strip_suffix("vecs").or_else(|| lit.strip_suffix("vec")) else { return false };
    let stem = stem.strip_suffix('v').filter(|s| !s.is_empty()).unwrap_or(stem);
    EXTENSION_STEMS.contains(&stem)
}

/// Whether `line` matches a vector-file extension literal: one compared
/// (`==`, `!=`, `ends_with(`, `Some(`) or used as a `match` arm.
fn matches_extension_literal(line: &str) -> bool {
    let mut rest = line;
    let mut offset = 0;
    while let Some(open) = rest.find('"') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('"') else { break };
        let lit = &after[..close];
        let before = line[..offset + open].trim_end();
        let following = after[close + 1..].trim_start();
        if is_vector_extension(lit)
            && (["==", "!=", "ends_with(", "Some(", "|"].iter().any(|m| before.ends_with(m))
                || ["=>", "|", "=="].iter().any(|m| following.starts_with(m)))
        {
            return true;
        }
        offset += open + 1 + close + 1;
        rest = &line[offset..];
    }
    false
}

/// Whether `line` looks like a record stride.
fn is_stride(line: &str) -> bool {
    STRIDE_MARKS.iter().any(|m| line.contains(m))
        || line.match_indices("4 + ").any(|(i, _)| !line[i + 4..].starts_with('4'))
}

/// The format-layer violations in `text`, as `line: reason`.
fn format_violations(text: &str) -> Vec<String> {
    let lines = code_lines(text);
    let mut found = Vec::new();
    for (i, &(n, line)) in lines.iter().enumerate() {
        if matches_extension_literal(line) {
            found.push(format!("{n}: matches an extension literal: {}", line.trim()));
        }
        if line.contains("i32::from_le_bytes") {
            let near = &lines[i.saturating_sub(6)..(i + 7).min(lines.len())];
            if near.iter().any(|(_, l)| is_stride(l)) {
                found.push(format!("{n}: reads a dimension header beside a stride: {}", line.trim()));
            }
        }
    }
    found
}

/// **No crate but veks-io knows the xvec record layout or matches
/// vector-file extensions** (FL-9).
#[test]
fn the_format_layer_is_veks_io() {
    let root = workspace_root();
    let mut violations = Vec::new();
    for (krate, path) in production_sources(&root) {
        let name = path.file_name().unwrap().to_string_lossy();
        if FORMAT_CRATES.contains(&krate.as_str()) || FORMAT_ALLOWED.iter().any(|(f, _)| *f == name) {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for v in format_violations(&text) {
            violations.push(format!("{}:{v}", path.strip_prefix(&root).unwrap().display()));
        }
    }
    assert!(
        violations.is_empty(),
        "format code outside veks-io — use veks_io's readers, writers, `xvec::record`, `ElementType` or `VecFormat`:\n{}",
        violations.join("\n")
    );
}

/// **The streaming model and the format enum are defined once**: the
/// stream traits in veks-apicore, `VecFormat` in veks-io.
#[test]
fn the_streaming_model_and_formats_are_defined_once() {
    let root = workspace_root();
    let owners: &[(&str, &str)] = &[
        ("trait VecSource", "veks-apicore"),
        ("trait VecSink", "veks-apicore"),
        ("trait VarlenSource", "veks-apicore"),
        ("trait VarlenSink", "veks-apicore"),
        ("enum VecFormat", "veks-io"),
    ];
    let mut violations = Vec::new();
    for (krate, path) in production_sources(&root) {
        let text = std::fs::read_to_string(&path).unwrap();
        for (n, line) in code_lines(&text) {
            for (decl, owner) in owners {
                let defines = line.split_whitespace().collect::<Vec<_>>().windows(2).any(|w| w.join(" ") == *decl);
                if defines && krate != *owner {
                    violations.push(format!("{}:{n}: {decl} belongs to {owner}", path.strip_prefix(&root).unwrap().display()));
                }
            }
        }
    }
    assert!(violations.is_empty(), "defined outside its owner:\n{}", violations.join("\n"));
}

/// A crate's manifest, as far as the graph checks read it.
#[derive(Debug, Default, PartialEq, Eq)]
struct Manifest {
    /// `[package] name`.
    name: String,
    /// What it depends on outside its tests: `[dependencies]`,
    /// `[build-dependencies]` and their `[target.….dependencies]` forms.
    dependencies: BTreeSet<String>,
    /// What only its tests, examples and benches depend on.
    dev_dependencies: BTreeSet<String>,
}

/// Read `text` (a `Cargo.toml`) for its name and dependency names. A
/// dependency is a `key = …` at the top level of a dependency table, or
/// a `[dependencies.<key>]` table; a multi-line inline table or array
/// contributes only its key.
fn parse_manifest(text: &str) -> Manifest {
    #[derive(PartialEq)]
    enum Section {
        Package,
        Normal,
        Dev,
        Other,
    }
    let mut m = Manifest::default();
    let mut section = Section::Other;
    let mut depth = 0i32;
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if depth == 0 && line.starts_with('[') {
            let header = line.trim_matches(|c| c == '[' || c == ']').trim();
            let (table, key) = match header.rsplit_once('.') {
                Some((t, k)) if !header.ends_with("dependencies") => (t, Some(k.trim_matches('"'))),
                _ => (header, None),
            };
            let kind = if table == "package" {
                Section::Package
            } else if table.starts_with("workspace") {
                // The workspace's shared tables declare no crate's edges.
                Section::Other
            } else if table.ends_with("dev-dependencies") {
                Section::Dev
            } else if table.ends_with("dependencies") {
                Section::Normal
            } else {
                Section::Other
            };
            match (key, &kind) {
                (Some(k), Section::Normal) => {
                    m.dependencies.insert(k.to_string());
                    section = Section::Other;
                }
                (Some(k), Section::Dev) => {
                    m.dev_dependencies.insert(k.to_string());
                    section = Section::Other;
                }
                _ => section = kind,
            }
            continue;
        }
        if depth == 0
            && let Some((key, value)) = line.split_once('=')
        {
            let key = key.trim().trim_matches('"').to_string();
            match section {
                Section::Package if key == "name" => m.name = value.trim().trim_matches('"').to_string(),
                Section::Normal => {
                    m.dependencies.insert(key);
                }
                Section::Dev => {
                    m.dev_dependencies.insert(key);
                }
                _ => {}
            }
        }
        depth += line.matches(['{', '[']).count() as i32 - line.matches(['}', ']']).count() as i32;
        depth = depth.max(0);
    }
    m
}

/// Every workspace manifest — each crate under `crates/` and the root
/// package — by crate name.
fn manifests(root: &Path) -> BTreeMap<String, Manifest> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root.join("crates")).unwrap().map(|e| e.unwrap().path()).collect();
    dirs.sort();
    dirs.push(root.to_path_buf());
    dirs.iter()
        .filter_map(|d| std::fs::read_to_string(d.join("Cargo.toml")).ok())
        .map(|text| parse_manifest(&text))
        .filter(|m| !m.name.is_empty())
        .map(|m| (m.name.clone(), m))
        .collect()
}

/// The layer `name` is on, if [`LAYERS`] places it.
fn layer_of(name: &str) -> Option<usize> {
    LAYERS.iter().position(|layer| layer.contains(&name))
}

/// The workspace crates `from` reaches through `[dependencies]`, itself
/// excluded.
fn reaches(graph: &BTreeMap<String, Manifest>, from: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from.to_string()];
    while let Some(name) = stack.pop() {
        let Some(m) = graph.get(&name) else { continue };
        for dep in m.dependencies.iter().filter(|d| graph.contains_key(*d)) {
            if seen.insert(dep.clone()) {
                stack.push(dep.clone());
            }
        }
    }
    seen.remove(from);
    seen
}

/// The violations of the graph rules in `graph`, as sentences.
fn graph_violations(graph: &BTreeMap<String, Manifest>) -> Vec<String> {
    let mut found = Vec::new();
    for (name, m) in graph {
        let Some(layer) = layer_of(name) else {
            found.push(format!("{name} is on no layer: place it in LAYERS"));
            continue;
        };
        for dep in m.dependencies.iter().filter(|d| graph.contains_key(*d)) {
            match layer_of(dep) {
                Some(l) if l < layer => {}
                _ => found.push(format!("{name} depends on {dep}, which is not on a lower layer")),
            }
        }
        if m.dependencies.contains("reqwest") && name != HTTP_CRATE {
            found.push(format!("{name} depends on reqwest; only {HTTP_CRATE} may (tests may, as a dev-dependency)"));
        }
    }
    for (from, to) in [
        ("veks-transport", "veks-io"),
        ("veks-io", "veks-transport"),
        ("veks-io", "vectordata"),
        ("veks-transport", "vectordata"),
    ] {
        if reaches(graph, from).contains(to) {
            found.push(format!("{from} reaches {to}"));
        }
    }
    if let Some(api) = graph.get("veks-apicore") {
        for dep in api.dependencies.iter().filter(|d| !APICORE_DEPENDENCIES.contains(&d.as_str())) {
            found.push(format!("veks-apicore depends on {dep}, outside its vocabulary"));
        }
    }
    found
}

/// **The graph points down, and only veks-transport depends on
/// reqwest** (principle 2, AC-1, TR-8, CB-1).
#[test]
fn the_dependency_graph_is_the_srds() {
    let graph = manifests(&workspace_root());
    for expected in ["veks-apicore", "veks-io", "veks-transport", "vectordata", "veks-pipeline", "veks", "vecd"] {
        assert!(graph.contains_key(expected), "no manifest for {expected}: {:?}", graph.keys());
    }
    let violations = graph_violations(&graph);
    assert!(violations.is_empty(), "the crate graph is not the SRD's:\n{}", violations.join("\n"));
}

/// Whether `line` (code) names reqwest.
fn names_reqwest(line: &str) -> bool {
    line.contains("reqwest::") || line.split_whitespace().collect::<Vec<_>>().windows(2).any(|w| w[0] == "use" && w[1].starts_with("reqwest"))
}

/// **No production source outside veks-transport names reqwest** (CB-2):
/// every other crate's requests go through veks-transport's `Transport`.
#[test]
fn only_veks_transport_names_reqwest() {
    let root = workspace_root();
    let mut violations = Vec::new();
    for (krate, path) in production_sources(&root) {
        if krate == HTTP_CRATE {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for (n, line) in code_lines(&text) {
            if names_reqwest(line) {
                violations.push(format!("{}:{n}: {}", path.strip_prefix(&root).unwrap().display(), line.trim()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "reqwest outside veks-transport — send through `vectordata::transport` / `veks_transport::Transport`:\n{}",
        violations.join("\n")
    );
}

/// The manifest reader and the graph rules, on the shapes they are for.
#[test]
fn the_graph_matchers_catch_what_they_are_for() {
    let text = r#"
[package]
name = "demo"   # the crate
version = "1.0.0"

[dependencies]
# reqwest = "0.13"   (commented out: not a dependency)
veks-io = { path = "../veks-io", features = [
    "npy",
    "slab",
] }
serde = "1"

[target.'cfg(unix)'.dependencies]
libc = "0.2"

[dependencies.url]
version = "2"

[dev-dependencies]
reqwest = { version = "0.13", features = ["blocking"] }

[features]
default = ["serde"]
"#;
    let m = parse_manifest(text);
    assert_eq!(m.name, "demo");
    let deps: Vec<&str> = m.dependencies.iter().map(String::as_str).collect();
    assert_eq!(deps, ["libc", "serde", "url", "veks-io"]);
    let dev: Vec<&str> = m.dev_dependencies.iter().map(String::as_str).collect();
    assert_eq!(dev, ["reqwest"]);

    let manifest = |name: &str, deps: &[&str]| Manifest {
        name: name.to_string(),
        dependencies: deps.iter().map(|d| d.to_string()).collect(),
        dev_dependencies: BTreeSet::new(),
    };
    let graph = |ms: Vec<Manifest>| -> BTreeMap<String, Manifest> { ms.into_iter().map(|m| (m.name.clone(), m)).collect() };
    let good = graph(vec![
        manifest("veks-apicore", &["veks-datatypes", "serde"]),
        manifest("veks-datatypes", &[]),
        manifest("veks-io", &["veks-apicore"]),
        manifest("veks-transport", &["veks-apicore", "reqwest"]),
        manifest("vectordata", &["veks-io", "veks-transport"]),
    ]);
    assert!(graph_violations(&good).is_empty(), "{:?}", graph_violations(&good));
    let bad = graph(vec![
        manifest("veks-apicore", &["veks-datatypes", "memmap2"]),
        manifest("veks-datatypes", &[]),
        manifest("veks-io", &["veks-apicore"]),
        manifest("veks-transport", &["veks-apicore", "veks-io"]),
        manifest("vectordata", &["veks-transport", "reqwest"]),
        manifest("newcomer", &[]),
    ]);
    let v = graph_violations(&bad);
    for expected in [
        "veks-transport depends on veks-io, which is not on a lower layer",
        "veks-transport reaches veks-io",
        "vectordata depends on reqwest",
        "veks-apicore depends on memmap2",
        "newcomer is on no layer",
    ] {
        assert!(v.iter().any(|s| s.starts_with(expected)), "missing {expected:?} in {v:#?}");
    }
    // Reaching is transitive.
    let chain = graph(vec![manifest("veks-io", &["a"]), manifest("a", &["vectordata"]), manifest("vectordata", &[])]);
    assert!(reaches(&chain, "veks-io").contains("vectordata"));

    assert!(names_reqwest("    let c = reqwest::blocking::Client::new();"));
    assert!(names_reqwest("use reqwest::header::HeaderMap;"));
    assert!(names_reqwest("use reqwest;"));
    assert!(!names_reqwest("let client = transport.send(req);"));
    let commented = "// reqwest::blocking::get(url)\n";
    assert!(code_lines(commented).iter().all(|(_, l)| !names_reqwest(l)));
}

/// The checks' own matchers, on the shapes they are for.
#[test]
fn the_matchers_catch_what_they_are_for() {
    for lit in ["fvec", "fvecs", ".ivecs", "bf16vecs", "ivvec", ".u8vvecs", "mvec"] {
        assert!(is_vector_extension(lit), "{lit}");
    }
    for lit in ["vec", "vecs", "slab", "f32", "neighbors", "fvecx", "vvec"] {
        assert!(!is_vector_extension(lit), "{lit}");
    }
    assert!(matches_extension_literal(r#"if ext == "fvecs" {"#));
    assert!(matches_extension_literal(r#"    "fvec" | "fvecs" => 4,"#));
    assert!(matches_extension_literal(r#"if name.ends_with(".ivecs") {"#));
    assert!(matches_extension_literal(r#"if path.extension() == Some("fvec") {"#));
    assert!(!matches_extension_literal(r#"let p = format!("{}.ivec", name);"#));
    assert!(!matches_extension_literal(r#"build_cache_path(dir, "neighbors", "ivec")"#));
    assert!(!matches_extension_literal(r#".unwrap_or("fvec");"#));

    let header_and_stride = "let dim = i32::from_le_bytes(h);\nlet stride = 4 + dim as usize * 4;\n";
    assert_eq!(format_violations(header_and_stride).len(), 1);
    let payload = "let v = i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());\n";
    assert!(format_violations(payload).is_empty(), "a packed i32 payload is not a header");
    let in_tests = "#[cfg(test)]\nmod tests {\n    if ext == \"fvecs\" {}\n}\n";
    assert!(format_violations(in_tests).is_empty());
    let commented = "// if ext == \"fvecs\" {}\n";
    assert!(format_violations(commented).is_empty());
}
