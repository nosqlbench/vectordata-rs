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
//!
//! The checks are textual and deliberately small: a line is code when
//! it is not a `//` comment, a file's tests (from a `#[cfg(test)]`
//! that opens a `mod`) are not production source, and "beside" is
//! within six code lines.

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
