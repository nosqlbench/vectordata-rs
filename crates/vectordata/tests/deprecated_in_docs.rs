// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! No document teaches a deprecated call. The compiled rustdoc examples
//! are held by `#![doc(test(attr(deny(deprecated))))]` in each crate;
//! what the compiler never sees is held here: the Rust blocks of the
//! workspace's markdown (the sysref, the tutorials, the READMEs, a
//! crate's `docs/` and `AGENTS.md`), and the rustdoc blocks marked
//! `ignore` or `text` in the library crates' sources. A deprecated
//! item's own doc comment is exempt; so is prose, where a migration
//! table names the old call beside its replacement.

use std::path::{Path, PathBuf};

use regex::Regex;

/// The deprecated calls — and the removed ones, which a document must
/// not teach either — and what replaces each.
const DEPRECATED: &[(&str, &str)] = &[
    (
        r"\.(base_vectors|query_vectors|neighbor_indices|neighbor_distances|prefiltered_neighbor_indices|prefiltered_neighbor_distances|postfiltered_neighbor_indices|postfiltered_neighbor_distances|metadata_results)\(\)",
        "the TestDataViewExt `*_as::<T>()` call",
    ),
    (r"\bview\.facet\(", "`view.facet_as::<T>(name)`"),
    (r"\bopen_facet_typed\b", "`view.facet_as::<T>(name)`"),
    (r"\b(TypedReader|TypedElement|TypedAccessError|VvecElement)\b", "`facet_as`, `open_vec_as`, `ScalarReader`, `T: Element`"),
    (r"\bdispatch_typed!", "`dispatch_element!` or `open_vec_native`"),
    (r"\bcan_open_as\(", "`ElementType::conversion_to`"),
    (r"MmapReader(::<[^>]+>)?::open_(fvec|ivec|mvec|dvec|bvec|svec)\(", "`MmapReader::<T>::open`"),
    (r"ScalarMmapReader(::<[^>]+>)?::open_[a-z0-9]+\(", "`ScalarMmapReader::<T>::open`"),
    (r"\bScalarType\b", "`ElementType`"),
];

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The `.md` files directly in `dir`.
fn markdown_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    files
}

/// The `.rs` files under `dir`, recursively.
fn rust_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for path in entries.filter_map(|e| e.ok().map(|e| e.path())) {
        if path.is_dir() {
            rust_under(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

/// Every markdown document a user reads for the API.
fn documents() -> Vec<PathBuf> {
    let root = workspace();
    let mut files = vec![root.join("README.md")];
    files.extend(markdown_in(&root.join("docs/sysref")));
    files.extend(markdown_in(&root.join("docs/tutorials")));
    let mut crates: Vec<PathBuf> =
        std::fs::read_dir(root.join("crates")).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    crates.sort();
    for krate in crates {
        for name in ["README.md", "AGENTS.md"] {
            if krate.join(name).is_file() {
                files.push(krate.join(name));
            }
        }
        files.extend(markdown_in(&krate.join("docs")));
    }
    files
}

/// The lines of a markdown document's ```` ```rust ```` blocks, with
/// their line numbers.
fn rust_blocks_of_markdown(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut in_rust = false;
    let mut in_other = false;
    for (n, line) in text.lines().enumerate() {
        let t = line.trim_start();
        if let Some(info) = t.strip_prefix("```") {
            if in_rust || in_other {
                in_rust = false;
                in_other = false;
            } else if info.trim().split(',').next() == Some("rust") {
                in_rust = true;
            } else {
                in_other = true;
            }
            continue;
        }
        if in_rust {
            out.push((n + 1, line));
        }
    }
    out
}

/// The lines of a source file's uncompiled rustdoc blocks (`ignore`,
/// `text`), with their line numbers, outside a deprecated item's own
/// doc comment.
fn uncompiled_doc_blocks(text: &str) -> Vec<(usize, &str)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let is_doc = |l: &str| l.trim_start().starts_with("///") || l.trim_start().starts_with("//!");
        if !is_doc(lines[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < lines.len() && is_doc(lines[i]) {
            i += 1;
        }
        // The attributes between the comment and its item.
        let deprecated_item = lines[i..]
            .iter()
            .take_while(|l| l.trim_start().starts_with("#[") || l.trim_start().starts_with("since") || l.trim_start().starts_with("note") || l.trim() == ")]")
            .any(|l| l.contains("#[deprecated"));
        if deprecated_item {
            continue;
        }
        let mut in_block = false;
        for (k, line) in lines.iter().enumerate().take(i).skip(start) {
            let body = line.trim_start().trim_start_matches("///").trim_start_matches("//!").trim_start();
            if let Some(info) = body.strip_prefix("```") {
                if in_block {
                    in_block = false;
                } else {
                    let info = info.trim();
                    in_block = info.split(',').any(|w| w == "ignore" || w == "text");
                }
                continue;
            }
            if in_block {
                out.push((k + 1, *line));
            }
        }
    }
    out
}

fn offences(file: &Path, lines: &[(usize, &str)], patterns: &[(Regex, &str)]) -> Vec<String> {
    let root = workspace();
    let shown = file.strip_prefix(&root).unwrap_or(file).display().to_string();
    lines
        .iter()
        .flat_map(|(n, line)| {
            patterns
                .iter()
                .filter(|(re, _)| re.is_match(line))
                .map(|(_, instead)| format!("{shown}:{n}: {} — use {instead}", line.trim()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn patterns() -> Vec<(Regex, &'static str)> {
    DEPRECATED.iter().map(|(p, instead)| (Regex::new(p).unwrap(), *instead)).collect()
}

/// **No markdown example calls a deprecated item.** A sysref or README
/// block that does teaches the old API, and nothing compiles it.
#[test]
fn no_markdown_example_calls_a_deprecated_item() {
    let patterns = patterns();
    let docs = documents();
    assert!(docs.len() > 10, "found only {} documents: {docs:?}", docs.len());
    let found: Vec<String> = docs
        .iter()
        .flat_map(|path| {
            let text = std::fs::read_to_string(path).unwrap();
            offences(path, &rust_blocks_of_markdown(&text), &patterns)
        })
        .collect();
    assert!(found.is_empty(), "deprecated calls in markdown examples:\n{}", found.join("\n"));
}

/// **No uncompiled rustdoc example calls a deprecated item**, outside
/// the deprecated item's own documentation.
#[test]
fn no_uncompiled_rustdoc_example_calls_a_deprecated_item() {
    let patterns = patterns();
    let mut files = Vec::new();
    for krate in ["vectordata", "veks-io", "veks-datatypes", "veks-core", "veks-pipeline", "veks"] {
        rust_under(&workspace().join("crates").join(krate).join("src"), &mut files);
    }
    files.sort();
    assert!(files.len() > 100, "found only {} sources", files.len());
    let found: Vec<String> = files
        .iter()
        .flat_map(|path| {
            let text = std::fs::read_to_string(path).unwrap();
            offences(path, &uncompiled_doc_blocks(&text), &patterns)
        })
        .collect();
    assert!(found.is_empty(), "deprecated calls in uncompiled rustdoc examples:\n{}", found.join("\n"));
}
