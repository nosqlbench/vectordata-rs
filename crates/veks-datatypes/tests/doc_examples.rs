// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Rustdoc examples by type (SRD element access, §13.11): every item
//! on the list below carries a fenced example that `cargo test --doc`
//! runs, and the element types named across the crate's runnable
//! examples in the element-model modules are exactly
//! [`ElementType::ALL`]. The parser mirrors veks-io's guard: a doc
//! comment is the run of `///` lines directly above an item, a fenced
//! block lies between two ```` ``` ```` lines, and a type is named when
//! a token is its name, bare or as the stem of an extension.

use std::path::PathBuf;

use veks_datatypes::ElementType;

/// The items that must carry a runnable example, by file and by the
/// start of their declaration line (trimmed).
const DOCUMENTED_ITEMS: &[(&str, &str)] = &[
    ("src/element_type.rs", "pub fn parse_extension("),
    ("src/element_type.rs", "pub fn aliases("),
    ("src/conversion.rs", "pub const fn conversion_to("),
    ("src/conversion.rs", "pub fn convert<"),
    ("src/element.rs", "pub trait Element:"),
    ("src/element.rs", "pub enum Value {"),
    ("src/lib.rs", "macro_rules! dispatch_element"),
];

/// The modules whose runnable examples, together, must name every type.
const ELEMENT_MODULES: &[&str] = &["src/lib.rs", "src/element_type.rs", "src/conversion.rs", "src/element.rs"];

/// The extension suffixes under which a type name still names the type.
const EXTENSION_SUFFIXES: &[&str] = &["vecs", "vec", "vvecs", "vvec"];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The text of a doc-comment line without its marker, or `None` when
/// the line is not a doc comment.
fn doc_text(line: &str) -> Option<&str> {
    let t = line.trim_start();
    for marker in ["///", "//!"] {
        if let Some(rest) = t.strip_prefix(marker) {
            // Not a plain `////` rule, and tolerate a missing space.
            if rest.starts_with('/') {
                return None;
            }
            return Some(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    None
}

/// A fenced block in a doc comment: its info string and its body.
#[derive(Debug)]
struct Fence {
    info: String,
    body: String,
}

/// The fenced blocks in a run of doc-comment texts.
fn fences(doc_lines: &[&str]) -> Vec<Fence> {
    let mut out = Vec::new();
    let mut open: Option<Fence> = None;
    for line in doc_lines {
        if let Some(rest) = line.trim_start().strip_prefix("```") {
            match open.take() {
                Some(block) => out.push(block),
                None => open = Some(Fence { info: rest.trim().to_string(), body: String::new() }),
            }
        } else if let Some(block) = open.as_mut() {
            block.body.push_str(line);
            block.body.push('\n');
        }
    }
    assert!(open.is_none(), "an unclosed fence: {open:?}");
    out
}

/// Whether `cargo test --doc` runs a block with this info string.
fn is_runnable(info: &str) -> bool {
    !info
        .split(|c: char| c == ',' || c.is_whitespace())
        .any(|word| matches!(word, "ignore" | "no_run" | "text" | "compile_fail"))
}

/// The `///` lines directly above the first line of `source` whose
/// trimmed text starts with `prefix`, skipping attributes between.
fn item_doc<'a>(source: &'a str, prefix: &str) -> Option<Vec<&'a str>> {
    let lines: Vec<&str> = source.lines().collect();
    let at = lines.iter().position(|l| l.trim_start().starts_with(prefix))?;
    let mut doc = Vec::new();
    for line in lines[..at].iter().rev() {
        let t = line.trim_start();
        if t.starts_with("#[") {
            continue;
        }
        match doc_text(line) {
            Some(text) if t.starts_with("///") => doc.push(text),
            _ => break,
        }
    }
    doc.reverse();
    Some(doc)
}

/// The element types named by the tokens of `body`.
fn types_named(body: &str) -> Vec<ElementType> {
    let names: Vec<(&str, ElementType)> = ElementType::ALL.iter().map(|t| (t.name(), *t)).collect();
    let mut found = Vec::new();
    for token in body.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        for (name, ty) in &names {
            let stem_of_extension = token
                .strip_prefix(name)
                .is_some_and(|rest| EXTENSION_SUFFIXES.contains(&rest));
            if (token == *name || stem_of_extension) && !found.contains(ty) {
                found.push(*ty);
            }
        }
    }
    found
}

#[test]
fn every_listed_item_has_a_runnable_example() {
    let mut missing = Vec::new();
    for (file, prefix) in DOCUMENTED_ITEMS {
        let source = std::fs::read_to_string(crate_root().join(file)).unwrap();
        let doc = item_doc(&source, prefix).unwrap_or_else(|| panic!("{file}: no item starts with {prefix:?}"));
        let runnable = fences(&doc).iter().filter(|f| is_runnable(&f.info)).count();
        if runnable == 0 {
            missing.push(format!("{file}: {prefix}"));
        }
    }
    assert!(missing.is_empty(), "items without a runnable example:\n{}", missing.join("\n"));
}

/// Every example in this crate runs; between them they name every type.
#[test]
fn the_examples_name_every_element_type() {
    let mut named: Vec<(ElementType, String)> = Vec::new();
    for file in ELEMENT_MODULES {
        let source = std::fs::read_to_string(crate_root().join(file)).unwrap();
        let doc: Vec<&str> = source.lines().filter_map(doc_text).collect();
        for fence in fences(&doc).iter().filter(|f| is_runnable(&f.info)) {
            for ty in types_named(&fence.body) {
                if !named.iter().any(|(t, _)| *t == ty) {
                    named.push((ty, file.to_string()));
                }
            }
        }
    }
    let missing: Vec<&ElementType> = ElementType::ALL.iter().filter(|t| !named.iter().any(|(n, _)| n == *t)).collect();
    assert!(missing.is_empty(), "element types no runnable example names: {missing:?}; named: {named:?}");
}
