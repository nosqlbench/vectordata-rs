// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Rustdoc examples by type (SRD element access, §13.11): every item
//! on the list below carries a fenced example that `cargo test --doc`
//! runs, no example in the crate is `ignore` or `no_run`, and the
//! element types named across the crate's examples are exactly
//! [`ElementType::ALL`].
//!
//! The parser is deliberately small: a doc comment is the run of
//! `///` (or `//!`) lines directly above an item, a fenced block is
//! what lies between two ```` ``` ```` lines inside it, and a type is
//! named when a token of a block is its name, bare or as the stem of
//! an extension (`bf16` in `.bf16vecs`).

use std::path::{Path, PathBuf};

use veks_io::ElementType;

/// The items that must carry a runnable example, by file and by the
/// start of their declaration line (trimmed). The declaration prefix
/// includes the opening `(` or `<` so that `open` is not `open_fvec`.
const DOCUMENTED_ITEMS: &[(&str, &str)] = &[
    ("src/lib.rs", "pub fn open("),
    ("src/lib.rs", "pub fn create("),
    ("src/lib.rs", "pub fn probe("),
    ("src/lib.rs", "pub fn open_varlen("),
    ("src/lib.rs", "pub fn create_varlen("),
    ("src/reader/xvec.rs", "pub struct XvecReader<"),
    ("src/reader/xvec.rs", "pub struct StreamReclaim<"),
    ("../veks-apicore/src/reader.rs", "fn read_into("),
    ("src/reader/mod.rs", "pub fn require_native<"),
    ("src/reader/scalar.rs", "pub struct ScalarReader<"),
    ("src/reader/vvec.rs", "pub struct IndexedVvecReader<"),
    ("../veks-apicore/src/reader.rs", "pub struct ConvertingReader<"),
    ("src/reader/any.rs", "pub fn open_vec_native("),
    ("src/reader/any.rs", "pub fn open_vec_as<"),
    ("src/reader/any.rs", "pub fn open_vec<"),
    ("src/reader/any.rs", "pub fn open_vvec<"),
    ("src/format.rs", "pub fn of("),
    ("src/format.rs", "pub fn from_extension("),
    ("src/format.rs", "pub fn preferred_extension("),
    ("src/format.rs", "pub fn element_type("),
    ("src/convert.rs", "pub fn convert_scalars("),
    ("src/xvec/reader.rs", "pub fn open("),
    ("src/xvec/reader.rs", "pub fn probe("),
    ("src/xvec/writer.rs", "pub fn open("),
];

/// The extension suffixes under which a type name still names the type.
const EXTENSION_SUFFIXES: &[&str] = &["vecs", "vec", "vvecs", "vvec"];

fn crate_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` file under `dir`, relative to the crate root, sorted.
fn source_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            source_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
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
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
        let source = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let Some(doc) = item_doc(&source, prefix) else {
            missing.push(format!("{file}: no line starts with `{prefix}`"));
            continue;
        };
        let blocks = fences(&doc);
        let runnable = blocks.iter().filter(|b| is_runnable(&b.info)).count();
        if runnable == 0 {
            missing.push(format!(
                "{file}: `{prefix}` has {} fenced block(s), none runnable",
                blocks.len()
            ));
        }
    }
    assert!(missing.is_empty(), "items without a runnable example:\n  {}", missing.join("\n  "));
}

#[test]
fn the_crate_root_examples_are_runnable() {
    let source = std::fs::read_to_string(crate_src().join("lib.rs")).unwrap();
    let root: Vec<&str> = source
        .lines()
        .take_while(|l| l.trim_start().starts_with("//!") || l.trim().is_empty() || l.trim_start().starts_with("//"))
        .filter_map(doc_text)
        .collect();
    let blocks = fences(&root);
    assert!(blocks.len() >= 3, "the crate root has {} fenced blocks", blocks.len());
    let held_back: Vec<&str> = blocks.iter().filter(|b| !is_runnable(&b.info)).map(|b| b.info.as_str()).collect();
    assert!(held_back.is_empty(), "crate-root blocks that do not run: {held_back:?}");
}

#[test]
fn no_example_in_the_crate_is_ignored_or_no_run() {
    let mut files = Vec::new();
    source_files(&crate_src(), &mut files);
    let mut held_back = Vec::new();
    for path in &files {
        let source = std::fs::read_to_string(path).unwrap();
        let doc: Vec<&str> = source.lines().filter_map(doc_text).collect();
        for block in fences(&doc) {
            if !is_runnable(&block.info) {
                held_back.push(format!("{}: ```{}", path.display(), block.info));
            }
        }
    }
    assert!(held_back.is_empty(), "examples that do not run:\n  {}", held_back.join("\n  "));
}

#[test]
fn the_examples_name_every_element_type() {
    let mut files = Vec::new();
    source_files(&crate_src(), &mut files);
    let mut named: Vec<ElementType> = Vec::new();
    let mut where_named: Vec<(ElementType, String)> = Vec::new();
    for path in &files {
        let source = std::fs::read_to_string(path).unwrap();
        let doc: Vec<&str> = source.lines().filter_map(doc_text).collect();
        for block in fences(&doc).iter().filter(|b| is_runnable(&b.info)) {
            for ty in types_named(&block.body) {
                if !named.contains(&ty) {
                    named.push(ty);
                    where_named.push((ty, path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap().display().to_string()));
                }
            }
        }
    }
    let missing: Vec<&str> = ElementType::ALL.iter().filter(|t| !named.contains(t)).map(|t| t.name()).collect();
    assert!(
        missing.is_empty(),
        "element types no runnable example names: {missing:?}\nnamed so far: {where_named:?}"
    );
    assert_eq!(named.len(), ElementType::ALL.len(), "a type outside ALL was counted: {named:?}");
}

// ── The parser itself ──────────────────────────────────────────────────

#[test]
fn a_fence_marked_no_run_or_ignore_is_not_runnable() {
    assert!(is_runnable(""));
    assert!(is_runnable("rust"));
    assert!(!is_runnable("rust,no_run"));
    assert!(!is_runnable("ignore"));
    assert!(!is_runnable("rust, ignore"));
}

#[test]
fn a_type_is_named_bare_or_as_an_extension_stem() {
    let found = types_named("let x = MmapReader::<u32>::open(&dir.join(\"a.bf16vecs\"));\n// 0u8 is not a name");
    assert_eq!(found, [ElementType::U32, ElementType::BF16]);
    assert!(types_named("f32vvecs i8vec").contains(&ElementType::F32));
    assert!(types_named("f32vvecs i8vec").contains(&ElementType::I8));
    assert!(types_named("F32 ElementType::I32 u8s").is_empty());
}

#[test]
fn an_item_doc_is_the_doc_lines_directly_above_it() {
    let source = "/// first\n/// ```\n/// code\n/// ```\n#[inline]\npub fn target() {}\n\n/// other\npub fn other() {}\n";
    let doc = item_doc(source, "pub fn target(").unwrap();
    assert_eq!(doc, ["first", "```", "code", "```"]);
    assert_eq!(item_doc(source, "pub fn other(").unwrap(), ["other"]);
    assert!(item_doc(source, "pub fn absent(").is_none());
}
