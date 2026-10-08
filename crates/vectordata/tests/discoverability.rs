// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The properties that keep the canonical API findable, checked against
//! the source (`docs/design/srd-api-discoverability.md` §5):
//!
//! - every module says in its first line whether it is library API,
//!   CLI support, or internal (DX-19);
//! - library modules never print or exit (DX-8, DX-16);
//! - every Common tasks row names its call with a checked link, and the
//!   examples it names exist (DX-27);
//! - the public fetch-like entry points are exactly the budgeted ones,
//!   and the duplicates among them are deprecated (DX-30).
//!
//! Intra-doc links themselves — in the crate docs and in `AGENTS.md`,
//! which the crate renders as `_agents` — are checked by
//! `cargo doc` with `-D warnings` in CI. The CLI ↔ library map is
//! checked against the real command tree in `src/shell.rs`.

use std::path::{Path, PathBuf};

const LABELS: [&str; 3] = ["Library API.", "CLI support.", "Internal."];

fn src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` file under `src/` except the binary shim and test
/// modules, with its path relative to `src/`.
fn source_files() -> Vec<(String, String)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().is_some_and(|n| n != "bin") {
                    walk(&p, root, out);
                }
            } else if p.extension().is_some_and(|x| x == "rs")
                && p.file_name().is_some_and(|n| n != "tests.rs")
            {
                let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
                out.push((rel, std::fs::read_to_string(&p).unwrap()));
            }
        }
    }
    let mut out = Vec::new();
    walk(&src(), &src(), &mut out);
    out.sort();
    out
}

/// The first line of a file's `//!` docs, if it has any.
fn module_head(text: &str) -> Option<&str> {
    text.lines().find_map(|l| l.strip_prefix("//! "))
}

/// The label a file's module carries: its own head's, or the head of
/// the top-level module it sits under.
fn layer_of(rel: &str, files: &[(String, String)]) -> Option<&'static str> {
    let own = files.iter().find(|(r, _)| r == rel).and_then(|(_, t)| module_head(t));
    if let Some(l) = own.and_then(|h| LABELS.iter().find(|l| h.starts_with(**l))) {
        return Some(l);
    }
    let top = rel.split('/').next().unwrap();
    let top_file = format!("{top}/mod.rs");
    files
        .iter()
        .find(|(r, _)| *r == top_file)
        .and_then(|(_, t)| module_head(t))
        .and_then(|h| LABELS.iter().find(|l| h.starts_with(**l)).copied())
}

/// `text` with every `#[cfg(test)]` item removed, and comment lines
/// blanked, so what remains is the code that ships.
fn shipped_code(text: &str) -> String {
    let mut out = String::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim() == "#[cfg(test)]" {
            // Skip the item that follows: to the end of its first
            // balanced brace block, or its terminating semicolon.
            let mut depth = 0i32;
            let mut opened = false;
            for l in lines.by_ref() {
                for c in l.chars() {
                    match c {
                        '{' => {
                            depth += 1;
                            opened = true;
                        }
                        '}' => depth -= 1,
                        _ => {}
                    }
                }
                if (opened && depth == 0) || (!opened && l.trim_end().ends_with(';')) {
                    break;
                }
            }
            continue;
        }
        let t = line.trim_start();
        if t.starts_with("//") {
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// **Every module's first doc line names its layer** (DX-19), so a
/// reader landing on any module page knows whether it is the library.
#[test]
fn every_module_head_names_its_layer() {
    let files = source_files();
    let mut unlabeled = Vec::new();
    for (rel, text) in &files {
        if rel == "lib.rs" {
            continue;
        }
        match module_head(text) {
            Some(h) if LABELS.iter().any(|l| h.starts_with(l)) => {}
            Some(h) => unlabeled.push(format!("{rel}: {h}")),
            None => {}
        }
    }
    assert!(
        unlabeled.is_empty(),
        "module docs must open with one of {LABELS:?}:\n{}",
        unlabeled.join("\n")
    );
    // Every public module has docs to carry the label.
    let lib = std::fs::read_to_string(src().join("lib.rs")).unwrap();
    for line in lib.lines() {
        let Some(name) = line.strip_prefix("pub mod ").and_then(|l| l.strip_suffix(';')) else {
            continue;
        };
        let file = files
            .iter()
            .find(|(r, _)| *r == format!("{name}.rs") || *r == format!("{name}/mod.rs"))
            .unwrap_or_else(|| panic!("no file for pub mod {name}"));
        assert!(module_head(&file.1).is_some(), "pub mod {name} has no module docs");
    }
}

/// Files allowed to write to the terminal although their module is
/// library API, and why. Each renders output the caller asked for.
const RENDERERS: &[(&str, &str)] = &[
    ("push/mod.rs", "Reporter: renders the ProgressSink a caller chose, and asks for confirmation only on a terminal sink"),
    ("push/transport/https.rs", "the live upload display, drawn only when the caller's sink is a terminal"),
    ("settings.rs", "the one-time notice that the library wrote the user's settings file; a change to the user's machine is never silent"),
];

/// **Library code never prints, prompts or exits** (DX-8, DX-16).
/// Diagnostics are values; progress goes through a sink the caller
/// passes. A command adapter — a module labelled CLI support — is where
/// output happens.
#[test]
fn library_modules_do_not_print_or_exit() {
    let files = source_files();
    let calls = ["eprintln!", "println!", "eprint!(", "print!(", "process::exit"];
    let mut found = Vec::new();
    for (rel, text) in &files {
        let layer = layer_of(rel, &files);
        if layer == Some("CLI support.") || RENDERERS.iter().any(|(f, _)| f == rel) {
            continue;
        }
        for (n, line) in shipped_code(text).lines().enumerate() {
            if let Some(c) = calls.iter().find(|c| line.contains(**c)) {
                found.push(format!("{rel}:{}: {c} — {}", n + 1, line.trim()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "library code writes to the terminal or exits; return a value or take a sink:\n{}",
        found.join("\n")
    );
    // An allow-listed renderer that no longer renders is a stale entry.
    for (rel, why) in RENDERERS {
        let text = &files.iter().find(|(r, _)| r == rel).unwrap_or_else(|| panic!("{rel} gone")).1;
        assert!(
            calls.iter().any(|c| shipped_code(text).contains(c)),
            "{rel} is allow-listed ({why}) but prints nothing; drop the entry"
        );
    }
}

/// The Common tasks table in the crate docs.
fn common_tasks() -> Vec<Vec<String>> {
    let lib = std::fs::read_to_string(src().join("lib.rs")).unwrap();
    let start = lib.find("//! ## Common tasks").expect("lib.rs has a Common tasks section");
    lib[start..]
        .lines()
        .skip_while(|l| !l.starts_with("//! | Task"))
        .skip(2)
        .take_while(|l| l.starts_with("//! |"))
        .map(|l| {
            l.trim_start_matches("//! ")
                .trim_matches('|')
                .split(" | ")
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect()
}

/// **Every Common tasks row names its call with a link rustdoc checks,
/// and every example it points at exists** (DX-27). A renamed or removed
/// canonical call then breaks the doc build instead of silently leaving
/// the index pointing at nothing.
#[test]
fn every_common_task_links_its_call_and_its_example_exists() {
    let rows = common_tasks();
    assert!(rows.len() >= 10, "the table lost its rows: {rows:?}");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for row in &rows {
        assert_eq!(row.len(), 3, "a row is task | call | example: {row:?}");
        assert!(row[1].contains("[`"), "'{}' names its call without a link", row[0]);
        for ex in row[2].split(", ").filter(|e| e.starts_with('`')) {
            let path = ex.trim_matches('`');
            assert!(root.join(path).is_file(), "'{}' points at {path}, which does not exist", row[0]);
        }
    }
    // The crate docs' snippets compile; none opts out with `ignore`.
    let lib = std::fs::read_to_string(src().join("lib.rs")).unwrap();
    assert!(!lib.contains("```ignore") && !lib.contains("rust,ignore"));
}

/// The public fetch-like entry points, by name: what each is, and
/// whether it is a deprecated duplicate of `fetch` (DX-30). Adding a
/// new one means adding it here — which is the moment to ask whether
/// it is a tenth way to do the same thing.
const FETCH_BUDGET: &[(&str, &str, bool)] = &[
    ("fetch", "the canonical call: view, group, catalog, selection", false),
    ("plan_fetch", "the planning half of fetch", false),
    ("prefetch_in_background", "the background form, with a handle", false),
    ("prefetch_plan", "low-level: one facet's plan", false),
    ("prefetch_plan_on", "low-level: one facet's plan on an open handle", false),
    ("prefetch_range", "page-cache hint on a mapped file; no network", false),
    ("prefetch_pages", "page-cache fill on a mapped file; no network", false),
    ("precache", "low-level: one file or facet, whole", false),
    ("prebuffer_with_progress", "low-level: one file or facet, whole, with transport progress", false),
    ("prebuffer_range_with_progress", "low-level: a byte range of one facet", false),
    ("prebuffer_all", "duplicate of fetch(all)", true),
    ("prebuffer_all_with_progress", "duplicate of fetch(all) with a sink", true),
    ("prefetch", "duplicate of fetch(facet, window)", true),
    ("prefetch_with_progress", "duplicate of fetch(facet, window) with a sink", true),
    ("prebuffer_all_profiles", "duplicate of group fetch", true),
    ("prebuffer_all_profiles_with_progress", "duplicate of group fetch with a sink", true),
    ("prebuffer_profiles_with_progress", "duplicate of group fetch with a sink", true),
];

/// **The fetch-like public surface is exactly the budget, and its
/// duplicates are deprecated** (DX-17, DX-30).
#[test]
fn the_fetch_surface_is_the_budgeted_one() {
    let files = source_files();
    let fetchlike = |name: &str| {
        let n = name.strip_prefix("plan_").unwrap_or(name);
        ["fetch", "prefetch", "precache", "prebuffer", "download"]
            .iter()
            .any(|p| n == *p || n.starts_with(&format!("{p}_")))
    };
    let mut seen: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for (rel, text) in &files {
        if layer_of(rel, &files) != Some("Library API.") {
            continue;
        }
        let code = shipped_code(text);
        let lines: Vec<&str> = code.lines().collect();
        let raw: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            // Public functions, and the provided methods of public
            // traits (indented `fn` with a body or a signature).
            let name = t
                .strip_prefix("pub fn ")
                .or_else(|| t.strip_prefix("fn ").filter(|_| line.starts_with("    fn ")));
            let Some(name) = name.map(|n| n.split(['(', '<']).next().unwrap()) else { continue };
            if !fetchlike(name) {
                continue;
            }
            seen.entry(name.to_string()).or_default().push(format!("{rel}:{}", i + 1));
            // Find the attributes above the item in the raw text.
            let Some(raw_i) = raw.iter().position(|l| l.trim_start() == t) else { continue };
            let attrs: String = raw[..raw_i]
                .iter()
                .rev()
                .take_while(|l| {
                    let l = l.trim_start();
                    l.starts_with("///") || l.starts_with("#[") || l.starts_with("note =")
                        || l.starts_with("since =") || l.starts_with(')') || l.is_empty()
                })
                .copied()
                .collect::<Vec<_>>()
                .join("\n");
            if let Some((_, _, deprecated)) = FETCH_BUDGET.iter().find(|(n, _, _)| *n == name) {
                assert_eq!(
                    attrs.contains("#[deprecated"),
                    *deprecated,
                    "{rel}:{}: `{name}` deprecated-ness does not match the budget",
                    i + 1
                );
            }
        }
    }
    let unbudgeted: Vec<_> = seen
        .iter()
        .filter(|(n, _)| !FETCH_BUDGET.iter().any(|(b, _, _)| b == n))
        .collect();
    assert!(
        unbudgeted.is_empty(),
        "new fetch-like public entry points; budget them in FETCH_BUDGET (or reuse `fetch`): {unbudgeted:?}"
    );
    for (name, why, _) in FETCH_BUDGET {
        assert!(seen.contains_key(*name), "budgeted `{name}` ({why}) no longer exists; drop it");
    }
}
