// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The API core stays the API core (`docs/design/srd-crate-boundaries.md`
//! §4, §7): it depends only on its vocabulary (AC-1), its hot-path
//! functions inline across crates (PR-2), and a trait one crate
//! implements and another consumes is defined here, not in an
//! implementation crate (AC-2).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// The keys of a manifest's `[dependencies]` table.
fn dependencies(manifest: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut inside = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            inside = t == "[dependencies]";
            continue;
        }
        if inside
            && !t.starts_with('#')
            && let Some((key, _)) = t.split_once('=')
        {
            out.insert(key.trim().to_string());
        }
    }
    out
}

/// What the API's vocabulary needs: the element model, and the
/// descriptor model's parsing, regex selectors and parse warnings.
const VOCABULARY: &[&str] =
    &["indexmap", "log", "regex", "serde", "serde_json", "serde_yaml", "thiserror", "url", "veks-datatypes"];

/// **AC-1: the API core holds no implementation**, so it depends on
/// nothing that implements one: not a network client, not a mapping,
/// not an implementation crate of the workspace.
#[test]
fn the_api_core_depends_only_on_its_vocabulary() {
    let manifest = std::fs::read_to_string(crates_dir().join("veks-apicore/Cargo.toml")).unwrap();
    let deps = dependencies(&manifest);
    let allowed: BTreeSet<String> = VOCABULARY.iter().map(|s| s.to_string()).collect();
    let extra: Vec<&String> = deps.difference(&allowed).collect();
    assert!(
        extra.is_empty(),
        "veks-apicore depends on {extra:?}: the API core holds no implementation, so a dependency belongs \
         there only when the API's vocabulary needs it (srd-crate-boundaries.md AC-1)"
    );
}

/// The non-generic `pub fn`s of a source, by name, each with whether
/// `#[inline]` sits on it.
fn public_functions(text: &str) -> Vec<(String, bool)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("pub fn ") else { continue };
        let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if rest[name.len()..].starts_with('<') {
            continue; // generic: instantiated in the caller's crate
        }
        let inline = lines[..i]
            .iter()
            .rev()
            .take_while(|l| {
                let l = l.trim_start();
                l.starts_with("///") || l.starts_with("#[")
            })
            .any(|l| l.trim_start().starts_with("#[inline"));
        out.push((name, inline));
    }
    out
}

/// The modules a per-record, per-chunk or per-page path calls into.
const HOT_MODULES: &[&str] = &["reader.rs", "source.rs", "stream.rs", "progress.rs"];

/// Non-generic functions of the hot modules that run once per open,
/// not per record: a vtable call or an out-of-line call there costs
/// nothing a caller can measure.
const COLD_IN_HOT_MODULES: &[(&str, &str)] = &[("new", "ConvertingReader::new: once per open")];

/// Every other module, and why nothing in it is on a hot path.
const COLD_MODULES: &[(&str, &str)] = &[
    ("lib.rs", "re-exports"),
    ("bytes.rs", "FacetBytes: opening, planning, whole ranges, whole-file hashing"),
    ("catalog.rs", "catalog loading"),
    ("error.rs", "built only on the error path (PR-6)"),
    ("fetch.rs", "fetch planning and per-range execution"),
    ("location.rs", "parsing a location at open"),
    ("metadata_schema.rs", "a slab's schema sidecar, read or written once"),
    ("model.rs", "descriptor model, at load"),
    ("overlay.rs", "attribute provenance, at load"),
    ("records.rs", "record facets: a page per record, through dyn containers by design"),
    ("transport.rs", "a transport's configuration, built once; a request value, built once per request"),
    ("verify.rs", "verification reports"),
    ("view.rs", "dataset views, at open; reads go to the readers they hand out"),
    ("dataset", "descriptor model, at load"),
];

/// **PR-2: what a hot path calls across the crate boundary inlines.**
/// The release profile has no LTO, so a non-generic function here is
/// inlined into veks-io's or vectordata's readers only when it says so;
/// every module is classed hot or cold, so a new one must be placed.
#[test]
fn hot_path_functions_are_inlinable_across_crates() {
    let src = crates_dir().join("veks-apicore/src");
    let mut unclassified = Vec::new();
    let mut not_inline = Vec::new();
    for entry in std::fs::read_dir(&src).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if HOT_MODULES.contains(&name.as_str()) {
            let text = std::fs::read_to_string(entry.path()).unwrap();
            for (f, inline) in public_functions(&text) {
                if !inline && !COLD_IN_HOT_MODULES.iter().any(|(c, _)| *c == f) {
                    not_inline.push(format!("{name}: {f}"));
                }
            }
        } else if !COLD_MODULES.iter().any(|(m, _)| *m == name) {
            unclassified.push(name);
        }
    }
    assert!(unclassified.is_empty(), "class these modules hot or cold (PR-2): {unclassified:?}");
    assert!(not_inline.is_empty(), "hot-path functions without #[inline] (PR-2): {not_inline:?}");
}

/// Every `.rs` file under `dir`, with `#[cfg(test)]` modules cut.
fn rust_sources(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(rust_sources(&p));
        } else if p.extension().is_some_and(|x| x == "rs") {
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            out.push((p, without_test_modules(&text)));
        }
    }
    out
}

/// `text` with every `#[cfg(test)] mod … { … }` removed.
fn without_test_modules(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find("#[cfg(test)]") {
        out.push_str(&rest[..i]);
        let after = &rest[i..];
        let Some(open) = after.find('{') else { break };
        if !after[..open].contains("mod ") {
            out.push_str("#[cfg(test)]");
            rest = &rest[i + "#[cfg(test)]".len()..];
            continue;
        }
        let mut depth = 0usize;
        let mut end = after.len();
        for (k, c) in after[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + k + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

/// The public traits an implementation crate's sources define.
fn public_traits(sources: &[(PathBuf, String)]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (_, text) in sources {
        for line in text.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("pub trait ") {
                out.insert(rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect());
            }
        }
    }
    out
}

/// Whether `text` names `word` as a whole identifier.
fn names(text: &str, word: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    text.match_indices(word).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + word.len()..].chars().next();
        !before.is_some_and(ident) && !after.is_some_and(ident)
    })
}

/// The crates that implement the API.
const IMPLEMENTATIONS: &[&str] = &["vectordata", "veks-io", "veks-transport"];

/// Public traits of an implementation crate that another crate uses,
/// and why each is not a contract of the API.
const NOT_CONTRACTS: &[(&str, &str)] = &[
    ("DatasetConfigIo", "an extension trait: vectordata's methods on the API's DatasetConfig, one implementation"),
    ("ElementTypeExt", "an extension trait: the path and URL forms of ElementType::from_extension, one implementation"),
    ("OpenSource", "veks-io's constructor protocol for its own generic open functions (PR-4); opening is not a contract"),
];

/// The violations of AC-2 under the workspace's `crates/` at `root`:
/// a public trait an implementation crate defines that another crate's
/// sources name.
fn misplaced_contracts(root: &Path) -> Vec<String> {
    let crates: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .flatten()
        .filter(|e| e.path().join("Cargo.toml").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let mut found = Vec::new();
    for imp in IMPLEMENTATIONS {
        let traits = public_traits(&rust_sources(&root.join(imp).join("src")));
        for other in crates.iter().filter(|c| c.as_str() != *imp) {
            let sources = rust_sources(&root.join(other).join("src"));
            for t in &traits {
                if NOT_CONTRACTS.iter().any(|(n, _)| n == t) {
                    continue;
                }
                if let Some((p, _)) = sources.iter().find(|(_, text)| names(text, t)) {
                    found.push(format!("{imp}::{t} is used by {other} ({})", p.display()));
                }
            }
        }
    }
    found
}

/// **AC-2: a trait one crate implements and another consumes is the
/// API core's.** An implementation crate's public trait used by another
/// crate is a contract in the wrong place, unless it is listed in
/// [`NOT_CONTRACTS`] with the reason it is not one.
#[test]
fn contracts_are_the_api_cores() {
    let found = misplaced_contracts(&crates_dir());
    assert!(found.is_empty(), "contracts defined outside veks-apicore (srd-crate-boundaries.md AC-2):\n  {}", found.join("\n  "));
}

/// The streaming model's names and where each is defined.
const STREAMING_MODEL: &[(&str, &str, &str)] = &[
    ("trait", "VecSource", "veks-apicore"),
    ("trait", "VecSink", "veks-apicore"),
    ("struct", "SourceMeta", "veks-apicore"),
    ("enum", "VecFormat", "veks-io"),
];

/// Every definition of a streaming-model name, as `crate: file`.
fn streaming_definitions(root: &Path) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(root).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        for (path, text) in rust_sources(&entry.path().join("src")) {
            for (kind, item, _) in STREAMING_MODEL {
                if text.lines().any(|l| l.trim_start().starts_with(&format!("pub {kind} {item}"))) {
                    found.push((format!("{item} in {name}"), path.display().to_string()));
                }
            }
        }
    }
    found
}

/// **FL-1: one streaming model.** `VecSource`, `VecSink` and
/// `SourceMeta` are the API core's and `VecFormat` is veks-io's; no
/// other crate defines its own, so nothing converts between two models
/// through a string.
#[test]
fn the_streaming_model_is_defined_once() {
    let found = streaming_definitions(&crates_dir());
    let misplaced: Vec<&(String, String)> = found
        .iter()
        .filter(|(what, _)| !STREAMING_MODEL.iter().any(|(_, item, home)| *what == format!("{item} in {home}")))
        .collect();
    assert!(misplaced.is_empty(), "streaming-model types defined outside their home (FL-1): {misplaced:?}");
}
