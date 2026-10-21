// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Attribute overlays: a host's own statement of a
//! dataset's attributes, layered over what its catalog says.
//!
//! A `knn_entries.yaml` catalog carries no `distance_function`, and a
//! host may know a base is normalized when nothing published says so.
//! An overlay says it once, in a file, and every reader of the dataset
//! sees it: `describe`, `list`, `explore` and `view.distance()` read the
//! same merged attributes, each with the [`AttributeSource`] it came
//! from.
//!
//! ```yaml
//! # <config_dir>/overlays.yaml — comments are kept on every edit
//! datasets:
//!   ada002-100k:                  # any dataset of this name
//!     distance_function: DOT_PRODUCT
//!     is_normalized: true
//!     fvec.ncentroids: 300        # a consumer's own key: namespaced, opaque
//! catalogs:
//!   bench:                        # only the entry in catalog `bench`
//!     ada002-100k:
//!       base_count: 99287
//! ```
//!
//! Lowest to highest: the dataset's own attributes, those inferred from
//! loose files' names, program overlays (registered with
//! [`CatalogSources::with_overlay`](crate::catalog::CatalogSources::with_overlay),
//! in order; two setting one key is refused), and the user's
//! `<config_dir>/overlays.yaml`. Within a file, a `catalogs.<c>.<name>`
//! block wins over `datasets.<name>` for the entry in catalog `c`. A
//! value replaces; it never merges into the one below.
//!
//! An overlay is believed, and checked: whenever a facet it makes a
//! claim about is opened, a disagreement — a `base_count` that is not
//! the base's, a `maxk` that is not the ground truth's width — is a
//! warning naming the file, the key and both values, once per process.
//! [`TestDataViewExt::verify`](crate::TestDataViewExt::verify) reports
//! it as a failed check.
//!
//! (`docs/design/srd-loose-files-overlays-integrity.md` §4.)

use std::fmt;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde_yaml::Value;

/// The file name of the user overlay, in the configuration directory.
pub const USER_OVERLAY_FILE: &str = "overlays.yaml";

/// Who stated an overlay.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OverlayKind {
    /// The user, in `<config_dir>/overlays.yaml`.
    User,
    /// A program, by the name it registered the overlay under.
    Program(String),
}

/// Where an attribute's value came from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttributeSource {
    /// The dataset's own descriptor or catalog listing.
    Dataset,
    /// Its loose files' names (`dataset::infer`, LF-12).
    Inferred,
    /// An overlay.
    Overlay {
        /// The overlay's file, or the program's name for one built in
        /// code.
        file: String,
        /// Who stated it.
        kind: OverlayKind,
    },
    /// Measured from the data, and recorded in this file
    /// (`<cache_root>/<dataset>/measured.yaml`), or not recorded (a
    /// local base, measured each time).
    Measured {
        /// The file the measurement is recorded in, when it is.
        file: Option<String>,
    },
}

impl fmt::Display for AttributeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttributeSource::Dataset => write!(f, "dataset"),
            AttributeSource::Inferred => write!(f, "inferred from file names"),
            AttributeSource::Overlay { file, kind: OverlayKind::User } => write!(f, "overlay {file}"),
            AttributeSource::Overlay { file, kind: OverlayKind::Program(name) } => {
                write!(f, "program overlay '{name}' ({file})")
            }
            AttributeSource::Measured { file: Some(file) } => write!(f, "measured, recorded in {file}"),
            AttributeSource::Measured { file: None } => write!(f, "measured"),
        }
    }
}

/// The attribute keys an overlay may set without a namespace: the
/// fields a dataset's attributes hold, and the counts a dataset
/// states. Any key holding a `.` is a consumer's own and is opaque.
pub const KNOWN_KEYS: &[&str] = &[
    "model",
    "model_revision",
    "model_revision_requested",
    "url",
    "distance_function",
    "license",
    "vendor",
    "notes",
    "is_normalized",
    "is_duplicate_vector_free",
    "is_zero_vector_free",
    "tags",
    "base_count",
    "dimension",
    "maxk",
];

/// Whether `key` is one an overlay may set: a known attribute, or a
/// namespaced (`consumer.key`) one.
pub fn is_settable(key: &str) -> bool {
    KNOWN_KEYS.contains(&key) || key.contains('.')
}

/// Keys and values for one dataset.
type Values = IndexMap<String, Value>;

/// One overlay: a user's file, or a program's statement.
#[derive(Debug, Clone)]
pub struct Overlay {
    kind: OverlayKind,
    file: String,
    datasets: IndexMap<String, Values>,
    catalogs: IndexMap<String, IndexMap<String, Values>>,
    warnings: Vec<String>,
}

impl Overlay {
    /// Read an overlay's YAML text, as stated by `kind` and found in
    /// `file` (a path, or a name for one built in code). An unknown key
    /// without a namespace is ignored with a warning, named with its
    /// file and dataset.
    pub fn parse(text: &str, kind: OverlayKind, file: impl Into<String>) -> Result<Self, String> {
        let file = file.into();
        let doc: Value = serde_yaml::from_str(text).map_err(|e| format!("{file}: {e}"))?;
        let mut overlay = Overlay { kind, file, datasets: IndexMap::new(), catalogs: IndexMap::new(), warnings: Vec::new() };
        let Some(map) = doc.as_mapping() else {
            if doc.is_null() {
                return Ok(overlay);
            }
            return Err(format!("{}: an overlay is a map with `datasets:` and `catalogs:`", overlay.file));
        };
        for (k, v) in map {
            match k.as_str() {
                Some("datasets") => {
                    for (name, values) in v.as_mapping().into_iter().flatten() {
                        let name = name.as_str().ok_or_else(|| format!("{}: a dataset name is not a string", overlay.file))?;
                        let values = overlay.values(name, values)?;
                        overlay.datasets.insert(name.to_string(), values);
                    }
                }
                Some("catalogs") => {
                    for (catalog, datasets) in v.as_mapping().into_iter().flatten() {
                        let catalog =
                            catalog.as_str().ok_or_else(|| format!("{}: a catalog name is not a string", overlay.file))?;
                        let mut by_name = IndexMap::new();
                        for (name, values) in datasets.as_mapping().into_iter().flatten() {
                            let name =
                                name.as_str().ok_or_else(|| format!("{}: a dataset name is not a string", overlay.file))?;
                            by_name.insert(name.to_string(), overlay.values(name, values)?);
                        }
                        overlay.catalogs.insert(catalog.to_string(), by_name);
                    }
                }
                other => overlay.warnings.push(format!(
                    "{}: '{}' is not a section of an overlay (datasets, catalogs); ignored",
                    overlay.file,
                    other.unwrap_or("?")
                )),
            }
        }
        for w in &overlay.warnings {
            log::warn!("{w}");
        }
        Ok(overlay)
    }

    /// The settable keys of one dataset's block.
    fn values(&mut self, dataset: &str, block: &Value) -> Result<Values, String> {
        let mut out = Values::new();
        for (k, v) in block.as_mapping().into_iter().flatten() {
            let key = k.as_str().ok_or_else(|| format!("{}: a key of '{dataset}' is not a string", self.file))?;
            if is_settable(key) {
                out.insert(key.to_string(), v.clone());
            } else {
                self.warnings.push(format!(
                    "{}: '{key}' for '{dataset}' is not a dataset attribute; ignored (a consumer's own key holds a '.', as `fvec.ncentroids`)",
                    self.file
                ));
            }
        }
        Ok(out)
    }

    /// A program's overlay, from YAML it holds.
    pub fn program(name: &str, text: &str) -> Result<Self, String> {
        Self::parse(text, OverlayKind::Program(name.to_string()), format!("<{name}>"))
    }

    /// The user overlay of the configuration directory `config_dir`,
    /// when it has one.
    pub fn load_user(config_dir: &Path) -> Option<Result<Self, String>> {
        let path = user_overlay_path(config_dir);
        let text = std::fs::read_to_string(&path).ok()?;
        Some(Self::parse(&text, OverlayKind::User, path.display().to_string()))
    }

    /// What reading it reported: unknown keys and sections, ignored.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Who stated it.
    pub fn kind(&self) -> &OverlayKind {
        &self.kind
    }

    /// Its file, or its program's name.
    pub fn file(&self) -> &str {
        &self.file
    }

    /// The values it states for `dataset` in catalog `catalog`: the
    /// `datasets` block, then the catalog's block over it.
    pub fn values_for(&self, catalog: Option<&str>, dataset: &str) -> Values {
        let mut out = Values::new();
        if let Some(v) = find(&self.datasets, dataset) {
            out.extend(v.clone());
        }
        if let Some(c) = catalog.and_then(|c| self.catalogs.get(c))
            && let Some(v) = find(c, dataset)
        {
            out.extend(v.clone());
        }
        out
    }

    fn source(&self) -> AttributeSource {
        AttributeSource::Overlay { file: self.file.clone(), kind: self.kind.clone() }
    }

    /// Every (catalog, dataset, key) this overlay sets, for conflicts.
    fn keys(&self) -> Vec<(Option<String>, String, String)> {
        let mut out = Vec::new();
        for (name, values) in &self.datasets {
            out.extend(values.keys().map(|k| (None, name.to_lowercase(), k.clone())));
        }
        for (catalog, by_name) in &self.catalogs {
            for (name, values) in by_name {
                out.extend(values.keys().map(|k| (Some(catalog.clone()), name.to_lowercase(), k.clone())));
            }
        }
        out
    }
}

/// A dataset's block by name, case-insensitively as catalogs match
/// names.
fn find<'a>(map: &'a IndexMap<String, Values>, dataset: &str) -> Option<&'a Values> {
    map.iter().find(|(k, _)| k.eq_ignore_ascii_case(dataset)).map(|(_, v)| v)
}

/// `<config_dir>/overlays.yaml`.
pub fn user_overlay_path(config_dir: &Path) -> PathBuf {
    config_dir.join(USER_OVERLAY_FILE)
}

/// Every overlay in force, in precedence order.
#[derive(Debug, Clone, Default)]
pub struct Overlays {
    program: Vec<Overlay>,
    user: Option<Overlay>,
}

/// One attribute an overlay sets, with its source.
#[derive(Debug, Clone, PartialEq)]
pub struct Stated {
    /// The key.
    pub key: String,
    /// The value.
    pub value: Value,
    /// Who stated it.
    pub source: AttributeSource,
}

impl Overlays {
    /// No overlays.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether there is none.
    pub fn is_empty(&self) -> bool {
        self.program.is_empty() && self.user.is_none()
    }

    /// Add a program's overlay, after the others. A key another program
    /// overlay sets for the same dataset is refused, naming both: two
    /// programs' statements are not settled by order.
    pub fn add_program(&mut self, overlay: Overlay) -> Result<(), String> {
        let theirs = overlay.keys();
        for other in &self.program {
            let ours = other.keys();
            if let Some((catalog, name, key)) = theirs.iter().find(|k| ours.contains(k)) {
                let where_ = catalog.as_deref().map(|c| format!(" in catalog '{c}'")).unwrap_or_default();
                return Err(format!(
                    "the overlays {} and {} both set '{key}' for '{name}'{where_}; state it in one",
                    other.file, overlay.file
                ));
            }
        }
        self.program.push(overlay);
        Ok(())
    }

    /// Set the user's overlay, over every program's.
    pub fn set_user(&mut self, overlay: Option<Overlay>) {
        self.user = overlay;
    }

    /// The overlays, lowest first.
    pub fn iter(&self) -> impl Iterator<Item = &Overlay> {
        self.program.iter().chain(self.user.iter())
    }

    /// What the overlays state for `dataset` in `catalog`, each key
    /// once, at the value of the highest overlay that sets it.
    pub fn resolve(&self, catalog: Option<&str>, dataset: &str) -> Vec<Stated> {
        let mut out: IndexMap<String, Stated> = IndexMap::new();
        for overlay in self.iter() {
            for (key, value) in overlay.values_for(catalog, dataset) {
                out.insert(key.clone(), Stated { key, value, source: overlay.source() });
            }
        }
        out.into_values().collect()
    }
}

/// The counts an overlay claims for a dataset, for checking against the
/// facets as they open (OV-5).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Claims {
    /// The dataset's name, for the warning.
    pub dataset: String,
    /// `base_count`, and who said it.
    pub base_count: Option<(u64, AttributeSource)>,
    /// `dimension`, and who said it.
    pub dimension: Option<(u64, AttributeSource)>,
    /// `maxk`, and who said it.
    pub maxk: Option<(u64, AttributeSource)>,
}

impl Claims {
    /// The count claims among `stated`.
    pub fn of(dataset: &str, stated: &[Stated]) -> Option<Self> {
        let get = |key: &str| {
            stated
                .iter()
                .find(|s| s.key == key)
                .and_then(|s| s.value.as_u64().map(|n| (n, s.source.clone())))
        };
        let claims = Claims { dataset: dataset.to_string(), base_count: get("base_count"), dimension: get("dimension"), maxk: get("maxk") };
        (claims.base_count.is_some() || claims.dimension.is_some() || claims.maxk.is_some()).then_some(claims)
    }

    /// What disagrees with an opened facet: `facet` holds `count`
    /// records of `dim`. Each as (key, claimed, observed, source).
    pub fn contradictions(&self, facet: &str, count: usize, dim: usize, whole_base: bool) -> Vec<(&'static str, u64, u64, AttributeSource)> {
        let mut out = Vec::new();
        let mut check = |key: &'static str, claim: &Option<(u64, AttributeSource)>, observed: u64| {
            if let Some((claimed, source)) = claim
                && *claimed != observed
            {
                out.push((key, *claimed, observed, source.clone()));
            }
        };
        match facet {
            "base_vectors" => {
                if whole_base {
                    check("base_count", &self.base_count, count as u64);
                }
                if count > 0 {
                    check("dimension", &self.dimension, dim as u64);
                }
            }
            "neighbor_indices" if count > 0 => check("maxk", &self.maxk, dim as u64),
            _ => {}
        }
        out
    }
}

/// Warn, once per process for each (dataset, key), that an overlay's
/// value disagrees with what was observed (OV-5). The value stays in
/// force; the warning says so.
pub(crate) fn warn_contradiction(dataset: &str, key: &str, claimed: &str, observed: &str, source: &AttributeSource) {
    static SAID: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    let id = (dataset.to_string(), key.to_string());
    if said.contains(&id) {
        return;
    }
    said.push(id);
    log::warn!(
        "{source} states {key}: {claimed} for '{dataset}', but {observed} was observed; the overlay's value is in force"
    );
}

// ── Editing the user overlay, line by line ─────────────────────────────

/// The YAML text of an overlay value.
fn render(value: &Value) -> Result<String, String> {
    crate::dataset::yaml_edit::render_scalar(value).map_err(|e| e.replace("a tag value", "an overlay value"))
}

/// Indentation of a line, in spaces.
fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The `key:` a line opens, when it opens one at exactly `at` spaces.
fn key_at(line: &str, at: usize) -> Option<&str> {
    if indent(line) != at || line.trim_start().starts_with('#') {
        return None;
    }
    let t = line.trim_start();
    let (k, _) = t.split_once(':')?;
    Some(k.trim().trim_matches(['\'', '"']))
}

/// The line range of the block a `key:` line at `at` opens: the line
/// itself through the last line indented deeper (comments and blanks
/// inside it included), within `range`.
fn block(lines: &[String], range: std::ops::Range<usize>, at: usize, key: &str) -> Option<std::ops::Range<usize>> {
    let start = range.clone().find(|&i| key_at(&lines[i], at).is_some_and(|k| k.eq_ignore_ascii_case(key)))?;
    let mut end = start + 1;
    let mut last = start + 1;
    while end < range.end {
        let l = &lines[end];
        if l.trim().is_empty() || l.trim_start().starts_with('#') {
            end += 1;
            continue;
        }
        if indent(l) <= at {
            break;
        }
        end += 1;
        last = end;
    }
    Some(start..last)
}

/// The text of `path` with `key: value` set for `dataset` (in
/// `catalog`, when given): the key's line replaced, or a line added at
/// the end of the dataset's block, which is added — with its section —
/// when there is none. Every other line, comments included, is kept.
pub fn set_text(text: &str, catalog: Option<&str>, dataset: &str, key: &str, value: &Value) -> Result<String, String> {
    if !is_settable(key) {
        return Err(format!("'{key}' is not a dataset attribute; a consumer's own key holds a '.', as `fvec.ncentroids`"));
    }
    let rendered = render(value)?;
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let (section, path): (&str, Vec<&str>) = match catalog {
        Some(c) => ("catalogs", vec![c, dataset]),
        None => ("datasets", vec![dataset]),
    };
    // The section at the top level.
    let mut range = match block(&lines, 0..lines.len(), 0, section) {
        Some(r) => r,
        None => {
            if lines.last().is_some_and(|l| !l.trim().is_empty()) {
                lines.push(String::new());
            }
            lines.push(format!("{section}:"));
            lines.len() - 1..lines.len()
        }
    };
    // Each level beneath it: a catalog, then the dataset.
    let mut at = 0;
    for name in path {
        at += 2;
        range = match block(&lines, range.start + 1..range.end, at, name) {
            Some(r) => r,
            None => {
                lines.insert(range.end, format!("{}{name}:", " ".repeat(at)));
                range.end..range.end + 1
            }
        };
    }
    at += 2;
    let line = format!("{}{key}: {rendered}", " ".repeat(at));
    match (range.start + 1..range.end).find(|&i| key_at(&lines[i], at) == Some(key)) {
        Some(i) => lines[i] = line,
        None => lines.insert(range.end, line),
    }
    let mut out = lines.join("\n");
    out.push('\n');
    Ok(out)
}

/// The text of `path` without `key` for `dataset` (in `catalog`, when
/// given), and whether it was there. A dataset block left empty is
/// removed; everything else is kept.
pub fn unset_text(text: &str, catalog: Option<&str>, dataset: &str, key: &str) -> (String, bool) {
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let (section, path): (&str, Vec<&str>) = match catalog {
        Some(c) => ("catalogs", vec![c, dataset]),
        None => ("datasets", vec![dataset]),
    };
    let Some(mut range) = block(&lines, 0..lines.len(), 0, section) else { return (text.to_string(), false) };
    let mut at = 0;
    for name in &path {
        at += 2;
        match block(&lines, range.start + 1..range.end, at, name) {
            Some(r) => range = r,
            None => return (text.to_string(), false),
        }
    }
    let Some(i) = (range.start + 1..range.end).find(|&i| key_at(&lines[i], at + 2) == Some(key)) else {
        return (text.to_string(), false);
    };
    lines.remove(i);
    // The dataset's block, now one line shorter: when nothing but
    // comments and blanks is left in it, it goes, its comments with it.
    let end = range.end - 1;
    if (range.start + 1..end).all(|j| lines[j].trim().is_empty() || lines[j].trim_start().starts_with('#')) {
        lines.drain(range.start..end);
    }
    let mut out = lines.join("\n");
    out.push('\n');
    (out, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_adds_the_section_block_and_key_and_keeps_comments() {
        let text = "# my overlays\n";
        let text = set_text(text, None, "ada", "distance_function", &Value::from("DOT_PRODUCT")).unwrap();
        assert_eq!(text, "# my overlays\n\ndatasets:\n  ada:\n    distance_function: DOT_PRODUCT\n");
        let text = set_text(&text, None, "ada", "is_normalized", &Value::from(true)).unwrap();
        let text = set_text(&text, None, "ada", "distance_function", &Value::from("COSINE")).unwrap();
        assert_eq!(text, "# my overlays\n\ndatasets:\n  ada:\n    distance_function: COSINE\n    is_normalized: true\n");
        let text = set_text(&text, Some("bench"), "ada", "base_count", &Value::from(99287)).unwrap();
        assert!(text.ends_with("catalogs:\n  bench:\n    ada:\n      base_count: 99287\n"), "{text}");
        let parsed = Overlay::parse(&text, OverlayKind::User, "o").unwrap();
        assert_eq!(parsed.values_for(Some("bench"), "ADA").len(), 3);
    }

    #[test]
    fn unset_removes_the_key_and_an_emptied_block() {
        let text = "datasets:\n  ada:\n    # why\n    distance_function: COSINE\n  other:\n    model: m\n";
        let (text, removed) = unset_text(text, None, "ada", "distance_function");
        assert!(removed);
        assert_eq!(text, "datasets:\n  other:\n    model: m\n", "{text}");
        let (_, removed) = unset_text(&text, None, "ada", "distance_function");
        assert!(!removed);
    }

    #[test]
    fn a_key_two_program_overlays_set_is_refused() {
        let mut all = Overlays::new();
        all.add_program(Overlay::program("a", "datasets:\n  ds:\n    distance_function: L2\n").unwrap()).unwrap();
        all.add_program(Overlay::program("b", "datasets:\n  ds:\n    model: m\n").unwrap()).unwrap();
        let err = all.add_program(Overlay::program("c", "datasets:\n  DS:\n    distance_function: COSINE\n").unwrap()).unwrap_err();
        assert!(err.contains("<a> and <c> both set 'distance_function' for 'ds'"), "{err}");
    }

    #[test]
    fn the_user_overlay_wins_and_a_catalog_block_beats_a_bare_one() {
        let mut all = Overlays::new();
        all.add_program(Overlay::program("p", "datasets:\n  ds:\n    distance_function: L2\n    maxk: 10\n").unwrap()).unwrap();
        all.set_user(Some(
            Overlay::parse("datasets:\n  ds:\n    distance_function: COSINE\ncatalogs:\n  c:\n    ds:\n      maxk: 100\n", OverlayKind::User, "u")
                .unwrap(),
        ));
        let stated = all.resolve(Some("c"), "ds");
        let get = |k: &str| stated.iter().find(|s| s.key == k).unwrap().clone();
        assert_eq!(get("distance_function").value, Value::from("COSINE"));
        assert_eq!(get("maxk").value, Value::from(100));
        assert!(matches!(get("maxk").source, AttributeSource::Overlay { kind: OverlayKind::User, .. }));
        let elsewhere = all.resolve(Some("other"), "ds");
        assert_eq!(elsewhere.iter().find(|s| s.key == "maxk").unwrap().value, Value::from(10));
    }

    #[test]
    fn an_unknown_bare_key_is_warned_and_ignored() {
        let o = Overlay::parse("datasets:\n  ds:\n    colour: blue\n    fvec.groups: [a, b]\n", OverlayKind::User, "u").unwrap();
        assert_eq!(o.values_for(None, "ds").keys().collect::<Vec<_>>(), ["fvec.groups"]);
        assert!(o.warnings()[0].contains("'colour' for 'ds' is not a dataset attribute"), "{:?}", o.warnings());
    }
}
