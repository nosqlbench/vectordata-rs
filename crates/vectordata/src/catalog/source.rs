// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Where a catalog's entries come from: the
//! [`CatalogSource`] trait, a catalog at a location ([`LocatedCatalog`]),
//! and a catalog a program builds ([`CatalogBuilder`] →
//! [`BuiltCatalog`]).
//!
//! A [`CatalogSources`](super::CatalogSources) set holds any mix of them,
//! in order; [`Catalog::of`](super::Catalog::of) asks each for its
//! entries and records which catalog every entry came from. A located
//! and a built catalog given the same content give the same entries:
//! both read it with one parser.
//!
//! ```
//! use vectordata::catalog::{Catalog, CatalogBuilder, CatalogSources};
//! use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
//! use vectordata::Location;
//!
//! let description = DatasetConfig::builder("demo")
//!     .profile("default", |p| p.facet(StandardFacet::BaseVectors, "base.fvec"))
//!     .build()
//!     .map_err(vectordata::Error::Other)?;
//! let entry = CatalogEntry::builder("demo")
//!     .home(&Location::parse("https://example.com/datasets/demo/")?)
//!     .dataset(description)
//!     .build()
//!     .map_err(vectordata::Error::Other)?;
//! let sources = CatalogSources::new().with(CatalogBuilder::named("mine").entry(entry).build());
//! let catalog = Catalog::of(&sources);
//! assert_eq!(catalog.lookup("demo")?.catalog_name(), Some("mine"));
//! # Ok::<(), vectordata::Error>(())
//! ```

use std::fmt;

use super::CatalogDiagnostic;
use crate::dataset::CatalogEntry;

/// A catalog: a named supply of [`CatalogEntry`] values.
///
/// The crate provides [`LocatedCatalog`] (a file, directory or URL) and
/// [`BuiltCatalog`] (from [`CatalogBuilder`]). A program may implement it
/// to serve entries from anywhere else.
pub trait CatalogSource: Send + Sync {
    /// The catalog's symbolic name: what provenance, ambiguity reports
    /// and the catalogs screen show.
    fn name(&self) -> &str;

    /// Where it came from, for display and diagnostics.
    fn origin(&self) -> CatalogOrigin;

    /// Its entries. `None` when there is no catalog there at all, such
    /// as a location with no catalog file; the set it belongs to decides
    /// whether that is an error. Problems with what is there (a file
    /// that does not parse, an entry skipped) go into `diagnostics`,
    /// never to stdout.
    fn entries(&self, diagnostics: &mut Vec<CatalogDiagnostic>) -> Option<Vec<CatalogEntry>>;
}

/// Where a [`CatalogSource`] came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogOrigin {
    /// A file, directory or URL, as written: a catalog location is a
    /// cache identity (its kept copy is named by it), so it is never
    /// re-spelled.
    Located {
        /// The location.
        location: String,
        /// The `catalogs.yaml` that names it, when one does.
        configured_in: Option<String>,
    },
    /// Built in memory by a program; `label` is what diagnostics and
    /// the catalogs screen show.
    Registered {
        /// The label.
        label: String,
    },
}

impl fmt::Display for CatalogOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatalogOrigin::Located { location, .. } => f.write_str(location),
            CatalogOrigin::Registered { label } => f.write_str(label),
        }
    }
}

/// Whether a catalog in a [`CatalogSources`](super::CatalogSources) set
/// must load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    /// Finding no catalog there is an error.
    Required,
    /// Finding no catalog there is unremarkable.
    Optional,
}

/// The catalog at a file, directory or URL.
///
/// A location naming a file (`.json`, `.yaml`, `.yml`, or an existing
/// local file) is that file. A directory or URL prefix is probed for
/// `catalog.json`, `catalog.yaml` and `knn_entries.yaml`, in that order;
/// the first found is the catalog. Either way the content's shape
/// decides how it reads: a list is a canonical catalog, a map is a
/// `knn_entries.yaml` document (jvector `DataSetLoaderSimpleMFD`
/// parity). Every entry records the document it was read from.
#[derive(Debug, Clone)]
pub struct LocatedCatalog {
    name: String,
    location: String,
    configured_in: Option<String>,
}

/// The file names a catalog directory is probed for, in order.
const CATALOG_FILES: [&str; 3] = ["catalog.json", "catalog.yaml", "knn_entries.yaml"];

impl LocatedCatalog {
    /// The catalog at `location`, called `name`.
    pub fn new(name: impl Into<String>, location: &crate::Location) -> Self {
        Self::as_written(name, location.to_string(), None)
    }

    /// The catalog at `location` as written, named in the
    /// `catalogs.yaml` at `configured_in` when one names it.
    pub(crate) fn as_written(
        name: impl Into<String>,
        location: impl Into<String>,
        configured_in: Option<String>,
    ) -> Self {
        LocatedCatalog { name: name.into(), location: location.into(), configured_in }
    }

    /// The location, as written.
    pub fn location(&self) -> &str {
        &self.location
    }

    /// The files this catalog may be: the location itself when it names
    /// a file, else the probe set under it.
    fn candidates(&self) -> Vec<String> {
        let names_file = super::sources::looks_like_catalog_file(&self.location)
            || (!crate::transport::is_remote_url(&self.location)
                && std::path::Path::new(&self.location).is_file());
        if names_file {
            vec![self.location.clone()]
        } else {
            let dir = crate::location::dir_form(&self.location);
            CATALOG_FILES.iter().map(|file| format!("{dir}{file}")).collect()
        }
    }
}

impl CatalogSource for LocatedCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    fn origin(&self) -> CatalogOrigin {
        CatalogOrigin::Located {
            location: self.location.clone(),
            configured_in: self.configured_in.clone(),
        }
    }

    fn entries(&self, diagnostics: &mut Vec<CatalogDiagnostic>) -> Option<Vec<CatalogEntry>> {
        self.candidates().into_iter().find_map(|file| {
            let content = read_catalog_file(&file)?;
            let base = crate::location::parent_form(&file);
            Some(parse_catalog_content(&content, Some(&base), Some(&file), diagnostics))
        })
    }
}

/// A catalog file's text, or `None` when it cannot be had: a local file
/// that is not there, or a remote one its server does not give (the
/// next probe is tried).
fn read_catalog_file(file: &str) -> Option<String> {
    if crate::transport::is_remote_url(file) {
        super::resolver::fetch_http(file).ok()
    } else {
        let path = std::path::Path::new(file);
        path.is_file().then(|| std::fs::read_to_string(path).ok()).flatten()
    }
}

/// The entries a catalog document holds, read by its shape: a list is a
/// canonical catalog (`catalog.json`/`catalog.yaml`), a map is a
/// `knn_entries.yaml` document. Relative entry paths and facets resolve
/// against `base`. Without a base, a relative one is a diagnostic and
/// its entry is skipped, rather than resolved against wherever the
/// process happens to run. `document` is recorded on every entry as the
/// file it was read from.
pub(crate) fn parse_catalog_content(
    content: &str,
    base: Option<&str>,
    document: Option<&str>,
    diagnostics: &mut Vec<CatalogDiagnostic>,
) -> Vec<CatalogEntry> {
    let named = document.unwrap_or("catalog content");
    // JSON first: a JSON document indented with tabs is not YAML.
    let value: serde_json::Value = match serde_json::from_str(content) {
        Ok(v) => v,
        Err(_) => match serde_yaml::from_str(content) {
            Ok(v) => v,
            Err(e) => {
                diagnostics.push(CatalogDiagnostic::error(format!("failed to parse catalog {named}: {e}")));
                return Vec::new();
            }
        },
    };
    let mut entries = match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|item| match super::resolver::remap_entry(item, base) {
                Ok(entry) => Some(entry),
                Err(e) => {
                    diagnostics.push(CatalogDiagnostic::warning(format!("skipping catalog entry: {e}")));
                    None
                }
            })
            .collect(),
        serde_json::Value::Object(ref map) => {
            let states_base = map
                .get("_defaults")
                .and_then(|d| d.get("base_url"))
                .is_some_and(|b| b.is_string());
            match base {
                None if !states_base => {
                    diagnostics.push(CatalogDiagnostic::error(format!(
                        "{named}: a knn_entries catalog needs a base or a _defaults.base_url \
                         to resolve its facets against"
                    )));
                    Vec::new()
                }
                _ => match super::knn_entries::parse_knn_entries_yaml(content, base.unwrap_or_default()) {
                    Ok(entries) => entries,
                    Err(e) => {
                        diagnostics.push(CatalogDiagnostic::error(format!("failed to parse {named}: {e}")));
                        Vec::new()
                    }
                },
            }
        }
        serde_json::Value::Null => Vec::new(),
        _ => {
            diagnostics.push(CatalogDiagnostic::error(format!(
                "catalog {named} is neither a list nor a map"
            )));
            Vec::new()
        }
    };
    if let Some(document) = document {
        for entry in &mut entries {
            entry.set_catalog_file(document);
        }
    }
    entries
}

/// A catalog built in memory. See [`CatalogBuilder`].
#[derive(Debug, Clone)]
pub struct BuiltCatalog {
    name: String,
    label: String,
    entries: Vec<CatalogEntry>,
    diagnostics: Vec<CatalogDiagnostic>,
}

impl CatalogSource for BuiltCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    fn origin(&self) -> CatalogOrigin {
        CatalogOrigin::Registered { label: self.label.clone() }
    }

    fn entries(&self, diagnostics: &mut Vec<CatalogDiagnostic>) -> Option<Vec<CatalogEntry>> {
        diagnostics.extend(self.diagnostics.iter().cloned());
        Some(self.entries.clone())
    }
}

/// Builds a catalog a program supplies: entries it built
/// ([`entry`](Self::entry)), or catalog content it holds
/// ([`content`](Self::content)), read as a catalog file of that shape is
/// read.
#[derive(Debug, Clone)]
pub struct CatalogBuilder {
    name: String,
    label: Option<String>,
    base: Option<String>,
    entries: Vec<CatalogEntry>,
    contents: Vec<String>,
}

impl CatalogBuilder {
    /// Start a catalog called `name`.
    pub fn named(name: impl Into<String>) -> Self {
        CatalogBuilder { name: name.into(), label: None, base: None, entries: Vec::new(), contents: Vec::new() }
    }

    /// Where relative entry paths and facets in [`content`](Self::content)
    /// resolve, read as a directory.
    pub fn base(mut self, base: crate::Location) -> Self {
        self.base = Some(crate::location::dir_form(&base.to_string()));
        self
    }

    /// An entry built with [`CatalogEntry::builder`].
    pub fn entry(mut self, entry: CatalogEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Several entries.
    pub fn entries(mut self, entries: impl IntoIterator<Item = CatalogEntry>) -> Self {
        self.entries.extend(entries);
        self
    }

    /// Catalog content: a canonical catalog list or a `knn_entries.yaml`
    /// document, in JSON or YAML.
    pub fn content(mut self, text: impl Into<String>) -> Self {
        self.contents.push(text.into());
        self
    }

    /// What diagnostics and the catalogs screen show for it; by default
    /// `registered:<name>`.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// The catalog. Content is read now: what it cannot give (it does
    /// not parse, or an entry's path is relative and there is no base)
    /// is kept as diagnostics, which the catalog reports when loaded.
    pub fn build(self) -> BuiltCatalog {
        let mut diagnostics = Vec::new();
        let mut entries = self.entries;
        for content in &self.contents {
            entries.extend(parse_catalog_content(content, self.base.as_deref(), None, &mut diagnostics));
        }
        let label = self.label.unwrap_or_else(|| format!("registered:{}", self.name));
        BuiltCatalog { name: self.name, label, entries, diagnostics }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Catalog, CatalogSources, Severity};

    const CANONICAL: &str = r#"[{"name":"ds","path":"ds/dataset.yaml","dataset_type":"dataset.yaml",
        "layout":{"profiles":{"default":{"base_vectors":"base.fvec"}}}}]"#;
    const KNN: &str = "knn-ds:default:\n  base: b.fvec\n  query: q.fvec\n  gt: g.ivec\n";

    /// Each entry as a comparable value: what a catalog file would say
    /// of it, and its home.
    fn shapes(entries: &[CatalogEntry]) -> Vec<(serde_json::Value, String)> {
        entries
            .iter()
            .map(|e| (serde_json::to_value(e).unwrap(), e.dataset_home_url().to_string()))
            .collect()
    }

    /// **A built catalog given a catalog's content holds what the
    /// located catalog holds**, for either shape: both read it with one
    /// parser, against the same base.
    #[test]
    fn a_built_catalog_equals_the_located_one_with_its_content() {
        for (file, content) in [("catalog.json", CANONICAL), ("knn_entries.yaml", KNN)] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(file), content).unwrap();
            let located = LocatedCatalog::new("c", &crate::Location::from_path(dir.path()));
            let built = CatalogBuilder::named("c").base(crate::Location::from_path(dir.path())).content(content).build();
            let mut diagnostics = Vec::new();
            let from_file = located.entries(&mut diagnostics).expect("the file is there");
            let from_content = built.entries(&mut diagnostics).unwrap();
            assert!(diagnostics.is_empty(), "{diagnostics:?}");
            assert_eq!(from_file.len(), 1, "{file}");
            assert_eq!(shapes(&from_file), shapes(&from_content), "{file}");
        }
    }

    /// **Content whose paths are relative needs a base**: without one
    /// the entry is skipped with a diagnostic, not resolved against the
    /// working directory.
    #[test]
    fn built_content_with_relative_paths_and_no_base_is_a_diagnostic() {
        for content in [CANONICAL, KNN] {
            let mut diagnostics = Vec::new();
            let entries = CatalogBuilder::named("c").content(content).build().entries(&mut diagnostics).unwrap();
            assert!(entries.is_empty(), "{content}");
            assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
            assert!(diagnostics[0].message.contains("base"), "{}", diagnostics[0]);
        }
        let absolute = CANONICAL.replace("ds/dataset.yaml", "https://h/ds/dataset.yaml");
        let mut diagnostics = Vec::new();
        let entries = CatalogBuilder::named("c").content(absolute).build().entries(&mut diagnostics).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
    }

    /// **Every entry of a located catalog records the document it was
    /// read from**, a canonical catalog's included. Only `knn_entries`
    /// entries once did, and the explorer's source view made up a path
    /// for the rest.
    #[test]
    fn a_located_catalogs_entries_record_their_document() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("catalog.json"), CANONICAL).unwrap();
        let catalog = Catalog::of(&CatalogSources::new().add_catalogs(&[dir.path().to_string_lossy().into_owned()]));
        let file = format!("{}catalog.json", crate::location::dir_form(&dir.path().to_string_lossy()));
        assert_eq!(catalog.lookup("ds").unwrap().catalog_file(), Some(file.as_str()));
    }

    /// **A registered catalog is a catalog like any other**: its entries
    /// carry its name, and a name it shares with a located catalog is
    /// ambiguous, never shadowed by order.
    #[test]
    fn a_registered_catalog_collides_like_a_located_one() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("catalog.json"), CANONICAL).unwrap();
        let entry = CatalogEntry::builder("ds")
            .home(&crate::Location::parse("https://h/ds/").unwrap())
            .dataset(crate::dataset::DatasetConfig::builder("ds").build().unwrap())
            .build()
            .unwrap();
        let sources = CatalogSources::new()
            .with(CatalogBuilder::named("mine").entry(entry).build())
            .add_catalogs(&[dir.path().to_string_lossy().into_owned()]);
        let catalog = Catalog::of(&sources);
        assert_eq!(catalog.datasets()[0].catalog_name(), Some("mine"), "added first, listed first");
        let err = catalog.lookup("ds").unwrap_err();
        let crate::Error::AmbiguousDataset { matches, .. } = &err else { panic!("{err}") };
        assert!(matches.iter().any(|m| m.contains("'mine': registered:mine")), "{matches:?}");
    }

    /// **A required catalog that gives nothing is an error; an optional
    /// one is not.** The requirement belongs to the set, not the source.
    #[test]
    fn only_a_required_catalog_that_gives_nothing_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let empty = LocatedCatalog::new("empty", &crate::Location::from_path(dir.path()));
        let required = Catalog::of(&CatalogSources::new().with(empty.clone()));
        assert_eq!(required.diagnostics().len(), 1, "{:?}", required.diagnostics());
        assert_eq!(required.diagnostics()[0].severity, Severity::Error);
        assert!(required.diagnostics()[0].message.contains("no catalog file found"));
        let optional = Catalog::of(&CatalogSources::new().with_optional(empty));
        assert!(optional.diagnostics().is_empty(), "{:?}", optional.diagnostics());
    }
}
