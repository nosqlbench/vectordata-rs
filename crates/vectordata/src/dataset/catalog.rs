// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Catalog discovery and loading for dataset indexes.
//!
//! A catalog is a JSON or YAML array of [`CatalogEntry`] values. Each entry
//! embeds the publishable parts of a [`DatasetConfig`]
//! (attributes and profiles) under a `layout` key, alongside the dataset
//! `name`, relative `path` to the `dataset.yaml`, and a `dataset_type`
//! discriminator.
//!
//! Use [`find_catalog`] to locate a `catalog.json` or `catalog.yaml` in a
//! directory, and [`load_catalog`] to parse it. These are used by both
//! catalog generation (`veks catalog generate`) and catalog consumption
//! (`veks datasets list`).
//!
//! ## Catalog file format
//!
//! ```yaml
//! - name: glove-100
//!   path: vendor/glove-100/dataset.yaml
//!   dataset_type: dataset.yaml
//!   layout:
//!     attributes:
//!       distance_function: L2
//!       license: MIT
//!     profiles:
//!       default:
//!         base_vectors: base.fvec
//!         query_vectors: query.fvec
//! ```

use std::path::Path;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::config::{DatasetAttributes, DatasetConfig};
use super::profile::DSProfileGroup;

/// The layout block embedded within each catalog entry.
///
/// Contains the publishable subset of a `DatasetConfig`: attributes and
/// profiles. The `upstream` pipeline section is intentionally excluded —
/// it describes build steps, not dataset metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogLayout {
    /// The `dataset.yaml` format version this entry stands for (V-13).
    ///
    /// A catalog is what a consumer reads **before fetching**, so the
    /// version belongs in it: a client too old for a dataset can then
    /// say so from the listing instead of downloading it and failing on
    /// a type error or a missing shard.
    ///
    /// Absent means 1, like everywhere else (V-2). Always written when a
    /// catalog is generated (V-25), so a consumer can refuse a dataset
    /// before fetching it (V-13).
    #[serde(default = "crate::model::base_format_version")]
    pub format_version: u32,
    /// Dataset-level attributes (model, distance function, license, etc.).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub attributes: Option<DatasetAttributes>,
    /// The profile tag schema (PS-19), naming tags in order.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub profile_tags: IndexMap<String, serde_yaml::Value>,
    /// Named profiles mapping view names to data sources.
    #[serde(default)]
    pub profiles: DSProfileGroup,
}

/// What a catalog entry holds of its dataset's description.
#[derive(Debug, Clone)]
pub enum EntryDescription {
    /// The whole description is in the entry: a `knn_entries.yaml`
    /// catalog's datasets, and entries built in code. Opening the entry
    /// fetches nothing more.
    Embedded(Box<DatasetConfig>),
    /// A summary for listing and selecting. The description is the file
    /// at `descriptor`, read when the entry is opened.
    Referenced {
        /// Where the description is, as the catalog wrote it, resolved
        /// against the catalog's location.
        descriptor: String,
        /// The catalog's type for it: `dataset.yaml`, or another a
        /// catalog lists, such as `hdf5`.
        dataset_type: String,
        /// The listing's view of the dataset.
        summary: Box<CatalogLayout>,
    },
}

/// One dataset in a catalog: its name, its home, and its description,
/// whole or as a summary ([`EntryDescription`]).
///
/// Build one with [`CatalogEntry::builder`]; the catalog parsers do.
/// The file form matches the upstream Java catalog specification, so
/// catalog files produced by veks can be consumed by the Java vectordata
/// toolchain and vice versa.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "EntryWire", into = "EntryWire")]
pub struct CatalogEntry {
    name: String,
    /// The dataset's home as written, ending in `/`.
    home: String,
    description: EntryDescription,
    catalog_file: Option<String>,
    catalog_name: Option<String>,
    /// What overlays state for this dataset, merged over its own
    /// attributes. Never written to a catalog file: a listing says what
    /// the catalog says.
    overlay: Option<std::sync::Arc<AppliedOverlay>>,
}

/// An entry's attributes with the overlays that apply to it merged in,
/// and what each overlay stated.
#[derive(Debug, Clone)]
pub(crate) struct AppliedOverlay {
    attributes: DatasetAttributes,
    stated: Vec<crate::overlay::Stated>,
}

/// A catalog entry as catalog files write it.
#[derive(Serialize, Deserialize)]
struct EntryWire {
    name: String,
    path: String,
    #[serde(default = "default_dataset_type")]
    dataset_type: String,
    layout: CatalogLayout,
}

/// The `dataset_type` of an entry whose catalog file holds its whole
/// description.
const EMBEDDED_TYPE: &str = "knn_entries.yaml";

fn default_dataset_type() -> String {
    "dataset.yaml".to_string()
}

impl TryFrom<EntryWire> for CatalogEntry {
    type Error = String;

    /// A `knn_entries.yaml` entry's `path` is its catalog's base, with
    /// no dataset segment, and its home is the base and its name. Any
    /// other entry's `path` is its descriptor, and its home is the
    /// descriptor's directory.
    fn try_from(wire: EntryWire) -> Result<Self, String> {
        CatalogEntry::from_listing(wire.name, wire.path, wire.dataset_type, wire.layout)
    }
}

impl From<CatalogEntry> for EntryWire {
    fn from(entry: CatalogEntry) -> Self {
        let layout = entry.layout();
        let dataset_type = entry.dataset_type().to_string();
        let path = match entry.description {
            EntryDescription::Referenced { descriptor, .. } => descriptor,
            EntryDescription::Embedded(_) => {
                let home = entry.home.trim_end_matches('/');
                home.strip_suffix(&format!("/{}", entry.name)).unwrap_or(home).to_string()
            }
        };
        EntryWire { name: entry.name, path, dataset_type, layout }
    }
}

/// One attribute of `attributes` by name, known or not.
fn attribute_of(attributes: &DatasetAttributes, key: &str) -> Option<serde_yaml::Value> {
    if let Some(v) = attributes.other.get(key) {
        return Some(v.clone());
    }
    let all = serde_yaml::to_value(attributes).ok()?;
    all.get(key).filter(|v| !v.is_null()).cloned()
}

/// The parent directory of `path` as written, ending in `/`; `path`
/// itself when it has no `/`.
fn parent_dir_form(path: &str) -> String {
    match path.rsplit_once('/') {
        Some((parent, _)) => format!("{parent}/"),
        None => path.to_string(),
    }
}

/// A catalog entry being built. See [`CatalogEntry::builder`].
#[derive(Debug, Clone)]
pub struct EntryBuilder {
    name: String,
    home: Option<String>,
    description: Option<EntryDescription>,
}

impl CatalogEntry {
    /// Start building the entry for the dataset called `name`.
    ///
    /// An entry holds either the whole description
    /// ([`dataset`](EntryBuilder::dataset)), and then needs a
    /// [`home`](EntryBuilder::home) to resolve its relative facets
    /// against, or a summary and where the description is
    /// ([`descriptor`](EntryBuilder::descriptor)), whose directory is its
    /// home unless one is given.
    ///
    /// ```
    /// use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
    /// use vectordata::Location;
    ///
    /// let description = DatasetConfig::builder("demo")
    ///     .profile("default", |p| p.facet(StandardFacet::BaseVectors, "base.fvec"))
    ///     .build()
    ///     .map_err(vectordata::Error::Other)?;
    /// let entry = CatalogEntry::builder("demo")
    ///     .home(&Location::parse("https://example.com/datasets/demo")?)
    ///     .dataset(description)
    ///     .build()
    ///     .map_err(vectordata::Error::Other)?;
    /// assert_eq!(entry.dataset_home_url(), "https://example.com/datasets/demo/");
    /// assert_eq!(
    ///     entry.resolve_facet_url("base.fvec").as_deref(),
    ///     Some("https://example.com/datasets/demo/base.fvec"),
    /// );
    /// # Ok::<(), vectordata::Error>(())
    /// ```
    pub fn builder(name: impl Into<String>) -> EntryBuilder {
        EntryBuilder { name: name.into(), home: None, description: None }
    }

    /// The entry a catalog file lists: `path` (as written, or resolved
    /// against the catalog's location), its `dataset_type`, and its
    /// `layout`. A `knn_entries.yaml` entry's `path` is its catalog's
    /// base, with no dataset segment, so its home is the base and its
    /// name, and its layout is its whole description. Any other entry's
    /// `path` is its descriptor, whose directory is its home.
    pub(crate) fn from_listing(
        name: String,
        path: String,
        dataset_type: String,
        layout: CatalogLayout,
    ) -> Result<Self, String> {
        let builder = CatalogEntry::builder(name.clone());
        if dataset_type == EMBEDDED_TYPE {
            let home = format!("{}/{}/", path.trim_end_matches('/'), name);
            builder.home_as_written(home).dataset(DatasetConfig::from_layout(&name, layout)?).build()
        } else {
            builder.descriptor_as_written(path, dataset_type, layout).build()
        }
    }

    /// The dataset's name, which the catalog finds it by.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What the entry holds of the dataset's description.
    pub fn description(&self) -> &EntryDescription {
        &self.description
    }

    /// The catalog's type for the entry: `knn_entries.yaml` for an
    /// embedded description, else what the catalog states
    /// (`dataset.yaml` when it states nothing).
    pub fn dataset_type(&self) -> &str {
        match &self.description {
            EntryDescription::Embedded(_) => EMBEDDED_TYPE,
            EntryDescription::Referenced { dataset_type, .. } => dataset_type,
        }
    }

    /// Where the entry points: its descriptor, or for an embedded
    /// description, its home. What a listing shows as the dataset's
    /// location.
    pub fn location(&self) -> &str {
        match &self.description {
            EntryDescription::Embedded(_) => &self.home,
            EntryDescription::Referenced { descriptor, .. } => descriptor,
        }
    }

    /// The `dataset.yaml` format version the entry stands for (V-13).
    pub fn format_version(&self) -> u32 {
        match &self.description {
            EntryDescription::Embedded(d) => d.format_version,
            EntryDescription::Referenced { summary, .. } => summary.format_version,
        }
    }

    /// Dataset-level attributes (model, distance function, license,
    /// etc.), with what overlays state merged in
    /// ([`crate::overlay`]); [`attribute_source`](Self::attribute_source)
    /// says where each came from.
    pub fn attributes(&self) -> Option<&DatasetAttributes> {
        match &self.overlay {
            Some(o) => Some(&o.attributes),
            None => self.own_attributes(),
        }
    }

    /// The attributes the dataset itself declares, before overlays.
    pub fn own_attributes(&self) -> Option<&DatasetAttributes> {
        match &self.description {
            EntryDescription::Embedded(d) => d.attributes.as_ref(),
            EntryDescription::Referenced { summary, .. } => summary.attributes.as_ref(),
        }
    }

    /// One attribute by name, a namespaced consumer key
    /// (`fvec.ncentroids`) as much as a known one, overlays merged in.
    pub fn attribute(&self, key: &str) -> Option<serde_yaml::Value> {
        attribute_of(self.attributes()?, key)
    }

    /// Where attribute `key`'s value came from: an overlay, or the
    /// dataset. `None` when nothing sets it.
    pub fn attribute_source(&self, key: &str) -> Option<crate::overlay::AttributeSource> {
        if let Some(stated) = self.overlay.as_ref().and_then(|o| o.stated.iter().find(|s| s.key == key)) {
            return Some(stated.source.clone());
        }
        self.own_attributes()
            .and_then(|a| attribute_of(a, key))
            .map(|_| crate::overlay::AttributeSource::Dataset)
    }

    /// One consumer's keys (`fvec.` for namespace `fvec`) and their
    /// values, overlays merged in.
    pub fn attributes_in(&self, namespace: &str) -> Vec<(String, serde_yaml::Value)> {
        let prefix = format!("{namespace}.");
        self.attributes()
            .map(|a| a.other.iter().filter(|(k, _)| k.starts_with(&prefix)).map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }

    /// What overlays state for this entry: each key, its value and
    /// its source. Empty when none applies.
    pub fn overlaid(&self) -> &[crate::overlay::Stated] {
        self.overlay.as_ref().map(|o| o.stated.as_slice()).unwrap_or(&[])
    }

    /// Merge `stated` over the entry's own attributes. A distance
    /// function the dataset declares differently is a warning (OV-5):
    /// the overlay's is in force.
    pub(crate) fn apply_overlay(&mut self, stated: Vec<crate::overlay::Stated>) {
        if stated.is_empty() {
            return;
        }
        let own = self.own_attributes().cloned().unwrap_or_default();
        let mut attributes = own.clone();
        for s in &stated {
            if s.key == "distance_function"
                && let (Some(declared), Some(claimed)) = (own.distance(), s.value.as_str().and_then(crate::DistanceFunction::parse))
                && declared != claimed
            {
                crate::overlay::warn_contradiction(&self.name, &s.key, claimed.name(), &format!("the dataset's own {}", declared.name()), &s.source);
            }
            if !attributes.set_known(&s.key, &s.value) {
                attributes.other.insert(s.key.clone(), s.value.clone());
            }
        }
        self.overlay = Some(std::sync::Arc::new(AppliedOverlay { attributes, stated }));
    }

    /// The profile tag schema (PS-19).
    pub fn profile_tags(&self) -> &IndexMap<String, serde_yaml::Value> {
        match &self.description {
            EntryDescription::Embedded(d) => &d.profile_tags,
            EntryDescription::Referenced { summary, .. } => &summary.profile_tags,
        }
    }

    /// The dataset's profiles, resolved.
    pub fn profiles(&self) -> &DSProfileGroup {
        match &self.description {
            EntryDescription::Embedded(d) => &d.profiles,
            EntryDescription::Referenced { summary, .. } => &summary.profiles,
        }
    }

    /// The entry's listing view, as a catalog file writes it under
    /// `layout`.
    pub fn layout(&self) -> CatalogLayout {
        match &self.description {
            EntryDescription::Embedded(d) => CatalogLayout {
                format_version: d.format_version,
                attributes: d.attributes.clone(),
                profile_tags: d.profile_tags.clone(),
                profiles: d.profiles.clone(),
            },
            EntryDescription::Referenced { summary, .. } => (**summary).clone(),
        }
    }

    /// Where the catalog document this entry was read from is (URL or
    /// path), when the catalog that loaded it knows. For a
    /// `knn_entries.yaml` catalog this is the only record of where the
    /// file lives: its `_defaults.base_url` may be another host. `None`
    /// for an entry built in code.
    pub fn catalog_file(&self) -> Option<&str> {
        self.catalog_file.as_deref()
    }

    /// The name of the configured catalog this entry came from (the
    /// `catalogs.yaml` key, or the synthesized `catalog-N`). `None` for
    /// an entry no catalog has loaded.
    pub fn catalog_name(&self) -> Option<&str> {
        self.catalog_name.as_deref()
    }

    /// Record which catalog document the entry was read from. Set by
    /// the loader that reads the document.
    pub(crate) fn set_catalog_file(&mut self, file: impl Into<String>) {
        self.catalog_file = Some(file.into());
    }

    /// Record which configured catalog adopted the entry.
    pub(crate) fn set_catalog_name(&mut self, name: impl Into<String>) {
        self.catalog_name = Some(name.into());
    }

    /// The dataset's *home URL* — the location the cache mirrors
    /// under `<cache_root>/<dataset_name>/`, always with a trailing
    /// slash, as written. For a `knn_entries.yaml` entry it is the
    /// catalog base and the dataset's name; for a referenced entry, the
    /// descriptor's directory. Single authority for this derivation:
    /// the group loader and every cache-path computation read it here.
    pub fn dataset_home_url(&self) -> &str {
        &self.home
    }

    /// Resolve a facet view's raw source path to an absolute URL/path
    /// to read. Window notation (`base.fvec[0..1M)`) is stripped, then
    /// the source is resolved against [`Self::dataset_home_url`] (the
    /// single authority for the dataset's home) by
    /// [`Location::resolve`](crate::Location::resolve), the rule the
    /// dataset open reads facets by: a URL or leading `/` is kept as
    /// written, `file://` becomes its path, and a relative source joins
    /// by URL rules, so `..` resolves. `None` for an empty source.
    ///
    /// This is the one resolver the picker's facet probes and cache
    /// drop, `ping`, and the `datasets list` filter/completion probes
    /// all go through, so each names the file the open reads.
    pub fn resolve_facet_url(&self, raw_path: &str) -> Option<String> {
        let raw = strip_window_suffix(raw_path);
        if raw.is_empty() {
            return None;
        }
        crate::Location::parse(&self.home)
            .and_then(|home| home.resolve(raw))
            .ok()
    }

    /// Returns view names from the default profile (convenience accessor).
    pub fn view_names(&self) -> Vec<&str> {
        self.profiles().view_names()
    }

    /// Returns the number of profiles.
    pub fn profile_count(&self) -> usize {
        self.profiles().profile_names().len()
    }

    /// Returns profile names.
    pub fn profile_names(&self) -> Vec<&str> {
        self.profiles().profile_names()
    }

    /// The profiles a selector names in this entry, size-ordered;
    /// `None` is `default` (PS-10). A catalog entry is what a consumer
    /// reads before fetching, so a selector resolves here the same way
    /// it does against the loaded dataset.
    pub fn select(
        &self,
        selector: Option<&str>,
    ) -> Result<Vec<String>, crate::dataset::selector::SelectionError> {
        self.profiles().select(selector)
    }

    /// The one profile a selector names in this entry (PS-10).
    pub fn select_one(
        &self,
        selector: Option<&str>,
    ) -> Result<String, crate::dataset::selector::SelectionError> {
        self.profiles().select_one(selector)
    }
}

impl EntryBuilder {
    /// The dataset's home: where its relative facets resolve, and what
    /// the cache mirrors under `<cache_root>/<name>/`. Read as a
    /// directory.
    pub fn home(self, home: &crate::Location) -> Self {
        self.home_as_written(home.to_string())
    }

    /// The home as a catalog wrote it: kept as written, because it is
    /// the identity the dataset's cache records.
    pub(crate) fn home_as_written(mut self, home: impl Into<String>) -> Self {
        self.home = Some(crate::location::dir_form(&home.into()));
        self
    }

    /// The whole description: the entry is embedded and opens with no
    /// further fetch.
    pub fn dataset(mut self, description: DatasetConfig) -> Self {
        self.description = Some(EntryDescription::Embedded(Box::new(description)));
        self
    }

    /// Where the description is (a `dataset.yaml`), with the summary a
    /// listing shows until it is opened.
    pub fn descriptor(self, descriptor: &crate::Location, summary: CatalogLayout) -> Self {
        self.descriptor_as_written(descriptor.to_string(), default_dataset_type(), summary)
    }

    /// The descriptor as a catalog wrote it, and the catalog's type for
    /// it.
    pub(crate) fn descriptor_as_written(
        mut self,
        descriptor: impl Into<String>,
        dataset_type: impl Into<String>,
        summary: CatalogLayout,
    ) -> Self {
        self.description = Some(EntryDescription::Referenced {
            descriptor: descriptor.into(),
            dataset_type: dataset_type.into(),
            summary: Box::new(summary),
        });
        self
    }

    /// The entry. Refused without a description, or for an embedded one
    /// without a home.
    pub fn build(self) -> Result<CatalogEntry, String> {
        let name = self.name;
        let description = self
            .description
            .ok_or_else(|| format!("catalog entry '{name}': no dataset or descriptor"))?;
        let home = match (&description, self.home) {
            (_, Some(home)) => home,
            (EntryDescription::Referenced { descriptor, .. }, None) => parent_dir_form(descriptor),
            (EntryDescription::Embedded(_), None) => {
                return Err(format!("catalog entry '{name}': an embedded dataset needs a home"));
            }
        };
        Ok(CatalogEntry { name, home, description, catalog_file: None, catalog_name: None, overlay: None })
    }
}

/// Strip a window notation (`[lo..hi)` or `(lo..hi]`) from a source
/// path, returning just the underlying file path. Source strings in
/// `dataset.yaml` use these brackets to denote subranges of a shared
/// base file (e.g. `base.fvec[0..1000000)`).
pub fn strip_window_suffix(source_path: &str) -> &str {
    match source_path.find(['[', '(']) {
        Some(bracket) => &source_path[..bracket],
        None => source_path,
    }
}

/// Load catalog entries from a `catalog.json` or `catalog.yaml` file.
///
/// Returns `Ok(entries)` on success, or an error string if the file cannot
/// be read or parsed.
pub fn load_catalog(path: &Path) -> Result<Vec<CatalogEntry>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "json" => serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse {}: {}", path.display(), e)),
        "yaml" | "yml" => serde_yaml::from_str(&content)
            .map_err(|e| format!("failed to parse {}: {}", path.display(), e)),
        _ => Err(format!("unsupported catalog format: {}", path.display())),
    }
}

/// Attempt to find and load a catalog file in the given directory.
///
/// Checks for `catalog.json` first, then `catalog.yaml`. Returns `None`
/// if neither exists.
pub fn find_catalog(dir: &Path) -> Option<Result<Vec<CatalogEntry>, String>> {
    let json_path = dir.join("catalog.json");
    if json_path.is_file() {
        return Some(load_catalog(&json_path));
    }
    let yaml_path = dir.join("catalog.yaml");
    if yaml_path.is_file() {
        return Some(load_catalog(&yaml_path));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_facet_url_joins_relative_against_home() {
        // A dataset.yaml-shaped entry: the facet source is relative to
        // the dataset.yaml's directory. Resolution must produce the
        // absolute URL the probe reads — the bug was a bare relative
        // string being treated as a local file.
        let json = r#"[{"name":"sift1m",
            "path":"https://host/bucket/sift1m/dataset.yaml",
            "dataset_type":"dataset.yaml",
            "layout":{"profiles":{"default":{
                "base_vectors":"profiles/base/base_vectors.fvecs"}}}}]"#;
        let e: Vec<CatalogEntry> = serde_json::from_str(json).unwrap();
        let e = &e[0];
        assert_eq!(
            e.resolve_facet_url("profiles/base/base_vectors.fvecs").as_deref(),
            Some("https://host/bucket/sift1m/profiles/base/base_vectors.fvecs")
        );
        // Window suffix is stripped before joining.
        assert_eq!(
            e.resolve_facet_url("profiles/base/base_vectors.fvecs[0..1000)").as_deref(),
            Some("https://host/bucket/sift1m/profiles/base/base_vectors.fvecs")
        );
        // Already-absolute sources pass through unchanged (knn_entries
        // catalogs pre-resolve their facet paths this way).
        assert_eq!(
            e.resolve_facet_url("s3://other/x.fvecs").as_deref(),
            Some("s3://other/x.fvecs")
        );
        assert_eq!(e.resolve_facet_url("").as_deref(), None);
    }

    /// **A catalog file reads and writes the same text**, for either
    /// kind of entry: the entry type changed underneath, and the files
    /// a generator writes and the Java toolchain reads must not.
    #[test]
    fn an_entry_writes_the_form_it_was_read_from() {
        for json in [
            r#"{"name":"ds","path":"vendor/ds/dataset.yaml","dataset_type":"dataset.yaml","layout":{"format_version":1,"profiles":{"default":{"base_vectors":"base.fvec"}}}}"#,
            r#"{"name":"h5","path":"https://h/x/h5.hdf5","dataset_type":"hdf5","layout":{"format_version":1,"profiles":{}}}"#,
            r#"{"name":"ds","path":"s3://bucket/base","dataset_type":"knn_entries.yaml","layout":{"format_version":1,"profiles":{"default":{"base_vectors":"s3://bucket/base/b.fvec"}}}}"#,
        ] {
            let entry: CatalogEntry = serde_json::from_str(json).unwrap();
            assert_eq!(serde_json::to_string(&entry).unwrap(), json);
        }
    }

    /// **An entry's kind is its description**: a `knn_entries.yaml`
    /// entry holds its whole description and is homed at its base and
    /// name; any other is a summary homed at its descriptor's directory.
    #[test]
    fn an_entry_is_embedded_or_referenced_by_its_listing() {
        let knn: CatalogEntry = serde_json::from_str(
            r#"{"name":"ds","path":"s3://bucket/base/","dataset_type":"knn_entries.yaml","layout":{"profiles":{}}}"#,
        )
        .unwrap();
        assert!(matches!(knn.description(), EntryDescription::Embedded(_)));
        assert_eq!(knn.dataset_home_url(), "s3://bucket/base/ds/");
        let listed: CatalogEntry = serde_json::from_str(
            r#"{"name":"ds","path":"https://h/a b/ds/dataset.yaml","layout":{"profiles":{}}}"#,
        )
        .unwrap();
        assert!(matches!(listed.description(), EntryDescription::Referenced { .. }));
        assert_eq!(listed.dataset_type(), "dataset.yaml");
        assert_eq!(listed.dataset_home_url(), "https://h/a b/ds/", "kept as written");
    }

    /// **An embedded entry needs a home** to resolve its facets against;
    /// a referenced one takes its descriptor's directory.
    #[test]
    fn an_embedded_entry_without_a_home_is_refused() {
        let description = DatasetConfig::builder("ds").build().unwrap();
        let err = CatalogEntry::builder("ds").dataset(description).build().unwrap_err();
        assert!(err.contains("needs a home"), "{err}");
        assert!(CatalogEntry::builder("ds").build().is_err());
    }

    #[test]
    fn test_roundtrip_json() {
        let json = r#"[
  {
    "name": "test-ds",
    "path": "vendor/test/dataset.yaml",
    "dataset_type": "dataset.yaml",
    "layout": {
      "attributes": {
        "distance_function": "COSINE",
        "license": "MIT"
      },
      "profiles": {
        "default": {
          "base_vectors": "base.fvec",
          "query_vectors": "query.fvec"
        }
      }
    }
  }
]"#;
        let entries: Vec<CatalogEntry> = serde_json::from_str(json).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test-ds");
        assert_eq!(entries[0].dataset_type(), "dataset.yaml");
        assert_eq!(
            entries[0].attributes().unwrap().distance_function.as_deref(),
            Some("COSINE")
        );
        assert_eq!(entries[0].profile_count(), 1);
        assert_eq!(entries[0].view_names().len(), 2);
    }

    #[test]
    fn test_roundtrip_yaml() {
        let yaml = r#"
- name: glove-100
  path: glove-100/dataset.yaml
  dataset_type: dataset.yaml
  layout:
    profiles:
      default:
        base_vectors: base.fvec
"#;
        let entries: Vec<CatalogEntry> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "glove-100");
        assert!(entries[0].attributes().is_none());
        assert_eq!(entries[0].profile_count(), 1);
    }

    #[test]
    fn test_default_dataset_type() {
        // Missing dataset_type should default to "dataset.yaml"
        let json = r#"[{
            "name": "x",
            "path": "x/dataset.yaml",
            "layout": { "profiles": {} }
        }]"#;
        let entries: Vec<CatalogEntry> = serde_json::from_str(json).unwrap();
        assert_eq!(entries[0].dataset_type(), "dataset.yaml");
    }

    #[test]
    fn test_load_catalog_json_file() {
        let tmp = tempfile::tempdir().unwrap();
        let json = r#"[{"name":"a","path":"a/dataset.yaml","dataset_type":"dataset.yaml","layout":{"profiles":{"default":{"base_vectors":"b.fvec"}}}}]"#;
        std::fs::write(tmp.path().join("catalog.json"), json).unwrap();

        let entries = load_catalog(&tmp.path().join("catalog.json")).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "a");
    }

    #[test]
    fn test_find_catalog_prefers_json() {
        let tmp = tempfile::tempdir().unwrap();
        let json = r#"[{"name":"from-json","path":"a/dataset.yaml","dataset_type":"dataset.yaml","layout":{"profiles":{}}}]"#;
        let yaml = "- name: from-yaml\n  path: a/dataset.yaml\n  dataset_type: dataset.yaml\n  layout:\n    profiles: {}\n";
        std::fs::write(tmp.path().join("catalog.json"), json).unwrap();
        std::fs::write(tmp.path().join("catalog.yaml"), yaml).unwrap();

        let entries = find_catalog(tmp.path()).unwrap().unwrap();
        assert_eq!(entries[0].name, "from-json");
    }

    #[test]
    fn test_find_catalog_falls_back_to_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = "- name: from-yaml\n  path: a/dataset.yaml\n  dataset_type: dataset.yaml\n  layout:\n    profiles: {}\n";
        std::fs::write(tmp.path().join("catalog.yaml"), yaml).unwrap();

        let entries = find_catalog(tmp.path()).unwrap().unwrap();
        assert_eq!(entries[0].name, "from-yaml");
    }

    #[test]
    fn test_find_catalog_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(find_catalog(tmp.path()).is_none());
    }
}
