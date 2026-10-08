// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Catalog resolver — loads catalog entries from multiple sources (local files
//! and HTTP URLs) and provides search methods.
//!
//! This is the Rust equivalent of the upstream Java `Catalog` class. It takes
//! a [`CatalogSources`] configuration and fetches `catalog.json` from each
//! location, remapping layout-embedded entries into a unified list.

use crate::dataset::{CatalogEntry, CatalogLayout};

use super::CatalogDiagnostic;
use super::knn_entries::parse_knn_entries_yaml;
use super::sources::{
    catalog_file_for, ensure_trailing_slash, looks_like_catalog_file, CatalogSources,
};

/// A resolved catalog containing dataset entries from all configured sources.
///
/// Library API. Build one with [`Catalog::of`], then go from a name to
/// data: [`open_spec`](Self::open_spec) for a `name:selector` spec,
/// [`open_profile`](Self::open_profile) for one profile,
/// [`fetch`](Self::fetch) to bring a spec's facets into the cache.
/// Problems met while loading the catalogs are kept, not printed — see
/// [`diagnostics`](Self::diagnostics).
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    entries: Vec<CatalogEntry>,
    diagnostics: Vec<CatalogDiagnostic>,
}

impl Catalog {
    /// Build a catalog by loading entries from all locations in the given sources.
    ///
    /// Loading never fails as a whole: a location that cannot be read
    /// or parsed contributes a [`CatalogDiagnostic`] instead of entries —
    /// an error for a required location, nothing for an optional one
    /// that is simply absent. Read them with
    /// [`diagnostics`](Self::diagnostics); each is also logged at
    /// `warn` level.
    pub fn of(sources: &CatalogSources) -> Self {
        let mut entries = Vec::new();
        let mut diagnostics = sources.diagnostics().to_vec();

        let mut load_named = |src: &super::sources::NamedCatalogSource, required: bool| {
            let before = entries.len();
            load_catalog_entries(&src.location, &mut entries, required, &mut diagnostics);
            // Stamp every entry this source contributed with the
            // catalog's symbolic name so downstream surfaces (the
            // picker's catalog toggle screen, listings) can group
            // and filter by catalog.
            for e in entries[before..].iter_mut() {
                e.catalog_name = Some(src.name.clone());
            }
        };

        for src in sources.required() {
            load_named(src, true);
        }

        for src in sources.optional() {
            load_named(src, false);
        }

        for d in &diagnostics {
            log::warn!("{d}");
        }
        Catalog { entries, diagnostics }
    }

    /// Returns all dataset entries in the catalog.
    pub fn datasets(&self) -> &[CatalogEntry] {
        &self.entries
    }

    /// Returns true if no entries are loaded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// What went wrong while the sources were read and the catalogs
    /// loaded: unreadable locations, unparseable files, skipped
    /// entries. Empty when everything loaded cleanly. The command-line
    /// tools print these to stderr; a library caller decides.
    pub fn diagnostics(&self) -> &[CatalogDiagnostic] {
        &self.diagnostics
    }

    /// Find a dataset by exact name (case-insensitive).
    ///
    /// `None` both when no dataset has the name and when several do;
    /// [`lookup`](Self::lookup) says which, with suggestions.
    pub fn find_exact(&self, name: &str) -> Option<&CatalogEntry> {
        self.lookup(name).ok()
    }

    /// Find a dataset by exact name (case-insensitive), or say why not:
    /// [`Error::UnknownDataset`](crate::Error::UnknownDataset), carrying
    /// the datasets whose names contain `name` as suggestions, or
    /// [`Error::AmbiguousDataset`](crate::Error::AmbiguousDataset).
    pub fn lookup(&self, name: &str) -> crate::Result<&CatalogEntry> {
        let matches: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.name.eq_ignore_ascii_case(name))
            .collect();
        match matches.as_slice() {
            [one] => Ok(one),
            [] => Err(crate::Error::UnknownDataset {
                name: name.to_string(),
                suggestions: self.suggestions(name).iter().map(|e| e.name.clone()).collect(),
            }),
            _ => Err(crate::Error::AmbiguousDataset {
                name: name.to_string(),
                matches: matches.iter().map(|e| e.name.clone()).collect(),
            }),
        }
    }

    /// Datasets whose names contain `search`, case-insensitively — the
    /// "did you mean" list for a name that was not found.
    pub fn suggestions(&self, search: &str) -> Vec<&CatalogEntry> {
        if search.is_empty() {
            return Vec::new();
        }
        let lower = search.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.name.to_lowercase().contains(&lower))
            .collect()
    }

    /// Match datasets by glob pattern against their names.
    pub fn match_glob(&self, pattern: &str) -> Vec<&CatalogEntry> {
        // Simple glob: convert to regex-ish matching
        let regex = glob_to_regex(pattern);
        self.match_regex(&regex)
    }

    /// Match datasets by regex pattern against their names.
    ///
    /// Only a simple subset of regex syntax is understood; a pattern
    /// outside it falls back to a case-insensitive substring match,
    /// logged at `warn` level.
    pub fn match_regex(&self, pattern: &str) -> Vec<&CatalogEntry> {
        // Use simple substring/pattern matching without pulling in regex crate
        match simple_regex_match(pattern) {
            Some(matcher) => self.entries.iter().filter(|e| matcher(&e.name)).collect(),
            None => {
                log::warn!("unsupported regex pattern '{pattern}', falling back to substring match");
                let lower = pattern.to_lowercase();
                self.entries
                    .iter()
                    .filter(|e| e.name.to_lowercase().contains(&lower))
                    .collect()
            }
        }
    }

    /// Open a dataset by name, returning a `TestDataGroup` ready for
    /// facet access.
    ///
    /// No URL construction needed — the catalog resolves the location.
    /// A name that is unknown or ambiguous fails as
    /// [`lookup`](Self::lookup) does. To open by a full
    /// `name:selector` spec, use [`open_spec`](Self::open_spec).
    ///
    /// ```no_run
    /// # use vectordata::catalog::{Catalog, CatalogSources};
    /// let catalog = Catalog::of(&CatalogSources::new().configure_default());
    /// let group = catalog.open("my-dataset")?;
    /// let view = group.profile("default").unwrap();
    /// let base = view.base_vectors()?;
    /// # Ok::<(), vectordata::Error>(())
    /// ```
    pub fn open(&self, name: &str) -> crate::Result<crate::TestDataGroup> {
        Self::open_entry(self.lookup(name)?)
    }

    /// Open a [`CatalogEntry`] directly, dispatching on its shape:
    /// `knn_entries.yaml`-shape entries synthesise the
    /// [`TestDataGroup`](crate::TestDataGroup) from their embedded layout (no
    /// per-dataset `dataset.yaml` to re-fetch), while canonical
    /// entries load through [`TestDataGroup::load`](crate::TestDataGroup::load) with the
    /// entry's absolute `dataset.yaml` URL. Callers that already
    /// hold a `CatalogEntry` (e.g. the picker iterating `datasets()`)
    /// use this to avoid the `find_exact` round-trip.
    pub fn open_entry(entry: &CatalogEntry) -> crate::Result<crate::TestDataGroup> {
        if entry.dataset_type == "knn_entries.yaml" {
            return crate::TestDataGroup::from_catalog_entry(entry);
        }
        crate::TestDataGroup::load(&entry.path)
    }

    /// Open what a dataset spec names: **the call for turning a string
    /// a user typed into data.**
    ///
    /// A spec is `<head>[:<selector>]`. The head is a catalog name, a
    /// local directory or `dataset.yaml`, or a URL; the selector names
    /// profiles — a bare name (`default`), an expression
    /// (`size=10m,predicates=uniform*`), or `profile=*` for all — and
    /// no selector means `default`. The split is the one
    /// [`DatasetSpec::parse`](crate::dataset::selector::DatasetSpec::parse)
    /// makes, which knows that a URL's port and a Windows drive letter
    /// are not selectors; do not split a spec on `:` yourself.
    ///
    /// ```no_run
    /// # use vectordata::catalog::{Catalog, CatalogSources};
    /// let catalog = Catalog::of(&CatalogSources::new().configure_default());
    /// let selection = catalog.open_spec("my-dataset:size=10m")?;
    /// let view = selection.view()?; // exactly one profile, or an error
    /// # Ok::<(), vectordata::Error>(())
    /// ```
    pub fn open_spec(&self, spec: &str) -> crate::Result<DatasetSelection> {
        // A path that exists is a path, whatever punctuation it holds.
        if std::path::Path::new(spec).exists() {
            return self.open_selection(spec, None);
        }
        let parsed = crate::dataset::selector::DatasetSpec::parse(spec).map_err(|e| {
            crate::Error::Selection {
                dataset: crate::dataset::selector::DatasetSpec::split_head(spec).0.to_string(),
                message: e.to_string(),
            }
        })?;
        self.open_selection(&parsed.head, parsed.selector.as_ref().map(|s| s.text()))
    }

    /// [`open_spec`](Self::open_spec) with the head and the selector
    /// given separately — for a caller whose selector arrives on its
    /// own, such as a `--profile` option. `None` selects `default`.
    pub fn open_selection(&self, head: &str, selector: Option<&str>) -> crate::Result<DatasetSelection> {
        let opened = |label: &str, e: crate::Error| {
            crate::Error::Other(format!("failed to open dataset '{label}': {e}"))
        };
        let (dataset, group) = if crate::transport::is_remote_url(head) || std::path::Path::new(head).exists() {
            let group = crate::TestDataGroup::load(head).map_err(|e| opened(head, e))?;
            (head.to_string(), group)
        } else {
            let entry = self.lookup(head)?;
            // Checked against the catalog's own profile list first, so a
            // selector that matches nothing fails before the dataset is
            // fetched (PS-9).
            entry.select(selector).map_err(|e| crate::Error::Selection {
                dataset: entry.name.clone(),
                message: e.to_string(),
            })?;
            let group = Self::open_entry(entry).map_err(|e| opened(&entry.name, e))?;
            (entry.name.clone(), group)
        };
        let profiles = group.select(selector).map_err(|e| crate::Error::Selection {
            dataset: dataset.clone(),
            message: e.to_string(),
        })?;
        Ok(DatasetSelection { dataset, group, profiles })
    }

    /// Fetch what a dataset spec names into the local cache, with
    /// progress: [`open_spec`](Self::open_spec), then
    /// [`TestDataGroup::fetch`](crate::TestDataGroup::fetch) over the
    /// profiles it selected.
    ///
    /// ```no_run
    /// # use vectordata::catalog::{Catalog, CatalogSources};
    /// use vectordata::fetch::{FetchRequest, TextMeter};
    /// let catalog = Catalog::of(&CatalogSources::new().configure_default());
    /// catalog.fetch(
    ///     "my-dataset:default",
    ///     &FetchRequest::facets(["base_vectors", "query_vectors"]),
    ///     &mut TextMeter::stderr("Fetch"),
    /// )?;
    /// # Ok::<(), vectordata::Error>(())
    /// ```
    pub fn fetch(
        &self,
        spec: &str,
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> crate::Result<crate::fetch::FetchReport> {
        self.open_spec(spec)?.fetch(request, progress)
    }

    /// Open the one profile of a dataset a selector names, returning
    /// the view directly.
    ///
    /// `profile` is a selector (PS-3): a bare name as it always was, or
    /// an expression such as `size=10m,predicates=uniform-2`. It must
    /// name exactly one profile; a set is an error here, and
    /// [`open_profiles`](Self::open_profiles) is the surface that takes
    /// one (PS-10). The same as `open_selection(name, Some(profile))?.view()`.
    ///
    /// ```no_run
    /// # use vectordata::catalog::{Catalog, CatalogSources};
    /// let catalog = Catalog::of(&CatalogSources::new().configure_default());
    /// let view = catalog.open_profile("my-dataset", "default")?;
    /// let base = view.base_vectors()?;
    /// # Ok::<(), vectordata::Error>(())
    /// ```
    pub fn open_profile(&self, name: &str, profile: &str) -> crate::Result<std::sync::Arc<dyn crate::view::TestDataView>> {
        let group = self.open(name)?;
        let selected = group.select_one(Some(profile)).map_err(|e| crate::Error::Selection {
            dataset: name.to_string(),
            message: e.to_string(),
        })?;
        group.profile(&selected).ok_or_else(|| crate::Error::Selection {
            dataset: name.to_string(),
            message: format!("profile '{selected}' not found"),
        })
    }

    /// Open every profile of a dataset a selector names, size-ordered,
    /// each with its name (PS-9). `None` opens `default`; `profile=*`
    /// opens them all (PS-10).
    pub fn open_profiles(
        &self,
        name: &str,
        selector: Option<&str>,
    ) -> crate::Result<Vec<(String, std::sync::Arc<dyn crate::view::TestDataView>)>> {
        self.open_selection(name, selector)?.views()
    }
}

/// What a dataset spec resolved to: the dataset, and the profiles its
/// selector picked, in size order. Returned by
/// [`Catalog::open_spec`] and [`Catalog::open_selection`].
#[derive(Debug)]
pub struct DatasetSelection {
    dataset: String,
    group: crate::TestDataGroup,
    profiles: Vec<String>,
}

impl DatasetSelection {
    /// The dataset as resolved: its catalog name, or the path or URL
    /// it was opened from.
    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    /// The selected profiles' names, size-ordered (PS-9).
    pub fn profiles(&self) -> &[String] {
        &self.profiles
    }

    /// The opened dataset, for anything beyond the selection.
    pub fn group(&self) -> &crate::TestDataGroup {
        &self.group
    }

    /// The view of the one selected profile. More than one match is an
    /// [`Error::Selection`](crate::Error::Selection) naming them — use
    /// [`views`](Self::views) for a set.
    pub fn view(&self) -> crate::Result<std::sync::Arc<dyn crate::view::TestDataView>> {
        match self.profiles.as_slice() {
            [one] => self.group.profile(one).ok_or_else(|| crate::Error::Selection {
                dataset: self.dataset.clone(),
                message: format!("profile '{one}' not found"),
            }),
            many => Err(crate::Error::Selection {
                dataset: self.dataset.clone(),
                message: format!(
                    "the selector matches {} profiles ({}); narrow it to one",
                    many.len(),
                    many.join(", ")
                ),
            }),
        }
    }

    /// A view of every selected profile, with its name.
    pub fn views(&self) -> crate::Result<Vec<(String, std::sync::Arc<dyn crate::view::TestDataView>)>> {
        self.profiles
            .iter()
            .map(|p| {
                self.group
                    .profile(p)
                    .map(|v| (p.clone(), v))
                    .ok_or_else(|| crate::Error::Selection {
                        dataset: self.dataset.clone(),
                        message: format!("profile '{p}' not found"),
                    })
            })
            .collect()
    }

    /// Fetch the selected profiles' facets — [`TestDataGroup::fetch`](crate::TestDataGroup::fetch)
    /// over [`profiles`](Self::profiles).
    pub fn fetch(
        &self,
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> crate::Result<crate::fetch::FetchReport> {
        self.group.fetch(&self.profiles, request, progress)
    }

    /// Plan a [`fetch`](Self::fetch) without fetching.
    pub fn plan_fetch(
        &self,
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> crate::Result<crate::fetch::FetchPlan> {
        self.group.plan_fetch(&self.profiles, request, progress)
    }
}

/// Load catalog entries from a single location string.
///
/// The location may be an HTTP URL or a local file/directory path.
/// For layout-embedded entries, the `path` field is resolved relative
/// to the base location to construct full URLs.
///
/// Resolution strategy:
///
/// 1. **Explicit catalog file** — when `location` ends in `.yaml`,
///    `.yml`, or `.json`, the file is fetched once and dispatched by
///    content shape: a YAML/JSON sequence is treated as a canonical
///    catalog array; a YAML mapping is treated as a `knn_entries`-
///    style document (jvector `DataSetLoaderSimpleMFD` parity). No
///    sibling probing — the file *is* the catalog regardless of name.
///
/// 2. **Directory cascade** — when `location` is a directory or URL
///    prefix, the canonical filename set is walked in order:
///    `catalog.json`, `catalog.yaml`, `knn_entries.yaml`.
fn load_catalog_entries(
    location: &str,
    entries: &mut Vec<CatalogEntry>,
    required: bool,
    diags: &mut Vec<CatalogDiagnostic>,
) {
    if looks_like_catalog_file(location) {
        if !load_from_explicit_catalog_file(location, entries, diags) && required {
            diags.push(CatalogDiagnostic::error(format!("could not load catalog from {location}")));
        }
        return;
    }

    // Directory cascade.
    if try_load_canonical_catalog(location, entries, diags) { return; }
    if try_load_knn_entries(location, entries, diags) { return; }

    if required {
        diags.push(CatalogDiagnostic::error(format!(
            "no catalog file found at {location} \
             (tried catalog.json, catalog.yaml, knn_entries.yaml)"
        )));
    } else {
        log::debug!(
            "optional catalog {} has no catalog.json / catalog.yaml / knn_entries.yaml",
            location,
        );
    }
}

/// Fetch the content of an explicit-filename catalog (one whose URL
/// or path ends in `.yaml`/`.yml`/`.json`) and dispatch by content
/// shape: top-level sequence → canonical catalog array; top-level
/// mapping → `knn_entries`-style. Returns `true` when the file was
/// fetched and parsed successfully into at least one entry, or when
/// the file was parsed but contained no entries (still a success —
/// the file exists). Returns `false` when the file could not be
/// fetched at all.
fn load_from_explicit_catalog_file(
    location: &str,
    entries: &mut Vec<CatalogEntry>,
    diags: &mut Vec<CatalogDiagnostic>,
) -> bool {
    let content = match fetch_location_content(location) {
        Some(c) => c,
        None => return false,
    };

    // Generic YAML parse: subsumes JSON (every JSON document is also
    // valid YAML). This gives us a single value we can shape-match on
    // without paying for two separate parse passes.
    let value: serde_yaml::Value = match serde_yaml::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            diags.push(CatalogDiagnostic::error(format!("failed to parse catalog {location}: {e}")));
            return true;
        }
    };

    let parent = parent_location_of(location);
    let base_url = ensure_trailing_slash(&parent);

    match value {
        serde_yaml::Value::Sequence(seq) => {
            for item in seq {
                let as_json: serde_json::Value = match serde_json::to_value(&item) {
                    Ok(v) => v,
                    Err(e) => {
                        diags.push(CatalogDiagnostic::warning(format!("skipping catalog entry: {e}")));
                        continue;
                    }
                };
                match remap_entry(&as_json, &base_url) {
                    Ok(entry) => entries.push(entry),
                    Err(e) => diags.push(CatalogDiagnostic::warning(format!("skipping catalog entry: {e}"))),
                }
            }
            true
        }
        serde_yaml::Value::Mapping(_) => {
            // knn_entries-style: parser resolves facet paths relative
            // to the catalog file's parent directory (matching
            // jvector's per-entry resolution against the entry's
            // cache directory, with `_defaults.base_url` overriding).
            match parse_knn_entries_yaml(&content, &parent) {
                Ok(mut parsed) => {
                    // The parser only sees the parent directory; the
                    // resolver knows which document these entries came
                    // from. Without this, "show me the catalog source"
                    // surfaces a fabricated `<base_url>/knn_entries.yaml`
                    // that may name the wrong host AND the wrong file.
                    for e in &mut parsed {
                        e.catalog_file = Some(location.to_string());
                    }
                    entries.extend(parsed);
                    true
                }
                Err(e) => {
                    diags.push(CatalogDiagnostic::error(format!("failed to parse {location}: {e}")));
                    true
                }
            }
        }
        _ => {
            diags.push(CatalogDiagnostic::error(format!(
                "catalog {location} is neither a sequence nor a mapping"
            )));
            true
        }
    }
}

/// Fetch the raw content of any catalog location (HTTP URL or local
/// path). Returns `None` if the file/URL is unreachable so callers
/// can fall through to alternative probes.
fn fetch_location_content(location: &str) -> Option<String> {
    // Anything the shared transport speaks goes through the HTTP
    // fetcher (which normalises `s3://` → virtual-hosted-style HTTPS
    // before the wire). Everything else is treated as a local path.
    if crate::transport::is_remote_url(location) {
        fetch_http(location).ok()
    } else {
        std::fs::read_to_string(location).ok()
    }
}

/// Return the parent directory of a catalog file URL or path, with
/// no trailing slash. For `https://a/b/c.yaml` → `https://a/b`. For
/// `/x/y/z.yaml` → `/x/y`. For something with no separators returns
/// the empty string.
fn parent_location_of(location: &str) -> String {
    match location.rfind('/') {
        Some(idx) => location[..idx].to_string(),
        None => String::new(),
    }
}

/// Try the canonical `catalog.{json,yaml}` path at `location`.
/// Returns `true` when the file was read and successfully parsed
/// (entries appended); `false` when no canonical catalog file was
/// found OR the file exists but parsing failed (the caller falls
/// through to the next probe; the parse failure is logged here).
fn try_load_canonical_catalog(
    location: &str,
    entries: &mut Vec<CatalogEntry>,
    diags: &mut Vec<CatalogDiagnostic>,
) -> bool {
    let catalog_url = catalog_file_for(location);

    let content = if crate::transport::is_remote_url(&catalog_url) {
        match fetch_http(&catalog_url) {
            Ok(c) => c,
            Err(_) => return false, // missing — let the caller try the next probe
        }
    } else {
        let path = std::path::Path::new(&catalog_url);
        let file_path = if path.is_dir() {
            let json = path.join("catalog.json");
            if json.is_file() {
                json
            } else {
                let yaml = path.join("catalog.yaml");
                if yaml.is_file() {
                    yaml
                } else {
                    return false;
                }
            }
        } else if path.is_file() {
            path.to_path_buf()
        } else {
            return false;
        };

        match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(_) => return false,
        }
    };

    let parsed: Vec<serde_json::Value> = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => match serde_yaml::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                diags.push(CatalogDiagnostic::error(format!("failed to parse catalog {catalog_url}: {e}")));
                return true; // file exists, but couldn't parse — don't fall through
            }
        },
    };

    let base_url = ensure_trailing_slash(location);
    for value in parsed {
        match remap_entry(&value, &base_url) {
            Ok(entry) => entries.push(entry),
            Err(e) => {
                diags.push(CatalogDiagnostic::warning(format!("skipping catalog entry: {e}")));
            }
        }
    }
    true
}

/// Try the legacy `knn_entries.yaml` format at `location` as a
/// fallback when no canonical catalog file was found. See
/// [`super::knn_entries`] for the format spec.
///
/// Returns `true` if a `knn_entries.yaml` was located and parsed
/// (whether or not any entries were produced); `false` when no
/// such file exists at the location.
fn try_load_knn_entries(
    location: &str,
    entries: &mut Vec<CatalogEntry>,
    diags: &mut Vec<CatalogDiagnostic>,
) -> bool {
    let url = knn_entries_url_for(location);
    let content = if url.starts_with("http://") || url.starts_with("https://") {
        match fetch_http(&url) {
            Ok(c) => c,
            Err(_) => return false,
        }
    } else {
        let p = std::path::Path::new(&url);
        if !p.is_file() { return false; }
        match std::fs::read_to_string(p) {
            Ok(c) => c,
            Err(_) => return false,
        }
    };

    match parse_knn_entries_yaml(&content, location) {
        Ok(mut parsed) => {
            for e in &mut parsed {
                e.catalog_file = Some(url.clone());
            }
            entries.extend(parsed);
            true
        }
        Err(e) => {
            diags.push(CatalogDiagnostic::error(format!("failed to parse {url}: {e}")));
            true
        }
    }
}

/// Resolve a catalog location to the implied `knn_entries.yaml`
/// URL/path. Mirrors [`super::sources::catalog_file_for`] but for
/// the legacy filename.
pub(crate) fn knn_entries_url_for(location: &str) -> String {
    if location.ends_with("/knn_entries.yaml") { return location.to_string(); }
    let p = std::path::Path::new(location);
    if p.is_file()
        && p.file_name().and_then(|n| n.to_str()) == Some("knn_entries.yaml")
    {
        return location.to_string();
    }
    let base = ensure_trailing_slash(location);
    format!("{}knn_entries.yaml", base)
}

/// Remap a raw JSON/YAML catalog entry into a `CatalogEntry`.
///
/// Layout-embedded entries have `layout.attributes` and `layout.profiles`
/// and a relative `path`. The name is derived from the directory component
/// of the path (matching Java's `dirNameOfPath`).
fn remap_entry(value: &serde_json::Value, base_url: &str) -> Result<CatalogEntry, String> {
    if value.get("layout").is_some() {
        // Layout-embedded entry — remap to normalized form
        let path = value
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("layout entry missing 'path'")?;

        // Use explicit name if present, otherwise derive from path
        let name = if let Some(n) = value.get("name").and_then(|v| v.as_str()) {
            n.to_string()
        } else {
            dir_name_of_path(path)
        };

        // Resolve path relative to base URL to get the full URL
        let full_path = format!("{}{}", base_url, path);

        let layout: CatalogLayout = serde_json::from_value(value["layout"].clone())
            .map_err(|e| format!("failed to parse layout: {}", e))?;

        let dataset_type = value
            .get("dataset_type")
            .and_then(|v| v.as_str())
            .unwrap_or("dataset.yaml")
            .to_string();

        Ok(CatalogEntry {
            name,
            path: full_path,
            dataset_type,
            layout,
            catalog_file: None,
            catalog_name: None,
        })
    } else {
        // Direct entry — attributes and profiles at top level (e.g. HDF5 datasets)
        let name = value
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("entry missing 'name'")?
            .to_string();

        let path_str = value
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let full_path = format!("{}{}", base_url, path_str);

        let dataset_type = value
            .get("dataset_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Build layout from top-level attributes and profiles
        let attributes = value
            .get("attributes")
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        let profiles = value
            .get("profiles")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok(CatalogEntry {
            name,
            path: full_path,
            dataset_type,
            layout: CatalogLayout {
                format_version: crate::model::FORMAT_VERSION_BASE,
                profile_tags: Default::default(),
                attributes,
                profiles,
            },
            catalog_file: None,
            catalog_name: None,
        })
    }
}

/// Extract the directory name from a path, similar to Java's `dirNameOfPath`.
///
/// If the path ends with `dataset.yaml`, returns the parent directory name.
/// Otherwise returns the last path component.
fn dir_name_of_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.is_empty() {
        return path.to_string();
    }
    let last = parts[parts.len() - 1];
    if last.eq_ignore_ascii_case("dataset.yaml") && parts.len() >= 2 {
        parts[parts.len() - 2].to_string()
    } else {
        last.to_string()
    }
}

/// Fetch content from a remote URL using the process-wide shared
/// `reqwest::blocking::Client` so cert loading and DNS/connection
/// pool state amortise across every catalog access. Accepts the URL
/// forms `is_remote_url` recognises — `http(s)://` pass through, and
/// `s3://bucket/key` is rewritten via `normalize_remote_url` to the
/// virtual-hosted HTTPS endpoint before the wire.
/// A remote catalog file's text: from its server, keeping a copy under
/// the cache root; from that copy when the server cannot be reached; and
/// only from it in offline mode. A server that answers with an error
/// status is an error, so a missing catalog file is never papered over
/// by an old copy.
fn fetch_http(url: &str) -> Result<String, String> {
    let kept = crate::settings::cache_dir().ok().map(|root| {
        root.join(crate::cache::layout::CATALOG_COPIES_DIR)
            .join(crate::cache::layout::kept_catalog_name(url))
    });
    let from_kept = |why: &str| -> Result<String, String> {
        match kept.as_ref().filter(|p| p.is_file()) {
            Some(path) => {
                log::warn!("{why}; using the copy of {url} kept at {}", path.display());
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
            }
            None => Err(format!("{why}, and no copy of {url} is kept")),
        }
    };
    if crate::settings::offline() {
        return from_kept("offline mode is on");
    }
    match fetch_http_online(url) {
        Ok(text) => {
            if let Some(path) = &kept {
                // Best effort: an unwritable cache only costs the offline copy.
                let _ = path.parent().map(std::fs::create_dir_all);
                let _ = std::fs::write(path, &text);
            }
            Ok(text)
        }
        Err(Fetch::Unreachable(e)) => from_kept(&format!("{url} is unreachable ({e})")),
        Err(Fetch::Failed(e)) => Err(e),
    }
}

/// Why an online catalog fetch failed.
enum Fetch {
    /// The server could not be reached at all.
    Unreachable(String),
    /// It answered, with an error or an unreadable body.
    Failed(String),
}

fn fetch_http_online(url: &str) -> Result<String, Fetch> {
    let client = crate::transport::shared_client_for(url);
    let normalized = crate::transport::normalize_remote_url(url);

    let mut rb = client.get(normalized.as_ref());
    // Authenticate the catalog fetch with the SAME resolution as data reads
    // (`apply_read_auth`/`resolve_read_token`): `$VECTORDATA_TOKEN`, else the
    // login-stored credential keyed by origin. So a catalog and the datasets it
    // points at authenticate identically — private vecd namespaces are
    // fetchable, not just public-read ones.
    if let Ok(parsed) = url::Url::parse(url)
        && let Some(token) = crate::credentials::resolve_read_token(&parsed) {
            rb = rb.bearer_auth(token);
        }
    let response = rb.send().map_err(|e| {
        let msg = format!("HTTP request to {url} failed: {e}");
        if e.is_connect() || e.is_timeout() { Fetch::Unreachable(msg) } else { Fetch::Failed(msg) }
    })?;

    let status = response.status();
    if !status.is_success() {
        return Err(Fetch::Failed(format!("HTTP {} from {}", status.as_u16(), url)));
    }

    response.text()
        .map_err(|e| Fetch::Failed(format!("failed to read response from {}: {}", url, e)))
}

/// Convert a simple glob pattern to a regex-style matcher.
///
/// Supports `*` (match any) and `?` (match one character).
fn glob_to_regex(pattern: &str) -> String {
    let mut regex = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '*' => regex.push_str(".*"),
            '?' => regex.push('.'),
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                regex.push('\\');
                regex.push(ch);
            }
            _ => regex.push(ch),
        }
    }
    regex.push('$');
    regex
}

/// Simple regex-like matcher for common patterns without pulling in the regex crate.
///
/// Supports: `^...$` anchored, `.*` any, `.` single char, literal text.
/// Returns `None` for patterns too complex to handle.
#[allow(clippy::type_complexity)]
fn simple_regex_match(pattern: &str) -> Option<Box<dyn Fn(&str) -> bool>> {
    // For simple substring/exact matching
    let pat = pattern.to_string();

    if pat.starts_with('^') && pat.ends_with('$') {
        // Anchored pattern — try to interpret
        let inner = &pat[1..pat.len() - 1];
        if inner == ".*" {
            return Some(Box::new(|_| true));
        }
        // Check if it's just literal with .* wildcards
        if inner.contains(".*") {
            let parts: Vec<&str> = inner.split(".*").collect();
            let owned: Vec<String> = parts.iter().map(|s| s.to_lowercase()).collect();
            return Some(Box::new(move |name: &str| {
                let lower = name.to_lowercase();
                let mut pos = 0;
                for part in &owned {
                    if part.is_empty() {
                        continue;
                    }
                    match lower[pos..].find(part.as_str()) {
                        Some(idx) => pos += idx + part.len(),
                        None => return false,
                    }
                }
                true
            }));
        }
        // Plain literal
        let lower = inner.to_lowercase();
        return Some(Box::new(move |name: &str| name.to_lowercase() == lower));
    }

    // Unanchored — treat as substring
    let lower = pat.to_lowercase();
    Some(Box::new(move |name: &str| {
        name.to_lowercase().contains(&lower)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    fn make_entry(name: &str, profiles: &[&str]) -> CatalogEntry {
        let mut profile_group = IndexMap::new();
        for p in profiles {
            profile_group.insert(
                p.to_string(),
                crate::dataset::DSProfile {
                    maxk: None,
                    base_count: None,
                    partition: false,
                    views: IndexMap::new(),
                    ..Default::default()
                },
            );
        }
        CatalogEntry {
            name: name.to_string(),
            path: format!("{}/dataset.yaml", name),
            dataset_type: "dataset.yaml".to_string(),
            catalog_file: None,
            catalog_name: None,
            layout: CatalogLayout {
                format_version: crate::model::FORMAT_VERSION_BASE,
                profile_tags: Default::default(),
                attributes: None,
                profiles: crate::dataset::DSProfileGroup::from_profiles(profile_group),
            },
        }
    }

    #[test]
    fn test_find_exact() {
        let catalog = Catalog {
            entries: vec![
                make_entry("vecs-128", &["default"]),
                make_entry("glove-100", &["default", "10m"]),
            ],
            ..Catalog::default()
        };

        assert!(catalog.find_exact("vecs-128").is_some());
        assert!(catalog.find_exact("VECS-128").is_some()); // case insensitive
        assert!(catalog.find_exact("nonexistent").is_none());
    }

    #[test]
    fn test_match_glob() {
        let catalog = Catalog {
            entries: vec![
                make_entry("vecs-128", &["default"]),
                make_entry("vecs-256", &["default"]),
                make_entry("glove-100", &["default"]),
            ],
            ..Catalog::default()
        };

        let matches = catalog.match_glob("vecs-*");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_dir_name_of_path() {
        assert_eq!(dir_name_of_path("myds/dataset.yaml"), "myds");
        assert_eq!(dir_name_of_path("a/b/dataset.yaml"), "b");
        assert_eq!(dir_name_of_path("myds"), "myds");
        assert_eq!(dir_name_of_path("a/b/c"), "c");
    }

    #[test]
    fn test_remap_layout_entry() {
        let json = serde_json::json!({
            "name": "test-ds",
            "path": "test-ds/dataset.yaml",
            "dataset_type": "dataset.yaml",
            "layout": {
                "attributes": {
                    "distance_function": "L2"
                },
                "profiles": {
                    "default": {
                        "base_vectors": "base.fvec"
                    }
                }
            }
        });

        let entry = remap_entry(&json, "https://example.com/data/").unwrap();
        assert_eq!(entry.name, "test-ds");
        assert_eq!(entry.path, "https://example.com/data/test-ds/dataset.yaml");
        assert!(entry.layout.attributes.is_some());
        assert!(!entry.layout.profiles.is_empty());
    }

    #[test]
    fn test_remap_layout_entry_derives_name() {
        let json = serde_json::json!({
            "path": "myds/dataset.yaml",
            "dataset_type": "dataset.yaml",
            "layout": {
                "profiles": {
                    "default": {
                        "base_vectors": "base.fvec"
                    }
                }
            }
        });

        let entry = remap_entry(&json, "https://example.com/").unwrap();
        assert_eq!(entry.name, "myds");
    }

    #[test]
    fn test_local_catalog_loading() {
        let tmp = tempfile::tempdir().unwrap();
        let json = serde_json::json!([
            {
                "name": "alpha",
                "path": "alpha/dataset.yaml",
                "dataset_type": "dataset.yaml",
                "layout": {
                    "profiles": {
                        "default": {
                            "base_vectors": "base.fvec"
                        }
                    }
                }
            }
        ]);
        std::fs::write(
            tmp.path().join("catalog.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();

        let sources = CatalogSources::new()
            .add_catalogs(&[tmp.path().to_string_lossy().to_string()]);

        let catalog = Catalog::of(&sources);
        assert_eq!(catalog.datasets().len(), 1);
        assert_eq!(catalog.datasets()[0].name, "alpha");
    }

    #[test]
    fn test_empty_sources() {
        let sources = CatalogSources::new();
        let catalog = Catalog::of(&sources);
        assert!(catalog.is_empty());
    }

    #[test]
    fn test_glob_to_regex() {
        assert_eq!(glob_to_regex("vecs-*"), "^vecs-.*$");
        assert_eq!(glob_to_regex("?est"), "^.est$");
        assert_eq!(glob_to_regex("exact"), "^exact$");
    }
}
