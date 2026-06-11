// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Catalog resolver — loads catalog entries from multiple sources (local files
//! and HTTP URLs) and provides search methods.
//!
//! This is the Rust equivalent of the upstream Java `Catalog` class. It takes
//! a [`CatalogSources`] configuration and fetches `catalog.json` from each
//! location, remapping layout-embedded entries into a unified list.

use crate::dataset::{CatalogEntry, CatalogLayout};

use super::knn_entries::parse_knn_entries_yaml;
use super::sources::{
    catalog_file_for, ensure_trailing_slash, looks_like_catalog_file, CatalogSources,
};

/// A resolved catalog containing dataset entries from all configured sources.
#[derive(Debug, Clone)]
pub struct Catalog {
    entries: Vec<CatalogEntry>,
}

impl Catalog {
    /// Build a catalog by loading entries from all locations in the given sources.
    ///
    /// Required locations that fail to load cause a fatal error (process exit).
    /// Optional locations that fail are silently ignored.
    pub fn of(sources: &CatalogSources) -> Self {
        let mut entries = Vec::new();

        for location in sources.required() {
            load_catalog_entries(location, &mut entries, true);
        }

        for location in sources.optional() {
            load_catalog_entries(location, &mut entries, false);
        }

        Catalog { entries }
    }

    /// Returns all dataset entries in the catalog.
    pub fn datasets(&self) -> &[CatalogEntry] {
        &self.entries
    }

    /// Returns true if no entries are loaded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find a dataset by exact name (case-insensitive).
    pub fn find_exact(&self, name: &str) -> Option<&CatalogEntry> {
        let matches: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.name.eq_ignore_ascii_case(name))
            .collect();
        match matches.len() {
            1 => Some(matches[0]),
            0 => None,
            _ => {
                eprintln!(
                    "ERROR: multiple datasets matching '{}': {}",
                    name,
                    matches.iter().map(|e| e.name.as_str()).collect::<Vec<_>>().join(", ")
                );
                None
            }
        }
    }

    /// Match datasets by glob pattern against their names.
    pub fn match_glob(&self, pattern: &str) -> Vec<&CatalogEntry> {
        // Simple glob: convert to regex-ish matching
        let regex = glob_to_regex(pattern);
        self.match_regex(&regex)
    }

    /// Match datasets by regex pattern against their names.
    pub fn match_regex(&self, pattern: &str) -> Vec<&CatalogEntry> {
        // Use simple substring/pattern matching without pulling in regex crate
        match simple_regex_match(pattern) {
            Some(matcher) => self.entries.iter().filter(|e| matcher(&e.name)).collect(),
            None => {
                eprintln!("warning: unsupported regex pattern '{}', falling back to substring match", pattern);
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
    /// This is the primary entry point for consumers: name → data.
    /// No URL construction needed — the catalog resolves the location.
    ///
    /// ```rust,ignore
    /// let catalog = Catalog::of(&CatalogSources::new().configure_default());
    /// let group = catalog.open("my-dataset")?;
    /// let view = group.profile("default").unwrap();
    /// let base = view.base_vectors()?;
    /// ```
    pub fn open(&self, name: &str) -> crate::Result<crate::TestDataGroup> {
        let entry = self.find_exact(name)
            .ok_or_else(|| crate::Error::Other(format!("dataset '{}' not found in catalog", name)))?;
        Self::open_entry(entry)
    }

    /// Open a [`CatalogEntry`] directly, dispatching on its shape:
    /// `knn_entries.yaml`-shape entries synthesise the
    /// [`TestDataGroup`] from their embedded layout (no
    /// per-dataset `dataset.yaml` to re-fetch), while canonical
    /// entries load through [`TestDataGroup::load`] with the
    /// entry's absolute `dataset.yaml` URL. Callers that already
    /// hold a `CatalogEntry` (e.g. the picker iterating `datasets()`)
    /// use this to avoid the `find_exact` round-trip.
    pub fn open_entry(entry: &CatalogEntry) -> crate::Result<crate::TestDataGroup> {
        if entry.dataset_type == "knn_entries.yaml" {
            return crate::TestDataGroup::from_catalog_entry(entry);
        }
        crate::TestDataGroup::load(&entry.path)
    }

    /// Open a specific profile of a dataset by name, returning the view
    /// directly.
    ///
    /// ```rust,ignore
    /// let catalog = Catalog::of(&CatalogSources::new().configure_default());
    /// let view = catalog.open_profile("my-dataset", "default")?;
    /// let base = view.base_vectors()?;
    /// ```
    pub fn open_profile(&self, name: &str, profile: &str) -> crate::Result<std::sync::Arc<dyn crate::view::TestDataView>> {
        let group = self.open(name)?;
        group.profile(profile)
            .ok_or_else(|| crate::Error::Other(format!("profile '{}' not found in dataset '{}'", profile, name)))
    }

    /// Print all datasets to stderr, highlighting any that contain the
    /// search term as a substring.
    pub fn list_datasets(&self, search: &str) {
        if !search.is_empty() {
            eprintln!("Dataset '{}' not found.", search);
        }

        if self.entries.is_empty() {
            eprintln!("No datasets are available in the catalog.");
            return;
        }

        if !search.is_empty() {
            let lower = search.to_lowercase();
            let matches: Vec<_> = self
                .entries
                .iter()
                .filter(|e| e.name.to_lowercase().contains(&lower))
                .collect();
            if !matches.is_empty() {
                eprintln!("Did you mean one of these datasets?");
                for entry in &matches {
                    print_dataset_with_profiles(entry);
                }
                eprintln!();
            }
        }

        eprintln!("Available datasets ({} total):", self.entries.len());
        for entry in &self.entries {
            print_dataset_with_profiles(entry);
        }
    }
}

/// Print a dataset entry with its profile names to stderr.
fn print_dataset_with_profiles(entry: &CatalogEntry) {
    let profiles: Vec<String> = entry.profile_names().into_iter().map(|s| s.to_string()).collect();
    if profiles.is_empty() {
        eprintln!("  - {} (no profiles)", entry.name);
    } else {
        eprintln!("  - {} (profiles: {})", entry.name, profiles.join(", "));
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
fn load_catalog_entries(location: &str, entries: &mut Vec<CatalogEntry>, required: bool) {
    if looks_like_catalog_file(location) {
        if !load_from_explicit_catalog_file(location, entries) && required {
            eprintln!("error: could not load catalog from {}", location);
        }
        return;
    }

    // Directory cascade.
    if try_load_canonical_catalog(location, entries) { return; }
    if try_load_knn_entries(location, entries) { return; }

    if required {
        eprintln!(
            "ERROR: no catalog file found at {} \
             (tried catalog.json, catalog.yaml, knn_entries.yaml)",
            location,
        );
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
fn load_from_explicit_catalog_file(location: &str, entries: &mut Vec<CatalogEntry>) -> bool {
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
            eprintln!("error: failed to parse catalog {}: {}", location, e);
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
                        eprintln!("warning: skipping catalog entry: {}", e);
                        continue;
                    }
                };
                match remap_entry(&as_json, &base_url) {
                    Ok(entry) => entries.push(entry),
                    Err(e) => eprintln!("warning: skipping catalog entry: {}", e),
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
                    eprintln!("error: failed to parse {}: {}", location, e);
                    true
                }
            }
        }
        _ => {
            eprintln!(
                "ERROR: catalog {} is neither a sequence nor a mapping",
                location,
            );
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
fn try_load_canonical_catalog(location: &str, entries: &mut Vec<CatalogEntry>) -> bool {
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
                eprintln!("error: failed to parse catalog {}: {}", catalog_url, e);
                return true; // file exists, but couldn't parse — don't fall through
            }
        },
    };

    let base_url = ensure_trailing_slash(location);
    for value in parsed {
        match remap_entry(&value, &base_url) {
            Ok(entry) => entries.push(entry),
            Err(e) => {
                eprintln!("warning: skipping catalog entry: {}", e);
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
fn try_load_knn_entries(location: &str, entries: &mut Vec<CatalogEntry>) -> bool {
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
            eprintln!("error: failed to parse {}: {}", url, e);
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
                attributes,
                profiles,
            },
            catalog_file: None,
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
fn fetch_http(url: &str) -> Result<String, String> {
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
    let response = rb.send()
        .map_err(|e| format!("HTTP request to {} failed: {}", url, e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("HTTP {} from {}", status.as_u16(), url));
    }

    response.text()
        .map_err(|e| format!("failed to read response from {}: {}", url, e))
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
                },
            );
        }
        CatalogEntry {
            name: name.to_string(),
            path: format!("{}/dataset.yaml", name),
            dataset_type: "dataset.yaml".to_string(),
            catalog_file: None,
            layout: CatalogLayout {
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
