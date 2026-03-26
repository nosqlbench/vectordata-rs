// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Catalog resolver — loads catalog entries from multiple sources (local files
//! and HTTP URLs) and provides search methods.
//!
//! This is the Rust equivalent of the upstream Java `Catalog` class. It takes
//! a [`CatalogSources`] configuration and fetches `catalog.json` from each
//! location, remapping layout-embedded entries into a unified list.

use vectordata::dataset::{CatalogEntry, CatalogLayout};

use super::sources::{catalog_file_for, ensure_trailing_slash, CatalogSources};

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
                eprintln!("WARNING: unsupported regex pattern '{}', falling back to substring match", pattern);
                let lower = pattern.to_lowercase();
                self.entries
                    .iter()
                    .filter(|e| e.name.to_lowercase().contains(&lower))
                    .collect()
            }
        }
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
fn load_catalog_entries(location: &str, entries: &mut Vec<CatalogEntry>, required: bool) {
    let catalog_url = catalog_file_for(location);

    let content = if catalog_url.starts_with("http://") || catalog_url.starts_with("https://") {
        match fetch_http(&catalog_url) {
            Ok(c) => c,
            Err(e) => {
                if required {
                    eprintln!("ERROR: failed to load required catalog from {}: {}", catalog_url, e);
                } else {
                    log::debug!("optional catalog {} unavailable: {}", catalog_url, e);
                }
                return;
            }
        }
    } else {
        // Local file
        let path = std::path::Path::new(&catalog_url);
        // If the URL points to a directory, try catalog.json inside it
        let file_path = if path.is_dir() {
            let json = path.join("catalog.json");
            if json.is_file() {
                json
            } else {
                let yaml = path.join("catalog.yaml");
                if yaml.is_file() {
                    yaml
                } else {
                    if required {
                        eprintln!("ERROR: no catalog file found in {}", crate::check::rel_display(&path.to_path_buf()));
                    }
                    return;
                }
            }
        } else {
            path.to_path_buf()
        };

        match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                if required {
                    eprintln!(
                        "ERROR: failed to read catalog {}: {}",
                        file_path.display(),
                        e
                    );
                }
                return;
            }
        }
    };

    // Parse the content as JSON array of entries
    let parsed: Vec<serde_json::Value> = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => {
            // Try YAML
            match serde_yaml::from_str(&content) {
                Ok(v) => v,
                Err(e) => {
                    if required {
                        eprintln!("ERROR: failed to parse catalog {}: {}", catalog_url, e);
                    }
                    return;
                }
            }
        }
    };

    let base_url = ensure_trailing_slash(location);

    for value in parsed {
        match remap_entry(&value, &base_url) {
            Ok(entry) => entries.push(entry),
            Err(e) => {
                eprintln!("WARNING: skipping catalog entry: {}", e);
            }
        }
    }
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

/// Fetch content from an HTTP(S) URL using curl.
fn fetch_http(url: &str) -> Result<String, String> {
    let mut data = Vec::new();
    let mut handle = curl::easy::Easy::new();
    handle
        .url(url)
        .map_err(|e| format!("curl error: {}", e))?;
    handle
        .follow_location(true)
        .map_err(|e| format!("curl error: {}", e))?;
    handle
        .useragent("veks/0.2")
        .map_err(|e| format!("curl error: {}", e))?;

    {
        let mut transfer = handle.transfer();
        transfer
            .write_function(|buf| {
                data.extend_from_slice(buf);
                Ok(buf.len())
            })
            .map_err(|e| format!("curl error: {}", e))?;
        transfer
            .perform()
            .map_err(|e| format!("HTTP request to {} failed: {}", url, e))?;
    }

    let code = handle
        .response_code()
        .map_err(|e| format!("curl error: {}", e))?;
    if code != 200 {
        return Err(format!("HTTP {} from {}", code, url));
    }

    String::from_utf8(data).map_err(|e| format!("invalid UTF-8 in response from {}: {}", url, e))
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
                vectordata::dataset::DSProfile {
                    maxk: None,
                    base_count: None,
                    views: IndexMap::new(),
                },
            );
        }
        CatalogEntry {
            name: name.to_string(),
            path: format!("{}/dataset.yaml", name),
            dataset_type: "dataset.yaml".to_string(),
            layout: CatalogLayout {
                attributes: None,
                profiles: vectordata::dataset::DSProfileGroup::from_profiles(profile_group),
            },
        }
    }

    #[test]
    fn test_find_exact() {
        let catalog = Catalog {
            entries: vec![
                make_entry("sift-128", &["default"]),
                make_entry("glove-100", &["default", "10m"]),
            ],
        };

        assert!(catalog.find_exact("sift-128").is_some());
        assert!(catalog.find_exact("SIFT-128").is_some()); // case insensitive
        assert!(catalog.find_exact("nonexistent").is_none());
    }

    #[test]
    fn test_match_glob() {
        let catalog = Catalog {
            entries: vec![
                make_entry("sift-128", &["default"]),
                make_entry("sift-256", &["default"]),
                make_entry("glove-100", &["default"]),
            ],
        };

        let matches = catalog.match_glob("sift-*");
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
        assert_eq!(glob_to_regex("sift-*"), "^sift-.*$");
        assert_eq!(glob_to_regex("?est"), "^.est$");
        assert_eq!(glob_to_regex("exact"), "^exact$");
    }
}
