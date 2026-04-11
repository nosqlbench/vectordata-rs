// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Catalog source discovery — locates catalog files from config directories,
//! explicit paths, and URLs.
//!
//! Mirrors the upstream Java `TestDataSources` class. The resolution chain:
//!
//! 1. A config directory (default `~/.config/vectordata/`) contains a
//!    `catalogs.yaml` file listing catalog locations.
//! 2. Each location is a directory path, file path, or HTTP(S) URL.
//! 3. Directory paths are resolved to `catalog.json` within that directory.
//! 4. All locations are collected into required and optional lists.

use std::path::Path;

/// Default configuration directory for catalog discovery.
pub const DEFAULT_CONFIG_DIR: &str = "~/.config/vectordata";

/// A collection of catalog source locations.
///
/// Catalog sources are URLs or file paths pointing to directories or files
/// that contain `catalog.json` or `catalog.yaml`. Sources are divided into
/// required (failure is fatal) and optional (failure is silently ignored).
#[derive(Debug, Clone, Default)]
pub struct CatalogSources {
    /// Required catalog locations — loading failures are fatal.
    required: Vec<String>,
    /// Optional catalog locations — loading failures are silently ignored.
    optional: Vec<String>,
}

impl CatalogSources {
    /// Create an empty source set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure from the default config directory (`~/.config/vectordata`).
    ///
    /// Loads `catalogs.yaml` from the directory. If the file does not exist,
    /// returns self unchanged (non-fatal for the default location).
    pub fn configure_default(mut self) -> Self {
        let config_dir = expand_tilde(DEFAULT_CONFIG_DIR);
        if let Ok(locations) = load_config(&config_dir) {
            self.optional.extend(locations);
        }
        self
    }

    /// Configure from a specific config directory.
    ///
    /// Loads `catalogs.yaml` from the directory. If the file does not exist,
    /// prints a warning but continues.
    pub fn configure(mut self, config_dir: &str) -> Self {
        let config_dir = expand_tilde(config_dir);
        match load_config(&config_dir) {
            Ok(locations) => self.required.extend(locations),
            Err(e) => eprintln!("WARNING: {}", e),
        }
        self
    }

    /// Add required catalog locations.
    ///
    /// Each path can be a directory (containing `catalog.json` or
    /// `catalogs.yaml`), a file path, or an HTTP(S) URL.
    pub fn add_catalogs(mut self, paths: &[String]) -> Self {
        for path in paths {
            let expanded = expand_tilde(path);
            self.required.extend(resolve_catalog_path(&expanded));
        }
        self
    }

    /// Add optional catalog locations (failures are silently ignored).
    pub fn add_optional_catalogs(mut self, paths: &[String]) -> Self {
        for path in paths {
            let expanded = expand_tilde(path);
            self.optional.extend(resolve_catalog_path(&expanded));
        }
        self
    }

    /// Returns required catalog locations.
    pub fn required(&self) -> &[String] {
        &self.required
    }

    /// Returns optional catalog locations.
    pub fn optional(&self) -> &[String] {
        &self.optional
    }

    /// Returns true if no catalog sources are configured.
    pub fn is_empty(&self) -> bool {
        self.required.is_empty() && self.optional.is_empty()
    }
}

/// Read raw catalog entries from `catalogs.yaml` without resolving them.
///
/// Returns the list of strings exactly as written in the file.
pub fn raw_catalog_entries(config_dir: &str) -> Vec<String> {
    let dir = std::path::Path::new(config_dir);
    let catalogs_yaml = dir.join("catalogs.yaml");
    if !catalogs_yaml.is_file() { return Vec::new(); }
    let content = match std::fs::read_to_string(&catalogs_yaml) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    serde_yaml::from_str(&content).unwrap_or_default()
}

/// Load `catalogs.yaml` from a config directory.
///
/// The file contains a YAML list of strings, each a catalog location.
fn load_config(config_dir: &str) -> Result<Vec<String>, String> {
    let dir = Path::new(config_dir);
    let catalogs_yaml = dir.join("catalogs.yaml");

    if !catalogs_yaml.is_file() {
        return Err(format!(
            "no catalogs.yaml found in {}",
            dir.display()
        ));
    }

    let content = std::fs::read_to_string(&catalogs_yaml)
        .map_err(|e| format!("failed to read {}: {}", catalogs_yaml.display(), e))?;

    let entries: Vec<String> = serde_yaml::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {}", catalogs_yaml.display(), e))?;

    // Resolve each entry relative to the config directory
    let mut locations = Vec::new();
    for entry in entries {
        let expanded = expand_tilde(&entry);
        locations.extend(resolve_catalog_path(&expanded));
    }

    Ok(locations)
}

/// Resolve a catalog path to a concrete URL or file path.
///
/// Smart detection:
/// - HTTP(S) URLs are kept as-is
/// - Directories containing `catalogs.yaml`: load recursively
/// - Directories containing `catalog.json`: use the directory URL
/// - Plain files: use directly
fn resolve_catalog_path(path: &str) -> Vec<String> {
    // HTTP URLs pass through
    if path.starts_with("http://") || path.starts_with("https://") {
        return vec![path.to_string()];
    }

    let p = Path::new(path);

    if p.is_dir() {
        // Directory with catalogs.yaml → load recursively
        let catalogs_yaml = p.join("catalogs.yaml");
        if catalogs_yaml.is_file()
            && let Ok(locations) = load_config(path) {
            return locations;
        }

        // Directory with catalog.json or catalog.yaml → use it
        let catalog_json = p.join("catalog.json");
        if catalog_json.is_file() {
            return vec![path.to_string()];
        }
        let catalog_yaml = p.join("catalog.yaml");
        if catalog_yaml.is_file() {
            return vec![path.to_string()];
        }

        // Directory without any catalog file
        eprintln!(
            "WARNING: directory {} has no catalogs.yaml or catalog.json",
            path
        );
        return vec![];
    }

    if p.is_file() {
        return vec![path.to_string()];
    }

    // Path doesn't exist — might be a remote URL without scheme
    vec![path.to_string()]
}

/// Expand `~` prefix to the user's home directory.
pub fn expand_tilde(path: &str) -> String {
    if (path.starts_with("~/") || path == "~")
        && let Some(home) = home_dir() {
        return format!("{}{}", home, &path[1..]);
    }
    path.to_string()
}

/// Get the user's home directory.
fn home_dir() -> Option<String> {
    std::env::var("HOME").ok()
}

/// Ensure a URL/path string ends with `/`.
pub fn ensure_trailing_slash(s: &str) -> String {
    if s.ends_with('/') {
        s.to_string()
    } else {
        format!("{}/", s)
    }
}

/// Resolve a catalog location to the actual catalog.json URL/path.
///
/// If the location already ends with `catalog.json`, returns it as-is.
/// Otherwise appends `catalog.json`.
pub fn catalog_file_for(location: &str) -> String {
    if location.ends_with("/catalog.json") || location.ends_with("/catalog.yaml") {
        return location.to_string();
    }

    // If it's a path to an existing file that looks like a catalog, use it
    let p = Path::new(location);
    if p.is_file()
        && let Some(name) = p.file_name().and_then(|n| n.to_str())
        && name.starts_with("catalog") && (name.ends_with(".json") || name.ends_with(".yaml")) {
        return location.to_string();
    }

    let base = ensure_trailing_slash(location);
    format!("{}catalog.json", base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde() {
        let expanded = expand_tilde("~/foo/bar");
        assert!(!expanded.starts_with('~'));
        assert!(expanded.ends_with("/foo/bar"));
    }

    #[test]
    fn test_expand_tilde_no_tilde() {
        assert_eq!(expand_tilde("/absolute/path"), "/absolute/path");
        assert_eq!(expand_tilde("relative/path"), "relative/path");
    }

    #[test]
    fn test_ensure_trailing_slash() {
        assert_eq!(ensure_trailing_slash("http://example.com"), "http://example.com/");
        assert_eq!(ensure_trailing_slash("http://example.com/"), "http://example.com/");
        assert_eq!(ensure_trailing_slash("/path/to/dir"), "/path/to/dir/");
    }

    #[test]
    fn test_catalog_file_for() {
        assert_eq!(
            catalog_file_for("http://example.com/data"),
            "http://example.com/data/catalog.json"
        );
        assert_eq!(
            catalog_file_for("http://example.com/data/"),
            "http://example.com/data/catalog.json"
        );
        assert_eq!(
            catalog_file_for("http://example.com/data/catalog.json"),
            "http://example.com/data/catalog.json"
        );
    }

    #[test]
    fn test_resolve_http_url() {
        let locations = resolve_catalog_path("https://example.com/catalogs");
        assert_eq!(locations, vec!["https://example.com/catalogs"]);
    }

    #[test]
    fn test_sources_empty() {
        let sources = CatalogSources::new();
        assert!(sources.is_empty());
    }

    #[test]
    fn test_load_config_from_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = "- /some/catalog\n- https://example.com/data\n";
        std::fs::write(tmp.path().join("catalogs.yaml"), yaml).unwrap();

        let locations = load_config(tmp.path().to_str().unwrap()).unwrap();
        // /some/catalog doesn't exist so it still gets returned as-is
        assert!(locations.iter().any(|l| l == "/some/catalog"));
        assert!(locations.iter().any(|l| l == "https://example.com/data"));
    }

    #[test]
    fn test_load_config_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let result = load_config(tmp.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_dir_with_catalog_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("catalog.json"), "[]").unwrap();

        let locations = resolve_catalog_path(tmp.path().to_str().unwrap());
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0], tmp.path().to_str().unwrap());
    }

    #[test]
    fn test_configure_add_catalogs() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("catalog.json"), "[]").unwrap();

        let sources = CatalogSources::new()
            .add_catalogs(&[tmp.path().to_string_lossy().to_string()]);
        assert_eq!(sources.required().len(), 1);
    }
}
