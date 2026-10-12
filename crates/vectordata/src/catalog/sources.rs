// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Catalog source discovery — locates catalog files from config directories,
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
    /// Required catalog sources — loading failures are fatal.
    required: Vec<NamedCatalogSource>,
    /// Optional catalog sources — loading failures are silently ignored.
    optional: Vec<NamedCatalogSource>,
    /// Problems met while reading configuration and locations.
    diagnostics: Vec<super::CatalogDiagnostic>,
}

impl CatalogSources {
    /// Create an empty source set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure from the default config directory ([`config_dir`] —
    /// `$VECTORDATA_HOME`, else `~/.config/vectordata`).
    ///
    /// Composes the env-aware [`config_dir`] resolver with the dir-taking
    /// [`Self::configure_optional`] loader; both are tested independently, so
    /// this is correct by construction and needs no env-mutating test.
    pub fn configure_default(self) -> Self {
        self.configure_optional(&config_dir())
    }

    /// Load `catalogs.yaml` from an explicit config directory as *optional*
    /// sources. Having no `catalogs.yaml` is the ordinary case and says
    /// nothing; one that cannot be read or parsed records a warning in
    /// [`diagnostics`](Self::diagnostics). The env-free seam behind
    /// [`Self::configure_default`].
    pub fn configure_optional(mut self, config_dir: &str) -> Self {
        let config_dir = expand_tilde(config_dir);
        if !Path::new(&config_dir).join("catalogs.yaml").is_file() {
            return self;
        }
        match load_config(&config_dir, &mut self.diagnostics) {
            Ok(locations) => self.optional.extend(locations),
            Err(e) => self.diagnostics.push(super::CatalogDiagnostic::warning(e)),
        }
        self
    }

    /// Configure from a specific config directory.
    ///
    /// Loads `catalogs.yaml` from the directory. If the file cannot be
    /// read, records a warning in [`diagnostics`](Self::diagnostics) and
    /// continues.
    pub fn configure(mut self, config_dir: &str) -> Self {
        let config_dir = expand_tilde(config_dir);
        match load_config(&config_dir, &mut self.diagnostics) {
            Ok(locations) => self.required.extend(locations),
            Err(e) => self.diagnostics.push(super::CatalogDiagnostic::warning(e)),
        }
        self
    }

    /// Problems met while reading configuration and resolving
    /// locations, such as a directory with no catalog file in it.
    /// [`Catalog::of`](super::Catalog::of) carries these forward into
    /// its own [`diagnostics`](super::Catalog::diagnostics).
    pub fn diagnostics(&self) -> &[super::CatalogDiagnostic] {
        &self.diagnostics
    }

    /// Add required catalog locations.
    ///
    /// Each path can be a directory (containing `catalog.json` or
    /// `catalogs.yaml`), a file path, or an HTTP(S) URL. Sources from
    /// a directory's `catalogs.yaml` keep the names that file gives
    /// them; any other path gets the next free stringified index.
    pub fn add_catalogs(mut self, paths: &[String]) -> Self {
        for path in paths {
            let expanded = expand_tilde(path);
            for (named, location) in resolve_catalog_path(&expanded, &mut self.diagnostics) {
                let name = named.unwrap_or_else(|| self.synthesize_name());
                self.required.push(NamedCatalogSource { name, location });
            }
        }
        self
    }

    /// Next made-up name for a source whose name can't be determined
    /// (ad-hoc CLI locations): the next free stringified index,
    /// matching the promotion rule for legacy list files.
    fn synthesize_name(&self) -> String {
        let taken: std::collections::HashSet<&str> = self.required.iter()
            .chain(self.optional.iter())
            .map(|s| s.name.as_str())
            .collect();
        let mut n = self.required.len() + self.optional.len();
        loop {
            let candidate = n.to_string();
            if !taken.contains(candidate.as_str()) {
                return candidate;
            }
            n += 1;
        }
    }

    /// Add optional catalog locations (failures are silently ignored).
    pub fn add_optional_catalogs(mut self, paths: &[String]) -> Self {
        for path in paths {
            let expanded = expand_tilde(path);
            for (named, location) in resolve_catalog_path(&expanded, &mut self.diagnostics) {
                let name = named.unwrap_or_else(|| self.synthesize_name());
                self.optional.push(NamedCatalogSource { name, location });
            }
        }
        self
    }

    /// Returns required catalog locations.
    pub fn required(&self) -> &[NamedCatalogSource] {
        &self.required
    }

    /// Returns optional catalog sources.
    pub fn optional(&self) -> &[NamedCatalogSource] {
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
    named_catalog_entries(config_dir).into_iter().map(|n| n.location).collect()
}

/// Pure core of the numbered catalog shortcut (`--at 2` → second
/// configured catalog). `None` when `value` isn't a positive integer
/// (it's a literal URL/path); `Some(Ok(url))` when the 1-based index
/// resolves against `configured`; `Some(Err(n))` when the index is
/// out of range.
pub fn lookup_catalog_index(value: &str, configured: &[String]) -> Option<Result<String, usize>> {
    let n: usize = value.parse().ok().filter(|n| *n >= 1)?;
    Some(configured.get(n - 1).cloned().ok_or(n))
}

/// Resolve a `--at`/`--catalog` value: a positive integer is a 1-based
/// index into the configured catalog list (see `config catalog list`);
/// anything else passes through as a literal URL/path. An out-of-range
/// index is an error naming it.
///
/// Idempotent: resolving an already-resolved URL is a no-op, so
/// callers layered above (e.g. binary dispatch arms) may resolve
/// early without harm.
pub fn resolve_catalog_value(value: &str) -> crate::Result<String> {
    let configured = raw_catalog_entries(&config_dir());
    match lookup_catalog_index(value, &configured) {
        None => Ok(value.to_string()),
        Some(Ok(url)) => Ok(url),
        Some(Err(n)) => Err(crate::Error::Other(format!(
            "catalog #{n} not found; list configured catalogs with \
             the `config catalog list` subcommand."
        ))),
    }
}

/// [`resolve_catalog_value`] over a slice — the seam the shared
/// `datasets` run paths use so numbered shortcuts work identically
/// from every binary.
pub fn resolve_catalog_values(values: &[String]) -> crate::Result<Vec<String>> {
    values.iter().map(|v| resolve_catalog_value(v)).collect()
}

/// A configured catalog source: symbolic name + location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedCatalogSource {
    /// Symbolic name — the map key in the map form of
    /// `catalogs.yaml`, or a stringified 0-based index for the legacy
    /// list form (and any other entry whose name can't be
    /// determined).
    pub name: String,
    /// Catalog URL or path. [`named_catalog_entries`] gives it
    /// verbatim from the file; [`CatalogSources`] holds it resolved,
    /// with a relative path joined onto the directory of the
    /// `catalogs.yaml` that lists it.
    pub location: String,
}

/// Read `catalogs.yaml` from a config directory as named sources.
/// Missing/unreadable/malformed files yield an empty list.
pub fn named_catalog_entries(config_dir: &str) -> Vec<NamedCatalogSource> {
    let catalogs_yaml = std::path::Path::new(config_dir).join("catalogs.yaml");
    if !catalogs_yaml.is_file() { return Vec::new(); }
    let content = match std::fs::read_to_string(&catalogs_yaml) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    parse_named_catalogs(&content).unwrap_or_default()
}

/// Parse `catalogs.yaml` content into named sources. Two supported
/// shapes:
///
/// - **Map of names to locations** — the default going forward:
///   ```yaml
///   protected: https://host/path/protected-catalog.yaml
///   local-lab: /data/catalogs/lab
///   ```
/// - **Legacy list of locations** — still accepted; entries promote
///   to stringified 0-based indexes as their names (`0`, `1`, …) in
///   file order:
///   ```yaml
///   - https://host/path/protected-catalog.yaml   # name: "0"
///   ```
///
/// Map keys win as names; anything name-less gets the stringified
/// index so every catalog is addressable.
pub fn parse_named_catalogs(content: &str) -> Result<Vec<NamedCatalogSource>, String> {
    let value: serde_yaml::Value = serde_yaml::from_str(content)
        .map_err(|e| format!("parse catalogs.yaml: {e}"))?;
    let mut out = Vec::new();
    match value {
        serde_yaml::Value::Sequence(seq) => {
            for (i, item) in seq.into_iter().enumerate() {
                let Some(loc) = item.as_str() else { continue };
                out.push(NamedCatalogSource {
                    name: i.to_string(),
                    location: loc.to_string(),
                });
            }
        }
        serde_yaml::Value::Mapping(map) => {
            for (i, (k, v)) in map.into_iter().enumerate() {
                let Some(loc) = v.as_str() else { continue };
                let name = k.as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| i.to_string());
                out.push(NamedCatalogSource { name, location: loc.to_string() });
            }
        }
        serde_yaml::Value::Null => {}
        other => {
            return Err(format!(
                "catalogs.yaml must be a list of locations or a map of name: location, got {other:?}"));
        }
    }
    Ok(out)
}

/// Load `catalogs.yaml` from a config directory.
///
/// The file is a map of name → location, or a legacy list of
/// locations. A relative local location is resolved against
/// `config_dir`, not the process's working directory, so a
/// `catalogs.yaml` means the same thing wherever it is read from.
fn load_config(
    config_dir: &str,
    diags: &mut Vec<super::CatalogDiagnostic>,
) -> Result<Vec<NamedCatalogSource>, String> {
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

    let entries = parse_named_catalogs(&content)
        .map_err(|e| format!("failed to parse {}: {e}", catalogs_yaml.display()))?;

    // Resolve each entry relative to the config directory. An entry
    // naming a directory with its own catalogs.yaml yields that file's
    // sources under that file's names.
    let mut locations = Vec::new();
    for entry in entries {
        let expanded = relative_to(dir, &expand_tilde(&entry.location));
        for (named, location) in resolve_catalog_path(&expanded, diags) {
            let name = named.unwrap_or_else(|| entry.name.clone());
            locations.push(NamedCatalogSource { name, location });
        }
    }

    Ok(locations)
}

/// `location` as written in a `catalogs.yaml` in `dir`: a relative
/// local path is joined onto `dir`; URLs and absolute paths are kept.
fn relative_to(dir: &Path, location: &str) -> String {
    if crate::location::has_scheme(location) || Path::new(location).is_absolute() {
        return location.to_string();
    }
    dir.join(location).to_string_lossy().into_owned()
}

/// Resolve a catalog path to concrete URLs or file paths, each with
/// the name a nested `catalogs.yaml` gave it (`None` when the path
/// names a catalog directly and the caller must name it).
///
/// Smart detection:
/// - Remote URLs (`http(s)://`, `s3://`) are kept as-is — the
///   resolver routes them through [`crate::transport`] which handles
///   the S3→virtual-hosted-HTTPS rewrite at fetch time.
/// - Directories containing `catalogs.yaml`: load recursively
/// - Directories containing `catalog.json`: use the directory URL
/// - Plain files: use directly
fn resolve_catalog_path(
    path: &str,
    diags: &mut Vec<super::CatalogDiagnostic>,
) -> Vec<(Option<String>, String)> {
    // Any URL the shared transport speaks (currently `http://`,
    // `https://`, `s3://`) passes through verbatim — the catalog
    // fetcher normalises on the way to the wire.
    if crate::transport::is_remote_url(path) {
        return vec![(None, path.to_string())];
    }

    let p = Path::new(path);

    if p.is_dir() {
        // Directory with catalogs.yaml → load recursively
        let catalogs_yaml = p.join("catalogs.yaml");
        if catalogs_yaml.is_file()
            && let Ok(sources) = load_config(path, diags) {
            return sources.into_iter().map(|s| (Some(s.name), s.location)).collect();
        }

        // Directory with catalog.json or catalog.yaml → use it
        let catalog_json = p.join("catalog.json");
        if catalog_json.is_file() {
            return vec![(None, path.to_string())];
        }
        let catalog_yaml = p.join("catalog.yaml");
        if catalog_yaml.is_file() {
            return vec![(None, path.to_string())];
        }

        // Fallback: `knn_entries.yaml` (legacy simplified format).
        // Treated as a complete catalog by the resolver — see
        // `super::knn_entries`.
        let knn_entries = p.join("knn_entries.yaml");
        if knn_entries.is_file() {
            return vec![(None, path.to_string())];
        }

        // Directory without any catalog file
        diags.push(super::CatalogDiagnostic::warning(format!(
            "directory {path} has no catalogs.yaml / catalog.json / catalog.yaml / knn_entries.yaml"
        )));
        return vec![];
    }

    if p.is_file() {
        return vec![(None, path.to_string())];
    }

    // Path doesn't exist — might be a remote URL without scheme
    vec![(None, path.to_string())]
}

/// Expand `~` prefix to the user's home directory.
pub fn expand_tilde(path: &str) -> String {
    if (path.starts_with("~/") || path == "~")
        && let Some(home) = home_dir() {
        return format!("{}{}", home, &path[1..]);
    }
    path.to_string()
}

/// The client configuration directory that holds `catalogs.yaml`.
///
/// `$VECTORDATA_HOME` takes precedence over `~/.config/vectordata`,
/// mirroring [`crate::settings::settings_path`] and the credentials store.
/// Routing every catalog read and write through here means that one env
/// var isolates *all* client config — catalogs, settings, and credentials —
/// under a single root, so tests and tutorials never touch the real config.
///
/// This is the thin env-reading wrapper; the resolution logic lives in the
/// pure `config_dir_from` so it can be tested without mutating the
/// process-wide environment.
pub fn config_dir() -> String {
    config_dir_from(std::env::var_os("VECTORDATA_HOME"))
}

/// Pure resolver behind [`config_dir`]: the given `$VECTORDATA_HOME` value if
/// set, else the tilde-expanded `~/.config/vectordata` default.
fn config_dir_from(vectordata_home: Option<std::ffi::OsString>) -> String {
    match vectordata_home {
        Some(root) => root.to_string_lossy().into_owned(),
        None => expand_tilde(DEFAULT_CONFIG_DIR),
    }
}

/// Get the user's home directory.
fn home_dir() -> Option<String> {
    std::env::var("HOME").ok()
}

/// Resolve a catalog location to the actual catalog file URL/path.
///
/// If `location` already names a YAML or JSON file (any filename ending
/// in `.yaml`, `.yml`, or `.json`), it is returned verbatim. This is
/// what lets back-ends point at non-standard filenames like
/// `protected-catalog.yaml` — the file *is* the catalog regardless of
/// its name, and the resolver's content-shape dispatch decides how to
/// parse it.
///
/// Otherwise — when `location` names a directory — the canonical
/// `catalog.json` is appended. The directory cascade in
/// `resolver::load_catalog_entries` walks the documented filename set
/// (`catalog.json` → `catalog.yaml` → `knn_entries.yaml`).
pub fn catalog_file_for(location: &str) -> String {
    if looks_like_catalog_file(location) {
        return location.to_string();
    }

    // Local file paths may exist with non-standard names; honour the
    // filesystem if it's actually there.
    let p = Path::new(location);
    if p.is_file() {
        return location.to_string();
    }

    format!("{}catalog.json", crate::location::dir_form(location))
}

/// Returns true when `location` ends in `.yaml`, `.yml`, or `.json`
/// — the strong signal that it names a catalog file directly rather
/// than a directory to probe. Matches `DataSetLoaderSimpleMFD`'s
/// `*.{yaml,yml}` discovery rule (jvector parity).
pub fn looks_like_catalog_file(location: &str) -> bool {
    let lower = location.to_ascii_lowercase();
    let trimmed = lower.trim_end_matches('/');
    trimmed.ends_with(".yaml") || trimmed.ends_with(".yml") || trimmed.ends_with(".json")
}

#[cfg(test)]
mod tests {
    /// catalogs.yaml accepts both shapes: a map of name → location
    /// (the default going forward) and the legacy list, whose entries
    /// promote to stringified 0-based indexes as names.
    #[test]
    fn named_catalogs_map_and_list_forms() {
        let map_form = "\n# main\nprotected: https://h/p/protected-catalog.yaml\nlab: /data/lab\n";
        let named = super::parse_named_catalogs(map_form).unwrap();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].name, "protected");
        assert_eq!(named[0].location, "https://h/p/protected-catalog.yaml");
        assert_eq!(named[1].name, "lab");

        let list_form = "- https://h/a/\n- https://h/b/\n";
        let named = super::parse_named_catalogs(list_form).unwrap();
        assert_eq!(named[0].name, "0");
        assert_eq!(named[1].name, "1");

        assert!(super::parse_named_catalogs("just a string").is_err());
        assert!(super::parse_named_catalogs("").unwrap().is_empty());
    }

    use super::*;

    #[test]
    fn catalog_index_lookup_rules() {
        let configured = vec!["https://a/".to_string(), "https://b/".to_string()];
        assert_eq!(lookup_catalog_index("https://a/", &configured), None);
        assert_eq!(lookup_catalog_index("0", &configured), None);
        assert_eq!(lookup_catalog_index("1", &configured), Some(Ok("https://a/".to_string())));
        assert_eq!(lookup_catalog_index("3", &configured), Some(Err(3)));
    }

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
    fn config_dir_prefers_vectordata_home() {
        // The regression guard: a set $VECTORDATA_HOME wins over ~/.config,
        // so catalog reads/writes stay inside an isolated client home.
        let resolved = config_dir_from(Some(std::ffi::OsString::from("/tmp/iso-home")));
        assert_eq!(resolved, "/tmp/iso-home");
    }

    #[test]
    fn config_dir_falls_back_to_default_when_home_unset() {
        let resolved = config_dir_from(None);
        assert!(resolved.ends_with(".config/vectordata"), "got {resolved}");
        assert!(!resolved.starts_with('~'), "must be tilde-expanded, got {resolved}");
    }

    #[test]
    fn configure_optional_loads_from_the_given_dir_only() {
        // An explicit dir with no catalogs.yaml yields nothing — proving the
        // loader reads the dir it is handed and never a real config elsewhere.
        let empty = tempfile::tempdir().unwrap();
        let s = CatalogSources::new().configure_optional(&empty.path().to_string_lossy());
        assert!(s.optional().is_empty() && s.required().is_empty());

        // With a catalogs.yaml in that dir, its entries are loaded.
        std::fs::write(
            empty.path().join("catalogs.yaml"),
            "- https://example.invalid/catalog/\n",
        )
        .unwrap();
        let s = CatalogSources::new().configure_optional(&empty.path().to_string_lossy());
        assert!(
            s.optional().iter().any(|e| e.location.contains("example.invalid")),
            "expected the dir's catalogs.yaml entry, got {:?}",
            s.optional()
        );
    }

    /// **Configuration problems are diagnostics on the sources**, never
    /// stderr: a required config directory without `catalogs.yaml`, a
    /// location that is a directory with no catalog file in it (named
    /// directly or listed in `catalogs.yaml`), and a malformed
    /// `catalogs.yaml` — which the default, optional configuration
    /// reports too. Only an absent optional `catalogs.yaml` says
    /// nothing: having none is the ordinary case.
    #[test]
    fn configuration_problems_are_diagnostics() {
        use super::super::Severity;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        let s = CatalogSources::new().configure(&path);
        assert_eq!(s.diagnostics().len(), 1, "{:?}", s.diagnostics());
        assert_eq!(s.diagnostics()[0].severity, Severity::Warning);
        assert!(s.diagnostics()[0].message.contains("no catalogs.yaml found"), "{}", s.diagnostics()[0]);
        assert!(CatalogSources::new().configure_optional(&path).diagnostics().is_empty());

        let bare = dir.path().join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        let s = CatalogSources::new().add_catalogs(&[bare.to_string_lossy().to_string()]);
        assert!(s.required().is_empty());
        assert_eq!(s.diagnostics().len(), 1, "{:?}", s.diagnostics());
        assert!(s.diagnostics()[0].message.contains("has no catalogs.yaml / catalog.json"), "{}", s.diagnostics()[0]);

        std::fs::write(dir.path().join("catalogs.yaml"), format!("lab: {}\n", bare.display())).unwrap();
        let s = CatalogSources::new().configure_optional(&path);
        assert!(s.optional().is_empty());
        assert_eq!(s.diagnostics().len(), 1, "a listed bare directory: {:?}", s.diagnostics());

        std::fs::write(dir.path().join("catalogs.yaml"), "lab: [unclosed\n").unwrap();
        for s in [CatalogSources::new().configure(&path), CatalogSources::new().configure_optional(&path)] {
            assert_eq!(s.diagnostics().len(), 1, "a malformed catalogs.yaml: {:?}", s.diagnostics());
            assert!(s.diagnostics()[0].message.contains("failed to parse"), "{}", s.diagnostics()[0]);
            assert!(s.required().is_empty() && s.optional().is_empty());
        }
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
        let locations = resolve_catalog_path("https://example.com/catalogs", &mut Vec::new());
        assert_eq!(locations, vec![(None, "https://example.com/catalogs".to_string())]);
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

        let sources = load_config(tmp.path().to_str().unwrap(), &mut Vec::new()).unwrap();
        // /some/catalog doesn't exist so it still gets returned as-is
        assert!(sources.iter().any(|s| s.location == "/some/catalog"));
        assert!(sources.iter().any(|s| s.location == "https://example.com/data"));
        // Legacy list entries promote to stringified 0-based indexes.
        assert!(sources.iter().any(|s| s.name == "0"));
        assert!(sources.iter().any(|s| s.name == "1"));
    }

    #[test]
    fn test_load_config_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let result = load_config(tmp.path().to_str().unwrap(), &mut Vec::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_dir_with_catalog_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("catalog.json"), "[]").unwrap();

        let locations = resolve_catalog_path(tmp.path().to_str().unwrap(), &mut Vec::new());
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0], (None, tmp.path().to_str().unwrap().to_string()));
    }

    /// A relative entry in catalogs.yaml names a file beside that
    /// catalogs.yaml, whatever the process's working directory is.
    #[test]
    fn a_relative_entry_resolves_against_its_catalogs_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("catalogs.yaml"), "shipped: shipped-catalog.yaml\n").unwrap();
        std::fs::write(tmp.path().join("shipped-catalog.yaml"), "[]").unwrap();

        let sources = CatalogSources::new().configure(tmp.path().to_str().unwrap());
        let expected = tmp.path().join("shipped-catalog.yaml");
        assert_eq!(sources.required().len(), 1);
        assert_eq!(sources.required()[0].name, "shipped");
        assert_eq!(sources.required()[0].location, expected.to_string_lossy());
    }

    /// A directory added ad hoc keeps the names its own catalogs.yaml
    /// gives, through any depth of nesting, rather than being numbered.
    #[test]
    fn a_directory_added_by_path_keeps_its_catalogs_yaml_names() {
        let tmp = tempfile::tempdir().unwrap();
        let inner = tmp.path().join("inner");
        std::fs::create_dir(&inner).unwrap();
        std::fs::write(inner.join("catalogs.yaml"), "deep: deep-catalog.yaml\n").unwrap();
        std::fs::write(inner.join("deep-catalog.yaml"), "[]").unwrap();
        std::fs::write(
            tmp.path().join("catalogs.yaml"),
            "near: https://example.com/near/\nouter: inner\n",
        ).unwrap();

        let sources = CatalogSources::new()
            .add_catalogs(&[tmp.path().to_string_lossy().to_string()]);
        let got: Vec<(&str, &str)> = sources.required().iter()
            .map(|s| (s.name.as_str(), s.location.as_str()))
            .collect();
        let deep = inner.join("deep-catalog.yaml");
        assert_eq!(got, vec![
            ("near", "https://example.com/near/"),
            ("deep", deep.to_str().unwrap()),
        ]);
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
