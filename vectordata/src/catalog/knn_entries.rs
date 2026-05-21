// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Fallback parser for the legacy `knn_entries.yaml` catalog format.
//!
//! A `knn_entries.yaml` file is a flat map keyed by quoted strings
//! of the form `"<dataset>:<profile>"`, where each value is a map
//! of facet aliases to relative paths. An optional `_defaults`
//! entry may carry a `base_url` for resolving those relative paths.
//!
//! Example:
//!
//! ```yaml
//! _defaults:
//!   base_url: s3://my-bucket/datasets
//!
//! "my-dataset:default":
//!   base: profiles/base/base_vectors.fvecs
//!   query: profiles/base/query_vectors.fvecs
//!   gt: profiles/default/neighbor_indices.ivecs
//!
//! "my-dataset:100k":
//!   base: profiles/base/base_vectors.fvecs
//!   query: profiles/base/query_vectors.fvecs
//!   gt: profiles/100k/neighbor_indices.ivecs
//! ```
//!
//! When `vectordata`'s catalog resolver is pointed at a directory
//! that contains `knn_entries.yaml` (instead of `catalog.json` /
//! `catalog.yaml`), [`parse_knn_entries_yaml`] turns the contents
//! into the same [`CatalogEntry`] shape the canonical loader
//! produces.

use indexmap::IndexMap;

use crate::dataset::{
    profile::{DSProfile, DSProfileGroup, DSView},
    source::DSSource,
    CatalogEntry, CatalogLayout,
};

/// Facet aliases used in `knn_entries.yaml`, mapped to the
/// canonical facet keys the rest of the codebase expects.
///
/// Anything not in this table is passed through verbatim — so a
/// legacy file that uses `base_vectors` directly (or some
/// custom facet name) round-trips without translation.
fn canonical_facet_name(alias: &str) -> &str {
    match alias {
        "base" => "base_vectors",
        "query" => "query_vectors",
        "gt" => "neighbor_indices",
        // Carry-through for legacy/custom names.
        other => other,
    }
}

/// Parse a `knn_entries.yaml` document into one [`CatalogEntry`]
/// per distinct dataset name. Profiles are aggregated under each
/// entry; facet paths are resolved relative to the catalog
/// location (the directory containing `knn_entries.yaml`) — the
/// optional `_defaults.base_url` overrides that when present.
///
/// `catalog_location` should be the directory URL or path that
/// `knn_entries.yaml` lives in (with or without trailing `/`).
pub fn parse_knn_entries_yaml(
    content: &str,
    catalog_location: &str,
) -> Result<Vec<CatalogEntry>, String> {
    // Top-level shape: a map of String -> serde_yaml::Value.
    let top: serde_yaml::Mapping = serde_yaml::from_str(content)
        .map_err(|e| format!("parse knn_entries.yaml: {e}"))?;

    // `_defaults` carries per-catalog defaults that are folded into
    // each entry. The two fields we honour mirror jvector
    // `DataSetLoaderSimpleMFD`: `base_url` (where to fetch from)
    // and `cache_dir` (where the local copies live, if any).
    let (default_base_url, default_cache_dir) =
        extract_defaults(&top, catalog_location);

    // Accumulate by dataset name → (profile name → resolved views).
    // We resolve every facet path eagerly here so the synthesized
    // layout can carry absolute paths for facets whose local copy
    // exists, and absolute URLs for facets that still need to be
    // fetched. Downstream the view layer's local-source short-
    // circuit handles them uniformly.
    let mut grouped: IndexMap<String, IndexMap<String, IndexMap<String, String>>> =
        IndexMap::new();

    for (key_v, val_v) in top.iter() {
        let key = match key_v.as_str() {
            Some(s) => s,
            None => continue,
        };
        // Skip all `_`-prefixed keys (`_defaults`, `_meta`,
        // `_include`, etc.). They are control-plane metadata, not
        // dataset entries.
        if key.starts_with('_') { continue; }
        let (dataset_name, profile_name) = match parse_dataset_profile_key(key) {
            Some(t) => t,
            None => {
                log::warn!("knn_entries.yaml: skipping malformed key '{key}'");
                continue;
            }
        };

        let raw = match val_v.as_mapping() {
            Some(m) => m,
            None => {
                log::warn!("knn_entries.yaml: entry '{key}' is not a map; skipping");
                continue;
            }
        };

        // Per-entry overrides for cache_dir / base_url. Each
        // override replaces (not extends) the corresponding
        // default. Env vars are expanded eagerly so the resolved
        // strings can be compared against the filesystem.
        let entry_base_url = raw
            .get(serde_yaml::Value::String("base_url".to_string()))
            .and_then(|v| v.as_str())
            .map(expand_env_vars)
            .map(|s| ensure_trailing_slash(&s))
            .unwrap_or_else(|| default_base_url.clone());
        let entry_cache_dir = raw
            .get(serde_yaml::Value::String("cache_dir".to_string()))
            .and_then(|v| v.as_str())
            .map(expand_env_vars)
            .or_else(|| default_cache_dir.clone());

        let mut view_map: IndexMap<String, String> = IndexMap::new();
        for (fk, fv) in raw {
            let alias = match fk.as_str() {
                Some(s) => s,
                None => continue,
            };
            // Skip the configuration fields we already consumed —
            // they aren't facets.
            if alias == "base_url" || alias == "cache_dir" { continue; }
            let rel = match fv.as_str() {
                Some(s) => s,
                None => continue,
            };
            let canonical = canonical_facet_name(alias).to_string();
            let resolved = resolve_facet_path(
                rel,
                entry_cache_dir.as_deref(),
                &entry_base_url,
            );
            view_map.insert(canonical, resolved);
        }

        grouped
            .entry(dataset_name.to_string())
            .or_default()
            .insert(profile_name.to_string(), view_map);
    }

    // Materialize CatalogEntries.
    let mut out: Vec<CatalogEntry> = Vec::with_capacity(grouped.len());
    for (name, profiles) in grouped {
        let mut profile_group = DSProfileGroup::default();
        for (profile_name, views) in profiles {
            let mut profile = DSProfile {
                maxk: None,
                base_count: None,
                partition: false,
                views: IndexMap::new(),
            };
            for (facet, path) in views {
                let view = DSView {
                    source: DSSource {
                        path,
                        namespace: None,
                        window: Default::default(),
                    },
                    window: None,
                };
                profile.views.insert(facet, view);
            }
            profile_group.profiles.insert(profile_name, profile);
        }
        out.push(CatalogEntry {
            name: name.clone(),
            // For knn_entries-shaped catalogs there's no per-dataset
            // dataset.yaml — the synthesized layout *is* the
            // dataset description. Point the path at the catalog
            // location so downstream consumers can resolve a URL
            // for the dataset's "home directory" when they need
            // one (e.g. for precache status display).
            path: default_base_url.trim_end_matches('/').to_string(),
            dataset_type: "knn_entries.yaml".to_string(),
            layout: CatalogLayout {
                attributes: None,
                profiles: profile_group,
            },
        });
    }
    Ok(out)
}

/// Extract `(default_base_url, default_cache_dir)` from a parsed
/// `_defaults` mapping. `base_url` defaults to the catalog
/// location (with a trailing `/` so `join_url` composes cleanly).
/// `cache_dir` has no default — its absence simply means "no
/// local cache layer; always use the remote URL".
fn extract_defaults(top: &serde_yaml::Mapping, catalog_location: &str) -> (String, Option<String>) {
    let mut base_url = ensure_trailing_slash(catalog_location);
    let mut cache_dir: Option<String> = None;
    if let Some(defaults) = top
        .get(serde_yaml::Value::String("_defaults".to_string()))
        .and_then(|v| v.as_mapping())
    {
        if let Some(bu) = defaults
            .get(serde_yaml::Value::String("base_url".to_string()))
            .and_then(|v| v.as_str())
        {
            base_url = ensure_trailing_slash(&expand_env_vars(bu));
        }
        if let Some(cd) = defaults
            .get(serde_yaml::Value::String("cache_dir".to_string()))
            .and_then(|v| v.as_str())
        {
            cache_dir = Some(expand_env_vars(cd));
        }
    }
    (base_url, cache_dir)
}

/// Resolve a single facet's relative path to its effective
/// source. Matches jvector `DataSetLoaderSimpleMFD::loadDataSet`:
/// if `<cache_dir>/<rel>` exists on disk, that local copy is
/// used; otherwise the remote URL composed from `<base_url><rel>`
/// is used. An absolute URL or absolute path in `rel` short-
/// circuits both branches (the catalog entry is naming a specific
/// location, not a layout-relative file).
fn resolve_facet_path(rel: &str, cache_dir: Option<&str>, base_url: &str) -> String {
    // Absolute reference in the entry overrides everything.
    if rel.contains("://") || rel.starts_with('/') {
        return rel.to_string();
    }
    // Prefer a local cached copy if one exists. The check is
    // best-effort: if cache_dir is itself relative, it is
    // interpreted relative to the process CWD (matching
    // `Paths.get` in jvector).
    if let Some(cd) = cache_dir {
        let candidate = std::path::PathBuf::from(cd).join(rel);
        if candidate.exists() {
            // Emit the absolute path so the view-layer
            // local-source short-circuit triggers without
            // re-checking the filesystem.
            return std::fs::canonicalize(&candidate)
                .ok()
                .and_then(|p| p.to_str().map(|s| s.to_string()))
                .unwrap_or_else(|| candidate.to_string_lossy().to_string());
        }
    }
    join_url(base_url, rel)
}

/// Expand `${VAR}` and `${VAR:-default}` references in a string
/// against the process environment. Mirrors jvector
/// `DataSetLoaderSimpleMFD::expandEnvVars`. An unset variable
/// with no default left in place is an error — but at this layer
/// we degrade to a literal pass-through so the failure surfaces
/// later (at facet-open time) with a more actionable error than a
/// catalog-parse rejection that hides the whole dataset list.
fn expand_env_vars(value: &str) -> String {
    if !value.contains("${") { return value.to_string(); }
    let mut out = String::with_capacity(value.len());
    let bytes = value.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'$' && bytes[i + 1] == b'{' {
            // find matching '}'
            if let Some(close_rel) = value[i + 2..].find('}') {
                let inner = &value[i + 2..i + 2 + close_rel];
                let (var_name, default_value) = match inner.split_once(":-") {
                    Some((n, d)) => (n, Some(d)),
                    None => (inner, None),
                };
                match std::env::var(var_name) {
                    Ok(v) => out.push_str(&v),
                    Err(_) => match default_value {
                        Some(d) => out.push_str(d),
                        None => {
                            // Unset and no default — preserve the
                            // literal so the parse doesn't fail
                            // catastrophically. A facet-open will
                            // surface the bad path.
                            out.push_str(&value[i..i + 2 + close_rel + 1]);
                        }
                    },
                }
                i = i + 2 + close_rel + 1;
                continue;
            }
        }
        // No expansion at this byte — copy through.
        // Use char_indices semantics by reading a full char.
        let ch = value[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Split a catalog entry key into `(dataset, profile)`.
///
/// Two key shapes are accepted:
///
/// - `"<dataset>:<profile>"` (knn_entries.yaml) — colon-delimited
///   with both halves non-empty.
/// - `"<dataset>"` (jvector `DataSetLoaderSimpleMFD` format) — no
///   colon; profile defaults to `"default"` so the entry surfaces
///   as a single-default-profile dataset.
///
/// Returns `None` for keys with empty components on either side of
/// a colon (e.g. `":foo"`, `"foo:"`).
pub(crate) fn parse_dataset_profile_key(key: &str) -> Option<(&str, &str)> {
    if let Some((name, profile)) = key.split_once(':') {
        if name.is_empty() || profile.is_empty() { return None; }
        return Some((name, profile));
    }
    if key.is_empty() { return None; }
    Some((key, "default"))
}

/// Append a trailing `/` if missing. Idempotent for empty strings
/// (returns `"/"`).
fn ensure_trailing_slash(s: &str) -> String {
    if s.ends_with('/') { s.to_string() } else { format!("{s}/") }
}

/// Concatenate `base` (assumed to end with `/`) and `rel`, taking
/// care to avoid `//` collisions when `rel` happens to start with
/// `/`. Absolute `rel` (containing `://` or starting with `/`)
/// shortcuts the base entirely so a caller can opt into per-entry
/// overrides without having to strip the default.
fn join_url(base: &str, rel: &str) -> String {
    if rel.contains("://") { return rel.to_string(); }
    if rel.starts_with('/') {
        // Absolute path component — keep just the scheme://host of
        // the base, if it's a URL; otherwise the literal path.
        if let Some(scheme_end) = base.find("://") {
            // Find the next '/' after the scheme to slice the host.
            let after_scheme = scheme_end + 3;
            if let Some(host_end) = base[after_scheme..].find('/') {
                let host_prefix = &base[..after_scheme + host_end];
                return format!("{host_prefix}{rel}");
            }
            // base = "scheme://host" with no path — append rel.
            return format!("{base}{rel}");
        }
        return rel.to_string();
    }
    format!("{base}{rel}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generic knn_entries-shape sample exercising the user-
    /// supplied pattern (`_defaults.base_url` + colon-keyed
    /// dataset:profile entries). Synthetic names — the live
    /// production catalogs use the same layout.
    const SAMPLE_YAML: &str = r#"
_defaults:
  base_url: s3://example-bucket/sample-prefix/sample-dataset

"sample-dataset:default":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/default/neighbor_indices.ivecs

"sample-dataset:100k":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/100k/neighbor_indices.ivecs

"sample-dataset:200k":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/200k/neighbor_indices.ivecs
"#;

    #[test]
    fn parses_user_supplied_sample() {
        let entries = parse_knn_entries_yaml(SAMPLE_YAML, "https://host/sample-dataset/").unwrap();
        // Single dataset, three profiles.
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.name, "sample-dataset");
        assert_eq!(entry.dataset_type, "knn_entries.yaml");
        let profile_names: Vec<&str> = entry.layout.profiles.profile_names();
        assert_eq!(profile_names, vec!["default", "100k", "200k"]);

        // Facet alias mapping: base→base_vectors, query→query_vectors, gt→neighbor_indices.
        let default = entry.layout.profiles.profile("default").unwrap();
        let view_names: Vec<&str> = default.views.keys().map(|s| s.as_str()).collect();
        assert!(view_names.contains(&"base_vectors"));
        assert!(view_names.contains(&"query_vectors"));
        assert!(view_names.contains(&"neighbor_indices"));

        // Facet paths resolved against the explicit `_defaults.base_url`,
        // not the catalog location (the s3:// URL wins).
        let bv = &default.views.get("base_vectors").unwrap().source.path;
        assert_eq!(
            bv,
            "s3://example-bucket/sample-prefix/sample-dataset/profiles/base/base_vectors.fvecs"
        );
    }

    /// When `_defaults.cache_dir` points at a directory that
    /// contains the named relative file, the facet path resolves
    /// to the absolute local path (preferred over the remote
    /// `base_url`). This is the jvector `DataSetLoaderSimpleMFD`
    /// "use cached copy when available" semantic.
    #[test]
    fn cache_dir_local_copy_preferred_over_remote() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path();
        // Create the cached file with the relative layout the
        // entry references.
        std::fs::create_dir_all(cache_dir.join("subdir")).unwrap();
        let cached_file = cache_dir.join("subdir/base.fvecs");
        std::fs::write(&cached_file, b"\x01\x00\x00\x00").unwrap();

        let yaml = format!(
            r#"
_defaults:
  base_url: s3://example/remote-prefix/
  cache_dir: {cache_root}

sample-ds:
  base: subdir/base.fvecs
  query: subdir/query.fvecs
  gt: subdir/gt.ivecs
"#,
            cache_root = cache_dir.display(),
        );
        let entries = parse_knn_entries_yaml(&yaml, "https://host/").unwrap();
        assert_eq!(entries.len(), 1);
        let default = entries[0].layout.profiles.profile("default").unwrap();
        let bv = &default.views.get("base_vectors").unwrap().source.path;
        // Cached → absolute local path.
        let expected = std::fs::canonicalize(&cached_file).unwrap();
        assert_eq!(bv, expected.to_str().unwrap());
        // Not cached → falls back to remote URL.
        let qv = &default.views.get("query_vectors").unwrap().source.path;
        assert_eq!(qv, "s3://example/remote-prefix/subdir/query.fvecs");
    }

    /// `${VAR}` and `${VAR:-default}` references in
    /// `_defaults.cache_dir` and `_defaults.base_url` are expanded
    /// against the process environment.
    #[test]
    fn env_var_expansion_in_defaults() {
        let yaml = r#"
_defaults:
  base_url: https://${BUCKET_HOST:-default-host.example.com}/prefix/
  cache_dir: ${VECTORDATA_TEST_NONEXISTENT_DIR:-/no/such/dir/exists}

sample-ds:
  base: a/b/c.fvecs
"#;
        let entries = parse_knn_entries_yaml(yaml, "https://host/").unwrap();
        let default = entries[0].layout.profiles.profile("default").unwrap();
        // cache_dir's default value points at a path that doesn't
        // exist → falls through to remote URL with expanded
        // base_url default.
        let bv = &default.views.get("base_vectors").unwrap().source.path;
        assert_eq!(bv, "https://default-host.example.com/prefix/a/b/c.fvecs");
    }

    /// Per-entry `cache_dir` and `base_url` override `_defaults`.
    #[test]
    fn per_entry_cache_dir_overrides_defaults() {
        let yaml = r#"
_defaults:
  base_url: https://default-host/prefix/
  cache_dir: /not/the/right/place

sample-ds:
  base_url: https://override-host/private/
  cache_dir: /still-nonexistent-override
  base: a/b/c.fvecs
"#;
        let entries = parse_knn_entries_yaml(yaml, "https://host/").unwrap();
        let default = entries[0].layout.profiles.profile("default").unwrap();
        // Both override cache_dir candidates are missing → falls
        // back to the per-entry base_url override.
        let bv = &default.views.get("base_vectors").unwrap().source.path;
        assert_eq!(bv, "https://override-host/private/a/b/c.fvecs");
    }

    /// Without `_defaults.base_url`, paths resolve relative to the
    /// catalog location passed in.
    #[test]
    fn falls_back_to_catalog_location_when_no_defaults() {
        let yaml = r#"
"d:default":
  base: a/b/c.fvecs
"#;
        let entries = parse_knn_entries_yaml(yaml, "https://host/d").unwrap();
        let path = &entries[0]
            .layout
            .profiles
            .profile("default")
            .unwrap()
            .views
            .get("base_vectors")
            .unwrap()
            .source
            .path;
        assert_eq!(path, "https://host/d/a/b/c.fvecs");
    }

    /// Multiple datasets in one file group correctly into separate
    /// entries.
    #[test]
    fn groups_multiple_datasets() {
        let yaml = r#"
"alpha:default":
  base: a/base.fvecs
"alpha:large":
  base: a/base.fvecs
"beta:default":
  base: b/base.fvecs
"#;
        let entries = parse_knn_entries_yaml(yaml, "https://host/").unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
        let alpha = entries.iter().find(|e| e.name == "alpha").unwrap();
        assert_eq!(alpha.layout.profiles.profile_names(), vec!["default", "large"]);
        let beta = entries.iter().find(|e| e.name == "beta").unwrap();
        assert_eq!(beta.layout.profiles.profile_names(), vec!["default"]);
    }

    /// A facet alias not in the mapping table passes through with
    /// its original name. Lets future / custom catalogs use
    /// arbitrary facet keys without a code change here.
    #[test]
    fn passes_through_unknown_facet_aliases() {
        let yaml = r#"
"d:default":
  base: a.fvecs
  metadata_content: meta.slab
  custom_facet: cf.bin
"#;
        let entries = parse_knn_entries_yaml(yaml, "https://host/").unwrap();
        let views = &entries[0].layout.profiles.profile("default").unwrap().views;
        assert!(views.contains_key("base_vectors"));
        assert!(views.contains_key("metadata_content"));
        assert!(views.contains_key("custom_facet"));
    }

    /// Malformed keys (empty halves of a colon-form key) are
    /// skipped with a log, not a fatal parse error. Bare-name keys
    /// without a colon are accepted as SimpleMFD-shape datasets
    /// with the implicit `default` profile.
    #[test]
    fn skips_malformed_keys() {
        let yaml = r#"
"no-colon-here":
  base: a.fvecs
":empty-name":
  base: a.fvecs
"empty-profile:":
  base: a.fvecs
"ok:default":
  base: a.fvecs
"#;
        let entries = parse_knn_entries_yaml(yaml, "https://host/").unwrap();
        let mut names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["no-colon-here", "ok"]);
        let no_colon = entries.iter().find(|e| e.name == "no-colon-here").unwrap();
        assert_eq!(no_colon.profile_names(), vec!["default"]);
    }

    /// An absolute URL in a per-entry path bypasses the base_url
    /// prefix — useful for one-off entries that point elsewhere.
    #[test]
    fn absolute_url_overrides_base_url() {
        let yaml = r#"
_defaults:
  base_url: s3://my-bucket/parent
"d:default":
  base: https://other-host/base.fvecs
  query: profiles/q.fvecs
"#;
        let entries = parse_knn_entries_yaml(yaml, "https://host/d/").unwrap();
        let views = &entries[0].layout.profiles.profile("default").unwrap().views;
        assert_eq!(views.get("base_vectors").unwrap().source.path, "https://other-host/base.fvecs");
        assert_eq!(views.get("query_vectors").unwrap().source.path, "s3://my-bucket/parent/profiles/q.fvecs");
    }

    /// Garbage YAML errors cleanly.
    #[test]
    fn rejects_invalid_yaml() {
        let yaml = "not: valid: yaml:";
        let r = parse_knn_entries_yaml(yaml, "https://host/");
        assert!(r.is_err());
    }

    /// Empty content parses to zero entries (not an error — the
    /// directory may legitimately have an empty catalog).
    #[test]
    fn empty_yaml_parses_to_empty() {
        let yaml = "{}";
        let entries = parse_knn_entries_yaml(yaml, "https://host/").unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn join_url_strips_double_slashes() {
        assert_eq!(join_url("https://h/p/", "a/b"), "https://h/p/a/b");
        assert_eq!(join_url("https://h/p", "a/b"), "https://h/pa/b"); // explicit no-slash form
    }

    #[test]
    fn join_url_passes_absolute_urls_through() {
        assert_eq!(join_url("https://h/p/", "s3://other/x"), "s3://other/x");
    }

    #[test]
    fn join_url_absolute_path_targets_host_root() {
        assert_eq!(join_url("https://h/p/d/", "/abs/path"), "https://h/abs/path");
    }

    #[test]
    fn parse_dataset_profile_key_basic() {
        assert_eq!(parse_dataset_profile_key("a:b"), Some(("a", "b")));
        assert_eq!(parse_dataset_profile_key("foo-bar:default"), Some(("foo-bar", "default")));
        // jvector DataSetLoaderSimpleMFD shape: bare dataset name
        // maps to the implicit `default` profile.
        assert_eq!(parse_dataset_profile_key("bare-name"), Some(("bare-name", "default")));
        assert_eq!(parse_dataset_profile_key(""), None);
        assert_eq!(parse_dataset_profile_key(":empty-name"), None);
        assert_eq!(parse_dataset_profile_key("empty-profile:"), None);
    }
}
