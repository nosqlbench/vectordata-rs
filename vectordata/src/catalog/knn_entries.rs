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

    let mut base_url = ensure_trailing_slash(catalog_location);
    if let Some(defaults) = top
        .get(serde_yaml::Value::String("_defaults".to_string()))
        .and_then(|v| v.as_mapping())
    {
        if let Some(bu) = defaults
            .get(serde_yaml::Value::String("base_url".to_string()))
            .and_then(|v| v.as_str())
        {
            base_url = ensure_trailing_slash(bu);
        }
    }

    // Accumulate by dataset name → (profile name → views map).
    let mut grouped: IndexMap<String, IndexMap<String, IndexMap<String, String>>> =
        IndexMap::new();

    for (key_v, val_v) in top.iter() {
        let key = match key_v.as_str() {
            Some(s) => s,
            None => continue,
        };
        if key == "_defaults" { continue; }
        let (dataset_name, profile_name) = match parse_dataset_profile_key(key) {
            Some(t) => t,
            None => {
                // Be permissive — skip malformed keys with a log
                // entry rather than failing the whole catalog.
                log::warn!("knn_entries.yaml: skipping malformed key '{key}'");
                continue;
            }
        };

        let facets = match val_v.as_mapping() {
            Some(m) => m,
            None => {
                log::warn!("knn_entries.yaml: entry '{key}' is not a map; skipping");
                continue;
            }
        };

        let mut view_map: IndexMap<String, String> = IndexMap::new();
        for (fk, fv) in facets {
            let alias = match fk.as_str() {
                Some(s) => s,
                None => continue,
            };
            let rel = match fv.as_str() {
                Some(s) => s,
                None => continue,
            };
            let canonical = canonical_facet_name(alias).to_string();
            view_map.insert(canonical, rel.to_string());
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
            for (facet, rel_path) in views {
                let full_path = join_url(&base_url, &rel_path);
                let view = DSView {
                    source: DSSource {
                        path: full_path,
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
            path: base_url.trim_end_matches('/').to_string(),
            dataset_type: "knn_entries.yaml".to_string(),
            layout: CatalogLayout {
                attributes: None,
                profiles: profile_group,
            },
        });
    }
    Ok(out)
}

/// Split a `"<dataset>:<profile>"` key. Returns `None` for keys
/// missing a colon or with empty components.
pub(crate) fn parse_dataset_profile_key(key: &str) -> Option<(&str, &str)> {
    let (name, profile) = key.split_once(':')?;
    if name.is_empty() || profile.is_empty() { return None; }
    Some((name, profile))
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

    /// The exact format the user pointed at — a few representative
    /// entries from the ibm-datapile-1b knn_entries.yaml.
    const SAMPLE_YAML: &str = r#"
_defaults:
  base_url: s3://jvector-datasets-infratest-20260401/5bcb3e5c3ca3694d492cf3449947869fbe9d3e14_private/ibm-datapile-1b

"ibm-datapile-1b:default":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/default/neighbor_indices.ivecs

"ibm-datapile-1b:100k":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/100k/neighbor_indices.ivecs

"ibm-datapile-1b:200k":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/200k/neighbor_indices.ivecs
"#;

    #[test]
    fn parses_user_supplied_sample() {
        let entries = parse_knn_entries_yaml(SAMPLE_YAML, "https://host/ibm-datapile-1b/").unwrap();
        // Single dataset, three profiles.
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.name, "ibm-datapile-1b");
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
            "s3://jvector-datasets-infratest-20260401/5bcb3e5c3ca3694d492cf3449947869fbe9d3e14_private/ibm-datapile-1b/profiles/base/base_vectors.fvecs"
        );
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

    /// Malformed keys (no colon, empty halves) are skipped with a
    /// log, not a fatal parse error. Keeps a partially-corrupt
    /// catalog usable.
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
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "ok");
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
        assert_eq!(parse_dataset_profile_key("no-colon"), None);
        assert_eq!(parse_dataset_profile_key(":empty-name"), None);
        assert_eq!(parse_dataset_profile_key("empty-profile:"), None);
    }
}
