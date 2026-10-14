// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Fallback parser for the legacy `knn_entries.yaml` catalog format.
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

use crate::dataset::{profile::DSView, source::DSSource, CatalogEntry, DatasetConfig};

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

/// The environment variable jvector reads for where local copies of a
/// catalog's files live when the catalog does not say.
pub const DATASET_CACHE_DIR_ENV: &str = "DATASET_CACHE_DIR";

/// Parse a `knn_entries.yaml` document into one [`CatalogEntry`]
/// per distinct dataset name. Profiles are aggregated under each
/// entry; facet paths are resolved relative to the catalog
/// location (the directory containing `knn_entries.yaml`) — the
/// optional `_defaults.base_url` overrides that when present.
///
/// Each facet is read from a local copy when there is one, as jvector
/// reads it ([`resolve_facet_path`]): the local directory is the
/// entry's `cache_dir`, else `$DATASET_CACHE_DIR`, else the catalog's
/// own directory when the catalog is local.
///
/// `catalog_location` should be the directory URL or path that
/// `knn_entries.yaml` lives in (with or without trailing `/`).
pub fn parse_knn_entries_yaml(
    content: &str,
    catalog_location: &str,
) -> Result<Vec<CatalogEntry>, String> {
    let env_cache_dir = std::env::var(DATASET_CACHE_DIR_ENV).ok().filter(|v| !v.is_empty());
    parse_knn_entries_yaml_with(content, catalog_location, env_cache_dir.as_deref())
}

/// [`parse_knn_entries_yaml`] with `$DATASET_CACHE_DIR`'s value given,
/// so the lookup is a function of its inputs.
pub(crate) fn parse_knn_entries_yaml_with(
    content: &str,
    catalog_location: &str,
    env_cache_dir: Option<&str>,
) -> Result<Vec<CatalogEntry>, String> {
    // Top-level shape: a map of String -> serde_yaml::Value.
    let top = top_level(content)?;

    // `_defaults` carries per-catalog defaults that are folded into
    // each entry. The two fields we honour mirror jvector
    // `DataSetLoaderSimpleMFD`: `base_url` (where to fetch from)
    // and `cache_dir` (where the local copies live, if any).
    let (default_base_url, default_cache_dir) =
        extract_defaults(&top, catalog_location);
    // A local catalog's own directory is where its files are, unless the
    // catalog or the environment names another; jvector's last resort.
    let local_catalog_dir = crate::Location::parse(catalog_location)
        .ok()
        .and_then(|l| l.as_path().map(|p| p.to_string_lossy().into_owned()));

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
            .map(|s| crate::location::dir_form(&s))
            .unwrap_or_else(|| default_base_url.clone());
        let entry_cache_dir = raw
            .get(serde_yaml::Value::String("cache_dir".to_string()))
            .and_then(|v| v.as_str())
            .map(expand_env_vars)
            .or_else(|| default_cache_dir.clone())
            .or_else(|| env_cache_dir.map(str::to_string))
            .or_else(|| local_catalog_dir.clone());

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

    // Each dataset is an embedded entry: its home is the base and its
    // name, and every facet keeps the path resolved above, as written.
    let mut out: Vec<CatalogEntry> = Vec::with_capacity(grouped.len());
    for (name, profiles) in grouped {
        let description = profiles
            .into_iter()
            .fold(DatasetConfig::builder(name.clone()), |dataset, (profile_name, views)| {
                dataset.profile(&profile_name, |profile| {
                    views.into_iter().fold(profile, |profile, (facet, path)| {
                        let source = DSSource { path, ..Default::default() };
                        profile.view(&facet, DSView { source, ..Default::default() })
                    })
                })
            })
            .build()
            .map_err(|e| format!("dataset '{name}': {e}"))?;
        out.push(
            CatalogEntry::builder(name.clone())
                .home_as_written(format!("{}/{name}/", default_base_url.trim_end_matches('/')))
                .dataset(description)
                .build()?,
        );
    }
    Ok(out)
}

/// The document's top-level map. A key given twice keeps its last
/// entry, as jvector's YAML loader and earlier releases read it, and is
/// warned about: hand-kept files repeat an entry when a newer copy of
/// the data is appended below the old one.
fn top_level(content: &str) -> Result<serde_yaml::Mapping, String> {
    struct LastWins(serde_yaml::Mapping, Vec<String>);
    impl<'de> serde::Deserialize<'de> for LastWins {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct Visit;
            impl<'de> serde::de::Visitor<'de> for Visit {
                type Value = LastWins;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("a map of catalog entries")
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<LastWins, A::Error> {
                    let mut out = LastWins(serde_yaml::Mapping::new(), Vec::new());
                    while let Some((key, value)) = map.next_entry::<serde_yaml::Value, serde_yaml::Value>()? {
                        if out.0.insert(key.clone(), value).is_some() {
                            out.1.push(key.as_str().map(str::to_string).unwrap_or_else(|| format!("{key:?}")));
                        }
                    }
                    Ok(out)
                }
            }
            deserializer.deserialize_map(Visit)
        }
    }
    let LastWins(top, repeated) = serde_yaml::from_str(content).map_err(|e| format!("parse knn_entries.yaml: {e}"))?;
    for key in repeated {
        log::warn!("knn_entries.yaml: entry '{key}' is given more than once; the last one is used");
    }
    Ok(top)
}

/// Every facet path a `knn_entries.yaml` declares, as written and
/// before `_defaults` resolve it: `(entry key, facet alias, path)`, in
/// file order. What a publish checks against the files beside the
/// catalog, which is where they will be published from.
pub(crate) fn declared_facet_paths(content: &str) -> Result<Vec<(String, String, String)>, String> {
    let top = top_level(content)?;
    let mut out = Vec::new();
    for (key, value) in &top {
        let (Some(key), Some(facets)) = (key.as_str(), value.as_mapping()) else { continue };
        if key.starts_with('_') || parse_dataset_profile_key(key).is_none() {
            continue;
        }
        for (alias, path) in facets {
            let (Some(alias), Some(path)) = (alias.as_str(), path.as_str()) else { continue };
            if alias != "base_url" && alias != "cache_dir" {
                out.push((key.to_string(), alias.to_string(), path.to_string()));
            }
        }
    }
    Ok(out)
}

/// Extract `(default_base_url, default_cache_dir)` from a parsed
/// `_defaults` mapping. `base_url` defaults to the catalog
/// location (with a trailing `/` so `concat_form` composes cleanly).
/// `cache_dir` has no default — its absence simply means "no
/// local cache layer; always use the remote URL".
fn extract_defaults(top: &serde_yaml::Mapping, catalog_location: &str) -> (String, Option<String>) {
    let mut base_url = crate::location::dir_form(catalog_location);
    let mut cache_dir: Option<String> = None;
    if let Some(defaults) = top
        .get(serde_yaml::Value::String("_defaults".to_string()))
        .and_then(|v| v.as_mapping())
    {
        if let Some(bu) = defaults
            .get(serde_yaml::Value::String("base_url".to_string()))
            .and_then(|v| v.as_str())
        {
            base_url = crate::location::dir_form(&expand_env_vars(bu));
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
/// is used. `cache_dir` is the local directory as jvector chooses it:
/// the entry's (or `_defaults`') `cache_dir`, else `$DATASET_CACHE_DIR`,
/// else a local catalog's own directory. An absolute URL or absolute path in `rel` short-
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
            // re-checking the filesystem. Absolute, not canonical: a
            // file kept as a symlink (a dataset's facet linked to its
            // source) is named where the catalog names it, as jvector's
            // `cacheDir.resolve(file)` does, not where the link points.
            return std::path::absolute(&candidate)
                .unwrap_or(candidate)
                .to_string_lossy()
                .into_owned();
        }
    }
    crate::location::concat_form(base_url, rel)
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
        assert_eq!(entry.name(), "sample-dataset");
        assert_eq!(entry.dataset_type(), "knn_entries.yaml");
        let profile_names: Vec<&str> = entry.profiles().profile_names();
        assert_eq!(profile_names, vec!["default", "100k", "200k"]);

        // Facet alias mapping: base→base_vectors, query→query_vectors, gt→neighbor_indices.
        let default = entry.profiles().profile("default").unwrap();
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
        let default = entries[0].profiles().profile("default").unwrap();
        let bv = &default.views.get("base_vectors").unwrap().source.path;
        // Cached → absolute local path.
        let expected = std::path::absolute(&cached_file).unwrap();
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
        let default = entries[0].profiles().profile("default").unwrap();
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
        let default = entries[0].profiles().profile("default").unwrap();
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
            .profiles()
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
        let names: Vec<&str> = entries.iter().map(|e| e.name()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
        let alpha = entries.iter().find(|e| e.name() == "alpha").unwrap();
        assert_eq!(alpha.profiles().profile_names(), vec!["default", "large"]);
        let beta = entries.iter().find(|e| e.name() == "beta").unwrap();
        assert_eq!(beta.profiles().profile_names(), vec!["default"]);
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
        let views = &entries[0].profiles().profile("default").unwrap().views;
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
        let mut names: Vec<&str> = entries.iter().map(|e| e.name()).collect();
        names.sort();
        assert_eq!(names, vec!["no-colon-here", "ok"]);
        let no_colon = entries.iter().find(|e| e.name() == "no-colon-here").unwrap();
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
        let views = &entries[0].profiles().profile("default").unwrap().views;
        assert_eq!(views.get("base_vectors").unwrap().source.path, "https://other-host/base.fvecs");
        assert_eq!(views.get("query_vectors").unwrap().source.path, "s3://my-bucket/parent/profiles/q.fvecs");
    }

    /// **The local directory for an entry's files is chosen as jvector
    /// chooses it**: the entry's `cache_dir`, else `$DATASET_CACHE_DIR`,
    /// else a local catalog's own directory. A file found there is read
    /// from there; any other comes from `base_url`. A remote catalog has
    /// no directory on disk to look in.
    #[test]
    fn local_copies_are_found_where_jvector_looks() {
        let tmp = tempfile::tempdir().unwrap();
        let (catalog, env, named) = (tmp.path().join("catalog"), tmp.path().join("env"), tmp.path().join("named"));
        for (dir, file) in [(&catalog, "beside.fvec"), (&env, "env.fvec"), (&named, "named.fvec")] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join(file), b"x").unwrap();
        }
        let local = |dir: &std::path::Path, file: &str| std::path::absolute(dir.join(file)).unwrap().to_string_lossy().into_owned();
        let source = |yaml: &str, location: &str, env_dir: Option<&str>, facet: &str| {
            let entries = parse_knn_entries_yaml_with(yaml, location, env_dir).unwrap();
            entries[0].profiles().profile("default").unwrap().views[facet].source.path.clone()
        };
        let facets = "  base: beside.fvec\n  query: env.fvec\n  gt: named.fvec\n";
        let plain = format!("_defaults:\n  base_url: s3://b/ds\nds:\n{facets}");
        let at = catalog.to_str().unwrap();

        // The catalog's directory, with nothing else named.
        assert_eq!(source(&plain, at, None, "base_vectors"), local(&catalog, "beside.fvec"));
        assert_eq!(source(&plain, at, None, "query_vectors"), "s3://b/ds/env.fvec");
        // $DATASET_CACHE_DIR outranks the catalog's directory.
        let env_dir = env.to_str();
        assert_eq!(source(&plain, at, env_dir, "query_vectors"), local(&env, "env.fvec"));
        assert_eq!(source(&plain, at, env_dir, "base_vectors"), "s3://b/ds/beside.fvec");
        // A named cache_dir outranks both.
        let with_cache = format!("_defaults:\n  base_url: s3://b/ds\n  cache_dir: {}\nds:\n{facets}", named.display());
        assert_eq!(source(&with_cache, at, env_dir, "neighbor_indices"), local(&named, "named.fvec"));
        assert_eq!(source(&with_cache, at, env_dir, "query_vectors"), "s3://b/ds/env.fvec");
        // A file kept as a symlink is named where the catalog names it,
        // not where the link points.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(env.join("env.fvec"), catalog.join("linked.fvec")).unwrap();
            let linked = "_defaults:\n  base_url: s3://b/ds\nds:\n  base: linked.fvec\n";
            assert_eq!(source(linked, at, None, "base_vectors"), local(&catalog, "linked.fvec"));
        }
        // A remote catalog looks in no local directory of its own.
        let remote = format!("ds:\n{facets}");
        assert_eq!(source(&remote, "https://host/ds/", None, "base_vectors"), "https://host/ds/beside.fvec");
    }

    /// **An entry given twice keeps its last copy**, as jvector's loader
    /// and earlier releases read it, rather than failing the whole file:
    /// hand-kept catalogs append a newer copy of an entry below the old.
    #[test]
    fn a_repeated_entry_keeps_its_last_copy() {
        let yaml = "\"d:default\":\n  base: old/b.fvec\n\"d:other\":\n  base: o.fvec\n\"d:default\":\n  base: new/b.fvec\n";
        let entries = parse_knn_entries_yaml_with(yaml, "https://host/", None).unwrap();
        let profiles = entries[0].profiles();
        assert_eq!(profiles.profile("default").unwrap().views["base_vectors"].source.path, "https://host/new/b.fvec");
        assert!(profiles.profile("other").is_some());
        let declared = declared_facet_paths(yaml).unwrap();
        assert_eq!(declared.iter().filter(|(k, _, _)| k == "d:default").map(|(_, _, p)| p.as_str()).collect::<Vec<_>>(), ["new/b.fvec"]);
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
