// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the `knn_entries.yaml` catalog fallback.
//!
//! Verifies that every documented probe site — `Catalog::of` for
//! multi-dataset catalogs and `TestDataGroup::load` for the
//! single-dataset path — recognises a directory that contains
//! `knn_entries.yaml` (but no `catalog.json` / `catalog.yaml` /
//! `dataset.yaml`) and synthesizes the canonical view.

mod support;

use vectordata::catalog::{Catalog, CatalogSources};

use support::testserver::TestServer;

const KNN_ENTRIES_YAML: &str = r#"
_defaults:
  base_url: file:///tmp/knn-fixture/parent

"alpha:default":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/default/neighbor_indices.ivecs

"alpha:100k":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/100k/neighbor_indices.ivecs

"beta:default":
  base: beta/base.fvecs
  query: beta/query.fvecs
  gt: beta/gt.ivecs
"#;

/// Catalog resolver finds `knn_entries.yaml` when no canonical
/// catalog file is present, parses every dataset, and surfaces
/// per-profile facets via the canonical [`CatalogEntry`] shape.
#[test]
fn catalog_resolver_discovers_knn_entries_yaml() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("knn_entries.yaml"), KNN_ENTRIES_YAML).unwrap();

    let sources = CatalogSources::new().add_catalogs(&[tmp.path().to_string_lossy().to_string()]);
    let catalog = Catalog::of(&sources);

    let names: Vec<&str> = catalog.datasets().iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"alpha"), "got names: {names:?}");
    assert!(names.contains(&"beta"), "got names: {names:?}");

    let alpha = catalog.find_exact("alpha").expect("alpha resolves");
    let alpha_profiles: Vec<&str> = alpha.profile_names();
    assert!(alpha_profiles.contains(&"default"));
    assert!(alpha_profiles.contains(&"100k"));

    let beta = catalog.find_exact("beta").expect("beta resolves");
    assert_eq!(beta.profile_names(), vec!["default"]);

    // Facet alias mapping: base→base_vectors, gt→neighbor_indices.
    let default_profile = alpha.layout.profiles.profile("default").unwrap();
    let view_names: Vec<&str> = default_profile.views.keys().map(|s| s.as_str()).collect();
    assert!(view_names.contains(&"base_vectors"));
    assert!(view_names.contains(&"query_vectors"));
    assert!(view_names.contains(&"neighbor_indices"));

    // Paths resolved against the `_defaults.base_url` from the file
    // (the catalog directory is the fallback, but the explicit
    // base_url wins).
    let bv = &default_profile.views.get("base_vectors").unwrap().source.path;
    assert!(
        bv.starts_with("file:///tmp/knn-fixture/parent/"),
        "base_vectors path should resolve under _defaults.base_url, got {bv}",
    );
    assert!(bv.ends_with("profiles/base/base_vectors.fvecs"));
}

/// When a directory has no `_defaults.base_url`, facet paths
/// resolve relative to the catalog directory itself. This is the
/// common case for local fixtures and self-contained dataset
/// directories.
#[test]
fn no_defaults_uses_catalog_directory_as_base() {
    let yaml = r#"
"d:default":
  base: a/b/c.fvecs
"#;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("knn_entries.yaml"), yaml).unwrap();

    let sources = CatalogSources::new().add_catalogs(&[tmp.path().to_string_lossy().to_string()]);
    let catalog = Catalog::of(&sources);
    let d = catalog.find_exact("d").unwrap();
    let path = &d
        .layout
        .profiles
        .profile("default")
        .unwrap()
        .views
        .get("base_vectors")
        .unwrap()
        .source
        .path;
    assert!(
        path.ends_with("/a/b/c.fvecs"),
        "expected path to end with /a/b/c.fvecs, got {path}",
    );
    // Must contain the tmpdir path as the prefix (used as the
    // implicit base when no `_defaults.base_url` is supplied).
    let prefix = tmp.path().to_string_lossy();
    assert!(
        path.contains(prefix.as_ref()),
        "expected {prefix} in {path}",
    );
}

/// Canonical `catalog.json` wins over `knn_entries.yaml` when both
/// exist — the fallback only fires when no canonical file is
/// present. This guarantees the simplified format never silently
/// shadows a hand-curated catalog.
#[test]
fn canonical_catalog_takes_precedence_over_knn_entries() {
    let tmp = tempfile::tempdir().unwrap();
    // A trivial canonical catalog that should be preferred.
    std::fs::write(
        tmp.path().join("catalog.json"),
        r#"[{"name":"canonical","path":"canonical","layout":{"profiles":{}}}]"#,
    ).unwrap();
    // A knn_entries.yaml alongside that would otherwise also resolve.
    std::fs::write(
        tmp.path().join("knn_entries.yaml"),
        r#""shadowed:default": { base: x.fvecs, query: y.fvecs, gt: z.ivecs }"#,
    ).unwrap();

    let sources = CatalogSources::new().add_catalogs(&[tmp.path().to_string_lossy().to_string()]);
    let catalog = Catalog::of(&sources);
    let names: Vec<&str> = catalog.datasets().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["canonical"],
        "canonical catalog should win — knn_entries.yaml should not contribute when a canonical catalog is present",
    );
}

/// `TestDataGroup::load_from_path` already had a `knn_entries.yaml`
/// fallback; this test pins the behavior so a future refactor
/// doesn't drop it.
#[test]
fn test_data_group_load_from_path_falls_back_to_knn_entries() {
    let yaml = r#"
"myset:default":
  base: profiles/base/base.fvec
  query: profiles/base/query.fvec
  gt: profiles/base/gt.ivec
"#;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("knn_entries.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap())
        .expect("knn_entries fallback must produce a usable TestDataGroup");
    assert!(
        group.profile("default").is_some(),
        "default profile from knn_entries.yaml should be discoverable",
    );
}

/// HTTP fixture: a real test server serves `knn_entries.yaml`
/// (no `catalog.{json,yaml}`) and the catalog resolver discovers
/// the datasets through the live network round-trip. Mirrors how
/// the laion / ibm-datapile S3-hosted catalogs are consumed in
/// production.
#[test]
fn http_catalog_resolver_discovers_knn_entries_yaml() {
    let tmp = tempfile::tempdir().unwrap();
    // Lay out the dataset-shaped tree the user pointed at:
    //   <root>/ibm-datapile-1b/knn_entries.yaml
    let dataset_dir = tmp.path().join("ibm-datapile-1b");
    std::fs::create_dir(&dataset_dir).unwrap();
    let yaml = r#"
_defaults:
  base_url: REPLACED_AT_RUNTIME

"ibm-datapile-1b:default":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/default/neighbor_indices.ivecs

"ibm-datapile-1b:100k":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/100k/neighbor_indices.ivecs
"#;
    let server = TestServer::start(tmp.path()).unwrap();
    let base = server.base_url();
    let dataset_url = format!("{base}/ibm-datapile-1b");
    // Substitute the real base_url so the synthesized facet paths
    // point at the running fixture server.
    let yaml = yaml.replace("REPLACED_AT_RUNTIME", &dataset_url);
    std::fs::write(dataset_dir.join("knn_entries.yaml"), yaml).unwrap();

    let sources = CatalogSources::new().add_catalogs(&[dataset_url.clone()]);
    let catalog = Catalog::of(&sources);

    let names: Vec<&str> = catalog.datasets().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["ibm-datapile-1b"], "got names: {names:?}");

    let entry = catalog.find_exact("ibm-datapile-1b").expect("dataset resolves");
    assert_eq!(entry.dataset_type, "knn_entries.yaml");
    let profiles: Vec<&str> = entry.profile_names();
    assert!(profiles.contains(&"default"));
    assert!(profiles.contains(&"100k"));

    // Facet alias mapping survives the HTTP round-trip.
    let default = entry.layout.profiles.profile("default").unwrap();
    let view_names: Vec<&str> = default.views.keys().map(|s| s.as_str()).collect();
    assert!(view_names.contains(&"base_vectors"));
    assert!(view_names.contains(&"query_vectors"));
    assert!(view_names.contains(&"neighbor_indices"));

    // Paths resolved against the explicit `_defaults.base_url`.
    let bv = &default.views.get("base_vectors").unwrap().source.path;
    assert!(
        bv.starts_with(&dataset_url) && bv.ends_with("profiles/base/base_vectors.fvecs"),
        "expected facet path to live under the dataset URL, got {bv}",
    );
}

/// HTTP fixture: `TestDataGroup::load` against a URL with only a
/// `knn_entries.yaml` works the same as the local-file path.
/// Confirms the `load_from_url` fallback branch matches behaviour
/// the user expects from production S3-hosted catalogs.
#[test]
fn http_test_data_group_load_from_url_falls_back_to_knn_entries() {
    let tmp = tempfile::tempdir().unwrap();
    let dataset_dir = tmp.path().join("alpha");
    std::fs::create_dir(&dataset_dir).unwrap();
    std::fs::write(
        dataset_dir.join("knn_entries.yaml"),
        r#"
"alpha:default":
  base: base.fvec
  query: query.fvec
  gt: gt.ivec
"#,
    )
    .unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    let dataset_url = format!("{}/alpha", server.base_url());

    let group = vectordata::TestDataGroup::load(&dataset_url)
        .expect("HTTP knn_entries.yaml fallback must produce a usable TestDataGroup");
    assert!(
        group.profile("default").is_some(),
        "default profile from HTTP-served knn_entries.yaml should be discoverable",
    );
}

/// HTTP fixture: a catalog file is canonical only when it parses.
/// Specifically the resolver must NOT confuse an HTTP 404 for a
/// canonical-catalog parse error — it falls through to the
/// knn_entries.yaml probe instead. Regression guard for the
/// fallthrough logic in `load_catalog_entries`.
#[test]
fn http_404_on_canonical_falls_through_to_knn_entries() {
    let tmp = tempfile::tempdir().unwrap();
    // ONLY knn_entries.yaml — no catalog.json / catalog.yaml.
    std::fs::write(
        tmp.path().join("knn_entries.yaml"),
        r#"
"only:default":
  base: base.fvec
  query: query.fvec
  gt: gt.ivec
"#,
    )
    .unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    let sources = CatalogSources::new().add_catalogs(&[server.base_url()]);
    let catalog = Catalog::of(&sources);
    let names: Vec<&str> = catalog.datasets().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["only"]);
}

/// When a `knn_entries.yaml` describes multiple datasets, the
/// `TestDataGroup::load` fallback picks the dataset matching the
/// containing directory name. This is the laion-style layout
/// where the catalog file lives inside a per-dataset directory.
#[test]
fn test_data_group_picks_dataset_matching_directory_name() {
    let yaml = r#"
"alpha:default":
  base: alpha/base.fvec
  query: alpha/query.fvec
  gt: alpha/gt.ivec

"beta:default":
  base: beta/base.fvec
  query: beta/query.fvec
  gt: beta/gt.ivec

"beta:large":
  base: beta/base.fvec
  query: beta/query.fvec
  gt: beta/gt-large.ivec
"#;
    let parent = tempfile::tempdir().unwrap();
    let beta_dir = parent.path().join("beta");
    std::fs::create_dir(&beta_dir).unwrap();
    std::fs::write(beta_dir.join("knn_entries.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(beta_dir.to_str().unwrap())
        .expect("multi-dataset knn_entries fallback must work");
    // Both beta profiles available, no alpha bleed-through.
    assert!(group.profile("default").is_some());
    assert!(group.profile("large").is_some());
    let alpha_view = group.profile("alpha");
    assert!(alpha_view.is_none(), "alpha profile should not appear under beta");
}
