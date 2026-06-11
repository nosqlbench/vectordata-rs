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

use std::sync::LazyLock;

use vectordata::catalog::{Catalog, CatalogSources};

use support::testserver::TestServer;

/// Test-wide cache root under `target/tmp` (reaped by `cargo clean`),
/// mirroring the `http_storage.rs` convention. Installed lazily by
/// tests that actually open facet storage.
static TEST_CACHE_DIR: LazyLock<tempfile::TempDir> = LazyLock::new(|| {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).expect("create test cache root")
});

fn init_test_cache() {
    vectordata::settings::override_cache_dir_for_process(
        TEST_CACHE_DIR.path().to_path_buf(),
    );
}

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
/// large S3-hosted catalogs are consumed in production.
#[test]
fn http_catalog_resolver_discovers_knn_entries_yaml() {
    let tmp = tempfile::tempdir().unwrap();
    // Lay out the dataset-shaped tree the user pointed at:
    //   <root>/example-1b/knn_entries.yaml
    let dataset_dir = tmp.path().join("example-1b");
    std::fs::create_dir(&dataset_dir).unwrap();
    let yaml = r#"
_defaults:
  base_url: REPLACED_AT_RUNTIME

"example-1b:default":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/default/neighbor_indices.ivecs

"example-1b:100k":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/100k/neighbor_indices.ivecs
"#;
    let server = TestServer::start(tmp.path()).unwrap();
    let base = server.base_url();
    let dataset_url = format!("{base}/example-1b");
    // Substitute the real base_url so the synthesized facet paths
    // point at the running fixture server.
    let yaml = yaml.replace("REPLACED_AT_RUNTIME", &dataset_url);
    std::fs::write(dataset_dir.join("knn_entries.yaml"), yaml).unwrap();

    let sources = CatalogSources::new().add_catalogs(std::slice::from_ref(&dataset_url));
    let catalog = Catalog::of(&sources);

    let names: Vec<&str> = catalog.datasets().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["example-1b"], "got names: {names:?}");

    let entry = catalog.find_exact("example-1b").expect("dataset resolves");
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

/// `Catalog::open` must synthesize a `TestDataGroup` directly
/// from a `knn_entries.yaml`-shape entry's embedded layout — it
/// cannot try to re-load `dataset.yaml` from the entry's `path`
/// because that path is just the catalog's base URL and there is
/// no `dataset.yaml` at that location. Regression guard for the
/// "Failed to read dataset configuration: No such file or
/// directory" failure that hit `vectordata explore` against an `s3://`
/// SimpleMFD catalog.
#[test]
fn catalog_open_synthesizes_group_for_knn_entries_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    let base = server.base_url();
    let yaml = r#"
_defaults:
  base_url: s3://example-bucket/some-prefix/

ds-x:
  base: x/base.fvecs
  query: x/query.fvecs
  gt: x/gt.ivecs
"#.to_string();
    std::fs::write(tmp.path().join("entries.yaml"), &yaml).unwrap();
    let sources = CatalogSources::new()
        .add_catalogs(&[format!("{base}/entries.yaml")]);
    let catalog = Catalog::of(&sources);

    // Calling open() must not attempt to fetch `dataset.yaml` from
    // the s3:// base; instead it builds the group from the
    // embedded layout.
    let group = catalog.open("ds-x").expect("knn_entries entry must open via embedded layout");
    let view = group.profile("default").expect("default profile resolves");
    // The synthesised view exposes the facet sources verbatim
    // (already absolutised by the parser).
    let bv = view.facet_source("base_vectors").expect("base_vectors source");
    assert_eq!(bv, "s3://example-bucket/some-prefix/x/base.fvecs");
}

/// HTTP fixture: a URL pointing at a jvector
/// `DataSetLoaderSimpleMFD` catalog (plain dataset-name keys, no
/// `":profile"` suffix) surfaces every entry through
/// `Catalog::of`. Regression guard for the production
/// `protected-catalog.yaml` shape — previously, every entry was
/// silently skipped as "malformed" because the parser required a
/// colon.
#[test]
fn http_catalog_resolver_accepts_simple_mfd_shape() {
    let tmp = tempfile::tempdir().unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    let base = server.base_url();
    let yaml = format!(
        r#"
_defaults:
  cache_dir: ${{DATASET_CACHE_DIR:-dataset_cache}}/private-mirror
  base_url: {base}/private-root/

ds-a-1m:
  base: ds-a/base.fvecs
  query: ds-a/query.fvecs
  gt: ds-a/gt.ivecs

ds-b-10m:
  base: ds-b/base.fvecs
  query: ds-b/query.fvecs
  gt: ds-b/gt.ivecs

ds-c-1m:
  base: ds-c/base.fvecs
  query: ds-c/query.fvecs
  gt: ds-c/gt.ivecs
"#
    );
    std::fs::write(tmp.path().join("nonstandard-catalog.yaml"), &yaml).unwrap();
    let catalog_url = format!("{base}/nonstandard-catalog.yaml");
    let sources = CatalogSources::new().add_catalogs(&[catalog_url]);
    let catalog = Catalog::of(&sources);

    let mut names: Vec<&str> = catalog.datasets().iter().map(|e| e.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["ds-a-1m", "ds-b-10m", "ds-c-1m"], "got names: {names:?}");

    // Each SimpleMFD entry is a single-default-profile dataset.
    let ds_a = catalog.find_exact("ds-a-1m").expect("ds-a-1m resolves");
    assert_eq!(ds_a.profile_names(), vec!["default"]);
    let default = ds_a.layout.profiles.profile("default").unwrap();
    let bv = &default.views.get("base_vectors").unwrap().source.path;
    assert!(
        bv.starts_with(&format!("{base}/private-root/")) && bv.ends_with("ds-a/base.fvecs"),
        "expected facet path resolved against _defaults.base_url, got {bv}",
    );
}

/// HTTP fixture: a URL that explicitly names a YAML catalog file
/// with a non-standard basename (e.g. `protected-catalog.yaml`) is
/// recognised by the back-end as a knn_entries-shape catalog,
/// without any filename probing. This is `DataSetLoaderSimpleMFD`
/// parity — the catalog file IS the catalog regardless of name.
#[test]
fn http_catalog_resolver_accepts_arbitrary_yaml_filename() {
    let tmp = tempfile::tempdir().unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    let base = server.base_url();

    let yaml = format!(
        r#"
_defaults:
  base_url: {base}/protected

"private-ds:default":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt: profiles/default/neighbor_indices.ivecs
"#
    );
    // Non-standard filename, served from server root.
    std::fs::write(tmp.path().join("protected-catalog.yaml"), &yaml).unwrap();

    let catalog_url = format!("{base}/protected-catalog.yaml");
    let sources = CatalogSources::new().add_catalogs(std::slice::from_ref(&catalog_url));
    let catalog = Catalog::of(&sources);

    let names: Vec<&str> = catalog.datasets().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["private-ds"], "got names: {names:?}");

    let entry = catalog.find_exact("private-ds").expect("dataset resolves");
    let default = entry.layout.profiles.profile("default").unwrap();
    let bv = &default.views.get("base_vectors").unwrap().source.path;
    assert!(
        bv.starts_with(&format!("{base}/protected/")) && bv.ends_with("profiles/base/base_vectors.fvecs"),
        "expected facet path resolved against _defaults.base_url, got {bv}",
    );
}

/// HTTP fixture: a URL that explicitly names a YAML file holding a
/// canonical *array* catalog (the `catalog.yaml` shape) under a
/// non-standard basename is still recognised. Confirms the content-
/// shape dispatch picks the canonical branch correctly.
#[test]
fn http_catalog_resolver_accepts_arbitrary_canonical_yaml() {
    let tmp = tempfile::tempdir().unwrap();
    // Layout-embedded canonical catalog under a non-standard name.
    let yaml = r#"
- name: alpha
  path: alpha/dataset.yaml
  dataset_type: dataset.yaml
  layout:
    profiles:
      default:
        base_vectors: base.fvec
"#;
    std::fs::write(tmp.path().join("my-private.yaml"), yaml).unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    let catalog_url = format!("{}/my-private.yaml", server.base_url());

    let sources = CatalogSources::new().add_catalogs(&[catalog_url]);
    let catalog = Catalog::of(&sources);

    let names: Vec<&str> = catalog.datasets().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["alpha"], "got names: {names:?}");
}

/// HTTP fixture: `TestDataGroup::load` against a URL that names a
/// non-standard catalog file resolves to the right dataset by
/// dispatching on content shape (knn_entries-style map).
#[test]
fn http_test_data_group_load_from_url_with_arbitrary_yaml_filename() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("protected-catalog.yaml"),
        r#"
"alpha:default":
  base: base.fvec
  query: query.fvec
  gt: gt.ivec
"#,
    )
    .unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}/protected-catalog.yaml", server.base_url());

    let group = vectordata::TestDataGroup::load(&url)
        .expect("explicit non-standard yaml URL must produce a usable TestDataGroup");
    assert!(
        group.profile("default").is_some(),
        "default profile from HTTP-served non-standard yaml should be discoverable",
    );
}

/// Local-path equivalent: pointing `TestDataGroup::load` at a yaml
/// file with any name (not just `dataset.yaml` or `knn_entries.yaml`)
/// dispatches by content shape.
#[test]
fn test_data_group_load_from_path_with_arbitrary_yaml_filename() {
    let tmp = tempfile::tempdir().unwrap();
    let catalog_path = tmp.path().join("protected-catalog.yaml");
    std::fs::write(
        &catalog_path,
        r#"
"alpha:default":
  base: base.fvec
  query: query.fvec
  gt: gt.ivec
"#,
    )
    .unwrap();

    let group = vectordata::TestDataGroup::load(catalog_path.to_str().unwrap())
        .expect("local non-standard yaml path must produce a usable TestDataGroup");
    assert!(group.profile("default").is_some());
}

/// When a `knn_entries.yaml` describes multiple datasets, the
/// `TestDataGroup::load` fallback picks the dataset matching the
/// containing directory name. This is the knn_entries-style layout
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

/// End-to-end: a local `knn_entries.yaml` whose facet entries are
/// **relative paths** must resolve against the catalog file's
/// directory AND those paths must actually open and read
/// correctly through the full `TestDataGroup → view →
/// VectorReader` stack. Earlier tests confirm the path *shape*
/// (`no_defaults_uses_catalog_directory_as_base`); this confirms
/// the assembled path is functionally usable — covers the
/// open+mmap+byte-read chain that the user-facing API exercises.
#[test]
fn local_knn_entries_relative_paths_open_and_read_correctly() {
    use byteorder::{LittleEndian, WriteBytesExt};
    use vectordata::TestDataGroup;

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // Lay out a self-contained dataset directory with the facet
    // files in subdirectories — exercises multi-segment relative
    // paths (not just `base.fvec` next to the yaml).
    //
    //   <root>/
    //     knn_entries.yaml
    //     profiles/base/base_vectors.fvec
    //     profiles/base/query_vectors.fvec
    //     profiles/default/neighbor_indices.ivec
    std::fs::create_dir_all(root.join("profiles/base")).unwrap();
    std::fs::create_dir_all(root.join("profiles/default")).unwrap();

    let write_fvec = |path: &std::path::Path, count: usize, dim: usize| {
        let mut f = std::fs::File::create(path).unwrap();
        for i in 0..count {
            f.write_i32::<LittleEndian>(dim as i32).unwrap();
            for d in 0..dim {
                f.write_f32::<LittleEndian>((i * 100 + d) as f32).unwrap();
            }
        }
    };
    let write_ivec = |path: &std::path::Path, count: usize, dim: usize| {
        let mut f = std::fs::File::create(path).unwrap();
        for i in 0..count {
            f.write_i32::<LittleEndian>(dim as i32).unwrap();
            for d in 0..dim {
                f.write_i32::<LittleEndian>((i * 10 + d) as i32).unwrap();
            }
        }
    };

    write_fvec(&root.join("profiles/base/base_vectors.fvec"), 16, 4);
    write_fvec(&root.join("profiles/base/query_vectors.fvec"), 8, 4);
    write_ivec(&root.join("profiles/default/neighbor_indices.ivec"), 8, 3);

    // No `_defaults.base_url` — facet paths resolve against
    // the catalog directory (this is the documented local-only
    // self-contained-dataset shape).
    let yaml = r#"
"local-ds:default":
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  gt: profiles/default/neighbor_indices.ivec
"#;
    std::fs::write(root.join("knn_entries.yaml"), yaml).unwrap();

    // Open via the same TestDataGroup API a downstream consumer
    // would call. This goes catalog → knn_entries parser →
    // resolved layout (with absolute filesystem paths produced
    // by `join_url`) → view → reader.
    let group = TestDataGroup::load(root.to_str().unwrap())
        .expect("local knn_entries.yaml opens via TestDataGroup");
    let view = group.profile("default").expect("default profile exists");

    // 1. base_vectors — reads through the view trait's
    //    `base_vectors()` accessor.
    let base = view.base_vectors().expect("base_vectors opens");
    assert_eq!(base.count(), 16);
    assert_eq!(base.dim(), 4);
    for i in 0..base.count() {
        let v = base.get(i).unwrap();
        assert_eq!(v, vec![
            (i * 100) as f32,
            (i * 100 + 1) as f32,
            (i * 100 + 2) as f32,
            (i * 100 + 3) as f32,
        ], "base record {i} mismatch");
    }

    // 2. query_vectors — exercises a different facet so the
    //    resolution isn't accidentally hardcoded to one slot.
    let query = view.query_vectors().expect("query_vectors opens");
    assert_eq!(query.count(), 8);
    for i in 0..query.count() {
        assert_eq!(query.get(i).unwrap(), vec![
            (i * 100) as f32,
            (i * 100 + 1) as f32,
            (i * 100 + 2) as f32,
            (i * 100 + 3) as f32,
        ]);
    }

    // 3. neighbor_indices — verifies the `gt` → `neighbor_indices`
    //    facet alias mapping carries through to a working
    //    integer reader.
    let nbrs = view.neighbor_indices().expect("neighbor_indices opens");
    assert_eq!(nbrs.count(), 8);
    assert_eq!(nbrs.dim(), 3);
    let row = nbrs.get(4).unwrap();
    assert_eq!(row, vec![40i32, 41, 42]);

    // 4. The view must report local storage — no precache or
    //    HTTP round-trips involved. The local-path
    //    short-circuit at `view::is_local_facet_source` is the
    //    code path that lets this work even when the parser
    //    emits absolute paths joined against the catalog dir.
    let base_storage = view.open_facet_storage("base_vectors").unwrap();
    assert!(base_storage.is_local(),
        "facet against a local knn_entries.yaml must open as local mmap");
    assert!(base_storage.is_complete(),
        "local facet has no fill state — already complete");
}

/// Symmetric coverage: when `Catalog::of` synthesises the group
/// from a local `knn_entries.yaml` (rather than
/// `TestDataGroup::load_from_path`), the resolved facet paths
/// must still open + read correctly. This is the entry point
/// `vectordata datasets list` and similar tools use.
#[test]
fn local_knn_entries_via_catalog_open_reads_correctly() {
    use byteorder::{LittleEndian, WriteBytesExt};

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("d/a/b")).unwrap();
    let mut f = std::fs::File::create(root.join("d/a/b/base.fvec")).unwrap();
    for i in 0..8u32 {
        f.write_i32::<LittleEndian>(2).unwrap();
        f.write_f32::<LittleEndian>(i as f32).unwrap();
        f.write_f32::<LittleEndian>((i + 1) as f32).unwrap();
    }
    drop(f);

    let yaml = r#"
"d:default":
  base: a/b/base.fvec
"#;
    std::fs::write(root.join("d/knn_entries.yaml"), yaml).unwrap();

    let sources = CatalogSources::new()
        .add_catalogs(&[root.join("d").to_string_lossy().to_string()]);
    let catalog = Catalog::of(&sources);
    let group = catalog.open("d").expect("d resolves through Catalog::open");
    let view = group.profile("default").expect("default profile");
    let base = view.base_vectors().expect("base_vectors opens");
    assert_eq!(base.count(), 8);
    let v = base.get(3).unwrap();
    assert_eq!(v, vec![3.0f32, 4.0]);
}

/// Facets that live OUTSIDE a dataset's home URL (legal in
/// knn_entries catalogs — several sized datasets sharing one remote
/// directory of files) must still cache under the dataset-keyed
/// layout: `<cache_root>/<dataset>/<basename>`, flat, with a
/// `.publish_url` binding alongside. Never URL-keyed, never the old
/// `<dataset>/s3:/...` nested-URL shape (whose colon broke Windows).
#[test]
fn out_of_home_facets_cache_flat_under_dataset_dir() {
    let tmp = tempfile::tempdir().unwrap();
    // Files live in a `shared/` directory that is a *sibling* of the
    // per-dataset home URLs (`<base>/root/ds-a-1m/`, `.../ds-a-6m/`).
    let shared = tmp.path().join("root/shared");
    std::fs::create_dir_all(&shared).unwrap();
    // Minimal valid fvecs: one record, dim 2.
    let mut rec: Vec<u8> = Vec::new();
    rec.extend_from_slice(&2i32.to_le_bytes());
    rec.extend_from_slice(&1.0f32.to_le_bytes());
    rec.extend_from_slice(&2.0f32.to_le_bytes());
    std::fs::write(shared.join("base_1m.fvecs"), &rec).unwrap();
    std::fs::write(shared.join("base_6m.fvecs"), &rec).unwrap();
    std::fs::write(shared.join("query_10k.fvecs"), &rec).unwrap();

    let server = TestServer::start(tmp.path()).unwrap();
    let base = server.base_url();
    let yaml = format!(
        r#"
_defaults:
  base_url: {base}/root/

ds-a-1m:
  base: shared/base_1m.fvecs
  query: shared/query_10k.fvecs
  gt: shared/gt.ivecs

ds-a-6m:
  base: shared/base_6m.fvecs
  query: shared/query_10k.fvecs
  gt: shared/gt.ivecs
"#
    );
    std::fs::write(tmp.path().join("root/knn_entries.yaml"), &yaml).unwrap();
    init_test_cache();

    let sources = CatalogSources::new()
        .add_catalogs(&[format!("{base}/root/knn_entries.yaml")]);
    let catalog = Catalog::of(&sources);

    let open_query_path = |name: &str| {
        let group = catalog.open(name).expect("dataset opens");
        let view = group.profile("default").expect("default profile");
        let storage = view.open_facet_storage("query_vectors").expect("storage opens");
        storage.cache_path().expect("chunk store pre-sizes the cache file")
    };
    // The resolver records which document each entry came from —
    // this is what the picker's Source action displays. It must be
    // the actual catalog file, not a name fabricated from base_url.
    for e in catalog.datasets() {
        assert_eq!(e.catalog_file.as_deref(),
            Some(format!("{base}/root/knn_entries.yaml").as_str()),
            "entry '{}' must record its source document", e.name);
    }

    let q_1m = open_query_path("ds-a-1m");
    let q_6m = open_query_path("ds-a-6m");

    let cache_root = TEST_CACHE_DIR.path();
    // Dataset-keyed layout: each dataset holds its own flat copy.
    assert_eq!(q_1m, cache_root.join("ds-a-1m/query_10k.fvecs"),
        "out-of-home facet must cache flat under its dataset directory");
    assert_eq!(q_6m, cache_root.join("ds-a-6m/query_10k.fvecs"));

    // No URL-shaped path components anywhere under the dataset dirs.
    for q in [&q_1m, &q_6m] {
        let rel = q.strip_prefix(cache_root).unwrap();
        for comp in rel.components() {
            let c = comp.as_os_str().to_string_lossy();
            assert!(!c.contains(':'),
                "URL-shaped path component '{c}' in cache path {rel:?}");
        }
    }

    // Single binding: `.publish_url` recorded at download time names
    // the dataset's home URL, so a later `vectordata push` of the
    // cached directory is already bound.
    for ds in ["ds-a-1m", "ds-a-6m"] {
        let marker = cache_root.join(ds).join(".publish_url");
        let content = std::fs::read_to_string(&marker)
            .unwrap_or_else(|e| panic!("missing {}: {e}", marker.display()));
        let parsed = vectordata::push::binding::parse_publish_url(&content)
            .expect("binding parses");
        assert_eq!(parsed.url, format!("{base}/root/{ds}/"),
            "binding must name the dataset home URL");
    }
}

/// Two facets of one profile mapping to the same cache file from
/// different URLs (basename collision between out-of-home facets)
/// must be rejected before any byte lands — a shared cache path with
/// divergent source bytes would be silent corruption.
#[test]
fn out_of_home_basename_collision_is_a_hard_error() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("root/dir-a");
    let b = tmp.path().join("root/dir-b");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    let mut rec: Vec<u8> = Vec::new();
    rec.extend_from_slice(&2i32.to_le_bytes());
    rec.extend_from_slice(&1.0f32.to_le_bytes());
    rec.extend_from_slice(&2.0f32.to_le_bytes());
    std::fs::write(a.join("vectors.fvecs"), &rec).unwrap();
    std::fs::write(b.join("vectors.fvecs"), &rec).unwrap();

    let server = TestServer::start(tmp.path()).unwrap();
    let base = server.base_url();
    let yaml = format!(
        r#"
_defaults:
  base_url: {base}/root/

ds-x:
  base: dir-a/vectors.fvecs
  query: dir-b/vectors.fvecs
  gt: dir-a/gt.ivecs
"#
    );
    std::fs::write(tmp.path().join("root/knn_entries.yaml"), &yaml).unwrap();
    init_test_cache();

    let sources = CatalogSources::new()
        .add_catalogs(&[format!("{base}/root/knn_entries.yaml")]);
    let catalog = Catalog::of(&sources);
    let group = catalog.open("ds-x").expect("dataset opens");
    let view = group.profile("default").expect("default profile");

    let msg = match view.open_facet_storage("base_vectors") {
        Ok(_) => panic!("colliding basenames must refuse to open"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("collision"), "unexpected error: {msg}");
}

/// `config catalog add` verification contract: a source is accepted
/// only when it parses AND its data endpoint answers a catalog ping.
/// A catalog whose YAML is fine but whose facet files are absent
/// (dead base_url, wrong prefix, decommissioned bucket) must be
/// refused before it is recorded.
#[test]
fn catalog_source_verification_requires_live_endpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let live = tmp.path().join("live");
    std::fs::create_dir_all(live.join("ds")).unwrap();
    let mut rec: Vec<u8> = Vec::new();
    rec.extend_from_slice(&2i32.to_le_bytes());
    rec.extend_from_slice(&1.0f32.to_le_bytes());
    rec.extend_from_slice(&2.0f32.to_le_bytes());
    std::fs::write(live.join("ds/base.fvecs"), &rec).unwrap();
    std::fs::write(live.join("ds/query.fvecs"), &rec).unwrap();
    let mut gt: Vec<u8> = Vec::new();
    gt.extend_from_slice(&1i32.to_le_bytes());
    gt.extend_from_slice(&0i32.to_le_bytes());
    std::fs::write(live.join("ds/gt.ivecs"), &gt).unwrap();

    let server = TestServer::start(tmp.path()).unwrap();
    let base = server.base_url();
    init_test_cache();

    // Healthy: document parses, facets answer.
    std::fs::write(live.join("knn_entries.yaml"), format!(
        r#"
_defaults:
  base_url: {base}/live/

good-ds:
  base: ds/base.fvecs
  query: ds/query.fvecs
  gt: ds/gt.ivecs
"#
    )).unwrap();
    let n = vectordata::config::verify_catalog_source(&format!("{base}/live/knn_entries.yaml"))
        .expect("live endpoint must verify");
    assert_eq!(n, 1);

    // Parses fine, but every facet 404s: must be refused.
    let dead = tmp.path().join("dead");
    std::fs::create_dir_all(&dead).unwrap();
    std::fs::write(dead.join("knn_entries.yaml"), format!(
        r#"
_defaults:
  base_url: {base}/nowhere/

ghost-ds:
  base: ds/base.fvecs
  query: ds/query.fvecs
  gt: ds/gt.ivecs
"#
    )).unwrap();
    let err = vectordata::config::verify_catalog_source(&format!("{base}/dead/knn_entries.yaml"))
        .expect_err("dead data endpoint must be refused");
    assert!(err.contains("catalog ping failed"), "unexpected refusal: {err}");

    // Unreachable / empty location: refused at the parse stage.
    let err = vectordata::config::verify_catalog_source(&format!("{base}/absent/"))
        .expect_err("missing catalog must be refused");
    assert!(err.contains("no datasets"), "unexpected refusal: {err}");
}
