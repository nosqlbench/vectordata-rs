// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **`--at` keeps the overlays in force exactly once.** A run's `--at`
//! names which catalogs to read; the program's and the user's overlays
//! still apply (`docs/design/srd-loose-files-overlays-integrity.md`
//! OV-2, OV-3). Naming several catalogs must not register a program
//! overlay once per name, which OV-2 refuses as two programs setting
//! one key.

use std::path::Path;

use vectordata::catalog::{Catalog, CatalogBuilder, CatalogSources};
use vectordata::datasets::at_sources;
use vectordata::overlay::{AttributeSource, Overlay, OverlayKind};
use vectordata::DistanceFunction;

/// A `knn_entries.yaml` catalog at `dir` listing `name`, which states no
/// distance function.
fn knn_catalog(dir: &Path, name: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("knn_entries.yaml"),
        format!("_defaults:\n  base_url: {}\n\"{name}:default\":\n  base: base.fvecs\n  query: query.fvecs\n  gt: gt.ivecs\n", dir.display()),
    )
    .unwrap();
}

/// A configuration directory naming two catalogs, `c1` holding `ds` and
/// `c2` holding `other`.
fn config_with_two_catalogs(root: &Path) -> String {
    let (c1, c2) = (root.join("c1"), root.join("c2"));
    knn_catalog(&c1, "ds");
    knn_catalog(&c2, "other");
    let home = root.join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("catalogs.yaml"), format!("c1: {}\nc2: {}\n", c1.display(), c2.display())).unwrap();
    home.to_string_lossy().into_owned()
}

/// The program: a built catalog `fvec` and an overlay stating `ds`'s metric.
fn program() -> CatalogSources {
    CatalogSources::new()
        .with(CatalogBuilder::named("fvec").build())
        .with_overlay(Overlay::program("fvec", "datasets:\n  ds:\n    distance_function: COSINE\n").unwrap())
        .unwrap()
}

/// **Two names, one program overlay**: the set `at_sources` returns
/// carries the overlay once, reports nothing, and the catalog it opens
/// still reads the overlaid value, sourced to the program.
#[test]
fn at_sources_with_two_names_carries_a_program_overlay_once() {
    let tmp = tempfile::tempdir().unwrap();
    let config = config_with_two_catalogs(tmp.path());
    let candidates = CatalogSources::new().configure(&config).extend(program());
    assert!(candidates.diagnostics().is_empty(), "{:?}", candidates.diagnostics());

    let sources = at_sources(&["fvec".into(), "c1".into(), "c2".into()], &candidates).unwrap();
    let names: Vec<&str> = sources.iter().map(|(s, _)| s.name()).collect();
    assert_eq!(names, ["fvec", "c1", "c2"]);
    assert!(sources.diagnostics().is_empty(), "{:?}", sources.diagnostics());
    let programs = sources.overlays().iter().filter(|o| matches!(o.kind(), OverlayKind::Program(_))).count();
    assert_eq!(programs, 1, "the program overlay once");

    let catalog = Catalog::of(&sources);
    assert!(catalog.diagnostics().is_empty(), "{:?}", catalog.diagnostics());
    let ds = catalog.find_exact("ds").unwrap();
    assert_eq!(ds.attributes().and_then(|a| a.distance()), Some(DistanceFunction::Cosine));
    assert!(matches!(ds.attribute_source("distance_function"), Some(AttributeSource::Overlay { kind: OverlayKind::Program(ref p), .. }) if p == "fvec"));
}

/// **A location alone keeps the overlays too**: `--at <path>` names
/// which catalog to read, not what the program knows of datasets.
#[test]
fn at_sources_with_a_location_keeps_the_overlays() {
    let tmp = tempfile::tempdir().unwrap();
    let config = config_with_two_catalogs(tmp.path());
    let candidates = CatalogSources::new().configure(&config).extend(program());
    let c1 = tmp.path().join("c1").to_string_lossy().into_owned();

    let sources = at_sources(&[c1], &candidates).unwrap();
    assert!(sources.diagnostics().is_empty(), "{:?}", sources.diagnostics());
    let catalog = Catalog::of(&sources);
    let ds = catalog.find_exact("ds").unwrap();
    assert_eq!(ds.attributes().and_then(|a| a.distance()), Some(DistanceFunction::Cosine));
}

/// **Two programs setting one key are still refused** (OV-2), whether
/// registered on one set or brought together by `extend`, and the same
/// overlay registered twice is two registrations.
#[test]
fn two_program_overlays_setting_one_key_are_still_refused() {
    let other = || Overlay::program("other", "datasets:\n  ds:\n    distance_function: L2\n").unwrap();

    let err = program().with_overlay(other()).unwrap_err();
    assert!(err.contains("both set 'distance_function' for 'ds'"), "{err}");

    let joined = program().extend(CatalogSources::new().with_overlay(other()).unwrap());
    let said: Vec<String> = joined.diagnostics().iter().map(|d| d.to_string()).collect();
    assert!(said.iter().any(|d| d.contains("both set 'distance_function' for 'ds'")), "{said:?}");

    let again = Overlay::program("fvec", "datasets:\n  ds:\n    distance_function: COSINE\n").unwrap();
    assert!(program().with_overlay(again).is_err(), "registering an overlay twice is refused");
}
