// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata datasets precache a b …` precaches several specs as one
//! run: one plan, one progress display, one closing report, and every
//! spec's bytes in the cache.

mod support;

use std::path::PathBuf;
use std::process::Command;

use support::fixtures::{write_catalogs, write_fvec, write_mref};
use support::testserver::TestServer;

fn make_tmp() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// **Two specs, one report.** Both datasets are served, each fetched
/// in full into the cache, under one heading, one progress display
/// (one fetch counting both facets, one total over both) and one `done`
/// line; each facet is named by its dataset and profile.
#[test]
fn several_specs_precache_as_one_run() {
    let tmp = make_tmp();
    let served = tmp.path().join("served");
    for (name, seed) in [("first", 1.0e3), ("second", 2.0e3)] {
        let ds = served.join(name);
        std::fs::create_dir_all(&ds).unwrap();
        write_fvec(&ds.join("base.fvec"), 300, 8, seed);
        write_mref(&ds.join("base.fvec"));
        std::fs::write(
            ds.join("dataset.yaml"),
            format!("name: {name}\nprofiles:\n  default:\n    base_vectors: base.fvec\n"),
        )
        .unwrap();
    }
    write_catalogs(&served, &["first", "second"]);
    let server = TestServer::start(&served).unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let cache = home.join("cache");
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", cache.display())).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(["datasets", "precache", "--at", &server.base_url(), "first:default", "second:default"])
        .env("VECTORDATA_HOME", &home)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{}\n{err}", String::from_utf8_lossy(&out.stdout));

    let precache: Vec<&str> = err.lines().filter(|l| l.starts_with("Precache")).collect();
    // One display: one fetch begun, its facets counted across both
    // specs, and one total running to 100% of both.
    let fetching: Vec<&&str> = precache.iter().filter(|l| l.starts_with("Precache: fetching ")).collect();
    assert_eq!(fetching.len(), 1, "one progress display:\n{err}");
    assert!(fetching[0].starts_with("Precache: fetching 2 facet(s)"), "{err}");
    assert!(precache.iter().any(|l| l.contains("[1/2] first:default/base_vectors: started")), "{err}");
    assert!(precache.iter().any(|l| l.contains("[2/2] second:default/base_vectors: started")), "{err}");
    assert!(precache.iter().any(|l| l.ends_with("total 100% (21.1 KiB/21.1 KiB)")), "{err}");
    let done: Vec<&&str> = precache.iter().filter(|l| l.starts_with("Precache done:")).collect();
    assert_eq!(done.len(), 1, "one report:\n{err}");
    assert!(done[0].starts_with("Precache done: 2 facet(s)"), "{err}");
    assert_eq!(err.matches("Prebuffering 2 facet(s) across 2 profiles").count(), 1, "one plan:\n{err}");

    // Every byte of both, in the cache.
    for name in ["first", "second"] {
        let cached = cache.join(name).join("base.fvec");
        let served = std::fs::read(served.join(name).join("base.fvec")).unwrap();
        assert_eq!(std::fs::read(&cached).unwrap_or_default(), served, "{}", cached.display());
    }
}

/// **A spec that fails stops the run before anything is fetched**: the
/// second spec names no dataset, so the first is not downloaded either.
#[test]
fn a_bad_spec_among_several_fetches_nothing() {
    let tmp = make_tmp();
    let served = tmp.path().join("served");
    let ds = served.join("first");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 300, 8, 1.0e3);
    write_mref(&ds.join("base.fvec"));
    std::fs::write(ds.join("dataset.yaml"), "name: first\nprofiles:\n  default:\n    base_vectors: base.fvec\n").unwrap();
    write_catalogs(&served, &["first"]);
    let server = TestServer::start(&served).unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let cache = home.join("cache");
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", cache.display())).unwrap();

    let run = |specs: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_vectordata"))
            .args(["datasets", "precache", "--at", &server.base_url()])
            .args(specs)
            .env("VECTORDATA_HOME", &home)
            .env("VECTORDATA_NO_UPDATE_CHECK", "1")
            .env_remove(vectordata::settings::OFFLINE_ENV)
            .output()
            .unwrap()
    };
    let out = run(&["first:default", "nowhere:default"]);
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!cache.join("first").join("base.fvec").exists(), "nothing fetched");

    let out = run(&["first:default", "first"]);
    assert_eq!(out.status.code(), Some(2), "a bare spec is refused: {}", String::from_utf8_lossy(&out.stderr));
    assert!(!cache.join("first").join("base.fvec").exists(), "nothing fetched");

    let out = run(&[]);
    assert_eq!(out.status.code(), Some(2), "no spec at all: {}", String::from_utf8_lossy(&out.stderr));
}
