// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **A jvector catalog's `cache_dir` is the dataset's cache**, managed
//! as any cache is, and the commands act on the cache that is active
//! for them: a command's `--cache-dir` first, then the cache a dataset's
//! catalog names, then the configured one.
//!
//! Each run is a child process with its own `VECTORDATA_HOME`, so the
//! configured cache is the test's, and no process-wide cache override
//! (which outranks a catalog's) is in force.

mod support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use support::fixtures::write_fvec;
use support::testserver::TestServer;

fn make_tmp() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// A served `base.fvec`, a configuration home whose configured cache is
/// `<tmp>/home/cache`, and a jvector catalog listing `jv`, whose
/// `_defaults.cache_dir` is `<tmp>/named`: `(server, home, catalog file,
/// served file)`.
fn setup(tmp: &Path) -> (TestServer, PathBuf, PathBuf, PathBuf) {
    let served = tmp.join("served");
    std::fs::create_dir_all(&served).unwrap();
    write_fvec(&served.join("base.fvec"), 50, 4, 1.0);
    let server = TestServer::start(&served).unwrap();
    let home = tmp.join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", home.join("cache").display())).unwrap();
    let catalog = tmp.join("jv.yaml");
    std::fs::write(
        &catalog,
        format!(
            "_defaults:\n  base_url: {}/\n  cache_dir: {}\njv:\n  base: base.fvec\n",
            server.base_url(),
            tmp.join("named").display()
        ),
    )
    .unwrap();
    (server, home, catalog, served.join("base.fvec"))
}

fn vectordata(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(args)
        .env("VECTORDATA_HOME", home)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .env_remove(vectordata::catalog::knn_entries::DATASET_CACHE_DIR_ENV)
        .output()
        .unwrap()
}

fn ok(out: &Output) -> String {
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{text}");
    text
}

/// **A fetch lands in the catalog's cache**, `<cache_dir>/<dataset>/`,
/// not in the configured cache.
#[test]
fn a_fetch_lands_in_the_cache_its_catalog_names() {
    let tmp = make_tmp();
    let (_server, home, catalog, served) = setup(tmp.path());
    ok(&vectordata(&home, &["datasets", "precache", "--at", catalog.to_str().unwrap(), "jv:default"]));

    let kept = tmp.path().join("named/jv/base.fvec");
    assert_eq!(std::fs::read(&kept).unwrap_or_default(), std::fs::read(&served).unwrap(), "{}", kept.display());
    assert!(!home.join("cache/jv").exists(), "the configured cache holds a copy");
}

/// **A command's `--cache-dir` outranks the catalog's**: the cache a
/// user names for the run is the one the run uses.
#[test]
fn a_commands_cache_dir_outranks_the_catalogs() {
    let tmp = make_tmp();
    let (_server, home, catalog, served) = setup(tmp.path());
    let explicit = tmp.path().join("explicit");
    ok(&vectordata(
        &home,
        &["datasets", "precache", "--at", catalog.to_str().unwrap(), "--cache-dir", explicit.to_str().unwrap(), "jv:default"],
    ));

    let kept = explicit.join("jv/base.fvec");
    assert_eq!(std::fs::read(&kept).unwrap_or_default(), std::fs::read(&served).unwrap(), "{}", kept.display());
    assert!(!tmp.path().join("named/jv").exists(), "the catalog's cache was used over --cache-dir");
}

/// **The cache commands act on the cache named for them**: `cache list
/// --cache-dir` lists what the catalog's cache holds, and `cache prune
/// --cache-dir` removes it from there.
#[test]
fn the_cache_commands_act_on_the_cache_given_them() {
    let tmp = make_tmp();
    let (_server, home, catalog, _) = setup(tmp.path());
    ok(&vectordata(&home, &["datasets", "precache", "--at", catalog.to_str().unwrap(), "jv:default"]));
    let named = tmp.path().join("named");

    let listed = ok(&vectordata(&home, &["cache", "list", "--cache-dir", named.to_str().unwrap()]));
    assert!(listed.contains("jv"), "{listed}");
    let configured = ok(&vectordata(&home, &["cache", "list"]));
    assert!(!configured.lines().any(|l| l.split_whitespace().next() == Some("jv")), "{configured}");

    ok(&vectordata(&home, &["cache", "prune", "--dataset", "jv", "--cache-dir", named.to_str().unwrap()]));
    assert!(!named.join("jv").exists(), "prune left {}", named.join("jv").display());
}
