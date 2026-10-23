// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **`veks datasets cache-status` reads the cache active for it**: its
//! `--cache-dir` when given, else the dataset's cache — the one its
//! catalog names (a jvector catalog's `cache_dir`), else the configured
//! one. Each run is a child process with its own `VECTORDATA_HOME`.

mod support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use support::testserver::TestServer;

fn make_tempdir() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

fn veks(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_veks"))
        .args(args)
        .env("VECTORDATA_HOME", home)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .env_remove("VECTORDATA_OFFLINE")
        .env_remove("DATASET_CACHE_DIR")
        .output()
        .unwrap()
}

fn ok(out: &Output) -> String {
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{text}");
    text
}

/// The `Cache dir:` line a status prints (relative to the working
/// directory when under it).
fn cache_dir_line(text: &str) -> String {
    text.lines().find(|l| l.trim_start().starts_with("Cache dir:")).unwrap_or_else(|| panic!("{text}")).to_string()
}

#[test]
fn cache_status_reads_the_catalogs_cache_or_the_one_given() {
    let tmp = make_tempdir();
    let served = tmp.path().join("served");
    std::fs::create_dir_all(&served).unwrap();
    let mut fvec = Vec::new();
    for i in 0..20u32 {
        fvec.extend_from_slice(&2i32.to_le_bytes());
        fvec.extend_from_slice(&(i as f32).to_le_bytes());
        fvec.extend_from_slice(&(i as f32 + 0.5).to_le_bytes());
    }
    std::fs::write(served.join("base.fvec"), &fvec).unwrap();
    let server = TestServer::start(&served).unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", home.join("cache").display())).unwrap();
    let named = tmp.path().join("named");
    let catalog = tmp.path().join("jv.yaml");
    std::fs::write(
        &catalog,
        format!("_defaults:\n  base_url: {}/\n  cache_dir: {}\njv:\n  base: base.fvec\n", server.base_url(), named.display()),
    )
    .unwrap();
    let at = catalog.to_str().unwrap();
    ok(&veks(&home, &["datasets", "precache", "--at", at, "--dataset", "jv:default"]));
    assert!(named.join("jv/base.fvec").is_file());

    // The dataset's cache: the one its catalog names.
    let status = ok(&veks(&home, &["datasets", "cache-status", "-d", "jv", "--at", at]));
    assert!(cache_dir_line(&status).ends_with(&format!("{}", Path::new("named").join("jv").display())), "{status}");
    assert!(!status.contains("Not cached"), "{status}");

    // A cache given for the command is the one read.
    let other = tmp.path().join("other");
    let status = ok(&veks(&home, &["datasets", "cache-status", "-d", "jv", "--at", at, "--cache-dir", other.to_str().unwrap()]));
    assert!(cache_dir_line(&status).ends_with(&format!("{}", Path::new("other").join("jv").display())), "{status}");
    assert!(status.contains("Not cached"), "{status}");

    // `--all` covers the catalog's cache as well as the configured one.
    let all = ok(&veks(&home, &["datasets", "cache-status", "--all", "--at", at]));
    assert!(all.contains("Cache status: jv"), "{all}");
}
