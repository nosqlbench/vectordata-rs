// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata datasets precache` with standard error going to a file
//! or a pipe reports its progress as log lines: whole lines, no
//! carriage-return redraws or terminal controls, ending at 100%.

mod support;

use std::path::PathBuf;
use std::process::Command;

use support::fixtures::{write_fvec, write_mref};
use support::testserver::TestServer;

fn make_tmp() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

#[test]
fn precache_into_a_pipe_writes_log_lines() {
    let tmp = make_tmp();
    let served = tmp.path().join("served");
    let ds = served.join("logged");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 400, 8, 3.1e8);
    write_mref(&ds.join("base.fvec"));
    std::fs::write(ds.join("dataset.yaml"), "name: logged\nprofiles:\n  default:\n    base_vectors: base.fvec\n").unwrap();
    std::fs::write(
        served.join("catalog.json"),
        r#"[{"name":"logged","path":"logged/dataset.yaml","dataset_type":"dataset.yaml","layout":{"profiles":{"default":{}}}}]"#,
    )
    .unwrap();
    let server = TestServer::start(&served).unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", home.join("cache").display())).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(["datasets", "precache", "--at", &server.base_url(), "logged:default"])
        .env("VECTORDATA_HOME", &home)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{}\n{err}", String::from_utf8_lossy(&out.stdout));
    assert!(!err.contains('\r') && !err.contains('\u{1b}'), "no in-place redraws in a pipe:\n{err:?}");
    let lines: Vec<&str> = err.lines().filter(|l| l.starts_with("Precache")).collect();
    assert!(lines.iter().any(|l| l.contains(": started, ")), "{err}");
    assert!(lines.iter().any(|l| l.contains("total 100%")), "{err}");
    assert!(lines.last().is_some_and(|l| l.starts_with("Precache done: 1 facet(s)")), "{err}");
}
