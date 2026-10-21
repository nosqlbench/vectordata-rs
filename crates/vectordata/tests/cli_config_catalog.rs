// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `config catalog add` and `remove`, run as the binary a user runs,
//! from the directory a relative path is relative to: a path is saved
//! as the directory it names, under that directory's name, so it
//! resolves from anywhere and `remove` finds it by either spelling.

mod support;

use std::path::Path;
use std::process::{Command, Output};

use support::fixtures::{write_catalogs, write_fvec};

/// A `VECTORDATA_HOME` whose settings name a cache inside it.
fn home(root: &Path) -> std::path::PathBuf {
    let home = root.join("home");
    std::fs::create_dir_all(home.join("cache")).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", home.join("cache").display())).unwrap();
    home
}

/// A local catalog at `root` holding one dataset, `name`.
fn catalog_with(root: &Path, name: &str) {
    let ds = root.join(name);
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 10, 4, 1.0);
    std::fs::write(ds.join("dataset.yaml"), format!("name: {name}\nprofiles:\n  default:\n    base_vectors: base.fvec\n"))
        .unwrap();
    write_catalogs(root, &[name]);
}

fn vectordata(home: &Path, cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(args)
        .current_dir(cwd)
        .env("VECTORDATA_HOME", home)
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .output()
        .expect("run vectordata")
}

fn catalogs_yaml(home: &Path) -> String {
    std::fs::read_to_string(home.join("catalogs.yaml")).unwrap_or_default()
}

/// **`config catalog add .` saves `<dirname>: /abs/path`.** It saved
/// `0: .` — relative, so it resolved only from that directory, and
/// named by index, as `.` slugs to nothing.
#[test]
fn add_saves_a_relative_path_absolute_and_named_after_its_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let home = home(tmp.path());
    let stage = tmp.path().join("stage");
    catalog_with(&stage, "demo");
    let abs = stage.to_string_lossy().into_owned();

    let out = vectordata(&home, &stage, &["config", "catalog", "add", "."]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(catalogs_yaml(&home).trim(), format!("stage: {abs}"));

    // The same directory again, by its absolute spelling, is the one
    // already configured.
    let out = vectordata(&home, tmp.path(), &["config", "catalog", "add", &abs]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Already configured"));
    assert_eq!(catalogs_yaml(&home).trim(), format!("stage: {abs}"));

    // A relative path through a parent resolves to the directory too.
    let out = vectordata(&home, &stage.join("demo"), &["config", "catalog", "remove", ".."]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(catalogs_yaml(&home).trim(), "");
}

/// **`config catalog remove` finds a path by either spelling**: the
/// relative one it was added by, or the absolute one it was saved as.
#[test]
fn remove_finds_a_path_by_its_relative_or_absolute_spelling() {
    let tmp = tempfile::tempdir().unwrap();
    let home = home(tmp.path());
    let stage = tmp.path().join("stage");
    catalog_with(&stage, "demo");
    let abs = stage.to_string_lossy().into_owned();

    for remove_by in [".", abs.as_str()] {
        let out = vectordata(&home, &stage, &["config", "catalog", "add", "."]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let out = vectordata(&home, &stage, &["config", "catalog", "remove", remove_by]);
        assert!(out.status.success(), "remove {remove_by}: {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(catalogs_yaml(&home).trim(), "", "remove {remove_by}");
    }
}
