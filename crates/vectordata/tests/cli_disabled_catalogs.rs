// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `disabled_catalogs` in `settings.yaml` means one thing everywhere:
//! a disabled catalog is not assembled with the configured ones by any
//! command or host program, a catalog named for the run (`--at <name>`,
//! `--catalog <name>`) is read all the same, numbered shortcuts count
//! the catalogs as `catalogs.yaml` writes them, `config catalog list`
//! marks it, and `config catalog disable`/`enable` set it.
//!
//! Each binary run gets `VECTORDATA_HOME` in its own environment; this
//! process's environment is never changed.

#![cfg(feature = "cli")]

mod support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use support::fixtures::{write_catalogs, write_fvec};
use vectordata::Location;
use vectordata::catalog::{CatalogBuilder, CatalogSources};
use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};

/// A local catalog at `root` holding the datasets `names`.
fn catalog_with(root: &Path, names: &[&str]) {
    for name in names {
        let ds = root.join(name);
        std::fs::create_dir_all(&ds).unwrap();
        write_fvec(&ds.join("base.fvec"), 10, 4, 1.0);
        std::fs::write(ds.join("dataset.yaml"), format!("name: {name}\nprofiles:\n  default:\n    base_vectors: base.fvec\n"))
            .unwrap();
    }
    write_catalogs(root, names);
}

/// A config directory under `root` configuring `c1` (holding `ds`) and
/// `c2` (holding `ds` and `ds2`), in that order, with `settings` as its
/// `settings.yaml` after a `cache_dir` line.
fn config(root: &Path, settings: &str) -> PathBuf {
    let home = root.join("home");
    std::fs::create_dir_all(home.join("cache")).unwrap();
    catalog_with(&root.join("c1"), &["ds"]);
    catalog_with(&root.join("c2"), &["ds", "ds2"]);
    std::fs::write(
        home.join("catalogs.yaml"),
        format!("c1: {}\nc2: {}\n", root.join("c1").display(), root.join("c2").display()),
    )
    .unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n{settings}", home.join("cache").display())).unwrap();
    home
}

fn vectordata(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(args)
        .env("VECTORDATA_HOME", home)
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .output()
        .expect("run vectordata")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Both streams of `out`, for a failure message.
fn both(out: &Output) -> String {
    format!("stdout:\n{}\nstderr:\n{}", text(&out.stdout), text(&out.stderr))
}

/// **A disabled catalog's datasets are not listed and do not
/// describe.** `ds2` is in `c2` alone; with `c2` disabled, `datasets
/// list` leaves it out and `describe ds2` does not find it.
#[test]
fn a_disabled_catalogs_datasets_are_absent() {
    let tmp = tempfile::tempdir().unwrap();
    let home = config(tmp.path(), "disabled_catalogs: c2\n");

    let out = vectordata(&home, &["datasets", "list"]);
    assert!(out.status.success(), "{}", both(&out));
    let listed = text(&out.stdout);
    assert!(listed.contains("ds"), "{}", both(&out));
    assert!(!listed.contains("ds2"), "a disabled catalog's dataset was listed: {}", both(&out));

    let out = vectordata(&home, &["datasets", "describe", "ds2"]);
    assert_eq!(out.status.code(), Some(1), "{}", both(&out));
    assert!(text(&out.stderr).contains("Dataset 'ds2' not found"), "{}", both(&out));
}

/// **A dataset in an enabled and a disabled catalog is not ambiguous.**
/// `ds` is in both `c1` and `c2`; with `c2` disabled it describes from
/// `c1`, with no "name the catalog" refusal.
#[test]
fn a_dataset_shared_with_a_disabled_catalog_is_not_ambiguous() {
    let tmp = tempfile::tempdir().unwrap();
    let home = config(tmp.path(), "disabled_catalogs: c2\n");

    let out = vectordata(&home, &["datasets", "describe", "ds"]);
    assert!(out.status.success(), "{}", both(&out));
    assert!(!text(&out.stderr).contains("--at <catalog>"), "{}", both(&out));
    assert!(text(&out.stdout).contains("Dataset: ds"), "{}", both(&out));
}

/// **A catalog named for the run is read though disabled**: `--at c2`
/// and `--catalog c2` both find `ds2`, and `--catalog c2` adds it to the
/// enabled ones rather than replacing them.
#[test]
fn a_disabled_catalog_named_for_the_run_is_read() {
    let tmp = tempfile::tempdir().unwrap();
    let home = config(tmp.path(), "disabled_catalogs: c2\n");

    let out = vectordata(&home, &["datasets", "describe", "ds2", "--at", "c2"]);
    assert!(out.status.success(), "--at c2: {}", both(&out));
    assert!(text(&out.stdout).contains("Dataset: ds2"), "{}", both(&out));

    let out = vectordata(&home, &["datasets", "list", "--catalog", "c2"]);
    assert!(out.status.success(), "--catalog c2: {}", both(&out));
    assert!(text(&out.stdout).contains("ds2"), "{}", both(&out));

    // `--catalog` adds c2 to c1, so `ds` is in two catalogs again.
    let out = vectordata(&home, &["datasets", "describe", "ds", "--catalog", "c2"]);
    assert_eq!(out.status.code(), Some(1), "{}", both(&out));
    assert!(text(&out.stderr).contains("--at <catalog>"), "{}", both(&out));
}

/// **`--at <number>` counts every catalog `catalogs.yaml` writes.**
/// `--at 2` is `c2` whether `c1` is disabled or not, and whether `c2`
/// itself is.
#[test]
fn a_numbered_catalog_is_the_same_whatever_is_disabled() {
    for settings in ["", "disabled_catalogs: c1\n", "disabled_catalogs: c2\n", "disabled_catalogs: c1, c2\n"] {
        let tmp = tempfile::tempdir().unwrap();
        let home = config(tmp.path(), settings);
        let out = vectordata(&home, &["datasets", "describe", "ds2", "--at", "2"]);
        assert!(out.status.success(), "with {settings:?}: {}", both(&out));
        let out = vectordata(&home, &["datasets", "describe", "ds2", "--at", "1"]);
        assert_eq!(out.status.code(), Some(1), "with {settings:?}, #1 is c1, which has no ds2: {}", both(&out));
    }
}

/// **`config catalog list` shows a disabled catalog, marked.**
#[test]
fn config_catalog_list_marks_a_disabled_catalog() {
    let tmp = tempfile::tempdir().unwrap();
    let home = config(tmp.path(), "disabled_catalogs: c2\n");

    let out = vectordata(&home, &["config", "catalog", "list"]);
    assert!(out.status.success(), "{}", both(&out));
    let stdout = text(&out.stdout);
    let line = |name: &str| {
        stdout
            .lines()
            .find(|l| l.split_whitespace().nth(1) == Some(name))
            .unwrap_or_else(|| panic!("no line for {name}: {stdout}"))
            .to_string()
    };
    assert!(line("c1").trim_start().starts_with("1."), "{stdout}");
    assert!(!line("c1").contains("(disabled)"), "{stdout}");
    assert!(line("c2").trim_start().starts_with("2."), "{stdout}");
    assert!(line("c2").contains("(disabled)"), "{stdout}");
}

/// **`config catalog disable` and `enable` edit `disabled_catalogs`,
/// keeping every comment**: a disable is seen by the next command,
/// disabling again says so and succeeds, enable undoes it, and an
/// unknown name is refused naming the configured ones.
#[test]
fn config_catalog_disable_and_enable_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let home = config(tmp.path(), "# my settings\nprotect_settings: true\n# end\n");
    let settings = || std::fs::read_to_string(home.join("settings.yaml")).unwrap();

    let out = vectordata(&home, &["config", "catalog", "disable", "c2"]);
    assert!(out.status.success(), "{}", both(&out));
    assert!(settings().contains("disabled_catalogs: c2"), "{}", settings());
    let out = vectordata(&home, &["datasets", "describe", "ds2"]);
    assert_eq!(out.status.code(), Some(1), "disabled: {}", both(&out));

    let out = vectordata(&home, &["config", "catalog", "disable", "c2"]);
    assert!(out.status.success(), "{}", both(&out));
    assert!(text(&out.stdout).contains("already disabled"), "{}", both(&out));

    let out = vectordata(&home, &["config", "catalog", "disable", "c1"]);
    assert!(out.status.success(), "{}", both(&out));
    assert!(settings().contains("disabled_catalogs: c1, c2"), "{}", settings());

    let out = vectordata(&home, &["config", "catalog", "enable", "c2"]);
    assert!(out.status.success(), "{}", both(&out));
    assert!(settings().contains("disabled_catalogs: c1\n"), "{}", settings());
    let out = vectordata(&home, &["datasets", "describe", "ds2"]);
    assert!(out.status.success(), "enabled again: {}", both(&out));

    let out = vectordata(&home, &["config", "catalog", "enable", "c2"]);
    assert!(out.status.success(), "{}", both(&out));
    assert!(text(&out.stdout).contains("not disabled"), "{}", both(&out));

    for comment in ["# my settings", "protect_settings: true", "# end"] {
        assert!(settings().contains(comment), "lost {comment:?}: {}", settings());
    }

    let before = settings();
    for verb in ["disable", "enable"] {
        let out = vectordata(&home, &["config", "catalog", verb, "c3"]);
        assert_eq!(out.status.code(), Some(1), "{}", both(&out));
        let stderr = text(&out.stderr);
        assert!(stderr.contains("'c3'") && stderr.contains("c1") && stderr.contains("c2"), "{}", both(&out));
    }
    assert_eq!(settings(), before, "a refused name writes nothing");
}

/// A program catalog `bench` holding `alpha`.
fn bench(dir: &Path) -> CatalogSources {
    let home = dir.join("alpha");
    std::fs::create_dir_all(&home).unwrap();
    write_fvec(&home.join("base.fvec"), 3, 2, 1.0);
    let description = DatasetConfig::builder("alpha")
        .profile("default", |p| p.facet(StandardFacet::BaseVectors, "base.fvec"))
        .build()
        .unwrap();
    let entry = CatalogEntry::builder("alpha")
        .home(&Location::from_path(&home))
        .dataset(description)
        .build()
        .unwrap();
    CatalogSources::new().with(CatalogBuilder::named("bench").entry(entry).build())
}

fn words(w: &[&str]) -> Vec<String> {
    w.iter().map(|s| s.to_string()).collect()
}

/// **A host program does not see a disabled configured catalog, and
/// its own catalog is disabled only by name.** Run in-process with
/// `shell::run_with_catalogs` over `--configdir`, whose `settings.yaml`
/// is the one its `catalogs.yaml` is read with.
#[test]
fn a_host_program_does_not_see_a_disabled_configured_catalog() {
    let tmp = tempfile::tempdir().unwrap();
    let home = config(tmp.path(), "disabled_catalogs: c2\n");
    let configdir = home.to_string_lossy().into_owned();
    let program = bench(tmp.path());
    let run = |args: &[&str]| {
        let mut argv = words(args);
        argv.extend(words(&["--configdir", &configdir]));
        vectordata::shell::run_with_catalogs("bench", argv, program.clone())
    };

    assert_eq!(run(&["datasets", "describe", "ds2"]), 1, "c2 is disabled");
    assert_eq!(run(&["datasets", "describe", "ds"]), 0, "c1 alone has ds");
    assert_eq!(run(&["datasets", "describe", "alpha"]), 0, "the program's catalog is not disabled");
    assert_eq!(run(&["datasets", "describe", "ds2", "--at", "c2"]), 0, "named for the run");

    std::fs::write(
        home.join("settings.yaml"),
        format!("cache_dir: {}\ndisabled_catalogs: c2, bench\n", home.join("cache").display()),
    )
    .unwrap();
    assert_eq!(run(&["datasets", "describe", "alpha"]), 1, "the program's catalog, disabled by name");
    assert_eq!(run(&["datasets", "describe", "alpha", "--at", "bench"]), 0, "and named for the run");
}
