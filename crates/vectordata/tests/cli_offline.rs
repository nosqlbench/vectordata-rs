// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `vectordata` commands that show or set offline behaviour, run as
//! the binary a user runs, each in its own `VECTORDATA_HOME`.

use std::path::Path;
use std::process::{Command, Output};

fn make_tmp() -> tempfile::TempDir {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// A `VECTORDATA_HOME` whose settings name a cache inside it, with a
/// comment of the user's own to be kept.
fn home(root: &Path) -> std::path::PathBuf {
    let home = root.join("home");
    std::fs::create_dir_all(home.join("cache")).unwrap();
    std::fs::write(
        home.join("settings.yaml"),
        format!("# my own note\ncache_dir: {}\n", home.join("cache").display()),
    )
    .unwrap();
    home
}

/// Run `vectordata` with `args` in `home`, with `env` set and offline
/// mode left to the settings unless `env` says otherwise.
fn vectordata(home: &Path, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vectordata"));
    cmd.args(args)
        .env("VECTORDATA_HOME", home)
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().expect("run vectordata")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// **`config set offline` persists the switch without touching the rest
/// of the file, and `config get offline` reports what is in effect** —
/// naming `VECTORDATA_OFFLINE` only when it is what decided.
#[test]
fn config_sets_and_reports_offline_mode() {
    let tmp = make_tmp();
    let home = home(tmp.path());

    let set = vectordata(&home, &["config", "set", "offline", "on"], &[]);
    assert!(set.status.success(), "{}", stderr(&set));
    assert!(stdout(&set).starts_with("offline = on\nSaved to "), "{}", stdout(&set));
    let settings = std::fs::read_to_string(home.join("settings.yaml")).unwrap();
    assert!(settings.contains("# my own note") && settings.contains("cache_dir:"), "{settings}");
    assert!(settings.lines().any(|l| l.trim() == "offline: on"), "{settings}");

    let get = |env: &[(&str, &str)]| stdout(&vectordata(&home, &["config", "get", "offline"], env));
    assert_eq!(get(&[]), "on\n");
    assert_eq!(get(&[("VECTORDATA_OFFLINE", "0")]), "off (from VECTORDATA_OFFLINE=0)\n");
    assert_eq!(get(&[("VECTORDATA_OFFLINE", "maybe")]), "on\n", "an unrecognized value decides nothing");
    assert!(stdout(&vectordata(&home, &["config", "get"], &[])).contains("offline: on"));

    let bad = vectordata(&home, &["config", "set", "offline", "sometimes"], &[]);
    assert!(!bad.status.success());
    assert!(stderr(&bad).contains("invalid offline value 'sometimes'"), "{}", stderr(&bad));
    assert_eq!(std::fs::read_to_string(home.join("settings.yaml")).unwrap(), settings, "unchanged");

    let off = vectordata(&home, &["config", "set", "offline", "off"], &[("VECTORDATA_OFFLINE", "1")]);
    assert!(off.status.success());
    assert!(stdout(&off).contains("VECTORDATA_OFFLINE is set in this environment and takes precedence"), "{}", stdout(&off));
    assert_eq!(get(&[]), "off\n");
}
