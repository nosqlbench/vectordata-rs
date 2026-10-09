// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `vectordata` commands that show or set offline behaviour, run as
//! the binary a user runs, each in its own `VECTORDATA_HOME`: `config
//! get/set offline`, `datasets ping` on a cached dataset whose server is
//! gone or not to be asked, and the kept catalogs in `cache list`.

mod support;

use std::path::Path;
use std::process::{Command, Output};

use support::fixtures::{write_fvec, write_mref};
use support::testserver::TestServer;

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

/// Serve `name`, with two facets, from a remote catalog under `root`.
fn served_catalog(root: &Path, name: &str, seed: f32) {
    let ds = root.join(name);
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 300, 4, seed);
    write_mref(&ds.join("base.fvec"));
    write_fvec(&ds.join("query.fvec"), 20, 4, seed + 0.5);
    std::fs::write(
        ds.join("dataset.yaml"),
        format!("name: {name}\nprofiles:\n  default:\n    base_vectors: base.fvec\n    query_vectors: query.fvec\n"),
    )
    .unwrap();
    std::fs::write(
        root.join("catalog.json"),
        format!(
            r#"[{{"name":"{name}","path":"{name}/dataset.yaml","dataset_type":"dataset.yaml","layout":{{"profiles":{{"default":{{}}}}}}}}]"#
        ),
    )
    .unwrap();
}

/// **`datasets ping` on a cached dataset says why it could not check**,
/// and fails: with the server gone, each complete facet is reported
/// unreachable but still readable; in offline mode it is reported not
/// checked, with no request made. The catalog loads from its kept copy
/// either way, and `cache list` shows that copy in its own section.
#[test]
fn ping_and_cache_list_report_a_cached_dataset_whose_server_is_gone() {
    let tmp = make_tmp();
    let home = home(tmp.path());
    let served = tmp.path().join("served");
    served_catalog(&served, "pinged", 3.0e8);
    let server = TestServer::start(&served).unwrap();
    let at = server.base_url();

    let precache = vectordata(&home, &["datasets", "precache", "--at", &at, "pinged:default"], &[]);
    assert!(precache.status.success(), "{}\n{}", stdout(&precache), stderr(&precache));
    let online = vectordata(&home, &["datasets", "ping", "--at", &at, "pinged"], &[]);
    assert!(online.status.success(), "{}\n{}", stdout(&online), stderr(&online));

    let before = server.accepted_connections();
    let offline = vectordata(&home, &["datasets", "ping", "--at", &at, "pinged"], &[("VECTORDATA_OFFLINE", "1")]);
    let out = stdout(&offline);
    assert!(!offline.status.success(), "{out}");
    assert_eq!(out.matches("NOT CHECKED: offline mode is on (a complete copy is cached)").count(), 2, "{out}");
    assert_eq!(server.accepted_connections(), before, "offline ping contacted the server");

    drop(server);
    let gone = vectordata(&home, &["datasets", "ping", "--at", &at, "pinged"], &[]);
    let out = stdout(&gone);
    assert!(!gone.status.success(), "{out}");
    assert_eq!(
        out.matches("FAILED: unreachable (a complete copy is cached and still readable offline)").count(),
        2,
        "{out}\n{}",
        stderr(&gone)
    );

    let list = vectordata(&home, &["cache", "list"], &[]);
    let out = stdout(&list);
    assert!(list.status.success(), "{}", stderr(&list));
    let section = out
        .split("Kept catalog copies (read when a catalog server is unreachable, or offline):\n")
        .nth(1)
        .unwrap_or_else(|| panic!("no kept-catalog section:\n{out}"));
    let first_row = section.lines().next().unwrap();
    assert!(first_row.trim_start().starts_with(".catalogs"), "{out}");
    assert!(first_row.ends_with("(1 file)"), "{out}");
}
