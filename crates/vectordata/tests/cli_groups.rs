// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The shape of the `vectordata` command line, run as the binary a user
//! runs, each in its own `VECTORDATA_HOME`: a group named alone lists
//! its subcommands, and the explorer is `datasets explore`, with
//! `explore` as its short form.

use std::path::Path;
use std::process::{Command, Output};

fn make_tmp() -> tempfile::TempDir {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

fn vectordata(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(args)
        .env("VECTORDATA_HOME", home)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .output()
        .expect("run vectordata")
}

fn text(o: &Output) -> (String, String) {
    (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// **A group named alone lists its subcommands.** `cache` and `config
/// catalog` print their help and succeed, as `datasets` does; given an
/// option but no subcommand, a group is a usage error that still lists
/// them.
#[test]
fn a_bare_group_lists_its_subcommands() {
    let tmp = make_tmp();
    for (group, sub) in [(&["cache"][..], "prune"), (&["config", "catalog"][..], "disable"), (&["datasets"][..], "describe")] {
        let out = vectordata(tmp.path(), group);
        let (stdout, stderr) = text(&out);
        assert!(out.status.success(), "{group:?}: {stdout}{stderr}");
        assert!(stdout.contains("Commands:") && stdout.contains(sub), "{group:?}: {stdout}");
    }

    let out = vectordata(tmp.path(), &["datasets", "--configdir", tmp.path().to_str().unwrap()]);
    let (stdout, stderr) = text(&out);
    assert!(out.status.success(), "{stdout}{stderr}");
    assert!(stdout.contains("explore"), "{stdout}");

    let out = vectordata(tmp.path(), &["config", "catalog", "--no-such-flag"]);
    let (_, stderr) = text(&out);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
}

/// **A bare `cache` says where the cache is and how much it holds**,
/// with the lines `cache list` opens with, above its subcommands.
#[test]
fn a_bare_cache_opens_with_its_directory_and_size() {
    let tmp = make_tmp();
    let cache = tmp.path().join("the-cache");
    std::fs::create_dir_all(cache.join("ds")).unwrap();
    std::fs::write(cache.join("ds/base.fvec"), vec![7u8; 1 << 20]).unwrap();
    std::fs::write(tmp.path().join("settings.yaml"), format!("cache_dir: {}\n", cache.display())).unwrap();

    let (listed, _) = text(&vectordata(tmp.path(), &["cache", "list"]));
    let header: Vec<&str> = listed.lines().take(2).collect();
    assert!(header[0] == format!("Cache directory: {}", cache.display()), "{listed}");
    assert!(header[1].starts_with("Total on disk:") && !header[1].ends_with(" 0 B"), "{listed}");

    let out = vectordata(tmp.path(), &["cache"]);
    let (stdout, stderr) = text(&out);
    assert!(out.status.success(), "{stdout}{stderr}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines[..2], header[..], "{stdout}");
    let usage = lines.iter().position(|l| l.starts_with("Usage:")).unwrap_or_else(|| panic!("{stdout}"));
    assert!(usage > 2 && stdout.contains("prune-legacy"), "{stdout}");
}

/// **The explorer is `datasets explore`, and `explore` is its short
/// form**: both are listed and read the run's catalogs the same way —
/// an `--at` name no catalog has is refused by either.
#[test]
fn explore_is_a_datasets_command_with_a_short_form() {
    let tmp = make_tmp();
    let (datasets_help, _) = text(&vectordata(tmp.path(), &["datasets", "--help"]));
    assert!(datasets_help.lines().any(|l| l.trim_start().starts_with("explore ")), "{datasets_help}");

    for form in [&["datasets", "explore"][..], &["explore"][..]] {
        let mut args = form.to_vec();
        args.push("--help");
        let out = vectordata(tmp.path(), &args);
        let (stdout, stderr) = text(&out);
        assert!(out.status.success(), "{form:?}: {stdout}{stderr}");
        assert!(stdout.contains("--dataset") && stdout.contains("--at"), "{form:?}: {stdout}");

        let mut args = form.to_vec();
        args.extend(["--at", "nowhere", "--configdir", tmp.path().to_str().unwrap()]);
        let out = vectordata(tmp.path(), &args);
        let (stdout, stderr) = text(&out);
        assert!(!out.status.success(), "{form:?}: {stdout}{stderr}");
        assert!(stderr.contains("no catalog is named 'nowhere'"), "{form:?}: {stderr}");
    }
}
