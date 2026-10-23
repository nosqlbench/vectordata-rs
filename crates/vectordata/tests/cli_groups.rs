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
