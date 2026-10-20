// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `datasets describe` (and `ping`'s lookup), run as the binary a user
//! runs: a name two catalogs hold is reported as ambiguous, never as
//! missing, which is the opposite of the truth; and a series is
//! described by the files it spans, the count the planner fetches.

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

/// A local catalog under `root` holding one dataset, `name`.
fn catalog_with(root: &Path, name: &str) {
    let ds = root.join(name);
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 10, 4, 1.0);
    std::fs::write(ds.join("dataset.yaml"), format!("name: {name}\nprofiles:\n  default:\n    base_vectors: base.fvec\n"))
        .unwrap();
    write_catalogs(root, &[name]);
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

/// **`datasets describe` and `datasets ping` name both catalogs of a
/// doubled name**, as `precache` does, and say how to choose. They
/// reported it "not found in any configured catalog".
#[test]
fn describe_and_ping_report_a_doubled_name_as_ambiguous() {
    let tmp = tempfile::tempdir().unwrap();
    let home = home(tmp.path());
    let (a, b) = (tmp.path().join("cat-a"), tmp.path().join("cat-b"));
    catalog_with(&a, "twin");
    catalog_with(&b, "twin");
    let (a, b) = (a.to_string_lossy().into_owned(), b.to_string_lossy().into_owned());

    for command in ["describe", "ping"] {
        let out = vectordata(&home, &["datasets", command, "--catalog", &a, "--catalog", &b, "twin"]);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{command}: {err}");
        assert!(err.contains("multiple datasets match 'twin'"), "{command}: {err}");
        assert!(err.contains("--at"), "{command} names the remedy: {err}");
        assert!(!err.contains("not found"), "{command}: {err}");
    }

    // One catalog named: the same commands find it.
    let out = vectordata(&home, &["datasets", "describe", "--at", &a, "twin"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

/// **A uniform series is described by its files, not its one pattern**:
/// `shards` is the count `precache --plan` fetches and
/// `FacetDescriptor::shard_count` holds, and each file is listed. It
/// printed `shards 1` and the pattern, for any number of files.
#[test]
fn describe_counts_the_files_a_uniform_series_spans() {
    let tmp = tempfile::tempdir().unwrap();
    let home = home(tmp.path());
    let root = tmp.path().join("cat");
    let ds = root.join("sharded");
    std::fs::create_dir_all(&ds).unwrap();
    for (i, records) in [4, 4, 2].iter().enumerate() {
        write_fvec(&ds.join(format!("base__{i:04}.fvec")), *records, 4, i as f32);
    }
    std::fs::write(
        ds.join("dataset.yaml"),
        "format_version: 2\nname: sharded\nprofiles:\n  default:\n    base_vectors:\n      source: base__NNNN.fvec\n      \
         shard_stride: 4\n      shard_count: 3\n      record_count: 10\n",
    )
    .unwrap();
    write_catalogs(&root, &["sharded"]);

    let out = vectordata(&home, &["datasets", "describe", "--at", &root.to_string_lossy(), "sharded"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
    let shards = text.lines().find(|l| l.trim_start().starts_with("shards ")).unwrap_or_else(|| panic!("{text}"));
    assert_eq!(shards.split_whitespace().nth(1), Some("3"), "{text}");
    for i in 0..3 {
        assert!(text.contains(&format!("base__{i:04}.fvec")), "file {i} listed: {text}");
    }
    assert!(text.contains("pattern       base__NNNN.fvec"), "{text}");
}
