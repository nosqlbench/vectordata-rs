// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The CLI embedded in another program sees that program's catalogs
//! (`shell::bin_main_with_catalogs`): a dataset only the program
//! registered resolves by name, as a configured one does.

#![cfg(feature = "cli")]

use vectordata::Location;
use vectordata::catalog::{CatalogBuilder, CatalogSources};
use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};

/// Three two-dimensional f32 vectors in `.fvec` form.
fn write_fvec(path: &std::path::Path) {
    let mut buf = Vec::new();
    for i in 0..3 {
        buf.extend(2i32.to_le_bytes());
        buf.extend((i as f32).to_le_bytes());
        buf.extend((i as f32 + 0.5).to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

/// **A dataset the program registered is found by name** by an
/// embedded `datasets precache --plan`, with no catalog configured. A
/// lookup that missed would exit the process with status 1, failing
/// this test; returning is the pass.
#[test]
fn an_embedded_cli_finds_a_dataset_its_program_registered() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("bench-ds");
    std::fs::create_dir_all(&home).unwrap();
    write_fvec(&home.join("b.fvec"));
    let config = tmp.path().join("config");
    std::fs::create_dir_all(&config).unwrap();

    let description = DatasetConfig::builder("bench-ds")
        .profile("default", |p| p.facet(StandardFacet::BaseVectors, "b.fvec"))
        .build()
        .unwrap();
    let entry = CatalogEntry::builder("bench-ds")
        .home(&Location::from_path(&home))
        .dataset(description)
        .build()
        .unwrap();
    let program = CatalogSources::new().with(CatalogBuilder::named("bench").entry(entry).build());

    let argv: Vec<String> = ["datasets", "precache", "bench-ds:default", "--plan", "--configdir"]
        .iter()
        .map(|s| s.to_string())
        .chain([config.to_string_lossy().into_owned()])
        .collect();
    vectordata::shell::bin_main_with_catalogs("embedded", argv, program);
}

/// **`--at` names the program's catalog**, so a run can read it and
/// nothing configured: `--at bench` finds the dataset only `bench`
/// holds. It was taken for a path that does not exist, and the run
/// found no catalog.
#[test]
fn at_names_the_programs_catalog() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("bench-ds");
    std::fs::create_dir_all(&home).unwrap();
    write_fvec(&home.join("b.fvec"));
    let config = tmp.path().join("config");
    std::fs::create_dir_all(&config).unwrap();
    let description = DatasetConfig::builder("bench-ds")
        .profile("default", |p| p.facet(StandardFacet::BaseVectors, "b.fvec"))
        .build()
        .unwrap();
    let entry = CatalogEntry::builder("bench-ds")
        .home(&Location::from_path(&home))
        .dataset(description)
        .build()
        .unwrap();
    let program = CatalogSources::new().with(CatalogBuilder::named("bench").entry(entry).build());
    let argv: Vec<String> = ["datasets", "precache", "--at", "bench", "bench-ds:default", "--plan", "--configdir"]
        .iter()
        .map(|s| s.to_string())
        .chain([config.to_string_lossy().into_owned()])
        .collect();
    vectordata::shell::bin_main_with_catalogs("embedded", argv, program);
}

/// **A host completes the embedded CLI through its own tree**: the
/// node grafted under its `vectordata` subcommand answers the commands
/// beneath it, and `shell::complete` answers the same words for a host
/// not built on veks-completion (LS-5).
#[test]
fn a_host_completes_the_embedded_cli_from_its_own_tree() {
    let mut host = veks_completion::CommandTree::new("fvec");
    host.root = veks_completion::Node::group(vec![("bench", veks_completion::Node::empty_group())])
        .with_child("vectordata", vectordata::shell::completion_node());
    // The completer's first word is the host's own name.
    let through_host = veks_completion::complete(&host, &["fvec", "vectordata", "datasets", "desc"]);
    assert!(through_host.iter().any(|c| c == "describe"), "{through_host:?}");
    let direct = vectordata::shell::complete(&["datasets", "desc"]);
    assert!(direct.iter().any(|c| c == "describe"), "{direct:?}");
}
