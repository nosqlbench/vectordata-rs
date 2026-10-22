// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **The cache layout does not change** (`docs/design/srd-crate-boundaries.md`
//! principle 5, TR-7): every path a fetch writes, for a representative
//! dataset served over loopback, is the list recorded from the code
//! before the transport moved into `veks-transport`.
//!
//! The dataset covers each way a remote file reaches the cache: merkle
//! published (`.mrkl`), unpublished (the chunk store's `.chunks`), a
//! variable-length facet whose offset index is rebuilt beside its copy,
//! a scalar, a facet in a profile subdirectory, the catalog's kept copy,
//! the kept `dataset.yaml`, and a file opened by bare URL with no
//! dataset around it (the URL-derived directory). The port of the
//! loopback server is the one part of a path that varies, and is
//! written `PORT`.

mod support;

use std::path::Path;

use vectordata::fetch::{FetchRequest, Silent};
use vectordata::{TestDataViewExt, VectorReader};

use support::fixtures::{write_catalogs, write_fvec, write_ivvec, write_mref};
use support::testserver::TestServer;

/// Every path under `root`, relative and `/`-separated, directories
/// included, sorted.
fn every_path(root: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        entries.sort();
        for path in entries {
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            if path.is_dir() {
                out.push(format!("{rel}/"));
                walk(root, &path, out);
            } else {
                out.push(rel);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

/// What the code before the move wrote, recorded by running this test
/// against it.
const RECORDED: &[&str] = &[
    ".catalogs/",
    ".catalogs/127.0.0.1_PORT_catalog.json",
    "127.0.0.1_PORT/",
    "127.0.0.1_PORT/loose/",
    "127.0.0.1_PORT/loose/.publish_url",
    "127.0.0.1_PORT/loose/origin.json",
    "127.0.0.1_PORT/loose/x.fvec",
    "127.0.0.1_PORT/loose/x.fvec.mrkl",
    "layout/",
    "layout/.publish_url",
    "layout/IDXFOR__results.ivvec.i32",
    "layout/base.fvec",
    "layout/base.fvec.mrkl",
    "layout/dataset.yaml",
    "layout/labels.u8",
    "layout/labels.u8.mrkl",
    "layout/origin.json",
    "layout/profiles/",
    "layout/profiles/small/",
    "layout/profiles/small/base.fvec",
    "layout/profiles/small/base.fvec.chunks",
    "layout/query.fvec",
    "layout/query.fvec.chunks",
    "layout/results.ivvec",
    "layout/results.ivvec.mrkl",
];

#[test]
fn a_fetch_writes_the_paths_it_always_wrote() {
    let served = tempfile::tempdir().unwrap();
    let ds = served.path().join("layout");
    std::fs::create_dir_all(ds.join("profiles/small")).unwrap();
    write_fvec(&ds.join("base.fvec"), 2000, 8, 1.0e8);
    write_mref(&ds.join("base.fvec"));
    write_fvec(&ds.join("query.fvec"), 50, 8, 2.0e8);
    write_ivvec(&ds.join("results.ivvec"), 300, 7);
    write_mref(&ds.join("results.ivvec"));
    let labels: Vec<u8> = (0..2000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(ds.join("labels.u8"), &labels).unwrap();
    write_mref(&ds.join("labels.u8"));
    write_fvec(&ds.join("profiles/small/base.fvec"), 100, 8, 3.0e8);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: layout\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         query_vectors: query.fvec\n    metadata_results: results.ivvec\n    \
         metadata_content: labels.u8\n  small:\n    base_vectors: profiles/small/base.fvec\n",
    )
    .unwrap();
    // A file with no dataset around it, opened by its URL.
    std::fs::create_dir_all(served.path().join("loose")).unwrap();
    write_fvec(&served.path().join("loose/x.fvec"), 10, 4, 4.0e8);
    write_mref(&served.path().join("loose/x.fvec"));
    write_catalogs(served.path(), &["layout"]);

    let server = TestServer::start(served.path()).unwrap();
    let cache = tempfile::tempdir().unwrap();
    vectordata::settings::override_cache_dir_for_process(cache.path().to_path_buf());

    let catalog = vectordata::catalog::Catalog::of(
        &vectordata::catalog::CatalogSources::new().add_catalogs(&[server.base_url()]),
    );
    for profile in ["default", "small"] {
        let view = catalog.open_profile("layout", profile).unwrap();
        let report = view.fetch(&FetchRequest::all(), &mut Silent).unwrap();
        assert!(report.facets.iter().all(|f| f.complete), "{report:?}");
        let base = view.base_vectors_as::<f32>().unwrap();
        assert!(base.get(base.count() - 1).is_ok());
        if profile == "default" {
            assert!(view.query_vectors_as::<f32>().unwrap().get(49).is_ok());
            assert_eq!(view.metadata_results_as::<i32>().unwrap().get(299).unwrap(), vec![7 + 299]);
            assert_eq!(view.facet_as::<u8>("metadata_content").unwrap().get(300).unwrap()[0], (300 % 251) as u8);
        }
    }
    let loose = vectordata::XvecReader::<f32>::open(&format!("{}loose/x.fvec", server.base_url())).unwrap();
    assert_eq!(loose.count(), 10);
    loose.get(9).unwrap();

    let port = server.port().to_string();
    let written: Vec<String> =
        every_path(cache.path()).into_iter().map(|p| p.replace(&format!("127.0.0.1_{port}"), "127.0.0.1_PORT")).collect();
    assert_eq!(
        written,
        RECORDED,
        "the cache layout changed (srd-crate-boundaries.md principle 5):\n{}",
        written.join("\n")
    );
}
