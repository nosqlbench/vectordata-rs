// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A fetched dataset opens and reads with no network, and a changed
//! upstream is still caught where the network is meant to be used.

mod support;

use std::path::Path;
use std::sync::LazyLock;

use support::fixtures::{fvec_value, write_fvec, write_ivvec, write_mref};
use support::testserver::TestServer;
use vectordata::TestDataGroup;
use vectordata::fetch::{FetchRequest, Silent};

static TEST_CACHE_DIR: LazyLock<tempfile::TempDir> = LazyLock::new(|| {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).expect("create test cache root")
});

fn make_tmp() -> tempfile::TempDir {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// A dataset with every kind of remote facet: merkle-published vectors,
/// a merkle-published variable-length facet with no index sidecar, a
/// merkle-published scalar, and vectors with no `.mref` at all (the
/// chunk-store path).
fn dataset(root: &Path, name: &str, seed: f32) {
    let ds = root.join(name);
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 2000, 8, seed);
    write_mref(&ds.join("base.fvec"));
    write_ivvec(&ds.join("results.ivvec"), 300, seed as i32);
    write_mref(&ds.join("results.ivvec"));
    let labels: Vec<u8> = (0..2000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(ds.join("labels.u8"), &labels).unwrap();
    write_mref(&ds.join("labels.u8"));
    write_fvec(&ds.join("query.fvec"), 50, 8, seed + 0.5);
    std::fs::write(
        ds.join("dataset.yaml"),
        format!(
            "name: {name}\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
             query_vectors: query.fvec\n    metadata_results: results.ivvec\n    \
             metadata_content: labels.u8\n"
        ),
    )
    .unwrap();
}

const CHILD_URL: &str = "VECTORDATA_TEST_OFFLINE_URL";
const CHILD_CACHE: &str = "VECTORDATA_TEST_OFFLINE_CACHE";
const CHILD_SEED: &str = "VECTORDATA_TEST_OFFLINE_SEED";

/// **A warmed cache needs no network.** Fetch a dataset, stop its
/// server, and in a fresh process open it by URL and read every kind of
/// facet: the definition comes from the copy the fetch kept, complete
/// data opens from disk (no `.mref`, no HEAD), the offset index from
/// the copy kept beside the cache file. A fetch on the warmed cache
/// then succeeds too, reporting that the upstream could not be checked.
///
/// The server being gone is the proof: any request the open or the
/// reads made would fail.
#[test]
fn a_fetched_dataset_opens_and_reads_with_the_server_gone() {
    let tmp = make_tmp();
    dataset(tmp.path(), "warm", 3.0e7);
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let url = format!("{}warm/", server.base_url());
    {
        let group = TestDataGroup::load(&url).unwrap();
        let view = group.profile("default").unwrap();
        let report = view.fetch(&FetchRequest::all(), &mut Silent).unwrap();
        assert!(report.facets.iter().all(|f| f.complete));
    }
    drop(server);

    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["read_offline_in_a_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_URL, &url)
        .env(CHILD_CACHE, TEST_CACHE_DIR.path())
        .env(CHILD_SEED, "30000000")
        .output()
        .expect("run the offline process");
    assert!(
        child.status.success(),
        "the offline process failed:\n{}\n{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
}

/// The offline half of the test above; does nothing unless launched by
/// it.
#[test]
fn read_offline_in_a_child() {
    let (Ok(url), Ok(cache), Ok(seed)) =
        (std::env::var(CHILD_URL), std::env::var(CHILD_CACHE), std::env::var(CHILD_SEED))
    else {
        return;
    };
    let seed: f32 = seed.parse().unwrap();
    vectordata::settings::override_cache_dir_for_process(cache.into());

    // First, before anything else in this process holds the file: a file
    // of the dataset opened by bare URL lands in the dataset's own cache
    // directory — the copy the fetch filled — so it opens complete with
    // the server gone, rather than keying a second, empty entry.
    {
        let by_url = vectordata::XvecReader::<f32>::open(&format!("{url}base.fvec"))
            .expect("a URL open finds the dataset's cached copy");
        assert!(vectordata::VectorReader::is_complete(&by_url));
        assert_eq!(vectordata::VectorReader::get(&by_url, 1999).unwrap()[7], fvec_value(seed, 8, 1999, 7));
    }

    let group = TestDataGroup::load(&url).expect("the kept dataset.yaml opens it offline");
    let view = group.profile("default").unwrap();

    let base = view.base_vectors().expect("complete merkle data opens offline");
    assert_eq!(base.get(1999).unwrap()[7], fvec_value(seed, 8, 1999, 7));
    assert!(base.get_slice(0).is_some(), "mapped");
    let query = view.query_vectors().expect("complete chunk-store data opens offline");
    assert_eq!(query.get(49).unwrap()[0], fvec_value(seed + 0.5, 8, 49, 0));
    let results = view.metadata_results().expect("the kept offset index opens offline");
    assert_eq!(results.get(299).unwrap(), vec![seed as i32 + 299]);
    let labels: vectordata::TypedReader<u8> =
        vectordata::open_facet_typed(&*view, "metadata_content").unwrap();
    assert_eq!(labels.get_native(300).unwrap(), (300 % 251) as u8);

    let report = view.fetch(&FetchRequest::all(), &mut Silent).expect("a warm fetch works offline");
    assert!(report.facets.iter().all(|f| f.complete && !f.upstream_checked), "{report:?}");
    assert_eq!(report.bytes_fetched(), 0);

}

/// **A changed upstream is caught by fetch, not by open.** Opening a
/// complete copy does not ask the server, so it keeps serving what was
/// verified; the fetch that would bring the dataset up to date asks,
/// and says the cache is stale.
#[test]
fn a_changed_upstream_is_stale_at_fetch_time() {
    let tmp = make_tmp();
    dataset(tmp.path(), "changing", 4.0e7);
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let url = format!("{}changing/", server.base_url());
    {
        let group = TestDataGroup::load(&url).unwrap();
        group.profile("default").unwrap().fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent).unwrap();
    }

    // The publisher replaces the file.
    write_fvec(&tmp.path().join("changing/base.fvec"), 2000, 8, 5.0e7);
    write_mref(&tmp.path().join("changing/base.fvec"));

    let group = TestDataGroup::load(&url).unwrap();
    let view = group.profile("default").unwrap();
    let base = view.base_vectors().expect("the complete copy opens as cached");
    assert_eq!(base.get(0).unwrap()[0], fvec_value(4.0e7, 8, 0, 0), "serving the verified copy");
    let err = view
        .fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent)
        .expect_err("a fetch revalidates and finds the upstream changed");
    assert!(err.to_string().contains("stale"), "{err}");
}

const STRICT_CATALOG: &str = "VECTORDATA_TEST_STRICT_CATALOG";

/// Serve `name` with a remote catalog file listing it.
fn served_catalog(root: &Path, name: &str, seed: f32) {
    dataset(root, name, seed);
    std::fs::write(
        root.join("catalog.json"),
        format!(
            r#"[{{"name":"{name}","path":"{name}/dataset.yaml","dataset_type":"dataset.yaml","layout":{{"profiles":{{"default":{{}}}}}}}}]"#
        ),
    )
    .unwrap();
}

/// **Offline mode makes no request at all**, and still opens a fetched
/// dataset by name. The fetch runs online here, keeping the catalog,
/// the definition, the data and the index; then a process with
/// `VECTORDATA_OFFLINE=1` opens the dataset through the catalog and
/// reads it while the server stays up — and the server accepts no
/// connection from it. A facet that was never fetched is refused, saying
/// offline mode is on.
#[test]
fn offline_mode_opens_a_fetched_dataset_by_name_without_a_single_request() {
    let tmp = make_tmp();
    served_catalog(tmp.path(), "strict", 6.0e7);
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    {
        let catalog = vectordata::catalog::Catalog::of(
            &vectordata::catalog::CatalogSources::new().add_catalogs(&[server.base_url()]),
        );
        let view = catalog.open_profile("strict", "default").unwrap();
        view.fetch(
            &FetchRequest::facets(["base_vectors", "metadata_results", "metadata_content"]),
            &mut Silent,
        )
        .unwrap();
    }
    let before = server.accepted_connections();

    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["read_strictly_offline_in_a_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(STRICT_CATALOG, server.base_url())
        .env(CHILD_CACHE, TEST_CACHE_DIR.path())
        .env(CHILD_SEED, "60000000")
        .env(vectordata::settings::OFFLINE_ENV, "1")
        .output()
        .expect("run the offline process");
    assert!(
        child.status.success(),
        "the offline process failed:\n{}\n{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    assert_eq!(server.accepted_connections(), before, "offline mode contacted the server");
}

/// The offline half of the test above.
#[test]
fn read_strictly_offline_in_a_child() {
    let (Ok(catalog_url), Ok(cache), Ok(seed)) =
        (std::env::var(STRICT_CATALOG), std::env::var(CHILD_CACHE), std::env::var(CHILD_SEED))
    else {
        return;
    };
    assert!(vectordata::settings::offline());
    let seed: f32 = seed.parse().unwrap();
    vectordata::settings::override_cache_dir_for_process(cache.into());

    let catalog = vectordata::catalog::Catalog::of(
        &vectordata::catalog::CatalogSources::new().add_catalogs(&[catalog_url]),
    );
    assert!(catalog.diagnostics().is_empty(), "the kept catalog copy loads: {:?}", catalog.diagnostics());
    let view = catalog.open_profile("strict", "default").expect("the dataset opens by name");
    assert_eq!(view.base_vectors().unwrap().get(5).unwrap()[0], fvec_value(seed, 8, 5, 0));
    assert_eq!(view.metadata_results().unwrap().get(7).unwrap(), vec![seed as i32 + 7]);
    let labels: vectordata::TypedReader<u8> =
        vectordata::open_facet_typed(&*view, "metadata_content").unwrap();
    assert_eq!(labels.get_native(9).unwrap(), 9);

    let refused = match view.query_vectors() {
        Ok(_) => panic!("a facet never fetched cannot be read offline"),
        Err(e) => e.to_string(),
    };
    assert!(refused.contains("offline mode is on"), "{refused}");
}

/// **A remote catalog loads from its kept copy when its server is
/// gone**, and the copy is named after the catalog's URL.
#[test]
fn a_remote_catalog_loads_from_its_kept_copy_when_unreachable() {
    let tmp = make_tmp();
    served_catalog(tmp.path(), "listed", 7.0e7);
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let sources = vectordata::catalog::CatalogSources::new().add_catalogs(&[server.base_url()]);
    assert_eq!(vectordata::catalog::Catalog::of(&sources).datasets().len(), 1);

    let port = server.port();
    let kept = TEST_CACHE_DIR.path().join(".catalogs").join(format!("127.0.0.1_{port}_catalog.json"));
    assert!(kept.is_file(), "the copy is kept at {}", kept.display());

    drop(server);
    let offline = vectordata::catalog::Catalog::of(&sources);
    assert!(offline.diagnostics().is_empty(), "{:?}", offline.diagnostics());
    assert_eq!(offline.datasets()[0].name, "listed");
}

/// **A changed upstream is caught for a file with no `.mref` too.** The
/// chunk-store copy has only the file's size to compare; when the
/// publisher's file changes size, the fetch says the cache is stale.
#[test]
fn a_resized_upstream_without_an_mref_is_stale_at_fetch_time() {
    let tmp = make_tmp();
    dataset(tmp.path(), "resized", 8.0e7);
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let url = format!("{}resized/", server.base_url());
    let fetch_query = || {
        TestDataGroup::load(&url)
            .unwrap()
            .profile("default")
            .unwrap()
            .fetch(&FetchRequest::facets(["query_vectors"]), &mut Silent)
    };
    let first = fetch_query().unwrap();
    assert!(first.facets[0].complete && first.facets[0].upstream_checked);
    assert!(fetch_query().unwrap().facets[0].upstream_checked, "an unchanged upstream is current");

    write_fvec(&tmp.path().join("resized/query.fvec"), 60, 8, 8.5e7);
    let err = fetch_query().expect_err("the upstream changed size");
    assert!(err.to_string().contains("stale"), "{err}");
}

/// **A dataset opened by its `dataset.yaml` URL keeps that file too**,
/// so the same URL opens it with the server gone.
#[test]
fn a_dataset_opened_by_its_yaml_url_opens_with_the_server_gone() {
    let tmp = make_tmp();
    dataset(tmp.path(), "by-yaml", 9.0e7);
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let url = format!("{}by-yaml/dataset.yaml", server.base_url());
    TestDataGroup::load(&url)
        .unwrap()
        .profile("default")
        .unwrap()
        .fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent)
        .unwrap();
    drop(server);

    let group = TestDataGroup::load(&url).expect("the kept dataset.yaml opens it");
    let base = group.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(base.get(1234).unwrap()[3], fvec_value(9.0e7, 8, 1234, 3));
}

/// **A `knn_entries.yaml` dataset opens with the server gone.** The
/// directory cascade asks for `dataset.yaml` first; with the server
/// unreachable and no copy of that kept, it goes on to the kept
/// `knn_entries.yaml` rather than failing at the first question.
#[test]
fn a_knn_entries_dataset_opens_with_the_server_gone() {
    let tmp = make_tmp();
    let ds = tmp.path().join("knn-ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 500, 8, 1.1e8);
    write_mref(&ds.join("base.fvec"));
    write_fvec(&ds.join("query.fvec"), 20, 8, 1.15e8);
    write_ivvec(&ds.join("gt.ivec"), 20, 0);
    std::fs::write(
        ds.join("knn_entries.yaml"),
        "\"knn-ds:default\":\n  base: base.fvec\n  query: query.fvec\n  gt: gt.ivec\n",
    )
    .unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let url = format!("{}knn-ds/", server.base_url());
    TestDataGroup::load(&url)
        .unwrap()
        .profile("default")
        .unwrap()
        .fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent)
        .unwrap();
    drop(server);

    let group = TestDataGroup::load(&url).expect("the kept knn_entries.yaml opens it");
    let base = group.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(base.get(499).unwrap()[0], fvec_value(1.1e8, 8, 499, 0));
}

/// **A kept definition belongs to its origin.** A same-named dataset
/// from another server opens online, but neither replaces the first
/// server's kept copy nor, once its own server is gone, is answered
/// with it.
#[test]
fn a_kept_definition_is_never_served_for_another_origin() {
    let first = make_tmp();
    let second = make_tmp();
    dataset(first.path(), "twin-ds", 1.2e8);
    dataset(second.path(), "twin-ds", 1.3e8);
    std::fs::write(
        second.path().join("twin-ds/dataset.yaml"),
        "name: twin-ds\nprofiles:\n  default:\n    base_vectors: base.fvec\n  other:\n    base_vectors: base.fvec\n",
    )
    .unwrap();
    let a = TestServer::start(first.path()).unwrap();
    let b = TestServer::start(second.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let kept = TEST_CACHE_DIR.path().join("twin-ds/dataset.yaml");

    TestDataGroup::load(&format!("{}twin-ds/", a.base_url())).unwrap();
    let a_yaml = std::fs::read_to_string(&kept).expect("the first origin's copy is kept");
    let b_url = format!("{}twin-ds/", b.base_url());
    let from_b = TestDataGroup::load(&b_url).expect("the other origin opens online");
    assert!(from_b.profile("other").is_some(), "with its own definition");
    assert_eq!(std::fs::read_to_string(&kept).unwrap(), a_yaml, "the first origin's copy is untouched");

    drop(b);
    assert!(TestDataGroup::load(&b_url).is_err(), "the first origin's copy is not served for the second");
}

const PARTIAL_URL: &str = "VECTORDATA_TEST_PARTIAL_URL";

/// **Offline, a partly fetched facet serves what it holds.** A window
/// of the base vectors is fetched online; a process in offline mode
/// opens the dataset and reads inside the window, and a read outside
/// it is refused saying offline mode is on — not a request, not a hang.
#[test]
fn offline_mode_reads_what_a_partial_fetch_holds() {
    let tmp = make_tmp();
    dataset(tmp.path(), "partial", 1.4e8);
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let url = format!("{}partial/", server.base_url());
    let window = vectordata::dataset::source::parse_window("[0..100]").unwrap();
    let report = TestDataGroup::load(&url)
        .unwrap()
        .profile("default")
        .unwrap()
        .fetch(&FetchRequest::facets(["base_vectors"]).window(window), &mut Silent)
        .unwrap();
    assert!(!report.facets[0].complete, "only a window was fetched");
    let before = server.accepted_connections();

    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["read_partially_offline_in_a_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(PARTIAL_URL, &url)
        .env(CHILD_CACHE, TEST_CACHE_DIR.path())
        .env(CHILD_SEED, "140000000")
        .env(vectordata::settings::OFFLINE_ENV, "1")
        .output()
        .expect("run the offline process");
    assert!(
        child.status.success(),
        "the offline process failed:\n{}\n{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    assert_eq!(server.accepted_connections(), before, "offline mode contacted the server");
}

/// The offline half of the test above.
#[test]
fn read_partially_offline_in_a_child() {
    let (Ok(url), Ok(cache), Ok(seed)) =
        (std::env::var(PARTIAL_URL), std::env::var(CHILD_CACHE), std::env::var(CHILD_SEED))
    else {
        return;
    };
    assert!(vectordata::settings::offline());
    let seed: f32 = seed.parse().unwrap();
    vectordata::settings::override_cache_dir_for_process(cache.into());

    let group = TestDataGroup::load(&url).expect("the kept dataset.yaml opens it offline");
    let base = group.profile("default").unwrap().base_vectors().expect("a partial copy opens offline");
    assert!(!vectordata::VectorReader::is_complete(&*base));
    assert_eq!(base.get(42).unwrap()[5], fvec_value(seed, 8, 42, 5));
    let started = std::time::Instant::now();
    let refused = base.get(1999).expect_err("a record outside the fetched window").to_string();
    assert!(refused.contains("offline mode is on"), "{refused}");
    // Refused at once: the refusal is not retried with backoff.
    assert!(started.elapsed() < std::time::Duration::from_secs(5), "refused after {:?}", started.elapsed());
}

/// Occupy `port` with a listener that never accepts, and fill its
/// accept queue: a further connection is then neither accepted nor
/// refused, and hangs — what a dead host behind a firewall looks like.
fn hanging_listener(port: u16) -> (std::net::TcpListener, Vec<std::net::TcpStream>) {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).expect("rebind the stopped server's port");
    let addr = listener.local_addr().unwrap();
    let mut held = Vec::new();
    for _ in 0..4096 {
        match std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300)) {
            Ok(s) => held.push(s),
            Err(_) => return (listener, held),
        }
    }
    panic!("the accept queue never filled");
}

/// **A server that hangs instead of refusing is given up on in
/// seconds.** The connection attempt times out, and the kept copy is
/// used, as for a refused connection — rather than waiting out the
/// operating system's TCP timeout of minutes.
#[test]
fn a_hanging_server_falls_back_to_the_kept_copy_in_seconds() {
    let tmp = make_tmp();
    dataset(tmp.path(), "hanging", 1.5e8);
    let server = TestServer::start(tmp.path()).unwrap();
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let url = format!("{}hanging/", server.base_url());
    TestDataGroup::load(&url)
        .unwrap()
        .profile("default")
        .unwrap()
        .fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent)
        .unwrap();
    let port = server.port();
    drop(server);
    let _hang = hanging_listener(port);

    let started = std::time::Instant::now();
    let group = TestDataGroup::load(&url).expect("the kept copy, once the connection times out");
    let waited = started.elapsed();
    assert!(waited < std::time::Duration::from_secs(30), "waited {waited:?}");
    let base = group.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(base.get(7).unwrap()[1], fvec_value(1.5e8, 8, 7, 1));
}
