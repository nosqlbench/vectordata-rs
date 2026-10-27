// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **A remote variable-length facet published without its offset index
//! opens when the caller accepts a whole-file fetch**, and says so: the
//! lazy open refuses, the fetching open fetches the facet and opens it,
//! and a warning names the file and the index its publisher should add.
//! An indexed facet opens without a fetch or a warning.

mod support;

use std::sync::{LazyLock, Mutex, OnceLock};

use support::testserver::TestServer;
use vectordata::fetch::FetchEvent;
use vectordata::io::IoError;
use vectordata::{Error, TestDataGroup, TestDataViewExt};

/// Every `warn` line logged in this test binary.
static LOGGED: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Warn
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            LOGGED.lock().unwrap().push(r.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn capture() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        log::set_logger(&Capture).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    });
}

fn warnings_naming(text: &str) -> Vec<String> {
    LOGGED.lock().unwrap().iter().filter(|l| l.contains(text)).cloned().collect()
}

/// This binary's cache, under `target/tmp`, never the user's.
static CACHE: LazyLock<tempfile::TempDir> = LazyLock::new(|| {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
});

fn served_dataset(name: &str, file: &str) -> (tempfile::TempDir, TestServer) {
    vectordata::settings::override_cache_dir_for_process(CACHE.path().to_path_buf());
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let mut bytes = Vec::new();
    for r in [&[3i32, 1][..], &[], &[2]] {
        bytes.extend((r.len() as i32).to_le_bytes());
        r.iter().for_each(|x| bytes.extend(x.to_le_bytes()));
    }
    std::fs::write(dir.path().join(file), bytes).unwrap();
    std::fs::write(dir.path().join("dataset.yaml"), format!("name: {name}\nprofiles:\n  default:\n    metadata_results: {file}\n")).unwrap();
    let server = TestServer::start(dir.path()).unwrap();
    (dir, server)
}

/// The fetch events a sink heard: how many facets began fetching.
fn begins(events: &[String]) -> usize {
    events.iter().filter(|e| e.starts_with("Begin")).count()
}

/// **The fetching open fetches what the lazy open refuses, once, and
/// warns that the dataset should carry the index.**
#[test]
fn an_unindexed_remote_vvec_opens_by_fetching_it_with_a_warning() {
    capture();
    let (_dir, server) = served_dataset("unindexed", "results_a.ivvec");
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    match view.metadata_results_as::<i32>() {
        Err(Error::VectorIo(IoError::OffsetIndexUnavailable { .. })) => {}
        other => panic!("the lazy open refuses: {:?}", other.map(|r| r.count())),
    }
    assert!(warnings_naming("results_a.ivvec").is_empty(), "a refusal is not the warning");

    let mut heard = Vec::new();
    let mut sink = |e: &FetchEvent<'_>| heard.push(format!("{e:?}"));
    let results = view.metadata_results_fetching_as::<i32>(&mut sink).expect("opens once fetched");
    assert_eq!(results.count(), 3);
    assert_eq!(results.get(0).unwrap(), vec![3, 1]);
    assert_eq!(results.get(2).unwrap(), vec![2]);
    assert_eq!(begins(&heard), 1, "the facet is fetched, with progress: {heard:?}");

    let warned = warnings_naming("results_a.ivvec");
    assert_eq!(warned.len(), 1, "{warned:?}");
    for part in ["without an offset index", "optimized upstream", "ask its publisher", "IDXFOR__results_a.ivvec.i32"] {
        assert!(warned[0].contains(part), "{part}: {}", warned[0]);
    }

    // Once fetched, it opens as an indexed facet does: no fetch, no warning.
    let mut again = Vec::new();
    let mut sink = |e: &FetchEvent<'_>| again.push(format!("{e:?}"));
    assert_eq!(view.vvec_facet_fetching_as::<i32>("metadata_results", &mut sink).unwrap().count(), 3);
    assert_eq!(begins(&again), 0, "{again:?}");
    assert_eq!(warnings_naming("results_a.ivvec").len(), 1);
}

/// **An indexed remote vvec opens lazily through the fetching open**:
/// nothing is fetched and nothing is warned.
#[test]
fn an_indexed_remote_vvec_opens_without_a_fetch_or_a_warning() {
    capture();
    let (dir, server) = served_dataset("indexed", "results_b.ivvec");
    vectordata::vvec_index::prepare(&dir.path().join("results_b.ivvec"), false, &mut |_| {}).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let mut heard = Vec::new();
    let mut sink = |e: &FetchEvent<'_>| heard.push(format!("{e:?}"));
    let results = view.metadata_results_fetching_as::<i32>(&mut sink).unwrap();
    assert_eq!(results.get(1).unwrap(), Vec::<i32>::new());
    assert_eq!(begins(&heard), 0, "{heard:?}");
    assert!(warnings_naming("results_b.ivvec").is_empty());
}
