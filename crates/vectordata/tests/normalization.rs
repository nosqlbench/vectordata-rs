// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Whether a base is unit length, and on whose word
//! (`TestDataViewExt::normalization`, IN-4/IN-5): a declared value
//! checked against a sample, the sample alone when nothing declares
//! it, recorded for a remote base by its content identity and never
//! re-taken, and a contradiction reported (V8) and, for an overlay's
//! claim, warned about.

mod support;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, OnceLock};

use support::testserver::TestServer;
use vectordata::overlay::{AttributeSource, Overlay, Overlays};
use vectordata::verify::NoProgress;
use vectordata::{TestDataGroup, TestDataViewExt};

static CACHE: LazyLock<tempfile::TempDir> = LazyLock::new(|| {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
});

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

/// `n` vectors of dimension 2: of norm 1, or of norm `scale`.
fn base(path: &Path, n: usize, scale: f32) {
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..n {
        let a = i as f32 * 0.01;
        f.write_all(&2i32.to_le_bytes()).unwrap();
        for x in [a.cos() * scale, a.sin() * scale] {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

fn dataset(dir: &Path, name: &str, attrs: &str) {
    std::fs::write(dir.join("dataset.yaml"), format!("name: {name}\n{attrs}profiles:\n  default:\n    base_vectors: base.fvecs\n")).unwrap();
}

/// **A dataset's own declaration the sample contradicts is reported
/// as V8**, and the declared value is kept, marked contradicted.
#[test]
fn a_declared_normalization_the_sample_contradicts_is_v8() {
    let dir = tempfile::tempdir().unwrap();
    base(&dir.path().join("base.fvecs"), 50, 3.0);
    dataset(dir.path(), "claims-unit", "attributes:\n  is_normalized: true\n");
    let view = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap().profile("default").unwrap();
    let n = view.normalization().unwrap();
    assert!(n.value && n.contradicted && !n.measured.value, "{n:?}");
    assert_eq!(n.source, AttributeSource::Dataset);
    let err = view.verify(&mut NoProgress).unwrap_err().to_string();
    assert!(err.contains("V8 base_vectors") && err.contains("is declared is_normalized: true"), "{err}");
}

/// **An overlay's claim the sample contradicts is warned about**, once,
/// and stays the answer.
#[test]
fn an_overlay_normalization_the_sample_contradicts_is_warned() {
    capture();
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("overlaid-unit");
    std::fs::create_dir(&dir).unwrap();
    base(&dir.join("base.fvecs"), 50, 3.0);
    dataset(&dir, "overlaid-unit", "");
    let mut overlays = Overlays::new();
    overlays.add_program(Overlay::program("host", "datasets:\n  overlaid-unit:\n    is_normalized: true\n").unwrap()).unwrap();
    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap().with_overlays(&overlays);
    let view = group.profile("default").unwrap();
    for _ in 0..2 {
        let n = view.normalization().unwrap();
        assert!(n.value && n.contradicted);
        assert!(matches!(n.source, AttributeSource::Overlay { .. }));
    }
    let warned: Vec<String> = LOGGED.lock().unwrap().iter().filter(|l| l.contains("'overlaid-unit'")).cloned().collect();
    assert_eq!(warned.len(), 1, "{warned:?}");
    assert!(warned[0].contains("states is_normalized: true") && warned[0].contains("in force"), "{warned:?}");
}

/// **A local base nothing declares is measured each time**, with no
/// file written into the cache: a local dataset has no cache directory.
#[test]
fn a_local_base_is_measured_and_nothing_is_recorded() {
    vectordata::settings::override_cache_dir_for_process(CACHE.path().to_path_buf());
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("local-measured");
    std::fs::create_dir(&dir).unwrap();
    base(&dir.join("base.fvecs"), 2000, 1.0);
    dataset(&dir, "local-measured", "");
    let view = TestDataGroup::load(dir.to_str().unwrap()).unwrap().profile("default").unwrap();
    let n = view.normalization().unwrap();
    assert!(n.value);
    assert_eq!(n.measured.sample, 1000, "1000 of 2000, a stride apart");
    assert_eq!(n.source, AttributeSource::Measured { file: None });
    assert!(!CACHE.path().join("local-measured").join("measured.yaml").exists());
}

/// **A remote base's measurement is recorded by its content identity
/// and never re-taken**: corrupting the cached bytes afterwards does not
/// change the answer, because the identity the cache knows is that of
/// what was fetched.
#[test]
fn a_remote_measurement_is_recorded_by_content_id_and_reused() {
    vectordata::settings::override_cache_dir_for_process(CACHE.path().to_path_buf());
    let served = tempfile::tempdir().unwrap();
    let dir = served.path().join("remote-unit");
    std::fs::create_dir(&dir).unwrap();
    base(&dir.join("base.fvecs"), 1500, 1.0);
    dataset(&dir, "remote-unit", "");
    let server = TestServer::start(served.path()).unwrap();
    let group = TestDataGroup::load(&format!("{}/remote-unit/", server.base_url())).unwrap();
    let view = group.profile("default").unwrap();
    let storage = view.open_facet_storage("base_vectors").unwrap();
    storage.precache().unwrap();
    let id = storage.known_content_id().expect("fetched whole: the recorded root");

    let first = view.normalization().unwrap();
    assert!(first.value);
    let AttributeSource::Measured { file: Some(file) } = &first.source else { panic!("{:?}", first.source) };
    let recorded = std::fs::read_to_string(file).unwrap();
    assert!(recorded.contains(&id.hex()) && recorded.contains("is_normalized"), "{recorded}");

    // Make the cached bytes non-unit: a re-taken sample would see it.
    let cached = PathBuf::from(file).parent().unwrap().join("base.fvecs");
    let mut bytes = std::fs::read(&cached).unwrap();
    for rec in 0..1500 {
        bytes[rec * 12 + 4..rec * 12 + 8].copy_from_slice(&9.0f32.to_le_bytes());
    }
    std::fs::write(&cached, &bytes).unwrap();
    let again = view.normalization().unwrap();
    assert_eq!((again.value, again.measured.worst_deviation), (first.value, first.measured.worst_deviation), "read from the record, not re-sampled");
}
