// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A program that embeds vectordata choosing its own cache directory
//! with `settings::set_cache_dir`, and `precache --cache-dir` doing the
//! same from the command line: data lands in the chosen cache, the
//! user's configured cache and `settings.yaml` are left as they were,
//! and a choice that cannot take effect is refused. The choice is
//! per-process, so each case runs in a fresh process.

mod support;

use std::path::{Path, PathBuf};
use std::process::Command;

use support::fixtures::{fvec_value, write_fvec, write_mref};
use support::testserver::TestServer;
use vectordata::fetch::{FetchRequest, Silent};
use vectordata::settings::{self, CacheDirConflict};

const CASE: &str = "VECTORDATA_TEST_CACHE_DIR_CASE";
const HOME: &str = "VECTORDATA_TEST_CACHE_DIR_HOME";

fn make_tmp() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// A `VECTORDATA_HOME` whose settings name a cache inside it, with a
/// comment of the user's own. Returns the home and its settings text.
fn user_home(root: &Path) -> (PathBuf, String) {
    let home = root.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let text = format!("# my own note\ncache_dir: {}\n", home.join("user-cache").display());
    std::fs::write(home.join("settings.yaml"), &text).unwrap();
    (home, text)
}

/// Serve dataset `name` with one merkle-published base facet.
fn serve(root: &Path, name: &str, seed: f32) -> TestServer {
    let ds = root.join("served").join(name);
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 400, 8, seed);
    write_mref(&ds.join("base.fvec"));
    std::fs::write(
        ds.join("dataset.yaml"),
        format!("name: {name}\nprofiles:\n  default:\n    base_vectors: base.fvec\n"),
    )
    .unwrap();
    std::fs::write(
        root.join("served/catalog.json"),
        format!(
            r#"[{{"name":"{name}","path":"{name}/dataset.yaml","dataset_type":"dataset.yaml","layout":{{"profiles":{{"default":{{}}}}}}}}]"#
        ),
    )
    .unwrap();
    TestServer::start(&root.join("served")).unwrap()
}

/// Run the child half of `case` in a fresh process with `home` as its
/// `VECTORDATA_HOME`.
fn in_child(case: &str, home: &Path) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([case, "--exact", "--nocapture", "--test-threads=1"])
        .env(CASE, case)
        .env(HOME, home)
        .env("VECTORDATA_HOME", home)
        .env_remove(settings::OFFLINE_ENV)
        .output()
        .expect("run the child");
    let report = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{report}");
    assert!(report.contains("1 passed"), "the child case did not run:\n{report}");
}

/// The child's home, when this process is running `case`.
fn child_home(case: &str) -> Option<PathBuf> {
    (std::env::var(CASE).ok()? == case).then(|| std::env::var_os(HOME).map(PathBuf::from))?
}

/// **A program keeps its own cache.** With the user's settings naming a
/// cache, a program that sets its own fetches into it: the user's cache
/// is never created and `settings.yaml` is unchanged.
#[test]
fn a_program_keeps_its_own_cache() {
    let tmp = make_tmp();
    let (home, settings_text) = user_home(tmp.path());
    in_child("own_cache_in_a_child", &home);
    assert!(home.join("own-cache/owned/base.fvec").is_file(), "the data is in the program's cache");
    assert!(!home.join("user-cache").exists(), "the user's cache was not touched");
    assert_eq!(std::fs::read_to_string(home.join("settings.yaml")).unwrap(), settings_text);
}

#[test]
fn own_cache_in_a_child() {
    let Some(home) = child_home("own_cache_in_a_child") else { return };
    let own = home.join("own-cache");
    settings::set_cache_dir(&own).expect("the first choice is taken");
    settings::set_cache_dir(&own).expect("repeating it changes nothing");
    let other = home.join("other-cache");
    assert_eq!(
        settings::set_cache_dir(&other),
        Err(CacheDirConflict::AlreadySet { current: own.clone(), requested: other })
    );
    assert_eq!(settings::cache_dir().unwrap(), own);

    let server = serve(&home, "owned", 2.1e8);
    let group = vectordata::TestDataGroup::load(&format!("{}owned/", server.base_url())).unwrap();
    let view = group.profile("default").unwrap();
    view.fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent).unwrap();
    assert_eq!(view.base_vectors().unwrap().get(399).unwrap()[7], fvec_value(2.1e8, 8, 399, 7));
}

/// **A choice made after a dataset file was opened in the configured
/// cache is refused**, and the process keeps the cache it has:
/// switching midway would split its data between two caches. Asking
/// where the configured cache is does not count.
#[test]
fn a_late_cache_choice_is_refused() {
    let tmp = make_tmp();
    let (home, _) = user_home(tmp.path());
    in_child("late_choice_in_a_child", &home);
    assert!(home.join("user-cache/late/base.fvec").is_file());
    assert!(!home.join("own-cache").exists());
}

#[test]
fn late_choice_in_a_child() {
    let Some(home) = child_home("late_choice_in_a_child") else { return };
    let configured = settings::cache_dir().unwrap();
    assert_eq!(configured, home.join("user-cache"));
    let server = serve(&home, "late", 2.3e8);
    let group = vectordata::TestDataGroup::load(&format!("{}late/", server.base_url())).unwrap();
    let view = group.profile("default").unwrap();
    view.fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent).unwrap();

    let own = home.join("own-cache");
    let err = settings::set_cache_dir(&own).unwrap_err();
    assert_eq!(err, CacheDirConflict::ConfiguredCacheHoldsData { configured: configured.clone(), requested: own });
    assert!(err.to_string().contains("before the first dataset is opened"), "{err}");
    assert_eq!(settings::cache_dir().unwrap(), configured);
    settings::set_cache_dir(&configured).expect("the configured cache itself is no move");
}

/// **Catalogs and definitions may be loaded before the choice.** A
/// remote catalog and a dataset's `dataset.yaml` are loaded — their
/// copies kept in the configured cache — and only then does the
/// program set its own: the choice is taken, the dataset's files land
/// in it, and the copies kept from then on are kept there too.
#[test]
fn catalogs_may_be_loaded_before_the_choice() {
    let tmp = make_tmp();
    let (home, _) = user_home(tmp.path());
    in_child("catalog_first_in_a_child", &home);
    assert!(home.join("user-cache/.catalogs").is_dir(), "the first copy went to the configured cache");
    assert!(home.join("own-cache/.catalogs").is_dir(), "later copies go to the program's cache");
    assert!(home.join("own-cache/first/base.fvec").is_file());
    assert!(!home.join("user-cache/first/base.fvec").exists(), "no dataset bytes in the configured cache");
}

#[test]
fn catalog_first_in_a_child() {
    let Some(home) = child_home("catalog_first_in_a_child") else { return };
    let server = serve(&home, "first", 2.4e8);
    let sources = vectordata::catalog::CatalogSources::new().add_catalogs(&[server.base_url()]);
    let catalog = vectordata::catalog::Catalog::of(&sources);
    assert!(catalog.diagnostics().is_empty(), "{:?}", catalog.diagnostics());
    let group = catalog.open_spec("first:default").unwrap();

    let own = home.join("own-cache");
    settings::set_cache_dir(&own).expect("loading catalogs and definitions is not using the cache for data");
    group.fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent).unwrap();
    assert_eq!(group.view().unwrap().base_vectors().unwrap().get(5).unwrap()[2], fvec_value(2.4e8, 8, 5, 2));
    let again = vectordata::catalog::Catalog::of(&sources);
    assert!(again.diagnostics().is_empty(), "{:?}", again.diagnostics());
}

/// **`precache --cache-dir` fetches into that directory**, leaving the
/// configured cache and `settings.yaml` as they were.
#[test]
fn precache_fetches_into_the_cache_dir_it_is_given() {
    let tmp = make_tmp();
    let (home, settings_text) = user_home(tmp.path());
    let server = serve(tmp.path(), "flagged", 2.2e8);
    let own = tmp.path().join("flag-cache");
    let out = Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(["datasets", "precache", "--at", &server.base_url(), "flagged:default", "--cache-dir"])
        .arg(&own)
        .env("VECTORDATA_HOME", &home)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .env_remove(settings::OFFLINE_ENV)
        .output()
        .unwrap();
    let report = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{report}");
    assert!(own.join("flagged/base.fvec").is_file(), "{report}");
    assert!(!home.join("user-cache").exists(), "{report}");
    assert_eq!(std::fs::read_to_string(home.join("settings.yaml")).unwrap(), settings_text);
}
