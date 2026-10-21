// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Attribute overlays (`crate::overlay`): a host's statement of a
//! dataset's attributes, seen by every reader of it — the catalog
//! entry, the opened view, `describe` — each with its source, and
//! checked against the facets as they open
//! (`docs/design/srd-loose-files-overlays-integrity.md` §4).

mod support;

use std::path::Path;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use support::fixtures::{write_catalogs, write_fvec};
use vectordata::catalog::{Catalog, CatalogSources};
use vectordata::overlay::{AttributeSource, Overlay, OverlayKind};
use vectordata::{DistanceFunction, TestDataViewExt};

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

/// A `knn_entries.yaml` catalog at `dir` listing `name`, which states no
/// distance function, as such catalogs never do.
fn knn_catalog(dir: &Path, name: &str) {
    std::fs::write(
        dir.join("knn_entries.yaml"),
        format!("_defaults:\n  base_url: {}\n\"{name}:default\":\n  base: base.fvecs\n  query: query.fvecs\n  gt: gt.ivecs\n", dir.display()),
    )
    .unwrap();
}

/// A configuration directory whose `overlays.yaml` is `text`.
fn home_with_overlay(root: &Path, text: &str) -> std::path::PathBuf {
    let home = root.join("home");
    std::fs::create_dir_all(home.join("cache")).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", home.join("cache").display())).unwrap();
    std::fs::write(home.join("overlays.yaml"), text).unwrap();
    home
}

/// **A user overlay states a knn_entries dataset's metric**, and the
/// catalog entry and the opened view both read it, sourced to the file.
#[test]
fn a_user_overlay_gives_a_knn_entries_dataset_its_metric() {
    let tmp = tempfile::tempdir().unwrap();
    let cat = tmp.path().join("cat");
    std::fs::create_dir_all(&cat).unwrap();
    knn_catalog(&cat, "ada002-100k");
    let home = home_with_overlay(tmp.path(), "datasets:\n  ada002-100k:\n    distance_function: DOT_PRODUCT\n");
    let sources = CatalogSources::new().configure(home.to_str().unwrap()).add_catalogs(&[cat.display().to_string()]);
    let catalog = Catalog::of(&sources);

    let entry = catalog.find_exact("ada002-100k").unwrap();
    assert_eq!(entry.attributes().and_then(|a| a.distance()), Some(DistanceFunction::DotProduct));
    assert_eq!(entry.own_attributes().and_then(|a| a.distance()), None, "the listing itself says nothing");
    assert!(
        matches!(entry.attribute_source("distance_function"), Some(AttributeSource::Overlay { kind: OverlayKind::User, ref file }) if file.ends_with("overlays.yaml"))
    );

    let view = catalog.open_spec("ada002-100k").unwrap().view().unwrap();
    assert_eq!(view.distance(), Some(DistanceFunction::DotProduct));
}

/// **Precedence**: a program overlay is beaten by the user's, and a
/// catalog-qualified block beats a bare one — for that catalog only.
#[test]
fn the_user_beats_a_program_and_a_catalog_block_beats_a_bare_one() {
    let tmp = tempfile::tempdir().unwrap();
    let (a, b) = (tmp.path().join("a"), tmp.path().join("b"));
    for d in [&a, &b] {
        std::fs::create_dir_all(d).unwrap();
    }
    knn_catalog(&a, "prec-ds");
    knn_catalog(&b, "prec-other");
    let home = home_with_overlay(
        tmp.path(),
        "datasets:\n  prec-ds:\n    distance_function: COSINE\ncatalogs:\n  b:\n    prec-other:\n      model: from-catalog-block\n",
    );
    let sources = CatalogSources::new()
        .with_overlay(Overlay::program("prog", "datasets:\n  prec-ds:\n    distance_function: L2\n    model: prog-model\n  prec-other:\n    model: prog-model\n").unwrap())
        .unwrap()
        .configure(home.to_str().unwrap())
        .add_named_catalog("a", &a.display().to_string())
        .add_named_catalog("b", &b.display().to_string());
    let catalog = Catalog::of(&sources);
    let ds = catalog.find_exact("prec-ds").unwrap();
    assert_eq!(ds.attributes().and_then(|x| x.distance()), Some(DistanceFunction::Cosine), "the user's over the program's");
    assert_eq!(ds.attributes().and_then(|x| x.model.clone()).as_deref(), Some("prog-model"), "what the user does not state, the program's");
    assert!(matches!(ds.attribute_source("model"), Some(AttributeSource::Overlay { kind: OverlayKind::Program(ref p), .. }) if p == "prog"));
    let other = catalog.find_exact("prec-other").unwrap();
    assert_eq!(other.attributes().and_then(|x| x.model.clone()).as_deref(), Some("from-catalog-block"));
}

/// **A consumer's namespaced keys read back** through the entry and
/// the group, and an unknown bare key is warned about and ignored.
#[test]
fn namespaced_keys_read_back_and_unknown_bare_keys_are_ignored() {
    let tmp = tempfile::tempdir().unwrap();
    let cat = tmp.path().join("cat");
    std::fs::create_dir_all(&cat).unwrap();
    knn_catalog(&cat, "ns-ds");
    let home = home_with_overlay(tmp.path(), "datasets:\n  ns-ds:\n    fvec.ncentroids: 300\n    fvec.groups: [a, b]\n    colour: blue\n");
    let sources = CatalogSources::new().configure(home.to_str().unwrap()).add_catalogs(&[cat.display().to_string()]);
    assert!(sources.diagnostics().iter().any(|d| d.to_string().contains("'colour' for 'ns-ds' is not a dataset attribute")));
    let catalog = Catalog::of(&sources);
    let entry = catalog.find_exact("ns-ds").unwrap();
    assert_eq!(entry.attribute("fvec.ncentroids").and_then(|v| v.as_u64()), Some(300));
    assert_eq!(entry.attributes_in("fvec").len(), 2);
    assert_eq!(entry.attribute("colour"), None);
    let group = catalog.open_spec("ns-ds").unwrap();
    assert_eq!(group.group().attribute("fvec.ncentroids").and_then(|v| v.as_u64()), Some(300));
}

/// A local dataset `name` under `root`: 10 dim-4 vectors as its base,
/// 3 queries, listed in a catalog at `root`.
fn local_dataset(root: &Path, name: &str) {
    let ds = root.join(name);
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 10, 4, 1.0);
    write_fvec(&ds.join("query.fvec"), 3, 4, 2.0);
    std::fs::write(ds.join("dataset.yaml"), format!("name: {name}\nprofiles:\n  default:\n    base_vectors: base.fvec\n    query_vectors: query.fvec\n")).unwrap();
    write_catalogs(root, &[name]);
}

/// **An overlay's count that the base contradicts is a warning, once,
/// and stays in force** (OV-5); verification reports it as V7.
#[test]
fn a_contradicted_overlay_count_warns_once_and_stays_in_force() {
    capture();
    let tmp = tempfile::tempdir().unwrap();
    let cat = tmp.path().join("cat");
    local_dataset(&cat, "wrong-count");
    let home = home_with_overlay(tmp.path(), "datasets:\n  wrong-count:\n    base_count: 12\n");
    let sources = CatalogSources::new().configure(home.to_str().unwrap()).add_catalogs(&[cat.display().to_string()]);
    let selection = Catalog::of(&sources).open_spec("wrong-count").unwrap();
    let view = selection.view().unwrap();
    for _ in 0..3 {
        assert_eq!(view.base_vectors_as::<f32>().unwrap().count(), 10);
    }
    let warned = warnings_naming("'wrong-count'");
    assert_eq!(warned.len(), 1, "{warned:?}");
    assert!(warned[0].contains("states base_count: 12") && warned[0].contains("base_vectors holding 10") && warned[0].contains("in force"), "{warned:?}");
    assert_eq!(selection.group().attribute("base_count").and_then(|v| v.as_u64()), Some(12), "the overlay's value stays");
    let err = view.verify(&mut vectordata::verify::NoProgress).unwrap_err().to_string();
    assert!(err.contains("V7 base_vectors") && err.contains("base_count attribute states 12"), "{err}");
}

/// **An overlay's distance the dataset declares otherwise is a
/// warning**, at the merge, before any facet opens.
#[test]
fn an_overlay_distance_the_dataset_contradicts_is_a_warning() {
    capture();
    let tmp = tempfile::tempdir().unwrap();
    let cat = tmp.path().join("cat");
    local_dataset(&cat, "declared-l2");
    let yaml = cat.join("declared-l2/dataset.yaml");
    let text = std::fs::read_to_string(&yaml).unwrap().replace("profiles:", "attributes:\n  distance_function: L2\nprofiles:");
    std::fs::write(&yaml, text).unwrap();
    write_catalogs(&cat, &["declared-l2"]);
    let home = home_with_overlay(tmp.path(), "datasets:\n  declared-l2:\n    distance_function: COSINE\n");
    let sources = CatalogSources::new().configure(home.to_str().unwrap()).add_catalogs(&[cat.display().to_string()]);
    let catalog = Catalog::of(&sources);
    assert_eq!(catalog.find_exact("declared-l2").unwrap().attributes().and_then(|a| a.distance()), Some(DistanceFunction::Cosine));
    let warned = warnings_naming("'declared-l2'");
    assert!(warned.iter().any(|w| w.contains("distance_function: COSINE") && w.contains("the dataset's own L2")), "{warned:?}");
}

fn vectordata_bin(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(args)
        .env("VECTORDATA_HOME", home)
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .output()
        .expect("run vectordata")
}

/// **`config overlay set` keeps the file's comments, and `describe`
/// prints the overlay's value with its source**, through `--at` too.
#[test]
fn config_overlay_set_keeps_comments_and_describe_shows_the_source() {
    let tmp = tempfile::tempdir().unwrap();
    let cat = tmp.path().join("cat");
    std::fs::create_dir_all(&cat).unwrap();
    knn_catalog(&cat, "cli-ds");
    let home = home_with_overlay(tmp.path(), "# my notes on datasets\n");
    let out = vectordata_bin(&home, &["config", "overlay", "set", "cli-ds", "distance_function", "DOT_PRODUCT"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = vectordata_bin(&home, &["config", "overlay", "set", "cli-ds", "fvec.ncentroids", "300"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = std::fs::read_to_string(home.join("overlays.yaml")).unwrap();
    assert_eq!(text, "# my notes on datasets\n\ndatasets:\n  cli-ds:\n    distance_function: DOT_PRODUCT\n    fvec.ncentroids: 300\n");

    for at in [false, true] {
        let cat_s = cat.display().to_string();
        let mut args = vec!["datasets", "describe"];
        if at {
            args.extend(["--at", &cat_s]);
        } else {
            args.extend(["--catalog", &cat_s]);
        }
        args.push("cli-ds");
        let out = vectordata_bin(&home, &args);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{stdout}{}", String::from_utf8_lossy(&out.stderr));
        assert!(stdout.contains("DOT_PRODUCT  (overlay ") && stdout.contains("overlays.yaml)"), "--at {at}: {stdout}");
        assert!(stdout.contains("fvec.ncentroids") && stdout.contains("300"), "--at {at}: {stdout}");
    }

    let out = vectordata_bin(&home, &["config", "overlay", "unset", "cli-ds", "fvec.ncentroids"]);
    assert!(out.status.success());
    assert!(!std::fs::read_to_string(home.join("overlays.yaml")).unwrap().contains("ncentroids"));
    assert!(!vectordata_bin(&home, &["config", "overlay", "set", "cli-ds", "colour", "blue"]).status.success());
}
