// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Every file a dataset publishes about itself must say what its
//! `dataset.yaml` says.
//!
//! `dataset.yaml` is the authority; the catalogs (`catalog.json`,
//! `catalog.yaml`, `knn_entries.yaml`) and `dataset.json` are projections
//! of it, written by the finalize steps. A projection that adds,
//! drops or moves a facet misleads every consumer that lists before it
//! opens, so an end-to-end test that builds a dataset asserts the
//! projections against the reader's view of the `dataset.yaml`: the same
//! view a program opening the dataset gets.
//!
//! The two sides are reached by independent paths. The reader parses
//! `dataset.yaml` (`TestDataGroup::load`). The projection is read back
//! from the file the generator wrote, through the parser a consumer uses,
//! and projected into the same reader view.

use std::path::Path;

use vectordata::TestDataGroup;
use vectordata::catalog::{Catalog, CatalogSources};
use vectordata::model::DatasetConfig as ReaderView;

/// Mark `dir` as a dataset to publish, so the `generate-catalog` step
/// writes its catalogs (the generator lists only datasets with this
/// sentinel).
pub fn mark_publishable(dir: &Path) {
    std::fs::write(dir.join(".publish"), "").expect("write .publish");
}

/// Assert every projection of the dataset at `dir` mirrors its
/// `dataset.yaml`. The catalogs and `dataset.json` must exist: a
/// publishable dataset whose finalize pass wrote none of them is itself
/// the failure.
pub fn assert_projections_mirror_dataset_yaml(dir: &Path) {
    let reader = TestDataGroup::load(&dir.to_string_lossy())
        .unwrap_or_else(|e| panic!("open {} by its dataset.yaml: {e}", dir.display()));
    let name = vectordata::dataset::DatasetConfig::load(&dir.join("dataset.yaml"))
        .unwrap_or_else(|e| panic!("load {}/dataset.yaml: {e}", dir.display()))
        .name;
    let expected = as_value(reader.config());

    // catalog.json, through the catalog parser a consumer uses.
    let sources = CatalogSources::new().add_catalogs(&[dir.to_string_lossy().into_owned()]);
    let catalog = Catalog::of(&sources);
    assert!(catalog.diagnostics().is_empty(), "catalog at {}: {:?}", dir.display(), catalog.diagnostics());
    let entry = catalog
        .lookup(&name)
        .unwrap_or_else(|e| panic!("catalog at {} does not list '{name}': {e}", dir.display()));
    let listed = ReaderView::from_parts(
        entry.format_version(),
        entry.attributes(),
        entry.profile_tags(),
        entry.profiles(),
    );
    assert_same(&expected, &as_value(&listed), "catalog.json", dir);

    // catalog.yaml is the same document as catalog.json.
    let json: serde_json::Value = serde_json::from_str(&read(dir, "catalog.json")).expect("parse catalog.json");
    let yaml: serde_json::Value = serde_yaml::from_str(&read(dir, "catalog.yaml")).expect("parse catalog.yaml");
    assert_eq!(json, yaml, "catalog.yaml and catalog.json differ in {}", dir.display());

    // dataset.json, read as a description.
    let published = vectordata::dataset::DatasetConfig::parse(&read(dir, "dataset.json")).expect("parse dataset.json");
    assert_same(&expected, &as_value(&ReaderView::from_description(&published)), "dataset.json", dir);

    // knn_entries.yaml lists exactly the profiles whose base, query and
    // ground truth are each a whole file, at those files; a profile
    // needing a window, a series or a namespace is not listed, and with
    // none to list there is no file.
    let expected: std::collections::BTreeMap<String, [String; 3]> = reader
        .config()
        .profiles
        .iter()
        .filter_map(|(profile, p)| {
            let facets: std::collections::BTreeMap<_, _> = p.facets().into_iter().collect();
            let whole = |facet: &str| whole_file(facets.get(facet)?).map(|f| dir.join(f).to_string_lossy().into_owned());
            Some((profile.clone(), [whole("base_vectors")?, whole("query_vectors")?, whole("neighbor_indices")?]))
        })
        .collect();
    let knn_path = dir.join("knn_entries.yaml");
    if expected.is_empty() {
        assert!(!knn_path.exists(), "knn_entries.yaml in {} lists profiles it cannot state exactly", dir.display());
        return;
    }
    let knn = vectordata::catalog::knn_entries::parse_knn_entries_yaml(&read(dir, "knn_entries.yaml"), &dir.to_string_lossy())
        .expect("parse knn_entries.yaml");
    let mut listed = std::collections::BTreeMap::new();
    for knn_entry in &knn {
        assert_eq!(knn_entry.name(), name, "knn_entries.yaml in {} names another dataset", dir.display());
        for (profile, p) in &knn_entry.profiles().profiles {
            let file = |facet: &str| p.view(facet).map(|v| v.path().to_string()).unwrap_or_default();
            listed.insert(profile.clone(), [file("base_vectors"), file("query_vectors"), file("neighbor_indices")]);
        }
    }
    assert_eq!(listed, expected, "knn_entries.yaml in {} does not mirror dataset.yaml", dir.display());
}

/// The file a facet is, when it is one whole file: no series, no
/// window, no namespace inside it.
fn whole_file(facet: &vectordata::model::FacetConfig) -> Option<&str> {
    let source = facet.source()?;
    let windowed = facet.window().is_some() || source.ends_with([']', ')']);
    (!windowed && !source.contains(':')).then_some(source)
}


/// The reader view as a value to compare, with each facet in one form.
///
/// A file's window can be written on its path (`b.fvec[0..100]`) or as
/// a field (`{source: b.fvec, window: "[0..100]"}`), and `[0..100)` and
/// `[0..100]` are two spellings of the same bounds. Which form a writer
/// chose is not what a dataset contains, so each facet is compared as its
/// file (or pattern, with its series fields) and its bounds.
fn as_value(view: &ReaderView) -> serde_json::Value {
    let mut value = serde_json::to_value(view).expect("serialize the reader view");
    if let Some(profiles) = value.get_mut("profiles").and_then(|p| p.as_object_mut()) {
        for profile in profiles.values_mut().filter_map(|p| p.as_object_mut()) {
            for (key, facet) in profile.iter_mut() {
                if !matches!(key.as_str(), "base_count" | "maxk" | "partition" | "attributes" | "inherits") {
                    *facet = one_form(facet);
                }
            }
        }
    }
    value
}

/// A facet as `{source, window, ..series fields}`, the window as `lo..hi`
/// and absent when there is none.
fn one_form(facet: &serde_json::Value) -> serde_json::Value {
    let mut fields = match facet {
        serde_json::Value::String(s) => serde_json::Map::from_iter([("source".to_string(), s.clone().into())]),
        serde_json::Value::Object(m) => m.clone(),
        other => return other.clone(),
    };
    let mut window = fields.remove("window").and_then(|w| w.as_str().map(bounds));
    if let Some(source) = fields.get("source").and_then(|s| s.as_str()).map(str::to_string)
        && let Some(open) = source.rfind('[')
        && source.ends_with([']', ')'])
    {
        window = window.or(Some(bounds(&source[open..])));
        fields.insert("source".into(), source[..open].to_string().into());
    }
    if let Some(window) = window {
        fields.insert("window".into(), window.into());
    }
    serde_json::Value::Object(fields)
}

/// `[0..100]`, `[0..100)` and `0..100` as `0..100`.
fn bounds(window: &str) -> String {
    window.trim().trim_start_matches(['[', '(']).trim_end_matches([']', ')']).replace(' ', "")
}

fn assert_same(expected: &serde_json::Value, actual: &serde_json::Value, what: &str, dir: &Path) {
    if expected != actual {
        panic!(
            "{what} in {} does not mirror dataset.yaml\n--- dataset.yaml ---\n{}\n--- {what} ---\n{}",
            dir.display(),
            serde_json::to_string_pretty(expected).unwrap(),
            serde_json::to_string_pretty(actual).unwrap(),
        );
    }
}

fn read(dir: &Path, file: &str) -> String {
    std::fs::read_to_string(dir.join(file))
        .unwrap_or_else(|e| panic!("{} was not written in {}: {e}", file, dir.display()))
}
