// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Internal. What a [`DatasetConfig`] does beyond its own file: read from a URL,
//! and reconcile its shard declarations with the files on disk. The
//! descriptor is the API core's; these reach the network and the
//! filesystem, so they are vectordata's (`srd-crate-boundaries.md` §4).

use crate::dataset::DatasetConfig;

/// [`DatasetConfig`] read from anywhere vectordata reaches, and
/// reconciled with the files a pipeline wrote. Bring it into scope to
/// call these as methods.
pub trait DatasetConfigIo: Sized {
    /// Load the `dataset.yaml` a location names: a local directory or
    /// file, or a URL naming the dataset's directory or the file, read
    /// as [`TestDataGroup::load`](crate::TestDataGroup::load) reads it
    /// (auth, and the kept copy when the server cannot be reached). A
    /// file that states no `name` is named after its directory.
    fn load_from(location: &crate::Location) -> Result<Self, String>;

    /// Make every facet declaration describe the files that are
    /// actually there: a facet the pipeline wrote as a series gains its
    /// shard pattern, stride, count and record total. `base` is the
    /// directory declared paths are relative to. Returns how many
    /// declarations changed.
    fn reconcile_shard_declarations(&mut self, base: &std::path::Path) -> usize;
}

impl DatasetConfigIo for DatasetConfig {
    fn load_from(location: &crate::Location) -> Result<Self, String> {
        match location {
            crate::Location::Path(path) => {
                let file = if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                    path.clone()
                } else {
                    path.join("dataset.yaml")
                };
                DatasetConfig::load(&file)
            }
            crate::Location::Url(url) => {
                let text = crate::group::fetch_dataset_yaml(url.as_str()).map_err(|e| e.to_string())?;
                let mut config = DatasetConfig::parse(&text)
                    .map_err(|e| format!("Failed to parse the dataset.yaml at {url}: {e}"))?;
                if config.name.is_empty() {
                    let dir = if url.path().ends_with(".yaml") || url.path().ends_with(".yml") {
                        location.parent()
                    } else {
                        Some(location.clone())
                    };
                    config.name = dir
                        .and_then(|d| d.last_segment().map(str::to_string))
                        .unwrap_or_default();
                }
                Ok(config)
            }
        }
    }

    fn reconcile_shard_declarations(&mut self, base: &std::path::Path) -> usize {
        let mut changed = 0;
        for profile in self.profiles.profiles.values_mut() {
            for view in profile.views.values_mut() {
                // A **view-level** window is not disqualifying and is
                // deliberately not checked here: it is in facet
                // ordinals, so it means the same range whether the
                // facet is one file or five. Every sized profile
                // declares one, and skipping those would leave them
                // naming a filename nothing wrote while the unwindowed
                // profile named the series.
                //
                // A window on the **source** is a different thing — an
                // entry window, in that file's own ordinals (SH-67).
                // If the facet became a series, `source` is merely its
                // first entry, so carrying that range onto the pattern
                // would silently reinterpret it in a coordinate space
                // it was never written in. Left alone instead.
                if !view.source.window.is_empty() || view.source.namespace.is_some() {
                    continue;
                }
                let path = view.source.path.clone();
                if crate::dataset::shards::has_shard_field(&path) {
                    continue; // already declared as a series
                }
                let declared = base.join(&path);
                if declared.exists() {
                    continue;
                }
                let shards = crate::dataset::shards::discover_shards(&declared);
                if shards.is_empty() {
                    continue;
                }
                let Some(shape) = crate::dataset::shards::observe_series(&shards) else {
                    continue;
                };
                // The `source:` keeps whatever directory prefix it
                // had — and its window and namespace, which are
                // untouched — and gains the shard field (SH-47). The
                // prefix is kept as the declared text: a declaration
                // is `/`-separated on every platform, and rebuilding
                // it through `Path::join` would write `\` on Windows.
                let (dir, name) = match path.rsplit_once('/') {
                    Some((d, n)) => (Some(d), n),
                    None => (None, path.as_str()),
                };
                if name.is_empty() {
                    continue;
                }
                let (basename, ext) = match name.rsplit_once('.') {
                    Some((b, e)) => (b.to_string(), e.to_string()),
                    None => (name.to_string(), String::new()),
                };
                let pattern = crate::dataset::shards::shard_source_spec(&basename, &ext);
                let spec = match dir {
                    Some(d) => format!("{d}/{pattern}"),
                    None => pattern,
                };
                view.source.path = spec;
                view.shard_stride = Some(shape.stride);
                view.shard_count = Some(shape.shards);
                view.record_count = Some(shape.records);
                changed += 1;
            }
        }
        changed
    }
}

#[cfg(test)]
mod reconcile_shard_tests {
    use super::DatasetConfigIo;
    use crate::dataset::DatasetConfig;

    fn tmpdir() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// Write `n` records of dim 2 as shard `i` of `base/b.fvecs`.
    fn shard(dir: &std::path::Path, i: u32, n: usize) {
        std::fs::create_dir_all(dir.join("base")).unwrap();
        let mut bytes = Vec::new();
        for r in 0..n {
            bytes.extend_from_slice(&2i32.to_le_bytes());
            for k in 0..2 {
                bytes.extend_from_slice(&((r * 2 + k) as f32).to_le_bytes());
            }
        }
        let name = crate::dataset::shards::shard_name("b", "fvecs", i);
        std::fs::write(dir.join("base").join(name), bytes).unwrap();
    }

    fn config_of(yaml: &str) -> DatasetConfig {
        serde_yaml::from_str(yaml).expect("valid dataset")
    }

    /// **Saving is a fixpoint** (SH-67): a set under a layer inherits the
    /// sharded base as a series windowed through its own field, so the
    /// writer suppresses it as the parent's and a second save writes
    /// the same bytes. Before this, the set's base was cut through its
    /// source, emitted on one save and suppressed on the next, so the
    /// file alternated between two forms on every run.
    #[test]
    fn saving_a_layered_set_is_a_fixpoint() {
        let yaml = "format_version: 3\nname: t\nprofiles:\n  default:\n    base_vectors:\n      source: profiles/base/base__NNNN.fvecs\n      shard_stride: 100\n      shard_count: 5\n      record_count: 495\n    metadata_content: profiles/base/m.slab\n    query_vectors: q.fvec\n    neighbor_indices: g.ivec\n  10m:\n    base_count: 10\n    inherits: default\n    neighbor_indices: profiles/10m/g.ivec\n  10m-unfiltered:\n    base_count: 10\n    inherits: default\n    neighbor_indices: profiles/10m/g.ivec\n  10m-set:\n    base_count: 10\n    inherits: 10m-unfiltered\n    metadata_predicates: profiles/10m-set/p.slab\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dataset.yaml");
        std::fs::write(&path, yaml).unwrap();
        let config = DatasetConfig::load(&path).unwrap();
        let set = &config.profiles.profiles["10m-set"];
        let base = &set.views["base_vectors"];
        assert!(base.is_series());
        assert!(base.source.window.is_empty(), "a series is not cut through its source: {}", base.source);
        assert_eq!(base.window.as_ref().map(|w| (w.0[0].min_incl, w.0[0].max_excl)), Some((0, 10)));
        let once = config.to_expanded_yaml_string(&path).unwrap();
        assert!(!once.contains("base__NNNN.fvecs[0..10"), "{once}");
        std::fs::write(&path, &once).unwrap();
        let again = DatasetConfig::load(&path).unwrap().to_expanded_yaml_string(&path).unwrap();
        assert_eq!(again, once, "a second save writes the same bytes");
    }

    /// **A windowed view is reconciled too.** Sized profiles declare
    /// `source: …/b.fvecs` with a window, and skipping them would leave
    /// every one of them naming a file nothing wrote while the
    /// unwindowed profile named the series — the declaration
    /// disagreeing with itself.
    #[test]
    fn a_windowed_view_follows_the_series_and_keeps_its_window() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 10);
        shard(tmp.path(), 1, 10);
        shard(tmp.path(), 2, 4);

        let mut c = config_of(
            "name: d\nprofiles:\n  default:\n    base_vectors: base/b.fvecs\n\
             \x20 p100k:\n    base_vectors:\n      source: base/b.fvecs\n      \
             window: \"[0..10]\"\n",
        );
        assert_eq!(c.reconcile_shard_declarations(tmp.path()), 2, "both views");

        for profile in ["default", "p100k"] {
            let v = c.profiles.profiles[profile].views["base_vectors"].clone();
            assert_eq!(v.source.path, "base/b__NNNN.fvecs", "{profile}");
            assert_eq!(v.shard_stride, Some(10), "{profile}");
            assert_eq!(v.shard_count, Some(3), "{profile}");
            assert_eq!(v.record_count, Some(24), "{profile}");
        }
        // The window is untouched — it is in facet ordinals, and the
        // facet's ordinals did not move.
        let sized = &c.profiles.profiles["p100k"].views["base_vectors"];
        assert!(sized.window.is_some(), "the view-level window survived");
        assert!(c.profiles.profiles["default"].views["base_vectors"].window.is_none());
    }

    /// **An entry window on the source is left alone.** That range is
    /// in the file's own ordinals (SH-67), and `source` would be only
    /// the first entry of a series — so rewriting it onto the pattern
    /// would reinterpret it in a coordinate space it was never written
    /// in. The declaration stays as it is, and a reader still resolves
    /// the facet, rather than being handed a confidently wrong range.
    #[test]
    fn an_entry_window_on_the_source_is_not_rewritten() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 10);
        shard(tmp.path(), 1, 10);

        let mut c = config_of(
            "name: d\nprofiles:\n  default:\n    base_vectors: \"base/b.fvecs[0..5)\"\n",
        );
        assert!(
            !c.profiles.profiles["default"].views["base_vectors"]
                .source
                .window
                .is_empty(),
            "fixture must carry an entry window"
        );
        assert_eq!(c.reconcile_shard_declarations(tmp.path()), 0);
        assert_eq!(
            c.profiles.profiles["default"].views["base_vectors"].source.path,
            "base/b.fvecs",
        );
    }

    /// A facet whose file is still there is left alone — the rule is
    /// "the file is gone and shards are present", not "shards exist".
    #[test]
    fn a_facet_still_present_as_one_file_is_untouched() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 10);
        // …and the plain file exists too.
        std::fs::write(tmp.path().join("base/b.fvecs"), [0u8; 12]).unwrap();

        let mut c = config_of("name: d\nprofiles:\n  default:\n    base_vectors: base/b.fvecs\n");
        assert_eq!(c.reconcile_shard_declarations(tmp.path()), 0);
        assert_eq!(
            c.profiles.profiles["default"].views["base_vectors"].source.path,
            "base/b.fvecs"
        );
    }

    /// A declaration already naming a series is not rewritten again,
    /// so the pass is idempotent across repeated runs.
    #[test]
    fn reconciling_twice_changes_nothing_the_second_time() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 10);
        shard(tmp.path(), 1, 5);

        let mut c = config_of("name: d\nprofiles:\n  default:\n    base_vectors: base/b.fvecs\n");
        assert_eq!(c.reconcile_shard_declarations(tmp.path()), 1);
        assert_eq!(c.reconcile_shard_declarations(tmp.path()), 0, "idempotent");
    }

    /// A facet that is simply missing — no file, no shards — is left
    /// as declared rather than being blamed on sharding.
    #[test]
    fn a_missing_facet_is_not_mistaken_for_a_series() {
        let tmp = tmpdir();
        std::fs::create_dir_all(tmp.path().join("base")).unwrap();
        let mut c = config_of("name: d\nprofiles:\n  default:\n    base_vectors: base/b.fvecs\n");
        assert_eq!(c.reconcile_shard_declarations(tmp.path()), 0);
    }
}

#[cfg(test)]
mod load_from_tests {
    use super::DatasetConfigIo;
    use crate::dataset::DatasetConfig;

    #[test]
    fn a_dataset_without_a_name_takes_its_directory_name() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("ada-1k");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("dataset.yaml"), "profiles:\n  default:\n    base_vectors: base.fvec\n").unwrap();
        let config = DatasetConfig::load(&dir.join("dataset.yaml")).expect("a nameless dataset loads");
        assert_eq!(config.name, "ada-1k");
        let by_location = DatasetConfig::load_from(&crate::Location::from_path(&dir)).unwrap();
        assert_eq!(by_location.name, "ada-1k");
    }
}
