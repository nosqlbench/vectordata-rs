// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! What older and newer builds make of each other's `dataset.yaml`.
//!
//! Two directions, and they fail differently:
//!
//! * **Backwards** — this build reading a dataset written before
//!   sharding existed, and this build *writing* one that a pre-sharding
//!   build can still read. The second half is the one that rots
//!   silently: a v2-only key leaking into an unsharded output breaks
//!   every older reader, and every test in the suite still passes.
//! * **Forwards** — a pre-sharding build reading a sharded dataset. It
//!   cannot succeed, and the requirement is that it cannot *appear* to
//!   succeed either (SH-71): resolving a series to its first shard
//!   would answer with a fraction of the facet and no error.
//!
//! The forward direction is tested against a faithful copy of the
//! pre-sharding `FacetConfig`, reproduced from the commit that
//! introduced sharding. Serde decides what an old build accepts from
//! the type alone, so the copy answers the question the old binary
//! would have.
//!
//! See `docs/design/srd-multifile-facet-shards.md` and
//! `docs/design/srd-dataset-format-version.md`.

use serde::Deserialize;
use vectordata::dataset::Sharding;

/// The pre-sharding facet declaration, verbatim: an untagged pair of a
/// bare filename and a `{source, window}` object. No shard fields, and
/// — the part that decides the outcome — no `deny_unknown_fields`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum V1FacetConfig {
    Simple(String),
    Detailed {
        source: String,
        #[serde(default)]
        #[allow(dead_code)]
        window: Option<String>,
    },
}

impl V1FacetConfig {
    /// What an old build would have opened.
    fn source(&self) -> &str {
        match self {
            Self::Simple(s) | Self::Detailed { source: s, .. } => s,
        }
    }
}

fn write_fvec(path: &std::path::Path, dim: i32, records: usize, first: usize) {
    use std::io::Write as _;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for i in 0..records {
        f.write_all(&dim.to_le_bytes()).unwrap();
        for d in 0..dim {
            let v = (first + i) as f32 + d as f32 / 100.0;
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    f.flush().unwrap();
}

// ── forwards: an old build meeting a series ────────────────────────

/// **A uniform series resolves to the pattern, never to a shard**
/// (SH-71, SH-74).
///
/// An old build matches the `Detailed` arm — untagged serde ignores the
/// shard keys it has no field for — and comes away with
/// `base__NNNN.fvec`. That name is on no filesystem, so the open fails
/// by name. The alternative the design had to avoid is an old build
/// resolving the same declaration to `base__0000.fvec` and reporting a
/// fifth of the dataset as the whole of it.
#[test]
fn an_old_build_resolves_a_uniform_series_to_a_name_that_cannot_open() {
    let tmp = tempfile::tempdir().unwrap();
    for s in 0..5 {
        write_fvec(&tmp.path().join(format!("base__{s:04}.fvec")), 4, 100, s * 100);
    }

    let facet: V1FacetConfig = serde_yaml::from_str(
        "source: base__NNNN.fvec\nshard_stride: 100\nshard_count: 5\nrecord_count: 500\n",
    )
    .expect("an old build parses the mapping — it has no field to reject");

    assert_eq!(facet.source(), "base__NNNN.fvec");
    assert!(
        !tmp.path().join(facet.source()).exists(),
        "the resolved name must not be openable, or the failure is silent"
    );
    // And specifically not a shard: the shard files are right there.
    assert!(tmp.path().join("base__0000.fvec").exists());
    assert_ne!(facet.source(), "base__0000.fvec");
}

/// **An explicit series is refused at parse** (SH-71).
///
/// The explicit form spells `source:` as a sequence, which matches
/// neither old arm. An old build stops at the declaration rather than
/// at a missing file — an even earlier and clearer failure than the
/// uniform form's.
#[test]
fn an_old_build_cannot_even_parse_an_explicit_series() {
    let err = serde_yaml::from_str::<V1FacetConfig>(
        "source:\n  - part_a.fvec=100\n  - part_b.fvec=100\nrecord_count: 200\n",
    )
    .expect_err("a sequence source matches no pre-sharding arm");
    let _ = err;
}

/// **A collapsed single shard is readable by an old build** (SH-4,
/// SH-83) — the whole point of collapsing.
///
/// Output that fits one shard is spelled as a plain filename, so
/// forwards compatibility becomes "supports sharded facets" rather than
/// "supports sharded descriptions": a dataset that never needed
/// splitting is not made unreadable by a writer that knows how to
/// split.
#[test]
fn an_old_build_reads_a_collapsed_single_shard_output() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_fvec(&src.join("base_vectors.fvec"), 4, 60, 0);
    std::fs::write(
        src.join("dataset.yaml"),
        "name: src\nprofiles:\n  default:\n    base_vectors: base_vectors.fvec\n",
    )
    .unwrap();

    // A stride large enough that the output is one shard.
    let out = tmp.path().join("out");
    assert_eq!(
        vectordata::datasets::derive::run(
            src.to_str().unwrap(),
            "default",
            &out,
            "",
            &[],
            &[],
            Some("derived"),
            true,
            Sharding::Stride(1000),
        ),
        0
    );

    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    let doc: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
    let raw = doc["profiles"]["default"]["base_vectors"].clone();
    let facet: V1FacetConfig =
        serde_yaml::from_value(raw).expect("a collapsed output parses as pre-sharding");

    // The name an old build resolves is a file that exists and holds
    // the whole facet.
    let path = out.join(facet.source());
    assert!(path.exists(), "old build resolves {:?}", facet.source());
    assert!(!facet.source().contains("NNNN"));
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        60 * (4 + 4 * 4),
        "the one file is the whole facet, not a fraction"
    );
}

// ── backwards: this build and pre-sharding datasets ────────────────

/// **An unsharded output carries no v2-only key** (V-5).
///
/// Absence of `format_version` is the headline, but any shard key
/// leaking into an unsharded declaration breaks an old reader just as
/// thoroughly — `record_count` is ignored by an untagged `Detailed`,
/// while a `source` promoted to a one-element sequence is fatal. The
/// assertion is on the whole document rather than on one key, because
/// the failure mode is a key nobody thought to check.
#[test]
fn an_unsharded_output_is_spelled_exactly_as_it_always_was() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_fvec(&src.join("base_vectors.fvec"), 4, 40, 0);
    std::fs::write(
        src.join("dataset.yaml"),
        "name: src\nprofiles:\n  default:\n    base_vectors: base_vectors.fvec\n",
    )
    .unwrap();

    let out = tmp.path().join("out");
    assert_eq!(
        vectordata::datasets::derive::run(
            src.to_str().unwrap(),
            "default",
            &out,
            "",
            &[],
            &[],
            Some("derived"),
            true,
            Sharding::Whole,
        ),
        0
    );

    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    for key in [
        "format_version",
        "shard_stride",
        "shard_count",
        "record_count",
    ] {
        assert!(!yaml.contains(key), "v2-only key `{key}` leaked:\n{yaml}");
    }

    // And every facet still parses under the pre-sharding types, which
    // is the property the key list is a proxy for.
    let doc: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
    let profiles = doc["profiles"].as_mapping().expect("profiles");
    let mut checked = 0;
    for (_, profile) in profiles {
        for (name, raw) in profile.as_mapping().expect("profile mapping") {
            let name = name.as_str().unwrap_or("");
            // Only facet-shaped entries; a profile also carries scalars
            // like `name` and structured non-facet keys.
            if !matches!(raw, serde_yaml::Value::String(_) | serde_yaml::Value::Mapping(_)) {
                continue;
            }
            if let Ok(f) = serde_yaml::from_value::<V1FacetConfig>(raw.clone()) {
                assert!(
                    !f.source().contains("NNNN"),
                    "facet `{name}` resolved to a shard pattern"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no facet was checked:\n{yaml}");
}

/// **Every pre-sharding spelling still loads and reads.**
///
/// The suite exercises v1 datasets everywhere, but incidentally. This
/// pins the four spellings a v1 `dataset.yaml` may use — bare name,
/// `{source}`, `{source, window}`, and the `path[a..b)` suffix — against
/// one dataset, so a change to the untagged enum's arm order shows up
/// as a failure here rather than as a subtly wrong window somewhere
/// else.
#[test]
fn every_pre_sharding_facet_spelling_still_reads() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 4, 100, 0);
    write_fvec(&ds.join("query.fvec"), 4, 100, 0);
    write_fvec(&ds.join("detail.fvec"), 4, 100, 0);
    write_fvec(&ds.join("suffix.fvec"), 4, 100, 0);

    std::fs::write(
        ds.join("dataset.yaml"),
        "name: v1\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20   query_vectors:\n\
        \x20     source: query.fvec\n\
        \x20   query_terms:\n\
        \x20     source: detail.fvec\n\
        \x20     window: 10..40\n\
        \x20   query_filters: suffix.fvec[20..50)\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    let bare = view.base_vectors().unwrap();
    assert_eq!(bare.count(), 100);
    assert_eq!(bare.get(0).unwrap()[0], 0.0);

    let object = view.facet("query_vectors").unwrap();
    assert_eq!(object.count(), 100);

    let windowed = view.facet("query_terms").unwrap();
    assert_eq!(windowed.count(), 30, "an explicit window field clips");
    assert_eq!(windowed.get(0).unwrap()[0], 10.0);

    let suffixed = view.facet("query_filters").unwrap();
    assert_eq!(suffixed.count(), 30, "a source suffix clips");
    assert_eq!(suffixed.get(0).unwrap()[0], 20.0);
}

// ── the two loaders ────────────────────────────────────────────────

/// **Both loaders read the version identically** (V-12).
///
/// A dataset accepted by one route and refused by the other makes the
/// *transport* decide whether it is readable: open it as a local
/// `dataset.yaml` and it fails, reach it through a catalog and it
/// loads, then dies on a type error or a missing `__NNNN` file — the
/// diagnosis the field exists to replace.
///
/// Asserted as agreement rather than as two separate expectations, so
/// the test fails whichever side drifts.
#[test]
fn both_loaders_agree_about_every_version_case() {
    let cases = [
        (
            "from the future",
            "format_version: 99\nname: f\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
            false,
        ),
        (
            "understating its content",
            "format_version: 1\nname: u\nprofiles:\n  default:\n    base_vectors:\n      \
             source: b__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
             record_count: 250\n",
            false,
        ),
        (
            "unannotated but sharded",
            "name: a\nprofiles:\n  default:\n    base_vectors:\n      \
             source: b__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
             record_count: 250\n",
            true,
        ),
        (
            "stating more than it needs",
            "format_version: 2\nname: g\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
            true,
        ),
        (
            "plain v1",
            "name: p\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
            true,
        ),
    ];

    for (label, yaml, expected) in cases {
        let client = serde_yaml::from_str::<vectordata::model::DatasetConfig>(yaml).is_ok();
        let catalog = serde_yaml::from_str::<vectordata::dataset::DatasetConfig>(yaml).is_ok();
        assert_eq!(
            client, catalog,
            "the loaders disagree about a dataset {label}: \
             client={client}, catalog={catalog}"
        );
        assert_eq!(client, expected, "a dataset {label} should load = {expected}");
    }
}

/// **A catalog entry carries the version** (V-13), so a consumer can
/// refuse before fetching rather than after.
#[test]
fn a_generated_catalog_entry_carries_the_datasets_version() {
    let sharded = "format_version: 2\nname: s\nprofiles:\n  default:\n    base_vectors:\n      \
                   source: b__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
                   record_count: 250\n";
    let cfg: vectordata::dataset::DatasetConfig = serde_yaml::from_str(sharded).unwrap();
    assert_eq!(cfg.format_version, 2);

    let layout = vectordata::dataset::CatalogLayout {
        format_version: cfg.format_version,
        attributes: cfg.attributes.clone(),
        profiles: cfg.profiles.clone(),
    };
    let json = serde_json::to_string(&layout).unwrap();
    assert!(
        json.contains("\"format_version\":2"),
        "the version must reach the catalog: {json}"
    );

    // And version 1 adds no key, so a catalog of pre-versioning
    // datasets is unchanged by this.
    let plain: vectordata::dataset::DatasetConfig =
        serde_yaml::from_str("name: p\nprofiles:\n  default:\n    base_vectors: b.fvec\n").unwrap();
    let layout = vectordata::dataset::CatalogLayout {
        format_version: plain.format_version,
        attributes: None,
        profiles: plain.profiles.clone(),
    };
    let json = serde_json::to_string(&layout).unwrap();
    assert!(!json.contains("format_version"), "{json}");
}

/// **A save states the version the content needs, not the one it was
/// loaded with** (V-4).
///
/// Folding it from the declarations is what keeps the field honest
/// through an edit: a dataset that gained a series says so, and the key
/// never appears on one that does not need it.
#[test]
fn a_saved_dataset_states_the_version_its_content_needs() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("dataset.yaml");

    let sharded: vectordata::dataset::DatasetConfig = serde_yaml::from_str(
        "name: s\nprofiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n      \
         shard_stride: 100\n      shard_count: 3\n      record_count: 250\n",
    )
    .unwrap();
    let out = sharded.to_expanded_yaml_string(&path).unwrap();
    assert!(
        out.contains("format_version: 2"),
        "an unannotated sharded dataset states what it needs on save:\n{out}"
    );

    let plain: vectordata::dataset::DatasetConfig =
        serde_yaml::from_str("name: p\nprofiles:\n  default:\n    base_vectors: b.fvec\n").unwrap();
    let out = plain.to_expanded_yaml_string(&path).unwrap();
    assert!(
        !out.contains("format_version"),
        "an unsharded dataset must not acquire the key by being saved:\n{out}"
    );
}
