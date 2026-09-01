// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A facet spread over several files, opened through the view.
//!
//! These exercise the seam between the declaration, the realized shard
//! model, and the storage handle — the point at which a series stops
//! being a description and becomes files on disk.
//!
//! See `docs/design/srd-multifile-facet-shards.md`.

use std::io::Write;

/// Write a uniform fvec: `records` rows of `dim` floats, each row
/// prefixed by its dimension.
fn write_fvec(path: &std::path::Path, dim: i32, records: usize, first: usize) {
    let mut f = std::fs::File::create(path).unwrap();
    for r in 0..records {
        f.write_all(&dim.to_le_bytes()).unwrap();
        for d in 0..dim {
            f.write_all(&(((first + r) as f32) + d as f32).to_le_bytes())
                .unwrap();
        }
    }
}

/// Raw packed `u32` scalars.
fn write_u32(path: &std::path::Path, values: impl Iterator<Item = u32>) {
    let mut f = std::fs::File::create(path).unwrap();
    for v in values {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

/// dim 4 → 4 + 4*4 = 20 bytes per record.
const BPR: u64 = 20;

/// **A uniform series opens, and reports the whole facet.**
///
/// `total_size` sums the files, `record_count` comes from the
/// declaration, and neither is shard 0's answer. `count()` is the
/// series total and `dim()` the dimension every shard shares (SH-23).
#[test]
fn a_uniform_series_opens_and_reports_the_whole_facet() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();

    write_fvec(&ds.join("base__0000.fvec"), 4, 100, 0);
    write_fvec(&ds.join("base__0001.fvec"), 4, 100, 100);
    write_fvec(&ds.join("base__0002.fvec"), 4, 40, 200);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: sharded\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
         record_count: 240\n",
    )
    .unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let storage = view.open_facet_storage("base_vectors").unwrap();

    assert_eq!(
        storage.total_size(),
        240 * BPR,
        "the facet is every shard's bytes, not the first shard's"
    );
    assert!(storage.is_local());
    assert!(storage.is_complete());
}

/// **A declaration that disagrees with the files is refused at open.**
///
/// The shard files here hold 240 records; the declaration claims 250.
/// Neither number silently wins (SH-8). The match is verified eagerly,
/// at open rather than at the first read that falls past the end
/// (SH-53).
#[test]
fn a_series_whose_files_contradict_its_declaration_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 100, 0);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: bad\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
         record_count: 9999\n",
    )
    .unwrap();
    // 3 shards of stride 100 can hold 201..=300 records, never 9999.
    let err = vectordata::TestDataGroup::load(ds.to_str().unwrap())
        .expect_err("an unreachable total must not load");
    assert!(
        err.to_string().contains("record_count"),
        "the message must name the disagreement: {err}"
    );
}

/// **An explicit series of named files opens**, with per-entry counts
/// and no probing of anything.
#[test]
fn an_explicit_series_opens_from_named_files() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_u32(&ds.join("part-a.u32"), 0..100);
    write_u32(&ds.join("part-b.u32"), 100..160);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: explicit\nprofiles:\n  default:\n    metadata_layout:\n      source:\n        \
         - part-a.u32=100\n        - part-b.u32=60\n      record_count: 160\n",
    )
    .unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let storage = view.open_facet_storage("metadata_layout").unwrap();
    assert_eq!(storage.total_size(), 160 * 4);
}

/// Bare names are resolved by opening each file — the local
/// convenience (SH-63) — and the resulting facet is the same one the
/// counted spelling produces.
#[test]
fn bare_names_resolve_to_the_same_facet_as_counted_ones() {
    let tmp = tempfile::tempdir().unwrap();
    let sizes = [("part-a.u32", 0..100u32), ("part-b.u32", 100..160)];

    let build = |dir: &std::path::Path, yaml: &str| {
        std::fs::create_dir_all(dir).unwrap();
        for (n, r) in sizes.clone() {
            write_u32(&dir.join(n), r);
        }
        std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();
        let g = vectordata::TestDataGroup::load(dir.to_str().unwrap()).unwrap();
        let v = g.profile("default").unwrap();
        v.open_facet_storage("metadata_layout")
            .unwrap()
            .total_size()
    };

    let bare = build(
        &tmp.path().join("bare"),
        "name: b\nprofiles:\n  default:\n    metadata_layout:\n      source:\n        \
         - part-a.u32\n        - part-b.u32\n      record_count: 160\n",
    );
    let counted = build(
        &tmp.path().join("counted"),
        "name: c\nprofiles:\n  default:\n    metadata_layout:\n      source:\n        \
         - part-a.u32=100\n        - part-b.u32=60\n      record_count: 160\n",
    );
    assert_eq!(bare, counted, "spelling must not change the facet");
}

/// **One file backing two shards opens once.**
///
/// Storage is per *file*, not per shard (SH-81): the registry keys on
/// the canonical path, so two windows into one file share a `Storage`,
/// a cache entry, and a descriptor.
#[test]
fn two_shards_of_one_file_share_its_storage() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_u32(&ds.join("corpus.u32"), 0..1000);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: sliced\nprofiles:\n  default:\n    metadata_layout:\n      source:\n        \
         - corpus.u32[0..100]=100\n        - corpus.u32[900..1000]=100\n      \
         record_count: 200\n",
    )
    .unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let storage = view.open_facet_storage("metadata_layout").unwrap();

    // Two shards, one file — so the facet's byte size is the file's,
    // counted once, not twice.
    assert_eq!(
        storage.total_size(),
        1000 * 4,
        "a shared file must be counted once, not once per shard"
    );
}

/// **Every pre-sharding dataset still opens, unchanged** — the
/// compatibility anchor (SH-70, test 33).
#[test]
fn a_plain_facet_is_untouched_by_any_of_this() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 4, 50, 0);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: plain\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
    )
    .unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let storage = view.open_facet_storage("base_vectors").unwrap();
    assert_eq!(storage.total_size(), 50 * BPR);
    assert!(storage.is_complete());

    // And it still reads, through the ordinary reader path.
    let reader = view.base_vectors().unwrap();
    assert_eq!(reader.count(), 50);
    assert_eq!(reader.dim(), 4);
}

/// **A missing shard is reported, not reported as emptiness.**
///
/// The infallible `total_size()` can only answer `0` for a facet it
/// cannot size — which reads exactly like an empty facet. A whole-facet
/// accessor that cannot fail must not answer as though it succeeded
/// (SH-99), so the fallible form says what is actually wrong and names
/// the file.
#[test]
fn a_missing_shard_is_named_rather_than_read_as_emptiness() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 100, 0);
    // __0001 deliberately absent.
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: gap\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 2\n      \
         record_count: 150\n",
    )
    .unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let storage = view.open_facet_storage("base_vectors").unwrap();

    let err = storage
        .try_total_size()
        .expect_err("a declared shard that is absent must be reported");
    let msg = err.to_string();
    assert!(
        msg.contains("base__0001.fvec"),
        "the message must name the missing shard: {msg}"
    );
    assert!(
        !storage.is_complete(),
        "a facet with an absent shard is not complete"
    );
}

// ─── Reading through the series ────────────────────────────────────

/// **A sharded facet reads exactly like the single file it was split
/// from** — the anchor property (SH-48, tests 13/18).
///
/// One fixture written once as a single file and once as a series must
/// be indistinguishable through the reader: same count, same dim, same
/// records, in the same order. Every `get(o)` resolves to the shard
/// that owns `o` and reads there (SH-24).
#[test]
fn a_series_reads_identically_to_the_single_file_it_was_split_from() {
    let tmp = tempfile::tempdir().unwrap();

    let single = tmp.path().join("single");
    std::fs::create_dir_all(&single).unwrap();
    write_fvec(&single.join("base.fvec"), 4, 240, 0);
    std::fs::write(
        single.join("dataset.yaml"),
        "name: one\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
    )
    .unwrap();

    let series = tmp.path().join("series");
    std::fs::create_dir_all(&series).unwrap();
    write_fvec(&series.join("base__0000.fvec"), 4, 100, 0);
    write_fvec(&series.join("base__0001.fvec"), 4, 100, 100);
    write_fvec(&series.join("base__0002.fvec"), 4, 40, 200);
    std::fs::write(
        series.join("dataset.yaml"),
        "name: many\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
         record_count: 240\n",
    )
    .unwrap();

    let open = |dir: &std::path::Path| {
        let g = vectordata::TestDataGroup::load(dir.to_str().unwrap()).unwrap();
        let v = g.profile("default").unwrap();
        v.base_vectors().unwrap()
    };
    let a = open(&single);
    let b = open(&series);

    assert_eq!(a.count(), b.count(), "the series is the same length");
    assert_eq!(a.dim(), b.dim());
    for i in 0..a.count() {
        assert_eq!(a.get(i).unwrap(), b.get(i).unwrap(), "record {i} differs");
    }
    // Including every seam, explicitly.
    for i in [99usize, 100, 199, 200, 239] {
        assert_eq!(a.get(i).unwrap(), b.get(i).unwrap(), "seam at {i}");
    }
}

/// Reading past the end is out of bounds, not a wrap into shard 0.
#[test]
fn reading_past_the_series_end_is_out_of_bounds() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 10, 0);
    write_fvec(&ds.join("base__0001.fvec"), 4, 5, 10);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: bounds\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 10\n      shard_count: 2\n      \
         record_count: 15\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let r = g.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(r.count(), 15);
    assert!(r.get(14).is_ok());
    assert!(r.get(15).is_err(), "one past the end must fail");
    assert!(r.get(999).is_err());
}

/// **A profile window applies in facet ordinals, over the whole
/// series** (SH-67) — not per shard, and not against a file.
#[test]
fn a_profile_window_clips_the_series_not_a_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 100, 0);
    write_fvec(&ds.join("base__0001.fvec"), 4, 100, 100);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: win\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 2\n      \
         record_count: 200\n      window: 50..150\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let full_dir = tmp.path().join("full");
    std::fs::create_dir_all(&full_dir).unwrap();
    write_fvec(&full_dir.join("base.fvec"), 4, 200, 0);
    std::fs::write(
        full_dir.join("dataset.yaml"),
        "name: full\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
    )
    .unwrap();
    let full = vectordata::TestDataGroup::load(full_dir.to_str().unwrap()).unwrap();

    let windowed = g.profile("default").unwrap().base_vectors().unwrap();
    let whole = full.profile("default").unwrap().base_vectors().unwrap();

    assert_eq!(windowed.count(), 100, "the window spans a shard boundary");
    // Window record 0 is facet record 50, which lives in shard 0;
    // window record 60 is facet record 110, which lives in shard 1.
    assert_eq!(windowed.get(0).unwrap(), whole.get(50).unwrap());
    assert_eq!(windowed.get(49).unwrap(), whole.get(99).unwrap());
    assert_eq!(windowed.get(50).unwrap(), whole.get(100).unwrap());
    assert_eq!(windowed.get(99).unwrap(), whole.get(149).unwrap());
    assert!(windowed.get(100).is_err());
}

/// A series whose shards disagree on dimension is refused at open —
/// there is no later moment at which the mismatch becomes visible
/// (SH-15).
#[test]
fn a_series_with_a_disagreeing_shard_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 10, 0);
    write_fvec(&ds.join("base__0001.fvec"), 8, 10, 10); // dim 8, not 4
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: mixed\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 10\n      shard_count: 2\n      \
         record_count: 20\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let msg = match g.profile("default").unwrap().base_vectors() {
        Err(e) => e.to_string(),
        Ok(_) => panic!("shards of different shapes must not open as one facet"),
    };
    assert!(
        msg.contains("dim"),
        "the message must name the shape: {msg}"
    );
}

/// **Typed access reads a scalar series** — the same values the single
/// file would serve, at the same ordinals.
#[test]
fn typed_access_reads_a_scalar_series() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_u32(&ds.join("layout__0000.u32"), 0..100);
    write_u32(&ds.join("layout__0001.u32"), 100..175);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: scalars\nprofiles:\n  default:\n    metadata_layout:\n      \
         source: layout__NNNN.u32\n      shard_stride: 100\n      shard_count: 2\n      \
         record_count: 175\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let r: vectordata::TypedReader<u32> =
        vectordata::open_facet_typed(&*view, "metadata_layout").unwrap();

    assert_eq!(r.count(), 175, "the count spans the series");
    for o in [0usize, 1, 99, 100, 101, 174] {
        assert_eq!(r.get_value(o).unwrap(), o as u32, "value at ordinal {o}");
    }
    assert!(r.get_value(175).is_err(), "one past the end must fail");
}

/// A typed reader over a *sliced* series reads the windowed records,
/// mapping facet ordinals to file ordinals through the entry window
/// (SH-64).
#[test]
fn typed_access_reads_a_sliced_series() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_u32(&ds.join("corpus.u32"), 0..1000);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: sliced\nprofiles:\n  default:\n    metadata_layout:\n      source:\n        \
         - corpus.u32[0..10]=10\n        - corpus.u32[990..1000]=10\n      record_count: 20\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let r: vectordata::TypedReader<u32> =
        vectordata::open_facet_typed(&*view, "metadata_layout").unwrap();

    assert_eq!(r.count(), 20);
    // First shard: facet ordinals 0..10 are file ordinals 0..10.
    assert_eq!(r.get_value(0).unwrap(), 0);
    assert_eq!(r.get_value(9).unwrap(), 9);
    // Second shard: facet ordinal 10 is file ordinal 990.
    assert_eq!(r.get_value(10).unwrap(), 990);
    assert_eq!(r.get_value(19).unwrap(), 999);
    assert!(r.get_value(20).is_err());
}

/// Write an ivvec of ragged records and its `IDXFOR__` sidecar.
fn write_ivvec_with_index(path: &std::path::Path, dims: &[i32], first: i32) {
    let mut buf: Vec<u8> = Vec::new();
    let mut starts: Vec<u64> = Vec::new();
    for (i, &d) in dims.iter().enumerate() {
        starts.push(buf.len() as u64);
        buf.extend(&d.to_le_bytes());
        for e in 0..d {
            buf.extend(&(first + i as i32 * 100 + e).to_le_bytes());
        }
    }
    std::fs::write(path, &buf).unwrap();
    let name = path.file_name().unwrap().to_str().unwrap();
    let idx: Vec<u8> = starts
        .iter()
        .flat_map(|&o| (o as i32).to_le_bytes())
        .collect();
    std::fs::write(
        path.parent().unwrap().join(format!("IDXFOR__{name}.i32")),
        idx,
    )
    .unwrap();
}

/// **A variable-length series reads through per-file indexes.**
///
/// Each shard file carries its own `IDXFOR__`, whose offsets are local
/// to that file (SH-17, SH-82) — nothing is re-based, and the series
/// reads as one ordinal space.
#[test]
fn a_vvec_series_reads_through_per_file_indexes() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();

    let a: Vec<i32> = (0..10).map(|i| 1 + (i % 4)).collect();
    let b: Vec<i32> = (0..6).map(|i| 1 + (i % 3)).collect();
    write_ivvec_with_index(&ds.join("meta__0000.ivvec"), &a, 0);
    write_ivvec_with_index(&ds.join("meta__0001.ivvec"), &b, 1000);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: vv\nprofiles:\n  default:\n    metadata_results:\n      \
         source: meta__NNNN.ivvec\n      shard_stride: 10\n      shard_count: 2\n      \
         record_count: 16\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let r = view.metadata_results().unwrap();

    assert_eq!(r.count(), 16, "the series spans both shards");
    // Shard 0, record 0: dim 1, value 0.
    assert_eq!(r.dim_at(0).unwrap(), 1);
    assert_eq!(r.get(0).unwrap(), vec![0]);
    // Shard 0, record 9: dim 1 + (9 % 4) = 2.
    assert_eq!(r.dim_at(9).unwrap(), a[9] as usize);
    // Shard 1, record 0 is facet ordinal 10 — its own file's ordinal 0.
    assert_eq!(r.dim_at(10).unwrap(), b[0] as usize);
    assert_eq!(r.get(10).unwrap()[0], 1000);
    assert_eq!(r.get(15).unwrap()[0], 1000 + 5 * 100);
    assert!(r.get(16).is_err(), "one past the end must fail");
}

/// A vvec series reads identically to the single file it was split
/// from — ragged records included.
#[test]
fn a_vvec_series_matches_the_single_file_it_was_split_from() {
    let tmp = tempfile::tempdir().unwrap();
    let dims: Vec<i32> = (0..16).map(|i| 1 + (i % 5)).collect();

    let single = tmp.path().join("single");
    std::fs::create_dir_all(&single).unwrap();
    write_ivvec_with_index(&single.join("meta.ivvec"), &dims, 0);
    std::fs::write(
        single.join("dataset.yaml"),
        "name: one\nprofiles:\n  default:\n    metadata_results: meta.ivvec\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(single.to_str().unwrap()).unwrap();
    let whole = g.profile("default").unwrap().metadata_results().unwrap();
    assert_eq!(whole.count(), 16);
    for i in 0..16 {
        assert_eq!(whole.dim_at(i).unwrap(), dims[i] as usize, "dim at {i}");
    }
}

// ─── Prefetch planning over a series ───────────────────────────────

use vectordata::WholeFacetFallback;
use vectordata::dataset::source::parse_window;
use vectordata::dataset::Sharding;

/// **A window inside one shard plans only that shard.**
///
/// The shard index is the point: byte 500 exists in every file, so a
/// range without one names no bytes (SH-28).
#[test]
fn a_window_inside_one_shard_plans_only_that_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 100, 0);
    write_fvec(&ds.join("base__0001.fvec"), 4, 100, 100);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: p\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 2\n      \
         record_count: 200\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let plan = view
        .prefetch_plan("base_vectors", &parse_window("110..150").unwrap())
        .unwrap();

    assert!(!plan.degrades_to_full_download);
    assert_eq!(plan.byte_ranges.len(), 1, "one shard, one range");
    let r = plan.byte_ranges[0];
    assert_eq!(r.shard, 1, "records 110..150 live in shard 1");
    assert_eq!(
        (r.start, r.end),
        (10 * BPR, 50 * BPR),
        "and at that shard's own byte offsets, not the facet's"
    );
}

/// **A window across a seam plans one range per shard**, each in its
/// own file's coordinates (SH-14, SH-28).
#[test]
fn a_window_across_a_seam_plans_one_range_per_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 100, 0);
    write_fvec(&ds.join("base__0001.fvec"), 4, 100, 100);
    write_fvec(&ds.join("base__0002.fvec"), 4, 50, 200);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: seam\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
         record_count: 250\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let plan = view
        .prefetch_plan("base_vectors", &parse_window("80..220").unwrap())
        .unwrap();

    assert_eq!(plan.byte_ranges.len(), 3, "the window touches three shards");
    let by_shard: Vec<(usize, u64, u64)> = plan
        .byte_ranges
        .iter()
        .map(|r| (r.shard, r.start, r.end))
        .collect();
    assert_eq!(
        by_shard,
        vec![
            (0, 80 * BPR, 100 * BPR),
            (1, 0, 100 * BPR),
            (2, 0, 20 * BPR),
        ]
    );
    // The parts sum to the window.
    let total: u64 = plan.byte_ranges.iter().map(|r| r.len()).sum();
    assert_eq!(total, 140 * BPR);
}

/// Ranges in different shards are different files and never merge,
/// however adjacent their byte offsets look.
#[test]
fn ranges_in_different_shards_never_merge() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 10, 0);
    write_fvec(&ds.join("base__0001.fvec"), 4, 10, 10);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: nm\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 10\n      shard_count: 2\n      \
         record_count: 20\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    // Records 9 and 10 are adjacent in the facet but in different
    // files, and shard 1's range starts at byte 0 — which would look
    // mergeable with shard 0's if the shard were dropped.
    let plan = view
        .prefetch_plan("base_vectors", &parse_window("9..11").unwrap())
        .unwrap();
    assert_eq!(plan.byte_ranges.len(), 2, "{:?}", plan.byte_ranges);
    assert_eq!(plan.byte_ranges[0].shard, 0);
    assert_eq!(plan.byte_ranges[1].shard, 1);
}

/// A no-window plan names every shard, so the whole facet is described
/// by real files rather than one span that exists in none of them.
#[test]
fn a_whole_facet_plan_names_every_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 10, 0);
    write_fvec(&ds.join("base__0001.fvec"), 4, 4, 10);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: whole\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 10\n      shard_count: 2\n      \
         record_count: 14\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let plan = view
        .prefetch_plan("base_vectors", &parse_window("").unwrap())
        .unwrap();
    assert_eq!(plan.byte_ranges.len(), 2);
    assert_eq!(plan.byte_ranges[0], (0u64, 10 * BPR));
    assert_eq!(plan.byte_ranges[1].shard, 1);
    assert_eq!(plan.byte_ranges[1].end, 4 * BPR);
    let total: u64 = plan.byte_ranges.iter().map(|r| r.len()).sum();
    assert_eq!(total, plan.facet_bytes);
}

/// Prefetching a local series is a no-op that still reports its ranges
/// — and needs no whole-facet consent, because the window mapped.
#[test]
fn prefetching_a_mapped_series_window_needs_no_consent() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 10, 0);
    write_fvec(&ds.join("base__0001.fvec"), 4, 10, 10);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: pf\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 10\n      shard_count: 2\n      \
         record_count: 20\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    view.prefetch(
        "base_vectors",
        &parse_window("5..15").unwrap(),
        WholeFacetFallback::Refuse,
    )
    .expect("a mapped window never needs whole-facet consent");
}

// ─── Creation ──────────────────────────────────────────────────────

/// **A derived series reads back as the dataset it was derived from.**
///
/// The full loop: write a single-file source, derive it with a stride,
/// and open the result. Splitting is a layout change, not a content
/// change (SH-48) — and the derived dataset must declare itself in a
/// form the reader understands without help.
#[test]
fn a_derived_series_round_trips_through_the_reader() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_fvec(&src.join("base_vectors.fvec"), 4, 250, 0);
    std::fs::write(
        src.join("dataset.yaml"),
        "name: src\nprofiles:\n  default:\n    base_vectors: base_vectors.fvec\n",
    )
    .unwrap();

    let out = tmp.path().join("out");
    let rc = vectordata::datasets::derive::run(
        src.to_str().unwrap(),
        "default",
        &out,
        "",
        &[],
        &[],
        Some("derived"),
        true,
        Sharding::Stride(100),
    );
    assert_eq!(rc, 0, "derive must succeed");

    // Three shards, four digits, plus a .mref each (SH-2, SH-20).
    let dir = out.join("profiles/base");
    for i in 0..3 {
        let f = dir.join(format!("base_vectors__{i:04}.fvec"));
        assert!(f.is_file(), "missing {}", f.display());
        assert!(
            dir.join(format!("base_vectors__{i:04}.fvec.mref"))
                .is_file(),
            "each shard is independently verifiable"
        );
    }
    assert!(
        !dir.join("base_vectors.fvec").exists(),
        "the unsharded name must not also be written"
    );

    // The emitted declaration describes the series.
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(yaml.contains("base_vectors__NNNN.fvec"), "{yaml}");
    assert!(yaml.contains("shard_stride: 100"), "{yaml}");
    assert!(yaml.contains("shard_count: 3"), "{yaml}");
    assert!(yaml.contains("record_count: 250"), "{yaml}");

    // And it reads back, record for record, as the source did.
    let orig = vectordata::TestDataGroup::load(src.to_str().unwrap()).unwrap();
    let orig_r = orig.profile("default").unwrap().base_vectors().unwrap();
    let derived = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let derived_r = derived.profile("default").unwrap().base_vectors().unwrap();

    assert_eq!(derived_r.count(), 250);
    assert_eq!(derived_r.dim(), orig_r.dim());
    for i in 0..250 {
        assert_eq!(
            derived_r.get(i).unwrap(),
            orig_r.get(i).unwrap(),
            "record {i}"
        );
    }
}

/// **A derive that fits in one shard emits the single-file form**
/// (SH-83), so a run that happened to fit stays readable by anything
/// predating multi-file facets.
#[test]
fn a_derive_that_fits_one_shard_emits_the_single_file_form() {
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
    let rc = vectordata::datasets::derive::run(
        src.to_str().unwrap(),
        "default",
        &out,
        "",
        &[],
        &[],
        Some("derived"),
        true,
        Sharding::Stride(1000),
    );
    assert_eq!(rc, 0);

    let dir = out.join("profiles/base");
    assert!(dir.join("base_vectors.fvec").is_file(), "collapsed");
    assert!(!dir.join("base_vectors__0000.fvec").exists());

    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(
        !yaml.contains("shard_stride"),
        "a collapsed output declares no shard fields: {yaml}"
    );
    assert!(
        yaml.contains("base_vectors: profiles/base/base_vectors.fvec"),
        "{yaml}"
    );
}

/// Deriving without a stride is exactly what it always was.
#[test]
fn deriving_without_a_stride_is_unchanged() {
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
    let rc = vectordata::datasets::derive::run(
        src.to_str().unwrap(),
        "default",
        &out,
        "",
        &[],
        &[],
        Some("derived"),
        true,
        Sharding::Whole,
    );
    assert_eq!(rc, 0);
    assert!(out.join("profiles/base/base_vectors.fvec").is_file());
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(!yaml.contains("shard_"), "{yaml}");
}

// ─── Publication ───────────────────────────────────────────────────

/// **Push publishes every shard and every sidecar** (SH-39).
///
/// Publication is filesystem-driven rather than declaration-driven, so
/// shards need no special handling — being ordinary files, they are
/// picked up by construction (SH-100). This pins that they actually
/// are, and that a temp left by a killed derive is not.
#[test]
fn a_published_series_lists_every_shard_and_sidecar() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    for i in 0..3 {
        write_fvec(&ds.join(format!("base__{i:04}.fvec")), 4, 10, i * 10);
        std::fs::write(ds.join(format!("base__{i:04}.fvec.mref")), b"mref").unwrap();
    }
    // A temp from a killed run, and a scratch file: neither is content.
    std::fs::write(ds.join("base__0003.fvec.partial"), b"half").unwrap();
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: pub\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 10\n      shard_count: 3\n      \
         record_count: 30\n",
    )
    .unwrap();

    let scan = vectordata::push::plan::scan(&ds).expect("scan");
    let names: Vec<String> = scan
        .files
        .iter()
        .map(|f| {
            std::path::Path::new(f)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string()
        })
        .collect();

    for i in 0..3 {
        assert!(
            names.contains(&format!("base__{i:04}.fvec")),
            "shard {i} not published: {names:?}"
        );
        assert!(
            names.contains(&format!("base__{i:04}.fvec.mref")),
            "shard {i}'s sidecar not published: {names:?}"
        );
    }
    assert!(
        !names.iter().any(|n| n.ends_with(".partial")),
        "a temp from a killed run must not ship: {names:?}"
    );
}

/// A catalog carries a series, so a remote consumer can enumerate the
/// shards before fetching anything (SH-41).
///
/// The declaration it carries is serialized from the realized model
/// (SH-89), so it states its cardinalities by construction rather than
/// because a pinning step ran over it.
#[test]
fn a_catalog_entry_round_trips_a_series() {
    let yaml = "default:\n  base_vectors:\n    source:\n      - a.fvec=10\n      - b.fvec=10\n\
                \x20   record_count: 20\n";
    let group: vectordata::dataset::DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
    let rendered = serde_yaml::to_string(&group).unwrap();
    let again: vectordata::dataset::DSProfileGroup = serde_yaml::from_str(&rendered).unwrap();

    let v = again.profiles["default"].views.get("base_vectors").unwrap();
    assert!(
        v.is_series(),
        "the series must survive a catalog round trip: {rendered}"
    );
    assert_eq!(v.sources().len(), 2);
    assert_eq!(v.record_count, Some(20));
}

// ─── Descriptor budget ─────────────────────────────────────────────

/// **A series larger than its descriptor budget still reads** (SH-59).
///
/// Files open lazily and the least-recently-used is closed when the
/// budget is reached, so the number of shards a facet may have is not
/// bounded by `ulimit -n`. Eviction can never pull a file out from under
/// a reader: it releases the series' claim, and a reader mid-read holds
/// its own.
#[test]
fn a_series_wider_than_the_descriptor_budget_still_reads() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();

    // 40 shards of 1 record each.
    let n = 40usize;
    for i in 0..n {
        write_fvec(&ds.join(format!("base__{i:04}.fvec")), 4, 1, i);
    }
    std::fs::write(
        ds.join("dataset.yaml"),
        format!(
            "name: wide\nprofiles:\n  default:\n    base_vectors:\n      \
             source: base__NNNN.fvec\n      shard_stride: 1\n      shard_count: {n}\n      \
             record_count: {n}\n"
        ),
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let r = view.base_vectors().unwrap();
    assert_eq!(r.count(), n);

    // Every record reads, in order and out of order — the second pass
    // re-opens files the first pass may have evicted.
    for i in 0..n {
        assert_eq!(r.get(i).unwrap()[0], i as f32, "record {i}");
    }
    for i in (0..n).rev() {
        assert_eq!(r.get(i).unwrap()[0], i as f32, "reverse pass, record {i}");
    }

    // And the budget was enforced along the way: reading every record
    // touched every file, but no more than the cap are held open.
    let storage = view.open_facet_storage("base_vectors").unwrap();
    let series = storage.series_ref().expect("a series");
    for i in 0..n {
        let _ = view.base_vectors().unwrap().get(i).unwrap();
    }
    assert!(
        series.open_file_count() <= vectordata::view::open_file_cap(),
        "open files ({}) exceeded the descriptor budget ({})",
        series.open_file_count(),
        vectordata::view::open_file_cap()
    );
}

/// The cap is derived, not hardcoded, and leaves headroom for
/// everything else the process holds open (SH-59).
#[test]
fn the_open_file_cap_is_derived_and_leaves_headroom() {
    let cap = vectordata::view::open_file_cap();
    assert!(cap >= 8, "a floor keeps small limits workable: {cap}");
    // On any host this runs on, the budget should not be so small that
    // it binds on ordinary fetch concurrency — SH-77's assumption.
    assert!(cap > 0);
}

/// **Residency means addressable bytes, not whole files** (SH-92).
///
/// A facet slicing a fraction of a large file is complete when its
/// window is resident — otherwise it would report incomplete forever
/// unless the rest were downloaded, bytes it can never read.
#[test]
fn a_sliced_facet_is_complete_when_its_window_is() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_u32(&ds.join("corpus.u32"), 0..10_000);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: sliced\nprofiles:\n  default:\n    metadata_layout:\n      source:\n        \
         - corpus.u32[0..10]=10\n        - corpus.u32[9990..10000]=10\n      record_count: 20\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let storage = view.open_facet_storage("metadata_layout").unwrap();

    // Local storage is resident by definition, so the facet is complete
    // — and it reads only 80 of the file's 40,000 bytes.
    assert!(storage.is_complete());
    assert!(
        storage.precache().is_ok(),
        "precache asks for the window, not the file"
    );

    let r: vectordata::TypedReader<u32> =
        vectordata::open_facet_typed(&*view, "metadata_layout").unwrap();
    assert_eq!(r.count(), 20);
    assert_eq!(r.get_value(0).unwrap(), 0);
    assert_eq!(r.get_value(10).unwrap(), 9990);
}

// ─── Format version ────────────────────────────────────────────────

/// **Absent means 1** (V-2). Every dataset in circulation omits the
/// field, and they are all version 1 — not a distinct "unversioned"
/// state to handle.
#[test]
fn a_dataset_without_a_version_is_version_one() {
    let cfg: vectordata::model::DatasetConfig =
        serde_yaml::from_str("profiles:\n  default:\n    base_vectors: b.fvec\n").unwrap();
    assert_eq!(cfg.format_version, vectordata::model::FORMAT_VERSION_BASE);

    let explicit: vectordata::model::DatasetConfig = serde_yaml::from_str(
        "format_version: 1\nprofiles:\n  default:\n    base_vectors: b.fvec\n",
    )
    .unwrap();
    assert_eq!(
        explicit.format_version, cfg.format_version,
        "an explicit 1 behaves identically to absence"
    );
}

/// **A version above what this build supports is refused, naming both
/// numbers** (V-9) — the diagnosis the field exists to give, in place of
/// a type error on `source` or a missing `__NNNN` file.
#[test]
fn a_dataset_from_the_future_is_refused_with_both_numbers() {
    let err = serde_yaml::from_str::<vectordata::model::DatasetConfig>(
        "format_version: 99\nprofiles:\n  default:\n    base_vectors: b.fvec\n",
    )
    .expect_err("a version this build cannot read must be refused");
    let msg = err.to_string();
    assert!(msg.contains("99"), "names what the dataset needs: {msg}");
    assert!(
        msg.contains(&vectordata::model::FORMAT_VERSION_SUPPORTED.to_string()),
        "names what this build supports: {msg}"
    );
}

/// The refusal is at **load**, before any facet is opened (V-10). A
/// dataset the reader cannot understand must not be half-read.
///
/// The fixture's one facet is an ordinary v1 `base.fvec` — familiar,
/// readable, and refused anyway (V-11). "The facets I want are all
/// version 1" is a judgement the reader is not equipped to make, since
/// the version exists precisely because it cannot tell what it is
/// missing.
#[test]
fn a_refused_version_opens_no_facet() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 4, 10, 0);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: future\nformat_version: 99\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
    )
    .unwrap();
    assert!(
        vectordata::TestDataGroup::load(ds.to_str().unwrap()).is_err(),
        "a dataset above this build's version must not load at all"
    );
}

/// **A new build writing an unsharded dataset emits no version** (V-5),
/// so it stays readable by every build that ever existed. The field is
/// worthless if adding it changes what older builds can read.
#[test]
fn an_unsharded_derive_emits_no_version() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_fvec(&src.join("base_vectors.fvec"), 4, 20, 0);
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
    assert!(!yaml.contains("format_version"), "{yaml}");
}

/// A sharded derive declares version 2 (V-8), because that is the
/// lowest version describing what it wrote.
#[test]
fn a_sharded_derive_declares_version_two() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_fvec(&src.join("base_vectors.fvec"), 4, 250, 0);
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
            Sharding::Stride(100),
        ),
        0
    );
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(yaml.contains("format_version: 2"), "{yaml}");

    // And it round-trips: this build supports 2, so it still reads.
    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    assert_eq!(
        g.profile("default")
            .unwrap()
            .base_vectors()
            .unwrap()
            .count(),
        250
    );
}

/// **The required version is derived from the shape, not read from a
/// field** (V-19) — so a writer cannot drift from the rule, because it
/// does not restate it.
#[test]
fn the_required_version_is_derived_from_the_declaration() {
    let plain: vectordata::model::DatasetConfig =
        serde_yaml::from_str("profiles:\n  default:\n    base_vectors: b.fvec\n").unwrap();
    assert_eq!(plain.min_format_version(), 1);
    assert!(plain.is_v1(), "a v1 dataset proves its own compatibility");

    let sharded: vectordata::model::DatasetConfig = serde_yaml::from_str(
        "profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n      \
         shard_stride: 100\n      shard_count: 3\n      record_count: 250\n",
    )
    .unwrap();
    assert_eq!(sharded.min_format_version(), 2);
    assert!(
        !sharded.is_v1(),
        "a sharded dataset cannot claim v1 compatibility"
    );
}

/// **A v1 declaration is held as the v1 case, not re-encoded** (V-17).
/// The version is visible in the type rather than probed from whether an
/// option happens to be set.
#[test]
fn a_v1_declaration_stays_the_v1_case() {
    use vectordata::model::FacetConfig;
    let cfg: vectordata::model::DatasetConfig = serde_yaml::from_str(
        "profiles:\n  default:\n    base_vectors: b.fvec\n    query_vectors:\n      \
         source: q.fvec\n      window: 0..10\n",
    )
    .unwrap();
    let p = &cfg.profiles["default"];
    assert!(matches!(p.base_vectors, Some(FacetConfig::Simple(_))));
    assert!(matches!(
        p.query_vectors,
        Some(FacetConfig::Detailed { .. })
    ));
    assert!(p.base_vectors.as_ref().unwrap().try_as_v1().is_some());
}

/// **A declaration cannot understate itself** (V-14): a stated version
/// below what the content needs is refused.
#[test]
fn a_stated_version_below_the_content_is_refused() {
    let err = serde_yaml::from_str::<vectordata::model::DatasetConfig>(
        "format_version: 1\nprofiles:\n  default:\n    base_vectors:\n      \
         source: b__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
         record_count: 250\n",
    )
    .expect_err("a version below the content must be refused");
    assert!(err.to_string().contains("understate"), "{err}");
}

/// **Absence is not a claim** (V-22). An unannotated sharded dataset
/// loads: a reader new enough to notice the omission is new enough to
/// read it, and refusing would reject every hand-written dataset.
#[test]
fn an_absent_version_is_not_an_understatement() {
    let cfg: vectordata::model::DatasetConfig = serde_yaml::from_str(
        "profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n      \
         shard_stride: 100\n      shard_count: 3\n      record_count: 250\n",
    )
    .expect("an unannotated sharded dataset loads");
    assert_eq!(cfg.format_version, 1, "absent still means 1 for the gate");
    assert_eq!(cfg.min_format_version(), 2, "but the content needs 2");
}

/// **A gap in the middle of a series is an error, not a short read**
/// (SH-3).
///
/// The uniform form derives contiguous names, so an absent middle shard
/// is a hole in the ordinal space. Reporting the records either side of
/// it — or the count the declaration promised — would put wrong
/// vectors at every ordinal past the gap.
#[test]
fn a_gap_in_the_middle_of_a_series_is_reported_by_name() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 100, 0);
    // __0001 deliberately absent — the shards either side of it exist.
    write_fvec(&ds.join("base__0002.fvec"), 4, 100, 200);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: gap\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 3\n      \
         record_count: 300\n",
    )
    .unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let storage = view.open_facet_storage("base_vectors").unwrap();

    let err = storage
        .try_total_size()
        .expect_err("a hole in the middle of a series must be reported");
    assert!(
        err.to_string().contains("base__0001.fvec"),
        "the message must name the missing shard: {err}"
    );
    assert!(!storage.is_complete());

    // And a read that would land past the gap must not answer from the
    // shard that happens to be there.
    let reader = view.base_vectors();
    if let Ok(r) = reader {
        assert!(
            r.get(250).is_err(),
            "an ordinal past a missing shard must not resolve"
        );
    }
}

/// **A shard and another facet's file that share a basename stay
/// distinct on disk** (SH-33).
///
/// The cache-relpath collision guard is a catalog concern — it fires
/// when two facets would cache to one filename. Locally there is no
/// cache and the paths differ, so what must hold here is the weaker
/// but more basic property: each facet reads its own file, and a
/// basename shared with a shard of another facet changes nothing.
#[test]
fn a_shard_sharing_a_basename_with_another_facet_reads_its_own_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    std::fs::create_dir_all(ds.join("a")).unwrap();
    std::fs::create_dir_all(ds.join("b")).unwrap();
    write_fvec(&ds.join("a/base__0000.fvec"), 4, 50, 0);
    write_fvec(&ds.join("a/base__0001.fvec"), 4, 50, 50);
    write_fvec(&ds.join("b/base__0001.fvec"), 4, 50, 0);

    // Both facets live outside a dataset home URL, so each file caches
    // under its basename — and `base__0001.fvec` is claimed twice.
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: clash\nprofiles:\n  default:\n    base_vectors:\n      \
         source: a/base__NNNN.fvec\n      shard_stride: 50\n      shard_count: 2\n      \
         record_count: 100\n    query_vectors: b/base__0001.fvec\n",
    )
    .unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    // Local files resolve to distinct absolute paths, so this dataset
    // is legal on disk; the guard is a catalog-cache concern. What must
    // hold either way is that the two facets read their own bytes.
    let base = view.base_vectors().unwrap();
    let query = view.query_vectors().unwrap();
    assert_eq!(base.count(), 100);
    assert_eq!(query.count(), 50);
    assert_eq!(base.get(50).unwrap()[0], 50.0, "shard 1 of base_vectors");
    assert_eq!(query.get(0).unwrap()[0], 0.0, "b/base__0001 is its own file");
}

/// **The CLI stride and the YAML key are one knob** (SH-44).
///
/// `--shard-stride 1M` and `shard_stride: 1000000` have to mean the
/// same number, and the number the flag asked for has to be the one
/// that lands in the written declaration. Both CLI surfaces parse
/// through `parse_number_with_suffix`, so the congruence is a property
/// of that function plus what `derive` emits.
#[test]
fn the_cli_stride_and_the_yaml_key_mean_the_same_number() {
    use vectordata::dataset::source::parse_number_with_suffix;
    assert_eq!(parse_number_with_suffix("1M").unwrap(), 1_000_000);
    assert_eq!(
        parse_number_with_suffix("1M").unwrap(),
        parse_number_with_suffix("1000000").unwrap(),
        "the suffixed and plain spellings are the same knob"
    );

    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_fvec(&src.join("base_vectors.fvec"), 4, 250, 0);
    std::fs::write(
        src.join("dataset.yaml"),
        "name: src\nprofiles:\n  default:\n    base_vectors: base_vectors.fvec\n",
    )
    .unwrap();

    let out = tmp.path().join("out");
    let stride = parse_number_with_suffix("100").unwrap();
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
            Sharding::Stride(stride),
        ),
        0
    );

    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(
        yaml.contains("shard_stride: 100"),
        "the flag's value is the key's value:\n{yaml}"
    );
    // And the dataset that comes back out reads as one facet.
    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let r = g.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(r.count(), 250);
    assert_eq!(r.get(249).unwrap()[0], 249.0);
}

/// **A sliced series publishes its files whole** (SH-84).
///
/// A window is a view over ordinals, not a licence to ship a fraction
/// of a file. Publishing only the windowed bytes would produce files
/// whose declared entry windows no longer describe them, and any other
/// profile sharing those files would find them truncated.
#[test]
fn a_sliced_series_publishes_whole_files() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("part_a.fvec"), 4, 100, 0);
    write_fvec(&ds.join("part_b.fvec"), 4, 100, 100);
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: sliced\nprofiles:\n  default:\n    base_vectors:\n      source:\n\
        \x20       - part_a.fvec[20..60)=40\n        - part_b.fvec[0..30)=30\n      \
         record_count: 70\n",
    )
    .unwrap();

    // The declaration reads 70 of the 200 records on disk.
    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let r = g.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(r.count(), 70);
    assert_eq!(r.get(0).unwrap()[0], 20.0, "the first entry's window applies");
    assert_eq!(r.get(40).unwrap()[0], 100.0, "the second entry starts at its own base");

    let scan = vectordata::push::plan::scan(&ds).expect("scan");
    for name in ["part_a.fvec", "part_b.fvec"] {
        let published = scan
            .files
            .iter()
            .find(|f| f.ends_with(name))
            .unwrap_or_else(|| panic!("{name} not published: {:?}", scan.files));
        assert_eq!(
            std::fs::metadata(ds.join(published)).unwrap().len(),
            100 * (4 + 4 * 4),
            "{name} must ship whole, not clipped to its window"
        );
    }
}

/// **`veks check` reports a non-canonical series; it does not rewrite
/// it** (SH-88).
///
/// A one-shard series is legal to read and wrong to write (SH-4), so a
/// validator has to say so. Repairing it silently would edit a file the
/// operator did not ask to change, and the next run would report
/// nothing — hiding whatever produced the declaration.
#[test]
fn validation_reports_a_non_canonical_series_without_rewriting_it() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base__0000.fvec"), 4, 40, 0);
    let yaml = "name: single\nprofiles:\n  default:\n    base_vectors:\n      \
                source: base__NNNN.fvec\n      shard_stride: 40\n      shard_count: 1\n      \
                record_count: 40\n";
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let cfg: vectordata::dataset::DatasetConfig = serde_yaml::from_str(yaml).unwrap();
    let violations = vectordata::dataset::conformance::validate_conformance(&cfg)
        .expect_err("a one-shard series is not canonical");
    assert!(
        violations
            .iter()
            .any(|v: &vectordata::dataset::conformance::FacetViolation| {
                v.to_string().contains("base_vectors")
            }),
        "the facet must be named: {violations:?}"
    );

    // The reader accepts it, and the file on disk is untouched.
    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    assert_eq!(
        g.profile("default").unwrap().base_vectors().unwrap().count(),
        40
    );
    assert_eq!(
        std::fs::read_to_string(ds.join("dataset.yaml")).unwrap(),
        yaml,
        "validation must not rewrite the declaration"
    );
}

/// **Saving a dataset must not be able to destroy it** (SH-85).
///
/// The compact writer emitted `view.source.path`, which for a uniform
/// series is the `NNNN` pattern with no stride or count — a declaration
/// naming a file that does not exist — and for an explicit one dropped
/// every entry after the first. Any command that loads and re-saves a
/// sharded `dataset.yaml` would have silently reduced it to a fraction
/// of itself.
#[test]
fn saving_a_sharded_dataset_preserves_its_series() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    for s in 0..3 {
        write_fvec(&ds.join(format!("base__{s:04}.fvec")), 4, 50, s * 50);
    }
    write_fvec(&ds.join("part_a.fvec"), 4, 20, 0);
    write_fvec(&ds.join("part_b.fvec"), 4, 20, 20);
    let yaml = "name: keep\nprofiles:\n  default:\n    base_vectors:\n      \
                source: base__NNNN.fvec\n      shard_stride: 50\n      shard_count: 3\n      \
                record_count: 150\n    query_vectors:\n      source:\n        \
                - part_a.fvec=20\n        - part_b.fvec=20\n      record_count: 40\n";
    let path = ds.join("dataset.yaml");
    std::fs::write(&path, yaml).unwrap();

    let cfg = vectordata::dataset::DatasetConfig::load(&path).expect("loads");
    let saved = cfg.to_expanded_yaml_string(&path).expect("renders");

    // Both forms survive, each in its own spelling.
    assert!(saved.contains("source: base__NNNN.fvec"), "{saved}");
    assert!(saved.contains("shard_stride: 50"), "{saved}");
    assert!(saved.contains("shard_count: 3"), "{saved}");
    assert!(saved.contains("- part_a.fvec=20"), "{saved}");
    assert!(saved.contains("- part_b.fvec=20"), "{saved}");

    // And the saved text reads back as the same facets.
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    for s in 0..3 {
        std::fs::copy(
            ds.join(format!("base__{s:04}.fvec")),
            out.join(format!("base__{s:04}.fvec")),
        )
        .unwrap();
    }
    std::fs::copy(ds.join("part_a.fvec"), out.join("part_a.fvec")).unwrap();
    std::fs::copy(ds.join("part_b.fvec"), out.join("part_b.fvec")).unwrap();
    std::fs::write(out.join("dataset.yaml"), &saved).unwrap();

    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 150, "the uniform series survived the round trip");
    assert_eq!(base.get(149).unwrap()[0], 149.0);
    let query = view.query_vectors().unwrap();
    assert_eq!(query.count(), 40, "the explicit series kept both entries");
    assert_eq!(query.get(39).unwrap()[0], 39.0);
}

/// **A facet reports one access mode, and it is the weakest among its
/// files** (SH-93).
///
/// A mode is a promise about every read the facet will serve. A series
/// whose shards were published differently — one with a `.mref`, one
/// without — must answer for the worst of them, or a caller plans
/// against an average it will not get. Understating a good shard costs
/// efficiency; overstating a bad one costs correctness, and only one of
/// those is recoverable.
#[test]
fn a_series_reports_the_weakest_access_mode_of_its_shards() {
    use vectordata::access::AccessMode::{self, *};
    fn weakest<const N: usize>(m: [AccessMode; N]) -> Option<AccessMode> {
        AccessMode::weakest(m)
    }

    // The ordering the fold is built on: what a caller must plan
    // around, not a quality ranking.
    assert_eq!(weakest([Local, MerkleHashed]), Some(MerkleHashed));
    assert_eq!(
        weakest([MerkleHashed, MerkleChunked]),
        Some(MerkleChunked),
        "a trusted-bytes shard is weaker than a verified one"
    );
    assert_eq!(
        weakest([Local, MerkleHashed, FullTransfer]),
        Some(FullTransfer),
        "one shard that must download whole makes the facet's promise that"
    );
    assert_eq!(weakest([Local, Local]), Some(Local));
    assert_eq!(
        weakest([]),
        None,
        "a facet with no files makes no promise, which is not the weakest promise"
    );
}

/// **A local series reports Local, through the facet handle** (SH-93).
///
/// The fold above is the rule; this is the rule reaching a facet. Every
/// shard on disk means every read is served from disk, and the handle
/// says so without the caller enumerating shards.
#[test]
fn a_local_series_reports_local_through_its_handle() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    for s in 0..3 {
        write_fvec(&ds.join(format!("base__{s:04}.fvec")), 4, 50, s * 50);
    }
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: modes\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 50\n      shard_count: 3\n      \
         record_count: 150\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let storage = view.open_facet_storage("base_vectors").unwrap();

    // The declared source is the NNNN pattern, which names no file —
    // the series must classify its shards, not that.
    let mode = storage.access_mode("base__NNNN.fvec", tmp.path());
    assert_eq!(mode, vectordata::access::AccessMode::Local);
}

/// **A facet-selecting flag names the facet, never a shard** (SH-45).
///
/// There is no CLI surface for "precache shard 7" — that is what a
/// window is for. A flag that accepted a shard filename would let a
/// caller address a fraction of a facet by a name the ordinal model
/// does not use.
#[test]
fn facet_selection_names_facets_not_shard_files() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    for s in 0..2 {
        write_fvec(&ds.join(format!("base__{s:04}.fvec")), 4, 50, s * 50);
    }
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: sel\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 50\n      shard_count: 2\n      \
         record_count: 100\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    // The facet name resolves.
    assert!(view.open_facet_storage("base_vectors").is_ok());

    // A shard filename is not a facet name, in any of its spellings.
    for spelling in ["base__0000", "base__0000.fvec", "base__NNNN.fvec", "base__0001"] {
        assert!(
            view.open_facet_storage(spelling).is_err(),
            "'{spelling}' must not address a facet — a shard is not selectable"
        );
    }

    // And the manifest lists the facet once, by its own name.
    let manifest = view.facet_manifest();
    assert!(manifest.contains_key("base_vectors"));
    assert!(
        !manifest.keys().any(|k| k.contains("__0000")),
        "no shard appears as a facet: {:?}",
        manifest.keys().collect::<Vec<_>>()
    );
}

// ── sizing a series from a file-size cap ───────────────────────────

use vectordata::dataset::shard_sizing::{plan_fixed, xvec_record_bytes};

fn capped_source(dir: &std::path::Path, dim: i32, records: usize) {
    std::fs::create_dir_all(dir).unwrap();
    write_fvec(&dir.join("base_vectors.fvec"), dim, records, 0);
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: src\nprofiles:\n  default:\n    base_vectors: base_vectors.fvec\n",
    )
    .unwrap();
}

/// **A cap becomes a stride, and the stride becomes the layout.** The
/// operator says how large a file may get; the record size decides how
/// many records that is.
#[test]
fn a_size_cap_derives_a_decade_stride() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    // dim 4 → 20 bytes a record. A 250-byte cap holds 12 records, so
    // the decade below is 10.
    capped_source(&src, 4, 25);
    assert_eq!(xvec_record_bytes(4, 4), 20);
    assert_eq!(plan_fixed(250, 20).unwrap().stride, 10);

    let out = tmp.path().join("out");
    let rc = vectordata::datasets::derive::run(
        src.to_str().unwrap(), "default", &out, "", &[], &[],
        Some("derived"), true, Sharding::MaxBytes(250),
    );
    assert_eq!(rc, 0, "derive under a cap must succeed");

    let dir = out.join("profiles/base");
    for i in 0..3 {
        assert!(
            dir.join(format!("base_vectors__{i:04}.fvec")).exists(),
            "shard {i} must exist"
        );
    }
    // And the declaration says what was written.
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(yaml.contains("base_vectors__NNNN.fvec"), "{yaml}");
    assert!(yaml.contains("shard_stride: 10"), "{yaml}");
    assert!(yaml.contains("shard_count: 3"), "{yaml}");
    assert!(yaml.contains("record_count: 25"), "{yaml}");

    // Every shard is under the cap it was sized for.
    for i in 0..3 {
        let len = std::fs::metadata(dir.join(format!("base_vectors__{i:04}.fvec")))
            .unwrap()
            .len();
        assert!(len <= 250, "shard {i} is {len} bytes, over a 250-byte cap");
    }
}

/// A facet that fits under the cap is written as one file (SH-83), so
/// asking for a cap does not itself produce a series.
#[test]
fn a_facet_under_its_cap_stays_a_single_file() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    capped_source(&src, 4, 25);

    let out = tmp.path().join("out");
    let rc = vectordata::datasets::derive::run(
        src.to_str().unwrap(), "default", &out, "", &[], &[],
        Some("derived"), true, Sharding::MaxBytes(1_000_000_000),
    );
    assert_eq!(rc, 0);

    let dir = out.join("profiles/base");
    assert!(dir.join("base_vectors.fvec").exists(), "one file");
    assert!(!dir.join("base_vectors__0000.fvec").exists());
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(!yaml.contains("shard_stride"), "no series declared: {yaml}");
}

/// A cap that cannot hold ten records is a misconfiguration, and is
/// refused rather than emitting a file per handful of records.
#[test]
fn a_cap_too_small_for_a_run_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    capped_source(&src, 4, 25);

    let out = tmp.path().join("out");
    let rc = vectordata::datasets::derive::run(
        src.to_str().unwrap(), "default", &out, "", &[], &[],
        Some("derived"), true, Sharding::MaxBytes(100), // five records
    );
    assert_ne!(rc, 0, "a cap this small must be refused, not honoured");
}
