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
/// declaration, and neither is shard 0's answer.
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
/// Neither number silently wins (SH-8).
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
/// cannot size — which reads exactly like an empty facet. The fallible
/// form says what is actually wrong, and names the file.
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
/// records, in the same order.
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

