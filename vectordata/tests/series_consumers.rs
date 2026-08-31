// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Consumers that read one file, meeting a facet that is several.
//!
//! A series has no single path. Every consumer written before sharding
//! assumed one, and the interesting question is not whether they work —
//! most cannot — but whether they **say so**. The explicit series form
//! names a real file first, so a consumer that reaches for the first
//! source gets a readable file and a plausible answer over a fraction
//! of the facet, with no error anywhere (SH-74, SH-79).
//!
//! These pin the two acceptable outcomes: read the whole series, or
//! refuse by name. Nothing in between.
//!
//! See `docs/design/srd-multifile-facet-shards.md`.

use std::io::Write as _;

fn write_fvec(path: &std::path::Path, dim: i32, records: usize, first: usize) {
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

fn write_u32(path: &std::path::Path, values: impl Iterator<Item = u32>) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for v in values {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

/// A dataset whose `base_vectors` is an explicit two-file series, with
/// both files present and readable.
fn explicit_series(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    write_fvec(&dir.join("part_a.fvec"), 4, 100, 0);
    write_fvec(&dir.join("part_b.fvec"), 4, 100, 100);
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: series\nprofiles:\n  default:\n    base_vectors:\n      source:\n\
        \x20       - part_a.fvec=100\n        - part_b.fvec=100\n      record_count: 200\n",
    )
    .unwrap();
}

/// A dataset whose `base_vectors` is a uniform series.
fn uniform_series(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    for s in 0..2 {
        write_fvec(&dir.join(format!("base__{s:04}.fvec")), 4, 100, s * 100);
    }
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: series\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 2\n      \
         record_count: 200\n",
    )
    .unwrap();
}

// ── the typed reader ───────────────────────────────────────────────

/// **Both entry points to a typed reader answer the same** (SH-79).
///
/// `open_facet_typed` exists as a free function over the trait and as
/// a method on the concrete view. The free one grew series support;
/// the method resolved a single source and errored. Two spellings of
/// one operation must not disagree about whether a dataset is
/// readable.
#[test]
fn the_typed_reader_reads_a_series_from_either_entry_point() {
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

    let by_fn: vectordata::TypedReader<u32> =
        vectordata::open_facet_typed(&*view, "metadata_layout")
            .expect("the free function reads a series");
    assert_eq!(by_fn.count(), 175);

    let concrete = g.generic_view("default").unwrap();
    let by_method: vectordata::TypedReader<u32> = concrete
        .open_facet_typed("metadata_layout")
        .expect("the concrete view reads a series too");
    assert_eq!(by_method.count(), by_fn.count());
    for o in [0usize, 99, 100, 174] {
        assert_eq!(
            by_method.get_value(o).unwrap(),
            by_fn.get_value(o).unwrap(),
            "ordinal {o}"
        );
    }
}

// ── whole-facet reads across the seam ──────────────────────────────

/// **A series read end to end equals the file it was split from.**
///
/// The access pattern a neighbour computation uses — every base vector
/// in order, then arbitrary re-reads — against both spellings of the
/// same 200 vectors. This is the numerical statement that the ordinal
/// algebra, the per-file offsets, and the seam all agree.
#[test]
fn a_full_sequential_and_random_pass_matches_the_unsplit_file() {
    let tmp = tempfile::tempdir().unwrap();

    let whole = tmp.path().join("whole");
    std::fs::create_dir_all(&whole).unwrap();
    write_fvec(&whole.join("base.fvec"), 4, 200, 0);
    std::fs::write(
        whole.join("dataset.yaml"),
        "name: whole\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
    )
    .unwrap();

    let split = tmp.path().join("split");
    explicit_series(&split);

    let gw = vectordata::TestDataGroup::load(whole.to_str().unwrap()).unwrap();
    let gs = vectordata::TestDataGroup::load(split.to_str().unwrap()).unwrap();
    let rw = gw.profile("default").unwrap().base_vectors().unwrap();
    let rs = gs.profile("default").unwrap().base_vectors().unwrap();

    assert_eq!(rw.count(), rs.count());
    assert_eq!(rw.dim(), rs.dim());

    for i in 0..rw.count() {
        assert_eq!(rw.get(i).unwrap(), rs.get(i).unwrap(), "record {i}");
    }
    // Out of order, crossing the seam repeatedly.
    for i in [199usize, 0, 100, 99, 150, 1, 198, 101] {
        assert_eq!(rw.get(i).unwrap(), rs.get(i).unwrap(), "random read {i}");
    }
}

/// The same equality under a profile window that straddles the seam,
/// which is where a per-shard clip can silently drop or duplicate the
/// records either side of a file boundary.
#[test]
fn a_window_across_the_seam_matches_the_unsplit_file() {
    let tmp = tempfile::tempdir().unwrap();

    let whole = tmp.path().join("whole");
    std::fs::create_dir_all(&whole).unwrap();
    write_fvec(&whole.join("base.fvec"), 4, 200, 0);
    std::fs::write(
        whole.join("dataset.yaml"),
        "name: whole\nprofiles:\n  default:\n    base_vectors: base.fvec[80..130)\n",
    )
    .unwrap();

    let split = tmp.path().join("split");
    std::fs::create_dir_all(&split).unwrap();
    write_fvec(&split.join("part_a.fvec"), 4, 100, 0);
    write_fvec(&split.join("part_b.fvec"), 4, 100, 100);
    std::fs::write(
        split.join("dataset.yaml"),
        "name: series\nprofiles:\n  default:\n    base_vectors:\n      source:\n\
        \x20       - part_a.fvec=100\n        - part_b.fvec=100\n      record_count: 200\n      \
         window: 80..130\n",
    )
    .unwrap();

    let gw = vectordata::TestDataGroup::load(whole.to_str().unwrap()).unwrap();
    let gs = vectordata::TestDataGroup::load(split.to_str().unwrap()).unwrap();
    let rw = gw.profile("default").unwrap().base_vectors().unwrap();
    let rs = gs.profile("default").unwrap().base_vectors().unwrap();

    assert_eq!(rw.count(), 50);
    assert_eq!(rs.count(), 50, "the window clips the series, not a shard");
    for i in 0..50 {
        assert_eq!(rw.get(i).unwrap(), rs.get(i).unwrap(), "windowed record {i}");
    }
    assert_eq!(rs.get(0).unwrap()[0], 80.0);
    assert_eq!(rs.get(49).unwrap()[0], 129.0);
}

