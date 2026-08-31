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

// ── derive ─────────────────────────────────────────────────────────

/// **Deriving from an explicit series copies the whole facet** (SH-38).
///
/// This is the case that used to fail silently: `part_a.fvec` is a real
/// file of 100 records, so a plan built from the view's first source
/// would copy half the base and write a `dataset.yaml` claiming to be
/// complete. The derived dataset must hold all 200.
#[test]
fn deriving_from_an_explicit_series_copies_every_record() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    explicit_series(&src);
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
            None,
        ),
        0,
        "deriving from a series must succeed"
    );

    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let r = g.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(r.count(), 200, "every record of every shard");
    for i in 0..200usize {
        assert_eq!(r.get(i).unwrap()[0], i as f32, "derived record {i}");
    }
}

/// The uniform form derives the same way — the two spellings differ
/// only in how the file list is derived, and the copy must not be able
/// to tell them apart.
#[test]
fn deriving_from_a_uniform_series_copies_every_record() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    uniform_series(&src);
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
            None,
        ),
        0
    );

    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let r = g.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(r.count(), 200);
    assert_eq!(r.get(0).unwrap()[0], 0.0);
    assert_eq!(r.get(99).unwrap()[0], 99.0);
    assert_eq!(r.get(100).unwrap()[0], 100.0, "across the source seam");
    assert_eq!(r.get(199).unwrap()[0], 199.0);
}

/// **Re-striding is a copy** (SH-38).
///
/// The source's shard boundaries and the output's have nothing to do
/// with each other. A 2×100 source written at a stride of 60 becomes
/// four shards — 60, 60, 60, 20 — and reads back as the same 200
/// records in the same order.
#[test]
fn a_series_re_strides_to_a_different_shard_layout() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    explicit_series(&src);
    let out = tmp.path().join("out");

    assert_eq!(
        vectordata::datasets::derive::run(
            src.to_str().unwrap(),
            "default",
            &out,
            "",
            &[],
            &[],
            Some("restrided"),
            true,
            Some(60),
        ),
        0
    );

    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(yaml.contains("shard_stride: 60"), "{yaml}");
    assert!(yaml.contains("shard_count: 4"), "{yaml}");
    assert!(yaml.contains("record_count: 200"), "{yaml}");

    // Four files, and the last is the short one.
    for s in 0..4 {
        assert!(
            out.join(format!("profiles/base/base_vectors__{s:04}.fvec")).exists(),
            "shard {s} missing"
        );
    }
    let last = std::fs::metadata(out.join("profiles/base/base_vectors__0003.fvec"))
        .unwrap()
        .len();
    assert_eq!(last, 20 * (4 + 4 * 4), "the last shard holds the remainder");

    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let r = g.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(r.count(), 200);
    for i in 0..200usize {
        assert_eq!(r.get(i).unwrap()[0], i as f32, "re-strided record {i}");
    }
}

/// A window over a series derives the window, in the series' ordinal
/// space — not a window into shard 0.
#[test]
fn deriving_a_windowed_series_slices_the_series() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_fvec(&src.join("part_a.fvec"), 4, 100, 0);
    write_fvec(&src.join("part_b.fvec"), 4, 100, 100);
    std::fs::write(
        src.join("dataset.yaml"),
        "name: series\nprofiles:\n  default:\n    base_vectors:\n      source:\n\
        \x20       - part_a.fvec=100\n        - part_b.fvec=100\n      record_count: 200\n      \
         window: 80..130\n",
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
            Some("windowed"),
            true,
            None,
        ),
        0
    );

    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let r = g.profile("default").unwrap().base_vectors().unwrap();
    assert_eq!(r.count(), 50, "the window is in the series' ordinals");
    assert_eq!(r.get(0).unwrap()[0], 80.0);
    assert_eq!(r.get(19).unwrap()[0], 99.0, "last record of the first shard");
    assert_eq!(r.get(20).unwrap()[0], 100.0, "first of the second");
    assert_eq!(r.get(49).unwrap()[0], 129.0);
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

/// **The refusal names a path that works** (SH-38).
///
/// A command whose kernel reads one mmapped file cannot take a series,
/// but "cannot" is only acceptable when there is something the operator
/// can do instead. Deriving with no stride writes the series back as a
/// single file, so the message points there — and the round trip is
/// asserted here, because a suggestion that does not work is worse than
/// no suggestion.
#[test]
fn deriving_a_series_without_a_stride_yields_the_single_file_a_kernel_needs() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    explicit_series(&src);
    let out = tmp.path().join("out");

    assert_eq!(
        vectordata::datasets::derive::run(
            src.to_str().unwrap(),
            "default",
            &out,
            "",
            &[],
            &[],
            Some("flat"),
            true,
            None,
        ),
        0
    );

    // Exactly one file, no shard pattern anywhere in the declaration.
    let flat = out.join("profiles/base/base_vectors.fvec");
    assert!(flat.is_file(), "the derived facet must be one file");
    assert_eq!(
        std::fs::metadata(&flat).unwrap().len(),
        200 * (4 + 4 * 4),
        "holding every record of the series"
    );
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    for key in ["NNNN", "shard_stride", "shard_count"] {
        assert!(!yaml.contains(key), "`{key}` leaked into a flat output:\n{yaml}");
    }

    // And it opens as the single-file facet a path-based kernel wants.
    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    assert!(
        view.facet_source("base_vectors").is_some(),
        "a flat facet resolves to one path"
    );
    let r = view.base_vectors().unwrap();
    assert_eq!(r.count(), 200);
    assert_eq!(r.get(199).unwrap()[0], 199.0);
}

// ── variable-length and container formats ──────────────────────────

fn write_ivvec(path: &std::path::Path, first: usize, count: usize) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for i in 0..count {
        let global = first + i;
        let dim = (global % 5) + 1;
        f.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            f.write_all(&((global * 100 + d) as i32).to_le_bytes()).unwrap();
        }
    }
    f.flush().unwrap();
}

/// **A vvec series derives as one facet** (SH-38).
///
/// A vvec is a self-describing stream — each record is its own
/// dimension followed by that many values, and the offset index is a
/// sidecar rather than part of the file — so its shards concatenate and
/// the copy reads them as one stream, exactly as a fixed-stride facet
/// is read.
#[test]
fn a_vvec_series_derives_into_one_facet() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_ivvec(&src.join("meta_a.ivvec"), 0, 30);
    write_ivvec(&src.join("meta_b.ivvec"), 30, 30);
    std::fs::write(
        src.join("dataset.yaml"),
        "name: vv\nprofiles:\n  default:\n    metadata_results:\n      source:\n\
        \x20       - meta_a.ivvec=30\n        - meta_b.ivvec=30\n      record_count: 60\n",
    )
    .unwrap();

    let out = tmp.path().join("out");
    assert_eq!(
        vectordata::datasets::derive::run(
            src.to_str().unwrap(), "default", &out, "", &[], &[],
            Some("vv-derived"), true, None,
        ),
        0,
        "a vvec series must derive"
    );

    // The derived file is the concatenation, byte for byte.
    let derived = out.join("profiles/base/metadata_results.ivvec");
    assert!(derived.is_file(), "expected {}", derived.display());
    let mut expected = std::fs::read(src.join("meta_a.ivvec")).unwrap();
    expected.extend(std::fs::read(src.join("meta_b.ivvec")).unwrap());
    assert_eq!(std::fs::read(&derived).unwrap(), expected);

    // And it reads back as 60 records with the right shapes.
    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let r = view.metadata_results().unwrap();
    assert_eq!(r.count(), 60, "every record of both shards");
    // Record shapes and values survive the concatenation.
    for o in [0usize, 29, 30, 59] {
        assert_eq!(r.dim_at(o).unwrap(), (o % 5) + 1, "dim at {o}");
        assert_eq!(r.get(o).unwrap()[0], (o * 100) as i32, "value at {o}");
    }
}

/// A window over a vvec series is a window in the **series'** ordinal
/// space, resolved by walking the concatenated records rather than any
/// one shard's sidecar.
#[test]
fn a_windowed_vvec_series_slices_the_series() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_ivvec(&src.join("meta_a.ivvec"), 0, 30);
    write_ivvec(&src.join("meta_b.ivvec"), 30, 30);
    std::fs::write(
        src.join("dataset.yaml"),
        "name: vv\nprofiles:\n  default:\n    metadata_results:\n      source:\n\
        \x20       - meta_a.ivvec=30\n        - meta_b.ivvec=30\n      record_count: 60\n      \
         window: 20..45\n",
    )
    .unwrap();

    let out = tmp.path().join("out");
    assert_eq!(
        vectordata::datasets::derive::run(
            src.to_str().unwrap(), "default", &out, "", &[], &[],
            Some("vv-window"), true, None,
        ),
        0
    );

    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let r = view.metadata_results().unwrap();
    assert_eq!(r.count(), 25, "a 25-record window across the seam");
    // The window starts at series ordinal 20 and crosses into the
    // second shard at 30.
    assert_eq!(r.get(0).unwrap()[0], 2000);
    assert_eq!(r.get(10).unwrap()[0], 3000, "first record of the second shard");
    assert_eq!(r.get(24).unwrap()[0], 4400);
}

/// **A sliced vvec shard is resolved, not refused** (SH-50, SH-67).
///
/// An entry window is in the file's own ordinals, and a vvec is a
/// self-describing stream, so walking it says where those records
/// start. An earlier cut refused this with "no fixed record size to
/// resolve that against" — a claim the same file disproves, since the
/// windowed single-file copy performs exactly that walk.
#[test]
fn a_sliced_vvec_shard_contributes_only_its_own_records() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    // 40 records each; the series takes the middle 10 of the first and
    // the first 5 of the second.
    write_ivvec(&src.join("part_a.ivvec"), 0, 40);
    write_ivvec(&src.join("part_b.ivvec"), 100, 40);
    std::fs::write(
        src.join("dataset.yaml"),
        "name: sliced-vv\nprofiles:\n  default:\n    metadata_results:\n      source:\n\
        \x20       - part_a.ivvec[10..20)=10\n        - part_b.ivvec[0..5)=5\n      \
         record_count: 15\n",
    )
    .unwrap();

    let out = tmp.path().join("out");
    assert_eq!(
        vectordata::datasets::derive::run(
            src.to_str().unwrap(), "default", &out, "", &[], &[],
            Some("sliced"), true, None,
        ),
        0,
        "a sliced vvec series must derive"
    );

    let g = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let r = view.metadata_results().unwrap();
    assert_eq!(r.count(), 15, "only the windowed records");
    // The first shard contributed its records 10..20, whose ids are
    // 10..20; the second contributed its 0..5, whose ids are 100..105.
    assert_eq!(r.get(0).unwrap()[0], 1000);
    assert_eq!(r.get(9).unwrap()[0], 1900);
    assert_eq!(r.get(10).unwrap()[0], 10000, "first record of the second shard");
    assert_eq!(r.get(14).unwrap()[0], 10400);
}

