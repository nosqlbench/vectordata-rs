// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `TestDataViewExt::verify`: a profile's facets belong together
//! (`docs/design/srd-loose-files-overlays-integrity.md` §5.1, V1–V7).

use std::io::Write;
use std::path::Path;

use vectordata::verify::{Check, NoProgress, VerifyOptions};
use vectordata::{Error, TestDataGroup, TestDataViewExt};

fn rows<T: Copy>(path: &Path, dim: i32, rows: &[Vec<T>], bytes: fn(T) -> Vec<u8>) {
    let mut f = std::fs::File::create(path).unwrap();
    for r in rows {
        let d = if dim < 0 { r.len() as i32 } else { dim };
        f.write_all(&d.to_le_bytes()).unwrap();
        for &v in r {
            f.write_all(&bytes(v)).unwrap();
        }
    }
}
fn f32s(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn i32s(v: i32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn u64s(v: u64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// A dataset of a 10-vector dim-4 base and 3 queries, with
/// `extra` lines under its default profile.
fn dataset(dir: &Path, extra: &str) -> TestDataGroup {
    let base: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 4]).collect();
    rows(&dir.join("base.fvecs"), 4, &base, f32s);
    rows(&dir.join("query.fvecs"), 4, &base[..3], f32s);
    std::fs::write(
        dir.join("dataset.yaml"),
        format!("name: v\nprofiles:\n  default:\n    base_vectors: base.fvecs\n    query_vectors: query.fvecs\n{extra}"),
    )
    .unwrap();
    TestDataGroup::load(dir.to_str().unwrap()).unwrap()
}

fn failures(group: &TestDataGroup) -> Vec<(Check, String)> {
    let view = group.profile("default").unwrap();
    match view.verify(&mut NoProgress) {
        Ok(_) => Vec::new(),
        Err(Error::Verification { failures }) => failures.into_iter().map(|f| (f.check, f.to_string())).collect(),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn a_consistent_profile_passes_every_check() {
    let dir = tempfile::tempdir().unwrap();
    rows(&dir.path().join("gt.ivecs"), 2, &[vec![0, 1], vec![1, 2], vec![2, 9]], i32s);
    rows(&dir.path().join("d.fvecs"), 2, &[vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]], f32s);
    let group = dataset(dir.path(), "    neighbor_indices: gt.ivecs\n    neighbor_distances: d.fvecs\n    base_count: 10\n");
    let report = group.profile("default").unwrap().verify(&mut NoProgress).unwrap();
    for check in [Check::V1Opens, Check::V2QueryDimension, Check::V3TruthRows, Check::V4TruthWidth, Check::V5TruthIndices, Check::V6Distances] {
        assert!(report.checked.contains(&check), "{check:?} ran: {:?}", report.checked);
    }
}

/// **V5: an index past the base is reported at its row and column** —
/// the one check that shows the truth was not computed against this
/// base.
#[test]
fn an_index_past_the_base_is_reported_at_its_row_and_column() {
    let dir = tempfile::tempdir().unwrap();
    rows(&dir.path().join("gt.ivecs"), 2, &[vec![0, 1], vec![1, 10], vec![2, 3]], i32s);
    let group = dataset(dir.path(), "    neighbor_indices: gt.ivecs\n");
    let found = failures(&group);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].0, Check::V5TruthIndices);
    assert!(
        found[0].1.contains("row 1 column 1 holds 10, which names no base vector of 10 base vectors; only a filtered truth may hold the -1 sentinel"),
        "{found:?}"
    );
}

/// **V5 in a filtered truth: `-1` ends a short row**, and is refused in
/// the plain one.
#[test]
fn the_minus_one_sentinel_is_a_filtered_truths_only() {
    let dir = tempfile::tempdir().unwrap();
    rows(&dir.path().join("gt.ivecs"), 2, &[vec![0, -1], vec![1, 2], vec![2, 3]], i32s);
    let filtered = dataset(dir.path(), "    prefiltered_neighbor_indices: gt.ivecs\n");
    assert!(failures(&filtered).is_empty());
    let plain = dataset(dir.path(), "    neighbor_indices: gt.ivecs\n");
    assert!(failures(&plain)[0].1.contains("holds -1"), "{:?}", failures(&plain));
}

/// **V3 and V6: rows that do not match**, equal and not merely enough.
#[test]
fn truth_rows_must_equal_the_queries_and_distances_the_truth() {
    let dir = tempfile::tempdir().unwrap();
    rows(&dir.path().join("gt.ivecs"), 2, &[vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]], i32s);
    rows(&dir.path().join("d.fvecs"), 2, &[vec![0.0, 1.0], vec![0.0, 1.0]], f32s);
    let group = dataset(dir.path(), "    neighbor_indices: gt.ivecs\n    neighbor_distances: d.fvecs\n");
    let found = failures(&group);
    assert!(found.iter().any(|(c, m)| *c == Check::V3TruthRows && m.contains("holds 4 rows for 3 queries")), "{found:?}");
    assert!(found.iter().any(|(c, m)| *c == Check::V6Distances && m.contains("holds 2 rows for the 4 rows of neighbor_indices")), "{found:?}");

    rows(&dir.path().join("gt.ivecs"), 2, &[vec![0, 1], vec![1, 2], vec![2, 3]], i32s);
    rows(&dir.path().join("d.fvecs"), 3, &[vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]], f32s);
    let found = failures(&dataset(dir.path(), "    neighbor_indices: gt.ivecs\n    neighbor_distances: d.fvecs\n"));
    assert!(found.iter().any(|(c, m)| *c == Check::V6Distances && m.contains("3 distances per row for the 2 neighbors")), "{found:?}");
}

/// **V2: queries of another dimension** are not of this base.
#[test]
fn queries_of_another_dimension_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    let group = dataset(dir.path(), "");
    rows(&dir.path().join("query.fvecs"), 3, &[vec![0.0; 3]], f32s);
    let found = failures(&group);
    assert!(found.iter().any(|(c, m)| *c == Check::V2QueryDimension && m.contains("has dimension 3; the base has 4")), "{found:?}");
}

/// **V7: a declared base count that is not the base's.**
#[test]
fn a_declared_base_count_must_be_the_bases() {
    let dir = tempfile::tempdir().unwrap();
    let found = failures(&dataset(dir.path(), "    base_count: 12\n"));
    assert!(found.iter().any(|(c, m)| *c == Check::V7DeclaredCount && m.contains("holds 10 vectors; base_count declares 12")), "{found:?}");
}

/// **V1: a facet that does not open**, and a u64 index that does not
/// fit i64, each named.
#[test]
fn a_facet_that_does_not_open_or_read_is_named() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("gt.ivecs"), [1u8, 0, 0, 0, 7]).unwrap();
    let found = failures(&dataset(dir.path(), "    neighbor_indices: gt.ivecs\n"));
    assert!(found.iter().any(|(c, m)| *c == Check::V1Opens && m.contains("neighbor_indices") && m.contains("does not open")), "{found:?}");

    rows(&dir.path().join("gt.u64vecs"), 1, &[vec![0u64], vec![u64::MAX], vec![1]], u64s);
    let found = failures(&dataset(dir.path(), "    neighbor_indices: gt.u64vecs\n"));
    assert!(found.iter().any(|(c, m)| *c == Check::V5TruthIndices && m.contains("does not fit in i64")), "{found:?}");
}

/// **A dataset still being built skips facets not yet written**, when
/// asked; by default an absent facet fails V1.
#[test]
fn an_absent_facet_is_skipped_only_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    let group = dataset(dir.path(), "    neighbor_indices: not_yet.ivecs\n");
    assert!(failures(&group).iter().any(|(c, _)| *c == Check::V1Opens));
    let view = group.profile("default").unwrap();
    let report = view.verify_report(&VerifyOptions { skip_absent: true, ..Default::default() }, &mut NoProgress);
    assert!(report.passed(), "{:?}", report.failures);
}

/// **The index scan reports its progress**, from 0 to every row.
#[test]
fn the_index_scan_reports_progress() {
    let dir = tempfile::tempdir().unwrap();
    rows(&dir.path().join("gt.ivecs"), 2, &[vec![0, 1], vec![1, 2], vec![2, 3]], i32s);
    let group = dataset(dir.path(), "    neighbor_indices: gt.ivecs\n");
    let mut seen = Vec::new();
    let mut sink = |label: &str, done: u64, total: u64| seen.push(format!("{label}: {done}/{total}"));
    group.profile("default").unwrap().verify(&mut sink).unwrap();
    assert_eq!(seen.first().map(String::as_str), Some("scanning neighbor_indices: 0/3"), "{seen:?}");
    assert_eq!(seen.last().map(String::as_str), Some("scanning neighbor_indices: 3/3"), "{seen:?}");
}

/// **A selection verifies every profile it selected**, each failure
/// named by its profile.
#[test]
fn a_selection_verifies_each_selected_profile() {
    let dir = tempfile::tempdir().unwrap();
    rows(&dir.path().join("gt.ivecs"), 1, &[vec![0], vec![1], vec![99]], i32s);
    dataset(dir.path(), "    neighbor_indices: gt.ivecs\n");
    let catalog = vectordata::catalog::Catalog::of(&vectordata::catalog::CatalogSources::new());
    let selection = catalog.open_spec(&format!("{}:default", dir.path().display())).unwrap();
    match selection.verify(&mut NoProgress) {
        Err(Error::Verification { failures }) => assert_eq!(failures[0].facet, "default:neighbor_indices"),
        other => panic!("{other:?}"),
    }
}
