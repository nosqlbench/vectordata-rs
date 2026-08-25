// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Prefetching a caller-named ordinal window.
//!
//! A profile's `window:` names a range someone wants repeatedly. It is a
//! convenience, not a fence: a caller that knows it is about to read
//! records 5M..6M should be able to say so whether or not a profile was
//! defined ahead of time that says the same thing. These tests pin that
//! the ad-hoc path exists, resolves to the same bytes the profile path
//! would, and reports its cost before spending it.

use std::io::Write;
use vectordata::dataset::source::parse_window;

fn write_fvec(path: &std::path::Path, dim: i32, records: usize) {
    let mut f = std::fs::File::create(path).unwrap();
    for r in 0..records {
        f.write_all(&dim.to_le_bytes()).unwrap();
        for d in 0..dim {
            f.write_all(&((r as f32) + d as f32).to_le_bytes()).unwrap();
        }
    }
}

/// A 100-record, dim-4 fvec: bytes-per-record = 4 + 4*4 = 20.
const BPR: u64 = 20;

fn dataset(dir: &std::path::Path) -> vectordata::TestDataGroup {
    let ds = dir.join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 100);
    let yaml = r#"
name: prefetch-test
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap()
}

/// **The escape hatch.** No profile declares a window here, and a
/// caller can still name one and have it resolve to the right bytes.
#[test]
fn an_arbitrary_window_resolves_without_a_profile_declaring_it() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("base_vectors", &parse_window("10..20").unwrap())
        .unwrap();

    assert!(
        !plan.degrades_to_full_download,
        "a uniform-stride facet must be windowable on demand"
    );
    assert_eq!(
        plan.byte_ranges,
        vec![(10 * BPR, 20 * BPR)],
        "records map to bytes at 4 + dim*elem_size"
    );
}

/// Several intervals resolve to several ranges. Nothing in the fetch
/// path needs one contiguous window, and a caller reading two clumps
/// should not have to make two calls to say so.
#[test]
fn a_multi_interval_window_resolves_every_interval() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("base_vectors", &parse_window("[0..10, 50..60]").unwrap())
        .unwrap();
    assert_eq!(
        plan.byte_ranges,
        vec![(0, 10 * BPR), (50 * BPR, 60 * BPR)],
        "both intervals must survive; the reader's single-window limit \
         is a reader limit, not a fetch limit"
    );
}

/// An empty window means the whole facet — the same thing precache has
/// always meant, reachable through the same call.
#[test]
fn an_empty_window_covers_the_whole_facet() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("base_vectors", &parse_window("").unwrap())
        .unwrap();
    assert_eq!(plan.byte_ranges, vec![(0, 100 * BPR)]);
}

/// Local storage has no chunks, so there is nothing to fetch and
/// nothing to plan — and a caller must read that as free rather than as
/// unknown.
#[test]
fn a_local_facet_costs_nothing_to_prefetch() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("base_vectors", &parse_window("10..20").unwrap())
        .unwrap();
    assert!(plan.fills.is_empty(), "local storage reports no chunk fill");
    assert_eq!(plan.bytes_to_fetch(), 0);
    assert_eq!(plan.chunks_to_fetch(), 0);

    // And actually running it is a no-op that succeeds.
    let report = view
        .prefetch("base_vectors", &parse_window("10..20").unwrap())
        .unwrap();
    assert_eq!(report.ranges_fetched, 1);
}

/// A window past the end of the facet clamps rather than failing — the
/// caller asked for "up to here", and here is where the data stops.
#[test]
fn a_window_past_the_end_clamps_to_the_facet() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("base_vectors", &parse_window("90..500").unwrap())
        .unwrap();
    assert_eq!(plan.byte_ranges, vec![(90 * BPR, 100 * BPR)]);
}

/// **The degrade case has to be visible.** A format whose record→byte
/// mapping this layer cannot compute does not silently prefetch a
/// wrong range or quietly do nothing — it reports that honouring the
/// request means fetching the facet whole.
#[test]
fn an_unmappable_format_reports_that_it_degrades() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 10);

    // A vvec facet: variable-length, so the stride is not computable
    // without the sibling offset index.
    let vv = ds.join("profiles/default/metadata_content.vvec");
    std::fs::write(&vv, [0u8; 64]).unwrap();

    let yaml = r#"
name: prefetch-degrade
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/metadata_content.vvec
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("2..4").unwrap())
        .unwrap();
    assert!(
        plan.degrades_to_full_download,
        "vvec has no computable stride at this layer and must say so"
    );
    assert!(
        plan.byte_ranges.is_empty(),
        "a partial plan beside the degrade flag would understate the cost"
    );
    assert_eq!(
        plan.bytes_to_fetch(),
        plan.facet_bytes,
        "the honest cost of the degrade is the whole facet"
    );
}

/// The ad-hoc path and the profile path must agree. A profile window is
/// a name for a range, so naming it or passing it should resolve
/// identically — otherwise the convenience is a second semantics.
#[test]
fn an_adhoc_window_matches_what_the_profile_form_resolves_to() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 100);

    // Two profiles over the same file: one declaring a window, one not.
    let yaml = r#"
name: prefetch-parity
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
  windowed:
    base_vectors: profiles/default/base_vectors.fvec[10..20)
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();

    let adhoc = group
        .profile("default")
        .unwrap()
        .prefetch_plan("base_vectors", &parse_window("10..20").unwrap())
        .unwrap();
    let declared = group
        .profile("windowed")
        .unwrap()
        .prefetch_plan("base_vectors", &parse_window("10..20").unwrap())
        .unwrap();

    assert_eq!(
        adhoc.byte_ranges, declared.byte_ranges,
        "a window someone typed and a window someone declared are the same window"
    );
}
