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

/// **Coalescing shows up as fewer requests, not just tidier ranges.**
///
/// Two intervals a few records apart resolve to two byte ranges, and on
/// a local facet — where there are no chunks — they stay two. The plan
/// reports both what was asked for and what will be issued, so the
/// difference is visible rather than inferred.
#[test]
fn the_plan_separates_what_was_asked_for_from_what_is_issued() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    // Adjacent intervals: 0..10 ends exactly where 10..20 begins, so
    // they are one range under any granularity.
    let touching = view
        .prefetch_plan("base_vectors", &parse_window("[0..10, 10..20]").unwrap())
        .unwrap();
    assert_eq!(
        touching.requested_ranges.len(),
        2,
        "two intervals were asked for"
    );
    assert_eq!(
        touching.byte_ranges,
        vec![(0, 20 * BPR)],
        "and they merge into one request"
    );
    assert_eq!(touching.requests(), 1);

    // Far apart, on local storage with no chunk granularity to bridge
    // them: two intervals, two requests.
    let apart = view
        .prefetch_plan("base_vectors", &parse_window("[0..10, 80..90]").unwrap())
        .unwrap();
    assert_eq!(apart.requested_ranges.len(), 2);
    assert_eq!(
        apart.byte_ranges,
        vec![(0, 10 * BPR), (80 * BPR, 90 * BPR)],
        "nothing bridges a gap this size"
    );
    assert_eq!(apart.requests(), 2);
}

/// Overlapping intervals are one fetch, and the plan says so — asking
/// for the same bytes twice is the thing coalescing exists to stop.
#[test]
fn overlapping_intervals_become_one_request() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("base_vectors", &parse_window("[0..30, 20..40]").unwrap())
        .unwrap();
    assert_eq!(plan.requests(), 1);
    assert_eq!(plan.byte_ranges, vec![(0, 40 * BPR)]);
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

/// Write a variable-length file: each record is a 4-byte dim followed
/// by `dim` i32 elements, so record sizes differ and no stride exists.
fn write_ivvec(path: &std::path::Path, dims: &[i32]) -> Vec<u64> {
    let mut offsets = Vec::new();
    let mut f = std::fs::File::create(path).unwrap();
    let mut at = 0u64;
    for &d in dims {
        offsets.push(at);
        f.write_all(&d.to_le_bytes()).unwrap();
        for e in 0..d {
            f.write_all(&e.to_le_bytes()).unwrap();
        }
        at += 4 + (d as u64) * 4;
    }
    offsets
}

/// **A vvec window resolves through its offset index.**
///
/// Variable-length records have no stride, so the only way an ordinal
/// becomes a byte offset is the index. Once it is loaded the mapping is
/// exact — `offsets[start]` to `offsets[end]`, not an estimate that a
/// reader then has to correct.
#[test]
fn a_vvec_window_resolves_through_its_offset_index() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 10);

    // Deliberately ragged: every record a different length.
    let dims = [1, 7, 3, 9, 2, 5, 4, 8];
    let offsets = write_ivvec(&ds.join("profiles/default/meta.ivvec"), &dims);

    let yaml = r#"
name: prefetch-vvec
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/meta.ivvec
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("2..5").unwrap())
        .unwrap();
    assert!(
        !plan.degrades_to_full_download,
        "an indexed vvec facet is windowable"
    );
    assert_eq!(
        plan.byte_ranges,
        vec![(offsets[2], offsets[5])],
        "records 2..5 are exactly offsets[2]..offsets[5]"
    );
}

/// **The index cost is reported, not hidden.** A vvec window cannot be
/// resolved without reading the whole offset index, and that read is
/// real work a caller should be able to see before it decides to
/// prefetch a hundred small windows one at a time.
#[test]
fn a_vvec_plan_reports_the_index_it_had_to_read() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 10);
    let dims = [1, 7, 3, 9, 2, 5, 4, 8];
    write_ivvec(&ds.join("profiles/default/meta.ivvec"), &dims);

    let yaml = r#"
name: prefetch-vvec-cost
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/meta.ivvec
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let vvec_plan = view
        .prefetch_plan("metadata_content", &parse_window("2..5").unwrap())
        .unwrap();
    assert_eq!(
        vvec_plan.prerequisite_bytes,
        (dims.len() * 8) as u64,
        "one u64 offset per record had to be read to resolve the window"
    );

    // A uniform-stride facet pays nothing: its stride comes from a
    // header read every reader does on first access anyway.
    let xvec_plan = view
        .prefetch_plan("base_vectors", &parse_window("2..5").unwrap())
        .unwrap();
    assert_eq!(xvec_plan.prerequisite_bytes, 0);
}

/// A vvec window running past the last record ends at the file, not at
/// a record that does not exist.
#[test]
fn a_vvec_window_past_the_end_ends_at_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 10);
    let dims = [1, 7, 3, 9];
    let offsets = write_ivvec(&ds.join("profiles/default/meta.ivvec"), &dims);
    let file_len = std::fs::metadata(ds.join("profiles/default/meta.ivvec"))
        .unwrap()
        .len();

    let yaml = r#"
name: prefetch-vvec-tail
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/meta.ivvec
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("2..99").unwrap())
        .unwrap();
    assert_eq!(plan.byte_ranges, vec![(offsets[2], file_len)]);
}

/// The window a prefetch resolves and the bytes a reader actually
/// touches have to be the same bytes — otherwise a prefetch warms one
/// range and the reader faults in another, which looks like the
/// prefetch silently did nothing.
#[test]
fn a_vvec_prefetch_covers_every_byte_the_reader_reads() {
    use vectordata::IndexedVvecReader;

    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 10);
    let dims = [3, 1, 8, 2, 6, 4];
    let offsets = write_ivvec(&ds.join("profiles/default/meta.ivvec"), &dims);
    let vv = ds.join("profiles/default/meta.ivvec");

    let yaml = r#"
name: prefetch-vvec-parity
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/meta.ivvec
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("1..4").unwrap())
        .unwrap();
    let (start, end) = plan.byte_ranges[0];

    let reader: IndexedVvecReader<i32> = IndexedVvecReader::open(vv.to_str().unwrap()).unwrap();

    // Every byte of every record in the window must lie inside the
    // range the prefetch resolved — a record whose tail falls outside
    // it would fault in a chunk the prefetch never asked for.
    for i in 1..4usize {
        let dim = reader.dim_at(i).unwrap();
        let record_start = offsets[i];
        let record_end = record_start + 4 + (dim as u64) * 4;
        assert!(
            record_start >= start && record_end <= end,
            "record {i} spans [{record_start}, {record_end}) but the prefetch \
             resolved [{start}, {end})"
        );
        assert_eq!(reader.get_bytes(i).unwrap().len(), dim * 4);
    }

    // And it must not overreach: record 4 is outside the window, so its
    // first byte must be at or after the end of the range.
    assert!(
        offsets[4] >= end,
        "the range should stop at record 4, not include it"
    );
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

    // A parquet facet: row-group structure, so a record range snaps
    // outward by an amount only the footer knows. Deferred by design.
    let pq = ds.join("profiles/default/metadata_content.parquet");
    std::fs::write(&pq, [0u8; 64]).unwrap();

    let yaml = r#"
name: prefetch-degrade
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/metadata_content.parquet
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("2..4").unwrap())
        .unwrap();
    assert!(
        plan.degrades_to_full_download,
        "parquet ordinal windowing is deferred and must say so"
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

// ─── The precache driver ───────────────────────────────────────────

use vectordata::datasets::precache::{PrecacheRequest, run};

fn request(spec: &str) -> PrecacheRequest {
    PrecacheRequest {
        dataset_spec: spec.to_string(),
        ..PrecacheRequest::default()
    }
}

/// A windowless run is the original behaviour, not a special case of
/// the windowed one — every caller that predates windows keeps working.
#[test]
fn a_windowless_run_still_precaches_everything() {
    let tmp = tempfile::tempdir().unwrap();
    let group_dir = tmp.path().join("ds");
    dataset(tmp.path());
    assert_eq!(run(request(group_dir.to_str().unwrap())), 0);
}

/// `--plan` reports without fetching, and reports success.
#[test]
fn plan_only_reports_and_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path());
    let ds = tmp.path().join("ds");
    let req = PrecacheRequest {
        plan_only: true,
        ..request(ds.to_str().unwrap())
    };
    assert_eq!(run(req), 0);
}

/// **A malformed window fails before anything is opened.** Discovering
/// it after a catalog round-trip wastes the user's time for no reason,
/// and the message has to name the separator.
#[test]
fn a_malformed_window_fails_early_with_a_usable_message() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path());
    let ds = tmp.path().join("ds");
    let req = PrecacheRequest {
        window: Some("0,1000".to_string()),
        plan_only: true,
        ..request(ds.to_str().unwrap())
    };
    assert_eq!(run(req), 2, "a bad window is a usage error, not a failure");
}

/// Naming a facet that does not exist stops the run rather than
/// quietly fetching the ones that do and reporting success.
#[test]
fn an_unknown_facet_stops_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path());
    let ds = tmp.path().join("ds");
    let req = PrecacheRequest {
        facets: vec!["not_a_facet".to_string()],
        plan_only: true,
        ..request(ds.to_str().unwrap())
    };
    assert_eq!(run(req), 2);
}

/// A window selects a subset, so it needs one profile to resolve
/// against — the same facet name means different bytes in different
/// profiles, and picking one silently would be a guess presented as a
/// result.
#[test]
fn a_window_against_several_profiles_asks_which_one() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 100);
    let yaml = r#"
name: multi
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
  other:
    base_vectors: profiles/default/base_vectors.fvec[0..10)
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let ambiguous = PrecacheRequest {
        window: Some("0..10".to_string()),
        plan_only: true,
        ..request(ds.to_str().unwrap())
    };
    assert_eq!(run(ambiguous), 2, "two profiles, no way to choose");

    // Naming the profile resolves it. A local path cannot carry a
    // `:profile` suffix — resolve_spec reads anything with a `/` as
    // naming every profile — so the field is the only way to say it.
    let named = PrecacheRequest {
        window: Some("0..10".to_string()),
        plan_only: true,
        profile: Some("default".to_string()),
        ..request(ds.to_str().unwrap())
    };
    assert_eq!(run(named), 0);
}
