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
use vectordata::WholeFacetFallback;
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
        .prefetch(
            "base_vectors",
            &parse_window("10..20").unwrap(),
            WholeFacetFallback::Refuse,
        )
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

    // A parquet facet. Ordinal windowing of parquet is excluded by
    // design — not unimplemented — so this is the settled answer for
    // the format rather than a placeholder for a better one.
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
        "parquet ordinal windowing is excluded by design and must say so"
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

// ─── Background prefetch ───────────────────────────────────────────

/// The plan is available immediately, before any fetching — that is the
/// point of computing it on the calling thread. A caller can look at the
/// cost and cancel before the worker has done anything.
#[test]
fn a_background_prefetch_reports_its_plan_before_finishing() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let handle = view
        .prefetch_in_background(
            "base_vectors",
            &parse_window("10..20").unwrap(),
            WholeFacetFallback::Refuse,
        )
        .unwrap();
    assert_eq!(
        handle.plan().byte_ranges,
        vec![(10 * BPR, 20 * BPR)],
        "the plan is known before the fetch is"
    );
    let report = handle.join().unwrap();
    assert_eq!(report.planned.byte_ranges, vec![(10 * BPR, 20 * BPR)]);
}

/// Joining waits, and a local facet finishes with nothing to do.
#[test]
fn joining_a_background_prefetch_waits_for_it() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let handle = view
        .prefetch_in_background(
            "base_vectors",
            &parse_window("0..100").unwrap(),
            WholeFacetFallback::Refuse,
        )
        .unwrap();
    let report = handle.join().unwrap();
    assert_eq!(report.ranges_fetched, 1, "one range, fetched");
}

/// **Cancellation is granular to a range, and partial work survives.**
/// A cancelled prefetch is work not finished, not work undone — the
/// ranges already fetched stay in the cache.
#[test]
fn cancelling_stops_the_worker_and_keeps_what_it_fetched() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let handle = view
        .prefetch_in_background(
            "base_vectors",
            &parse_window("0..100").unwrap(),
            WholeFacetFallback::Refuse,
        )
        .unwrap();
    handle.cancel();
    assert!(handle.is_cancelled());
    // Joining a cancelled prefetch is not an error: stopping early is
    // what was asked for.
    let report = handle.join().unwrap();
    assert!(report.ranges_fetched <= 1);
}

/// Dropping the handle detaches rather than blocking or aborting. The
/// bytes still land, which is what a caller who has moved on wants.
#[test]
fn dropping_the_handle_detaches_without_blocking() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    {
        let handle = view
            .prefetch_in_background(
                "base_vectors",
                &parse_window("0..50").unwrap(),
                WholeFacetFallback::Refuse,
            )
            .unwrap();
        assert_eq!(handle.plan().requests(), 1);
        // Dropped here without joining.
    }
    // The view is still usable and the facet still reads correctly.
    let reader = view.base_vectors().unwrap();
    assert_eq!(reader.count(), 100);
}

/// **Fetching a whole facet is something the caller says yes to.**
///
/// A window that cannot be resolved for its format is refused by
/// default: asking for records 2..4 and silently receiving the entire
/// facet is the surprise this whole feature exists to prevent. The plan
/// still reports it, so the caller can see the size and decide.
#[test]
fn an_unresolvable_window_is_refused_unless_allowed() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 10);
    std::fs::write(ds.join("profiles/default/m.parquet"), [0u8; 64]).unwrap();
    let yaml = r#"
name: bg-degrade
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/m.parquet
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let window = parse_window("2..4").unwrap();

    // Planning always tells you. It fetches nothing, so it needs no
    // permission — finding out is how you decide.
    let plan = view.prefetch_plan("metadata_content", &window).unwrap();
    assert!(plan.degrades_to_full_download);
    assert_eq!(plan.facet_bytes, 64);

    // Fetching without permission is refused, and the message carries
    // the size, because that is the decision being asked for.
    let refused = view
        .prefetch("metadata_content", &window, WholeFacetFallback::Refuse)
        .expect_err("an unresolvable window must not quietly fetch everything");
    assert!(refused.to_string().contains("whole facet"), "{refused}");
    assert!(refused.to_string().contains("64"), "{refused}");

    // Refusal is the default in every form.
    assert!(
        view.prefetch_in_background("metadata_content", &window, WholeFacetFallback::Refuse)
            .is_err()
    );
    assert_eq!(WholeFacetFallback::default(), WholeFacetFallback::Refuse);

    // With permission it proceeds.
    view.prefetch("metadata_content", &window, WholeFacetFallback::Allow)
        .unwrap();
    view.prefetch_in_background("metadata_content", &window, WholeFacetFallback::Allow)
        .unwrap()
        .join()
        .unwrap();
}

/// **No window is a request, not a fallback.** Prefetching an
/// unmappable facet with no window asks for the whole thing on purpose,
/// so it needs no permission and reports no degrade.
#[test]
fn a_windowless_prefetch_of_an_unmappable_facet_needs_no_permission() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 10);
    std::fs::write(ds.join("profiles/default/m.parquet"), [0u8; 64]).unwrap();
    let yaml = r#"
name: windowless-degrade
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/m.parquet
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("").unwrap())
        .unwrap();
    assert!(
        !plan.degrades_to_full_download,
        "asking for everything and getting everything is not a degrade"
    );
    assert_eq!(plan.byte_ranges, vec![(0, 64)]);

    // And so it needs no permission.
    view.prefetch(
        "metadata_content",
        &parse_window("").unwrap(),
        WholeFacetFallback::Refuse,
    )
    .unwrap();
}

/// Several background prefetches can run at once against one view —
/// each holds its own facet handle, so nothing is shared that would
/// need locking.
#[test]
fn several_background_prefetches_run_concurrently() {
    let tmp = tempfile::tempdir().unwrap();
    let group = dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let handles: Vec<_> = [(0u64, 20u64), (30, 50), (60, 90)]
        .iter()
        .map(|(a, b)| {
            view.prefetch_in_background(
                "base_vectors",
                &parse_window(&format!("{a}..{b}")).unwrap(),
                WholeFacetFallback::Refuse,
            )
            .unwrap()
        })
        .collect();

    for h in handles {
        let report = h.join().unwrap();
        assert_eq!(report.ranges_fetched, 1);
    }
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

/// The CLI gate: a window that cannot be resolved stops the run and
/// names the flag, rather than fetching everything and reporting
/// success.
#[test]
fn the_cli_refuses_a_whole_facet_fetch_without_the_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, 10);
    std::fs::write(ds.join("profiles/default/m.parquet"), [0u8; 64]).unwrap();
    let yaml = r#"
name: cli-degrade
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/m.parquet
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();
    let spec = ds.to_str().unwrap();

    let refused = PrecacheRequest {
        facets: vec!["metadata_content".to_string()],
        window: Some("2..4".to_string()),
        ..request(spec)
    };
    assert_eq!(run(refused), 2, "no flag, no whole-facet fetch");

    let allowed = PrecacheRequest {
        facets: vec!["metadata_content".to_string()],
        window: Some("2..4".to_string()),
        allow_whole_facet: true,
        ..request(spec)
    };
    assert_eq!(run(allowed), 0);

    // --plan is always allowed: it reports without fetching, which is
    // how a user finds out the flag is needed.
    let planned = PrecacheRequest {
        facets: vec!["metadata_content".to_string()],
        window: Some("2..4".to_string()),
        plan_only: true,
        ..request(spec)
    };
    assert_eq!(run(planned), 0);
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

// ─── Against a real remote facet ───────────────────────────────────
//
// Every test above runs against local storage, where a fetch is a
// no-op. That exercises the plumbing but not the machinery: nothing
// downloads, no counter advances, and cancellation has nothing to
// cancel. These run over HTTP with a published `.mref`, so the chunk
// bitmap is real and the numbers mean something.

mod support;

use support::testserver::TestServer;
use vectordata::merkle::MerkleRef;

/// Small enough that a modest fixture spans many chunks, so a window
/// covers some of them and not others.
const REMOTE_CHUNK: u64 = 4 * 1024;

/// One cache root for the whole binary.
///
/// `override_cache_dir_for_process` is process-wide, and these tests
/// run in parallel threads of one process — so a per-test override is
/// a race, with the last writer deciding where everyone caches. One
/// root, and a distinct dataset name per test, keeps them apart.
static TEST_CACHE_DIR: std::sync::LazyLock<tempfile::TempDir> = std::sync::LazyLock::new(|| {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).expect("create test cache root")
});

fn init_test_cache() {
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
}

/// Publish an fvec over HTTP with a `.mref`, and return a local
/// `dataset.yaml` pointing at it.
///
/// `name` must be unique per test: it becomes the cache subdirectory,
/// and two datasets sharing one while fetching from different servers
/// is a collision the cache correctly refuses.
fn remote_dataset(dir: &std::path::Path, name: &str) -> (TestServer, String) {
    init_test_cache();
    let published = dir.join("pub");
    std::fs::create_dir_all(&published).unwrap();

    // 4000 records of dim 16 → 68 bytes each, ~265 KiB, ~65 chunks.
    let data = published.join("base.fvec");
    write_fvec(&data, 16, 4000);
    let content = std::fs::read(&data).unwrap();
    MerkleRef::from_content(&content, REMOTE_CHUNK)
        .save(&published.join("base.fvec.mref"))
        .unwrap();

    let server = TestServer::start(&published).unwrap();
    let yaml = format!(
        "name: {name}\nprofiles:\n  default:\n    base_vectors: {}base.fvec\n",
        server.base_url()
    );
    let local_ds = dir.join(name);
    std::fs::create_dir_all(&local_ds).unwrap();
    std::fs::write(local_ds.join("dataset.yaml"), yaml).unwrap();
    (server, local_ds.to_string_lossy().to_string())
}

/// Publish an ivvec with its `IDXFOR__` sidecar and a `.mref`, plus an
/// unwindowable facet, alongside the fvec.
///
/// The plain fixture above covers one xvec facet. Several gaps need
/// more: a variable-length facet whose index must be fetched over HTTP,
/// and a facet with no ordinal mapping at all so the fallback gate can
/// be exercised where the whole facet actually costs something.
fn rich_remote_dataset(dir: &std::path::Path, name: &str) -> (TestServer, String) {
    init_test_cache();
    let published = dir.join("pub");
    std::fs::create_dir_all(&published).unwrap();

    let mref = |p: &std::path::Path| {
        let content = std::fs::read(p).unwrap();
        let mut out = p.to_path_buf().into_os_string();
        out.push(".mref");
        MerkleRef::from_content(&content, REMOTE_CHUNK)
            .save(std::path::Path::new(&out))
            .unwrap();
    };

    // dim 16 → 68 bytes per record; 4 KiB chunks hold ~60 records.
    let fvec = published.join("base.fvec");
    write_fvec(&fvec, 16, 4000);
    mref(&fvec);

    // Ragged records, so only the index can place them.
    let vv = published.join("meta.ivvec");
    let dims: Vec<i32> = (0..400).map(|i| 1 + (i % 17)).collect();
    let offsets = write_ivvec(&vv, &dims);
    mref(&vv);
    // The sidecar the reader and the prefetch path both look for.
    let total = std::fs::metadata(&vv).unwrap().len();
    let (ext, bytes): (&str, Vec<u8>) = if total <= i32::MAX as u64 {
        (
            "i32",
            offsets
                .iter()
                .flat_map(|&o| (o as i32).to_le_bytes())
                .collect(),
        )
    } else {
        (
            "i64",
            offsets
                .iter()
                .flat_map(|&o| (o as i64).to_le_bytes())
                .collect(),
        )
    };
    std::fs::write(published.join(format!("IDXFOR__meta.ivvec.{ext}")), bytes).unwrap();

    // No ordinal mapping: windowing it is excluded by design.
    let blob = published.join("blob.parquet");
    std::fs::write(&blob, vec![7u8; 40_000]).unwrap();
    mref(&blob);

    let server = TestServer::start(&published).unwrap();
    let base = server.base_url();
    let yaml = format!(
        "name: {name}\nprofiles:\n  default:\n    base_vectors: {base}base.fvec\n    \
         metadata_content: {base}meta.ivvec\n    metadata_predicates: {base}blob.parquet\n"
    );
    let local_ds = dir.join(name);
    std::fs::create_dir_all(&local_ds).unwrap();
    std::fs::write(local_ds.join("dataset.yaml"), yaml).unwrap();
    (server, local_ds.to_string_lossy().to_string())
}

/// Records per chunk for the fvec fixture: 4 KiB / 68 bytes.
const RECORDS_PER_CHUNK: u64 = REMOTE_CHUNK / 68;

/// **Coalescing on real chunks — the branch the local tests never
/// reach.**
///
/// Every other coalescing test runs against local storage, where
/// `cache_stats()` is `None` and merging falls back to byte adjacency.
/// That proves `coalesce_ranges` in isolation but never that
/// `prefetch_plan` hands it the right chunk size. Here the chunks are
/// real, so the chunk-space rule is what decides.
#[test]
fn coalescing_uses_real_chunk_boundaries() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = rich_remote_dataset(tmp.path(), "coalesce-chunks");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();

    let plan = |w: &str| {
        view.prefetch_plan("base_vectors", &parse_window(w).unwrap())
            .unwrap()
    };

    // Two windows inside chunk 0. Byte-adjacency would keep them apart;
    // chunk adjacency merges them, because they are already one fetch.
    let same_chunk = plan("[0..10, 12..20]");
    assert_eq!(same_chunk.requested_ranges.len(), 2);
    assert_eq!(
        same_chunk.requests(),
        1,
        "two ranges inside one chunk are one fetch, not two"
    );

    // Chunk 0 and chunk 1: adjacent, contiguous on the device.
    let adjacent = plan(&format!(
        "[0..10, {}..{}]",
        RECORDS_PER_CHUNK + 1,
        RECORDS_PER_CHUNK + 10
    ));
    assert_eq!(adjacent.requested_ranges.len(), 2);
    assert_eq!(adjacent.requests(), 1, "adjacent chunks merge");

    // Far apart, with whole chunks untouched between: no bridge.
    let apart = plan("[0..10, 200..210]");
    assert_eq!(apart.requested_ranges.len(), 2);
    assert_eq!(
        apart.requests(),
        2,
        "a gap of whole chunks must not be bridged"
    );
    assert!(
        apart.fills.len() == 2,
        "each request gets its own chunk accounting"
    );
}

/// **A remote vvec window resolves through an index fetched over HTTP.**
///
/// The local vvec tests build the index by walking an mmap. This is the
/// two-phase path as designed: fetch the sidecar, then window the data.
#[test]
fn a_remote_vvec_window_uses_the_published_index() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = rich_remote_dataset(tmp.path(), "vvec-remote");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("100..150").unwrap())
        .unwrap();
    assert!(
        !plan.degrades_to_full_download,
        "a published index makes a remote vvec windowable"
    );
    assert_eq!(plan.requested_ranges.len(), 1);
    assert!(
        plan.prerequisite_bytes > 0,
        "the index had to be read, and the plan says so"
    );
    let (start, end) = plan.requested_ranges[0];
    assert!(end > start && end <= plan.facet_bytes);
    assert!(
        end - start < plan.facet_bytes,
        "50 of 400 ragged records must be a fraction of the facet"
    );

    // And it fetches: afterwards the window reads as resident.
    view.prefetch(
        "metadata_content",
        &parse_window("100..150").unwrap(),
        WholeFacetFallback::Refuse,
    )
    .unwrap();
    let after = view
        .prefetch_plan("metadata_content", &parse_window("100..150").unwrap())
        .unwrap();
    assert!(after.is_resident());
}

/// **A window fetches its chunks and not the file.** This is the whole
/// claim, and it needs remote storage to mean anything: on a local
/// facet every plan reports zero and proves nothing.
#[test]
fn a_remote_window_fetches_only_its_chunks() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = remote_dataset(tmp.path(), "prefetch-window-chunks");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("base_vectors", &parse_window("100..200").unwrap())
        .unwrap();
    assert!(!plan.fills.is_empty(), "remote storage has chunk fills");
    assert!(
        plan.bytes_to_fetch() > 0,
        "nothing is resident yet, so there is work to do"
    );
    assert!(
        plan.bytes_to_fetch() < plan.facet_bytes,
        "a 100-record window must cost less than the whole {} byte facet, \
         got {}",
        plan.facet_bytes,
        plan.bytes_to_fetch()
    );

    let handle = view
        .prefetch_in_background(
            "base_vectors",
            &parse_window("100..200").unwrap(),
            WholeFacetFallback::Refuse,
        )
        .unwrap();
    handle.join().unwrap();

    // Asking again now costs nothing: the chunks are resident.
    let after = view
        .prefetch_plan("base_vectors", &parse_window("100..200").unwrap())
        .unwrap();
    assert!(
        after.is_resident(),
        "the window the prefetch just fetched must read as resident"
    );
    assert_eq!(after.bytes_to_fetch(), 0);
}

/// The background worker's counters advance against a real download,
/// and the handle reports done when it is.
#[test]
fn a_background_prefetch_advances_its_counters() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = remote_dataset(tmp.path(), "prefetch-counters");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();

    let handle = view
        .prefetch_in_background(
            "base_vectors",
            &parse_window("0..2000").unwrap(),
            WholeFacetFallback::Refuse,
        )
        .unwrap();
    let expected = handle.plan().bytes_to_fetch();
    assert!(expected > 0);

    let report = handle.join().unwrap();
    assert_eq!(report.ranges_fetched, 1);

    // What the plan said it would fetch is now resident.
    let after = view
        .prefetch_plan("base_vectors", &parse_window("0..2000").unwrap())
        .unwrap();
    assert!(after.is_resident());
}

/// **A prefetched window makes the reader's own reads free.** The
/// point of prefetch is that the chunks the reader wants are already
/// there — so a read after a prefetch must not fetch anything more.
#[test]
fn reading_a_prefetched_window_fetches_nothing_further() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = remote_dataset(tmp.path(), "prefetch-read-after");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();

    view.prefetch(
        "base_vectors",
        &parse_window("500..600").unwrap(),
        WholeFacetFallback::Refuse,
    )
    .unwrap();

    let before = view
        .open_facet_storage("base_vectors")
        .unwrap()
        .cache_stats()
        .map(|c| c.valid_chunks)
        .unwrap_or(0);

    // Read inside the prefetched window.
    let reader = view.base_vectors().unwrap();
    for i in 500..600 {
        let v = reader.get(i).unwrap();
        assert_eq!(v.len(), 16);
        // This file's write_fvec stores element (r, d) as r + d.
        assert_eq!(v[0], i as f32, "record {i} decodes correctly");
    }

    let after = view
        .open_facet_storage("base_vectors")
        .unwrap()
        .cache_stats()
        .map(|c| c.valid_chunks)
        .unwrap_or(0);
    assert_eq!(
        before, after,
        "reading inside a prefetched window must not fetch another chunk"
    );
}

/// **Progress is reported, not merely plumbed.**
///
/// Nothing else asserts these counters. The precache renderer adapts
/// them into its meter, so if the worker never advanced them — or
/// advanced them wrongly — every other test would still pass and the
/// meter would sit at zero.
#[test]
fn a_background_prefetch_reports_bytes_and_ranges() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = rich_remote_dataset(tmp.path(), "progress-counters");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();

    let handle = view
        .prefetch_in_background(
            "base_vectors",
            &parse_window("[0..200, 1000..1200, 3000..3200]").unwrap(),
            WholeFacetFallback::Refuse,
        )
        .unwrap();
    let planned_ranges = handle.plan().requests();
    let planned_bytes = handle.plan().bytes_to_fetch();
    assert!(planned_bytes > 0, "nothing is resident yet");

    let report = handle.join().unwrap();
    assert_eq!(
        report.ranges_fetched, planned_ranges,
        "every planned range must be accounted for"
    );

    // The blocking form reports through its callback.
    let mut seen_bytes = 0u64;
    let mut calls = 0usize;
    view.prefetch_with_progress(
        "metadata_content",
        &parse_window("0..400").unwrap(),
        WholeFacetFallback::Refuse,
        &mut |p| {
            calls += 1;
            seen_bytes = seen_bytes.max(p.downloaded_bytes());
        },
    )
    .unwrap();
    assert!(calls > 0, "the progress callback must actually fire");
    assert!(seen_bytes > 0, "and carry a byte count");
}

/// **Cancellation stops a real fetch, and partial work survives.**
///
/// The local cancel tests cancel an instantaneous no-op. Here there are
/// many separated ranges over HTTP, so cancelling between them is
/// observable: the run stops short of the plan and what was already
/// fetched stays cached.
#[test]
fn cancelling_a_real_fetch_stops_short_and_keeps_its_work() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = rich_remote_dataset(tmp.path(), "cancel-real");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();

    // Widely separated windows so they cannot coalesce: many ranges,
    // many chances to notice the flag between them.
    let spec_w: Vec<String> = (0..20)
        .map(|i| format!("{}..{}", i * 190, i * 190 + 5))
        .collect();
    let window = parse_window(&format!("[{}]", spec_w.join(", "))).unwrap();

    let handle = view
        .prefetch_in_background("base_vectors", &window, WholeFacetFallback::Refuse)
        .unwrap();
    let planned = handle.plan().requests();
    assert!(
        planned > 10,
        "the plan must have enough ranges to stop between"
    );

    handle.cancel();
    let report = handle.join().unwrap();
    assert!(
        report.ranges_fetched <= planned,
        "a cancelled run cannot exceed its plan"
    );

    // Whatever it did fetch is still cached — cancelling is stopping,
    // not undoing.
    let storage = view.open_facet_storage("base_vectors").unwrap();
    let resident = storage.cache_stats().map(|c| c.valid_chunks).unwrap_or(0);
    assert!(
        resident as usize >= report.ranges_fetched.min(1),
        "ranges already fetched stay resident"
    );
}

/// **A fetch failure reaches the caller.**
///
/// Two distinct paths, and it matters which is which:
///
/// - The endpoint is gone *before* the facet is opened → the failure
///   is in opening, and `prefetch_in_background` returns `Err`
///   immediately without spawning anything.
/// - The facet is already open and the endpoint dies → the failure is
///   in the worker, and `join` is the only place it can surface.
///
/// Without the second, a prefetch could report success while fetching
/// nothing, and the caller would hit faults later with no idea why.
///
/// **Ignored by default because it takes about six minutes.** Retries
/// are unconditional — [`RetryPolicy`] makes ten attempts with
/// exponential backoff capped at 30 s — so *every* failure mode is
/// slow, and there is no process-level override to shorten it. That is
/// worth knowing on its own: a CLI user pointed at a dead mirror waits
/// the same six minutes with no way to ask for fail-fast.
///
/// Run it with `cargo test -p vectordata --test prefetch_windows --
/// --ignored`.
#[test]
#[ignore = "slow: unconditional retry backoff takes ~6 minutes to exhaust"]
fn a_failed_fetch_surfaces_from_wherever_it_failed() {
    let tmp = tempfile::tempdir().unwrap();
    let (server, spec) = rich_remote_dataset(tmp.path(), "fetch-failure");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();
    let window = parse_window("1000..2000").unwrap();

    // Plan while the server is up, and hold a handle so the storage
    // registry keeps serving opens after it dies — otherwise the open
    // fails first and the worker never runs.
    let keep_open = view.open_facet_storage("base_vectors").unwrap();
    let plan = view.prefetch_plan("base_vectors", &window).unwrap();
    assert!(
        plan.bytes_to_fetch() > 0,
        "there must be work left to fail at"
    );

    drop(server);

    // The worker's failure, surfaced through join.
    let background = view
        .prefetch_in_background("base_vectors", &window, WholeFacetFallback::Refuse)
        .expect("opening still succeeds from the registry")
        .join();
    assert!(
        background.is_err(),
        "a fetch against a dead endpoint must not report success"
    );

    // And the blocking form fails the same way.
    assert!(
        view.prefetch("base_vectors", &window, WholeFacetFallback::Refuse)
            .is_err()
    );
    drop(keep_open);
}

/// The gate's expensive side: allowing the fallback on a remote facet
/// really does fetch the whole thing, which is what the caller consented
/// to and why consent is required.
#[test]
fn allowing_the_fallback_fetches_a_whole_remote_facet() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = rich_remote_dataset(tmp.path(), "allow-remote");
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();
    let window = parse_window("2..4").unwrap();

    let plan = view.prefetch_plan("metadata_predicates", &window).unwrap();
    assert!(plan.degrades_to_full_download);
    assert_eq!(plan.facet_bytes, 40_000);
    assert_eq!(
        plan.bytes_to_fetch(),
        plan.facet_bytes,
        "the honest cost of the fallback is the whole facet"
    );

    assert!(
        view.prefetch("metadata_predicates", &window, WholeFacetFallback::Refuse)
            .is_err(),
        "refused without consent even though it is only 40 KB"
    );

    view.prefetch("metadata_predicates", &window, WholeFacetFallback::Allow)
        .unwrap();
    let storage = view.open_facet_storage("metadata_predicates").unwrap();
    assert!(
        storage.is_complete(),
        "consenting to the whole facet fetches the whole facet"
    );
}

/// **A mixed selection is refused whole, before anything is fetched.**
///
/// The driver checks every facet's plan before fetching any of them,
/// specifically so a run cannot fetch the facets that window and then
/// fail on the one that does not — leaving the cache half populated for
/// a reason the user could have been told up front. Nothing tested
/// that: every other CLI test selects one facet.
#[test]
fn a_mixed_selection_is_refused_before_any_of_it_is_fetched() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, spec) = rich_remote_dataset(tmp.path(), "cli-mixed");

    let refused = PrecacheRequest {
        facets: vec![
            "base_vectors".to_string(),        // windowable
            "metadata_predicates".to_string(), // not
        ],
        window: Some("0..100".to_string()),
        ..request(&spec)
    };
    assert_eq!(run(refused), 2, "one unwindowable facet refuses the set");

    // And nothing was fetched: the windowable facet is untouched.
    let group = vectordata::TestDataGroup::load(&spec).unwrap();
    let view = group.profile("default").unwrap();
    let plan = view
        .prefetch_plan("base_vectors", &parse_window("0..100").unwrap())
        .unwrap();
    assert!(
        !plan.is_resident(),
        "a refused run must not have fetched the facets it could have"
    );

    // With consent the whole set proceeds.
    let allowed = PrecacheRequest {
        facets: vec![
            "base_vectors".to_string(),
            "metadata_predicates".to_string(),
        ],
        window: Some("0..100".to_string()),
        allow_whole_facet: true,
        ..request(&spec)
    };
    assert_eq!(run(allowed), 0);
}

/// A server that ignores `Range` has no chunk bitmap to plan against,
/// so a window has nothing to be partial about. The plan must say the
/// cost is the whole facet rather than reporting a partial fetch it
/// cannot perform.
#[test]
fn a_server_without_range_support_plans_the_whole_facet() {
    init_test_cache();
    let tmp = tempfile::tempdir().unwrap();
    let published = tmp.path().join("pub");
    std::fs::create_dir_all(&published).unwrap();
    let data = published.join("base.fvec");
    write_fvec(&data, 16, 2000);

    let server = TestServer::start_no_range(&published).unwrap();
    let yaml = format!(
        "name: no-range\nprofiles:\n  default:\n    base_vectors: {}base.fvec\n",
        server.base_url()
    );
    let ds = tmp.path().join("no-range");
    std::fs::create_dir_all(&ds).unwrap();
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let plan = view
        .prefetch_plan("base_vectors", &parse_window("100..200").unwrap())
        .unwrap();

    // The window resolves — the format is uniform-stride — but there
    // is nothing partial left to do: a server that ignores Range makes
    // the storage layer fetch the whole file at open, so by planning
    // time every chunk is resident.
    assert!(
        plan.is_resident(),
        "no-range storage arrives whole, so a window has nothing to fetch: \
         to_fetch={} facet={}",
        plan.bytes_to_fetch(),
        plan.facet_bytes
    );
    assert_eq!(plan.bytes_to_fetch(), 0);
    assert!(
        !plan.degrades_to_full_download,
        "the window was resolvable; the facet simply arrived early"
    );

    // Which means prefetching it is a no-op rather than a second
    // download, and needs no whole-facet consent.
    view.prefetch(
        "base_vectors",
        &parse_window("100..200").unwrap(),
        WholeFacetFallback::Refuse,
    )
    .unwrap();
}

// ─── Catalog-resolved profile selection ────────────────────────────

/// A local catalog declaring one dataset with two profiles over the
/// same fvec, so selecting between them changes which window resolves
/// against which facet.
fn catalog_fixture(dir: &std::path::Path) -> String {
    let root = dir.join("cat");
    std::fs::create_dir_all(root.join("data")).unwrap();
    write_fvec(&root.join("data/base.fvec"), 4, 100);
    write_fvec(&root.join("data/small.fvec"), 4, 20);
    let yaml = r#"
"windowed:default":
  base: data/base.fvec

"windowed:small":
  base: data/small.fvec
"#;
    std::fs::write(root.join("knn_entries.yaml"), yaml).unwrap();
    root.to_string_lossy().to_string()
}

/// **Profile selection works for a catalog-resolved dataset, by both
/// routes.**
///
/// A catalog name has no `/`, so unlike a local path it *can* carry a
/// `:profile` suffix — `resolve_spec` reads anything with a slash as
/// naming every profile. Both routes have to reach the same place, or
/// the flag and the suffix mean different things depending on how the
/// dataset was named.
#[test]
fn a_catalog_dataset_selects_its_profile_by_flag_or_suffix() {
    let tmp = tempfile::tempdir().unwrap();
    let catalog = catalog_fixture(tmp.path());

    let base = |spec: &str| PrecacheRequest {
        dataset_spec: spec.to_string(),
        extra_catalogs: vec![catalog.clone()],
        window: Some("0..10".to_string()),
        plan_only: true,
        ..PrecacheRequest::default()
    };

    // Two profiles and no way to choose: refused rather than guessed.
    assert_eq!(
        run(base("windowed")),
        2,
        "a windowed selection across two profiles must ask which one"
    );

    // The flag resolves it.
    assert_eq!(
        run(PrecacheRequest {
            profile: Some("small".to_string()),
            ..base("windowed")
        }),
        0
    );

    // And so does the suffix, which a catalog name can carry.
    assert_eq!(run(base("windowed:small")), 0);

    // A profile that does not exist fails rather than falling back.
    assert_eq!(
        run(PrecacheRequest {
            profile: Some("nonexistent".to_string()),
            ..base("windowed")
        }),
        1
    );

    // Exit codes alone would not prove the *right* profile was picked:
    // both succeed, so both could be resolving the same one. The two
    // profiles point at different files, so the plan's facet size is
    // what distinguishes them.
    use vectordata::catalog::resolver::Catalog;
    use vectordata::catalog::sources::CatalogSources;
    let sources = CatalogSources::new().add_catalogs(std::slice::from_ref(&catalog));
    let group = Catalog::of(&sources)
        .open("windowed")
        .expect("the catalog resolves the dataset");
    let window = parse_window("0..10").unwrap();

    let big = group
        .profile("default")
        .expect("default profile")
        .prefetch_plan("base_vectors", &window)
        .unwrap();
    let small = group
        .profile("small")
        .expect("small profile")
        .prefetch_plan("base_vectors", &window)
        .unwrap();

    assert_eq!(big.facet_bytes, 100 * 20, "default is the 100-record file");
    assert_eq!(small.facet_bytes, 20 * 20, "small is the 20-record file");
    assert_ne!(
        big.facet_bytes, small.facet_bytes,
        "the profiles must resolve to different facets, or selecting \
         between them proves nothing"
    );
    // The same window resolves to the same bytes in both, since the
    // stride is the same — what differs is what lies beyond it.
    assert_eq!(big.requested_ranges, small.requested_ranges);
}

/// The flag outranks the suffix when both are given, so a caller
/// scripting over a `name:profile` spec can override it without
/// rewriting the string.
#[test]
fn an_explicit_profile_flag_outranks_the_spec_suffix() {
    let tmp = tempfile::tempdir().unwrap();
    let catalog = catalog_fixture(tmp.path());

    // The suffix says `default` (100 records); the flag says `small`
    // (20). A window of 0..15 fits `small` and the run must succeed
    // against it.
    let req = PrecacheRequest {
        dataset_spec: "windowed:default".to_string(),
        extra_catalogs: vec![catalog],
        profile: Some("small".to_string()),
        window: Some("0..15".to_string()),
        plan_only: true,
        ..PrecacheRequest::default()
    };
    assert_eq!(run(req), 0);
}

/// **A sidecar is found beside the data, not beside the process.**
///
/// A `dataset.yaml` names its facets relative to the dataset root.
/// Resolving `IDXFOR__` from that string instead of from the file the
/// storage actually opened looks in the process working directory — so
/// a published index sitting next to the data is missed and every
/// window pays a full walk instead.
///
/// The fixture makes the difference observable: the data file has a
/// truncated final record, so walking it cannot succeed and only a
/// published index can place the records. If the sidecar is not found,
/// the window has no mapping at all.
#[test]
fn a_published_sidecar_beside_the_data_is_found() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("beside");
    std::fs::create_dir_all(&ds).unwrap();

    let vv = ds.join("meta.ivvec");
    let dims: Vec<i32> = (0..30).map(|i| 1 + (i % 5)).collect();
    let starts = write_ivvec(&vv, &dims);
    let complete = std::fs::metadata(&vv).unwrap().len();
    // A half-written final record: a walk stops at a boundary that is
    // not the end of the file and gives up.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&vv).unwrap();
        f.write_all(&[9u8, 0, 0]).unwrap();
    }

    let bytes: Vec<u8> = starts
        .iter()
        .flat_map(|&o| (o as i32).to_le_bytes())
        .collect();
    std::fs::write(ds.join("IDXFOR__meta.ivvec.i32"), bytes).unwrap();

    std::fs::write(
        ds.join("dataset.yaml"),
        "name: beside\nprofiles:\n  default:\n    metadata_content: meta.ivvec\n",
    )
    .unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("5..10").unwrap())
        .unwrap();
    assert!(
        !plan.degrades_to_full_download,
        "the sidecar beside the data places these records; only a \
         lookup against the working directory would miss it"
    );
    assert_eq!(plan.byte_ranges, vec![(starts[5], starts[10])]);
    assert!(complete > starts[10], "sanity: the window is interior");
}

/// **A rebuilt index is written beside the data, never into the
/// working directory.**
///
/// The walk's result is persisted so it is paid once. Deriving that
/// path from the caller's source string rather than the opened file
/// drops an `IDXFOR__` file into whatever directory the process is
/// running in — which then shadows every later open of any file with
/// the same basename, since the sidecar is found but the data beside it
/// is not.
#[test]
fn rebuilding_an_index_never_writes_into_the_working_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("rebuild");
    std::fs::create_dir_all(&ds).unwrap();

    // A basename no other test uses, so the working-directory probe
    // cannot collide with a parallel test's leavings. Clear any left
    // by an earlier run so the assertion below measures *this* one —
    // the failure mode being tested is a file appearing, and a stale
    // one would otherwise wedge the test permanently.
    let probe = std::path::Path::new("IDXFOR__cwdprobe.ivvec.i32");
    let _ = std::fs::remove_file(probe);
    let vv = ds.join("cwdprobe.ivvec");
    let dims: Vec<i32> = (0..25).map(|i| 1 + (i % 4)).collect();
    let starts = write_ivvec(&vv, &dims);
    // Deliberately no sidecar: the walk is the only way, and its
    // result gets persisted.

    std::fs::write(
        ds.join("dataset.yaml"),
        "name: rebuild\nprofiles:\n  default:\n    metadata_content: cwdprobe.ivvec\n",
    )
    .unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("3..8").unwrap())
        .unwrap();
    assert_eq!(plan.byte_ranges, vec![(starts[3], starts[8])]);

    assert!(
        !probe.exists(),
        "the rebuilt index landed in the working directory"
    );
    assert!(
        ds.join("IDXFOR__cwdprobe.ivvec.i32").is_file(),
        "the rebuilt index belongs beside the data it describes"
    );
}

/// **A sentinel sidecar describes N records, not N+1.**
///
/// `IDXFOR__` files are published in two layouts: `N` record starts,
/// and `N+1` entries whose last is the payload size, so a consumer can
/// take every extent as `offsets[i+1] - offsets[i]` without a special
/// case for the tail. Read verbatim, the second layout invents a
/// record beginning at EOF — the count is one too high and a window
/// reaching the end resolves to nothing.
#[test]
fn a_sentinel_sidecar_reads_as_the_records_it_describes() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("sentinel");
    std::fs::create_dir_all(&ds).unwrap();

    let vv = ds.join("meta.ivvec");
    let dims: Vec<i32> = (0..40).map(|i| 1 + (i % 7)).collect();
    let starts = write_ivvec(&vv, &dims);
    let total = std::fs::metadata(&vv).unwrap().len();

    // The sentinel layout: every start, then the payload size.
    let mut entries: Vec<u64> = starts.clone();
    entries.push(total);
    let bytes: Vec<u8> = entries
        .iter()
        .flat_map(|&o| (o as i32).to_le_bytes())
        .collect();
    std::fs::write(ds.join("IDXFOR__meta.ivvec.i32"), bytes).unwrap();

    std::fs::write(
        ds.join("dataset.yaml"),
        "name: sentinel\nprofiles:\n  default:\n    metadata_content: meta.ivvec\n",
    )
    .unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    // A window running to the last record must reach the end of the
    // file. With the sentinel kept, record 39 would start at
    // `starts[39]` and end at `starts[40]` — the phantom — leaving the
    // final record's bytes outside the window.
    let plan = view
        .prefetch_plan("metadata_content", &parse_window("38..40").unwrap())
        .unwrap();
    assert_eq!(
        plan.byte_ranges,
        vec![(starts[38], total)],
        "the last real record ends at the payload size"
    );
    assert_eq!(
        plan.prerequisite_bytes,
        40 * 8,
        "the index is counted as record starts, so both published \
         layouts report the same prerequisite"
    );
}

/// The starts-only layout is unchanged by the sentinel tolerance —
/// dropping a trailing entry must key on it *equalling the payload
/// size*, not on it being last.
#[test]
fn a_starts_only_sidecar_still_describes_every_record() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path().join("starts");
    std::fs::create_dir_all(&ds).unwrap();

    let vv = ds.join("meta.ivvec");
    let dims: Vec<i32> = (0..40).map(|i| 1 + (i % 7)).collect();
    let starts = write_ivvec(&vv, &dims);
    let total = std::fs::metadata(&vv).unwrap().len();
    let bytes: Vec<u8> = starts
        .iter()
        .flat_map(|&o| (o as i32).to_le_bytes())
        .collect();
    std::fs::write(ds.join("IDXFOR__meta.ivvec.i32"), bytes).unwrap();

    std::fs::write(
        ds.join("dataset.yaml"),
        "name: starts\nprofiles:\n  default:\n    metadata_content: meta.ivvec\n",
    )
    .unwrap();
    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("38..40").unwrap())
        .unwrap();
    assert_eq!(plan.byte_ranges, vec![(starts[38], total)]);
    assert_eq!(plan.prerequisite_bytes, 40 * 8);
}

/// **Planning must not perform the transfer it is pricing.**
///
/// A remote vvec with no published `IDXFOR__` sidecar has only one
/// remaining way to learn its record boundaries: walk the file record
/// by record. Over HTTP that walk touches every chunk — so a call
/// documented as reporting cost *before* anything moves would move the
/// entire facet, and the whole-facet consent gate would then fire after
/// the bytes had already arrived. The window has to degrade instead and
/// let the caller decide.
#[test]
fn planning_a_remote_vvec_without_a_sidecar_downloads_nothing() {
    init_test_cache();
    let tmp = tempfile::tempdir().unwrap();
    let published = tmp.path().join("pub");
    std::fs::create_dir_all(&published).unwrap();

    let vv = published.join("meta.ivvec");
    let dims: Vec<i32> = (0..2000).map(|i| 1 + (i % 23)).collect();
    write_ivvec(&vv, &dims);
    let content = std::fs::read(&vv).unwrap();
    MerkleRef::from_content(&content, REMOTE_CHUNK)
        .save(&published.join("meta.ivvec.mref"))
        .unwrap();
    // Deliberately no IDXFOR__meta.ivvec.* published.

    let server = TestServer::start(&published).unwrap();
    let yaml = format!(
        "name: no-sidecar\nprofiles:\n  default:\n    metadata_content: {}meta.ivvec\n",
        server.base_url()
    );
    let ds = tmp.path().join("no-sidecar");
    std::fs::create_dir_all(&ds).unwrap();
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let plan = view
        .prefetch_plan("metadata_content", &parse_window("100..200").unwrap())
        .unwrap();

    assert!(
        plan.degrades_to_full_download,
        "with no published index the window cannot be placed without \
         reading the whole file, which is what a degrade means"
    );
    assert!(
        !plan.is_resident(),
        "planning fetched the facet it was asked to price: to_fetch={} facet={}",
        plan.bytes_to_fetch(),
        plan.facet_bytes
    );
    assert_eq!(
        plan.bytes_to_fetch(),
        plan.facet_bytes,
        "a degraded plan costs the whole facet, and none of it is paid yet"
    );

    // And the gate still governs: the transfer happens only on consent.
    view.prefetch(
        "metadata_content",
        &parse_window("100..200").unwrap(),
        WholeFacetFallback::Refuse,
    )
    .expect_err("a degraded window must be refused without consent");

    let after = view
        .prefetch_plan("metadata_content", &parse_window("100..200").unwrap())
        .unwrap();
    assert!(
        !after.is_resident(),
        "a refused prefetch must leave the facet untouched"
    );

    view.prefetch(
        "metadata_content",
        &parse_window("100..200").unwrap(),
        WholeFacetFallback::Allow,
    )
    .expect("with consent the whole facet is fetched");
}

// ─── Format coverage the mapping path had gaps in ──────────────────
//
// Three places where the record→byte mapping disagreed with how the
// data is actually read or published. Each is a silent failure: a
// window that resolves to the wrong bytes, or a plan that moves the
// file it was asked to merely price.

/// Write a scalar facet: raw packed values, no header of any kind.
fn write_u32_scalar(path: &std::path::Path, values: &[u32]) {
    let mut f = std::fs::File::create(path).unwrap();
    for v in values {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

/// A dataset with two scalar facets and no vector facet windowing in
/// the way.
fn scalar_dataset(dir: &std::path::Path) -> vectordata::TestDataGroup {
    let ds = dir.join("scalars");
    std::fs::create_dir_all(&ds).unwrap();

    // Values from 7 up. The first four bytes are `7` — a plausible
    // dimension, which is exactly what makes the xvec path dangerous
    // here rather than merely unhelpful.
    let vals: Vec<u32> = (0..100).map(|i| 7 + i).collect();
    write_u32_scalar(&ds.join("layout.u32"), &vals);

    // Bytes chosen so the same four bytes read as a *negative* dim.
    let content: Vec<u8> = (0..200u32).map(|i| 200 + (i % 40) as u8).collect();
    std::fs::write(ds.join("content.u8"), content).unwrap();

    std::fs::write(
        ds.join("dataset.yaml"),
        "name: scalars\nprofiles:\n  default:\n    metadata_layout: layout.u32\n    \
         metadata_content: content.u8\n",
    )
    .unwrap();
    vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap()
}

/// **A scalar facet has no dimension header, so it must not be read as
/// if it had one.**
///
/// `TypedReader` addresses a scalar facet at `ordinal * elem_size`.
/// Routing it through the uniform-xvec branch instead takes the first
/// four bytes as a dimension and derives a stride of
/// `4 + dim * elem_size` — here `4 + 7*4 = 32` rather than `4`, so
/// every byte range past the first is wrong. Wrong, not absent: the
/// plan reports success and warms a region the reader never touches.
#[test]
fn a_scalar_window_maps_at_the_element_stride() {
    let tmp = tempfile::tempdir().unwrap();
    let group = scalar_dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_layout", &parse_window("2..5").unwrap())
        .unwrap();

    assert!(
        !plan.degrades_to_full_download,
        "a fixed-stride facet is windowable"
    );
    assert_eq!(
        plan.byte_ranges,
        vec![(8, 20)],
        "records 2..5 of a 4-byte scalar are bytes 8..20; reading the \
         leading 7 as a dimension would give 64..160 instead"
    );
    assert_eq!(
        plan.prerequisite_bytes, 0,
        "a fixed stride is known from the extension — nothing to read first"
    );
}

/// The other half of the same defect. When the leading bytes do *not*
/// pass the dimension sanity check, the xvec path returns no mapping at
/// all and a perfectly windowable facet degrades to a whole-facet
/// fetch — which the fallback gate then refuses.
#[test]
fn a_scalar_window_does_not_degrade_when_its_bytes_look_like_no_dimension() {
    let tmp = tempfile::tempdir().unwrap();
    let group = scalar_dataset(tmp.path());
    let view = group.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("10..60").unwrap())
        .unwrap();

    assert!(
        !plan.degrades_to_full_download,
        "the leading bytes are data, not a corrupt header"
    );
    assert_eq!(plan.byte_ranges, vec![(10, 60)], "one byte per record");
}

/// A scalar window is refusable on the same terms as any other, so the
/// mapping fix does not quietly widen what a caller consents to.
#[test]
fn a_scalar_window_prefetches_without_whole_facet_consent() {
    let tmp = tempfile::tempdir().unwrap();
    let group = scalar_dataset(tmp.path());
    let view = group.profile("default").unwrap();

    view.prefetch(
        "metadata_content",
        &parse_window("10..60").unwrap(),
        WholeFacetFallback::Refuse,
    )
    .expect("a mapped window never needs whole-facet consent");
    view.prefetch(
        "metadata_layout",
        &parse_window("2..5").unwrap(),
        WholeFacetFallback::Refuse,
    )
    .expect("a mapped window never needs whole-facet consent");
}
