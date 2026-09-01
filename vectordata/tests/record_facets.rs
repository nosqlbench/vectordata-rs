// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Reading a slab facet by ordinal, and typing it with a codec.
//!
//! The codecs were already here — stage 1 (`anode`) and stage 2
//! (`anode_vernacular`) — and the container was not, so no metadata or
//! predicate facet could be read through this crate's reader API at
//! all. These cover the container, the currying that types it, and the
//! property that makes both work for a series: a shard is an ordinary
//! slab based at zero, and the facet ordinal is resolved before the
//! container ever sees it.
//!
//! See `docs/design/srd-multifile-facet-shards.md` (SH-96, SH-98) and
//! `docs/design/metadata-facets-and-layout-namespace.md`.

use vectordata::formats::anode::ANode;
use vectordata::formats::anode_vernacular::Vernacular;
use vectordata::formats::mnode::{MNode, MValue};
use vectordata::records::{Anode, Serde, Text};

/// A metadata slab of `count` MNode records, ids starting at `first`.
fn write_metadata_slab(path: &std::path::Path, first: i32, count: i32) {
    let mut w =
        slabtastic::SlabWriter::new(path, slabtastic::WriterConfig::default()).unwrap();
    for i in first..first + count {
        let mut node = MNode::new();
        node.fields.insert("id".to_string(), MValue::Int32(i));
        node.fields.insert("bucket".to_string(), MValue::Int32(i % 4));
        w.add_record(&node.to_bytes()).unwrap();
    }
    w.finish().unwrap();
}

/// The struct a caller might want its records as.
#[derive(Debug, serde::Deserialize, PartialEq)]
struct Row {
    id: i32,
    bucket: i32,
}

fn single_facet_dataset(dir: &std::path::Path, records: i32) {
    std::fs::create_dir_all(dir).unwrap();
    write_metadata_slab(&dir.join("metadata_content.slab"), 0, records);
    std::fs::write(dir.join("base.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: meta\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();
}

// ── the container ──────────────────────────────────────────────────

/// **A slab facet reads by ordinal.** Before this there was no path
/// from a declared metadata facet to its records — the vector readers
/// are built on fixed-width elements, and a slab record is neither.
#[test]
fn a_slab_facet_reads_records_by_ordinal() {
    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 12);

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    assert_eq!(facet.count().unwrap(), 12);
    // The raw bytes are the escape hatch beneath every codec, and they
    // carry the dialect leader that makes stage 1 self-describing.
    let bytes = facet.record_bytes(0).unwrap();
    assert_eq!(bytes[0], vectordata::formats::anode::DIALECT_MNODE);
    assert!(facet.record_bytes(12).is_err(), "one past the end");
}

/// **The dialect comes from the record, not from the facet.** A reader
/// that took it from the facet table would put record identity in two
/// places when the bytes already carry it.
#[test]
fn the_record_says_which_dialect_it_is() {
    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 4);

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();
    let nodes = facet.decode(Anode);

    for o in 0..4u64 {
        match nodes.get(o).unwrap() {
            ANode::MNode(n) => {
                assert_eq!(n.fields.get("id"), Some(&MValue::Int32(o as i32)));
            }
            other => panic!("ordinal {o} decoded as {other:?}"),
        }
    }
}

// ── currying ───────────────────────────────────────────────────────

/// **A codec applied to a facet is a typed reader** — and the untyped
/// level is not a separate path, just the codec that stops after stage
/// one.
#[test]
fn applying_a_codec_types_the_same_facet() {
    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 6);

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    // Stage 1: the node.
    let nodes = facet.decode(Anode);
    assert!(matches!(nodes.get(2).unwrap(), ANode::MNode(_)));

    // Stages 1+2: text in a named vernacular.
    let cql = facet.decode(Text(Vernacular::Cql));
    let rendered = cql.get(2).unwrap();
    assert!(rendered.contains('2'), "CQL fragment for id=2: {rendered}");

    // Stages 1+2+serde: the caller's own type, by ordinal.
    let rows = facet.decode(Serde::<Row>::new());
    assert_eq!(rows.get(2).unwrap(), Row { id: 2, bucket: 2 });
    assert_eq!(rows.get(5).unwrap(), Row { id: 5, bucket: 1 });

    // All three read the same bytes — one container, three types.
    assert_eq!(nodes.count().unwrap(), 6);
    assert_eq!(cql.count().unwrap(), 6);
    assert_eq!(rows.count().unwrap(), 6);
}

/// A codec named at runtime resolves to the same decoding a typed one
/// performs — a lookup in front of one implementation, not a second.
#[test]
fn a_codec_named_at_runtime_decodes_identically() {
    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 3);

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    let by_type = facet.decode(Text(Vernacular::Cql)).get(1).unwrap();
    let named = vectordata::records::codec_by_name("cql").expect("cql is a known codec");
    let by_name = named.decode(&facet.record_bytes(1).unwrap()).unwrap();
    assert_eq!(by_type, by_name);

    assert!(vectordata::records::codec_by_name("not-a-codec").is_none());
}

/// Iteration decodes lazily and in order.
#[test]
fn records_iterate_in_ordinal_order() {
    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 5);

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    let ids: Vec<i32> = facet
        .decode(Serde::<Row>::new())
        .iter()
        .map(|r| r.unwrap().id)
        .collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4]);
}

// ── a sharded slab facet ───────────────────────────────────────────

/// A metadata facet spread over three slabs, each based at zero.
fn sharded_facet_dataset(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    // Each shard is an ordinary slab whose own ordinals start at zero;
    // the ids are what distinguish them, not the ordinals (SH-96).
    write_metadata_slab(&dir.join("meta__0000.slab"), 0, 10);
    write_metadata_slab(&dir.join("meta__0001.slab"), 10, 10);
    write_metadata_slab(&dir.join("meta__0002.slab"), 20, 5);
    std::fs::write(dir.join("base.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: sharded-meta\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_content:\n      source: meta__NNNN.slab\n      shard_stride: 10\n      \
         shard_count: 3\n      record_count: 25\n",
    )
    .unwrap();
}

/// **A slab series reads as one facet** (SH-18, SH-96).
///
/// The shard model resolves a facet ordinal to a shard and a local
/// ordinal; the slab resolves the local one. Neither level knows about
/// the other, which is exactly why shards can be ordinary slabs based
/// at zero.
#[test]
fn a_slab_series_reads_as_one_facet() {
    let tmp = tempfile::tempdir().unwrap();
    sharded_facet_dataset(tmp.path());

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    assert_eq!(facet.count().unwrap(), 25, "the series total, not a shard's");

    let rows = facet.decode(Serde::<Row>::new());
    for o in 0..25u64 {
        assert_eq!(
            rows.get(o).unwrap().id,
            o as i32,
            "facet ordinal {o} must reach the record that belongs to it"
        );
    }
    // Across every seam, in both directions.
    for o in [9u64, 10, 19, 20, 24, 10, 9, 0] {
        assert_eq!(rows.get(o).unwrap().id, o as i32);
    }
    assert!(rows.get(25).is_err(), "one past the end of the series");
}

/// **Shards carry relative ordinals** (SH-96).
///
/// Each shard is a slab based at zero — its own ordinal 0 is a real
/// record — and the global base lives only in the shard map. The proof
/// is that a shard read directly answers at 0..n while the same records
/// answer at their facet ordinals through the series.
#[test]
fn each_shard_is_a_slab_based_at_zero() {
    let tmp = tempfile::tempdir().unwrap();
    sharded_facet_dataset(tmp.path());

    // Directly: shard 2's own ordinal 0 holds the record the facet
    // calls ordinal 20.
    let shard = slabtastic::SlabReader::open(tmp.path().join("meta__0002.slab")).unwrap();
    assert_eq!(shard.total_records(), 5);
    let direct = vectordata::formats::anode::decode(&shard.get(0).unwrap()).unwrap();

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let through_series = view
        .open_facet_records("metadata_content")
        .unwrap()
        .decode(Anode)
        .get(20)
        .unwrap();

    assert_eq!(direct, through_series, "local ordinal 0 is facet ordinal 20");
}

/// **A namespace is a facet of its own**, and the same shard resolution
/// applies to it.
#[test]
fn a_sibling_namespace_reads_through_the_same_containers() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path()).unwrap();

    // One content record plus a schema namespace, as the producers write.
    let path = tmp.path().join("metadata_content.slab");
    let mut w =
        slabtastic::SlabWriter::new(&path, slabtastic::WriterConfig::default()).unwrap();
    let mut node = MNode::new();
    node.fields.insert("id".to_string(), MValue::Int32(7));
    node.fields.insert("bucket".to_string(), MValue::Int32(3));
    w.add_record(&node.to_bytes()).unwrap();
    w.start_namespace("schema").unwrap();
    w.add_record(br#"{"kind":"metadata"}"#).unwrap();
    w.finish().unwrap();

    std::fs::write(tmp.path().join("base.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "name: ns\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    // The default namespace holds content.
    assert_eq!(facet.count().unwrap(), 1);
    assert_eq!(facet.decode(Serde::<Row>::new()).get(0).unwrap().id, 7);

    // The schema namespace is the same containers under a different
    // name — no special case per document.
    let schema = facet.namespace("schema");
    assert_eq!(schema.count().unwrap(), 1);
    assert_eq!(
        schema.record_bytes(0).unwrap().as_ref(),
        br#"{"kind":"metadata"}"#
    );
    assert!(schema.name().ends_with(":schema"));
}

/// **An absent namespace is a normal state, not a failure.**
///
/// The layout copy is optional and several producers never write one,
/// so a facet asked for a namespace it lacks reports nothing rather
/// than reporting a broken file. That is the separation
/// `dataset::layout` already makes: open the file first, so a real I/O
/// or format failure stays one, and let only the namespace probe
/// answer "absent".
#[test]
fn an_absent_namespace_reports_empty_rather_than_failing() {
    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 3);

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    let missing = facet.namespace("layout");
    assert_eq!(missing.count().unwrap(), 0, "no layout namespace was written");
    assert!(matches!(
        missing.record_bytes(0),
        Err(vectordata::records::RecordError::OutOfBounds(0))
    ));

    // The content namespace is unaffected.
    assert_eq!(facet.count().unwrap(), 3);
}

/// **An embedded layout namespace does not travel into a sharded
/// content facet** (SH-98).
///
/// The standalone `metadata_layout.slab` is authoritative and the
/// embedded copy is a convenience. A sharded facet omits it rather than
/// duplicating a schema across every shard or placing it arbitrarily in
/// shard 0000 — either of which would invent a rule about where a
/// schema lives that the unsharded case never needed. Asked for it, the
/// facet reports nothing, from every shard alike.
#[test]
fn a_sharded_content_facet_carries_no_embedded_layout() {
    let tmp = tempfile::tempdir().unwrap();
    sharded_facet_dataset(tmp.path());

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    assert_eq!(facet.count().unwrap(), 25, "content is there");
    assert_eq!(
        facet.namespace("layout").count().unwrap(),
        0,
        "no shard carries the embedded copy"
    );
}

// ── what counts as a data facet ────────────────────────────────────

/// **A metadata slab is precached like any other data facet.**
///
/// The gate asked "what element width does this have?" and skipped
/// anything without one. For every fixed-width format that is the same
/// question as "is this data?"; for a slab it is not, so precaching a
/// dataset fetched the vectors and silently left the metadata behind.
#[test]
fn precaching_a_dataset_visits_its_slab_facets() {
    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 40);

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    let mut visited: Vec<String> = Vec::new();
    view.prebuffer_all_with_progress(&mut |facet: &str, _p| {
        if !visited.iter().any(|s| s == facet) {
            visited.push(facet.to_string());
        }
    })
    .unwrap();
    visited.sort();

    assert!(
        visited.iter().any(|f| f == "metadata_content"),
        "the metadata facet must be precached, not skipped: {visited:?}"
    );
    assert!(visited.iter().any(|f| f == "base_vectors"));
}

/// The same holds for a **sharded** slab facet, which additionally has
/// to report its format at all: a series has no single source path, and
/// the manifest inferred the format from that path.
#[test]
fn a_sharded_slab_facet_reports_its_format_and_is_precached() {
    let tmp = tempfile::tempdir().unwrap();
    sharded_facet_dataset(tmp.path());

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    assert_eq!(
        view.facet_manifest()
            .get("metadata_content")
            .and_then(|d| d.source_type.clone())
            .as_deref(),
        Some("slab"),
        "every shard shares one format, so a series has one to report"
    );

    let mut visited: Vec<String> = Vec::new();
    view.prebuffer_all_with_progress(&mut |facet: &str, _p| {
        if !visited.iter().any(|s| s == facet) {
            visited.push(facet.to_string());
        }
    })
    .unwrap();
    assert!(visited.iter().any(|f| f == "metadata_content"), "{visited:?}");
}

/// **A windowed facet still reports its format.**
///
/// A window contains dots, so taking the extension by splitting on `.`
/// yields `20)` for `base.fvec[0..20)` and the facet reports no format.
/// Harmless while only a human read it; load-bearing once anything
/// gates on it — a windowed profile would have stopped precaching its
/// own base vectors.
#[test]
fn a_windowed_source_still_reports_its_format() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path();
    write_metadata_slab(&ds.join("metadata_content.slab"), 0, 40);
    std::fs::write(ds.join("base.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: w\nprofiles:\n  default:\n    base_vectors: base.fvec[0..1)\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    assert_eq!(
        view.facet_manifest()
            .get("base_vectors")
            .and_then(|d| d.source_type.clone())
            .as_deref(),
        Some("fvec"),
        "the window suffix must not swallow the extension"
    );
}

// ── shape, and signalling ──────────────────────────────────────────

/// **A facet's shape is answerable before opening it.**
///
/// Some facets hold element runs and some hold opaque records. That is
/// a property of the data, not a gap in the reader, so a caller
/// handling both branches on it rather than trying one path and
/// interpreting the failure.
#[test]
fn a_facet_reports_which_shape_it_holds() {
    use vectordata::dataset::facet::FacetShape;

    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 4);

    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    assert_eq!(view.facet_shape("base_vectors").unwrap(), FacetShape::Elements);
    assert_eq!(view.facet_shape("metadata_content").unwrap(), FacetShape::Records);

    // And branching on it reaches a working reader either way.
    for name in ["base_vectors", "metadata_content"] {
        match view.facet_shape(name).unwrap() {
            FacetShape::Elements => {
                assert!(view.facet(name).is_ok(), "{name} should open as vectors");
            }
            FacetShape::Records => {
                assert!(
                    view.open_facet_records(name).is_ok(),
                    "{name} should open as records"
                );
            }
        }
    }
}

/// **Opening a record facet as vectors names the reader that works.**
///
/// The old failure was "cannot infer element size from extension
/// '.slab'" — true, and a description of the symptom rather than the
/// situation. The facet is readable; the caller is at the wrong door,
/// and the error's job is to say which door.
#[test]
fn opening_a_record_facet_as_vectors_points_at_the_right_reader() {
    use vectordata::dataset::facet::FacetShape;

    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 4);
    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    // Structured, so a caller can branch rather than parse prose.
    match view.facet("metadata_content") {
        Err(vectordata::Error::WrongFacetShape {
            facet,
            shape,
            attempted,
            reader,
        }) => {
            assert_eq!(facet, "metadata_content");
            assert_eq!(shape, FacetShape::Records);
            assert_eq!(attempted, FacetShape::Elements);
            assert!(reader.contains("open_facet_records"), "{reader}");
        }
        Err(other) => panic!("expected a shape mismatch, got {other}"),
        Ok(_) => panic!("a record facet must not open as vectors"),
    }

    // The typed and element-width paths agree with it.
    let msg = view.facet_element_type("metadata_content").unwrap_err().to_string();
    assert!(msg.contains("open_facet_records"), "{msg}");
    assert!(!msg.contains("unknown element type"), "{msg}");
}

/// **And the mirror.** An element facet brought to the record reader
/// would otherwise fail as a slab parse error about a footer, which
/// says nothing about what happened.
#[test]
fn opening_an_element_facet_as_records_points_back() {
    let tmp = tempfile::tempdir().unwrap();
    single_facet_dataset(tmp.path(), 4);
    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    match view.open_facet_records("base_vectors") {
        Err(vectordata::records::RecordError::WrongShape { facet, reader }) => {
            assert_eq!(facet, "base_vectors");
            assert!(reader.contains("facet()"), "{reader}");
        }
        other => panic!("expected a shape mismatch, got {other:?}"),
    }
}

/// A sharded record facet reports its shape too — the format is a
/// property of the series, not of the one path a series does not have.
#[test]
fn a_sharded_record_facet_reports_its_shape() {
    use vectordata::dataset::facet::FacetShape;

    let tmp = tempfile::tempdir().unwrap();
    sharded_facet_dataset(tmp.path());
    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    assert_eq!(view.facet_shape("metadata_content").unwrap(), FacetShape::Records);
    assert_eq!(
        view.open_facet_records("metadata_content").unwrap().count().unwrap(),
        25
    );
}

// ── the planner's slab branch ──────────────────────────────────────

/// A facet of `records` MNodes in 4 KiB pages, so a window spans some
/// pages and not others.
fn paged_facet(dir: &std::path::Path, records: i32) {
    std::fs::create_dir_all(dir).unwrap();
    let cfg = slabtastic::WriterConfig::new(4096, 4096, 1 << 20, false).unwrap();
    let mut w = slabtastic::SlabWriter::new(dir.join("metadata_content.slab"), cfg).unwrap();
    for i in 0..records {
        let mut n = MNode::new();
        n.fields.insert("id".to_string(), MValue::Int32(i));
        n.fields.insert("pad".to_string(), MValue::Text("x".repeat(40)));
        w.add_record(&n.to_bytes()).unwrap();
    }
    w.finish().unwrap();
    std::fs::write(dir.join("base.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: paged\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();
}

/// **A slab window costs the pages it spans, not the facet.**
///
/// Before the planner learned about slabs, `map_in_file` gave up at
/// `infer_elem_size(".slab") == 0` and every window priced as a whole
/// facet download — the reader fetching a page at a time while the
/// planner said otherwise.
#[test]
fn a_slab_window_prices_at_the_pages_it_spans() {
    use vectordata::dataset::source::parse_window;

    let tmp = tempfile::tempdir().unwrap();
    paged_facet(tmp.path(), 2000);
    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let total = view.open_facet_storage("metadata_content").unwrap().total_size();
    assert!(total > 40_000, "the fixture must span many pages: {total}");

    let narrow = view
        .prefetch_plan("metadata_content", &parse_window("10..20").unwrap())
        .unwrap();
    assert!(
        !narrow.degrades_to_full_download,
        "a ten-record window must not price as the whole facet"
    );
    let narrow_bytes: u64 = narrow.byte_ranges.iter().map(|r| r.end - r.start).sum();
    assert!(
        narrow_bytes > 0 && narrow_bytes < total / 4,
        "a ten-record window cost {narrow_bytes} of {total}"
    );

    // The whole facet costs the whole facet — the mapping is not merely
    // returning something small.
    let whole = view
        .prefetch_plan("metadata_content", &parse_window("0..2000").unwrap())
        .unwrap();
    let whole_bytes: u64 = whole.byte_ranges.iter().map(|r| r.end - r.start).sum();
    assert!(
        whole_bytes > narrow_bytes * 4,
        "every record costs more than ten: {whole_bytes} vs {narrow_bytes}"
    );
    assert!(whole_bytes <= total);
}

/// A window running past the last record ends at the last page rather
/// than at one that does not exist.
#[test]
fn a_slab_window_past_the_end_stops_at_the_last_page() {
    use vectordata::dataset::source::parse_window;

    let tmp = tempfile::tempdir().unwrap();
    paged_facet(tmp.path(), 500);
    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let total = view.open_facet_storage("metadata_content").unwrap().total_size();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("490..100000").unwrap())
        .unwrap();
    let bytes: u64 = plan.byte_ranges.iter().map(|r| r.end - r.start).sum();
    assert!(bytes > 0, "the tail records are still fetchable");
    assert!(bytes <= total, "and the plan does not run past the file");
}

/// A window whose lower bound is past the end maps to nothing, which
/// degrades rather than inventing a range.
#[test]
fn a_slab_window_starting_past_the_end_maps_to_nothing() {
    use vectordata::dataset::source::parse_window;

    let tmp = tempfile::tempdir().unwrap();
    paged_facet(tmp.path(), 100);
    let g = vectordata::TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();

    let plan = view
        .prefetch_plan("metadata_content", &parse_window("500..600").unwrap())
        .unwrap();
    assert!(
        plan.degrades_to_full_download,
        "an unmappable window degrades rather than fabricating a range"
    );
}

// ── sizing a shard from a variable-length facet ────────────────────

use vectordata::dataset::shard_sizing::{
    DEFAULT_MAX_SHARD_BYTES, DEFAULT_SAMPLE_RECORDS, RecordSize, plan,
};

/// A slab whose record `i` is `pad_bytes(i)` long, so the mean is
/// something a test can state rather than guess.
fn variable_slab(path: &std::path::Path, count: i32, pad_bytes: impl Fn(i32) -> usize) {
    let cfg = slabtastic::WriterConfig::new(4096, 4096, 1 << 20, false).unwrap();
    let mut w = slabtastic::SlabWriter::new(path, cfg).unwrap();
    for i in 0..count {
        let mut node = MNode::new();
        node.fields.insert("id".to_string(), MValue::Int32(i));
        node.fields
            .insert("pad".to_string(), MValue::Text("x".repeat(pad_bytes(i))));
        w.add_record(&node.to_bytes()).unwrap();
    }
    w.finish().unwrap();
}

fn open_facet(dir: &std::path::Path) -> vectordata::records::RecordFacet {
    let g = vectordata::TestDataGroup::load(dir.to_str().unwrap()).unwrap();
    g.profile("default").unwrap().open_facet_records("metadata_content").unwrap()
}

/// **A slab facet measures its own record size.** There is no stride
/// to divide a file-size cap by, so the facet is asked what its
/// records weigh and the answer carries a margin.
#[test]
fn a_slab_facet_samples_its_record_size() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path()).unwrap();
    variable_slab(&tmp.path().join("metadata_content.slab"), 2000, |_| 100);
    std::fs::write(tmp.path().join("base.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "name: meta\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();

    let facet = open_facet(tmp.path());
    let basis = facet.sample_record_size(DEFAULT_SAMPLE_RECORDS).unwrap().unwrap();

    let RecordSize::Sampled { mean, max, sampled } = basis else {
        panic!("a slab has no fixed record size: {basis:?}")
    };
    assert_eq!(sampled, DEFAULT_SAMPLE_RECORDS, "capped at the target");
    // Every record here is the same size, so mean and max agree.
    assert_eq!(mean, max);
    assert!((120..200).contains(&mean), "a ~100-byte pad plus MNode framing: {mean}");

    // And it plans a stride that keeps a full shard under the cap.
    let p = plan(DEFAULT_MAX_SHARD_BYTES, basis).unwrap();
    assert_eq!(p.record_bytes, mean * 2, "the 2x margin");
    assert!(p.projected_bytes() <= DEFAULT_MAX_SHARD_BYTES);
    assert_eq!(p.stride, vectordata::dataset::shard_sizing::floor_to_decade(p.stride));
}

/// The sample spans the facet, so a slab whose records grow with
/// ordinal is measured near its true mean rather than its first page.
#[test]
fn a_growing_slab_is_measured_across_its_whole_range() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path()).unwrap();
    // Records 0..1000 are small; 1000..2000 are ten times larger.
    variable_slab(&tmp.path().join("metadata_content.slab"), 2000, |i| {
        if i < 1000 { 20 } else { 1020 }
    });
    std::fs::write(tmp.path().join("base.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "name: meta\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();

    let facet = open_facet(tmp.path());
    let basis = facet.sample_record_size(200).unwrap().unwrap();
    let RecordSize::Sampled { mean, max, .. } = basis else { panic!("sampled") };

    // The true mean is ~570. A prefix sample would have said ~60.
    assert!((480..=660).contains(&mean), "mean across the range: {mean}");
    assert!(max > 1000, "the large half was reached: {max}");
}

/// An empty facet has nothing to measure, and says so rather than
/// reporting a zero-byte record and an unbounded stride.
#[test]
fn an_empty_facet_yields_no_basis() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path()).unwrap();
    variable_slab(&tmp.path().join("metadata_content.slab"), 0, |_| 0);
    std::fs::write(tmp.path().join("base.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "name: meta\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();

    let facet = open_facet(tmp.path());
    assert!(facet.sample_record_size(DEFAULT_SAMPLE_RECORDS).unwrap().is_none());
}
