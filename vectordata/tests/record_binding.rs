// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Binding records to operation parameters.
//!
//! Numbered as the acceptance cases in
//! `docs/design/srd-record-binding-and-forms.md` §10.

use std::collections::HashMap;

use vectordata::binding::{BindType, Binder, Layout};
use vectordata::formats::mnode::{MNode, MValue, TypeTag};
#[allow(unused_imports)]
use vectordata::formats::mnode::vernacular as render;

fn slab(path: &std::path::Path, records: &[MNode]) {
    let cfg = slabtastic::WriterConfig::new(4096, 4096, 1 << 20, false).unwrap();
    let mut w = slabtastic::SlabWriter::new(path, cfg).unwrap();
    for r in records {
        w.add_record(&r.to_bytes()).unwrap();
    }
    w.finish().unwrap();
}

fn slab_with_forms(path: &std::path::Path, records: &[MNode], forms: &[&str]) {
    let cfg = slabtastic::WriterConfig::new(4096, 4096, 1 << 20, false).unwrap();
    let mut w = slabtastic::SlabWriter::new(path, cfg).unwrap();
    for r in records {
        w.add_record(&r.to_bytes()).unwrap();
    }
    w.start_namespace("forms").unwrap();
    for f in forms {
        w.add_record(f.as_bytes()).unwrap();
    }
    w.finish().unwrap();
}

fn row(id: i64, tag: &str, score: f64) -> MNode {
    let mut n = MNode::new();
    n.insert("id".into(), MValue::Int(id));
    n.insert("tag".into(), MValue::Text(tag.into()));
    n.insert("score".into(), MValue::Float(score));
    n
}

fn dataset(dir: &std::path::Path, build: impl Fn(&std::path::Path)) {
    std::fs::create_dir_all(dir).unwrap();
    build(&dir.join("metadata_content.slab"));
    std::fs::write(dir.join("b.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: b\nprofiles:\n  default:\n    base_vectors: b.fvec\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();
}

fn open(dir: &std::path::Path) -> vectordata::records::RecordFacet {
    let g = vectordata::TestDataGroup::load(dir.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    view.open_facet_records("metadata_content").unwrap()
}

// ── cases 5, 6: the binding contract ───────────────────────────────

/// **Case 5** — a bound record is names and typed values, in declared
/// order. **Case 6** — the schema is obtained once, before any record.
#[test]
fn a_bound_record_is_named_typed_values_in_declared_order() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path(), |p| slab(p, &[row(1, "a", 1.5), row(2, "b", 2.5)]));
    let facet = open(tmp.path());

    // Once, before any record.
    let layout = Layout::discover(&facet).unwrap();
    assert_eq!(layout.names(), ["id", "tag", "score"]);
    assert_eq!(
        layout.types(),
        [BindType::Int64, BindType::Text, BindType::Float64]
    );

    let binder = Binder::all(&layout);
    assert_eq!(binder.parameters(), ["id", "tag", "score"]);

    // Per record, into the caller's buffer.
    let mut out = Vec::new();
    let bytes = facet.record_bytes(1).unwrap();
    binder.bind(&bytes, &mut out).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].as_i64(), Some(2));
    assert_eq!(out[1].as_str(), Some("b"));
    assert_eq!(out[2].as_f64(), Some(2.5));
}

/// A template may name fields in any order, and bind order follows the
/// template rather than the record.
#[test]
fn bind_order_follows_the_template_not_the_record() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path(), |p| slab(p, &[row(7, "z", 9.0)]));
    let facet = open(tmp.path());
    let layout = Layout::discover(&facet).unwrap();

    let binder = Binder::select(&layout, &["score", "id"]).unwrap();
    assert_eq!(binder.parameters(), ["score", "id"]);

    let mut out = Vec::new();
    let bytes = facet.record_bytes(0).unwrap();
    binder.bind(&bytes, &mut out).unwrap();
    assert_eq!(out[0].as_f64(), Some(9.0));
    assert_eq!(out[1].as_i64(), Some(7));
}

/// A field the facet does not have is refused when the binder is
/// built, naming what the facet does have — not on the first cycle.
#[test]
fn an_unknown_field_is_refused_at_compile_time() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path(), |p| slab(p, &[row(1, "a", 1.0)]));
    let facet = open(tmp.path());
    let layout = Layout::discover(&facet).unwrap();

    let err = Binder::select(&layout, &["id", "nope"]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("nope"), "{msg}");
    assert!(msg.contains("id, tag, score"), "names what it does have: {msg}");
}

/// **Names are the metadata names**, and a runtime may override one for
/// substitution — applied at compile time, leaving the facet's own name
/// untouched.
#[test]
fn a_parameter_may_be_renamed_without_touching_the_field() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path(), |p| slab(p, &[row(3, "c", 3.5)]));
    let facet = open(tmp.path());
    let layout = Layout::discover(&facet).unwrap();

    let mut overrides = HashMap::new();
    overrides.insert("id".to_string(), "pk".to_string());
    let binder = Binder::all(&layout).with_overrides(&overrides);

    assert_eq!(binder.parameters(), ["pk", "tag", "score"]);
    // The facet still calls it `id`.
    assert_eq!(layout.names()[0], "id");

    let mut out = Vec::new();
    let bytes = facet.record_bytes(0).unwrap();
    binder.bind(&bytes, &mut out).unwrap();
    assert_eq!(out[0].as_i64(), Some(3), "the rename moved no data");
}

// ── cases 13, 14, 16: the type asymmetry (OT-A) ────────────────────

/// **Case 13** — a `Half` binds as a float, not as a smallint.
///
/// The rendering mapping calls it `smallint`, which is right in a
/// `CREATE TABLE` and would send the bit pattern as an integer in a
/// bind. Binding needs its own mapping, and this is the case that says
/// so.
#[test]
fn a_half_binds_as_a_float_not_an_integer() {
    assert_eq!(BindType::of_tag(TypeTag::Half), BindType::Float16);
    assert_ne!(BindType::of_tag(TypeTag::Half), BindType::Int16);
    // The rendering mapping still says smallint, and is still right
    // about DDL — the two answers coexist rather than one being wrong.
    //
    // The hazard is concrete: `MValue::Half` holds the binary16 *bit
    // pattern* as a u16. 0x3C00 is 1.0. A binder that took the
    // rendering type would send 15360 to a smallint column and nothing
    // downstream would notice.
    let mut n = MNode::new();
    n.insert("h".into(), MValue::Half(0x3C00));
    assert!(render::to_cql_schema(&n).contains("smallint"));
}

/// **Case 14** — a container's element type is not collapsed to text.
///
/// The tag alone does not determine it, and the honest answer is to
/// say so rather than guess. A caller preparing a statement against an
/// underdetermined type needs a form or a sample.
#[test]
fn container_element_types_are_undetermined_rather_than_guessed() {
    for tag in [TypeTag::List, TypeTag::Array] {
        assert_eq!(BindType::of_tag(tag), BindType::List(None));
        assert!(BindType::of_tag(tag).is_underdetermined());
    }
    assert_eq!(BindType::of_tag(TypeTag::Set), BindType::Set(None));
    assert_eq!(BindType::of_tag(TypeTag::Map), BindType::Map(None, None));
    assert!(BindType::of_tag(TypeTag::TypedMap).is_underdetermined());

    // A scalar is fully determined.
    assert!(!BindType::of_tag(TypeTag::Int).is_underdetermined());
    assert!(!BindType::of_tag(TypeTag::UuidV7).is_underdetermined());
}

/// **Case 16** — the mapping is exhaustive.
///
/// Not a property a test can prove — an unhandled tag fails to compile,
/// which is the real guarantee. What this pins is that every tag
/// currently defined has an answer, and that null carries no type of
/// its own.
#[test]
fn every_tag_has_a_bind_type() {
    let all = [
        TypeTag::Text, TypeTag::Int, TypeTag::Float, TypeTag::Bool, TypeTag::Bytes,
        TypeTag::Null, TypeTag::EnumStr, TypeTag::EnumOrd, TypeTag::List, TypeTag::Map,
        TypeTag::TextValidated, TypeTag::Ascii, TypeTag::Int32, TypeTag::Short,
        TypeTag::Decimal, TypeTag::Varint, TypeTag::Float32, TypeTag::Half,
        TypeTag::Millis, TypeTag::Nanos, TypeTag::Date, TypeTag::Time, TypeTag::DateTime,
        TypeTag::UuidV1, TypeTag::UuidV7, TypeTag::Ulid, TypeTag::Array, TypeTag::Set,
        TypeTag::TypedMap,
    ];
    assert_eq!(all.len(), 29, "the tag set is a cross-language contract");
    for tag in all {
        let _ = BindType::of_tag(tag);
    }
    // Null has no type of its own; the parameter's type comes from the
    // schema, not from the absence.
    assert_eq!(BindType::of_tag(TypeTag::Null), BindType::Null);
    // The temporals and identifiers stay distinguishable, which is what
    // the rendering mapping cannot preserve.
    assert_ne!(
        BindType::of_tag(TypeTag::UuidV1),
        BindType::of_tag(TypeTag::UuidV7)
    );
    assert_ne!(
        BindType::of_tag(TypeTag::Millis),
        BindType::of_tag(TypeTag::Nanos)
    );
}

// ── case 9: the allocation claim ───────────────────────────────────

/// **Case 9** — binding N records allocates nothing per field name.
///
/// Asserted by counting allocations rather than by reasoning about the
/// code, because the claim is the whole point of resolving names to
/// positions once and it would regress silently. A global allocator
/// shim counts every allocation the binding loop makes.
#[test]
fn binding_allocates_nothing_per_record() {
    use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
    use std::cell::Cell;

    // Counted **per thread**. The allocator is process-wide and the
    // suite runs in parallel, so a global counter would tally other
    // tests' allocations during the measured window and fail this one
    // for someone else's work.
    thread_local! {
        static COUNT: Cell<usize> = const { Cell::new(0) };
        static ARMED: Cell<bool> = const { Cell::new(false) };
    }

    struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
            // `try_with` because a thread tearing down has no locals
            // left, and counting is not worth a panic in an allocator.
            let _ = ARMED.try_with(|a| {
                if a.get() {
                    let _ = COUNT.try_with(|c| c.set(c.get() + 1));
                }
            });
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
            unsafe { System.dealloc(p, l) }
        }
    }

    #[global_allocator]
    static A: Counting = Counting;

    let tmp = tempfile::tempdir().unwrap();
    let rows: Vec<MNode> = (0..64).map(|i| row(i, "steady", i as f64)).collect();
    dataset(tmp.path(), |p| slab(p, &rows));
    let facet = open(tmp.path());

    let layout = Layout::discover(&facet).unwrap();
    let binder = Binder::select(&layout, &["id", "tag"]).unwrap();

    // Pull every record's bytes first, so the measured region contains
    // only the binding, not the container reads.
    let records: Vec<_> = (0..64u64)
        .map(|o| facet.record_bytes(o).unwrap().into_owned())
        .collect();

    let mut id_total = 0i64;
    let mut tag_len = 0usize;

    COUNT.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    for bytes in &records {
        binder
            .bind_each(bytes, |slot, f| match slot {
                0 => id_total += f.as_i64().unwrap_or(0),
                _ => tag_len += f.as_str().map(str::len).unwrap_or(0),
            })
            .unwrap();
    }
    ARMED.with(|a| a.set(false));
    let allocations = COUNT.with(|c| c.get());

    // The work really happened.
    assert_eq!(id_total, (0..64i64).sum::<i64>());
    assert_eq!(tag_len, 64 * "steady".len());

    assert_eq!(
        allocations, 0,
        "binding 64 records allocated {allocations} times; names are resolved \
         to positions once and values are read in place, so the per-record \
         path should allocate nothing"
    );
}
