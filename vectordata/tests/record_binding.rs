// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Binding records to operation parameters.
//!
//! Numbered as the acceptance cases in
//! `docs/design/srd-record-binding-and-forms.md` §10.

use std::collections::HashMap;

use vectordata::binding::{BindType, Binder, Form, Layout, forms_of, form_by_name};
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

// ── cases 1–4: forms ───────────────────────────────────────────────

/// **Case 1 / case 10 — the gate.** A facet with no `forms` namespace
/// offers exactly one form and binds unchanged. That is every dataset
/// in existence; absence is not an empty set.
#[test]
fn a_facet_without_forms_offers_one_implicit_form() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path(), |p| slab(p, &[row(1, "a", 1.0)]));
    let facet = open(tmp.path());

    let forms = forms_of(&facet).unwrap();
    assert_eq!(forms.len(), 1, "one implicit form, not none");
    assert_eq!(forms[0].name, Form::IMPLICIT);

    // And it binds every field under its own name.
    let layout = Layout::discover(&facet).unwrap();
    let binder = forms[0].binder(&layout).unwrap();
    assert_eq!(binder.parameters(), ["id", "tag", "score"]);
}

/// **Case 2** — declared forms are enumerable by name, and each
/// compiles to its own binder.
#[test]
fn declared_forms_are_enumerable_and_compile_independently() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path(), |p| {
        slab_with_forms(
            p,
            &[row(1, "a", 1.0)],
            &[
                r#"{"name":"row","operation":"insert","fields":["id","tag","score"]}"#,
                r#"{"name":"key","operation":"get","fields":["id"],"parameters":{"id":"pk"}}"#,
            ],
        )
    });
    let facet = open(tmp.path());
    let layout = Layout::discover(&facet).unwrap();

    let forms = forms_of(&facet).unwrap();
    let names: Vec<&str> = forms.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["row", "key"]);

    let row_binder = form_by_name(&facet, "row").unwrap().binder(&layout).unwrap();
    assert_eq!(row_binder.parameters(), ["id", "tag", "score"]);

    // A second form of the same records, binding a subset under a
    // different parameter name.
    let key_binder = form_by_name(&facet, "key").unwrap().binder(&layout).unwrap();
    assert_eq!(key_binder.parameters(), ["pk"]);
    assert_eq!(key_binder.types(), [BindType::Int64]);

    let mut out = Vec::new();
    let bytes = facet.record_bytes(0).unwrap();
    key_binder.bind(&bytes, &mut out).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].as_i64(), Some(1));
}

/// **Case 3** — a form the facet does not offer is refused, naming the
/// ones it does.
#[test]
fn an_unknown_form_is_refused_naming_what_is_offered() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path(), |p| {
        slab_with_forms(p, &[row(1, "a", 1.0)], &[r#"{"name":"row"}"#])
    });
    let facet = open(tmp.path());

    let msg = form_by_name(&facet, "document").unwrap_err().to_string();
    assert!(msg.contains("document"), "{msg}");
    assert!(msg.contains("row"), "names what is offered: {msg}");
}

/// **Case 4** — a form carrying keys this build does not know is
/// preserved, not rejected. A writer recording a capability this build
/// lacks is recording, not misbehaving.
#[test]
fn an_unrecognised_form_key_is_preserved() {
    let tmp = tempfile::tempdir().unwrap();
    dataset(tmp.path(), |p| {
        slab_with_forms(
            p,
            &[row(1, "a", 1.0)],
            &[r#"{"name":"row","consistency":"quorum","ttl_seconds":600}"#],
        )
    });
    let facet = open(tmp.path());

    let form = form_by_name(&facet, "row").unwrap();
    assert_eq!(form.extra.get("consistency").and_then(|v| v.as_str()), Some("quorum"));
    assert_eq!(form.extra.get("ttl_seconds").and_then(|v| v.as_u64()), Some(600));
    // And it still compiles — an unknown key is not a broken form.
    let layout = Layout::discover(&facet).unwrap();
    assert_eq!(form.binder(&layout).unwrap().parameters(), ["id", "tag", "score"]);
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

// ── cases 11, 12: series and remote ────────────────────────────────

/// **Case 11** — forms and schema are read from the series, not from
/// shard 0. A sharded facet's layout is the facet's.
#[test]
fn a_sharded_facet_binds_across_its_shards() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path();
    std::fs::create_dir_all(ds).unwrap();
    for sh in 0..2i64 {
        slab(
            &ds.join(format!("meta__{sh:04}.slab")),
            &(0..5)
                .map(|i| row(sh * 5 + i, "x", (sh * 5 + i) as f64))
                .collect::<Vec<_>>(),
        );
    }
    std::fs::write(ds.join("b.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: sh\nprofiles:\n  default:\n    base_vectors: b.fvec\n    \
         metadata_content:\n      source: meta__NNNN.slab\n      shard_stride: 5\n      \
         shard_count: 2\n      record_count: 10\n",
    )
    .unwrap();

    let facet = open(ds);
    assert_eq!(facet.count().unwrap(), 10);

    // One implicit form for the whole series, not one per shard.
    assert_eq!(forms_of(&facet).unwrap().len(), 1);

    let layout = Layout::discover(&facet).unwrap();
    let binder = Binder::all(&layout);
    // Across the seam, in the facet's ordinals. The callback form is
    // the loop form: no buffer outlives a record.
    for o in [0u64, 4, 5, 9] {
        let bytes = facet.record_bytes(o).unwrap();
        let mut id = None;
        binder
            .bind_each(&bytes, |slot, f| if slot == 0 { id = f.as_i64() })
            .unwrap();
        assert_eq!(id, Some(o as i64), "facet ordinal {o}");
    }
}

/// **Case 12** — binding stays incremental. A record costs its page,
/// not the facet, so a generator touching a scattered fraction of a
/// large facet fetches a fraction of it.
#[test]
fn binding_a_remote_facet_does_not_download_it() {
    let tmp = tempfile::tempdir().unwrap();
    let ds = tmp.path();
    std::fs::create_dir_all(ds).unwrap();
    let rows: Vec<MNode> = (0..2000).map(|i| row(i, "x", i as f64)).collect();
    slab(&ds.join("metadata_content.slab"), &rows);
    std::fs::write(ds.join("b.fvec"), [4u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: r\nprofiles:\n  default:\n    base_vectors: b.fvec\n    \
         metadata_content: metadata_content.slab\n",
    )
    .unwrap();

    let g = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = g.profile("default").unwrap();
    let facet = view.open_facet_records("metadata_content").unwrap();

    let layout = Layout::discover(&facet).unwrap();
    let binder = Binder::select(&layout, &["id"]).unwrap();
    for o in [0u64, 900, 1999] {
        let bytes = facet.record_bytes(o).unwrap();
        let mut id = None;
        binder.bind_each(&bytes, |_, f| id = f.as_i64()).unwrap();
        assert_eq!(id, Some(o as i64));
    }
    // The local case proves the addressing; the remote incrementality
    // itself is pinned in `http_storage.rs`, against a served facet.
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
