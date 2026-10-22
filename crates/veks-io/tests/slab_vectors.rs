// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A slab of vectors reports the element type its writer declared
//! (`docs/design/srd-crate-boundaries.md` acceptance test 5, the slab
//! case). A slab that declares none reads as bytes.

#![cfg(feature = "slab")]

use std::path::Path;

use veks_apicore::metadata_schema::{MetadataSchema, SCHEMA_NAMESPACE};
use veks_io::sink::{SinkConfig, open_sink};
use veks_io::slab::reader::SlabReader;
use veks_io::{ElementType, SourceMeta, VecFormat, VecSource};

fn config(dimension: u32, element_type: Option<ElementType>) -> SinkConfig {
    SinkConfig {
        dimension,
        element_type,
        slab_page_size: None,
        slab_namespace: 0,
        schema_sidecar: None,
        max_shard_bytes: None,
    }
}

fn write_slab(path: &Path, config: &SinkConfig, records: &[Vec<u8>]) {
    let mut sink = open_sink(path, VecFormat::Slab, config).unwrap();
    for (o, r) in records.iter().enumerate() {
        sink.write_record(o as i64, r);
    }
    sink.finish().unwrap();
}

/// Write `records` straight through slabtastic, with an optional raw
/// `:schema` record — the shape of a slab written before the vector
/// descriptor existed.
fn write_raw_slab(path: &Path, records: &[Vec<u8>], sidecar: Option<&[u8]>) {
    let mut w = slabtastic::SlabWriter::new(path, slabtastic::WriterConfig::default()).unwrap();
    for r in records {
        w.add_record(r).unwrap();
    }
    if let Some(bytes) = sidecar {
        w.start_namespace(SCHEMA_NAMESPACE).unwrap();
        w.add_record(bytes).unwrap();
    }
    w.finish().unwrap();
}

fn shape(source: &dyn VecSource) -> (u32, usize, Option<ElementType>) {
    (source.dimension(), source.element_size(), source.element_type())
}

fn probed(meta: &SourceMeta) -> (u32, usize, Option<ElementType>) {
    (meta.dimension, meta.element_size, meta.element_type)
}

fn all(mut source: Box<dyn VecSource>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(r) = source.next_record() {
        out.push(r);
    }
    out
}

/// **A slab of u8 vectors written by the writer reports a 1-byte
/// element and `U8`** — not 4 bytes whatever the data.
#[test]
fn a_slab_of_u8_vectors_reports_one_byte_and_u8() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("codes.slab");
    write_slab(&path, &config(3, Some(ElementType::U8)), &[vec![1, 2, 3], vec![4, 5, 6]]);

    let source = veks_io::open_format(&path, VecFormat::Slab).unwrap();
    assert_eq!(shape(source.as_ref()), (3, 1, Some(ElementType::U8)));
    assert_eq!(all(source), vec![vec![1, 2, 3], vec![4, 5, 6]]);
}

/// **An f32 slab reports `F32` and 4 bytes**, through the reader and
/// the probe alike.
#[test]
fn an_f32_slab_reports_f32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("emb.slab");
    let record: Vec<u8> = [1.0f32, -2.5].iter().flat_map(|v| v.to_le_bytes()).collect();
    write_slab(&path, &config(2, Some(ElementType::F32)), std::slice::from_ref(&record));

    let source = veks_io::open_format(&path, VecFormat::Slab).unwrap();
    assert_eq!(shape(source.as_ref()), (2, 4, Some(ElementType::F32)));
    assert_eq!(all(source), vec![record]);
    let meta = veks_io::probe_format(&path, VecFormat::Slab).unwrap();
    assert_eq!((probed(&meta), meta.record_count), ((2, 4, Some(ElementType::F32)), Some(1)));
}

/// **A slab that declares no type reads as bytes**: with no sidecar
/// (every slab written before the vector descriptor), with a
/// metadata sidecar of either era, or with one that is not JSON, the
/// type is `None`, the element 1 byte and the dimension the record's
/// length — never 4 bytes assumed.
#[test]
fn an_undeclared_slab_reads_as_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let records = vec![vec![0u8; 8], vec![1u8; 8]];
    let metadata = MetadataSchema::new("parquet:/x", vec![]).to_json_bytes().unwrap();
    let kindless = br#"{"version":1,"source":"parquet:/a","imported_at_epoch_secs":1,"record_count":null,"fields":[]}"#;
    for (name, sidecar) in [
        ("none", None),
        ("metadata", Some(metadata.as_slice())),
        ("kindless", Some(kindless.as_slice())),
        ("garbage", Some(b"not json".as_slice())),
    ] {
        let path = dir.path().join(format!("{name}.slab"));
        write_raw_slab(&path, &records, sidecar);
        let source = SlabReader::open(&path).unwrap();
        assert_eq!(shape(source.as_ref()), (8, 1, None), "{name}");
        assert_eq!(all(source), records, "{name}");
        assert_eq!(probed(&SlabReader::probe(&path).unwrap()), (8, 1, None), "{name}");
    }
    // The writer given no element type writes the same undeclared slab.
    let path = dir.path().join("written.slab");
    write_slab(&path, &config(8, None), &records);
    assert_eq!(shape(SlabReader::open(&path).unwrap().as_ref()), (8, 1, None));
}

/// **A vector sidecar that cannot be read is an error**, not a slab
/// of bytes: the declaration is there, and reading past it would be a
/// guess.
#[test]
fn an_unreadable_vector_sidecar_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.slab");
    write_raw_slab(&path, &[vec![0u8; 4]], Some(br#"{"version":1,"kind":"vector","element_type":"f128"}"#));
    let err = SlabReader::open(&path).err().unwrap();
    assert!(err.contains("unknown element type 'f128'"), "{err}");
    assert!(SlabReader::probe(&path).is_err());
}

/// **A record that is not whole elements of the declared type is an
/// error**, naming the type.
#[test]
fn a_record_not_whole_elements_of_its_type_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ragged.slab");
    write_slab(&path, &config(1, Some(ElementType::F64)), &[vec![0u8; 12]]);
    let err = SlabReader::open(&path).err().unwrap();
    assert!(err.contains("not a whole number of f64 elements"), "{err}");
}

/// **The type matrix**: a slab written at every element type reads
/// back at that type, width and dimension, bit for bit, through the
/// reader and the probe.
#[test]
fn every_element_type_round_trips_through_a_slab() {
    let dir = tempfile::tempdir().unwrap();
    for &ty in ElementType::ALL {
        let width = ty.byte_width();
        let dim = 3u32;
        let records: Vec<Vec<u8>> = (0..5u8)
            .map(|i| (0..dim as usize * width).map(|b| i.wrapping_mul(31).wrapping_add(b as u8)).collect())
            .collect();
        let path = dir.path().join(format!("{ty}.slab"));
        write_slab(&path, &config(dim, Some(ty)), &records);

        let source = veks_io::open_format(&path, VecFormat::Slab).unwrap();
        assert_eq!(shape(source.as_ref()), (dim, width, Some(ty)), "{ty}");
        assert_eq!(source.record_count(), Some(5), "{ty}");
        assert_eq!(all(source), records, "{ty}");
        assert_eq!(probed(&SlabReader::probe(&path).unwrap()), (dim, width, Some(ty)), "{ty}");
    }
}
