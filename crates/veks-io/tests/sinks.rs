// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The sinks over the type matrix (SRD element-access §13.10): every
//! writer sink over every vector and scalar format, one file and across
//! shards, and the positional writer over every uniform vector format,
//! each read back at its own type.

mod support;

use std::path::Path;

use support::samples::{Samples, encode, rotated_records, same_bits};
use veks_datatypes::dispatch_element;
use veks_io::sink::{SinkConfig, open_sharded_sink, open_sink};
use veks_io::xvec::positional::PositionalXvecWriter;
use veks_io::{Element, ScalarReader, VecFormat, VecSink, VectorReader, XvecReader};

/// Every format a sink can write whose records are a run of one
/// element type: the uniform vectors and the columns.
fn element_formats() -> impl Iterator<Item = VecFormat> {
    VecFormat::ALL.iter().copied().filter(|f| f.is_uniform_xvec() || f.is_scalar())
}

/// The records a test writes at `fmt`: five rotations of the
/// samples as vectors, or one sample per record as a column.
fn records_for<T: Samples>(fmt: VecFormat) -> Vec<Vec<T>> {
    if fmt.is_scalar() {
        T::samples().into_iter().map(|v| vec![v]).collect()
    } else {
        rotated_records::<T>(5)
    }
}

fn config(fmt: VecFormat, dim: u32) -> SinkConfig {
    SinkConfig {
        dimension: dim,
        element_type: fmt.element_type(),
        slab_page_size: None,
        slab_namespace: 0,
        schema_sidecar: None,
        max_shard_bytes: None,
    }
}

/// Read `path` back through veks-io's typed readers at `T` and
/// require every record, bit for bit, to be what was written.
fn assert_reads_back<T: Element>(path: &Path, fmt: VecFormat, expected: &[Vec<T>]) {
    if fmt.is_scalar() {
        let reader = ScalarReader::<T>::open_path(path).unwrap_or_else(|e| panic!("{fmt}: {e}"));
        assert_eq!(reader.count(), expected.len(), "{fmt}: count");
        for (i, rec) in expected.iter().enumerate() {
            let got = reader.get(i).unwrap_or_else(|e| panic!("{fmt}: record {i}: {e}"));
            assert!(same_bits(&got, rec), "{fmt}: record {i}");
        }
    } else {
        let reader = XvecReader::<T>::open_path(path).unwrap_or_else(|e| panic!("{fmt}: {e}"));
        assert_eq!((reader.dim(), reader.count()), (expected[0].len(), expected.len()), "{fmt}: shape");
        for (i, rec) in expected.iter().enumerate() {
            let got = reader.get(i).unwrap_or_else(|e| panic!("{fmt}: record {i}: {e}"));
            assert!(same_bits(&got, rec), "{fmt}: record {i}");
        }
    }
}

/// **`open_sink` writes every vector and scalar format**, and the
/// file reads back at the format's own type with every sample —
/// extremes, a subnormal, the infinities and a NaN — intact.
#[test]
fn open_sink_writes_every_vector_and_scalar_format_that_reads_back_at_its_type() {
    let tmp = tempfile::tempdir().unwrap();
    for fmt in element_formats() {
        let ty = fmt.element_type().unwrap();
        dispatch_element!(ty, T => {
            let path = tmp.path().join(format!("sink.{}", fmt.preferred_extension()));
            let expected = records_for::<T>(fmt);
            let mut sink = open_sink(&path, fmt, &config(fmt, expected[0].len() as u32)).unwrap();
            for (o, rec) in expected.iter().enumerate() {
                sink.write_record(o as i64, &encode(rec));
            }
            sink.finish().unwrap();
            assert_reads_back::<T>(&path, fmt, &expected);
        });
    }
}

/// **The sharded sink writes every vector and scalar format**: a
/// stride of two over five records lands three shards, each an
/// ordinary file of the format that reads back at its type, and
/// the shards in order are the records in order.
#[test]
fn the_sharded_sink_writes_every_vector_and_scalar_format_across_shards() {
    let tmp = tempfile::tempdir().unwrap();
    for fmt in element_formats() {
        let ty = fmt.element_type().unwrap();
        dispatch_element!(ty, T => {
            let path = tmp.path().join(format!("series.{}", fmt.preferred_extension()));
            let expected = records_for::<T>(fmt);
            let cfg = config(fmt, expected[0].len() as u32);
            let mut sink = open_sharded_sink(&path, fmt, &cfg, 2).unwrap();
            for (o, rec) in expected.iter().enumerate() {
                sink.write_record(o as i64, &encode(rec));
            }
            let outcome = sink.outcome().unwrap();
            assert_eq!(outcome.shard_count, Some(expected.len().div_ceil(2) as u32), "{fmt}");
            assert_eq!(outcome.files.len(), expected.len().div_ceil(2), "{fmt}");
            for (shard, file) in outcome.files.iter().enumerate() {
                let slice = &expected[shard * 2..((shard + 1) * 2).min(expected.len())];
                assert_reads_back::<T>(file, fmt, slice);
            }
        });
    }
}

/// **The mmap xvec writer writes every uniform vector format**: its
/// element width is the type's, and records written at their
/// ordinals read back at the type in order.
#[test]
fn the_positional_writer_writes_every_uniform_vector_format() {
    let tmp = tempfile::tempdir().unwrap();
    for fmt in element_formats().filter(|f| f.is_uniform_xvec()) {
        let ty = fmt.element_type().unwrap();
        dispatch_element!(ty, T => {
            let path = tmp.path().join(format!("mmap.{}", fmt.preferred_extension()));
            let expected = rotated_records::<T>(5);
            let writer = PositionalXvecWriter::create(&path, expected[0].len() as u32, T::SIZE, expected.len() as u64).unwrap();
            // Out of order, so the ordinal is what places a record.
            for (o, rec) in expected.iter().enumerate().rev() {
                writer.write_record_at(o as u64, &encode(rec));
            }
            writer.finish().unwrap();
            assert_reads_back::<T>(&path, fmt, &expected);
        });
    }
}
