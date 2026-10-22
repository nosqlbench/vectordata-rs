// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vector data format support.
//!
//! This module provides a unified interface for reading and writing vector data
//! in various formats (xvec, scalar, npy, parquet, slab). It is designed to be
//! used by multiple tools and follows a registry pattern similar to the Java
//! `VectorFileIO` approach.


// Wire format codecs re-exported from vectordata.
#[allow(dead_code)]
pub use vectordata::formats::mnode;
#[allow(dead_code)]
pub use vectordata::formats::pnode;
#[allow(dead_code)]
pub use vectordata::formats::anode;
#[allow(dead_code)]
pub use vectordata::formats::anode_vernacular;

/// The facet spec is vectordata's, the single authority: what a facet
/// is called, which formats it may take, how a resource classifies.
pub use vectordata::dataset::facet::{FacetFormat, FacetShape, StandardFacet};
pub mod parquet_compiler;
pub mod passage_table;
pub mod parquet_vector_compiler;
pub mod xvec_dir_compiler;
pub mod reader;
pub mod writer;


use vectordata::typed_access::{Container, ElementType};
use veks_io::VecFormat;

/// Bytes a slab spends per record beyond its payload: one entry in
/// the page's offset array.
pub const SLAB_RECORD_FRAMING: u64 = 4;

/// The format a facet is written in when its source holds `element`:
/// vectors and distances keep the source's element type (an `.i8vecs`
/// source writes `.i8vecs`, a `bf16` one `.bf16vecs`); indices are
/// `i32` vectors; the metadata facets are slabs; the quantization
/// columns are the columns the facet spec names.
pub fn preferred_vec_format(facet: StandardFacet, element: ElementType) -> VecFormat {
    use StandardFacet as F;
    match facet {
        F::BaseVectors | F::QueryVectors | F::NeighborDistances
        | F::PrefilteredNeighborDistances | F::PostfilteredNeighborDistances => {
            VecFormat::of(Container::Vec, element).unwrap_or(VecFormat::Fvec)
        }
        F::NeighborIndices | F::PrefilteredNeighborIndices | F::PostfilteredNeighborIndices => VecFormat::Ivec,
        F::MetadataContent | F::MetadataPredicates | F::MetadataResults | F::MetadataLayout => VecFormat::Slab,
        F::QuantScale => VecFormat::ScalarF32,
        F::QuantZeroPoint | F::QuantSum => VecFormat::ScalarI32,
    }
}

/// The element type a container source (npy, parquet, hdf5) holds,
/// which its probe states only as a width: containers hold floats, and
/// one byte is a code.
pub fn element_type_of_width(bytes: usize) -> ElementType {
    match bytes {
        1 => ElementType::U8,
        2 => ElementType::F16,
        8 => ElementType::F64,
        _ => ElementType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The format a facet is written in is keyed by element type**,
    /// not by width: an i8 source writes `.i8vecs` and a u16 source
    /// `.u16vecs`. The retired `Facet::preferred_format(element_size)`
    /// chose by width — `xvec_for_element_size` gave `Bvec` for every
    /// 1-byte type and `Svec` for every 2-byte one.
    #[test]
    fn preferred_vec_format_is_keyed_by_type() {
        use vectordata::dataset::StandardFacet as F;
        assert_eq!(preferred_vec_format(F::BaseVectors, ElementType::I8), VecFormat::I8vec);
        assert_eq!(preferred_vec_format(F::BaseVectors, ElementType::U16), VecFormat::U16vec);
        assert_eq!(preferred_vec_format(F::QueryVectors, ElementType::BF16), VecFormat::Bf16vec);
        assert_eq!(preferred_vec_format(F::NeighborDistances, ElementType::F32), VecFormat::Fvec);
        assert_eq!(preferred_vec_format(F::NeighborIndices, ElementType::F32), VecFormat::Ivec);
        assert_eq!(preferred_vec_format(F::MetadataContent, ElementType::U8), VecFormat::Slab);
        assert_eq!(preferred_vec_format(F::QuantScale, ElementType::U8), VecFormat::ScalarF32);
        assert_eq!(preferred_vec_format(F::QuantSum, ElementType::U8), VecFormat::ScalarI32);
        assert_eq!(element_type_of_width(1), ElementType::U8);
        assert_eq!(element_type_of_width(2), ElementType::F16);
        assert_eq!(element_type_of_width(4), ElementType::F32);
        assert_eq!(element_type_of_width(8), ElementType::F64);
    }

    /// **The facet spec veks-core answers from is vectordata's**: the
    /// `metadata_indices` alias resolves, which the retired copy had
    /// lost.
    #[test]
    fn metadata_indices_resolves_in_veks_core() {
        assert_eq!(StandardFacet::from_alias("metadata_indices"), Some(StandardFacet::MetadataResults));
        assert_eq!(StandardFacet::from_key("quant_scale"), Some(StandardFacet::QuantScale));
    }
}

/// The type matrix (SRD element-access §13.10): `preferred_vec_format`
/// over every facet and every element type, and every writer sink over
/// every vector and scalar format, each read back at its own type.
#[cfg(test)]
mod type_matrix {
    use std::path::Path;

    use veks_datatypes::dispatch_element;
    use veks_io::{ScalarReader, VectorReader, XvecReader};

    use super::*;
    use crate::formats::writer::mmap_xvec::MmapXvecWriter;
    use crate::formats::writer::{SinkConfig, open_sharded_sink, open_sink};
    use veks_io::VecSink;
    use crate::test_support::{Samples, encode, rotated_records, same_bits};
    use vectordata::typed_access::Element;

    /// **`preferred_vec_format` answers every facet at every element
    /// type by the documented contract**: a base, query or distance
    /// facet keeps the element type it is given in uniform vectors; an
    /// index facet is i32 vectors whatever it is given; the metadata
    /// facets are slabs; the quantization columns are the scalar
    /// formats the facet spec names, and every answer is one the facet
    /// spec accepts for that facet — except a distance facet given an
    /// integer type, which no pipeline produces (distances are written
    /// at the base's float type) and which the contract keeps as given.
    #[test]
    fn preferred_vec_format_answers_every_facet_at_every_element_type() {
        use StandardFacet as F;
        for &facet in F::PREFERRED_ORDER {
            for &ty in ElementType::ALL {
                let got = preferred_vec_format(facet, ty);
                let expected = match facet {
                    F::BaseVectors | F::QueryVectors | F::NeighborDistances
                    | F::PrefilteredNeighborDistances | F::PostfilteredNeighborDistances => {
                        VecFormat::of(Container::Vec, ty).unwrap()
                    }
                    F::NeighborIndices | F::PrefilteredNeighborIndices | F::PostfilteredNeighborIndices => VecFormat::Ivec,
                    F::MetadataContent | F::MetadataPredicates | F::MetadataResults | F::MetadataLayout => VecFormat::Slab,
                    F::QuantScale => VecFormat::ScalarF32,
                    F::QuantZeroPoint | F::QuantSum => VecFormat::ScalarI32,
                };
                assert_eq!(got, expected, "{} at {ty}", facet.key());
                if got.container() == Some(Container::Vec) {
                    let index_facet = matches!(facet, F::NeighborIndices | F::PrefilteredNeighborIndices | F::PostfilteredNeighborIndices);
                    assert_eq!(got.element_type(), Some(if index_facet { ElementType::I32 } else { ty }), "{} at {ty}", facet.key());
                }
                // The format a writer chooses is one the facet spec admits
                // for the facet, so a written facet passes conformance.
                let integer_distance = ty.is_integer()
                    && matches!(facet, F::NeighborDistances | F::PrefilteredNeighborDistances | F::PostfilteredNeighborDistances);
                if !integer_distance {
                    let spec = FacetFormat::from_extension(got.preferred_extension())
                        .unwrap_or_else(|| panic!("{} at {ty}: .{} names no facet format", facet.key(), got.preferred_extension()));
                    assert!(facet.accepts_format(spec), "{} at {ty}: {spec:?} is not in {:?}", facet.key(), facet.formats());
                }
            }
        }
    }

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
            source_format: fmt,
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
    fn the_mmap_xvec_writer_writes_every_uniform_vector_format() {
        let tmp = tempfile::tempdir().unwrap();
        for fmt in element_formats().filter(|f| f.is_uniform_xvec()) {
            let ty = fmt.element_type().unwrap();
            dispatch_element!(ty, T => {
                let path = tmp.path().join(format!("mmap.{}", fmt.preferred_extension()));
                let expected = rotated_records::<T>(5);
                let writer = MmapXvecWriter::create(&path, expected[0].len() as u32, T::SIZE, expected.len() as u64).unwrap();
                // Out of order, so the ordinal is what places a record.
                for (o, rec) in expected.iter().enumerate().rev() {
                    writer.write_record_at(o as u64, &encode(rec));
                }
                writer.finish().unwrap();
                assert_reads_back::<T>(&path, fmt, &expected);
            });
        }
    }
}
