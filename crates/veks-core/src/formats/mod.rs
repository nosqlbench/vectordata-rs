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


use vectordata::typed_access::{Container, ElementType};
use veks_io::VecFormat;

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
    use super::*;

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
}
