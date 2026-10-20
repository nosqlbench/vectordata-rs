// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vector data format support.
//!
//! This module provides a unified interface for reading and writing vector data
//! in various formats (xvec, scalar, npy, parquet, slab). It is designed to be
//! used by multiple tools and follows a registry pattern similar to the Java
//! `VectorFileIO` approach.

// Re-export veks-io traits so downstream code can access them via veks-core.
pub use veks_io::{VecSource as IoVecSource, VecSink as IoVecSink, SourceMeta as IoSourceMeta};

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
pub mod portable_io;
pub mod xvec_dir_compiler;
pub mod convert;
pub mod reader;
pub mod writer;

use std::fmt;

use vectordata::typed_access::{Container, ElementType};
use std::path::Path;

/// A vector data format recognized by veks.
///
/// **Vector formats** (xvec family) store one record per vector as
/// `[dim:i32, data...]`. Extensions end in `vec` or `vecs`.
///
/// **Scalar formats** store one value per ordinal as a flat packed array
/// with no header. Extension is the bare type name (e.g., `.u8`, `.i32`).
///
/// **Container formats** (npy, parquet, slab, hdf5) use their own structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecFormat {
    // ── Container formats ──────────────────────────────────────────
    /// NumPy `.npy` — float32 arrays.
    Npy,
    /// Apache Parquet — columnar float vectors.
    Parquet,
    /// Page-oriented slab binary format.
    Slab,
    /// HDF5 container — `file.hdf5#dataset` notation.
    Hdf5,

    // ── Vector formats (xvec: [dim:i32, data...] per record) ──────
    /// xvec float32 (4 bytes/element). Extension: `.fvec`
    Fvec,
    /// xvec float64 (8 bytes/element). Extension: `.dvec`
    Dvec,
    /// xvec float16 / half (2 bytes/element). Extension: `.mvec`
    Mvec,
    /// xvec bfloat16 (2 bytes/element). Extension: `.bf16vec`
    Bf16vec,
    /// xvec uint8 (1 byte/element). Extension: `.bvec` or `.u8vec`
    Bvec,
    /// xvec int8 (1 byte/element). Extension: `.i8vec`
    I8vec,
    /// xvec int16 (2 bytes/element). Extension: `.svec` or `.i16vec`
    Svec,
    /// xvec uint16 (2 bytes/element). Extension: `.u16vec`
    U16vec,
    /// xvec int32 (4 bytes/element). Extension: `.ivec` or `.i32vec`
    Ivec,
    /// xvec uint32 (4 bytes/element). Extension: `.u32vec`
    U32vec,
    /// xvec int64 (8 bytes/element). Extension: `.i64vec`
    I64vec,
    /// xvec uint64 (8 bytes/element). Extension: `.u64vec`
    U64vec,

    // ── Variable-length vector formats (vvec) ──────────────────────
    // Same per-record layout as xvec but records may have different
    // dimensions. Random access requires an offset index (IDXFOR__).
    /// vvec float32 (4 bytes/element). Extension: `.fvvec` or `.f32vvec`
    Fvvec,
    /// vvec float64 (8 bytes/element). Extension: `.dvvec` or `.f64vvec`
    Dvvec,
    /// vvec float16 (2 bytes/element). Extension: `.mvvec` or `.f16vvec`
    Mvvec,
    /// vvec bfloat16 (2 bytes/element). Extension: `.bf16vvec`
    Bf16vvec,
    /// vvec uint8 (1 byte/element). Extension: `.bvvec` or `.u8vvec`
    Bvvec,
    /// vvec int8 (1 byte/element). Extension: `.i8vvec`
    I8vvec,
    /// vvec int16 (2 bytes/element). Extension: `.svvec` or `.i16vvec`
    Svvec,
    /// vvec uint16 (2 bytes/element). Extension: `.u16vvec`
    U16vvec,
    /// vvec int32 (4 bytes/element). Extension: `.ivvec` or `.i32vvec`
    Ivvec,
    /// vvec uint32 (4 bytes/element). Extension: `.u32vvec`
    U32vvec,
    /// vvec int64 (8 bytes/element). Extension: `.i64vvec`
    I64vvec,
    /// vvec uint64 (8 bytes/element). Extension: `.u64vvec`
    U64vvec,

    // ── Scalar formats (flat packed, no header) ───────────────────
    /// Scalar uint8 (1 byte/element). Extension: `.u8`
    ScalarU8,
    /// Scalar int8 (1 byte/element). Extension: `.i8`
    ScalarI8,
    /// Scalar uint16 (2 bytes/element). Extension: `.u16`
    ScalarU16,
    /// Scalar int16 (2 bytes/element). Extension: `.i16`
    ScalarI16,
    /// Scalar uint32 (4 bytes/element). Extension: `.u32`
    ScalarU32,
    /// Scalar int32 (4 bytes/element). Extension: `.i32`
    ScalarI32,
    /// Scalar uint64 (8 bytes/element). Extension: `.u64`
    ScalarU64,
    /// Scalar int64 (8 bytes/element). Extension: `.i64`
    ScalarI64,
    /// Scalar float16 (2 bytes/element). Extension: `.f16`
    ScalarF16,
    /// Scalar bfloat16 (2 bytes/element). Extension: `.bf16`
    ScalarBf16,
    /// Scalar float32 (4 bytes/element). Extension: `.f32`
    ScalarF32,
    /// Scalar float64 (8 bytes/element). Extension: `.f64`
    ScalarF64,
}

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

impl VecFormat {
    /// Human-readable name matching the file extension.
    pub fn name(self) -> &'static str {
        match self {
            Self::Npy => "npy",
            Self::Parquet => "parquet",
            Self::Slab => "slab",
            Self::Hdf5 => "hdf5",
            Self::Fvec => "fvec",
            Self::Dvec => "dvec",
            Self::Mvec => "mvec",
            Self::Bf16vec => "bf16vec",
            Self::Bvec => "bvec",
            Self::I8vec => "i8vec",
            Self::Svec => "svec",
            Self::U16vec => "u16vec",
            Self::Ivec => "ivec",
            Self::U32vec => "u32vec",
            Self::I64vec => "i64vec",
            Self::U64vec => "u64vec",
            Self::Fvvec => "fvvec",
            Self::Dvvec => "dvvec",
            Self::Mvvec => "mvvec",
            Self::Bf16vvec => "bf16vvec",
            Self::Bvvec => "bvvec",
            Self::I8vvec => "i8vvec",
            Self::Svvec => "svvec",
            Self::U16vvec => "u16vvec",
            Self::Ivvec => "ivvec",
            Self::U32vvec => "u32vvec",
            Self::I64vvec => "i64vvec",
            Self::U64vvec => "u64vvec",
            Self::ScalarU8 => "u8",
            Self::ScalarI8 => "i8",
            Self::ScalarU16 => "u16",
            Self::ScalarI16 => "i16",
            Self::ScalarU32 => "u32",
            Self::ScalarI32 => "i32",
            Self::ScalarU64 => "u64",
            Self::ScalarI64 => "i64",
            Self::ScalarF16 => "f16",
            Self::ScalarBf16 => "bf16",
            Self::ScalarF32 => "f32",
            Self::ScalarF64 => "f64",
        }
    }

    /// Every format.
    pub const ALL: &[VecFormat] = &[
        Self::Npy, Self::Parquet, Self::Slab, Self::Hdf5,
        Self::Fvec, Self::Dvec, Self::Mvec, Self::Bf16vec, Self::Bvec, Self::I8vec, Self::Svec, Self::U16vec,
        Self::Ivec, Self::U32vec, Self::I64vec, Self::U64vec,
        Self::Fvvec, Self::Dvvec, Self::Mvvec, Self::Bf16vvec, Self::Bvvec, Self::I8vvec, Self::Svvec, Self::U16vvec,
        Self::Ivvec, Self::U32vvec, Self::I64vvec, Self::U64vvec,
        Self::ScalarU8, Self::ScalarI8, Self::ScalarU16, Self::ScalarI16,
        Self::ScalarU32, Self::ScalarI32, Self::ScalarU64, Self::ScalarI64,
        Self::ScalarF16, Self::ScalarBf16, Self::ScalarF32, Self::ScalarF64,
    ];

    /// The element type a vector, vvec or scalar format holds; `None`
    /// for a container, which describes its own.
    pub fn element_type(self) -> Option<ElementType> {
        use ElementType as E;
        Some(match self {
            Self::Npy | Self::Parquet | Self::Slab | Self::Hdf5 => return None,
            Self::Fvec | Self::Fvvec | Self::ScalarF32 => E::F32,
            Self::Dvec | Self::Dvvec | Self::ScalarF64 => E::F64,
            Self::Mvec | Self::Mvvec | Self::ScalarF16 => E::F16,
            Self::Bf16vec | Self::Bf16vvec | Self::ScalarBf16 => E::BF16,
            Self::Bvec | Self::Bvvec | Self::ScalarU8 => E::U8,
            Self::I8vec | Self::I8vvec | Self::ScalarI8 => E::I8,
            Self::Svec | Self::Svvec | Self::ScalarI16 => E::I16,
            Self::U16vec | Self::U16vvec | Self::ScalarU16 => E::U16,
            Self::Ivec | Self::Ivvec | Self::ScalarI32 => E::I32,
            Self::U32vec | Self::U32vvec | Self::ScalarU32 => E::U32,
            Self::I64vec | Self::I64vvec | Self::ScalarI64 => E::I64,
            Self::U64vec | Self::U64vvec | Self::ScalarU64 => E::U64,
        })
    }

    /// The layout of a vector, vvec or scalar format; `None` for a
    /// container.
    pub fn container(self) -> Option<Container> {
        Some(match self {
            Self::Npy | Self::Parquet | Self::Slab | Self::Hdf5 => return None,
            Self::Fvec | Self::Dvec | Self::Mvec | Self::Bf16vec | Self::Bvec | Self::I8vec | Self::Svec
            | Self::U16vec | Self::Ivec | Self::U32vec | Self::I64vec | Self::U64vec => Container::Vec,
            Self::Fvvec | Self::Dvvec | Self::Mvvec | Self::Bf16vvec | Self::Bvvec | Self::I8vvec | Self::Svvec
            | Self::U16vvec | Self::Ivvec | Self::U32vvec | Self::I64vvec | Self::U64vvec => Container::Vvec,
            Self::ScalarU8 | Self::ScalarI8 | Self::ScalarU16 | Self::ScalarI16 | Self::ScalarU32
            | Self::ScalarI32 | Self::ScalarU64 | Self::ScalarI64 | Self::ScalarF16 | Self::ScalarBf16
            | Self::ScalarF32 | Self::ScalarF64 => Container::Scalar,
        })
    }

    /// The format that holds `element_type` in `container`.
    pub fn of(container: Container, element_type: ElementType) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|f| f.container() == Some(container) && f.element_type() == Some(element_type))
    }

    /// True for the xvec family (vector formats with `[dim:i32, data...]` header).
    /// Includes both uniform (`vec`) and variable-length (`vvec`) formats.
    pub fn is_xvec(self) -> bool {
        self.is_uniform_xvec() || self.is_vvec()
    }

    /// True for uniform xvec (all records have the same dimension).
    pub fn is_uniform_xvec(self) -> bool {
        self.container() == Some(Container::Vec)
    }

    /// True for variable-length vector formats (vvec family).
    pub fn is_vvec(self) -> bool {
        self.container() == Some(Container::Vvec)
    }

    /// True for scalar formats (flat packed, no header, one value per ordinal).
    pub fn is_scalar(self) -> bool {
        self.container() == Some(Container::Scalar)
    }

    /// True for integer element types (scalar or vector).
    pub fn is_integer(self) -> bool {
        self.element_type().is_some_and(ElementType::is_integer)
    }

    /// Bytes per element. Returns 0 for container formats.
    pub fn element_size(self) -> usize {
        self.element_type().map_or(0, ElementType::byte_width)
    }

    /// Preferred file extension for writing a new file of this format:
    /// the historical plural where there is one (`fvecs`, `ivvecs`), the
    /// typed plural otherwise (`bf16vecs`); scalars and containers have
    /// no plural convention. Reading accepts every spelling; see
    /// [`from_extension`](Self::from_extension).
    pub fn preferred_extension(self) -> &'static str {
        match (self.container(), self.element_type()) {
            (Some(c @ (Container::Vec | Container::Vvec)), Some(ty)) => {
                ty.legacy_extension(c).unwrap_or(ty.extension(c))
            }
            _ => self.name(),
        }
    }

    /// Normalize an extension string to the preferred canonical form.
    pub fn canonical_extension(ext: &str) -> Option<&'static str> {
        Self::from_extension(ext).map(|f| f.preferred_extension())
    }

    /// Detect format from a file extension string: the containers by
    /// name, everything else by the element model's one table.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "npy" => Some(Self::Npy),
            "parquet" => Some(Self::Parquet),
            "slab" => Some(Self::Slab),
            "hdf5" | "h5" => Some(Self::Hdf5),
            other => {
                let (ty, container) = ElementType::parse_extension(other)?;
                Self::of(container, ty)
            }
        }
    }

    /// Detect format from a file path. Handles HDF5 `#` notation.
    pub fn detect_from_path(path: &Path) -> Option<Self> {
        let s = path.to_string_lossy();
        if let Some(hash_pos) = s.rfind('#') {
            let file_part = &s[..hash_pos];
            let ext = file_part.rsplit('.').next()?;
            if ext == "hdf5" || ext == "h5" {
                return Some(Self::Hdf5);
            }
        }
        let ext = path.extension()?.to_str()?;
        Self::from_extension(ext)
    }

    /// Detect format by scanning a directory's files.
    pub fn detect_from_directory(dir: &Path) -> Option<Self> {
        let entries: Vec<_> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .collect();
        if entries.is_empty() { return None; }

        let mut has_npy = false;
        let mut has_parquet = false;
        let mut has_xvec = None;
        let mut has_slab = false;
        for entry in &entries {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(ext) = name.rsplit('.').next() {
                match ext {
                    "npy" => has_npy = true,
                    "parquet" => has_parquet = true,
                    "slab" => has_slab = true,
                    _ => {
                        if has_xvec.is_none() {
                            has_xvec = Self::from_extension(ext).filter(|f| f.is_xvec());
                        }
                    }
                }
            }
        }
        if has_npy { Some(Self::Npy) }
        else if has_parquet { Some(Self::Parquet) }
        else if let Some(xvec) = has_xvec { Some(xvec) }
        else if has_slab { Some(Self::Slab) }
        else { None }
    }

    /// Auto-detect format from either a file path or a directory.
    pub fn detect(path: &Path) -> Option<Self> {
        if path.is_dir() { Self::detect_from_directory(path) }
        else { Self::detect_from_path(path) }
    }

    /// Compute expected file size for a given record count and dimension.
    ///
    /// For xvec: `records × (4 + dimension × element_size)`.
    /// For scalar: `records × element_size`.
    pub fn expected_file_size(self, record_count: u64, dimension: u32) -> Option<u64> {
        if self.is_vvec() {
            None // Variable-length — cannot predict size from count + dim
        } else if self.is_uniform_xvec() {
            let stride = 4u64 + (dimension as u64) * (self.element_size() as u64);
            Some(record_count * stride)
        } else if self.is_scalar() {
            Some(record_count * self.element_size() as u64)
        } else {
            None
        }
    }

    /// Bytes one record of `dimension` occupies on disk, framing
    /// included — the inverse of [`Self::expected_file_size`], and
    /// what a shard-size cap is divided by.
    ///
    /// `None` where a record has no fixed size to state: a vvec
    /// record carries its own dimension, and npy/parquet/hdf5 are
    /// container formats with their own structure. Those are sized by
    /// sampling instead (see
    /// [`sample_spread`](vectordata::dataset::shard_sizing::sample_spread)).
    ///
    /// `Slab` is `None` too, and for a subtler reason: a slab's record
    /// payload is the width of whatever was written *into* it, which
    /// is a property of the source, not of the slab. See
    /// [`SinkConfig::record_bytes`](writer::SinkConfig::record_bytes),
    /// which knows both.
    pub fn record_bytes(self, dimension: u32) -> Option<u64> {
        if self.is_uniform_xvec() {
            Some(4 + (dimension as u64) * (self.element_size() as u64))
        } else if self.is_scalar() {
            Some(self.element_size() as u64)
        } else {
            None
        }
    }

    /// Human-readable data type name.
    pub fn data_type_name(self) -> &'static str {
        match self.element_type() {
            Some(ty) => ty.long_name(),
            None if self == Self::Slab => "bytes",
            None => "float",
        }
    }

    /// Whether this format can be used as an output/sink format.
    pub fn is_writable(self) -> bool {
        self.is_xvec() || self.is_scalar() || self == Self::Slab
    }
}

impl fmt::Display for VecFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
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

    /// **veks-core's `VecFormat` names every element type in every
    /// layout**, and its spellings are the model's.
    #[test]
    fn vec_format_covers_the_element_model() {
        for &ty in ElementType::ALL {
            for container in [Container::Vec, Container::Vvec, Container::Scalar] {
                let fmt = VecFormat::of(container, ty).unwrap_or_else(|| panic!("{ty} in {container:?}"));
                assert_eq!((fmt.element_type(), fmt.container()), (Some(ty), Some(container)));
                assert_eq!(fmt.element_size(), ty.byte_width());
                assert_eq!(VecFormat::from_extension(ty.extension(container)), Some(fmt));
                for alias in ty.aliases(container) {
                    assert_eq!(VecFormat::from_extension(alias), Some(fmt), "{alias}");
                }
                assert_eq!(VecFormat::from_extension(fmt.preferred_extension()), Some(fmt));
            }
        }
        assert_eq!(VecFormat::Fvec.preferred_extension(), "fvecs");
        assert_eq!(VecFormat::Ivvec.preferred_extension(), "ivvecs");
        assert_eq!(VecFormat::Bf16vec.preferred_extension(), "bf16vecs");
        assert_eq!(VecFormat::Bf16vec.data_type_name(), "bfloat16");
        assert_eq!(VecFormat::Slab.data_type_name(), "bytes");
        assert_eq!(VecFormat::from_extension("h5"), Some(VecFormat::Hdf5));
    }
}

/// The type matrix (SRD element-access §13.10): `preferred_vec_format`
/// over every facet and every element type, and every writer sink over
/// every vector and scalar format, each read back at its own type.
#[cfg(test)]
mod type_matrix {
    use std::path::Path;

    use veks_datatypes::dispatch_element;
    use veks_io::MmapReader;
    use veks_io::scalar::mmap::ScalarMmapReader;

    use super::*;
    use crate::formats::writer::mmap_xvec::MmapXvecWriter;
    use crate::formats::writer::{SinkConfig, VecSink, open_sharded_sink, open_sink};
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
            let reader = ScalarMmapReader::<T>::open(path).unwrap_or_else(|e| panic!("{fmt}: {e}"));
            assert_eq!(reader.count(), expected.len(), "{fmt}: count");
            for (i, rec) in expected.iter().enumerate() {
                let got = reader.get(i).unwrap_or_else(|| panic!("{fmt}: record {i}"));
                assert!(same_bits(&[got], rec), "{fmt}: record {i}");
            }
        } else {
            let reader = MmapReader::<T>::open(path).unwrap_or_else(|e| panic!("{fmt}: {e}"));
            assert_eq!((reader.dim(), reader.count()), (expected[0].len(), expected.len()), "{fmt}: shape");
            for (i, rec) in expected.iter().enumerate() {
                let got = reader.get(i).unwrap_or_else(|| panic!("{fmt}: record {i}"));
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
