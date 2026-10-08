// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Type tag definitions for MNode field values.
//!
//! Each tag is a single byte discriminant that precedes the value bytes in the
//! MNode wire format. Tag assignments are stable and match the Java
//! `datatools-vectordata` implementation.

/// Type tags for MNode field values.
///
/// Each variant is a single-byte discriminant preceding the value bytes in
/// the MNode wire format. Assignments are stable and match the Java
/// `datatools-vectordata` implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TypeTag {
    /// UTF-8 string: `u32` LE byte length, then the bytes.
    Text = 0,
    /// 64-bit signed integer, little-endian.
    Int = 1,
    /// 64-bit IEEE 754 float, little-endian.
    Float = 2,
    /// Boolean as one byte; any nonzero value decodes as `true`.
    Bool = 3,
    /// Opaque byte string: `u32` LE length, then the bytes.
    Bytes = 4,
    /// Null value; no payload bytes.
    Null = 5,
    /// Enum value as a length-prefixed UTF-8 label.
    EnumStr = 6,
    /// Enum value as an `i32` LE ordinal.
    EnumOrd = 7,
    /// Heterogeneous list: `u32` LE count, then that many tagged values.
    List = 8,
    /// Nested MNode: `u32` LE byte length, then the encoded sub-record.
    Map = 9,
    /// Length-prefixed UTF-8 string that was validated on write; decodes
    /// to the same value as `Text`.
    TextValidated = 10,
    /// ASCII-only string, length-prefixed like `Text`.
    Ascii = 11,
    /// 32-bit signed integer, little-endian.
    Int32 = 12,
    /// 16-bit signed integer, little-endian.
    Short = 13,
    /// Arbitrary-precision decimal: `i32` LE scale, `u32` LE length, then
    /// the unscaled-value bytes. Decodes to raw bytes; the scale is discarded.
    Decimal = 14,
    /// Arbitrary-precision integer: `u32` LE length, then its bytes.
    /// Decodes to raw bytes.
    Varint = 15,
    /// 32-bit IEEE 754 float, little-endian.
    Float32 = 16,
    /// IEEE 754 half-precision float stored as raw `u16` LE bits.
    Half = 17,
    /// Timestamp as `i64` LE milliseconds since the Unix epoch.
    Millis = 18,
    /// Timestamp as `i64` LE epoch seconds followed by an `i32` LE
    /// nanosecond adjustment.
    Nanos = 19,
    /// Calendar date as a length-prefixed ISO 8601 string.
    Date = 20,
    /// Time of day as a length-prefixed ISO 8601 string.
    Time = 21,
    /// Date-time as a length-prefixed ISO 8601 string.
    DateTime = 22,
    /// Version 1 (time-based) UUID as 16 raw bytes.
    UuidV1 = 23,
    /// Version 7 (Unix-epoch, sortable) UUID as 16 raw bytes.
    UuidV7 = 24,
    /// ULID as 16 raw bytes.
    Ulid = 25,
    /// Homogeneous array: one element-tag byte, `u32` LE count, then that
    /// many untagged values of the element type.
    Array = 26,
    /// Unordered set: `u32` LE count, then that many tagged values.
    Set = 27,
    /// Map of tagged keys to tagged values: `u32` LE entry count, then
    /// alternating tagged key and tagged value.
    TypedMap = 28,
}

impl TypeTag {
    /// Convert a raw byte to a TypeTag, returning `None` for unknown values
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Text),
            1 => Some(Self::Int),
            2 => Some(Self::Float),
            3 => Some(Self::Bool),
            4 => Some(Self::Bytes),
            5 => Some(Self::Null),
            6 => Some(Self::EnumStr),
            7 => Some(Self::EnumOrd),
            8 => Some(Self::List),
            9 => Some(Self::Map),
            10 => Some(Self::TextValidated),
            11 => Some(Self::Ascii),
            12 => Some(Self::Int32),
            13 => Some(Self::Short),
            14 => Some(Self::Decimal),
            15 => Some(Self::Varint),
            16 => Some(Self::Float32),
            17 => Some(Self::Half),
            18 => Some(Self::Millis),
            19 => Some(Self::Nanos),
            20 => Some(Self::Date),
            21 => Some(Self::Time),
            22 => Some(Self::DateTime),
            23 => Some(Self::UuidV1),
            24 => Some(Self::UuidV7),
            25 => Some(Self::Ulid),
            26 => Some(Self::Array),
            27 => Some(Self::Set),
            28 => Some(Self::TypedMap),
            _ => None,
        }
    }

    /// Human-readable name for this type tag
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Int => "int",
            Self::Float => "float",
            Self::Bool => "bool",
            Self::Bytes => "bytes",
            Self::Null => "null",
            Self::EnumStr => "enum_str",
            Self::EnumOrd => "enum_ord",
            Self::List => "list",
            Self::Map => "map",
            Self::TextValidated => "text_validated",
            Self::Ascii => "ascii",
            Self::Int32 => "int32",
            Self::Short => "short",
            Self::Decimal => "decimal",
            Self::Varint => "varint",
            Self::Float32 => "float32",
            Self::Half => "half",
            Self::Millis => "millis",
            Self::Nanos => "nanos",
            Self::Date => "date",
            Self::Time => "time",
            Self::DateTime => "datetime",
            Self::UuidV1 => "uuid_v1",
            Self::UuidV7 => "uuid_v7",
            Self::Ulid => "ulid",
            Self::Array => "array",
            Self::Set => "set",
            Self::TypedMap => "typed_map",
        }
    }
}

impl std::fmt::Display for TypeTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_all_tags() {
        for i in 0..=28u8 {
            let tag = TypeTag::from_u8(i).unwrap_or_else(|| panic!("tag {} should be valid", i));
            assert_eq!(tag as u8, i);
        }
    }

    #[test]
    fn test_invalid_tag() {
        assert!(TypeTag::from_u8(29).is_none());
        assert!(TypeTag::from_u8(255).is_none());
    }
}
