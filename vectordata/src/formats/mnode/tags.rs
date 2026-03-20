// Copyright (c) DataStax, Inc.
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
    Text = 0,
    Int = 1,
    Float = 2,
    Bool = 3,
    Bytes = 4,
    Null = 5,
    EnumStr = 6,
    EnumOrd = 7,
    List = 8,
    Map = 9,
    TextValidated = 10,
    Ascii = 11,
    Int32 = 12,
    Short = 13,
    Decimal = 14,
    Varint = 15,
    Float32 = 16,
    Half = 17,
    Millis = 18,
    Nanos = 19,
    Date = 20,
    Time = 21,
    DateTime = 22,
    UuidV1 = 23,
    UuidV7 = 24,
    Ulid = 25,
    Array = 26,
    Set = 27,
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
