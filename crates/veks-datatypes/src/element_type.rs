// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The element types, and the one table that maps each
//! of them, in each container layout, to its file extensions.

use std::fmt;

/// The type of one element of a vector file.
///
/// The enum is non-exhaustive: a type added later must not break a
/// downstream match, so match with a wildcard arm or dispatch through
/// [`dispatch_element!`](crate::dispatch_element).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementType {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 binary16: 1 sign, 5 exponent, 10 fraction bits.
    F16,
    /// Brain float 16: 1 sign, 8 exponent, 7 fraction bits — the top
    /// half of an `f32`. Not [`F16`](Self::F16), and not `u16`.
    BF16,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
}

/// What kind of number an [`ElementType`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementKind {
    /// An unsigned integer.
    Unsigned,
    /// A two's-complement signed integer.
    Signed,
    /// A floating point number of the given format.
    Float(FloatFormat),
}

/// The encoding of a floating point element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatFormat {
    /// IEEE 754 (`f16`, `f32`, `f64`).
    Ieee,
    /// Brain float (`bf16`).
    Brain,
}

/// How elements are laid out in a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Container {
    /// A header-free column: one element after another, nothing else.
    Scalar,
    /// Uniform vectors: each record is a little-endian `i32` dimension
    /// followed by that many elements, and every record has the same
    /// dimension.
    Vec,
    /// Variable-length vectors: the same record layout as
    /// [`Vec`](Self::Vec), but each record may have its own dimension.
    Vvec,
}

/// One row of the extension table: a type, a layout, the spelling to
/// write, and the spellings to accept.
struct Spelling {
    ty: ElementType,
    container: Container,
    canonical: &'static str,
    /// The spelling the earliest files used, when there is one:
    /// `fvecs` for `f32vecs`, `ivecs` for `i32vecs`. Plural.
    legacy: Option<&'static str>,
    aliases: &'static [&'static str],
}

/// Every spelling this crate reads or writes. Canonical spellings are
/// the type name plus the layout (`f32`, `f32vecs`, `f32vvecs`); the
/// aliases are the singular forms and the historical letters
/// (`fvecs`, `ivecs`, `bvecs`, `svecs`, `mvecs`, `dvecs`).
const TABLE: &[Spelling] = &{
    use Container::*;
    use ElementType::*;
    macro_rules! rows {
        ($($ty:ident $name:literal $(legacy $l:literal)?;)*) => {
            [$(
                Spelling { ty: $ty, container: Scalar, canonical: $name, legacy: None, aliases: &[] },
                Spelling {
                    ty: $ty, container: Vec, canonical: concat!($name, "vecs"),
                    legacy: rows!(@legacy $($l)?),
                    aliases: rows!(@vec $name $(, $l)?),
                },
                Spelling {
                    ty: $ty, container: Vvec, canonical: concat!($name, "vvecs"),
                    legacy: rows!(@vlegacy $($l)?),
                    aliases: rows!(@vvec $name $(, $l)?),
                },
            )*]
        };
        (@legacy) => { None };
        (@legacy $l:literal) => { Some(concat!($l, "vecs")) };
        (@vlegacy) => { None };
        (@vlegacy $l:literal) => { Some(concat!($l, "vvecs")) };
        (@vec $name:literal) => { &[concat!($name, "vec")] };
        (@vec $name:literal, $l:literal) => { &[concat!($name, "vec"), concat!($l, "vecs"), concat!($l, "vec")] };
        (@vvec $name:literal) => { &[concat!($name, "vvec")] };
        (@vvec $name:literal, $l:literal) => { &[concat!($name, "vvec"), concat!($l, "vvecs"), concat!($l, "vvec")] };
    }
    rows! {
        U8 "u8" legacy "b";
        I8 "i8";
        U16 "u16";
        I16 "i16" legacy "s";
        U32 "u32";
        I32 "i32" legacy "i";
        U64 "u64";
        I64 "i64";
        F16 "f16" legacy "m";
        BF16 "bf16";
        F32 "f32" legacy "f";
        F64 "f64" legacy "d";
    }
};

impl ElementType {
    /// Every element type, in width order within each kind.
    pub const ALL: &[ElementType] = &[
        Self::U8,
        Self::I8,
        Self::U16,
        Self::I16,
        Self::U32,
        Self::I32,
        Self::U64,
        Self::I64,
        Self::F16,
        Self::BF16,
        Self::F32,
        Self::F64,
    ];

    /// The width of one element in bytes.
    pub const fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// The kind of number: unsigned, signed, or a float format.
    pub const fn kind(self) -> ElementKind {
        match self {
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => ElementKind::Unsigned,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => ElementKind::Signed,
            Self::F16 | Self::F32 | Self::F64 => ElementKind::Float(FloatFormat::Ieee),
            Self::BF16 => ElementKind::Float(FloatFormat::Brain),
        }
    }

    /// Whether this is an integer type of either sign.
    pub const fn is_integer(self) -> bool {
        matches!(self.kind(), ElementKind::Unsigned | ElementKind::Signed)
    }

    /// Whether this is a floating point type of either format.
    pub const fn is_float(self) -> bool {
        matches!(self.kind(), ElementKind::Float(_))
    }

    /// The type's name as Rust spells it: `u8`, `bf16`, `f32`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// The type's name as data tools spell it: `uint8`, `float32`,
    /// `bfloat16`.
    pub const fn long_name(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::I8 => "int8",
            Self::U16 => "uint16",
            Self::I16 => "int16",
            Self::U32 => "uint32",
            Self::I32 => "int32",
            Self::U64 => "uint64",
            Self::I64 => "int64",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    /// The type with this [`name`](Self::name), case-insensitively.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|t| t.name().eq_ignore_ascii_case(name))
    }

    /// The spelling to write for this type in the given layout:
    /// `f32` for a column, `f32vecs` for uniform vectors, `f32vvecs`
    /// for variable-length ones. No leading dot.
    pub fn extension(self, container: Container) -> &'static str {
        row(self, container).canonical
    }

    /// The spelling the earliest files used for this type in the
    /// given layout, when there is one: `fvecs` for `f32`, `ivecs` for
    /// `i32`, `bvecs` for `u8`, `svecs` for `i16`, `mvecs` for `f16`,
    /// `dvecs` for `f64`, and their `vvecs` forms. Columns have none.
    /// A writer that keeps emitting what it always has reads this
    /// before [`extension`](Self::extension).
    pub fn legacy_extension(self, container: Container) -> Option<&'static str> {
        row(self, container).legacy
    }

    /// Every spelling read for this type in the given layout other
    /// than the canonical one: the singular form, and the historical
    /// letter in both forms. No leading dot.
    ///
    /// ```
    /// use veks_datatypes::{Container, ElementType};
    /// assert_eq!(ElementType::I16.extension(Container::Vec), "i16vecs");                 // what a new writer emits
    /// assert_eq!(ElementType::I16.legacy_extension(Container::Vec), Some("svecs"));      // what the pipeline keeps emitting
    /// assert_eq!(ElementType::I16.aliases(Container::Vec), ["i16vec", "svecs", "svec"]);
    /// assert_eq!(ElementType::I8.legacy_extension(Container::Vec), None);                // i8 never had a letter
    /// assert_eq!(ElementType::F16.aliases(Container::Vvec), ["f16vvec", "mvvecs", "mvvec"]);
    /// assert!(ElementType::U64.aliases(Container::Scalar).is_empty(), "a column has one spelling");
    /// ```
    pub fn aliases(self, container: Container) -> &'static [&'static str] {
        row(self, container).aliases
    }

    /// The type and layout a file extension names, case-insensitively,
    /// with or without a leading dot. Canonical spellings and every
    /// alias are read.
    ///
    /// ```
    /// use veks_datatypes::{Container, ElementType};
    /// assert_eq!(ElementType::parse_extension("u16vecs"), Some((ElementType::U16, Container::Vec)));
    /// assert_eq!(ElementType::parse_extension(".DVECS"), Some((ElementType::F64, Container::Vec)));   // legacy, any case
    /// assert_eq!(ElementType::parse_extension("ivvec"), Some((ElementType::I32, Container::Vvec)));   // singular, variable-length
    /// assert_eq!(ElementType::parse_extension("bf16"), Some((ElementType::BF16, Container::Scalar))); // a column
    /// assert_eq!(ElementType::parse_extension("csv"), None);
    /// ```
    pub fn parse_extension(ext: &str) -> Option<(ElementType, Container)> {
        let ext = ext.strip_prefix('.').unwrap_or(ext);
        TABLE
            .iter()
            .find(|row| {
                row.canonical.eq_ignore_ascii_case(ext)
                    || row.aliases.iter().any(|a| a.eq_ignore_ascii_case(ext))
            })
            .map(|row| (row.ty, row.container))
    }

    /// The type a file extension names, whatever its layout.
    pub fn from_extension(ext: &str) -> Option<ElementType> {
        Self::parse_extension(ext).map(|(ty, _)| ty)
    }
}

fn row(ty: ElementType, container: Container) -> &'static Spelling {
    TABLE
        .iter()
        .find(|row| row.ty == ty && row.container == container)
        .expect("every element type has a row per container")
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Container {
    /// Every layout.
    pub const ALL: &[Container] = &[Self::Scalar, Self::Vec, Self::Vvec];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_type_has_a_row_per_container_and_round_trips() {
        for &ty in ElementType::ALL {
            for &container in Container::ALL {
                let ext = ty.extension(container);
                assert_eq!(ElementType::parse_extension(ext), Some((ty, container)), "{ext}");
                assert_eq!(ElementType::parse_extension(&format!(".{ext}")), Some((ty, container)));
                assert_eq!(ElementType::parse_extension(&ext.to_uppercase()), Some((ty, container)));
                for alias in ty.aliases(container) {
                    assert_eq!(ElementType::parse_extension(alias), Some((ty, container)), "{alias}");
                }
                if let Some(legacy) = ty.legacy_extension(container) {
                    assert!(ty.aliases(container).contains(&legacy), "{legacy} is an alias");
                }
            }
        }
    }

    #[test]
    fn canonical_spellings_are_the_type_name_plus_the_layout() {
        assert_eq!(ElementType::F32.extension(Container::Scalar), "f32");
        assert_eq!(ElementType::F32.extension(Container::Vec), "f32vecs");
        assert_eq!(ElementType::F32.extension(Container::Vvec), "f32vvecs");
        assert_eq!(ElementType::BF16.extension(Container::Vec), "bf16vecs");
        assert_eq!(ElementType::BF16.extension(Container::Scalar), "bf16");
    }

    #[test]
    fn the_historical_letters_name_the_types_they_always_did() {
        use Container::*;
        use ElementType::*;
        for (ext, want) in [
            ("fvecs", (F32, Vec)),
            ("fvec", (F32, Vec)),
            ("ivecs", (I32, Vec)),
            ("ivvecs", (I32, Vvec)),
            ("bvecs", (U8, Vec)),
            ("bvvec", (U8, Vvec)),
            ("svecs", (I16, Vec)),
            ("mvecs", (F16, Vec)),
            ("mvec", (F16, Vec)),
            ("dvecs", (F64, Vec)),
            ("u8", (U8, Scalar)),
            ("i64", (I64, Scalar)),
            ("f16", (F16, Scalar)),
            ("f64", (F64, Scalar)),
        ] {
            assert_eq!(ElementType::parse_extension(ext), Some(want), "{ext}");
        }
        assert_eq!(ElementType::F32.legacy_extension(Vec), Some("fvecs"));
        assert_eq!(ElementType::F32.legacy_extension(Vvec), Some("fvvecs"));
        assert_eq!(ElementType::F32.legacy_extension(Scalar), None);
        assert_eq!(ElementType::BF16.legacy_extension(Vec), None);
    }

    #[test]
    fn bf16_is_not_f16_and_not_u16() {
        assert_eq!(ElementType::parse_extension("bf16vecs"), Some((ElementType::BF16, Container::Vec)));
        assert_eq!(ElementType::parse_extension("mvecs"), Some((ElementType::F16, Container::Vec)));
        assert_eq!(ElementType::parse_extension("u16vecs"), Some((ElementType::U16, Container::Vec)));
        assert_eq!(ElementType::BF16.kind(), ElementKind::Float(FloatFormat::Brain));
        assert_eq!(ElementType::F16.kind(), ElementKind::Float(FloatFormat::Ieee));
        assert_eq!(ElementType::U16.kind(), ElementKind::Unsigned);
    }

    #[test]
    fn unknown_and_ambiguous_spellings_are_none() {
        for ext in ["", "vec", "vecs", "xvecs", "f32vecss", "slab", "txt", "fvecsx"] {
            assert_eq!(ElementType::parse_extension(ext), None, "{ext:?}");
        }
    }

    #[test]
    fn no_spelling_is_claimed_twice() {
        let mut seen = std::collections::HashSet::new();
        for row in TABLE {
            assert!(seen.insert(row.canonical), "{} claimed twice", row.canonical);
            for alias in row.aliases {
                assert!(seen.insert(alias), "{alias} claimed twice");
            }
        }
    }

    #[test]
    fn names_round_trip_and_widths_match_the_kind() {
        for &ty in ElementType::ALL {
            assert_eq!(ElementType::from_name(ty.name()), Some(ty));
            assert_eq!(ElementType::from_name(&ty.name().to_uppercase()), Some(ty));
            assert_eq!(ty.to_string(), ty.name());
            assert_eq!(ty.is_integer(), !ty.is_float());
        }
        assert_eq!(ElementType::from_name("float"), None);
        assert_eq!(ElementType::BF16.long_name(), "bfloat16");
        assert_eq!(ElementType::U8.long_name(), "uint8");
        assert_eq!(ElementType::ALL.len(), TABLE.len() / Container::ALL.len());
    }
}
