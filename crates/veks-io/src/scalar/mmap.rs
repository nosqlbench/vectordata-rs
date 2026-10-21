// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The mapped column reader under its earlier name: [`ScalarMmapReader`]
//! is [`ScalarReader`] over a local file.

use std::path::Path;

#[allow(deprecated)]
pub use crate::element::MmapError;
use crate::reader::{IoError, ScalarReader};
use crate::source::MappedFile;

/// [`ScalarReader`] over a [`MappedFile`], by the
/// name it had before the readers were generic over their source.
#[deprecated(note = "use `veks_io::ScalarReader::<T>::open_path`: the same reader")]
pub type ScalarMmapReader<T> = ScalarReader<T, MappedFile>;

// ── Type-specific constructors ─────────────────────────────────────────
//
// Each is `ScalarReader::<T>::open_path` with the type spelled in the
// name. They predate the element model.

macro_rules! deprecated_open {
    ($($t:ty, $name:ident, $ext:literal, $note:literal;)*) => {$(
        impl ScalarReader<$t, MappedFile> {
            #[doc = concat!("Open a `.", $ext, "` column; `ScalarReader::<", $ext, ">::open_path` with the type in the name.")]
            #[deprecated(since = "4.0.0", note = $note)]
            pub fn $name(path: &Path) -> Result<Self, IoError> {
                Self::open_path(path)
            }
        }
    )*};
}

deprecated_open! {
    u8, open_u8, "u8", "use `ScalarReader::<u8>::open_path`";
    i8, open_i8, "i8", "use `ScalarReader::<i8>::open_path`";
    u16, open_u16, "u16", "use `ScalarReader::<u16>::open_path`";
    i16, open_i16, "i16", "use `ScalarReader::<i16>::open_path`";
    u32, open_u32, "u32", "use `ScalarReader::<u32>::open_path`";
    i32, open_i32, "i32", "use `ScalarReader::<i32>::open_path`";
    u64, open_u64, "u64", "use `ScalarReader::<u64>::open_path`";
    i64, open_i64, "i64", "use `ScalarReader::<i64>::open_path`";
}
