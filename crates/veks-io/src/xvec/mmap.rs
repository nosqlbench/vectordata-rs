// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The mapped uniform reader under its earlier name: [`MmapReader`] is
//! [`XvecReader`] over a local file.

use std::path::Path;

#[allow(deprecated)]
pub use crate::element::MmapError;
use crate::reader::{IoError, XvecReader};
use crate::source::MappedFile;

/// [`XvecReader`] over a [`MappedFile`], by the name
/// it had before the readers were generic over their source.
#[deprecated(note = "use `veks_io::XvecReader::<T>::open_path`: the same reader")]
pub type MmapReader<T> = XvecReader<T, MappedFile>;

// ── Type-specific constructors ─────────────────────────────────────────
//
// Each is `XvecReader::<T>::open_path` with the type spelled in the
// name. They predate the element model.

macro_rules! deprecated_open {
    ($($t:ty, $name:ident, $ext:literal, $note:literal;)*) => {$(
        impl XvecReader<$t, MappedFile> {
            #[doc = concat!("Open a `.", $ext, "` file; `XvecReader::<", stringify!($t), ">::open_path` with the type in the name.")]
            #[deprecated(since = "4.0.0", note = $note)]
            pub fn $name(path: &Path) -> Result<Self, IoError> {
                Self::open_path(path)
            }
        }
    )*};
}

deprecated_open! {
    f32, open_fvec, "fvec", "use `XvecReader::<f32>::open_path`";
    i32, open_ivec, "ivec", "use `XvecReader::<i32>::open_path`";
    half::f16, open_mvec, "mvec", "use `XvecReader::<f16>::open_path`";
    f64, open_dvec, "dvec", "use `XvecReader::<f64>::open_path`";
    u8, open_bvec, "bvec", "use `XvecReader::<u8>::open_path`";
    i16, open_svec, "svec", "use `XvecReader::<i16>::open_path`";
}
