// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The element model.
//!
//! The model — [`ElementType`], [`Container`], the [`Element`] trait,
//! [`Value`], the extension table and the conversion table
//! ([`ElementType::conversion_to`]) — is `veks-datatypes`, re-exported
//! here and at the crate root; [`ElementTypeExt`] adds the path and URL
//! forms this crate reads extensions from.
//!
//! This module reads nothing itself. A facet is read at a type with
//! [`TestDataViewExt::facet_as`](crate::TestDataViewExt::facet_as) and
//! the per-facet `*_as` calls, a raw file with
//! [`io::open_vec_as`](crate::io::open_vec_as), both under the table
//! below.
//!
//! # Access modes
//!
//! What a file of one type may be read as is one table,
//! [`ElementType::conversion_to`]:
//!
//! - **Native**: open with the exact native type → zero-copy `&[T]` access
//! - **Widening**: open a file as a type that holds every value of its own → checked conversion, always succeeds
//! - **Cross-sign same width**: u8↔i8, u16↔i16, etc. → checked per-value, fails on overflow
//! - **Refused**: narrowing, a float as an integer, a lossy integer-to-float → rejected at open time
//!
//! # Examples
//!
//! ```rust,no_run
//! use vectordata::io::{open_vec_as, ScalarReader};
//! use vectordata::VectorReader;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Open with native type — zero-copy
//! let reader = ScalarReader::<u8>::open("metadata.u8")?;
//! let val: u8 = reader.as_slice().expect("a resident local column")[42];
//!
//! // Open with wider type — always succeeds
//! let reader = open_vec_as::<i32>("metadata.u8")?;
//! let val: i32 = reader.get(42)?[0];
//! # let _ = val; Ok(()) }
//! ```

use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════
// ElementType — the shared element model, plus its path forms
// ═══════════════════════════════════════════════════════════════════════

pub use veks_datatypes::{
    Container, Element, ElementConversion, ElementKind, ElementType, FloatFormat, Value, bf16, f16,
};

/// The path and URL forms of [`ElementType::from_extension`], and the
/// scalar-layout question, as this crate asks them.
///
/// Brought into scope with `use vectordata::ElementTypeExt` (or
/// `vectordata::typed_access::ElementTypeExt`).
pub trait ElementTypeExt: Sized {
    /// Infer the element type from the extension of a local path.
    ///
    /// Errors when the path has no extension or the extension is not
    /// one [`ElementType::from_extension`] recognises.
    fn from_path(path: impl AsRef<Path>) -> Result<Self, String>;

    /// Infer the element type from the extension of a URL's path
    /// (the text after the last `.`).
    ///
    /// Errors when that text is not a recognised extension.
    fn from_url(url: &url::Url) -> Result<Self, String>;

    /// Whether `path` names a scalar facet — raw packed values with no
    /// per-record header. See `crate::io::is_scalar_ext`, the single
    /// table both this and the URL form answer from.
    fn is_scalar_format(path: impl AsRef<Path>) -> bool;

    /// Whether a file of this type may be opened as a target type of
    /// `target_width` bytes — true unless that would narrow. Width
    /// alone no longer decides: [`ElementType::conversion_to`] does.
    #[deprecated(since = "4.0.0", note = "use `ElementType::conversion_to`, which answers by type rather than width")]
    fn can_open_as(self, target_width: usize) -> bool;
}

impl ElementTypeExt for ElementType {
    fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| format!("no extension: {}", path.display()))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}': {}", path.display()))
    }

    fn from_url(url: &url::Url) -> Result<Self, String> {
        let path = url.path();
        let ext = path.rsplit('.').next()
            .ok_or_else(|| format!("no extension in URL: {url}"))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}' in URL: {url}"))
    }

    fn is_scalar_format(path: impl AsRef<Path>) -> bool {
        let ext = path.as_ref().extension().and_then(|e| e.to_str()).unwrap_or("");
        crate::io::is_scalar_ext(ext)
    }

    fn can_open_as(self, target_width: usize) -> bool {
        target_width >= self.byte_width()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{IoError, ScalarReader, VectorReader, open_vec_as};
    use std::io::Write;

    fn make_tmp() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    fn src(path: &Path) -> &str {
        path.to_str().unwrap()
    }

    #[test]
    fn detect_scalar_types() {
        assert_eq!(ElementType::from_extension("u8"), Some(ElementType::U8));
        assert_eq!(ElementType::from_extension("i32"), Some(ElementType::I32));
        assert_eq!(ElementType::from_extension("u64"), Some(ElementType::U64));
    }

    #[test]
    fn detect_vector_types() {
        assert_eq!(ElementType::from_extension("ivec"), Some(ElementType::I32));
        assert_eq!(ElementType::from_extension("bvec"), Some(ElementType::U8));
        assert_eq!(ElementType::from_extension("fvec"), Some(ElementType::F32));
        assert_eq!(ElementType::from_extension("u16vec"), Some(ElementType::U16));
    }

    #[test]
    fn scalar_u8_native() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, [0u8, 42, 127, 255]).unwrap();
        let r = ScalarReader::<u8>::open(src(&path)).unwrap();
        assert_eq!(r.count(), 4);
        assert_eq!(r.dim(), 1);
        assert_eq!(r.conversion(), ElementConversion::Native);
        let s = r.as_slice().expect("a local column is mapped");
        assert_eq!(s[0], 0);
        assert_eq!(s[3], 255);
    }

    #[test]
    fn scalar_u8_as_i32_widening() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, [0u8, 42, 255]).unwrap();
        let r = open_vec_as::<i32>(src(&path)).unwrap();
        assert_ne!(r.conversion(), ElementConversion::Native);
        assert_eq!(r.get(0).unwrap()[0], 0i32);
        assert_eq!(r.get(2).unwrap()[0], 255);
    }

    #[test]
    fn scalar_u8_as_i8_overflow() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, [128u8]).unwrap();
        let r = open_vec_as::<i8>(src(&path)).unwrap();
        assert!(r.get(0).is_err());
    }

    #[test]
    fn narrowing_rejected() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&42i32.to_le_bytes()).unwrap();
        assert!(open_vec_as::<u8>(src(&path)).is_err());
        assert!(open_vec_as::<i8>(src(&path)).is_err());
    }

    #[test]
    fn ivec_i32_native() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.ivec");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&3i32.to_le_bytes()).unwrap();
        f.write_all(&10i32.to_le_bytes()).unwrap();
        f.write_all(&20i32.to_le_bytes()).unwrap();
        f.write_all(&30i32.to_le_bytes()).unwrap();
        drop(f);
        let r = open_vec_as::<i32>(src(&path)).unwrap();
        assert_eq!(r.count(), 1);
        assert_eq!(r.dim(), 3);
        assert_eq!(r.get(0).unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn out_of_bounds() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        let r = open_vec_as::<u8>(src(&path)).unwrap();
        assert!(matches!(r.get(3), Err(IoError::OutOfBounds(3))));
    }

    #[test]
    fn empty_scalar() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, []).unwrap();
        let r = open_vec_as::<u8>(src(&path)).unwrap();
        assert_eq!(r.count(), 0);
    }
}
