// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The element model.
//!
//! The model — [`ElementType`], [`Element`], the extension
//! table and the conversion table — is `veks-datatypes`, re-exported
//! here so a program that reads through this crate names them from
//! one place.

pub use veks_datatypes::{
    Container, Element, ElementConversion, ElementKind, ElementType, FloatFormat, bf16, f16,
};

/// The error a mapped reader raised before the readers shared one
/// error type: now [`IoError`](crate::IoError), which every reader
/// returns.
#[deprecated(note = "use `veks_io::IoError`, which every reader returns")]
pub type MmapError = crate::reader::IoError;
