// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Foundation modules shared by the [`veks`](https://docs.rs/veks) CLI
//! and its [`veks-pipeline`](https://docs.rs/veks-pipeline) engine.
//!
//! The crate sits between the dataset-access library
//! [`vectordata`](https://docs.rs/vectordata) and the toolkit built on
//! it. It holds what both the CLI and the pipeline need but neither
//! should own:
//!
//! - [`formats`] — vector and record file formats: the [`formats::VecFormat`]
//!   taxonomy and extension detection, [`formats::reader`] and
//!   [`formats::writer`] for xvec, npy, parquet and slab sources and sinks,
//!   element-type conversion ([`formats::convert`]), and the parquet
//!   compilers that turn columnar metadata into MNode records.
//! - [`ui`] — the progress and logging abstraction every long-running
//!   command reports through: a [`ui::UiHandle`] fronting a pluggable
//!   [`ui::UiSink`] (ratatui, plain text, headless, or a capturing test
//!   sink).
//! - [`filters`] — which files in a dataset directory are content,
//!   infrastructure or excluded (re-exported from `vectordata`, which owns
//!   the rules).
//! - [`term`] and [`paths`] — terminal styling and path display helpers.
//! - [`legacy_sweep`] — removal of pre-normalization singular-extension
//!   xvec links.
//!
//! ```
//! use veks_core::formats::{VecFormat, convert::convert_elements};
//!
//! assert_eq!(VecFormat::from_extension("fvecs"), Some(VecFormat::Fvec));
//!
//! // Widen one f16 element (1.0) to f32, as `transform convert` does.
//! let f16_one = half::f16::from_f32(1.0).to_le_bytes();
//! let f32_bytes = convert_elements(&f16_one, 2, 4).unwrap();
//! assert_eq!(f32::from_le_bytes(f32_bytes.try_into().unwrap()), 1.0);
//! ```

#![allow(dead_code)]
#![warn(missing_docs)]

/// Unified filtering rules — re-exported from `vectordata` (the base crate),
/// where the single definition lives so the push engine shares the same rules.
pub use vectordata::filters;
pub mod formats;
pub mod legacy_sweep;
pub mod paths;
pub mod term;
pub mod ui;
