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
//! - [`formats`] — [`formats::reader`] and [`formats::writer`] for xvec,
//!   npy, parquet and slab sources and sinks, the facet → format choice
//!   ([`formats::preferred_vec_format`]), and the parquet compilers that
//!   turn columnar metadata into MNode records. The format model itself
//!   (`VecFormat`, the streaming traits, element conversion) is veks-io's.
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
//! use veks_core::formats::preferred_vec_format;
//! use veks_io::{ElementType, VecFormat};
//! use vectordata::dataset::StandardFacet;
//!
//! // A bf16 base is written as bf16 vectors; ground truth is always i32.
//! assert_eq!(preferred_vec_format(StandardFacet::BaseVectors, ElementType::BF16), VecFormat::Bf16vec);
//! assert_eq!(preferred_vec_format(StandardFacet::NeighborIndices, ElementType::BF16), VecFormat::Ivec);
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

#[cfg(test)]
pub(crate) mod test_support;
