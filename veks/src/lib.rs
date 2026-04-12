// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Veks — CLI toolkit for vector dataset preparation.
//!
//! Provides pipeline execution, format conversion, bulk downloads, and
//! analysis tools for large-scale vector datasets.

#![allow(dead_code)]

// Re-export foundation modules from veks-core so that code using
// `crate::term`, `crate::filters`, etc. continues to compile.
pub use veks_core::filters;
pub use veks_core::formats;
pub use veks_core::paths;
pub use veks_core::term;
pub use veks_core::ui;

// Re-export pipeline from veks-pipeline.
pub use veks_pipeline::pipeline;

// Local modules (remain in this crate).
pub mod catalog;
pub mod check;
pub mod cli;
pub mod datasets;
pub mod explore;
pub mod prepare;
pub mod publish;
