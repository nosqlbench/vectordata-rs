// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks` — the CLI toolkit that builds, checks and publishes vector
//! datasets.
//!
//! Most users want the binary (`cargo install veks`): `veks prepare
//! bootstrap` turns source vectors and metadata into a dataset directory
//! with a `dataset.yaml`, `veks run` executes its pipeline (ground-truth
//! KNN, predicates, filtered KNN, merkle trees, docs), `veks check`
//! verifies it, and `veks publish` puts it at an endpoint. Datasets are
//! read back with [`vectordata`](https://docs.rs/vectordata).
//!
//! The library exists so other binaries can embed the same CLI:
//!
//! - [`shell::bin_main`] — the whole `veks` command line, as the binary
//!   runs it.
//! - [`prepare`] — bootstrap, the dataset wizard, tagging, cleanup and
//!   cache maintenance.
//! - [`check`] — dataset conformance and publish-readiness checks.
//! - [`publish`] and [`catalog`] — publishing and catalog generation.
//! - [`pipeline`] — re-exported from
//!   [`veks-pipeline`](https://docs.rs/veks-pipeline), the step runner
//!   and every pipeline command.
//!
//! ## Features
//!
//! - `knnutils` (default) — the knn_utils parity personality; pure Rust.
//! - `blas-system`, `simsimd`, `faiss` — native backends for the KNN
//!   engines, all opt-in (see `veks-pipeline`).
//! - `embed`, `embed-cuda` — in-process embedding.

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
// `pub mod explore` was migrated to `vectordata::explore`; the
// `vectordata explore` binary is the single entry point now.
pub mod prepare;
pub mod publish;
pub mod shell;
