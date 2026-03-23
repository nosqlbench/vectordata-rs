// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Veks — CLI toolkit for vector dataset preparation.
//!
//! Provides pipeline execution, format conversion, bulk downloads, and
//! analysis tools for large-scale vector datasets.
//!
//! # Major modules
//!
//! - [`pipeline`] — DAG-based command execution driven by `dataset.yaml`,
//!   including all pipeline command implementations (import, convert, analyze,
//!   bulk download, compute, generate, etc.)
//! - [`formats`] — vector I/O with [`formats::VecFormat`] (xvec, npy, parquet, slab)
//! - [`catalog`] — dataset discovery and catalog index generation
//! - [`datasets`] — dataset inventory, caching, and exploration commands
//! - [`ui`] — UI-agnostic progress, logging, and TUI rendering
//! - [`check`] — pre-flight validation (pipeline, bucket, merkle, integrity)
//! - [`publish`] — S3 dataset publishing via `aws s3 sync`
//! - [`cli`] — shell completion support built on clap

#![allow(dead_code)]

pub mod catalog;
pub mod check;
pub mod cli;
pub mod datasets;
pub mod explore;
pub mod formats;
pub mod pipeline;
pub mod prepare;
pub mod publish;
pub mod term;
pub mod ui;


