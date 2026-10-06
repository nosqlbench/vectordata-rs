// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The pipeline engine behind [`veks`](https://docs.rs/veks): the
//! `dataset.yaml`-driven step runner and every command it can run.
//!
//! A dataset's `dataset.yaml` declares its build as a list of steps, each
//! naming a command (`compute knn`, `generate predicates`, `merkle
//! create`, …) and its options. This crate turns that into work:
//!
//! - [`pipeline::run_pipeline`] — the `veks run` entry point: loads the
//!   dataset, expands per-profile steps, orders them as a DAG
//!   ([`pipeline::dag`]), skips steps whose outputs are fresh against the
//!   progress log ([`pipeline::progress`]), and executes the rest.
//! - [`pipeline::command::CommandOp`] — the trait every command
//!   implements: its options (which are its CLI flags and its YAML keys
//!   alike), its documentation, the artifacts it reads and writes, and
//!   `execute`.
//! - [`pipeline::registry::CommandRegistry`] — command path → factory;
//!   [`with_builtins`](pipeline::registry::CommandRegistry::with_builtins)
//!   registers every command in [`pipeline::commands`].
//! - The KNN engines and their kernels: `compute knn` and
//!   `compute knn-stdarch` on the native SIMD kernels of
//!   [`veks-simd`](https://docs.rs/veks-simd) (through
//!   [`pipeline::simd_distance`]), `compute knn-blas` on sgemm
//!   ([`pipeline::sgemm`]: pure-Rust `gemm`, or the system BLAS with the
//!   `blas-system` feature), and `compute knn-faiss` with `faiss`.
//!
//! ```
//! use veks_pipeline::pipeline::registry::CommandRegistry;
//!
//! let registry = CommandRegistry::with_builtins();
//! // `compute knn` is the alias a `dataset.yaml` names; the command it
//! // resolves to reports its canonical path.
//! let op = (registry.get("compute knn").unwrap())();
//! assert_eq!(op.command_path(), "compute knn-metal");
//! ```
//!
//! ## Features
//!
//! None are on by default; the default build is pure Rust.
//!
//! - `knnutils` — the knn_utils / numpy parity personality
//!   (`*-knnutils` commands, MT19937 shuffles).
//! - `blas-system` — `--backend system` for the sgemm scans (unix, links
//!   the system BLAS).
//! - `simsimd` — `--backend simsimd` for the pairwise scans.
//! - `faiss` — `compute knn-faiss` and its verifiers.
//! - `embed`, `embed-cuda` — in-process embedding (`generate embed`).

#![allow(dead_code)]

pub mod pipeline;
