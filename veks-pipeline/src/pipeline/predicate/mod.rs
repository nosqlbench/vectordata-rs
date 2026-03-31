// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Predicate module for predicated dataset generation.
//!
//! Provides types and utilities for generating and evaluating metadata
//! predicates over vector datasets, enabling hybrid search testing with
//! structured attribute filters.
//!
//! ## Sub-modules
//!
//! - [`attribute`] — Field types, attribute schema, and typed column storage.
//! - [`predicate`] — Predicate tree types and evaluation logic.
//! - [`generator`] — Random attribute and predicate generation.
//! - [`codec`] — Binary serialization for predicate trees.

#[allow(dead_code)]
pub mod attribute;
#[allow(dead_code)]
pub mod codec;
pub mod generator;
#[allow(dead_code)]
pub mod predicate;
