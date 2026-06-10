// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Core utilities for the veks toolkit.
//!
//! This crate contains stable foundation modules shared across the veks
//! binary and pipeline crates: terminal helpers, file filtering rules,
//! UI abstractions, and vector format I/O.

#![allow(dead_code)]

/// Unified filtering rules — re-exported from `vectordata` (the base crate),
/// where the single definition lives so the push engine shares the same rules.
pub use vectordata::filters;
pub mod formats;
pub mod legacy_sweep;
pub mod paths;
pub mod term;
pub mod ui;
