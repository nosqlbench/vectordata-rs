// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Core utilities for the veks toolkit.
//!
//! This crate contains stable foundation modules shared across the veks
//! binary and pipeline crates: terminal helpers, file filtering rules,
//! UI abstractions, and vector format I/O.

#![allow(dead_code)]

pub mod filters;
pub mod formats;
pub mod paths;
pub mod term;
pub mod ui;
