// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Re-exports data source types from the `dataset` crate.
//!
//! All source types (DSSource, DSWindow, DSInterval) and sugar parsing
//! are now owned by the `dataset` crate. This module re-exports them
//! for backwards compatibility within veks.

pub use vectordata::dataset::source::*;
