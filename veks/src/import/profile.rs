// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Re-exports profile types from the `dataset` crate.
//!
//! All profile types (DSProfileGroup, DSProfile, DSView) are now owned
//! by the `dataset` crate. This module re-exports them for backwards
//! compatibility within veks.

pub use vectordata::dataset::profile::*;
