// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dataset catalog infrastructure.
//!
//! Resolution and source management are provided by [`vectordata::catalog`]
//! and re-exported here. Catalog generation (`veks catalog generate`) is
//! CLI-specific and defined in this module.

pub use vectordata::catalog::resolver;
pub use vectordata::catalog::sources;

pub mod generate;
