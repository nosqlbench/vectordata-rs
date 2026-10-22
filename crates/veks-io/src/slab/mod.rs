// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Slab files as vector records (the `slab` feature): a reader of a
//! slab's content namespace as a stream, and a writer that seals it
//! with a schema sidecar.

pub mod reader;
pub mod writer;
