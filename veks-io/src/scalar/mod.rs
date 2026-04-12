// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Scalar format I/O — flat packed arrays with no header.
//!
//! Each file is a contiguous array of fixed-size elements. Ordinal N
//! is at byte offset `N * element_size`. Record count is
//! `file_size / element_size`. Dimension is always 1.

pub mod convert;
pub mod reader;
pub mod writer;
pub mod mmap;
