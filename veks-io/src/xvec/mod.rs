// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! xvec format readers and writers.
//!
//! The xvec family (fvec, ivec, bvec, dvec, mvec, svec) stores vectors as
//! contiguous records: `[dim:i32_le, element_0, element_1, ..., element_dim-1]`.
//! Each element is in the format's native type (f32, i32, u8, f64, f16, i16).

pub mod reader;
pub mod writer;
pub mod mmap;
