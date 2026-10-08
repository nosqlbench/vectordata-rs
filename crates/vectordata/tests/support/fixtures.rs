// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! Dataset files for tests: written, numbered so values are checkable,
//! and seeded so no two tests share content (and so a cache slot).

use std::path::Path;

use vectordata::merkle::MerkleRef;

/// The value `write_fvec` puts at element `d` of record `i`.
pub fn fvec_value(seed: f32, dim: usize, i: usize, d: usize) -> f32 {
    seed + (i * dim + d) as f32
}

/// `records` fvec records of `dim` elements, values from [`fvec_value`].
pub fn write_fvec(path: &Path, records: usize, dim: usize, seed: f32) {
    let mut buf = Vec::with_capacity(records * (4 + dim * 4));
    for i in 0..records {
        buf.extend_from_slice(&(dim as i32).to_le_bytes());
        for d in 0..dim {
            buf.extend_from_slice(&fvec_value(seed, dim, i, d).to_le_bytes());
        }
    }
    std::fs::write(path, buf).unwrap();
}

/// `records` one-element ivvec records holding `base + i`.
pub fn write_ivvec(path: &Path, records: usize, base: i32) {
    let mut buf = Vec::new();
    for i in 0..records as i32 {
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&(base + i).to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

/// Publish a `.mref` beside `path` with small chunks, so a small
/// fixture spans several of them.
pub fn write_mref(path: &Path) {
    let content = std::fs::read(path).unwrap();
    let mref = MerkleRef::from_content(&content, 4 * 1024);
    let mut p = path.to_path_buf().into_os_string();
    p.push(".mref");
    mref.save(Path::new(&p)).unwrap();
}
