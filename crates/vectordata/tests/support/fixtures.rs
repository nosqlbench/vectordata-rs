// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! Dataset files for tests: written, numbered so values are checkable,
//! and seeded so no two tests share content (and so a cache slot).

use std::path::Path;

use vectordata::merkle::MerkleRef;
use vectordata::Element;

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

/// Write the catalogs over the datasets under `root` the way a publish
/// does: each named dataset directory is marked publishable and the
/// catalog generator writes every level, so what a catalog lists is what
/// each `dataset.yaml` says. A test never writes a catalog by hand.
pub fn write_catalogs(root: &Path, datasets: &[&str]) {
    use vectordata::catalog::generate::CatalogTree;
    for name in datasets {
        std::fs::write(root.join(name).join(".publish"), "").unwrap();
    }
    let tree = CatalogTree::scan(root, &mut |_| {});
    tree.write(&tree.catalog_dirs(), "catalog", &mut |_| {}).unwrap();
}

/// `records` uniform records of `dim` elements of `T`, element `d` of
/// record `i` being `f(i, d)`, under a dimension header each.
pub fn write_xvec<T: Element>(path: &Path, records: usize, dim: usize, f: impl Fn(usize, usize) -> T) {
    let mut buf = Vec::with_capacity(records * (4 + dim * T::SIZE));
    let mut elem = vec![0u8; T::SIZE];
    for i in 0..records {
        buf.extend_from_slice(&(dim as i32).to_le_bytes());
        for d in 0..dim {
            f(i, d).write_le_bytes(&mut elem);
            buf.extend_from_slice(&elem);
        }
    }
    std::fs::write(path, buf).unwrap();
}

/// A header-free column of `values`.
pub fn write_column<T: Element>(path: &Path, values: &[T]) {
    let mut buf = vec![0u8; values.len() * T::SIZE];
    for (v, slot) in values.iter().zip(buf.chunks_exact_mut(T::SIZE)) {
        v.write_le_bytes(slot);
    }
    std::fs::write(path, buf).unwrap();
}

/// Variable-length records of `T`: each slice of `records` under its
/// own dimension header.
pub fn write_vvec<T: Element>(path: &Path, records: &[Vec<T>]) {
    let mut buf = Vec::new();
    let mut elem = vec![0u8; T::SIZE];
    for r in records {
        buf.extend_from_slice(&(r.len() as i32).to_le_bytes());
        for v in r {
            v.write_le_bytes(&mut elem);
            buf.extend_from_slice(&elem);
        }
    }
    std::fs::write(path, buf).unwrap();
}

/// A `dataset.yaml` under `dir` with one `default` profile whose facet
/// lines are `facets` (each `key: locator`, already indented or not).
pub fn write_dataset_yaml(dir: &Path, name: &str, facets: &[(&str, &str)]) {
    let mut yaml = format!("name: {name}\nprofiles:\n  default:\n");
    for (key, locator) in facets {
        yaml.push_str(&format!("    {key}: {locator}\n"));
    }
    std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();
}
