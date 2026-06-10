// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Unified file and directory filtering rules for the whole toolchain.
//!
//! All decisions about which files/directories to include or exclude —
//! for publishing/pushing, merkle coverage, catalog generation, workspace
//! checks, and sync operations — are defined here. No other module should
//! inline its own filtering predicates. This lives in `vectordata` (the base
//! crate) so the push engine can apply exactly the same rules as the higher
//! `veks publish` surface; `veks-core` and `veks` re-export it unchanged.

/// Directories that are always skipped during any tree walk.
///
/// Hidden directories (`.` prefix) are local workspace state.
/// Underscore-prefixed directories (`_`) are private/scratch.
/// `__pycache__` and `node_modules` are tooling artifacts.
/// `target` is the Rust build directory.
pub fn is_excluded_dir(name: &str) -> bool {
    name.starts_with('.')
        || name.starts_with('_')
        || name == "__pycache__"
        || name == "node_modules"
        || name == "target"
}

/// Files that are always excluded from publishing, merkle, and workspace enumeration.
///
/// Hidden files (`.` prefix) are local state.
/// Underscore-prefixed files (`_`) are private/scratch.
/// Temp/partial/pyc files are build/runtime artifacts.
pub fn is_excluded_file(name: &str) -> bool {
    name.starts_with('.')
        || name.starts_with('_')
        || name.ends_with(".tmp")
        || name.ends_with(".partial")
        || name.ends_with(".pyc")
}

/// Small infrastructure/metadata files that are frequently regenerated.
///
/// These files don't need merkle protection and are not counted for
/// catalog staleness comparisons. They ARE published (inside dataset
/// dirs) but don't trigger merkle or staleness checks.
const INFRASTRUCTURE_FILES: &[&str] = &[
    "dataset.yaml",
    "dataset.yml",
    "dataset.json",
    "dataset.jsonl",
    "dataset.log",
    "runlog.jsonl",
    "catalog.json",
    "catalog.yaml",
    "variables.yaml",
    "variables.json",
];

/// Check if a file is infrastructure metadata.
///
/// Infrastructure files are small, frequently regenerated files that
/// don't need merkle protection and are excluded from catalog staleness
/// comparisons.
pub fn is_infrastructure_file(name: &str) -> bool {
    INFRASTRUCTURE_FILES.contains(&name)
}

/// Files that belong at intermediate (on-path, non-dataset) directories
/// in the publish hierarchy.
///
/// Only catalog files and their merkle companions are published from
/// intermediate directories. Arbitrary data files at these levels are
/// not dataset content and must not be published.
pub fn is_intermediate_publishable(name: &str) -> bool {
    name == "catalog.json"
        || name == "catalog.yaml"
        || name == "catalog.json.mref"
        || name == "catalog.yaml.mref"
}

/// Files to skip when checking catalog staleness.
///
/// Catalog files themselves, merkle references, the append-only
/// provenance log, and the JSON format copy of dataset config are
/// not compared against catalog timestamps. These are all derived
/// or infrastructure files whose mtime may be newer than catalogs
/// without indicating actual content changes.
pub fn is_catalog_staleness_exempt(name: &str) -> bool {
    is_infrastructure_file(name)
        || name.ends_with(".mref")
}

/// Files that are themselves *derived metadata about another file* — provenance
/// sidecars (`*.provenance.json`) and merkle references (`*.mref`/`*.mrkl`).
///
/// These must never be given their own provenance sidecar or merkle reference,
/// or the suffixes compound without bound — e.g.
/// `query_vectors.fvecs.provenance.json.mref.provenance.json`. Both the merkle
/// generator and the provenance writer treat this as a hard skip.
pub fn is_derived_sidecar(name: &str) -> bool {
    name.ends_with(".provenance.json") || name.ends_with(".mref") || name.ends_with(".mrkl")
}

/// Files that should not receive their own merkle coverage.
///
/// Infrastructure files at the dataset root level (dataset.yaml, catalog
/// files, variables.yaml) are too small to benefit from chunked merkle
/// verification. Everything within `profiles/` gets coverage categorically.
/// Derived sidecars (provenance/merkle) and excluded files are always exempt —
/// merkle must not cover a provenance sidecar or another merkle file.
pub fn is_merkle_exempt(name: &str) -> bool {
    is_infrastructure_file(name) || is_derived_sidecar(name) || is_excluded_file(name)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_excluded_dirs() {
        assert!(is_excluded_dir(".hidden"));
        assert!(is_excluded_dir("_scratch"));
        assert!(is_excluded_dir("__pycache__"));
        assert!(is_excluded_dir("node_modules"));
        assert!(is_excluded_dir("target"));
        assert!(!is_excluded_dir("profiles"));
        assert!(!is_excluded_dir("datasets"));
        assert!(!is_excluded_dir("docs"), "docs/ is a standard dataset artifact directory");
    }

    #[test]
    fn test_excluded_files() {
        assert!(is_excluded_file(".gitignore"));
        assert!(is_excluded_file("_source.mvec"));
        assert!(is_excluded_file("data.tmp"));
        assert!(is_excluded_file("data.partial"));
        assert!(is_excluded_file("module.pyc"));
        assert!(!is_excluded_file("dataset.yaml"));
        assert!(!is_excluded_file("base.fvec"));
    }

    #[test]
    fn test_infrastructure_files() {
        assert!(is_infrastructure_file("dataset.yaml"));
        assert!(is_infrastructure_file("dataset.yml"));
        assert!(is_infrastructure_file("dataset.log"));
        assert!(is_infrastructure_file("catalog.json"));
        assert!(is_infrastructure_file("catalog.yaml"));
        assert!(is_infrastructure_file("variables.yaml"));
        assert!(!is_infrastructure_file("base.fvec"));
        assert!(!is_infrastructure_file("profiles"));
    }

    #[test]
    fn test_intermediate_publishable() {
        assert!(is_intermediate_publishable("catalog.json"));
        assert!(is_intermediate_publishable("catalog.yaml"));
        assert!(is_intermediate_publishable("catalog.json.mref"));
        assert!(is_intermediate_publishable("catalog.yaml.mref"));
        assert!(!is_intermediate_publishable("base_test.mvec"));
        assert!(!is_intermediate_publishable("dataset.yaml"));
    }

    #[test]
    fn test_derived_sidecar() {
        // Provenance sidecars and merkle refs are derived metadata.
        assert!(is_derived_sidecar("base.fvec.provenance.json"));
        assert!(is_derived_sidecar("base.fvec.mref"));
        assert!(is_derived_sidecar("data.mrkl"));
        // …including compounded ones (which is exactly what we must stop growing).
        assert!(is_derived_sidecar("q.fvecs.provenance.json.mref"));
        // Real content is not.
        assert!(!is_derived_sidecar("base.fvec"));
        assert!(!is_derived_sidecar("dataset.json"));
    }

    #[test]
    fn test_merkle_exempt() {
        // Infrastructure files at dataset root — exempt (too small)
        assert!(is_merkle_exempt("dataset.yaml"));
        assert!(is_merkle_exempt("catalog.json"));
        assert!(is_merkle_exempt("variables.yaml"));
        // Derived sidecars — exempt (merkle must not cover provenance/merkle files)
        assert!(is_merkle_exempt("base.fvec.mref"));
        assert!(is_merkle_exempt("data.mrkl"));
        assert!(is_merkle_exempt("base.fvec.provenance.json"), "provenance sidecars get no .mref");
        // Excluded files — exempt
        assert!(is_merkle_exempt(".hidden"));
        assert!(is_merkle_exempt("data.tmp"));
        // Data files — get merkle coverage
        assert!(!is_merkle_exempt("base.fvec"));
        assert!(!is_merkle_exempt("neighbor_indices.ivecs"));
        assert!(!is_merkle_exempt("metadata.slab"));
    }

    #[test]
    fn test_catalog_staleness_exempt() {
        assert!(is_catalog_staleness_exempt("catalog.json"));
        assert!(is_catalog_staleness_exempt("catalog.yaml"));
        assert!(is_catalog_staleness_exempt("dataset.yaml"));
        assert!(is_catalog_staleness_exempt("dataset.yml"));
        assert!(is_catalog_staleness_exempt("dataset.json"));
        assert!(is_catalog_staleness_exempt("dataset.log"));
        assert!(is_catalog_staleness_exempt("variables.yaml"));
        assert!(is_catalog_staleness_exempt("base.fvec.mref"));
        assert!(!is_catalog_staleness_exempt("base.fvec"));
    }
}
