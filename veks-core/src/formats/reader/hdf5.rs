// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! HDF5 path parsing and format detection.
//!
//! The `hdf5` library dependency has been removed. HDF5 files must be
//! pre-converted to fvec/ivec format using external tools (e.g., the
//! knn\_utils `hdf5_to_fvecs.py` script) before use with veks.
//!
//! This module retains:
//! - Path parsing (`file.hdf5#dataset`) for pipeline configuration
//! - Format detection (`.hdf5`/`.h5` extensions)
//! - Error messages directing users to convert

use std::path::Path;

use super::{SourceMeta, VecSource};

/// Parse an HDF5 path of the form `file.hdf5#dataset_name`.
///
/// Returns `(file_path, dataset_name)`. This is used for pipeline
/// configuration even though HDF5 I/O is no longer supported.
pub fn parse_hdf5_path(path: &Path) -> Option<(std::path::PathBuf, String)> {
    let s = path.to_string_lossy();
    let hash_pos = s.rfind('#')?;
    let file_part = &s[..hash_pos];
    let dataset_part = &s[hash_pos + 1..];
    if file_part.is_empty() || dataset_part.is_empty() {
        return None;
    }
    Some((std::path::PathBuf::from(file_part), dataset_part.to_string()))
}

/// List the 2-D datasets inside an HDF5 file.
///
/// Returns an error — HDF5 I/O is no longer supported. Convert the file
/// to fvec format first using `hdf5_to_fvecs.py` or similar.
pub fn list_datasets(path: &Path) -> Result<Vec<(String, u64, u32, usize)>, String> {
    Err(format!(
        "HDF5 I/O is no longer supported. Convert {} to fvec format first \
         (e.g., using knn_utils hdf5_to_fvecs.py).",
        path.display()
    ))
}

/// Open an HDF5 dataset as a [`VecSource`].
///
/// Returns an error — HDF5 I/O is no longer supported.
pub fn open(path: &Path) -> Result<Box<dyn VecSource>, String> {
    Err(format!(
        "HDF5 I/O is no longer supported. Convert {} to fvec format first \
         (e.g., using knn_utils hdf5_to_fvecs.py).",
        path.display()
    ))
}

/// Probe an HDF5 dataset for metadata without reading all data.
///
/// Returns an error — HDF5 I/O is no longer supported.
pub fn probe(path: &Path) -> Result<SourceMeta, String> {
    Err(format!(
        "HDF5 I/O is no longer supported. Convert {} to fvec format first \
         (e.g., using knn_utils hdf5_to_fvecs.py).",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hdf5_path() {
        let (file, ds) = parse_hdf5_path(Path::new("data.hdf5#train")).unwrap();
        assert_eq!(file, std::path::PathBuf::from("data.hdf5"));
        assert_eq!(ds, "train");
        assert!(parse_hdf5_path(Path::new("data.hdf5")).is_none());
        assert!(parse_hdf5_path(Path::new("#dataset")).is_none());
    }

    #[test]
    fn test_open_returns_error() {
        let result = open(Path::new("test.hdf5#train"));
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("no longer supported"), "unexpected error: {}", msg),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn test_probe_returns_error() {
        let result = probe(Path::new("test.hdf5#train"));
        assert!(result.is_err());
    }

    #[test]
    fn test_list_datasets_returns_error() {
        let result = list_datasets(Path::new("test.hdf5"));
        assert!(result.is_err());
    }
}
