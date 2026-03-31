// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! HDF5 dataset reader implementing [`VecSource`].
//!
//! Reads a single named dataset from an HDF5 file. The dataset name is
//! encoded in the path using `#` as a separator: `file.hdf5#dataset_name`.
//!
//! Uses the C-backed `hdf5` crate for full format compatibility.
//! Requires `libhdf5` at runtime (available via conda, apt, etc.).

use std::path::Path;

use super::{SourceMeta, VecSource};

/// Parse an HDF5 path of the form `file.hdf5#dataset_name`.
///
/// Returns `(file_path, dataset_name)`.
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
/// Returns `Vec<(name, rows, cols, element_size_bytes)>`.
pub fn list_datasets(path: &Path) -> Result<Vec<(String, u64, u32, usize)>, String> {
    let file = hdf5::File::open(path)
        .map_err(|e| format!("failed to open HDF5 {}: {}", path.display(), e))?;

    let mut results = Vec::new();
    for name in file.member_names().unwrap_or_default() {
        let ds = match file.dataset(&name) {
            Ok(ds) => ds,
            Err(_) => continue,
        };
        let shape = ds.shape();
        if shape.len() != 2 {
            continue;
        }
        let rows = shape[0] as u64;
        let cols = shape[1] as u32;
        let elem_size = ds.dtype().map(|dt| dt.size()).unwrap_or(4);
        results.push((name, rows, cols, elem_size));
    }
    Ok(results)
}

/// Open an HDF5 dataset as a [`VecSource`].
///
/// The `path` must use `file.hdf5#dataset_name` notation.
pub fn open(path: &Path) -> Result<Box<dyn VecSource>, String> {
    let (file_path, dataset_name) = parse_hdf5_path(path)
        .ok_or_else(|| format!("invalid HDF5 path (expected file.hdf5#dataset): {}", path.display()))?;

    let file = hdf5::File::open(&file_path)
        .map_err(|e| format!("failed to open HDF5 {}: {}", file_path.display(), e))?;

    let ds = file.dataset(&dataset_name)
        .map_err(|e| format!("failed to open dataset '{}': {}", dataset_name, e))?;

    let shape = ds.shape();
    if shape.len() != 2 {
        return Err(format!(
            "dataset '{}' has {} dimensions, expected 2",
            dataset_name, shape.len(),
        ));
    }
    let rows = shape[0] as u64;
    let cols = shape[1] as u32;
    let elem_size = ds.dtype().map(|dt| dt.size()).unwrap_or(4);

    // Determine whether the dataset is integer or float from the HDF5 dtype
    let is_integer = ds.dtype()
        .map(|dt| dt.is::<i32>() || dt.is::<i16>() || dt.is::<i64>() || dt.is::<u32>() || dt.is::<u8>())
        .unwrap_or(false);

    let raw_data: Vec<u8> = match elem_size {
        4 => {
            if is_integer {
                let arr = ds.read_raw::<i32>()
                    .map_err(|e| format!("cannot read dataset '{}': {}", dataset_name, e))?;
                arr.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                let arr = ds.read_raw::<f32>()
                    .map_err(|e| format!("cannot read dataset '{}': {}", dataset_name, e))?;
                arr.iter().flat_map(|v| v.to_le_bytes()).collect()
            }
        }
        8 => {
            let arr = ds.read_raw::<f64>()
                .map_err(|e| format!("cannot read dataset '{}': {}", dataset_name, e))?;
            arr.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        2 => {
            let arr = ds.read_raw::<i16>()
                .map_err(|e| format!("cannot read dataset '{}': {}", dataset_name, e))?;
            arr.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        1 => {
            let arr = ds.read_raw::<u8>()
                .map_err(|e| format!("cannot read dataset '{}': {}", dataset_name, e))?;
            arr
        }
        _ => return Err(format!("unsupported element size {} for '{}'", elem_size, dataset_name)),
    };

    let row_bytes = cols as usize * elem_size;

    Ok(Box::new(Hdf5Source {
        dimension: cols,
        elem_size,
        total_rows: rows,
        data: raw_data,
        row_bytes,
        cursor: 0,
    }))
}

/// Probe an HDF5 dataset for metadata without reading all data.
pub fn probe(path: &Path) -> Result<SourceMeta, String> {
    let (file_path, dataset_name) = parse_hdf5_path(path)
        .ok_or_else(|| format!("invalid HDF5 path: {}", path.display()))?;

    let file = hdf5::File::open(&file_path)
        .map_err(|e| format!("failed to open HDF5 {}: {}", file_path.display(), e))?;

    let ds = file.dataset(&dataset_name)
        .map_err(|e| format!("failed to open dataset '{}': {}", dataset_name, e))?;

    let shape = ds.shape();
    if shape.len() != 2 {
        return Err(format!("dataset '{}' is not 2-dimensional", dataset_name));
    }

    Ok(SourceMeta {
        dimension: shape[1] as u32,
        element_size: ds.dtype().map(|dt| dt.size()).unwrap_or(4),
        record_count: Some(shape[0] as u64),
    })
}

struct Hdf5Source {
    dimension: u32,
    elem_size: usize,
    total_rows: u64,
    data: Vec<u8>,
    row_bytes: usize,
    cursor: u64,
}

impl VecSource for Hdf5Source {
    fn dimension(&self) -> u32 { self.dimension }
    fn element_size(&self) -> usize { self.elem_size }
    fn record_count(&self) -> Option<u64> { Some(self.total_rows) }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        if self.cursor >= self.total_rows {
            return None;
        }
        let offset = self.cursor as usize * self.row_bytes;
        let end = offset + self.row_bytes;
        if end > self.data.len() {
            return None;
        }
        self.cursor += 1;
        Some(self.data[offset..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a small HDF5 file with known datasets for testing.
    fn create_test_hdf5(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("test.hdf5");
        let file = hdf5::File::create(&path).unwrap();

        // 5 vectors of dim 3 (f32)
        let base_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let base = file.new_dataset::<f32>().shape([5, 3]).create("base_vectors").unwrap();
        base.write_raw(&base_data).unwrap();

        // 2 query vectors of dim 3 (f32)
        let query_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let query = file.new_dataset::<f32>().shape([2, 3]).create("query_vectors").unwrap();
        query.write_raw(&query_data).unwrap();

        // 2×2 neighbor indices (i32)
        let idx_data: Vec<i32> = vec![0, 1, 2, 3];
        let indices = file.new_dataset::<i32>().shape([2, 2]).create("neighbor_indices").unwrap();
        indices.write_raw(&idx_data).unwrap();

        path
    }

    #[test]
    fn test_parse_hdf5_path() {
        let (file, ds) = parse_hdf5_path(Path::new("data.hdf5#train")).unwrap();
        assert_eq!(file, std::path::PathBuf::from("data.hdf5"));
        assert_eq!(ds, "train");
        assert!(parse_hdf5_path(Path::new("data.hdf5")).is_none());
        assert!(parse_hdf5_path(Path::new("#dataset")).is_none());
    }

    #[test]
    fn test_list_datasets() {
        let dir = tempfile::tempdir().unwrap();
        let hdf5_path = create_test_hdf5(dir.path());

        let datasets = list_datasets(&hdf5_path).unwrap();
        assert_eq!(datasets.len(), 3);
        let names: Vec<&str> = datasets.iter().map(|(n, _, _, _)| n.as_str()).collect();
        assert!(names.contains(&"base_vectors"));
        assert!(names.contains(&"query_vectors"));
        assert!(names.contains(&"neighbor_indices"));

        let base = datasets.iter().find(|(n, _, _, _)| n == "base_vectors").unwrap();
        assert_eq!(base.1, 5);  // rows
        assert_eq!(base.2, 3);  // cols
        assert_eq!(base.3, 4);  // f32 = 4 bytes
    }

    #[test]
    fn test_open_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let hdf5_path = create_test_hdf5(dir.path());

        let ds_path = dir.path().join("test.hdf5#base_vectors");
        let mut source = open(&ds_path).unwrap();
        assert_eq!(source.dimension(), 3);
        assert_eq!(source.element_size(), 4);
        assert_eq!(source.record_count(), Some(5));

        // First record should be [1.0, 2.0, 3.0] as LE bytes
        let record = source.next_record().unwrap();
        assert_eq!(record.len(), 12);
        let v0 = f32::from_le_bytes(record[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(record[4..8].try_into().unwrap());
        let v2 = f32::from_le_bytes(record[8..12].try_into().unwrap());
        assert_eq!((v0, v1, v2), (1.0, 2.0, 3.0));

        // Read all remaining
        let mut count = 1;
        while source.next_record().is_some() { count += 1; }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_open_i32_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let hdf5_path = create_test_hdf5(dir.path());

        let ds_path = dir.path().join("test.hdf5#neighbor_indices");
        let mut source = open(&ds_path).unwrap();
        assert_eq!(source.dimension(), 2);
        assert_eq!(source.element_size(), 4);
        assert_eq!(source.record_count(), Some(2));

        let record = source.next_record().unwrap();
        let v0 = i32::from_le_bytes(record[0..4].try_into().unwrap());
        let v1 = i32::from_le_bytes(record[4..8].try_into().unwrap());
        assert_eq!((v0, v1), (0, 1));
    }

    #[test]
    fn test_probe() {
        let dir = tempfile::tempdir().unwrap();
        let hdf5_path = create_test_hdf5(dir.path());

        let ds_path = dir.path().join("test.hdf5#query_vectors");
        let meta = probe(&ds_path).unwrap();
        assert_eq!(meta.dimension, 3);
        assert_eq!(meta.element_size, 4);
        assert_eq!(meta.record_count, Some(2));
    }

    #[test]
    fn test_nonexistent_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let hdf5_path = create_test_hdf5(dir.path());

        let ds_path = dir.path().join("test.hdf5#nonexistent");
        assert!(open(&ds_path).is_err());
    }
}
