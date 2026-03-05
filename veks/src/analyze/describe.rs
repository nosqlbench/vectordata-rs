// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Implementation of `veks analyze describe` — reports metadata and
//! normalization status for vector files and dataset facets.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use super::args::DescribeArgs;
use crate::formats::VecFormat;
use crate::formats::reader::{self, SourceMeta};
use crate::formats::reader::parquet::ParquetDirReader;
use crate::import::dataset::DatasetConfig;
use crate::import::facet::Facet;

/// Run the describe command.
pub fn run(args: DescribeArgs) {
    let source = &args.source;

    // Dataset facet mode: source is a .yaml/.yml file or --facet is specified
    let is_yaml = source
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "yaml" || e == "yml")
        .unwrap_or(false);

    if is_yaml || args.facet.is_some() {
        run_dataset_mode(&args);
    } else {
        run_file_mode(&args);
    }
}

/// Describe a single file or directory of vector data.
fn run_file_mode(args: &DescribeArgs) {
    let source = &args.source;

    let format = if let Some(ref fmt_str) = args.from {
        VecFormat::from_extension(fmt_str)
            .unwrap_or_else(|| panic!("Unknown format: {}", fmt_str))
    } else {
        VecFormat::detect(source)
            .unwrap_or_else(|| panic!("Cannot detect format for: {}", source.display()))
    };

    if format == VecFormat::Parquet {
        run_file_mode_parquet(source);
    } else {
        run_file_mode_standard(source, format);
    }
}

/// Describe a parquet source, reading the schema first and handling
/// non-vector parquet files gracefully.
fn run_file_mode_parquet(source: &Path) {
    let probe = ParquetDirReader::probe(source)
        .unwrap_or_else(|e| panic!("Failed to probe {}: {}", source.display(), e));

    println!("Analyzing: {}\n", source.display());
    println!("File Description:");
    println!("- File: {}", source.display());
    println!("- Format: parquet");
    println!("- File Count: {}", probe.file_count);
    println!("- Row Count: {}", probe.row_count);
    println!("- Schema:");
    println!("{}", probe.schema);

    if let Some(ref vmeta) = probe.vector_meta {
        println!(
            "- Data Type: {} ({} bytes/element)",
            VecFormat::Parquet.data_type_name(),
            vmeta.element_size
        );
        println!("- Dimensions: {}", vmeta.dimension);
        if let Some(count) = vmeta.record_count {
            println!("- Vector Count: {}", count);
        }

        let norm = check_normalization(source, VecFormat::Parquet, vmeta);
        println!("- Normalization: {}", norm.label);
        if let Some(advice) = &norm.dot_product_advice {
            println!("- Dot Product: {}", advice);
        }
    } else {
        println!("- Vectors: N/A (no list-of-float column found)");
    }
}

/// Describe a non-parquet source.
fn run_file_mode_standard(source: &Path, format: VecFormat) {
    let meta = reader::probe_source(source, format)
        .unwrap_or_else(|e| panic!("Failed to probe {}: {}", source.display(), e));

    println!("Analyzing: {}\n", source.display());
    println!("File Description:");
    println!("- File: {}", source.display());
    println!("- Format: {}", format.name());
    println!(
        "- Data Type: {} ({} bytes/element)",
        format.data_type_name(),
        meta.element_size
    );
    println!("- Dimensions: {}", meta.dimension);

    if let Some(count) = meta.record_count {
        println!("- Vector Count: {}", count);

        if format.is_xvec() {
            let record_size = 4 + (meta.dimension as u64) * (meta.element_size as u64);
            println!("- Record Size: {} bytes", record_size);

            if let Ok(file_meta) = fs::metadata(source) {
                let file_size = file_meta.len();
                println!("- File Size: {} bytes", file_size);
            }
        }
    }

    // Normalization check
    let norm = check_normalization(source, format, &meta);
    println!("- Normalization: {}", norm.label);
    if let Some(advice) = &norm.dot_product_advice {
        println!("- Dot Product: {}", advice);
    }
}

/// Describe views from a dataset.yaml's default profile.
fn run_dataset_mode(args: &DescribeArgs) {
    let source = &args.source;

    let config = DatasetConfig::load(source)
        .unwrap_or_else(|e| panic!("Failed to load dataset config: {}", e));

    let base_dir = source.parent().unwrap_or(Path::new("."));

    let facet_filter = args.facet.map(|f| f.key().to_string());

    let default_profile = match config.default_profile() {
        Some(p) => p,
        None => {
            eprintln!("Error: no default profile in {}", source.display());
            return;
        }
    };

    let mut described = false;
    for (key, view) in &default_profile.views {
        if let Some(ref filter) = facet_filter {
            if key != filter {
                continue;
            }
        }

        let facet = match Facet::from_key(key) {
            Some(f) => f,
            None => {
                eprintln!("Warning: unknown view key '{}', skipping", key);
                continue;
            }
        };

        // Resolve the file to probe from the view's source path
        let probe_path = {
            let view_path = base_dir.join(view.path());
            if view_path.exists() {
                view_path
            } else {
                eprintln!(
                    "Warning: view '{}': path '{}' does not exist, skipping",
                    key,
                    view_path.display()
                );
                continue;
            }
        };

        // Determine format
        let format = if let Some(ref fmt_override) = args.from {
            VecFormat::from_extension(fmt_override)
                .unwrap_or_else(|| panic!("Unknown format: {}", fmt_override))
        } else {
            VecFormat::detect(&probe_path)
                .unwrap_or_else(|| panic!("Cannot detect format for view '{}'", key))
        };

        if described {
            println!();
        }

        println!("Analyzing: {} view {}\n", source.display(), facet.key());
        println!("Dataset View Description:");
        println!("- View: {}", facet.key());
        println!("- Format: {}", format.name());

        if format == VecFormat::Parquet {
            match ParquetDirReader::probe(&probe_path) {
                Ok(probe) => {
                    println!("- File Count: {}", probe.file_count);
                    println!("- Row Count: {}", probe.row_count);
                    println!("- Schema:");
                    println!("{}", probe.schema);

                    if let Some(ref vmeta) = probe.vector_meta {
                        println!(
                            "- Data Type: {} ({} bytes/element)",
                            format.data_type_name(),
                            vmeta.element_size
                        );
                        println!("- Dimensions: {}", vmeta.dimension);
                        if let Some(count) = vmeta.record_count {
                            println!("- Vector Count: {}", count);
                        }
                        let norm = check_normalization(&probe_path, format, vmeta);
                        println!("- Normalization: {}", norm.label);
                        if let Some(advice) = &norm.dot_product_advice {
                            println!("- Dot Product: {}", advice);
                        }
                    } else {
                        println!("- Vectors: N/A (no list-of-float column found)");
                    }
                }
                Err(e) => {
                    eprintln!("Warning: view '{}': failed to probe: {}", key, e);
                    continue;
                }
            }
        } else {
            let meta = match reader::probe_source(&probe_path, format) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Warning: view '{}': failed to probe: {}", key, e);
                    continue;
                }
            };

            println!(
                "- Data Type: {} ({} bytes/element)",
                format.data_type_name(),
                meta.element_size
            );
            println!("- Dimensions: {}", meta.dimension);

            if let Some(count) = meta.record_count {
                println!("- Vector Count: {}", count);

                if format.is_xvec() {
                    let record_size = 4 + (meta.dimension as u64) * (meta.element_size as u64);
                    println!("- Record Size: {} bytes", record_size);
                }
            }

            let norm = check_normalization(&probe_path, format, &meta);
            println!("- Normalization: {}", norm.label);
            if let Some(advice) = &norm.dot_product_advice {
                println!("- Dot Product: {}", advice);
            }
        }

        described = true;
    }

    if !described {
        if let Some(ref filter) = facet_filter {
            eprintln!("Error: view '{}' not found in {}", filter, source.display());
        } else {
            eprintln!("Error: no views found in {}", source.display());
        }
    }
}

// -- Normalization detection ---------------------------------------------------

/// Result of normalization checking.
struct NormResult {
    label: String,
    dot_product_advice: Option<String>,
}

/// Check whether vectors in a file are L2-normalized.
///
/// Samples up to 100 vectors evenly spaced through the file and checks
/// whether `|sqrt(sum(v_i^2)) - 1.0| < 0.01`. Vectors are considered
/// normalized if >90% of samples pass.
fn check_normalization(path: &Path, format: VecFormat, meta: &SourceMeta) -> NormResult {
    match format {
        VecFormat::Fvec => check_normalization_fvec(path, meta),
        VecFormat::Hvec | VecFormat::Dvec | VecFormat::Svec => {
            check_normalization_xvec_seek(path, format, meta)
        }
        VecFormat::Npy => check_normalization_npy_seek(path, meta),
        VecFormat::Ivec | VecFormat::Bvec | VecFormat::Slab | VecFormat::Parquet => NormResult {
            label: "N/A (not float vectors)".to_string(),
            dot_product_advice: None,
        },
    }
}

/// Check normalization for fvec using MmapVectorReader O(1) random access.
fn check_normalization_fvec(path: &Path, meta: &SourceMeta) -> NormResult {
    let count = match meta.record_count {
        Some(c) if c > 0 => c,
        _ => {
            return NormResult {
                label: "N/A (empty file)".to_string(),
                dot_product_advice: None,
            }
        }
    };

    let reader = match MmapVectorReader::<f32>::open_fvec(path) {
        Ok(r) => r,
        Err(e) => {
            return NormResult {
                label: format!("N/A (read error: {})", e),
                dot_product_advice: None,
            }
        }
    };

    let sample_count = std::cmp::min(100, count as usize);
    let stride = if sample_count > 1 {
        count as usize / sample_count
    } else {
        1
    };

    let mut passed = 0usize;
    let mut checked = 0usize;

    for i in 0..sample_count {
        let idx = i * stride;
        if idx >= reader.count() {
            break;
        }
        if let Ok(vec) = reader.get(idx) {
            checked += 1;
            if is_normalized_f32(&vec) {
                passed += 1;
            }
        }
    }

    make_norm_result(passed, checked)
}

/// Check normalization for hvec/dvec/svec via seek-based sampling.
fn check_normalization_xvec_seek(
    path: &Path,
    format: VecFormat,
    meta: &SourceMeta,
) -> NormResult {
    let count = match meta.record_count {
        Some(c) if c > 0 => c,
        _ => {
            return NormResult {
                label: "N/A (empty file)".to_string(),
                dot_product_advice: None,
            }
        }
    };

    let elem_size = format.element_size() as u64;
    let dim = meta.dimension as u64;
    let record_stride = 4 + dim * elem_size;

    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            return NormResult {
                label: format!("N/A (read error: {})", e),
                dot_product_advice: None,
            }
        }
    };

    let sample_count = std::cmp::min(100, count as usize);
    let index_stride = if sample_count > 1 {
        count / sample_count as u64
    } else {
        1
    };

    let data_len = (dim * elem_size) as usize;
    let mut buf = vec![0u8; data_len];
    let mut passed = 0usize;
    let mut checked = 0usize;

    for i in 0..sample_count {
        let idx = i as u64 * index_stride;
        let offset = idx * record_stride + 4; // skip dimension header
        if file.seek(SeekFrom::Start(offset)).is_err() {
            break;
        }
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        checked += 1;

        let norm_sq = match format {
            VecFormat::Hvec => l2_norm_sq_half(&buf),
            VecFormat::Dvec => l2_norm_sq_f64(&buf),
            VecFormat::Svec => {
                // svec is short (i16), not float — skip normalization
                return NormResult {
                    label: "N/A (not float vectors)".to_string(),
                    dot_product_advice: None,
                };
            }
            _ => unreachable!(),
        };

        if (norm_sq.sqrt() - 1.0).abs() < 0.01 {
            passed += 1;
        }
    }

    make_norm_result(passed, checked)
}

/// Check normalization for npy via seek-based sampling.
///
/// Npy files are flat arrays on disk: after the header, data is contiguous
/// at `rows * cols * element_size` bytes. This allows O(1) random access
/// by seeking to `data_offset + record_index * record_stride`.
fn check_normalization_npy_seek(path: &Path, meta: &SourceMeta) -> NormResult {
    let count = match meta.record_count {
        Some(c) if c > 0 => c,
        _ => {
            return NormResult {
                label: "N/A (empty file)".to_string(),
                dot_product_advice: None,
            }
        }
    };

    // Find the first npy file (directory or single file)
    let first_file = if path.is_dir() {
        let mut entries: Vec<_> = match fs::read_dir(path) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".npy"))
                .map(|e| e.path())
                .collect(),
            Err(e) => {
                return NormResult {
                    label: format!("N/A (read error: {})", e),
                    dot_product_advice: None,
                }
            }
        };
        entries.sort();
        match entries.into_iter().next() {
            Some(f) => f,
            None => {
                return NormResult {
                    label: "N/A (no .npy files)".to_string(),
                    dot_product_advice: None,
                }
            }
        }
    } else {
        path.to_path_buf()
    };

    // Parse npy header to find data offset
    let (data_offset, file_rows) = match npy_data_offset(&first_file) {
        Ok(v) => v,
        Err(e) => {
            return NormResult {
                label: format!("N/A (header error: {})", e),
                dot_product_advice: None,
            }
        }
    };

    let elem_size = meta.element_size;
    let dim = meta.dimension as usize;
    let record_stride = dim * elem_size;

    let mut file = match fs::File::open(&first_file) {
        Ok(f) => f,
        Err(e) => {
            return NormResult {
                label: format!("N/A (read error: {})", e),
                dot_product_advice: None,
            }
        }
    };

    // Sample evenly across the total record count, but only from this file
    let sample_count = std::cmp::min(100, count as usize);
    let index_stride = if sample_count > 1 {
        count / sample_count as u64
    } else {
        1
    };

    let mut buf = vec![0u8; record_stride];
    let mut passed = 0usize;
    let mut checked = 0usize;

    for i in 0..sample_count {
        let idx = i as u64 * index_stride;
        // Only sample records that fall within this first file
        if idx >= file_rows {
            break;
        }
        let offset = data_offset + idx * record_stride as u64;
        if file.seek(SeekFrom::Start(offset)).is_err() {
            break;
        }
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        checked += 1;

        let norm_sq = match elem_size {
            4 => l2_norm_sq_f32_bytes(&buf, dim),
            8 => l2_norm_sq_f64(&buf),
            2 => l2_norm_sq_half(&buf),
            _ => {
                return NormResult {
                    label: "N/A (not float vectors)".to_string(),
                    dot_product_advice: None,
                };
            }
        };

        if (norm_sq.sqrt() - 1.0).abs() < 0.01 {
            passed += 1;
        }
    }

    make_norm_result(passed, checked)
}

/// Parse an npy file header and return `(data_offset, row_count)`.
///
/// The data offset is the byte position where the flat array data begins
/// (immediately after the header). This enables seek-based random access.
fn npy_data_offset(path: &Path) -> Result<(u64, u64), String> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let mut file = std::io::BufReader::new(
        fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?,
    );

    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|e| format!("Read error: {}", e))?;
    if &magic != b"\x93NUMPY" {
        return Err(format!("{} is not a valid npy file", path.display()));
    }

    let major = file.read_u8().map_err(|e| format!("Read error: {}", e))?;
    let _minor = file.read_u8().map_err(|e| format!("Read error: {}", e))?;

    let header_len_size: u64;
    let header_len: usize;
    if major >= 2 {
        header_len = file
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Read error: {}", e))? as usize;
        header_len_size = 4;
    } else {
        header_len = file
            .read_u16::<LittleEndian>()
            .map_err(|e| format!("Read error: {}", e))? as usize;
        header_len_size = 2;
    }

    // data_offset = magic(6) + version(2) + header_len_field + header_len
    let data_offset = 6 + 2 + header_len_size + header_len as u64;

    // Parse shape to get row count
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    let header = String::from_utf8_lossy(&header_bytes);

    let rows = extract_npy_rows(&header)
        .ok_or_else(|| format!("Could not parse shape from {}", path.display()))?;

    Ok((data_offset, rows as u64))
}

/// Extract the row count from an npy header shape tuple.
fn extract_npy_rows(header: &str) -> Option<usize> {
    let shape_idx = header.find("'shape'")?;
    let after = &header[shape_idx..];
    let paren_start = after.find('(')?;
    let paren_end = after.find(')')?;
    let inside = &after[paren_start + 1..paren_end];
    let parts: Vec<&str> = inside.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if parts.len() >= 1 {
        parts[0].parse().ok()
    } else {
        None
    }
}

// -- Norm helpers --------------------------------------------------------------

/// Check if an f32 slice is L2-normalized (||v|| ≈ 1.0).
fn is_normalized_f32(v: &[f32]) -> bool {
    let norm_sq: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    (norm_sq.sqrt() - 1.0).abs() < 0.01
}

/// Compute L2 norm squared from raw LE f32 bytes.
fn l2_norm_sq_f32_bytes(data: &[u8], dim: usize) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..dim {
        let offset = i * 4;
        if offset + 4 > data.len() {
            break;
        }
        let val = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as f64;
        sum += val * val;
    }
    sum
}

/// Compute L2 norm squared from raw LE f64 bytes.
fn l2_norm_sq_f64(data: &[u8]) -> f64 {
    let mut sum = 0.0f64;
    let count = data.len() / 8;
    for i in 0..count {
        let offset = i * 8;
        let val = f64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        sum += val * val;
    }
    sum
}

/// Compute L2 norm squared from raw LE half-precision (f16) bytes.
fn l2_norm_sq_half(data: &[u8]) -> f64 {
    let mut sum = 0.0f64;
    let count = data.len() / 2;
    for i in 0..count {
        let offset = i * 2;
        let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let val = half::f16::from_bits(bits).to_f64();
        sum += val * val;
    }
    sum
}

/// Build a `NormResult` from the pass/check counts.
fn make_norm_result(passed: usize, checked: usize) -> NormResult {
    if checked == 0 {
        return NormResult {
            label: "N/A (no samples)".to_string(),
            dot_product_advice: None,
        };
    }

    let ratio = passed as f64 / checked as f64;
    if ratio > 0.9 {
        NormResult {
            label: "NORMALIZED (||v||=1.0)".to_string(),
            dot_product_advice: Some(
                "Safe to use DOT_PRODUCT metric (vectors are unit-normalized)".to_string(),
            ),
        }
    } else {
        NormResult {
            label: "NOT NORMALIZED".to_string(),
            dot_product_advice: Some(
                "DO NOT use DOT_PRODUCT metric (vectors not normalized)\n\
                               Use EUCLIDEAN or COSINE instead"
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_normalized_f32_unit_vector() {
        // A simple 3D unit vector: [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
        let s = 1.0f32 / 3.0f32.sqrt();
        assert!(is_normalized_f32(&[s, s, s]));
    }

    #[test]
    fn test_is_normalized_f32_not_normalized() {
        assert!(!is_normalized_f32(&[1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_is_normalized_f32_axis_aligned() {
        assert!(is_normalized_f32(&[1.0, 0.0, 0.0]));
        assert!(is_normalized_f32(&[0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_l2_norm_sq_f32_bytes() {
        let v: Vec<u8> = [1.0f32, 0.0, 0.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let norm_sq = l2_norm_sq_f32_bytes(&v, 3);
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm_sq_f64() {
        let v: Vec<u8> = [0.5f64, 0.5, 0.5, 0.5]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let norm_sq = l2_norm_sq_f64(&v);
        assert!((norm_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm_sq_half() {
        // Unit vector [1.0, 0.0] in half precision
        let one = half::f16::from_f32(1.0);
        let zero = half::f16::from_f32(0.0);
        let mut data = Vec::new();
        data.extend_from_slice(&one.to_le_bytes());
        data.extend_from_slice(&zero.to_le_bytes());
        let norm_sq = l2_norm_sq_half(&data);
        assert!((norm_sq - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_make_norm_result_normalized() {
        let result = make_norm_result(95, 100);
        assert!(result.label.contains("NORMALIZED"));
        assert!(result.dot_product_advice.is_some());
        assert!(result.dot_product_advice.unwrap().contains("Safe"));
    }

    #[test]
    fn test_make_norm_result_not_normalized() {
        let result = make_norm_result(50, 100);
        assert!(result.label.contains("NOT NORMALIZED"));
        assert!(result.dot_product_advice.is_some());
        assert!(result.dot_product_advice.unwrap().contains("DO NOT"));
    }

    #[test]
    fn test_make_norm_result_empty() {
        let result = make_norm_result(0, 0);
        assert!(result.label.contains("N/A"));
    }
}
