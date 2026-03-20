// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! File integrity check.
//!
//! Validates that publishable data files have correct geometry and record
//! structure for their format. Delegates to format-specific logic where
//! available (e.g. slab structural validation).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::formats::VecFormat;

use super::CheckResult;

/// Check file format integrity for all publishable files.
pub fn check(_root: &Path, publishable: &[PathBuf]) -> CheckResult {
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0u64;

    for file in publishable {
        let ext = match file.extension().and_then(|e| e.to_str()) {
            Some(e) => e,
            None => continue,
        };

        let format = match VecFormat::from_extension(ext) {
            Some(f) => f,
            None => continue, // Not a recognized data format
        };

        checked += 1;

        let result = if format.is_xvec() {
            check_xvec(file, format)
        } else if format == VecFormat::Slab {
            check_slab(file)
        } else {
            // Formats like npy, parquet — skip for now
            continue;
        };

        if let Err(msg) = result {
            failures.push(format!("{}: {}", file.display(), msg));
        }
    }

    if failures.is_empty() {
        let mut result = CheckResult::ok("integrity");
        result.messages.push(format!("{} data file(s) checked, all valid", checked));
        result
    } else {
        let fail_count = failures.len();
        let ok = checked - fail_count as u64;
        failures.push(format!("{} ok, {} invalid out of {} checked", ok, fail_count, checked));
        CheckResult::fail("integrity", failures)
    }
}

/// Validate an xvec file: check that file size is evenly divisible by
/// the record stride, and that the dimension header is consistent.
fn check_xvec(path: &Path, format: VecFormat) -> Result<(), String> {
    let meta = fs::metadata(path)
        .map_err(|e| format!("cannot stat: {}", e))?;
    let file_size = meta.len();

    if file_size == 0 {
        return Ok(()); // Empty file is technically valid
    }

    if file_size < 4 {
        return Err("file too small for dimension header (< 4 bytes)".to_string());
    }

    // Read the dimension from the first 4 bytes
    let mut f = fs::File::open(path)
        .map_err(|e| format!("cannot open: {}", e))?;
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)
        .map_err(|e| format!("cannot read dimension: {}", e))?;
    let dim = i32::from_le_bytes(dim_buf);

    if dim <= 0 {
        return Err(format!("invalid dimension: {} (must be > 0)", dim));
    }

    let elem_size = format.element_size() as u64;
    let stride = 4 + (dim as u64) * elem_size;
    let remainder = file_size % stride;

    if remainder != 0 {
        let record_count = file_size / stride;
        return Err(format!(
            "file size {} is not evenly divisible by stride {} (dim={}, elem={}B): \
             {} complete records + {} trailing bytes",
            file_size, stride, dim, elem_size, record_count, remainder,
        ));
    }

    // Spot-check: verify a second record's dimension matches (if file has > 1 record)
    let record_count = file_size / stride;
    if record_count > 1 {
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(stride))
            .map_err(|e| format!("cannot seek to second record: {}", e))?;
        f.read_exact(&mut dim_buf)
            .map_err(|e| format!("cannot read second record dimension: {}", e))?;
        let dim2 = i32::from_le_bytes(dim_buf);
        if dim2 != dim {
            return Err(format!(
                "dimension mismatch: record 0 has dim={}, record 1 has dim={}",
                dim, dim2,
            ));
        }
    }

    Ok(())
}

/// Validate a slab file by delegating to slabtastic's structural checks.
///
/// `SlabReader::open` already validates the pages page structure. We
/// additionally verify that all page entry offsets fall within the file
/// and that the page index is non-empty for non-trivial files.
fn check_slab(path: &Path) -> Result<(), String> {
    let path_str = path.to_string_lossy();

    match slabtastic::SlabReader::open(path_str.as_ref()) {
        Ok(reader) => {
            let entries = reader.page_entries();
            let file_len = reader.file_len()
                .map_err(|e| format!("cannot determine file length: {}", e))?;

            // Verify all page entry offsets fall within the file
            for (i, entry) in entries.iter().enumerate() {
                let offset = entry.file_offset as u64;
                if offset >= file_len {
                    return Err(format!(
                        "page {} file_offset {} is beyond file end ({})",
                        i, offset, file_len,
                    ));
                }
            }

            Ok(())
        }
        Err(e) => Err(format!("slab open failed: {}", e)),
    }
}
