// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use super::VecSource;
use crate::formats::VecFormat;

/// Opens an xvec source, dispatching to `MmapVectorReader` for
/// fvec/ivec (the canonical implementation) and falling back to a manual
/// stream reader for bvec/dvec/hvec/svec.
pub fn open_xvec(path: &Path, format: VecFormat) -> Result<Box<dyn VecSource>, String> {
    match format {
        VecFormat::Fvec => open_fvec(path),
        VecFormat::Ivec => open_ivec(path),
        VecFormat::Bvec | VecFormat::Dvec | VecFormat::Hvec | VecFormat::Svec => {
            RawXvecReader::open(path, format)
        }
        _ => Err(format!("{} is not an xvec format", format)),
    }
}

// -- fvec via vectordata -------------------------------------------------------

/// Reads `.fvec` files using `MmapVectorReader<f32>` (mmap-based).
struct FvecReader {
    readers: Vec<MmapVectorReader<f32>>,
    current_reader_idx: usize,
    current_index: usize,
    dimension: u32,
    record_count: u64,
}

fn open_fvec(path: &Path) -> Result<Box<dyn VecSource>, String> {
    let files = collect_xvec_files(path, "fvec")?;
    let mut readers = Vec::with_capacity(files.len());
    let mut total: u64 = 0;
    let mut dim: Option<u32> = None;

    for f in &files {
        let r = MmapVectorReader::<f32>::open_fvec(f)
            .map_err(|e| format!("Failed to open {}: {}", f.display(), e))?;
        if let Some(d) = dim {
            if r.dim() as u32 != d {
                return Err(format!(
                    "Dimension mismatch in {}: expected {}, got {}",
                    f.display(),
                    d,
                    r.dim()
                ));
            }
        } else {
            dim = Some(r.dim() as u32);
        }
        total += r.count() as u64;
        readers.push(r);
    }

    let dimension = dim.ok_or("No fvec files found")?;
    Ok(Box::new(FvecReader {
        readers,
        current_reader_idx: 0,
        current_index: 0,
        dimension,
        record_count: total,
    }))
}

impl VecSource for FvecReader {
    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn element_size(&self) -> usize {
        4
    }

    fn record_count(&self) -> Option<u64> {
        Some(self.record_count)
    }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        loop {
            if self.current_reader_idx >= self.readers.len() {
                return None;
            }
            let reader = &self.readers[self.current_reader_idx];
            if self.current_index < reader.count() {
                let vec = reader.get(self.current_index).ok()?;
                self.current_index += 1;
                // Convert f32 slice to raw LE bytes
                let bytes: Vec<u8> = vec.iter().flat_map(|f: &f32| f.to_le_bytes()).collect();
                return Some(bytes);
            }
            self.current_reader_idx += 1;
            self.current_index = 0;
        }
    }
}

// -- ivec via vectordata -------------------------------------------------------

/// Reads `.ivec` files using `MmapVectorReader<i32>` (mmap-based).
struct IvecReader {
    readers: Vec<MmapVectorReader<i32>>,
    current_reader_idx: usize,
    current_index: usize,
    dimension: u32,
    record_count: u64,
}

fn open_ivec(path: &Path) -> Result<Box<dyn VecSource>, String> {
    let files = collect_xvec_files(path, "ivec")?;
    let mut readers = Vec::with_capacity(files.len());
    let mut total: u64 = 0;
    let mut dim: Option<u32> = None;

    for f in &files {
        let r = MmapVectorReader::<i32>::open_ivec(f)
            .map_err(|e| format!("Failed to open {}: {}", f.display(), e))?;
        if let Some(d) = dim {
            if r.dim() as u32 != d {
                return Err(format!(
                    "Dimension mismatch in {}: expected {}, got {}",
                    f.display(),
                    d,
                    r.dim()
                ));
            }
        } else {
            dim = Some(r.dim() as u32);
        }
        total += r.count() as u64;
        readers.push(r);
    }

    let dimension = dim.ok_or("No ivec files found")?;
    Ok(Box::new(IvecReader {
        readers,
        current_reader_idx: 0,
        current_index: 0,
        dimension,
        record_count: total,
    }))
}

impl VecSource for IvecReader {
    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn element_size(&self) -> usize {
        4
    }

    fn record_count(&self) -> Option<u64> {
        Some(self.record_count)
    }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        loop {
            if self.current_reader_idx >= self.readers.len() {
                return None;
            }
            let reader = &self.readers[self.current_reader_idx];
            if self.current_index < reader.count() {
                let vec = reader.get(self.current_index).ok()?;
                self.current_index += 1;
                // Convert i32 slice to raw LE bytes
                let bytes: Vec<u8> = vec.iter().flat_map(|i: &i32| i.to_le_bytes()).collect();
                return Some(bytes);
            }
            self.current_reader_idx += 1;
            self.current_index = 0;
        }
    }
}

// -- bvec/dvec/hvec/svec manual reader -----------------------------------------

/// Reads xvec formats not yet supported by `vectordata` (bvec, dvec, hvec, svec).
///
/// Wire format per record: `[dimension: i32 LE][elements: dim * element_size bytes]`
struct RawXvecReader {
    readers: Vec<BufReader<File>>,
    current_reader_idx: usize,
    dimension: u32,
    element_size: usize,
    record_count: Option<u64>,
}

impl RawXvecReader {
    fn open(path: &Path, format: VecFormat) -> Result<Box<dyn VecSource>, String> {
        let element_size = format.element_size();
        let ext = format.name();
        let mut files = collect_xvec_files(path, ext)?;

        // Read dimension from first record of first file
        let mut first = BufReader::new(
            File::open(&files[0])
                .map_err(|e| format!("Failed to open {}: {}", files[0].display(), e))?,
        );
        let dim = first
            .read_i32::<LittleEndian>()
            .map_err(|e| format!("Failed to read dimension: {}", e))?;
        if dim <= 0 {
            return Err(format!("Invalid dimension: {}", dim));
        }
        let dimension = dim as u32;

        // Calculate total record count from file sizes
        let record_size = 4 + (dimension as u64) * (element_size as u64);
        let total_records: u64 = files
            .iter()
            .filter_map(|f| fs::metadata(f).ok())
            .map(|m| m.len() / record_size)
            .sum();

        // Re-open all files (first one included, since we consumed part of it)
        drop(first);
        let readers: Vec<BufReader<File>> = files
            .drain(..)
            .map(|f| {
                BufReader::new(
                    File::open(&f)
                        .unwrap_or_else(|e| panic!("Failed to open {}: {}", f.display(), e)),
                )
            })
            .collect();

        Ok(Box::new(RawXvecReader {
            readers,
            current_reader_idx: 0,
            dimension,
            element_size,
            record_count: Some(total_records),
        }))
    }
}

impl VecSource for RawXvecReader {
    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn element_size(&self) -> usize {
        self.element_size
    }

    fn record_count(&self) -> Option<u64> {
        self.record_count
    }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        loop {
            if self.current_reader_idx >= self.readers.len() {
                return None;
            }
            let reader = &mut self.readers[self.current_reader_idx];

            match reader.read_i32::<LittleEndian>() {
                Ok(dim) => {
                    if dim != self.dimension as i32 {
                        panic!(
                            "Dimension mismatch: expected {}, got {}",
                            self.dimension, dim
                        );
                    }
                    let data_len = self.dimension as usize * self.element_size;
                    let mut buf = vec![0u8; data_len];
                    reader.read_exact(&mut buf).ok()?;
                    return Some(buf);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    self.current_reader_idx += 1;
                    continue;
                }
                Err(_) => return None,
            }
        }
    }
}

// -- shared helpers ------------------------------------------------------------

/// Collect xvec files from a path (single file or directory), sorted
fn collect_xvec_files(path: &Path, ext: &str) -> Result<Vec<std::path::PathBuf>, String> {
    let files: Vec<_> = if path.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.ends_with(ext) || name.ends_with(&format!("{}s", ext))
            })
            .map(|e| e.path())
            .collect();
        entries.sort();
        entries
    } else {
        vec![path.to_path_buf()]
    };

    if files.is_empty() {
        Err(format!("No {} files found in {}", ext, path.display()))
    } else {
        Ok(files)
    }
}
