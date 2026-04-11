// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Streaming reader for scalar formats.

use std::io::{BufReader, Read};
use std::path::Path;

use crate::format::VecFormat;
use crate::traits::{SourceMeta, VecSource};

/// Open a scalar file for streaming reading.
pub fn open(path: &Path, format: VecFormat) -> Result<Box<dyn VecSource>, String> {
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} has no element size", format));
    }
    let file = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let file_len = file.metadata()
        .map_err(|e| format!("stat {}: {}", path.display(), e))?
        .len() as usize;
    if file_len % elem_size != 0 {
        return Err(format!(
            "{}: file size {} not divisible by element size {}",
            path.display(), file_len, elem_size
        ));
    }
    let count = file_len / elem_size;
    Ok(Box::new(ScalarReader {
        reader: BufReader::with_capacity(4 * 1024 * 1024, file),
        element_size: elem_size,
        total: count,
        read: 0,
    }))
}

/// Probe a scalar file for metadata.
pub fn probe(path: &Path, format: VecFormat) -> Result<SourceMeta, String> {
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} has no element size", format));
    }
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("stat {}: {}", path.display(), e))?;
    let file_len = meta.len() as usize;
    if file_len % elem_size != 0 {
        return Err(format!(
            "{}: file size {} not divisible by element size {}",
            path.display(), file_len, elem_size
        ));
    }
    Ok(SourceMeta {
        dimension: 1,
        element_size: elem_size,
        record_count: Some((file_len / elem_size) as u64),
    })
}

struct ScalarReader {
    reader: BufReader<std::fs::File>,
    element_size: usize,
    total: usize,
    read: usize,
}

impl VecSource for ScalarReader {
    fn dimension(&self) -> u32 { 1 }
    fn element_size(&self) -> usize { self.element_size }
    fn record_count(&self) -> Option<u64> { Some(self.total as u64) }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        if self.read >= self.total {
            return None;
        }
        let mut buf = vec![0u8; self.element_size];
        match self.reader.read_exact(&mut buf) {
            Ok(()) => {
                self.read += 1;
                Some(buf)
            }
            Err(_) => None,
        }
    }
}
