// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use slabtastic::{SlabBatchIter, SlabReader as SlabtasticReader};

use super::VecSource;

/// Reads a `.slab` file containing vector data using the `slabtastic` crate.
///
/// Uses `batch_iter` for streaming page-at-a-time reads, avoiding loading
/// the entire file into memory at once.
pub struct SlabReader {
    batch_iter: SlabBatchIter,
    current_batch: Vec<(i64, Vec<u8>)>,
    current_idx: usize,
    dimension: u32,
    record_count: Option<u64>,
}

impl SlabReader {
    /// Open a slab file for reading
    pub fn open(path: &Path) -> Result<Box<dyn VecSource>, String> {
        let reader = SlabtasticReader::open(path)
            .map_err(|e| format!("Failed to open slab {}: {}", path.display(), e))?;

        let entries = reader.page_entries();
        if entries.is_empty() {
            return Err("Empty slab file".to_string());
        }

        // Determine dimension from the first record
        let first_ordinal = entries
            .iter()
            .min_by_key(|e| e.start_ordinal)
            .unwrap()
            .start_ordinal;
        let first_record = reader
            .get(first_ordinal)
            .map_err(|e| format!("Failed to read first record: {}", e))?;
        // Assume f32 elements (4 bytes each) as default
        let dimension = (first_record.len() / 4) as u32;

        // Get total record count by reading each page's metadata
        let mut total_records: u64 = 0;
        for entry in &entries {
            let page = reader
                .read_data_page(entry)
                .map_err(|e| format!("Failed to read page: {}", e))?;
            total_records += page.record_count() as u64;
        }

        // Use batch_iter for streaming reads (page-at-a-time, not all-in-memory)
        let batch_iter = reader.batch_iter(4096);

        Ok(Box::new(SlabReader {
            batch_iter,
            current_batch: Vec::new(),
            current_idx: 0,
            dimension,
            record_count: Some(total_records),
        }))
    }
}

impl VecSource for SlabReader {
    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn element_size(&self) -> usize {
        4 // slab vectors default to f32 element size
    }

    fn record_count(&self) -> Option<u64> {
        self.record_count
    }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        loop {
            if self.current_idx < self.current_batch.len() {
                let (_ordinal, data) = &self.current_batch[self.current_idx];
                self.current_idx += 1;
                return Some(data.clone());
            }

            // Fetch next batch
            match self.batch_iter.next_batch() {
                Ok(batch) if batch.is_empty() => return None,
                Ok(batch) => {
                    self.current_batch = batch;
                    self.current_idx = 0;
                }
                Err(_) => return None,
            }
        }
    }

}
