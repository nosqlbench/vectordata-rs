// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Streaming writer for scalar formats.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::format::VecFormat;
use crate::traits::VecSink;

/// Open a scalar file for sequential writing.
///
/// The `dimension` parameter from `create()` is ignored for scalar formats
/// (always 1). Records are raw element bytes with no header.
pub fn open(path: &Path, format: VecFormat) -> Result<Box<dyn VecSink>, String> {
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} has no element size", format));
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = std::fs::File::create(path)
        .map_err(|e| format!("create {}: {}", path.display(), e))?;
    Ok(Box::new(ScalarWriter {
        writer: BufWriter::with_capacity(4 * 1024 * 1024, file),
        element_size: elem_size,
    }))
}

struct ScalarWriter {
    writer: BufWriter<std::fs::File>,
    element_size: usize,
}

impl VecSink for ScalarWriter {
    fn write_record(&mut self, _ordinal: i64, data: &[u8]) {
        // Write raw element bytes — no dimension header
        debug_assert!(data.len() == self.element_size,
            "scalar write: expected {} bytes, got {}", self.element_size, data.len());
        let _ = self.writer.write_all(data);
    }

    fn finish(self: Box<Self>) -> Result<(), String> {
        let mut inner = self;
        inner.writer.flush()
            .map_err(|e| format!("flush error: {}", e))
    }
}
