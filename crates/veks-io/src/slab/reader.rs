// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Streaming [`VecSource`] over a `.slab` vector file.

use std::path::Path;

use slabtastic::{OpenProgress, SlabBatchIter, SlabReader as SlabtasticReader};
use veks_apicore::metadata_schema::{
    SCHEMA_KIND_VECTOR, SCHEMA_NAMESPACE, VectorSchema, sidecar_kind,
};

use crate::element::ElementType;
use crate::traits::{SourceMeta, VecSource};

/// Reads a `.slab` file containing vector data using the `slabtastic` crate.
///
/// Uses `batch_iter` for streaming page-at-a-time reads, avoiding loading
/// the entire file into memory at once.
///
/// The element type is the one the slab's schema sidecar declares (a
/// [`VectorSchema`], which veks-io's slab writer emits). A slab that
/// declares none — no sidecar, or a metadata or predicate one — reads
/// as bytes: [`element_type`](VecSource::element_type) is `None`,
/// [`element_size`](VecSource::element_size) is 1, the dimension counts
/// bytes, and opening it logs one warning saying so.
pub struct SlabReader {
    batch_iter: SlabBatchIter,
    current_batch: Vec<(i64, Vec<u8>)>,
    current_idx: usize,
    dimension: u32,
    element_type: Option<ElementType>,
    record_count: Option<u64>,
}

/// What a slab's schema sidecar declares about its records.
enum Declaration {
    /// A vector sidecar naming the element type.
    Declared(ElementType),
    /// No element type is declared, and why — for the warning.
    Undeclared(&'static str),
}

impl Declaration {
    fn element_type(&self) -> Option<ElementType> {
        match self {
            Self::Declared(ty) => Some(*ty),
            Self::Undeclared(_) => None,
        }
    }
}

/// Read the element type `path`'s `:schema` sidecar declares.
///
/// A vector sidecar that does not parse is an error: the declaration is
/// there, and reading the records as some other type would be a guess.
/// Anything else — no sidecar, one of another kind, one that is not
/// JSON — declares nothing.
fn declaration(path: &Path) -> Result<Declaration, String> {
    let namespaces = SlabtasticReader::list_namespaces(path)
        .map_err(|e| format!("Failed to list slab namespaces in {}: {e}", path.display()))?;
    if !namespaces.iter().any(|ns| ns.name == SCHEMA_NAMESPACE) {
        return Ok(Declaration::Undeclared("it has no schema sidecar"));
    }
    let reader = SlabtasticReader::open_namespace(path, Some(SCHEMA_NAMESPACE))
        .map_err(|e| format!("Failed to open the schema sidecar of {}: {e}", path.display()))?;
    if reader.total_records() == 0 {
        return Ok(Declaration::Undeclared("its schema sidecar is empty"));
    }
    let bytes = reader
        .get(0)
        .map_err(|e| format!("Failed to read the schema sidecar of {}: {e}", path.display()))?;
    let Ok(kind) = sidecar_kind(&bytes) else {
        return Ok(Declaration::Undeclared("its schema sidecar is not a JSON object"));
    };
    if kind != SCHEMA_KIND_VECTOR {
        return Ok(Declaration::Undeclared("its schema sidecar describes records, not vectors"));
    }
    let schema = VectorSchema::from_json_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Declaration::Declared(schema.element_type))
}

/// The dimension of a record of `record_len` bytes at `element_type`
/// (bytes when undeclared).
fn dimension_of(path: &Path, record_len: usize, element_type: Option<ElementType>) -> Result<u32, String> {
    let width = element_type.map_or(1, ElementType::byte_width);
    if !record_len.is_multiple_of(width) {
        return Err(format!(
            "{}: a record of {record_len} bytes is not a whole number of {} elements",
            path.display(),
            element_type.map_or("u8", ElementType::name),
        ));
    }
    u32::try_from(record_len / width)
        .map_err(|_| format!("{}: a record of {record_len} bytes is too long", path.display()))
}

impl SlabReader {
    /// Probe a slab file's shape without building its page index.
    ///
    /// Reads the pages page, at most two data page footers
    /// ([`slabtastic::SlabReader::probe`]) and the schema sidecar, and
    /// reports what [`open`](Self::open) would: the declared element
    /// type, or bytes when there is none. It logs nothing; the
    /// `element_type` of `None` is the signal.
    ///
    /// ```
    /// use veks_io::sink::{SinkConfig, open_sink};
    /// use veks_io::{ElementType, VecFormat};
    /// # let dir = tempfile::tempdir().unwrap();
    /// let path = dir.path().join("codes.slab");
    /// let config = SinkConfig {
    ///     dimension: 2,
    ///     element_type: Some(ElementType::U16),
    ///     slab_page_size: None,
    ///     slab_namespace: 0,
    ///     schema_sidecar: None,
    ///     max_shard_bytes: None,
    /// };
    /// let mut sink = open_sink(&path, VecFormat::Slab, &config).unwrap();
    /// sink.write_record(0, &[1, 0, 2, 0]); // two u16s
    /// sink.finish().unwrap();
    ///
    /// let meta = veks_io::slab::reader::SlabReader::probe(&path).unwrap();
    /// assert_eq!((meta.dimension, meta.element_size, meta.element_type), (2, 2, Some(ElementType::U16)));
    /// assert_eq!(meta.record_count, Some(1));
    /// ```
    pub fn probe(path: &Path) -> Result<SourceMeta, String> {
        let stats = SlabtasticReader::probe(path)
            .map_err(|e| format!("Failed to probe slab {}: {}", path.display(), e))?;
        if stats.page_count == 0 {
            return Err("Empty slab file".to_string());
        }
        let element_type = declaration(path)?.element_type();
        Ok(SourceMeta {
            dimension: dimension_of(path, stats.first_record_size, element_type)?,
            element_size: element_type.map_or(1, ElementType::byte_width),
            record_count: Some(stats.total_records),
            element_type,
        })
    }

    /// Open a slab file for reading, with progress output to stderr.
    ///
    /// A slab that declares no element type logs one warning naming
    /// the file and why, and reads as bytes.
    pub fn open(path: &Path) -> Result<Box<dyn VecSource>, String> {
        let reader = SlabtasticReader::open_with_progress(path, |p| {
            match p {
                OpenProgress::PagesPageRead { page_count } => {
                    log::info!("    slab index: {} pages to scan", page_count);
                }
                OpenProgress::IndexBuild { done, total } => {
                    log::info!("    slab index: {}/{} pages scanned", done, total);
                }
                OpenProgress::IndexComplete { total_records } => {
                    log::info!("    slab index: complete, {} total records", total_records);
                }
            }
        }).map_err(|e| format!("Failed to open slab {}: {}", path.display(), e))?;

        if reader.page_count() == 0 {
            return Err("Empty slab file".to_string());
        }

        let declaration = declaration(path)?;
        if let Declaration::Undeclared(why) = &declaration {
            log::warn!(
                "{}: element type undeclared ({why}); its records read as bytes",
                path.display()
            );
        }
        let element_type = declaration.element_type();

        // Determine dimension from the first record
        let entries = reader.page_entries();
        let first_ordinal = entries
            .iter()
            .min_by_key(|e| e.start_ordinal)
            .ok_or_else(|| format!("{}: a slab with pages lists no page entry", path.display()))?
            .start_ordinal;
        let first_record = reader
            .get(first_ordinal)
            .map_err(|e| format!("Failed to read first record: {}", e))?;
        let dimension = dimension_of(path, first_record.len(), element_type)?;

        // Total record count from cached page metadata (no extra I/O)
        let total_records = reader.total_records();

        // Use batch_iter for streaming reads (page-at-a-time, not all-in-memory)
        let batch_iter = reader.batch_iter(4096);

        Ok(Box::new(SlabReader {
            batch_iter,
            current_batch: Vec::new(),
            current_idx: 0,
            dimension,
            element_type,
            record_count: Some(total_records),
        }))
    }
}

impl VecSource for SlabReader {
    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn element_size(&self) -> usize {
        self.element_type.map_or(1, ElementType::byte_width)
    }

    fn element_type(&self) -> Option<ElementType> {
        self.element_type
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
