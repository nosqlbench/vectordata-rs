// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Slabtastic file writer.
//!
//! The writer accumulates records into an in-memory page and flushes it
//! to disk when the preferred page size is reached. Call
//! [`SlabWriter::finish`] to flush any remaining records and write the
//! trailing pages page (index).
//!
//! ## Write modes
//!
//! - **Single** — [`SlabWriter::add_record`] appends one record at a time.
//! - **Bulk** — [`SlabWriter::add_records`] appends a slice of records in
//!   one call, flushing pages as needed.
//! - **Async from iterator** — [`SlabWriter::write_from_iter_async`]
//!   spawns a background thread that consumes an iterator of records,
//!   writes them to a new file, and provides a [`SlabTask`]
//!   handle for progress polling.
//!
//! ## Flush-at-boundaries requirement
//!
//! The writer only issues writes of complete, serialized pages — it
//! never writes a partial page buffer. However, `write_all` does **not**
//! guarantee OS-level atomicity: a concurrent reader may observe a
//! partially-written page on disk. Readers must therefore validate each
//! candidate page's `[magic][size]` header against the observed file
//! size before reading the page body. See the
//! [concurrency model](../docs/explanation/concurrency.md) for details.
//!
//! ## Record-too-large
//!
//! In v1, a single record that exceeds the configured maximum page
//! capacity is rejected with [`SlabError::RecordTooLarge`].
//! There is no multi-page spanning for individual records.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::config::WriterConfig;
use crate::constants::{DEFAULT_NAMESPACE_INDEX, FOOTER_V1_SIZE, HEADER_SIZE, PageType};
use crate::error::{Result, SlabError};
use crate::footer::Footer;
use crate::namespaces_page::NamespacesPage;
use crate::page::Page;
use crate::pages_page::PagesPage;
use crate::task::{self, SlabTask};

/// Writes slabtastic files, accumulating records into pages and flushing
/// them according to the configured page size thresholds.
///
/// ## Write modes
///
/// - **Single** — [`add_record`](Self::add_record) appends one record.
/// - **Bulk** — [`add_records`](Self::add_records) appends a slice of
///   records in one call, flushing pages as needed.
/// - **Async** — [`write_from_iter_async`](Self::write_from_iter_async)
///   spawns a background thread that writes records from an iterator and
///   returns a pollable [`SlabTask`].
///
/// ## Lifecycle
///
/// 1. Create with [`SlabWriter::new`] (fresh file) or
///    [`SlabWriter::append`] (extend existing file).
/// 2. Call [`add_record`](Self::add_record) or
///    [`add_records`](Self::add_records) to append records. Pages are
///    flushed automatically when the preferred page size is reached.
/// 3. Call [`finish`](Self::finish) to flush the last page and write the
///    pages page.
///
/// ## Examples
///
/// ```rust,no_run
/// use slabtastic::{SlabWriter, WriterConfig};
///
/// # fn main() -> slabtastic::Result<()> {
/// let mut w = SlabWriter::new("out.slab", WriterConfig::default())?;
/// w.add_record(b"first")?;
/// w.add_record(b"second")?;
/// w.finish()?;
/// # Ok(())
/// # }
/// ```
pub struct SlabWriter {
    file: File,
    config: WriterConfig,
    current_page: Page,
    pages_page: PagesPage,
    file_offset: u64,
    next_ordinal: i64,
    page_start_ordinal: i64,
}

impl SlabWriter {
    /// Create a new slabtastic file at the given path.
    pub fn new<P: AsRef<Path>>(path: P, config: WriterConfig) -> Result<Self> {
        let file = File::create(path)?;
        Ok(SlabWriter {
            file,
            config,
            current_page: Page::new(0, PageType::Data),
            pages_page: PagesPage::new(),
            file_offset: 0,
            next_ordinal: 0,
            page_start_ordinal: 0,
        })
    }

    /// Open an existing slabtastic file for appending to the default namespace.
    ///
    /// Reads the existing pages page, then positions after the last data
    /// page so new data pages can be appended before a new pages page.
    pub fn append<P: AsRef<Path>>(path: P, config: WriterConfig) -> Result<Self> {
        Self::append_namespace(path, config, None)
    }

    /// Open an existing slabtastic file for appending to a specific namespace.
    ///
    /// When `namespace_name` is `None`, appends to the default namespace
    /// (index 1, name `""`). When `Some(name)`, finds the named namespace
    /// in the namespaces page and appends to it.
    ///
    /// For single-namespace files, specifying a non-default namespace name
    /// is an error.
    pub fn append_namespace<P: AsRef<Path>>(
        path: P,
        config: WriterConfig,
        namespace_name: Option<&str>,
    ) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Read the last 16 bytes to find the pages page footer
        let file_len = file.seek(SeekFrom::End(0))?;
        if file_len < (HEADER_SIZE + FOOTER_V1_SIZE) as u64 {
            return Err(SlabError::TruncatedPage {
                expected: HEADER_SIZE + FOOTER_V1_SIZE,
                actual: file_len as usize,
            });
        }

        file.seek(SeekFrom::End(-(FOOTER_V1_SIZE as i64)))?;
        let mut footer_buf = [0u8; FOOTER_V1_SIZE];
        file.read_exact(&mut footer_buf)?;
        let footer = Footer::read_from(&footer_buf)?;

        if footer.page_type != PageType::Pages && footer.page_type != PageType::Namespaces {
            return Err(SlabError::InvalidPageType(footer.page_type as u8));
        }

        // If the last page is a namespaces page, locate the target
        // namespace's pages page. Otherwise read the pages page directly.
        let old_pages_page = if footer.page_type == PageType::Namespaces {
            let ns_page_offset = file_len - footer.page_size as u64;
            file.seek(SeekFrom::Start(ns_page_offset))?;
            let mut ns_buf = vec![0u8; footer.page_size as usize];
            file.read_exact(&mut ns_buf)?;
            let ns_page = NamespacesPage::deserialize(&ns_buf)?;
            let ns_entries = ns_page.entries()?;

            let target_entry = if let Some(name) = namespace_name {
                ns_entries.iter().find(|e| e.name == name).cloned()
            } else {
                ns_entries
                    .iter()
                    .find(|e| {
                        e.namespace_index == DEFAULT_NAMESPACE_INDEX && e.name.is_empty()
                    })
                    .cloned()
            };

            let pp_offset = match target_entry {
                Some(entry) => entry.pages_page_offset,
                None => {
                    let available: Vec<String> = ns_entries
                        .iter()
                        .map(|e| {
                            if e.name.is_empty() {
                                format!("  index {}: (default)", e.namespace_index)
                            } else {
                                format!("  index {}: '{}'", e.namespace_index, e.name)
                            }
                        })
                        .collect();
                    let ns_desc = if let Some(name) = namespace_name {
                        format!("namespace '{name}' not found")
                    } else {
                        "no default namespace found".to_string()
                    };
                    return Err(SlabError::InvalidFooter(format!(
                        "{}. Available namespaces:\n{}",
                        ns_desc,
                        available.join("\n")
                    )));
                }
            };

            file.seek(SeekFrom::Start(pp_offset as u64))?;
            let mut hdr = [0u8; HEADER_SIZE];
            file.read_exact(&mut hdr)?;
            let pp_size = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
            file.seek(SeekFrom::Start(pp_offset as u64))?;
            let mut pages_buf = vec![0u8; pp_size];
            file.read_exact(&mut pages_buf)?;
            PagesPage::deserialize(&pages_buf)?
        } else {
            // Single-namespace file
            if let Some(name) = namespace_name
                && !name.is_empty() {
                    return Err(SlabError::InvalidFooter(format!(
                        "namespace '{}' not found; this is a single-namespace file",
                        name
                    )));
                }
            let pages_page_offset = file_len - footer.page_size as u64;
            file.seek(SeekFrom::Start(pages_page_offset))?;
            let mut pages_buf = vec![0u8; footer.page_size as usize];
            file.read_exact(&mut pages_buf)?;
            PagesPage::deserialize(&pages_buf)?
        };

        // Determine the next ordinal from existing entries
        let entries = old_pages_page.entries();
        let mut max_ordinal: i64 = 0;
        let mut data_end: u64 = 0;

        for entry in &entries {
            // Read each data page's footer to find its record count
            file.seek(SeekFrom::Start(entry.file_offset as u64))?;
            let mut hdr = [0u8; HEADER_SIZE];
            file.read_exact(&mut hdr)?;
            let page_size = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);

            // Read footer of this data page
            let footer_pos = entry.file_offset as u64 + page_size as u64 - FOOTER_V1_SIZE as u64;
            file.seek(SeekFrom::Start(footer_pos))?;
            let mut dfb = [0u8; FOOTER_V1_SIZE];
            file.read_exact(&mut dfb)?;
            let data_footer = Footer::read_from(&dfb)?;

            let page_end_ordinal =
                data_footer.start_ordinal + data_footer.record_count as i64;
            if page_end_ordinal > max_ordinal {
                max_ordinal = page_end_ordinal;
            }

            let page_end = entry.file_offset as u64 + page_size as u64;
            if page_end > data_end {
                data_end = page_end;
            }
        }

        // Position at the end of the last data page (before the old pages page)
        file.seek(SeekFrom::Start(data_end))?;

        // Build a new pages page carrying forward existing entries
        let mut pages_page = PagesPage::new();
        for entry in &entries {
            pages_page.add_entry(entry.start_ordinal, entry.file_offset);
        }

        Ok(SlabWriter {
            file,
            config,
            current_page: Page::new(max_ordinal, PageType::Data),
            pages_page,
            file_offset: data_end,
            next_ordinal: max_ordinal,
            page_start_ordinal: max_ordinal,
        })
    }

    /// Add a record with a specific starting ordinal.
    ///
    /// Records are accumulated into the current page. When the page
    /// reaches the preferred size threshold, it is automatically flushed.
    pub fn add_record(&mut self, data: &[u8]) -> Result<()> {
        // Check if this single record would exceed max page size
        let single_record_page_size =
            HEADER_SIZE + data.len() + (2 * 4) + FOOTER_V1_SIZE;
        if single_record_page_size > self.config.max_page_size as usize {
            return Err(SlabError::RecordTooLarge {
                record_size: data.len(),
                max_size: self.config.max_page_size as usize - HEADER_SIZE - 8 - FOOTER_V1_SIZE,
            });
        }

        // Check if adding this record would exceed preferred size
        self.current_page.add_record(data);
        self.next_ordinal += 1;
        let projected_size = self.current_page.serialized_size();

        if projected_size >= self.config.preferred_page_size as usize
            && self.current_page.record_count() > 0
        {
            self.flush_page()?;
        }

        Ok(())
    }

    /// Add a record, verifying that its ordinal matches the next expected
    /// ordinal.
    ///
    /// This is identical to [`add_record`](Self::add_record) except that
    /// the caller specifies the ordinal they expect. If `ordinal` does not
    /// equal [`next_ordinal`](Self::next_ordinal), the call fails with
    /// [`SlabError::OrdinalMismatch`] and nothing is written.
    pub fn add_record_at(&mut self, ordinal: i64, data: &[u8]) -> Result<()> {
        if ordinal != self.next_ordinal {
            return Err(SlabError::OrdinalMismatch {
                expected: self.next_ordinal,
                actual: ordinal,
            });
        }
        self.add_record(data)
    }

    /// Add multiple records in one call, flushing pages as needed.
    ///
    /// This is semantically equivalent to calling [`add_record`](Self::add_record)
    /// for each element but avoids per-call overhead.
    pub fn add_records(&mut self, records: &[&[u8]]) -> Result<()> {
        for &data in records {
            self.add_record(data)?;
        }
        Ok(())
    }

    /// Add multiple records, verifying that the first record's ordinal
    /// matches the next expected ordinal.
    ///
    /// If `start_ordinal` does not equal [`next_ordinal`](Self::next_ordinal),
    /// the call fails with [`SlabError::OrdinalMismatch`] and nothing is
    /// written. On success, records are assigned ordinals
    /// `start_ordinal .. start_ordinal + records.len()`.
    pub fn add_records_at(&mut self, start_ordinal: i64, records: &[&[u8]]) -> Result<()> {
        if start_ordinal != self.next_ordinal {
            return Err(SlabError::OrdinalMismatch {
                expected: self.next_ordinal,
                actual: start_ordinal,
            });
        }
        self.add_records(records)
    }

    /// Spawn a background thread that creates a new slabtastic file at
    /// `path`, writes all records from `iter`, calls `on_complete` when
    /// done, and returns a [`SlabTask<u64>`] whose result is the number
    /// of records written.
    ///
    /// Progress can be polled via [`SlabTask::progress`]. If the total
    /// number of records is not known in advance, `total` will remain
    /// zero until the iterator is exhausted.
    pub fn write_from_iter_async<I, F>(
        path: PathBuf,
        config: WriterConfig,
        iter: I,
        on_complete: F,
    ) -> SlabTask<u64>
    where
        I: Iterator<Item = Vec<u8>> + Send + 'static,
        F: FnOnce(u64) + Send + 'static,
    {
        let (progress, tracker) = task::new_progress();
        let handle = std::thread::spawn(move || {
            let mut writer = SlabWriter::new(&path, config)?;
            let mut count: u64 = 0;
            for data in iter {
                writer.add_record(&data)?;
                count += 1;
                tracker.inc();
            }
            tracker.set_total(count);
            writer.finish()?;
            tracker.mark_done();
            on_complete(count);
            Ok(count)
        });
        task::new_task(handle, progress)
    }

    /// Flush the current page to the file.
    pub fn flush_page(&mut self) -> Result<()> {
        if self.current_page.record_count() == 0 {
            return Ok(());
        }

        let mut bytes = self.current_page.serialize();

        // Apply alignment padding if configured
        let aligned = self.config.aligned_size(bytes.len());
        if aligned > bytes.len() {
            let raw_len = bytes.len();
            bytes.resize(aligned, 0);
            // Update page_size in header and footer to reflect padded size
            let new_size = aligned as u32;
            bytes[4..8].copy_from_slice(&new_size.to_le_bytes());
            // Rewrite footer at the END of the aligned buffer
            let mut footer = self.current_page.footer.clone();
            footer.page_size = new_size;
            footer.record_count = self.current_page.records.len() as u32;
            let mut footer_buf = [0u8; FOOTER_V1_SIZE];
            footer.write_to(&mut footer_buf);
            // Move footer to end of aligned buffer
            // First, zero out old footer location
            let old_footer_start = raw_len - FOOTER_V1_SIZE;
            for b in &mut bytes[old_footer_start..raw_len] {
                *b = 0;
            }
            bytes[aligned - FOOTER_V1_SIZE..aligned].copy_from_slice(&footer_buf);
            // Also move the offsets right before the new footer
            let record_count = self.current_page.records.len();
            let offsets_size = (record_count + 1) * 4;
            let old_offsets_start = old_footer_start - offsets_size;
            let new_offsets_start = aligned - FOOTER_V1_SIZE - offsets_size;
            if old_offsets_start != new_offsets_start {
                let offsets_data: Vec<u8> =
                    bytes[old_offsets_start..old_offsets_start + offsets_size].to_vec();
                // Zero out old location
                for b in &mut bytes[old_offsets_start..old_offsets_start + offsets_size] {
                    *b = 0;
                }
                bytes[new_offsets_start..new_offsets_start + offsets_size]
                    .copy_from_slice(&offsets_data);
            }
        }

        self.file.write_all(&bytes)?;

        // Record this page in the pages page
        self.pages_page
            .add_entry(self.page_start_ordinal, self.file_offset as i64);

        self.file_offset += bytes.len() as u64;

        // Start a new current page
        self.page_start_ordinal = self.next_ordinal;
        self.current_page = Page::new(self.next_ordinal, PageType::Data);

        Ok(())
    }

    /// Return the ordinal that will be assigned to the next record added.
    pub fn next_ordinal(&self) -> i64 {
        self.next_ordinal
    }

    /// Finish writing: flush the pending page and write the pages page.
    pub fn finish(&mut self) -> Result<()> {
        self.flush_page()?;

        // Serialize and write the pages page
        let pages_bytes = self.pages_page.serialize();
        self.file.write_all(&pages_bytes)?;
        self.file.flush()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::NamedTempFile;

    /// Write two records and verify the resulting file is structurally
    /// valid: non-empty, and the last 16 bytes parse as a Pages-type
    /// footer (confirming the pages page was written correctly).
    #[test]
    fn test_writer_creates_valid_file() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let config = WriterConfig::default();
        let mut writer = SlabWriter::new(&path, config).unwrap();
        writer.add_record(b"hello").unwrap();
        writer.add_record(b"world").unwrap();
        writer.finish().unwrap();

        // Verify file is non-empty and ends with a pages page footer
        let mut file = File::open(&path).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        assert!(data.len() > HEADER_SIZE + FOOTER_V1_SIZE);

        // Last 16 bytes should be a valid pages page footer
        let footer = Footer::read_from(&data[data.len() - FOOTER_V1_SIZE..]).unwrap();
        assert_eq!(footer.page_type, PageType::Pages);
    }

    /// Bulk add_records produces the same file contents as sequential
    /// add_record calls.
    #[test]
    fn test_add_records_matches_sequential() {
        use crate::reader::SlabReader;

        let tmp_bulk = NamedTempFile::new().unwrap();
        let path_bulk = tmp_bulk.path().to_path_buf();

        let tmp_seq = NamedTempFile::new().unwrap();
        let path_seq = tmp_seq.path().to_path_buf();

        let data: Vec<Vec<u8>> = (0..10)
            .map(|i| format!("item-{i}").into_bytes())
            .collect();
        let refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();

        // Bulk
        let config = WriterConfig::default();
        let mut w = SlabWriter::new(&path_bulk, config.clone()).unwrap();
        w.add_records(&refs).unwrap();
        w.finish().unwrap();

        // Sequential
        let mut w = SlabWriter::new(&path_seq, config).unwrap();
        for d in &data {
            w.add_record(d).unwrap();
        }
        w.finish().unwrap();

        // Both files should yield identical records
        let r_bulk = SlabReader::open(&path_bulk).unwrap();
        let r_seq = SlabReader::open(&path_seq).unwrap();
        let all_bulk = r_bulk.iter().unwrap();
        let all_seq = r_seq.iter().unwrap();
        assert_eq!(all_bulk, all_seq);
    }

    /// write_from_iter_async completes and progress reaches done.
    #[test]
    fn test_write_from_iter_async() {
        use crate::reader::SlabReader;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let records: Vec<Vec<u8>> = (0..20)
            .map(|i| format!("async-{i}").into_bytes())
            .collect();

        let callback_called = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&callback_called);

        let task = SlabWriter::write_from_iter_async(
            path.clone(),
            WriterConfig::default(),
            records.clone().into_iter(),
            move |_count| {
                cb.store(true, Ordering::Release);
            },
        );

        let count = task.wait().unwrap();
        assert_eq!(count, 20);
        assert!(callback_called.load(Ordering::Acquire));

        // Verify records
        let reader = SlabReader::open(&path).unwrap();
        for (i, expected) in records.iter().enumerate() {
            assert_eq!(reader.get(i as i64).unwrap(), *expected);
        }
    }

    /// add_record_at succeeds when the ordinal matches next_ordinal.
    #[test]
    fn test_add_record_at_correct_ordinal() {
        use crate::reader::SlabReader;

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();
        assert_eq!(w.next_ordinal(), 0);
        w.add_record_at(0, b"zero").unwrap();
        assert_eq!(w.next_ordinal(), 1);
        w.add_record_at(1, b"one").unwrap();
        w.add_record_at(2, b"two").unwrap();
        w.finish().unwrap();

        let r = SlabReader::open(&path).unwrap();
        assert_eq!(r.get(0).unwrap(), b"zero");
        assert_eq!(r.get(1).unwrap(), b"one");
        assert_eq!(r.get(2).unwrap(), b"two");
    }

    /// add_record_at fails with OrdinalMismatch when the ordinal is wrong.
    #[test]
    fn test_add_record_at_wrong_ordinal() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();
        w.add_record(b"first").unwrap();

        match w.add_record_at(5, b"wrong") {
            Err(SlabError::OrdinalMismatch {
                expected: 1,
                actual: 5,
            }) => {}
            other => panic!("expected OrdinalMismatch, got {other:?}"),
        }

        // Writer state is unchanged — next_ordinal is still 1
        assert_eq!(w.next_ordinal(), 1);
        w.add_record_at(1, b"correct").unwrap();
        w.finish().unwrap();
    }

    /// add_records_at succeeds when start_ordinal matches.
    #[test]
    fn test_add_records_at_correct_ordinal() {
        use crate::reader::SlabReader;

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();
        w.add_record(b"zero").unwrap();
        w.add_records_at(1, &[b"one", b"two", b"three"]).unwrap();
        assert_eq!(w.next_ordinal(), 4);
        w.finish().unwrap();

        let r = SlabReader::open(&path).unwrap();
        assert_eq!(r.get(0).unwrap(), b"zero");
        assert_eq!(r.get(1).unwrap(), b"one");
        assert_eq!(r.get(3).unwrap(), b"three");
    }

    /// add_records_at fails with OrdinalMismatch when start_ordinal is wrong.
    #[test]
    fn test_add_records_at_wrong_ordinal() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();

        match w.add_records_at(10, &[b"a", b"b"]) {
            Err(SlabError::OrdinalMismatch {
                expected: 0,
                actual: 10,
            }) => {}
            other => panic!("expected OrdinalMismatch, got {other:?}"),
        }

        // Nothing was written
        assert_eq!(w.next_ordinal(), 0);
    }
}
