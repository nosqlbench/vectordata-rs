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
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::config::WriterConfig;
use crate::constants::{DEFAULT_NAMESPACE_INDEX, FOOTER_V1_SIZE, HEADER_SIZE, PageType};
use crate::error::{Result, SlabError};
use crate::footer::Footer;
use crate::namespaces_page::{NamespaceEntry, NamespacesPage};
use crate::page::Page;
use crate::pages_page::PagesPage;
use crate::task::{self, SlabTask};

/// The largest namespace index reserved for caller-defined namespaces.
///
/// The on-disk footer encodes the namespace index in one byte where
/// `0` is reserved as invalid, `1` is the default namespace `""`,
/// `2..=127` are user-defined, and `128..=255` are reserved. A single
/// file therefore supports at most 127 distinct namespaces (the
/// default plus 126 named).
const MAX_NAMESPACE_INDEX: u8 = 127;

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
/// 3. *(Optional)* Call [`start_namespace`](Self::start_namespace) to
///    seal the current namespace and begin a new one. Subsequent
///    records land in the new namespace; the file's pages page tail is
///    automatically promoted to a namespaces page at finish time.
/// 4. Call [`finish`](Self::finish) to flush the last page and write
///    the index trailer (pages page in single-namespace mode, namespaces
///    page in multi-namespace mode).
///
/// ## Examples
///
/// Single-namespace file:
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
///
/// Multi-namespace file (default + `"schema"`):
///
/// ```rust,no_run
/// use slabtastic::{SlabWriter, WriterConfig};
///
/// # fn main() -> slabtastic::Result<()> {
/// let mut w = SlabWriter::new("out.slab", WriterConfig::default())?;
/// w.add_record(b"content-record")?;       // default namespace
/// w.start_namespace("schema")?;
/// w.add_record(b"{\"version\":1}")?;     // \"schema\" namespace
/// w.finish()?;
/// # Ok(())
/// # }
/// ```
pub struct SlabWriter {
    /// File handle wrapped in a `BufWriter` so successive
    /// page-sized `write_all` calls (page bytes, pages-page
    /// trailer, namespaces-page trailer) coalesce into larger
    /// writev-friendly chunks. The buffer is sized to comfortably
    /// hold a typical preferred-page-size payload without
    /// fragmenting the syscall. `BufWriter`'s pass-through
    /// behaviour for writes larger than the buffer keeps oversize
    /// pages efficient too.
    file: BufWriter<File>,
    config: WriterConfig,
    current_page: Page,
    pages_page: PagesPage,
    file_offset: u64,
    next_ordinal: i64,
    page_start_ordinal: i64,
    /// Index assigned to the data pages and pages page currently being
    /// built. Starts at [`DEFAULT_NAMESPACE_INDEX`].
    current_namespace_index: u8,
    /// Name of the namespace currently being built (`""` for the default).
    current_namespace_name: String,
    /// Entries for namespaces that have already been sealed via
    /// [`SlabWriter::start_namespace`] (or carried forward from an
    /// existing multi-namespace file in
    /// [`SlabWriter::append_namespace`]). Non-empty means `finish` must
    /// emit a [`NamespacesPage`] trailer.
    sealed_namespaces: Vec<NamespaceEntry>,
}

impl SlabWriter {
    /// Create a new slabtastic file at the given path.
    ///
    /// The writer starts in the default namespace (index 1, name `""`).
    /// To produce a multi-namespace file, call
    /// [`start_namespace`](Self::start_namespace) between record batches
    /// — [`finish`](Self::finish) will then emit a
    /// [`NamespacesPage`] trailer instead of a bare pages page.
    pub fn new<P: AsRef<Path>>(path: P, config: WriterConfig) -> Result<Self> {
        // 8 MiB buffer: comfortably absorbs the typical 4 MiB
        // preferred-page payloads emitted on bulk-write paths, and
        // BufWriter passes through writes larger than the buffer
        // so oversized records stay zero-copy.
        const WRITE_BUF: usize = 8 * 1024 * 1024;
        let file = BufWriter::with_capacity(WRITE_BUF, File::create(path)?);
        Ok(SlabWriter {
            file,
            config,
            current_page: Page::new(0, PageType::Data),
            pages_page: PagesPage::new(),
            file_offset: 0,
            next_ordinal: 0,
            page_start_ordinal: 0,
            current_namespace_index: DEFAULT_NAMESPACE_INDEX,
            current_namespace_name: String::new(),
            sealed_namespaces: Vec::new(),
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
        // namespace's pages page and remember the sibling namespaces so
        // we can carry them forward through `finish`. Otherwise read
        // the pages page directly.
        let (old_pages_page, target_index, target_name, sealed_namespaces) =
            if footer.page_type == PageType::Namespaces {
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

                let target = match target_entry {
                    Some(entry) => entry,
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

                let pp_offset = target.pages_page_offset;
                file.seek(SeekFrom::Start(pp_offset as u64))?;
                let mut hdr = [0u8; HEADER_SIZE];
                file.read_exact(&mut hdr)?;
                let pp_size = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
                file.seek(SeekFrom::Start(pp_offset as u64))?;
                let mut pages_buf = vec![0u8; pp_size];
                file.read_exact(&mut pages_buf)?;
                let pp = PagesPage::deserialize(&pages_buf)?;

                let target_idx = target.namespace_index;
                let target_nm = target.name.clone();
                // Carry forward every sibling namespace verbatim — their
                // pages pages already live at known offsets and will
                // remain valid.
                let sealed: Vec<NamespaceEntry> = ns_entries
                    .into_iter()
                    .filter(|e| e.namespace_index != target_idx)
                    .collect();
                (pp, target_idx, target_nm, sealed)
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
                let pp = PagesPage::deserialize(&pages_buf)?;
                (pp, DEFAULT_NAMESPACE_INDEX, String::new(), Vec::new())
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

        // Build a new pages page carrying forward existing entries.
        // Tag the in-memory pages page and the initial data page with
        // the target namespace's index so newly emitted pages match.
        let mut pages_page = PagesPage::new();
        pages_page.page.footer.namespace_index = target_index;
        for entry in &entries {
            pages_page.add_entry(entry.start_ordinal, entry.file_offset);
        }
        let mut current_page = Page::new(max_ordinal, PageType::Data);
        current_page.footer.namespace_index = target_index;

        // BufWriter wrapping preserves the underlying File's seek
        // position, so writes resume at `data_end` (the position
        // we explicitly seeked to above). Same 8 MiB buffer as the
        // `new()` path so append matches create on write throughput.
        const WRITE_BUF: usize = 8 * 1024 * 1024;
        let file = BufWriter::with_capacity(WRITE_BUF, file);
        Ok(SlabWriter {
            file,
            config,
            current_page,
            pages_page,
            file_offset: data_end,
            next_ordinal: max_ordinal,
            page_start_ordinal: max_ordinal,
            current_namespace_index: target_index,
            current_namespace_name: target_name,
            sealed_namespaces,
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

    /// Add a record by taking ownership of its bytes — no clone.
    ///
    /// Behaviourally identical to [`add_record`](Self::add_record),
    /// but accepts the record as `Vec<u8>` so callers that already
    /// own the bytes avoid the per-record memcpy that
    /// [`Page::add_record`] performs. Use this on the hot path of
    /// bulk-write workloads (predicate-merge phases, segment-cache
    /// consolidation) where the caller has just produced the buffer
    /// and is about to drop it anyway.
    pub fn add_record_owned(&mut self, data: Vec<u8>) -> Result<()> {
        let single_record_page_size =
            HEADER_SIZE + data.len() + (2 * 4) + FOOTER_V1_SIZE;
        if single_record_page_size > self.config.max_page_size as usize {
            return Err(SlabError::RecordTooLarge {
                record_size: data.len(),
                max_size: self.config.max_page_size as usize - HEADER_SIZE - 8 - FOOTER_V1_SIZE,
            });
        }

        self.current_page.add_record_owned(data);
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
        self.current_page.footer.namespace_index = self.current_namespace_index;

        Ok(())
    }

    /// Return the ordinal that will be assigned to the next record added.
    pub fn next_ordinal(&self) -> i64 {
        self.next_ordinal
    }

    /// Return the name of the namespace currently being written.
    /// Empty string for the default namespace.
    pub fn current_namespace(&self) -> &str {
        &self.current_namespace_name
    }

    /// Seal the current namespace and begin a new one with `name`.
    ///
    /// The writer flushes the in-flight data page, emits the current
    /// namespace's pages page at the file's tail, records its entry,
    /// and rotates state so subsequent records land in a new namespace
    /// with the next available index. The eventual
    /// [`finish`](Self::finish) call will emit a [`NamespacesPage`]
    /// trailer covering every namespace.
    ///
    /// The first namespace (the one created implicitly by
    /// [`new`](Self::new) or selected by
    /// [`append_namespace`](Self::append_namespace)) is sealed by the
    /// first call to this method. Subsequent calls seal whichever
    /// namespace they just started.
    ///
    /// # Errors
    ///
    /// - [`SlabError::InvalidNamespace`] if `name` is empty (the empty
    ///   string is reserved for the default namespace).
    /// - [`SlabError::InvalidNamespace`] if a namespace with the same
    ///   name has already been sealed in this writer session.
    /// - [`SlabError::InvalidNamespace`] if the namespace index would
    ///   exceed `127`. A single file supports at most 127 namespaces
    ///   (the default plus 126 named).
    pub fn start_namespace(&mut self, name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(SlabError::InvalidNamespace(
                "namespace name must be non-empty (\"\" is reserved for the default)"
                    .to_string(),
            ));
        }
        if name == self.current_namespace_name
            || self.sealed_namespaces.iter().any(|e| e.name == name)
        {
            return Err(SlabError::InvalidNamespace(format!(
                "namespace '{name}' is already present in this writer"
            )));
        }
        if self.current_namespace_index >= MAX_NAMESPACE_INDEX {
            return Err(SlabError::InvalidNamespace(format!(
                "namespace index {} would exceed maximum {MAX_NAMESPACE_INDEX}",
                self.current_namespace_index as u16 + 1
            )));
        }

        // Seal the current namespace: flush in-flight records, then
        // write its pages page at the file tail.
        self.seal_current_namespace()?;

        // Rotate state for the new namespace.
        let next_index = self.current_namespace_index + 1;
        self.current_namespace_index = next_index;
        self.current_namespace_name = name.to_string();
        self.pages_page = PagesPage::new();
        self.pages_page.page.footer.namespace_index = next_index;
        self.next_ordinal = 0;
        self.page_start_ordinal = 0;
        self.current_page = Page::new(0, PageType::Data);
        self.current_page.footer.namespace_index = next_index;

        Ok(())
    }

    /// Flush the in-flight data page, serialize the current namespace's
    /// pages page, write it to the file tail, and record its entry in
    /// `sealed_namespaces`. Leaves the writer's namespace-rotation
    /// state untouched — callers (start_namespace, finish) own that.
    fn seal_current_namespace(&mut self) -> Result<()> {
        self.flush_page()?;
        self.pages_page.page.footer.namespace_index = self.current_namespace_index;
        let pp_offset = self.file_offset as i64;
        let pp_bytes = self.pages_page.serialize();
        self.file.write_all(&pp_bytes)?;
        self.file_offset += pp_bytes.len() as u64;
        self.sealed_namespaces.push(NamespaceEntry {
            namespace_index: self.current_namespace_index,
            name: self.current_namespace_name.clone(),
            pages_page_offset: pp_offset,
        });
        Ok(())
    }

    /// Finish writing: flush the pending page and write the trailing
    /// index pages.
    ///
    /// In single-namespace mode (no calls to
    /// [`start_namespace`](Self::start_namespace)) the trailer is just
    /// the [`PagesPage`]. When two or more namespaces have been written
    /// the trailer is a [`NamespacesPage`] following each namespace's
    /// own pages page, and the reader will route by name.
    pub fn finish(&mut self) -> Result<()> {
        if self.sealed_namespaces.is_empty() {
            // Single-namespace file: tag and emit the pages page as the
            // sole trailer.
            self.flush_page()?;
            self.pages_page.page.footer.namespace_index = self.current_namespace_index;
            let pages_bytes = self.pages_page.serialize();
            self.file.write_all(&pages_bytes)?;
            self.file.flush()?;
            return Ok(());
        }

        // Multi-namespace file: seal the namespace currently being
        // built, then emit a namespaces page mapping name → pages-page
        // offset for every namespace in the file.
        self.seal_current_namespace()?;
        let mut ns_page = NamespacesPage::new();
        for entry in &self.sealed_namespaces {
            ns_page.add_entry(entry);
        }
        let ns_bytes = ns_page.serialize();
        self.file.write_all(&ns_bytes)?;
        self.file_offset += ns_bytes.len() as u64;
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

    /// Writing default + one named namespace produces a file whose
    /// trailing footer is a Namespaces page, and the reader can open
    /// each namespace independently and read back its own records.
    #[test]
    fn test_start_namespace_roundtrip() {
        use crate::reader::SlabReader;

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();
        w.add_record(b"content-0").unwrap();
        w.add_record(b"content-1").unwrap();
        w.start_namespace("schema").unwrap();
        w.add_record(b"{\"version\":1}").unwrap();
        w.finish().unwrap();

        // Trailer must be a Namespaces page now.
        let bytes = std::fs::read(&path).unwrap();
        let tail =
            Footer::read_from(&bytes[bytes.len() - FOOTER_V1_SIZE..]).unwrap();
        assert_eq!(tail.page_type, PageType::Namespaces);

        // Default namespace reads its own records.
        let r_default = SlabReader::open(&path).unwrap();
        assert_eq!(r_default.get(0).unwrap(), b"content-0");
        assert_eq!(r_default.get(1).unwrap(), b"content-1");
        assert!(r_default.get(2).is_err());

        // Schema namespace reads independently with its own ordinal 0.
        let r_schema = SlabReader::open_namespace(&path, Some("schema")).unwrap();
        assert_eq!(r_schema.get(0).unwrap(), b"{\"version\":1}");
        assert!(r_schema.get(1).is_err());

        // The namespaces listing carries both entries.
        let ns = SlabReader::list_namespaces(&path).unwrap();
        let names: Vec<String> = ns.iter().map(|e| e.name.clone()).collect();
        assert!(names.iter().any(|n| n.is_empty()));
        assert!(names.iter().any(|n| n == "schema"));
    }

    /// Three namespaces (default + two named) all survive a write/read
    /// cycle, including a namespace that wasn't given any records.
    #[test]
    fn test_start_namespace_multiple_including_empty() {
        use crate::reader::SlabReader;

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();
        w.add_record(b"d-0").unwrap();
        w.start_namespace("alpha").unwrap();
        // No records — `alpha` is an empty namespace.
        w.start_namespace("beta").unwrap();
        w.add_record(b"b-0").unwrap();
        w.add_record(b"b-1").unwrap();
        w.finish().unwrap();

        let r_default = SlabReader::open(&path).unwrap();
        assert_eq!(r_default.get(0).unwrap(), b"d-0");

        let r_alpha = SlabReader::open_namespace(&path, Some("alpha")).unwrap();
        assert!(r_alpha.get(0).is_err()); // empty namespace

        let r_beta = SlabReader::open_namespace(&path, Some("beta")).unwrap();
        assert_eq!(r_beta.get(0).unwrap(), b"b-0");
        assert_eq!(r_beta.get(1).unwrap(), b"b-1");
    }

    /// `start_namespace("")` is rejected — the empty name is the
    /// default namespace and can't be created twice.
    #[test]
    fn test_start_namespace_empty_name_rejected() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();
        match w.start_namespace("") {
            Err(SlabError::InvalidNamespace(_)) => {}
            other => panic!("expected InvalidNamespace, got {other:?}"),
        }
    }

    /// Re-using a namespace name within one writer session is rejected.
    #[test]
    fn test_start_namespace_duplicate_name_rejected() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();
        w.start_namespace("schema").unwrap();
        match w.start_namespace("schema") {
            Err(SlabError::InvalidNamespace(_)) => {}
            other => panic!("expected InvalidNamespace, got {other:?}"),
        }
    }

    /// Tagging data pages with the current namespace_index — when a
    /// flush happens *across* a namespace boundary, the second
    /// namespace's pages carry index 2, not 1.
    #[test]
    fn test_start_namespace_tags_data_pages() {
        use crate::reader::SlabReader;

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut w = SlabWriter::new(&path, WriterConfig::default()).unwrap();
        w.add_record(b"a").unwrap();
        w.start_namespace("two").unwrap();
        w.add_record(b"b").unwrap();
        w.finish().unwrap();

        // Reader can find the right page for each namespace's ordinal 0
        // independently — the only way that works is if the data page
        // footer carried the correct namespace_index AND the per-
        // namespace pages page only references its own data pages.
        let r1 = SlabReader::open(&path).unwrap();
        assert_eq!(r1.get(0).unwrap(), b"a");
        let r2 = SlabReader::open_namespace(&path, Some("two")).unwrap();
        assert_eq!(r2.get(0).unwrap(), b"b");
    }
}
