// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Slabtastic file reader.
//!
//! Opening a file reads the trailing pages page to build an in-memory
//! ordinal-to-offset index. Records can then be accessed in four modes:
//!
//! - **Point get** — [`SlabReader::get`] fetches a single record by ordinal.
//! - **Batched iteration** — [`SlabReader::batch_iter`] returns a
//!   [`SlabBatchIter`] that yields records in configurable-size batches,
//!   suitable for streaming pipelines.
//! - **Sink read** — [`SlabReader::read_all_to_sink`] writes all record
//!   data sequentially to any [`std::io::Write`] sink. For background
//!   execution with progress polling, use the associated function
//!   [`SlabReader::read_to_sink_async`].
//! - **Multi-batch concurrent read** —
//!   [`SlabReader::multi_batch_get`] submits multiple independent batch
//!   read requests for concurrent execution using scoped threads.
//!   Results are returned in submission order as [`BatchReadResult`]
//!   values, with `None` for missing ordinals (partial success).
//!
//! ## Performance
//!
//! The file is memory-mapped at open time. Point gets use interpolation
//! search over a pre-built page index with cached per-page metadata
//! (page_size, record_count), so a `get()` call performs:
//!
//! 1. **Interpolation search** — O(1) expected for uniform ordinal
//!    distributions (~1–2 probes vs ~12 for binary search).
//! 2. **Two memory loads** — offset-pair lookup from the mmap.
//! 3. **One memcpy** — record bytes from the mmap into the output buffer.
//!
//! Zero syscalls per get. All data is accessed directly through the mmap.
//!
//! ## Sparse ordinals
//!
//! Ordinal ranges need not be contiguous — a file may have gaps between
//! pages (e.g. ordinals 0–99 and 200–299 with nothing in between). This
//! coarse chunk-level sparsity supports step-wise incremental changes.
//! Requesting an ordinal that falls in a gap returns
//! [`SlabError::OrdinalNotFound`].
//!
//! ## Concurrent / incremental reading
//!
//! Multiple readers may open the same file concurrently, each with its
//! own file descriptor and mmap. A reader may also observe an
//! actively-written file incrementally by validating each page's
//! `[magic][size]` header before reading it. However, the reader must
//! not assume atomic writes; pages should only be read once their header
//! confirms they are fully written. This incremental mode is inherently
//! optimistic and should only be used when the writer is streaming an
//! immutable version of the data.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
#[cfg(unix)]
use memmap2::Advice;

use crate::constants::{DEFAULT_NAMESPACE_INDEX, FOOTER_V1_SIZE, HEADER_SIZE, PageType};
use crate::error::{Result, SlabError};
use crate::footer::Footer;
use crate::namespaces_page::{NamespaceEntry, NamespacesPage};
use crate::page::Page;
use crate::pages_page::{PageEntry, PagesPage};
use crate::task::{self, SlabTask};

/// The result of one batch within a multi-batch read.
///
/// Each requested ordinal has a corresponding entry in `records`,
/// in the same order as the input. Found records contain `Some(data)`;
/// missing ordinals contain `None`.
pub struct BatchReadResult {
    /// Ordinal–data pairs in submission order. `None` means the ordinal
    /// was not found.
    pub records: Vec<(i64, Option<Vec<u8>>)>,
}

impl BatchReadResult {
    /// Returns `true` if no records were found in this batch.
    pub fn is_empty(&self) -> bool {
        self.records.iter().all(|(_, data)| data.is_none())
    }

    /// Returns the number of records that were found.
    pub fn found_count(&self) -> usize {
        self.records.iter().filter(|(_, data)| data.is_some()).count()
    }

    /// Returns the number of ordinals that were not found.
    pub fn missing_count(&self) -> usize {
        self.records.iter().filter(|(_, data)| data.is_none()).count()
    }
}

/// Cached per-page metadata built at open time.
///
/// Stores the page geometry (size, record count) alongside the
/// pages-page entry so that [`SlabReader::get_into`] can locate a
/// record's bytes without any per-call I/O parsing.
#[derive(Debug, Clone, Copy)]
struct PageIndexEntry {
    /// Starting ordinal for this page.
    start_ordinal: i64,
    /// Byte offset of this page within the file.
    file_offset: u64,
    /// Total page size in bytes (from the page header).
    page_size: u32,
    /// Number of records in this page (from the page footer).
    record_count: u32,
}

/// Reads slabtastic files, supporting random access by ordinal,
/// batched iteration, streaming sink reads, and multi-batch concurrent
/// reads.
///
/// ## Read modes
///
/// - **Point get** — [`get`](Self::get) fetches a single record by ordinal.
/// - **Batched** — [`batch_iter`](Self::batch_iter) yields configurable-size
///   batches of `(ordinal, data)` pairs. An empty batch signals exhaustion.
/// - **Sink** — [`read_all_to_sink`](Self::read_all_to_sink) writes all
///   records to an [`std::io::Write`] sink. For background execution with
///   progress polling, see [`read_to_sink_async`](Self::read_to_sink_async).
/// - **Multi-batch** — [`multi_batch_get`](Self::multi_batch_get) submits
///   multiple independent batch read requests for concurrent execution
///   using scoped threads. Results are returned in submission order as
///   [`BatchReadResult`] values with partial success for missing ordinals.
///
/// ## Opening semantics
///
/// [`SlabReader::open`] memory-maps the file and reads the trailing pages
/// page to build the ordinal-to-offset index. Each data page's header and
/// footer are scanned once to cache page_size and record_count. If the
/// file is truncated or does not end with a pages page, an error is
/// returned.
///
/// ## Sparse ordinals
///
/// Requesting an ordinal that falls in a gap between pages returns
/// [`SlabError::OrdinalNotFound`].
///
/// ## Examples
///
/// ```rust,no_run
/// use slabtastic::SlabReader;
///
/// # fn main() -> slabtastic::Result<()> {
/// let mut r = SlabReader::open("data.slab")?;
/// let record = r.get(0)?;
/// println!("record 0: {} bytes", record.len());
///
/// let all = r.iter()?;
/// for (ordinal, data) in &all {
///     println!("ordinal {ordinal}: {} bytes", data.len());
/// }
/// # Ok(())
/// # }
/// ```
pub struct SlabReader {
    /// File handle, kept alive for the mmap and for [`SlabBatchIter`].
    file: File,
    /// Memory-mapped view of the entire file.
    mmap: Mmap,
    /// The deserialized pages page (index of all data pages).
    pages_page: PagesPage,
    /// Pre-built page index with cached per-page metadata, sorted by
    /// `start_ordinal`. Used by [`get_into`](Self::get_into) for
    /// zero-syscall, zero-parse record lookup.
    page_index: Vec<PageIndexEntry>,
}

impl SlabReader {
    /// Open a slabtastic file for reading using the default namespace.
    ///
    /// Memory-maps the file, reads the trailing page to build the
    /// ordinal-to-offset index, and pre-caches per-page metadata
    /// (page_size, record_count) for zero-syscall point gets.
    ///
    /// The last page must be either a pages page (type 1) for
    /// single-namespace files, or a namespaces page (type 3) for
    /// multi-namespace files. In the latter case the default namespace's
    /// pages page is located via the namespaces page.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_namespace(path, None)
    }

    /// Open a slabtastic file for reading, targeting a specific namespace.
    ///
    /// When `namespace_name` is `None`, the default namespace (index 1,
    /// name `""`) is used. For single-namespace files (pages-page
    /// terminal), this is always the case.
    ///
    /// When `namespace_name` is `Some(name)`, the namespaces page is
    /// searched for a matching entry. If no match is found, an error is
    /// returned listing the available namespaces.
    pub fn open_namespace<P: AsRef<Path>>(path: P, namespace_name: Option<&str>) -> Result<Self> {
        let file = File::open(path)?;

        // Safety: the file is opened read-only and we keep the File
        // handle alive for the lifetime of the Mmap.
        let mmap = unsafe { Mmap::map(&file).map_err(SlabError::from)? };

        // Hint the kernel to back this mapping with transparent huge pages
        // (2 MB) where possible, reducing TLB misses for large files.
        #[cfg(unix)]
        let _ = mmap.advise(Advice::HugePage);

        let file_len = mmap.len();

        if file_len < HEADER_SIZE + FOOTER_V1_SIZE {
            return Err(SlabError::TruncatedPage {
                expected: HEADER_SIZE + FOOTER_V1_SIZE,
                actual: file_len,
            });
        }

        // Read the last 16 bytes for the terminal page footer
        let footer = Footer::read_from(&mmap[file_len - FOOTER_V1_SIZE..])?;

        if footer.page_type != PageType::Pages && footer.page_type != PageType::Namespaces {
            return Err(SlabError::InvalidPageType(footer.page_type as u8));
        }

        let pages_page = if footer.page_type == PageType::Pages {
            // Single-namespace file — if user asked for a non-default
            // namespace, that's an error.
            if let Some(name) = namespace_name
                && !name.is_empty() {
                    return Err(SlabError::InvalidFooter(format!(
                        "namespace '{}' not found; this is a single-namespace file",
                        name
                    )));
                }
            let pp_start = file_len - footer.page_size as usize;
            PagesPage::deserialize(&mmap[pp_start..file_len])?
        } else {
            // Namespaces page — locate the target namespace's pages page
            let ns_start = file_len - footer.page_size as usize;
            let ns_page = Page::deserialize(&mmap[ns_start..file_len])?;
            let ns_page_obj = NamespacesPage { page: ns_page };
            let ns_entries = ns_page_obj.entries()?;

            let target_entry = if let Some(name) = namespace_name {
                // Find namespace by name
                ns_entries.iter().find(|e| e.name == name)
            } else {
                // Find default namespace (index DEFAULT_NAMESPACE_INDEX, name "")
                ns_entries.iter().find(|e| {
                    e.namespace_index == DEFAULT_NAMESPACE_INDEX && e.name.is_empty()
                })
            };

            let pp_offset = match target_entry {
                Some(entry) => entry.pages_page_offset as usize,
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

            // Read page_size from the pages page header
            let pp_size = u32::from_le_bytes(
                mmap[pp_offset + 4..pp_offset + 8].try_into().unwrap(),
            ) as usize;
            PagesPage::deserialize(&mmap[pp_offset..pp_offset + pp_size])?
        };

        // Build page index with cached per-page metadata.
        // Each data page's header and footer are validated here so that
        // get_into() can skip per-call validation entirely.
        let entries = pages_page.entries();
        let mut page_index = Vec::with_capacity(entries.len());
        for entry in &entries {
            let offset = entry.file_offset as usize;
            let page_size = u32::from_le_bytes(
                mmap[offset + 4..offset + 8].try_into().unwrap(),
            );
            if (page_size as usize) < HEADER_SIZE + FOOTER_V1_SIZE {
                return Err(SlabError::TruncatedPage {
                    expected: HEADER_SIZE + FOOTER_V1_SIZE,
                    actual: page_size as usize,
                }
                .with_context(Some(offset as u64), None, None));
            }
            let footer_start = offset + page_size as usize - FOOTER_V1_SIZE;
            let page_type = mmap[footer_start + 12];
            if page_type != PageType::Data as u8 {
                return Err(SlabError::InvalidPageType(page_type)
                    .with_context(Some(offset as u64), None, None));
            }
            let mut rc = [0u8; 4];
            rc[..3].copy_from_slice(&mmap[footer_start + 5..footer_start + 8]);
            let record_count = u32::from_le_bytes(rc);
            page_index.push(PageIndexEntry {
                start_ordinal: entry.start_ordinal,
                file_offset: entry.file_offset as u64,
                page_size,
                record_count,
            });
        }

        Ok(SlabReader {
            file,
            mmap,
            pages_page,
            page_index,
        })
    }

    /// List all namespaces in a slabtastic file without fully opening
    /// for reading.
    ///
    /// For single-namespace files (pages-page terminal), returns a single
    /// entry for the default namespace. For multi-namespace files,
    /// returns all entries from the namespaces page.
    pub fn list_namespaces<P: AsRef<Path>>(path: P) -> Result<Vec<NamespaceEntry>> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file).map_err(SlabError::from)? };
        let file_len = mmap.len();

        if file_len < HEADER_SIZE + FOOTER_V1_SIZE {
            return Err(SlabError::TruncatedPage {
                expected: HEADER_SIZE + FOOTER_V1_SIZE,
                actual: file_len,
            });
        }

        let footer = Footer::read_from(&mmap[file_len - FOOTER_V1_SIZE..])?;

        if footer.page_type == PageType::Pages {
            // Single-namespace file — return default namespace entry
            let pp_start = file_len - footer.page_size as usize;
            Ok(vec![NamespaceEntry {
                namespace_index: DEFAULT_NAMESPACE_INDEX,
                name: String::new(),
                pages_page_offset: pp_start as i64,
            }])
        } else if footer.page_type == PageType::Namespaces {
            let ns_start = file_len - footer.page_size as usize;
            let ns_page = NamespacesPage::deserialize(&mmap[ns_start..file_len])?;
            ns_page.entries()
        } else {
            Err(SlabError::InvalidPageType(footer.page_type as u8))
        }
    }

    /// Get a record by its ordinal value.
    ///
    /// Delegates to [`get_into`](Self::get_into) with a fresh buffer.
    /// For repeated lookups, prefer `get_ref` for zero-copy access or
    /// `get_into` with a reusable buffer to avoid per-call allocation.
    pub fn get(&self, ordinal: i64) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.get_into(ordinal, &mut buf)?;
        Ok(buf)
    }

    /// Get a zero-copy reference to a record's bytes in the mmap.
    ///
    /// Returns a slice that borrows directly from the memory-mapped
    /// file with no allocation or copy. This is the fastest access
    /// path — the only work is interpolation search + two offset loads.
    ///
    /// The returned slice is valid for the lifetime of the reader.
    pub fn get_ref(&self, ordinal: i64) -> Result<&[u8]> {
        let idx = self
            .find_page_interpolation(ordinal)
            .ok_or(SlabError::OrdinalNotFound(ordinal))?;
        let entry = &self.page_index[idx];

        let local_index = (ordinal - entry.start_ordinal) as usize;
        if local_index >= entry.record_count as usize {
            return Err(SlabError::OrdinalNotFound(ordinal).with_context(
                Some(entry.file_offset),
                None,
                Some(ordinal),
            ));
        }

        let page_start = entry.file_offset as usize;
        let page_size = entry.page_size as usize;
        let record_count = entry.record_count as usize;

        let offset_count = record_count + 1;
        let offsets_start = page_start + page_size - FOOTER_V1_SIZE - offset_count * 4;
        let entry_pos = offsets_start + local_index * 4;

        let rec_start = u32::from_le_bytes(
            self.mmap[entry_pos..entry_pos + 4].try_into().unwrap(),
        ) as usize;
        let rec_end = u32::from_le_bytes(
            self.mmap[entry_pos + 4..entry_pos + 8].try_into().unwrap(),
        ) as usize;

        Ok(&self.mmap[page_start + rec_start..page_start + rec_end])
    }

    /// Get a record by ordinal, writing into the provided buffer.
    ///
    /// Uses the pre-built page index (interpolation search + cached
    /// metadata) and the memory-mapped file for zero-syscall access:
    ///
    /// 1. **Interpolation search** — O(1) expected page lookup.
    /// 2. **Offset-pair load** — 8 bytes from the mmap.
    /// 3. **Record copy** — direct memcpy from the mmap.
    ///
    /// The buffer is resized to fit the record data; previous contents
    /// are overwritten. For zero-copy access, use [`get_ref`](Self::get_ref)
    /// instead.
    pub fn get_into(&self, ordinal: i64, buf: &mut Vec<u8>) -> Result<()> {
        let data = self.get_ref(ordinal)?;
        buf.clear();
        buf.extend_from_slice(data);
        Ok(())
    }

    /// Submit multiple batch read requests for concurrent execution.
    ///
    /// Each batch is a slice of ordinals to look up. All batches execute
    /// concurrently using scoped threads. Results are returned in the
    /// same order as the input batches, with each batch's records in
    /// the same order as the requested ordinals.
    ///
    /// Individual ordinals that are not found produce `None` in their
    /// position rather than failing the entire batch, enabling partial
    /// success.
    pub fn multi_batch_get(&self, batches: &[&[i64]]) -> Vec<BatchReadResult> {
        if batches.len() <= 1 {
            // Single batch or empty: skip thread spawning
            return batches
                .iter()
                .map(|batch| self.execute_batch(batch))
                .collect();
        }

        std::thread::scope(|s| {
            let handles: Vec<_> = batches
                .iter()
                .map(|batch| s.spawn(|| self.execute_batch(batch)))
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        })
    }

    /// Execute a single batch of ordinal lookups, returning partial results.
    fn execute_batch(&self, ordinals: &[i64]) -> BatchReadResult {
        let records = ordinals
            .iter()
            .map(|&ord| {
                let data = self.get(ord).ok();
                (ord, data)
            })
            .collect();
        BatchReadResult { records }
    }

    /// Interpolation search over the page index.
    ///
    /// For files with roughly uniform record counts per page (the common
    /// case), this finds the containing page in 1–2 probes instead of
    /// the ~12 iterations a binary search needs for ~3 000 pages. This
    /// subsumes Eytzinger-layout binary search, which would reduce cache
    /// misses but still require O(log n) probes.
    ///
    /// Falls back to binary search if the interpolation guess is off by
    /// more than a bounded scan distance.
    fn find_page_interpolation(&self, ordinal: i64) -> Option<usize> {
        let entries = &self.page_index;
        let len = entries.len();
        if len == 0 || ordinal < entries[0].start_ordinal {
            return None;
        }
        if len == 1 {
            return Some(0);
        }

        let first = entries[0].start_ordinal;
        let last = entries[len - 1].start_ordinal;

        // Interpolation guess assuming uniform ordinal distribution
        let guess = if last <= first {
            0
        } else {
            let frac = (ordinal - first) as f64 / (last - first) as f64;
            (frac * (len - 1) as f64).clamp(0.0, (len - 1) as f64) as usize
        };

        // Bounded linear scan from the guess (≤ MAX_SCAN steps).
        // For uniform distributions the guess is within ±1 of correct.
        const MAX_SCAN: usize = 4;

        if entries[guess].start_ordinal <= ordinal {
            // Guess is at or before target — scan right
            let limit = (guess + MAX_SCAN + 1).min(len);
            for i in guess..limit {
                if i + 1 >= len || entries[i + 1].start_ordinal > ordinal {
                    return Some(i);
                }
            }
        } else {
            // Guess overshot — scan left
            let start = guess.saturating_sub(MAX_SCAN);
            for i in (start..guess).rev() {
                if entries[i].start_ordinal <= ordinal
                    && (i + 1 >= len || entries[i + 1].start_ordinal > ordinal) {
                        return Some(i);
                    }
            }
        }

        // Fallback: standard binary search (handles non-uniform distributions)
        match entries.binary_search_by_key(&ordinal, |e| e.start_ordinal) {
            Ok(i) => Some(i),
            Err(0) => None,
            Err(i) => Some(i - 1),
        }
    }

    /// Check whether the file contains a record for the given ordinal.
    pub fn contains(&self, ordinal: i64) -> bool {
        self.get(ordinal).is_ok()
    }

    /// Return the number of data pages in this file.
    pub fn page_count(&self) -> usize {
        self.pages_page.entry_count()
    }

    /// Return the page entries (index) from the pages page.
    pub fn page_entries(&self) -> Vec<PageEntry> {
        self.pages_page.entries()
    }

    /// Iterate all records in ordinal order, yielding `(ordinal, data)` pairs.
    pub fn iter(&self) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut entries = self.pages_page.entries();
        // Sort by start_ordinal to ensure ordinal order
        entries.sort_by_key(|e| e.start_ordinal);

        let mut result = Vec::new();
        for entry in &entries {
            let page = self.read_data_page(entry)?;
            for i in 0..page.record_count() {
                let ordinal = page.start_ordinal() + i as i64;
                let data = page.get_record(i).unwrap().to_vec();
                result.push((ordinal, data));
            }
        }

        Ok(result)
    }

    /// Iterate records within the half-open ordinal range `[start, end)`,
    /// yielding `(ordinal, data)` pairs in ordinal order.
    ///
    /// Only pages that overlap the requested range are read; pages
    /// entirely outside the range are skipped. Within qualifying pages,
    /// only records whose ordinal falls in `[start, end)` are returned.
    pub fn iter_range(&self, start: i64, end: i64) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut entries = self.pages_page.entries();
        entries.sort_by_key(|e| e.start_ordinal);

        let mut result = Vec::new();
        for entry in &entries {
            // Find the cached page index entry to get the record count
            // without deserializing the page.
            let idx = match self.page_index.iter().position(|p| {
                p.file_offset == entry.file_offset as u64
            }) {
                Some(i) => i,
                None => continue,
            };
            let pi = &self.page_index[idx];
            let page_start = pi.start_ordinal;
            let page_end = page_start + pi.record_count as i64;

            // Skip pages entirely outside the range
            if page_end <= start || page_start >= end {
                continue;
            }

            let page = self.read_data_page(entry)?;
            for i in 0..page.record_count() {
                let ordinal = page.start_ordinal() + i as i64;
                if ordinal >= end {
                    break;
                }
                if ordinal >= start {
                    let data = page.get_record(i).unwrap().to_vec();
                    result.push((ordinal, data));
                }
            }
        }

        Ok(result)
    }

    /// Return the total file length in bytes.
    pub fn file_len(&self) -> Result<u64> {
        Ok(self.mmap.len() as u64)
    }

    /// Read and deserialize a page starting at the given byte offset.
    ///
    /// Reads directly from the mmap. Useful for forward traversal of
    /// the file without relying on the pages page index.
    ///
    /// Errors are enriched with the file offset for diagnostics.
    pub fn read_page_at_offset(&self, offset: u64) -> Result<Page> {
        let offset = offset as usize;
        let page_size = u32::from_le_bytes(
            self.mmap[offset + 4..offset + 8].try_into().unwrap(),
        ) as usize;

        Page::deserialize(&self.mmap[offset..offset + page_size]).map_err(|e| {
            e.with_context(Some(offset as u64), None, None)
        })
    }

    /// Consume this reader and return a [`SlabBatchIter`] that yields
    /// records in batches of up to `batch_size`.
    ///
    /// Each call to [`SlabBatchIter::next_batch`] returns up to
    /// `batch_size` `(ordinal, data)` pairs. An empty vector signals
    /// that all records have been consumed.
    pub fn batch_iter(self, batch_size: usize) -> SlabBatchIter {
        let mut entries = self.pages_page.entries();
        entries.sort_by_key(|e| e.start_ordinal);
        SlabBatchIter {
            file: self.file,
            entries,
            batch_size,
            page_idx: 0,
            record_idx: 0,
            current_page: None,
        }
    }

    /// Write all records (in ordinal order) to `sink`, returning the
    /// number of records written.
    ///
    /// Each record's raw bytes are written directly with no framing or
    /// length prefix.
    pub fn read_all_to_sink<W: Write>(&self, sink: &mut W) -> Result<u64> {
        let mut entries = self.pages_page.entries();
        entries.sort_by_key(|e| e.start_ordinal);

        let mut count: u64 = 0;
        for entry in &entries {
            let page = self.read_data_page(entry)?;
            for i in 0..page.record_count() {
                let data = page.get_record(i).unwrap();
                sink.write_all(data).map_err(SlabError::from)?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Spawn a background thread that reads all records from `path` and
    /// writes them to `sink`, calling `on_complete` when finished.
    ///
    /// Returns a [`SlabTask<u64>`] whose progress can be polled and
    /// whose result is the total number of records written.
    ///
    /// The `sink` and `on_complete` callback are moved into the
    /// background thread.
    pub fn read_to_sink_async<W, F>(
        path: PathBuf,
        mut sink: W,
        on_complete: F,
    ) -> SlabTask<u64>
    where
        W: Write + Send + 'static,
        F: FnOnce(u64) + Send + 'static,
    {
        let (progress, tracker) = task::new_progress();
        let handle = std::thread::spawn(move || {
            let reader = SlabReader::open(&path)?;
            let mut entries = reader.pages_page.entries();
            entries.sort_by_key(|e| e.start_ordinal);

            // Read all pages to compute total record count
            let mut total_records: u64 = 0;
            let mut pages: Vec<Page> = Vec::with_capacity(entries.len());
            for entry in &entries {
                let page = reader.read_data_page(entry)?;
                total_records += page.record_count() as u64;
                pages.push(page);
            }
            tracker.set_total(total_records);

            let mut count: u64 = 0;
            for page in &pages {
                for i in 0..page.record_count() {
                    let data = page.get_record(i).unwrap();
                    sink.write_all(data).map_err(SlabError::from)?;
                    count += 1;
                    tracker.inc();
                }
            }
            tracker.mark_done();
            on_complete(count);
            Ok(count)
        });
        task::new_task(handle, progress)
    }

    /// Read and deserialize a data page from the mmap.
    ///
    /// Errors are enriched with the file offset from the page entry so
    /// that callers see where in the file the problem occurred.
    pub fn read_data_page(&self, entry: &PageEntry) -> Result<Page> {
        let offset = entry.file_offset as usize;
        let page_size = u32::from_le_bytes(
            self.mmap[offset + 4..offset + 8].try_into().unwrap(),
        ) as usize;

        Page::deserialize(&self.mmap[offset..offset + page_size]).map_err(|e| {
            e.with_context(Some(entry.file_offset as u64), None, None)
        })
    }
}

/// Batched iterator over all records in a slabtastic file.
///
/// Created by [`SlabReader::batch_iter`]. Each call to
/// [`next_batch`](Self::next_batch) returns up to `batch_size` records
/// as `(ordinal, data)` pairs. An empty vector signals exhaustion —
/// per the design doc: "if the reader returns 0 then the requestor
/// should assume there are no more."
pub struct SlabBatchIter {
    file: File,
    entries: Vec<PageEntry>,
    batch_size: usize,
    page_idx: usize,
    record_idx: usize,
    current_page: Option<Page>,
}

impl SlabBatchIter {
    /// Return the next batch of up to `batch_size` records.
    ///
    /// Returns an empty vector when all records have been consumed.
    pub fn next_batch(&mut self) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut batch = Vec::with_capacity(self.batch_size);

        while batch.len() < self.batch_size {
            // Load the current page if needed
            if self.current_page.is_none() {
                if self.page_idx >= self.entries.len() {
                    break;
                }
                let entry = self.entries[self.page_idx];
                let page = self.read_data_page(&entry)?;
                self.current_page = Some(page);
                self.record_idx = 0;
            }

            let page = self.current_page.as_ref().unwrap();
            if self.record_idx >= page.record_count() {
                self.current_page = None;
                self.page_idx += 1;
                continue;
            }

            let ordinal = page.start_ordinal() + self.record_idx as i64;
            let data = page.get_record(self.record_idx).unwrap().to_vec();
            batch.push((ordinal, data));
            self.record_idx += 1;
        }

        Ok(batch)
    }

    /// Read and deserialize a data page (internal helper).
    fn read_data_page(&mut self, entry: &PageEntry) -> Result<Page> {
        self.file
            .seek(SeekFrom::Start(entry.file_offset as u64))?;

        let mut hdr = [0u8; HEADER_SIZE];
        self.file.read_exact(&mut hdr)?;
        let page_size =
            u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;

        self.file
            .seek(SeekFrom::Start(entry.file_offset as u64))?;
        let mut page_buf = vec![0u8; page_size];
        self.file.read_exact(&mut page_buf)?;

        Page::deserialize(&page_buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WriterConfig;
    use crate::writer::SlabWriter;
    use tempfile::NamedTempFile;

    /// Write 3 records, open with `SlabReader`, and verify each ordinal
    /// returns the correct data. Also confirms that requesting an
    /// ordinal beyond the last record produces an error.
    #[test]
    fn test_reader_basic() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let config = WriterConfig::default();
        let mut writer = SlabWriter::new(&path, config).unwrap();
        writer.add_record(b"alpha").unwrap();
        writer.add_record(b"beta").unwrap();
        writer.add_record(b"gamma").unwrap();
        writer.finish().unwrap();

        let reader = SlabReader::open(&path).unwrap();
        assert_eq!(reader.get(0).unwrap(), b"alpha");
        assert_eq!(reader.get(1).unwrap(), b"beta");
        assert_eq!(reader.get(2).unwrap(), b"gamma");
        assert!(reader.get(3).is_err());
    }

    /// Helper: write N records and return the temp path.
    fn write_test_file(n: usize) -> (tempfile::TempPath, Vec<Vec<u8>>) {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.into_temp_path();

        let config = WriterConfig::default();
        let mut writer = SlabWriter::new(&path, config).unwrap();
        let records: Vec<Vec<u8>> = (0..n)
            .map(|i| format!("rec-{i:04}").into_bytes())
            .collect();
        for rec in &records {
            writer.add_record(rec).unwrap();
        }
        writer.finish().unwrap();
        (path, records)
    }

    /// batch_iter with batch_size=1 yields one record per call.
    #[test]
    fn test_batch_iter_size_one() {
        let (path, records) = write_test_file(5);
        let reader = SlabReader::open(&path).unwrap();
        let mut iter = reader.batch_iter(1);

        for (i, expected) in records.iter().enumerate() {
            let batch = iter.next_batch().unwrap();
            assert_eq!(batch.len(), 1, "batch {i} should have 1 record");
            assert_eq!(&batch[0].1, expected);
        }
        let empty = iter.next_batch().unwrap();
        assert!(empty.is_empty(), "should be exhausted");
    }

    /// batch_iter with batch_size == total returns everything in one call.
    #[test]
    fn test_batch_iter_size_total() {
        let (path, records) = write_test_file(5);
        let reader = SlabReader::open(&path).unwrap();
        let mut iter = reader.batch_iter(5);

        let batch = iter.next_batch().unwrap();
        assert_eq!(batch.len(), 5);
        for (i, (ord, data)) in batch.iter().enumerate() {
            assert_eq!(*ord, i as i64);
            assert_eq!(data, &records[i]);
        }
        assert!(iter.next_batch().unwrap().is_empty());
    }

    /// batch_iter with batch_size > total returns all records in one call.
    #[test]
    fn test_batch_iter_size_larger_than_total() {
        let (path, records) = write_test_file(3);
        let reader = SlabReader::open(&path).unwrap();
        let mut iter = reader.batch_iter(100);

        let batch = iter.next_batch().unwrap();
        assert_eq!(batch.len(), 3);
        for (i, (_, data)) in batch.iter().enumerate() {
            assert_eq!(data, &records[i]);
        }
        assert!(iter.next_batch().unwrap().is_empty());
    }

    /// multi_batch_get with 3 batches returns correct data in order.
    #[test]
    fn test_multi_batch_basic() {
        let (path, records) = write_test_file(10);
        let reader = SlabReader::open(&path).unwrap();

        let batch0: Vec<i64> = vec![0, 1, 2];
        let batch1: Vec<i64> = vec![5, 6];
        let batch2: Vec<i64> = vec![8, 9];
        let results = reader.multi_batch_get(&[&batch0, &batch1, &batch2]);

        assert_eq!(results.len(), 3);
        // Batch 0
        assert_eq!(results[0].records.len(), 3);
        for (i, &ord) in batch0.iter().enumerate() {
            assert_eq!(results[0].records[i].0, ord);
            assert_eq!(results[0].records[i].1.as_deref(), Some(records[ord as usize].as_slice()));
        }
        // Batch 1
        assert_eq!(results[1].records.len(), 2);
        for (i, &ord) in batch1.iter().enumerate() {
            assert_eq!(results[1].records[i].0, ord);
            assert_eq!(results[1].records[i].1.as_deref(), Some(records[ord as usize].as_slice()));
        }
        // Batch 2
        assert_eq!(results[2].records.len(), 2);
        for (i, &ord) in batch2.iter().enumerate() {
            assert_eq!(results[2].records[i].0, ord);
            assert_eq!(results[2].records[i].1.as_deref(), Some(records[ord as usize].as_slice()));
        }
    }

    /// multi_batch_get with mix of valid and invalid ordinals returns
    /// None for missing and Some for found.
    #[test]
    fn test_multi_batch_partial_success() {
        let (path, records) = write_test_file(5);
        let reader = SlabReader::open(&path).unwrap();

        let batch: Vec<i64> = vec![0, 999, 2, -1, 4];
        let results = reader.multi_batch_get(&[&batch]);

        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.records.len(), 5);
        assert_eq!(r.records[0].1.as_deref(), Some(records[0].as_slice()));
        assert!(r.records[1].1.is_none()); // 999 not found
        assert_eq!(r.records[2].1.as_deref(), Some(records[2].as_slice()));
        assert!(r.records[3].1.is_none()); // -1 not found
        assert_eq!(r.records[4].1.as_deref(), Some(records[4].as_slice()));
        assert_eq!(r.found_count(), 3);
        assert_eq!(r.missing_count(), 2);
        assert!(!r.is_empty());
    }

    /// multi_batch_get with all-invalid ordinals returns is_empty() == true.
    #[test]
    fn test_multi_batch_empty_result() {
        let (path, _) = write_test_file(5);
        let reader = SlabReader::open(&path).unwrap();

        let batch: Vec<i64> = vec![100, 200, 300];
        let results = reader.multi_batch_get(&[&batch]);

        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
        assert_eq!(results[0].found_count(), 0);
        assert_eq!(results[0].missing_count(), 3);
    }

    /// multi_batch_get preserves submission order regardless of batch size.
    #[test]
    fn test_multi_batch_preserves_order() {
        let (path, records) = write_test_file(20);
        let reader = SlabReader::open(&path).unwrap();

        // Batches of different sizes — order must match submission order
        let b0: Vec<i64> = vec![19];
        let b1: Vec<i64> = (0..10).collect();
        let b2: Vec<i64> = vec![15, 16, 17, 18, 19];
        let b3: Vec<i64> = vec![5, 10];
        let results = reader.multi_batch_get(&[&b0, &b1, &b2, &b3]);

        assert_eq!(results.len(), 4);

        // Verify each result corresponds to the correct batch
        assert_eq!(results[0].records.len(), 1);
        assert_eq!(results[0].records[0].0, 19);
        assert_eq!(results[0].records[0].1.as_deref(), Some(records[19].as_slice()));

        assert_eq!(results[1].records.len(), 10);
        for i in 0..10 {
            assert_eq!(results[1].records[i].0, i as i64);
        }

        assert_eq!(results[2].records.len(), 5);
        assert_eq!(results[2].records[0].0, 15);

        assert_eq!(results[3].records.len(), 2);
        assert_eq!(results[3].records[0].0, 5);
        assert_eq!(results[3].records[1].0, 10);
    }

    /// multi_batch_get with a single batch uses the inline optimization path.
    #[test]
    fn test_multi_batch_single_batch() {
        let (path, records) = write_test_file(5);
        let reader = SlabReader::open(&path).unwrap();

        let batch: Vec<i64> = vec![0, 2, 4];
        let results = reader.multi_batch_get(&[&batch]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].found_count(), 3);
        assert_eq!(results[0].records[0].1.as_deref(), Some(records[0].as_slice()));
        assert_eq!(results[0].records[1].1.as_deref(), Some(records[2].as_slice()));
        assert_eq!(results[0].records[2].1.as_deref(), Some(records[4].as_slice()));
    }

    /// read_all_to_sink writes correct data to a Vec<u8> sink.
    #[test]
    fn test_read_all_to_sink() {
        let (path, records) = write_test_file(4);
        let reader = SlabReader::open(&path).unwrap();
        let mut sink: Vec<u8> = Vec::new();
        let count = reader.read_all_to_sink(&mut sink).unwrap();
        assert_eq!(count, 4);

        // Sink should contain all record bytes concatenated
        let expected: Vec<u8> = records.iter().flat_map(|r| r.iter().copied()).collect();
        assert_eq!(sink, expected);
    }
}
