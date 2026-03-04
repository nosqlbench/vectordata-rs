// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! # Slabtastic
//!
//! A streamable, random-accessible, appendable data layout format for
//! non-uniform data by ordinal.
//!
//! ## File capacity
//!
//! Slabtastic supports files of up to 2^63 bytes. All file-level offsets
//! are twos-complement signed 8-byte little-endian integers.
//!
//! ## Format overview
//!
//! A slabtastic file (conventionally using the `.slab` extension — see
//! [`SLAB_EXTENSION`]) is a sequence of **pages** followed by a trailing
//! **pages page** (the index). Each page has the wire layout:
//!
//! ```text
//! [magic:4 "SLAB"][page_size:4][record data...][offsets:(n+1)*4][footer:16]
//! ```
//!
//! The footer encodes the starting ordinal, record count, page size, page
//! type, namespace index, and footer length — all in 16 little-endian bytes.
//! The header and footer both carry the page size, enabling both forward
//! and backward traversal of the file without the index.
//!
//! The **pages page** is always the last page in the file. Its records are
//! `(start_ordinal:8, file_offset:8)` tuples sorted by ordinal, enabling
//! O(log₂ n) binary-search lookup of any ordinal to its containing data
//! page.
//!
//! ## Reading
//!
//! [`SlabReader`] supports four access modes:
//!
//! - **Point get** — fetch a single record by ordinal.
//! - **Batched iteration** — [`SlabReader::batch_iter`] yields records in
//!   configurable-size batches via [`SlabBatchIter`].
//! - **Sink read** — [`SlabReader::read_all_to_sink`] streams all records
//!   to any [`std::io::Write`] sink. For background execution with
//!   progress polling, use [`SlabReader::read_to_sink_async`] which
//!   returns a [`SlabTask`].
//! - **Multi-batch concurrent read** —
//!   [`SlabReader::multi_batch_get`] submits multiple independent batch
//!   read requests for concurrent execution using scoped threads.
//!   Results are returned in submission order as [`BatchReadResult`]
//!   values with partial success for missing ordinals.
//!
//! ## Writing
//!
//! [`SlabWriter`] supports single-record, bulk, and async write modes:
//!
//! - **Single** — [`SlabWriter::add_record`] appends one record.
//! - **Bulk** — [`SlabWriter::add_records`] appends a slice of records.
//! - **Async from iterator** — [`SlabWriter::write_from_iter_async`]
//!   spawns a background thread and returns a pollable [`SlabTask`].
//!
//! ## Append-only semantics
//!
//! New data pages can be appended and a new pages page written without
//! modifying any existing page. The last pages page in the file is always
//! authoritative; earlier pages pages become logically dead.
//!
//! ## Sparse ordinals
//!
//! Ordinal ranges need not be contiguous. A file may have gaps between
//! pages (e.g. ordinals 0–99 and 200–299 with nothing in between).
//! This coarse chunk-level sparsity supports step-wise incremental
//! changes without rewriting existing pages. Requesting a missing
//! ordinal returns [`SlabError::OrdinalNotFound`].
//!
//! ## Interior mutation
//!
//! While not the primary use case, interior records can be mutated in
//! place when the replacement data fits within the existing record
//! boundaries — e.g. self-terminating formats (null-terminated strings)
//! or fixed-size values (32-bit integers). For more substantial
//! revisions, append a new data page and rewrite the pages page to
//! reference it instead of the original.
//!
//! ## Quick-start example
//!
//! ```rust,no_run
//! use slabtastic::{SlabWriter, SlabReader, WriterConfig};
//!
//! # fn main() -> slabtastic::Result<()> {
//! // Write
//! let config = WriterConfig::default();
//! let mut w = SlabWriter::new("demo.slab", config)?;
//! w.add_record(b"hello")?;
//! w.add_record(b"world")?;
//! w.finish()?;
//!
//! // Read
//! let mut r = SlabReader::open("demo.slab")?;
//! assert_eq!(r.get(0)?, b"hello");
//! assert_eq!(r.get(1)?, b"world");
//! # Ok(())
//! # }
//! ```

pub mod cli;
pub mod config;
pub mod constants;
pub mod error;
pub mod footer;
pub mod namespaces_page;
pub mod page;
pub mod pages_page;
pub mod reader;
pub mod task;
pub mod writer;

pub use config::WriterConfig;
pub use constants::{PageType, SLAB_EXTENSION};
pub use error::{Result, SlabError};
pub use footer::Footer;
pub use namespaces_page::{NamespaceEntry, NamespacesPage};
pub use page::Page;
pub use pages_page::{PageEntry, PagesPage};
pub use reader::{BatchReadResult, SlabBatchIter, SlabReader};
pub use task::{SlabProgress, SlabTask};
pub use writer::SlabWriter;
