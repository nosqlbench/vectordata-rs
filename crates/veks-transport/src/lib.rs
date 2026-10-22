// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! # veks-transport — remote access and streaming storage
//!
//! Every byte the workspace moves over a network goes through this
//! crate: one pooled blocking HTTP client ([`Transport`]) with its TLS
//! trust, offline gate, user agent, timeouts, retry policy and read
//! credentials; range reads and whole-file transfers ([`HttpTransport`],
//! [`Transport::fetch_text`], [`Transport::fetch_if_modified`]); uploads
//! ([`upload`]); AWS Signature Version 4 request signing ([`sigv4`]);
//! and the cache copies remote files are read through —
//! merkle-verified ([`VerifiedFile`], over [`CachedChannel`]) or
//! recorded as fetched ([`ChunkedFile`], over [`ChunkStore`]) — each a
//! veks-apicore [`ByteSource`](veks_apicore::ByteSource)
//! ([`RemoteSource`], [`RangeSource`]) that veks-io's readers read
//! (`docs/design/srd-crate-boundaries.md` §6).
//!
//! It knows no dataset, catalog or profile. It reads no environment
//! variable and no settings file: a [`TransportConfig`] value configures
//! it (TR-3). It computes no cache path: a [`CachePlacement`] says where
//! each copy and sidecar lives (TR-7). It does not depend on veks-io;
//! the two meet only through veks-apicore's `ByteSource` (TR-8).
//!
//! ## The seams
//!
//! | Contract or value | Defined in | Implemented by | Consumed here by |
//! |---|---|---|---|
//! | [`TransportConfig`], [`TlsTrust`] | veks-apicore | built by vectordata from its settings and environment, or by any program | [`Transport::new`] |
//! | [`Credentials`] | veks-apicore | vectordata (its credential store), any program; [`NoCredentials`] | every request that carries no authorisation of its own |
//! | [`RequestSigner`] | veks-apicore | [`sigv4::SigV4Signer`] here, any program (a signature scheme) | [`Transport::send`] |
//! | [`Request`] | veks-apicore | — (a value) | [`Transport::send`], answered by [`Response`] |
//! | [`TransportError`] | veks-apicore | — (a value) | every failed request |
//! | [`ByteSource`](veks_apicore::ByteSource) | veks-apicore | [`RemoteSource`], [`RangeSource`] here | veks-io's readers, vectordata's storage |
//! | [`DownloadProgress`] | veks-apicore | — (shared counters) | the chunk stores' progress callbacks |
//!
//! [`ChunkedTransport`] is this crate's own: the chunk stores fetch
//! through it, and nothing outside implements or consumes it (AC-2).
//!
//! ## Reading a remote file
//!
//! ```no_run
//! use veks_apicore::transport::TransportConfig;
//! use veks_apicore::ByteSource;
//! use veks_transport::{CachePlacement, RemoteSource, Transport};
//!
//! # fn main() -> std::io::Result<()> {
//! let transport = Transport::new(TransportConfig::default());
//! let placement = CachePlacement {
//!     root: "/tmp/cache/ds".into(),
//!     data: "/tmp/cache/ds/base.fvecs".into(),
//!     merkle_state: "/tmp/cache/ds/base.fvecs.mrkl".into(),
//!     chunk_map: "/tmp/cache/ds/base.fvecs.chunks".into(),
//! };
//! let source = RemoteSource::open(&transport, "https://data.example/ds/base.fvecs", &placement)?;
//! let header = source.read_bytes(0, 4)?; // fetches the first chunk only
//! source.precache()?; // the rest, verified when a .mref is published
//! assert!(source.mmap_base().is_some());
//! # let _ = header; Ok(()) }
//! ```
//!
//! ## Cost
//!
//! Hot paths dispatch statically: a reader is generic over its source,
//! and the cache files' per-read calls ([`VerifiedFile::mmap_slice`],
//! [`ChunkedFile::read_bytes`], …) are `#[inline]` so they inline into
//! the crate that reads through them (there is no LTO). Errors are built
//! only on the error path; progress is reported per chunk, never per
//! read (`srd-crate-boundaries.md` §4.1).

#![warn(missing_docs)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod chunk_store;
mod chunked;
mod client;
mod files;
mod http;
pub mod merkle;
mod placement;
mod pump;
mod retry;
pub mod sigv4;
pub mod upload;
mod verified;

pub use chunk_store::{ChunkStore, ChunksSidecar, DEFAULT_CHUNK_SIZE, complete_size_on_disk, read_sidecar};
pub use chunked::{ChunkRequest, ChunkedTransport, fetch_chunks_parallel};
pub use client::{Response, Transport, is_remote_url, normalize_remote_url};
pub use files::{ChunkedFile, RangeSource, RemoteFile, RemoteSource, VerifiedFile, fetch_reference, fetch_sibling};
pub use http::HttpTransport;
pub use placement::CachePlacement;
pub use retry::RetryPolicy;
pub use verified::CachedChannel;

pub use veks_apicore::transport::{
    Authorization, Body, Credentials, DEFAULT_PARALLEL_STREAMS, Method, NoCredentials, Request, RequestSigner, Timeouts,
    TlsTrust, TransportConfig,
};
pub use veks_apicore::{DownloadProgress, TransportError, TransportErrorKind};
