// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The vectordata API: what a program codes against to read vector
//! datasets, and what an implementation provides.
//!
//! This crate holds the contracts — the traits the workspace's crates
//! speak to one another through — and every value, descriptor and
//! error type their signatures name. It holds no implementation: no
//! file format, network, cache or dataset resolution. `vectordata`,
//! `veks-io` and `veks-transport` implement it, and re-export it at the
//! paths their users already know (`vectordata::TestDataView`,
//! `veks_io::VectorReader`, …). A program that needs only the
//! contracts, or a second implementation of one (a test double, another
//! program's byte source), depends on this crate alone
//! (`docs/design/srd-crate-boundaries.md` §4).
//!
//! ## The seams
//!
//! | Contract | What it is | Implemented by | Consumed by |
//! |---|---|---|---|
//! | [`ByteSource`] | where a reader's bytes come from | veks-io (`MappedFile`), vectordata (its storage) | veks-io's readers |
//! | [`VectorReader`], [`VvecReader`] | a file's records by ordinal at one element type | veks-io's readers; vectordata's windowed and sharded views | every reader of a dataset |
//! | [`VecSource`], [`VecSink`], [`VarlenSource`], [`VarlenSink`] | a file read or written front to back | veks-io, per format | veks-core, veks-pipeline, veks |
//! | [`ProgressSink`], [`DownloadProgress`] | how far long work has got | vectordata's meters, veks' UI, closures | every long phase |
//! | [`TestDataView`], [`TestDataViewExt`] | one profile of a dataset | vectordata | every dataset consumer |
//! | [`FacetBytes`] | one facet's bytes, wherever they live | vectordata | fetch planning, cache tools |
//! | [`FetchProgress`](fetch::FetchProgress) | a fetch's events | vectordata's meters, closures | [`TestDataView::fetch`] |
//! | [`CatalogSource`](catalog::CatalogSource) | a supply of catalog entries | vectordata (located, built) | vectordata's catalog resolution |
//! | [`RecordContainer`](records::RecordContainer), [`RecordCodec`](records::RecordCodec) | opaque records by ordinal, and their decoding | vectordata (slabs, codecs) | record facet readers |
//!
//! The vocabulary those contracts name is here too: the dataset
//! descriptor model ([`dataset`], [`model`]), locations ([`Location`]),
//! slab schema sidecars ([`metadata_schema`]), fetch plans and reports
//! ([`fetch`]), verification reports
//! ([`verify`]), attribute provenance ([`overlay`]), and the errors
//! ([`Error`], [`IoError`]). The element model is `veks-datatypes`',
//! re-exported here.
//!
//! ## Cost
//!
//! A contract costs nothing on a hot path: readers are generic over
//! their source, so a read is monomorphized for it; trait objects
//! appear at open time and around whole-range calls; progress is
//! reported per batch; an error is built only when there is one
//! (`srd-crate-boundaries.md` §4.1).

#![warn(missing_docs)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![doc(test(attr(deny(deprecated))))]

pub mod bytes;
pub mod catalog;
pub mod dataset;
pub mod error;
pub mod fetch;
pub mod location;
pub mod metadata_schema;
pub mod model;
pub mod overlay;
pub mod progress;
pub mod reader;
pub mod records;
pub mod source;
pub mod stream;
pub mod verify;
pub mod view;

pub use veks_datatypes::{
    Container, DistanceFunction, Element, ElementConversion, ElementKind, ElementType, FloatFormat,
    UnknownDistanceFunction, bf16, f16,
};

pub use bytes::{ContentId, ContentIdKind, FacetBytes};
pub use error::{Error, InsufficientCacheSpace, Result, TransportError, TransportErrorKind};
pub use location::Location;
pub use progress::{DownloadProgress, NoProgress, ProgressSink};
pub use reader::{AnyElement, AnyVectorReader, AnyVvecReader, ConvertingReader, DIM_UNDEFINED, IoError, VectorReader, VvecReader};
pub use source::{ByteSource, SourcePart};
pub use stream::{SourceMeta, VarlenRecord, VarlenSink, VarlenSource, VecSink, VecSource};
pub use view::{
    CacheStats, FacetDescriptor, MappedRange, PrefetchHandle, PrefetchPlan, PrefetchReport, QuantizedBase, RangeFill,
    ShardRange, TestDataView, TestDataViewExt, WholeFacetFallback,
};
