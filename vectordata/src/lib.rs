//! # Vector Data Tools
//!
//! A Rust library for reading and describing vector datasets defined by
//! `dataset.yaml` configuration files. Supports local file system access
//! (memory-mapped) and remote access via HTTP Range requests.
//!
//! ## Key modules
//!
//! - [`io`] — Vector readers for fvec, ivec, and mvec formats (mmap and HTTP).
//! - [`dataset`] — Configuration parsing for `dataset.yaml`: profiles, facets,
//!   pipelines, catalogs, and data source specifications.
//! - [`formats`] — Wire codecs for structured metadata (MNode), predicate
//!   trees (PNode), and the unified ANode wrapper.
//! - [`merkle`] — Content-addressed verification for dataset integrity.
//! - [`transport`] — HTTP transport layer for remote dataset access.
//! - [`cache`] — Cached channel utilities for buffered data delivery.
//!
//! ## Quick start
//!
//! ```no_run
//! use vectordata::TestDataGroup;
//!
//! fn main() -> anyhow::Result<()> {
//!     // Load from a local path or remote URL
//!     let group = TestDataGroup::load("https://example.com/datasets/glove-100/")?;
//!
//!     // Access a named profile (e.g., "default", "1M")
//!     if let Some(view) = group.profile("default") {
//!         let base = view.base_vectors()?;
//!         println!("{} vectors, dim={}", base.count(), base.dim());
//!
//!         let first = base.get(0)?;
//!         println!("First vector: {:?}", first);
//!     }
//!
//!     Ok(())
//! }
//! ```

/// Cached channel utilities for buffered data delivery.
pub mod cache;
/// Dataset catalog resolution — load catalogs from local and remote sources.
pub mod catalog;
/// Dataset configuration model (profiles, facets, pipelines, catalogs).
pub mod dataset;
/// Wire format codecs: MNode, PNode, ANode.
pub mod formats;
/// Content-addressed verification for dataset integrity.
pub mod merkle;
/// HTTP transport layer for remote dataset access.
pub mod transport;
/// Core data models for `dataset.yaml` parsing.
pub mod model;
/// Vector I/O: fvec, ivec, mvec readers (mmap and HTTP).
pub mod io;
/// Profile views for accessing dataset components.
pub mod view;
/// High-level dataset loading and profile access.
pub mod group;

pub use group::TestDataGroup;
pub use model::FacetConfig;
pub use view::{FacetDescriptor, TestDataView};
pub use io::VectorReader;

use thiserror::Error;

/// Top-level error type for the vectordata crate.
#[derive(Error, Debug)]
pub enum Error {
    /// A vector I/O operation failed (read, mmap, HTTP fetch).
    #[error("Vector IO error: {0}")]
    VectorIo(#[from] crate::io::IoError),
    /// The `dataset.yaml` file could not be read from disk or network.
    #[error("Failed to read dataset configuration: {0}")]
    ConfigIo(#[source] std::io::Error),
    /// The `dataset.yaml` content is not valid YAML or does not match the schema.
    #[error("Failed to parse dataset configuration: {0}")]
    ConfigParse(#[from] serde_yaml::Error),
    /// A URL string could not be parsed.
    #[error("Invalid URL: {0}")]
    UrlParse(#[from] url::ParseError),
    /// An HTTP request to a remote dataset failed.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    /// A required facet (e.g., `base_vectors`) is not defined in the profile.
    #[error("Required facet not defined: {0}")]
    MissingFacet(String),
    /// Catch-all for errors that do not fit other variants.
    #[error("{0}")]
    Other(String),
}

/// A specialized Result type for the library.
pub type Result<T> = std::result::Result<T, Error>;
