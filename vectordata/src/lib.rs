//! # vectordata — typed access to vector search datasets
//!
//! Find datasets by name, discover profiles and facets, read vectors.
//! No URLs or file paths needed — the catalog system resolves everything.
//!
//! ## Find and use a dataset
//!
//! ```no_run
//! use vectordata::catalog::sources::CatalogSources;
//! use vectordata::catalog::resolver::Catalog;
//!
//! fn main() -> anyhow::Result<()> {
//!     // Load catalogs from ~/.config/vectordata/catalogs.yaml
//!     let catalog = Catalog::of(&CatalogSources::new().configure_default());
//!
//!     // Open a dataset by name — two calls to vectors
//!     let view = catalog.open_profile("my-dataset", "default")?;
//!     let base = view.base_vectors()?;
//!     println!("{} vectors, dim={}", base.count(), base.dim());
//!     let v: Vec<f32> = base.get(42)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Discover profiles and facets
//!
//! ```no_run
//! # use vectordata::catalog::sources::CatalogSources;
//! # use vectordata::catalog::resolver::Catalog;
//! # let catalog = Catalog::of(&CatalogSources::new().configure_default());
//! let group = catalog.open("my-dataset")?;
//! for name in group.profile_names() {
//!     let view = group.profile(&name).unwrap();
//!     let manifest = view.facet_manifest();
//!     for (facet, desc) in &manifest {
//!         println!("  {}:{} → {}", name, facet,
//!             desc.source_type.as_deref().unwrap_or("?"));
//!     }
//! }
//! # Ok::<(), vectordata::Error>(())
//! ```
//!
//! ## Typed ordinal access
//!
//! For metadata and scalar facets, [`typed_access::TypedReader`] provides
//! ordinal-based access with automatic type widening:
//!
//! ```no_run
//! use vectordata::typed_access::{ElementType, TypedReader};
//!
//! // Open a scalar metadata file — u8 read as i32 (automatic widening)
//! let meta = TypedReader::<i32>::open_auto("metadata.u8", ElementType::U8)?;
//! let label: i32 = meta.get_value(42)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Caching
//!
//! Remote data is automatically cached locally with merkle verification.
//! Downloaded chunks are persisted to `~/.cache/vectordata/` (configurable
//! via `~/.config/vectordata/settings.yaml`). Once fully cached, readers
//! switch to mmap for zero-copy access.
//!
//! ## Module overview
//!
//! - **catalog** — find datasets by name from configured catalog sources
//! - **group** — load a dataset, list profiles, get views
//! - **view** — access facets on a profile (base vectors, GT, metadata)
//! - **typed_access** — ordinal-based typed readers with widening
//! - **io** — low-level vector file readers (mmap + HTTP)
//! - **cache** — merkle-verified download cache
//! - **merkle** — content-addressed integrity verification
//! - **dataset** — dataset.yaml configuration model

/// Merkle-verified download cache for remote datasets.
pub mod cache;
/// Catalog discovery — find datasets by name from configured sources.
///
/// This is the recommended entry point. Configure catalog sources in
/// `~/.config/vectordata/catalogs.yaml`, then use [`catalog::resolver::Catalog`]
/// to find and open datasets by name.
pub mod catalog;
/// Dataset configuration model (profiles, facets, pipelines, catalogs).
pub mod dataset;
/// Wire format codecs: MNode, PNode, ANode.
pub mod formats;
/// Content-addressed integrity verification (SHA-256 merkle trees).
pub mod merkle;
/// HTTP transport layer (internal — used by cache and readers).
pub mod transport;
/// Core data models for `dataset.yaml` parsing.
pub mod model;
/// Low-level vector file readers (mmap + cached HTTP).
///
/// Most users should access data through [`catalog`] → [`view::TestDataView`]
/// rather than opening files directly with [`io::open_vec`].
pub mod io;
/// Parser for `knn_entries.yaml` (jvector-compatible dataset index).
pub mod knn_entries;
/// Typed ordinal access with runtime type negotiation.
///
/// [`typed_access::TypedReader`] provides ordinal-based access to scalar
/// and vector data with automatic type widening. Access through profiles
/// via [`view::GenericTestDataView::open_facet_typed`].
pub mod typed_access;
/// Profile views — access facets (base vectors, GT, metadata) on a dataset profile.
pub mod view;
/// Dataset loading and profile access.
///
/// [`TestDataGroup::load`] opens a dataset from a path or URL.
/// Prefer [`catalog::resolver::Catalog::open`] for name-based access.
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
