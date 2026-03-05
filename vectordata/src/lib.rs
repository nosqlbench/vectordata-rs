//! # Vector Data Tools
//!
//! This library provides tools for reading vector datasets defined by a `dataset.yaml` configuration file.
//! It supports both local file system access and remote access via HTTP/HTTPS.
//!
//! ## Remote Dataset Access
//!
//! You can access datasets hosted on a remote server by providing an HTTP/HTTPS URL
//! to `TestDataGroup::load`. The URL should point to a directory containing a `dataset.yaml`
//! or directly to the `dataset.yaml` file. The library uses Range requests to efficiently
//! read only the required vector data.
//!
//! ```no_run
//! use vectordata::TestDataGroup;
//!
//! fn main() -> anyhow::Result<()> {
//!     // Load a dataset from a remote URL
//!     // This URL should point to a dataset.yaml or a directory containing one.
//!     let url = "https://example.com/datasets/glove-100/dataset.yaml";
//!     let group = TestDataGroup::load(url)?;
//!
//!     // Get a specific profile view (e.g., "default", "10k", etc.)
//!     if let Some(view) = group.profile("default") {
//!         // Access the base vectors
//!         let base_vectors = view.base_vectors()?;
//!         println!("Base vectors count: {}", base_vectors.count());
//!         println!("Vector dimension: {}", base_vectors.dim());
//!
//!         // Read the first vector
//!         let first_vector = base_vectors.get(0)?;
//!         println!("First vector: {:?}", first_vector);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod io;
pub mod view;
pub mod group;

pub use group::TestDataGroup;
pub use model::FacetConfig;
pub use view::TestDataView;
pub use io::VectorReader;

use thiserror::Error;

/// Top-level error type for the library.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Vector IO error: {0}")]
    VectorIo(#[from] crate::io::IoError),
    #[error("Failed to read dataset configuration: {0}")]
    ConfigIo(#[source] std::io::Error),
    #[error("Failed to parse dataset configuration: {0}")]
    ConfigParse(#[from] serde_yaml::Error),
    #[error("Invalid URL: {0}")]
    UrlParse(#[from] url::ParseError),
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Required facet not defined: {0}")]
    MissingFacet(String),
    #[error("{0}")]
    Other(String),
}

/// A specialized Result type for the library.
pub type Result<T> = std::result::Result<T, Error>;
