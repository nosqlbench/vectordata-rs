// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Find datasets by name: load the configured catalogs
//! ([`CatalogSources`] → [`Catalog`]), then open a dataset by name or
//! by `name:selector` spec ([`Catalog::open_spec`]).
//!
//! Provides multi-source catalog loading from local directories and remote
//! HTTP servers, with search by exact name, glob, and regex.
//!
//! ```rust,no_run
//! use vectordata::catalog::{Catalog, CatalogSources};
//!
//! let sources = CatalogSources::new().configure_default();
//! let catalog = Catalog::of(&sources);
//!
//! for entry in catalog.datasets() {
//!     println!("{} ({} profiles)", entry.name(), entry.profile_count());
//! }
//! ```

pub mod generate;
pub mod knn_entries;
pub mod resolver;
pub mod source;
pub mod sources;

pub use resolver::{Catalog, DatasetSelection};
pub use source::{BuiltCatalog, CatalogBuilder, LocatedCatalog, Requirement};
pub use veks_apicore::catalog::{CatalogDiagnostic, CatalogOrigin, CatalogSource, Severity};
pub use sources::CatalogSources;

