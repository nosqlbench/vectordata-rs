// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Dataset catalog resolution.
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
//!     println!("{} ({} profiles)", entry.name, entry.profile_count());
//! }
//! ```

pub mod resolver;
pub mod sources;

pub use resolver::Catalog;
pub use sources::CatalogSources;
