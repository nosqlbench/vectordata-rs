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
pub use source::{BuiltCatalog, CatalogBuilder, CatalogOrigin, CatalogSource, LocatedCatalog, Requirement};
pub use sources::CatalogSources;

/// Something that went wrong while catalog sources were read or
/// catalogs loaded — a location that could not be read, a file that
/// would not parse, an entry that was skipped.
///
/// Collected on [`CatalogSources::diagnostics`] and
/// [`Catalog::diagnostics`] rather than printed, so a library caller
/// decides what its users see. Displays as `error: …` or `warning: …`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogDiagnostic {
    /// How serious it is.
    pub severity: Severity,
    /// What happened, naming the location involved.
    pub message: String,
    /// The catalog it concerns, by its name in the
    /// [`CatalogSources`] set; `None` for a problem with the
    /// configuration itself, such as an unreadable `catalogs.yaml`.
    /// [`Catalog::of`] records it.
    pub catalog: Option<String>,
}

/// How serious a [`CatalogDiagnostic`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Something was skipped; what loaded is still usable.
    Warning,
    /// A required catalog could not be loaded; its datasets are missing.
    Error,
}

impl CatalogDiagnostic {
    /// Something was skipped; what loaded is still usable.
    pub fn warning(message: impl Into<String>) -> Self {
        CatalogDiagnostic { severity: Severity::Warning, message: message.into(), catalog: None }
    }

    /// A catalog, or part of one, could not be loaded.
    pub fn error(message: impl Into<String>) -> Self {
        CatalogDiagnostic { severity: Severity::Error, message: message.into(), catalog: None }
    }
}

impl std::fmt::Display for CatalogDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.severity {
            Severity::Warning => write!(f, "warning: {}", self.message),
            Severity::Error => write!(f, "error: {}", self.message),
        }
    }
}
