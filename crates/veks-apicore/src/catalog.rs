// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Where a catalog's entries come from ([`CatalogSource`]), and what
//! went wrong reading one ([`CatalogDiagnostic`]). The entries are
//! [`CatalogEntry`] values; vectordata
//! reads catalogs at locations, builds them in code, and resolves
//! dataset names across a set of them.

use std::fmt;

use crate::dataset::CatalogEntry;

/// A catalog: a named supply of [`CatalogEntry`] values.
///
/// vectordata provides `LocatedCatalog` (a file, directory or URL) and
/// `BuiltCatalog` (from `CatalogBuilder`). A program may implement it
/// to serve entries from anywhere else:
///
/// ```
/// use veks_apicore::catalog::{CatalogDiagnostic, CatalogOrigin, CatalogSource};
/// use veks_apicore::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
/// use veks_apicore::Location;
///
/// /// The datasets a program knows by itself.
/// struct Builtin;
///
/// impl CatalogSource for Builtin {
///     fn name(&self) -> &str {
///         "builtin"
///     }
///     fn origin(&self) -> CatalogOrigin {
///         CatalogOrigin::Registered { label: "the program's own".into() }
///     }
///     fn entries(&self, diagnostics: &mut Vec<CatalogDiagnostic>) -> Option<Vec<CatalogEntry>> {
///         let built = DatasetConfig::builder("demo")
///             .profile("default", |p| p.facet(StandardFacet::BaseVectors, "base.fvec"))
///             .build()
///             .and_then(|d| {
///                 let home = Location::parse("https://example.com/demo/").map_err(|e| e.to_string())?;
///                 CatalogEntry::builder("demo").home(&home).dataset(d).build()
///             });
///         match built {
///             Ok(entry) => Some(vec![entry]),
///             Err(e) => {
///                 diagnostics.push(CatalogDiagnostic::error(e));
///                 Some(Vec::new())
///             }
///         }
///     }
/// }
///
/// let mut diagnostics = Vec::new();
/// let entries = Builtin.entries(&mut diagnostics).unwrap();
/// assert_eq!(entries[0].name(), "demo");
/// assert!(diagnostics.is_empty());
/// ```
pub trait CatalogSource: Send + Sync {
    /// The catalog's symbolic name: what provenance, ambiguity reports
    /// and the catalogs screen show.
    fn name(&self) -> &str;

    /// Where it came from, for display and diagnostics.
    fn origin(&self) -> CatalogOrigin;

    /// Its entries. `None` when there is no catalog there at all, such
    /// as a location with no catalog file; the set it belongs to decides
    /// whether that is an error. Problems with what is there (a file
    /// that does not parse, an entry skipped) go into `diagnostics`,
    /// never to stdout.
    fn entries(&self, diagnostics: &mut Vec<CatalogDiagnostic>) -> Option<Vec<CatalogEntry>>;
}

/// Where a [`CatalogSource`] came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogOrigin {
    /// A file, directory or URL, as written: a catalog location is a
    /// cache identity (its kept copy is named by it), so it is never
    /// re-spelled.
    Located {
        /// The location.
        location: String,
        /// The `catalogs.yaml` that names it, when one does.
        configured_in: Option<String>,
    },
    /// Built in memory by a program; `label` is what diagnostics and
    /// the catalogs screen show.
    Registered {
        /// The label.
        label: String,
    },
}

impl fmt::Display for CatalogOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatalogOrigin::Located { location, .. } => f.write_str(location),
            CatalogOrigin::Registered { label } => f.write_str(label),
        }
    }
}

/// Something that went wrong while catalog sources were read or
/// catalogs loaded — a location that could not be read, a file that
/// would not parse, an entry that was skipped.
///
/// Collected on vectordata's `CatalogSources::diagnostics` and
/// `Catalog::diagnostics` rather than printed, so a library caller
/// decides what its users see. Displays as `error: …` or `warning: …`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogDiagnostic {
    /// How serious it is.
    pub severity: Severity,
    /// What happened, naming the location involved.
    pub message: String,
    /// The catalog it concerns, by its name in the
    /// `CatalogSources` set; `None` for a problem with the
    /// configuration itself, such as an unreadable `catalogs.yaml`.
    /// `Catalog::of` records it.
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
