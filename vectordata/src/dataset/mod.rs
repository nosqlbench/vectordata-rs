// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset configuration model for vector datasets.
//!
//! Defines the canonical `dataset.yaml` schema used to describe vector test
//! datasets, including profiles, pipeline definitions, facet mappings, and
//! catalog indexing.
//!
//! ## Sub-modules
//!
//! - [`config`] — Top-level `DatasetConfig` and `DatasetAttributes`.
//! - [`profile`] — `DSProfile`, `DSProfileGroup`, `DSView` with inheritance.
//! - [`facet`] — `StandardFacet` enum (canonical facet names and aliases).
//! - [`source`] — `DSSource`, `DSInterval`, `DSWindow` (path + range parsing).
//! - [`pipeline`] — `PipelineConfig`, `StepDef`, `OnPartial` (upstream build steps).
//! - [`catalog`] — `CatalogEntry`, `CatalogLayout` (dataset index files).

pub mod catalog;
pub mod config;
pub mod facet;
pub mod pipeline;
pub mod profile;
pub mod source;

pub use catalog::{CatalogEntry, CatalogLayout, find_catalog, load_catalog};
pub use config::{DatasetAttributes, DatasetConfig};
pub use facet::StandardFacet;
pub use pipeline::{OnPartial, PipelineConfig, StepDef};
pub use profile::{DSProfile, DSProfileGroup, DSView};
pub use source::{DSInterval, DSSource, DSWindow};
