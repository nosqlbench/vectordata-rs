// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset configuration model for vector datasets.
//!
//! Defines the canonical `dataset.yaml` schema used to describe vector test
//! datasets, including profiles, pipeline definitions, facet mappings, and
//! catalog indexing.

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
