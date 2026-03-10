// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset configuration model for vector datasets.
//!
//! This crate defines the canonical `dataset.yaml` schema used to describe
//! vector test datasets. It owns:
//!
//! - **Configuration model** — `DatasetConfig` with profiles, upstream
//!   pipeline definitions, and facet mappings.
//! - **Profile system** — `DSProfileGroup`, `DSProfile`, `DSView` with
//!   default-inheritance and alias resolution.
//! - **Source sugar** — `DSSource`, `DSWindow`, `DSInterval` with inline
//!   bracket/paren window notation and namespace support.
//! - **Standard facet registry** — `StandardFacet` enum defining canonical
//!   facet names, keys, and shorthand aliases.
//! - **Pipeline schema** — `PipelineConfig`, `StepDef`, `OnPartial` for
//!   the `upstream` block (command-agnostic).
//!
//! This crate is intentionally lightweight (serde + indexmap only) so that
//! both the `vectordata` access layer and the `veks` processing tool can
//! depend on it without pulling in heavy I/O or CLI dependencies.

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
