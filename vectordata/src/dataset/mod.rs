// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dataset configuration model for vector datasets.
//!
//! Defines the canonical `dataset.yaml` schema used to describe vector test
//! datasets, including profiles, pipeline definitions, facet mappings, and
//! catalog indexing.
//!
//! ## Sub-modules
//!
//! - **config** — Top-level `DatasetConfig` and `DatasetAttributes`.
//! - **strata** — `Strata`, `Stratum` (named sized-profile generators).
//! - **profile** — `DSProfile`, `DSProfileGroup`, `DSView` with inheritance.
//! - **facet** — `StandardFacet` enum (canonical facet names and aliases).
//! - **shard_sizing** — `ShardPlan`: how many records fit in a capped shard file.
//! - **source** — `DSSource`, `DSInterval`, `DSWindow` (path + range parsing).
//! - **pipeline** — `PipelineConfig`, `StepDef`, `OnPartial` (upstream build steps).
//! - **catalog** — `CatalogEntry`, `CatalogLayout` (dataset index files).

pub mod catalog;
pub mod conformance;
pub mod config;
pub mod expansion;
pub mod facet;
pub mod layout;
pub mod pipeline;
pub mod profile;
pub mod shard_sizing;
pub mod shards;
pub mod source;
pub mod strata;

pub use catalog::{CatalogEntry, CatalogLayout, find_catalog, load_catalog};
pub use config::{DatasetAttributes, DatasetConfig};
pub use expansion::{collect_all_steps, expand_per_profile_steps, expand_per_profile_steps_scoped, filter_steps_for_profile, resolve_steps};
pub use facet::StandardFacet;
pub use shard_sizing::{DEFAULT_MAX_SHARD_BYTES, RecordSize, ShardPlan, Sharding};
pub use shards::{discover_shards, shard_name, shard_source_spec};
pub use pipeline::{OnPartial, PipelineConfig, StepDef};
pub use profile::{DSProfile, DSProfileGroup, DSView};
pub use source::{DSInterval, DSSource, DSWindow};
pub use strata::{Strata, Stratum};
