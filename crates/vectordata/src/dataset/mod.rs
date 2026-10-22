// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Dataset configuration model for vector datasets.
//!
//! The descriptor model — the canonical `dataset.yaml` schema, its
//! profiles, facets, strata, pipeline and catalog entries — is the API
//! core's ([`veks_apicore::dataset`]) and is re-exported here at its
//! earlier paths. What touches the world beyond a descriptor's own file
//! is vectordata's and lives here too: reading a directory of loose
//! files as a dataset ([`infer`]), a slab's layout ([`layout`]),
//! finding a series' shard files on disk ([`shards::discover_shards`]),
//! and [`DatasetConfigIo`] (a descriptor at a URL, shard declarations
//! reconciled with the files written).

pub use veks_apicore::dataset::*;

mod config_io;
pub mod infer;
pub mod layout;
pub mod shards;

pub use config_io::DatasetConfigIo;
pub use shards::discover_shards;
