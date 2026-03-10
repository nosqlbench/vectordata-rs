// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset catalog infrastructure.
//!
//! Provides catalog generation, resolution, and source discovery for dataset
//! directories. Catalogs are hierarchical index files (`catalog.json`,
//! `catalog.yaml`) that describe all datasets discoverable under a directory
//! tree.
//!
//! The CLI entry point lives under `veks datasets catalog`; this module
//! provides the library types used by that command and the rest of the
//! inventory subsystem.

pub mod generate;
pub mod resolver;
pub mod sources;
