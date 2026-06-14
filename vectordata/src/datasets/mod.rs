// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Canonical implementation of `<binary> datasets …` subcommands.
//!
//! Both the `vectordata` binary and the `veks` CLI dispatch into
//! this module — there is exactly one implementation of each
//! command. The submodules return *exit codes* (`i32`) rather than
//! `Result`s so the dispatch layer in either binary can simply
//! `std::process::exit(code)`.

pub mod cache;
pub mod curlify;
pub mod derive;
pub mod describe;
pub mod drop_cache;
#[cfg(feature = "cli")]
pub mod dyncomp;
pub mod filter;
pub mod list;
pub mod precache;
pub mod ping;

use crate::catalog::sources::{self, CatalogSources};

/// Build [`CatalogSources`] from the `--configdir` / `--catalog` /
/// `--at` trio shared by the datasets subcommands. `--at` locations
/// override the configured catalogs entirely; otherwise `configdir`'s
/// `catalogs.yaml` is loaded and any `--catalog` extras appended.
///
/// Numbered catalog shortcuts (`--at 2`, `--catalog 1`) resolve
/// against the configured list in every position — this seam is what
/// makes the shortcuts work identically from every binary that
/// dispatches into this module.
pub fn build_sources(configdir: &str, extra_catalogs: &[String], at: &[String]) -> CatalogSources {
    let at = sources::resolve_catalog_values(at);
    let extra_catalogs = sources::resolve_catalog_values(extra_catalogs);
    let mut built = CatalogSources::new();
    if !at.is_empty() {
        built = built.add_catalogs(&at);
    } else {
        built = built.configure(configdir);
        if !extra_catalogs.is_empty() {
            built = built.add_catalogs(&extra_catalogs);
        }
    }
    built
}
