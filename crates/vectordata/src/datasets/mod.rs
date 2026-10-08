// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI support. The implementation of `<binary> datasets …`: both the
//! `vectordata` binary and `veks` dispatch here, so each subcommand has
//! exactly one implementation.
//!
//! **Not the library API.** Each command is a thin adapter: it parses
//! command-line shaped input, calls the library, prints, and returns a
//! process exit code (`i32`) so the dispatching binary can
//! `std::process::exit(code)`. A program that wants what a command does
//! calls the library function underneath it:
//!
//! | Command | Library call |
//! |---|---|
//! | `datasets precache` | [`TestDataView::fetch`](crate::TestDataView::fetch), [`TestDataGroup::fetch`](crate::TestDataGroup::fetch), [`Catalog::fetch`](crate::catalog::Catalog::fetch) |
//! | `datasets list` | [`Catalog::datasets`](crate::catalog::Catalog::datasets), [`Catalog::match_glob`](crate::catalog::Catalog::match_glob) |
//! | `datasets describe` | [`Catalog::open_spec`](crate::catalog::Catalog::open_spec), [`TestDataView::facet_manifest`](crate::TestDataView::facet_manifest) |
//! | `datasets cache` (veks) | [`cache_admin::list_entries`](crate::cache_admin::list_entries) |
//! | `datasets drop-cache` (veks) | [`cache_admin::prune_by_filter`](crate::cache_admin::prune_by_filter) is the library equivalent; the command still scans the cache itself |
//! | `datasets ping` | [`Catalog::open_spec`](crate::catalog::Catalog::open_spec), then a reader per facet ([`TestDataView::facet`](crate::TestDataView::facet)) |
//! | `datasets derive`, `datasets filter`, `datasets curlify` | no library equivalent; these are commands |
//!
//! The full map, for every command of both binaries, is in the crate's
//! `AGENTS.md` ([`crate::_agents`]).

pub(crate) mod shard_writer;
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
    let at = catalog_args(at);
    let extra_catalogs = catalog_args(extra_catalogs);
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

/// Resolve `--at`/`--catalog` values, numbered shortcuts included,
/// exiting with status 1 on an index that names no configured catalog.
/// The command-line boundary for
/// [`sources::resolve_catalog_values`].
pub fn catalog_args(values: &[String]) -> Vec<String> {
    match sources::resolve_catalog_values(values) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

/// [`Catalog::of`](crate::catalog::Catalog::of), printing what went
/// wrong while loading to stderr — the command-line rendering of
/// [`Catalog::diagnostics`](crate::catalog::Catalog::diagnostics).
pub fn open_catalog(sources: &CatalogSources) -> crate::catalog::Catalog {
    let catalog = crate::catalog::Catalog::of(sources);
    for d in catalog.diagnostics() {
        eprintln!("{d}");
    }
    catalog
}

/// Print why a dataset could not be found, with the catalog's
/// datasets listed so the user can see what is reachable: the
/// command-line rendering of [`Error::UnknownDataset`](crate::Error::UnknownDataset)
/// and [`Error::AmbiguousDataset`](crate::Error::AmbiguousDataset).
/// Anything else is printed as an error line.
pub fn report_lookup_failure(catalog: &crate::catalog::Catalog, error: &crate::Error) {
    let crate::Error::UnknownDataset { name, .. } = error else {
        eprintln!("error: {error}");
        return;
    };
    eprintln!("Dataset '{name}' not found.");
    if catalog.is_empty() {
        eprintln!("No datasets are available in the catalog.");
        return;
    }
    let near = catalog.suggestions(name);
    if !near.is_empty() {
        eprintln!("Did you mean one of these datasets?");
        for entry in near {
            print_dataset_with_profiles(entry);
        }
        eprintln!();
    }
    eprintln!("Available datasets ({} total):", catalog.datasets().len());
    for entry in catalog.datasets() {
        print_dataset_with_profiles(entry);
    }
}

/// One dataset and its profile names, on stderr.
fn print_dataset_with_profiles(entry: &crate::dataset::CatalogEntry) {
    let profiles: Vec<String> = entry.profile_names().into_iter().map(|s| s.to_string()).collect();
    if profiles.is_empty() {
        eprintln!("  - {} (no profiles)", entry.name);
    } else {
        eprintln!("  - {} (profiles: {})", entry.name, profiles.join(", "));
    }
}
