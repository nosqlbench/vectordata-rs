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

/// The catalogs a command runs against: what [`build_sources`] gives,
/// remembering which catalogs the user named for this run.
#[derive(Debug, Clone)]
pub struct RunSources {
    sources: CatalogSources,
    named: Vec<String>,
}

impl RunSources {
    /// The sources, for a caller that opens the catalog itself (shell
    /// completion, which must print nothing).
    pub fn sources(&self) -> &CatalogSources {
        &self.sources
    }

    /// Whether there is no catalog at all to read.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Open the catalogs, printing every problem to stderr (SRD §11,
    /// item 4): a configured catalog that fails to load is reported and
    /// the command carries on with the others. A catalog the user named
    /// for this run (`--at`, `--catalog`) that fails to load is an
    /// error: `Err` carries the exit status.
    pub fn open(&self) -> Result<crate::catalog::Catalog, i32> {
        let catalog = open_catalog(&self.sources);
        let failed: Vec<&str> = catalog
            .diagnostics()
            .iter()
            .filter(|d| d.severity == crate::catalog::Severity::Error)
            .filter_map(|d| d.catalog.as_deref())
            .filter(|name| self.named.iter().any(|n| n == name))
            .collect();
        match failed.first() {
            None => Ok(catalog),
            Some(name) => {
                eprintln!("error: catalog '{name}', named for this run, could not be loaded");
                Err(1)
            }
        }
    }
}

/// Build the catalogs a `datasets` command runs against, from the
/// catalogs the program running it supplies and the `--configdir` /
/// `--catalog` / `--at` trio: the one assembly every command uses.
/// `--at` locations replace every other catalog for the run;
/// otherwise `configdir`'s `catalogs.yaml` is read, `program`'s catalogs
/// follow it (empty for the `vectordata` binary; a program embedding
/// the CLI passes its own, [`bin_main_with_catalogs`](crate::shell::bin_main_with_catalogs)),
/// and any `--catalog` locations follow those. Catalogs given by `--at`
/// or `--catalog` are [named for the run](RunSources::open).
///
/// Numbered catalog shortcuts (`--at 2`, `--catalog 1`) resolve
/// against the configured list in every position — this seam is what
/// makes the shortcuts work identically from every binary that
/// dispatches into this module.
pub fn build_sources(program: &CatalogSources, configdir: &str, extra_catalogs: &[String], at: &[String]) -> RunSources {
    let at = catalog_args(at);
    let extra_catalogs = catalog_args(extra_catalogs);
    let (configured, named) = if at.is_empty() {
        (CatalogSources::new().configure(configdir).extend(program.clone()), extra_catalogs)
    } else {
        (CatalogSources::new(), at)
    };
    let before: Vec<String> = configured.iter().map(|(s, _)| s.name().to_string()).collect();
    let sources = configured.add_catalogs(&named);
    let named = sources
        .iter()
        .map(|(s, _)| s.name().to_string())
        .filter(|name| !before.contains(name))
        .collect();
    RunSources { sources, named }
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
        eprintln!("  - {} (no profiles)", entry.name());
    } else {
        eprintln!("  - {} (profiles: {})", entry.name(), profiles.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config directory whose `catalogs.yaml` names `lab`, a location
    /// with no catalog there.
    fn config_naming_a_missing_catalog(dir: &std::path::Path) -> String {
        let missing = dir.join("missing");
        std::fs::write(dir.join("catalogs.yaml"), format!("lab: {}\n", missing.display())).unwrap();
        dir.to_string_lossy().into_owned()
    }

    /// **A program's catalogs follow the configured ones and precede
    /// `--catalog`'s; `--at` replaces them all.** The order every
    /// command reads catalogs in, so an embedded CLI sees the catalogs
    /// its program registered.
    #[test]
    fn a_programs_catalogs_follow_the_configured_ones_and_at_replaces_them() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_naming_a_missing_catalog(dir.path());
        let program = CatalogSources::new().with(crate::catalog::CatalogBuilder::named("mine").build());
        let extra = dir.path().join("extra").to_string_lossy().into_owned();
        let names = |run: &RunSources| run.sources().iter().map(|(s, _)| s.name().to_string()).collect::<Vec<_>>();

        let run = build_sources(&program, &config, std::slice::from_ref(&extra), &[]);
        let got = names(&run);
        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(&got[..2], ["lab", "mine"], "configured, then the program's, then --catalog: {got:?}");

        let run = build_sources(&program, &config, &[], std::slice::from_ref(&extra));
        let got = names(&run);
        assert!(!got.iter().any(|n| n == "mine" || n == "lab"), "--at replaces both: {got:?}");
        assert_eq!(got.len(), 1, "{got:?}");

        let run = build_sources(&CatalogSources::new(), &config, &[], &[]);
        assert_eq!(names(&run), ["lab"], "no program catalogs: the configured ones alone");
    }

    /// **A configured catalog that fails to load is reported, and the
    /// command goes on; one named for the run is an error** (SRD §11,
    /// item 4). Each problem names the catalog it concerns.
    #[test]
    fn only_a_catalog_named_for_the_run_fails_the_command() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_naming_a_missing_catalog(dir.path());
        let configured = build_sources(&CatalogSources::new(), &config, &[], &[]).open().expect("a broken configured catalog is reported only");
        let problem = configured.diagnostics().iter().find(|d| d.catalog.as_deref() == Some("lab"));
        assert!(problem.is_some(), "{:?}", configured.diagnostics());

        let missing = dir.path().join("also-missing").to_string_lossy().into_owned();
        assert_eq!(build_sources(&CatalogSources::new(), &config, &[], std::slice::from_ref(&missing)).open().err(), Some(1), "--at");
        assert_eq!(build_sources(&CatalogSources::new(), &config, std::slice::from_ref(&missing), &[]).open().err(), Some(1), "--catalog");
    }
}

