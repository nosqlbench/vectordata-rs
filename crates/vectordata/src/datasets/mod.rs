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
//! | `datasets ping` | [`Catalog::open_spec`](crate::catalog::Catalog::open_spec), then a reader per facet ([`TestDataViewExt::facet_as`](crate::TestDataViewExt::facet_as)) |
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
pub mod init;
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
/// `program` is the program's own catalogs only, not the configured
/// ones, which are added here: a set built with `configure_default()`
/// lists each configured catalog twice, and every name in one becomes
/// ambiguous.
///
/// Numbered catalog shortcuts (`--at 2`, `--catalog 1`) resolve
/// against the configured list in every position — this seam is what
/// makes the shortcuts work identically from every binary that
/// dispatches into this module. An `--at` value may also name a
/// configured or program catalog ([`at_sources`]), so any ordered list
/// of catalogs is a list of `--at` values.
pub fn build_sources(program: &CatalogSources, configdir: &str, extra_catalogs: &[String], at: &[String]) -> RunSources {
    let at = catalog_args(at);
    let extra_catalogs = catalog_args(extra_catalogs);
    let candidates = CatalogSources::new().configure(configdir).extend(program.clone());
    if !at.is_empty() {
        // `--at` names which catalogs to read; what the user and the
        // program know of datasets (their overlays) still applies.
        let sources = at_sources(&at, &candidates)
            .unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            })
            .with_overlays_from(&candidates);
        let named = sources.iter().map(|(s, _)| s.name().to_string()).collect();
        return RunSources { sources, named };
    }
    let (configured, named) = (candidates, extra_catalogs);
    let before: Vec<String> = configured.iter().map(|(s, _)| s.name().to_string()).collect();
    let sources = configured.add_catalogs(&named);
    let named = sources
        .iter()
        .map(|(s, _)| s.name().to_string())
        .filter(|name| !before.contains(name))
        .collect();
    RunSources { sources, named }
}

/// The catalogs a run's `--at` values name, in their order, in place of
/// every other: each value is a location when it is a URL or a path
/// that exists (a path that exists is a path, as in
/// [`open_spec`](crate::catalog::Catalog::open_spec)); else the catalog
/// of `candidates` (the configured ones and the program's) of that
/// name; else a location, as before. A name two candidates share is an
/// error naming both, never settled by order. Numbered shortcuts are
/// resolved before this ([`catalog_args`]).
pub fn at_sources(values: &[String], candidates: &CatalogSources) -> Result<CatalogSources, String> {
    let mut sources = CatalogSources::new();
    for value in values {
        let is_location = value.contains("://")
            || std::path::Path::new(&crate::catalog::sources::expand_tilde(value)).exists();
        let named = if is_location { None } else { candidates.named(value)? };
        sources = match named {
            Some(named) => sources.extend(named),
            None => sources.add_catalogs(std::slice::from_ref(value)),
        };
    }
    Ok(sources)
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
        if matches!(error, crate::Error::AmbiguousDataset { .. }) {
            // Never resolved by catalog order: the user picks the catalog.
            eprintln!("Name the catalog to read it from with `--at <catalog>`.");
        }
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

    /// **`--at` takes catalog names as well as locations, in order**: a
    /// program catalog and a location in the order given, with nothing
    /// configured; a configured catalog by name; an existing path stays
    /// a path whatever catalog shares its name; a name two catalogs hold
    /// is refused naming both.
    #[test]
    fn at_names_a_program_or_configured_catalog_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_naming_a_missing_catalog(dir.path());
        let program = CatalogSources::new().with(crate::catalog::CatalogBuilder::named("fvec").build());
        let candidates = CatalogSources::new().configure(&config).extend(program.clone());
        let stage = dir.path().join("stage");
        std::fs::create_dir_all(&stage).unwrap();
        std::fs::write(stage.join("catalog.json"), "[]").unwrap();
        let stage = stage.to_string_lossy().into_owned();
        let names = |s: &CatalogSources| s.iter().map(|(c, _)| c.name().to_string()).collect::<Vec<_>>();

        let picked = at_sources(&["fvec".to_string(), stage.clone()], &candidates).unwrap();
        let got = names(&picked);
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0], "fvec", "the program's catalog first, as given: {got:?}");
        assert!(!got.contains(&"lab".to_string()), "nothing configured unless named: {got:?}");
        assert!(picked.iter().nth(1).unwrap().0.origin().to_string().contains("stage"), "then the location");

        assert_eq!(names(&at_sources(&["lab".to_string()], &candidates).unwrap()), ["lab"], "a configured catalog by name");

        // A path that exists is a path, even with a catalog of its name.
        let shadow = dir.path().join("fvec");
        std::fs::create_dir_all(&shadow).unwrap();
        std::fs::write(shadow.join("catalog.json"), "[]").unwrap();
        let shadow = shadow.to_string_lossy().into_owned();
        let named_fvec = CatalogSources::new().with(crate::catalog::CatalogBuilder::named(shadow.clone()).build());
        let picked = at_sources(std::slice::from_ref(&shadow), &named_fvec).unwrap();
        let origins: Vec<_> = picked.iter().map(|(c, _)| c.origin()).collect();
        assert_eq!(origins.len(), 1, "{origins:?}");
        assert!(!matches!(origins[0], crate::catalog::CatalogOrigin::Registered { .. }), "read as a location: {origins:?}");

        let twice = candidates.clone().extend(CatalogSources::new().with(crate::catalog::CatalogBuilder::named("fvec").build()));
        let err = at_sources(&["fvec".to_string()], &twice).unwrap_err();
        assert!(err.contains("'fvec'") && err.contains("2 catalogs"), "{err}");

        // Through build_sources: the run reads only what --at names.
        let run = build_sources(&program, &config, &[], &["fvec".to_string()]);
        assert_eq!(names(run.sources()), ["fvec"]);
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

