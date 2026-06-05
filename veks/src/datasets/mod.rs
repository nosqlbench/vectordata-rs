// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Root-level `veks datasets` command group.
//!
//! Provides inventory and exploration of datasets addressable by the catalog
//! system. Subcommands: list, plan, cache, curlify, precache, catalog.
//!
//! These are standalone commands — no pipeline context or StreamContext required.

pub(crate) mod precache;

// Every useful `datasets` subcommand now lives in
// `vectordata::datasets` so library consumers can drive them
// without depending on veks. Re-exported here so call sites in
// this file (and any veks-only consumer) keep the short path.
pub use vectordata::datasets::cache;
pub use vectordata::datasets::curlify;
pub use vectordata::datasets::drop_cache;
pub use vectordata::datasets::filter;
pub use vectordata::datasets::list;
pub use vectordata::datasets::ping;

use std::path::PathBuf;


/// Browse, search, and manage datasets
#[derive(veks_completion_derive::VeksCli)]
#[command(disable_help_subcommand = true)]
pub struct DatasetsArgs {
    #[command(subcommand)]
    pub command: DatasetsCommand,
}

/// Available subcommands under `veks datasets`.
#[derive(veks_completion_derive::VeksCli)]
pub enum DatasetsCommand {
    /// List available datasets from configured or specified catalogs
    #[command(alias = "ls")]
    List {
        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir())]
        configdir: String,

        /// Additional catalog directories, file paths, or HTTP URLs
        #[arg(long)]
        catalog: Vec<String>,

        /// Catalog URLs or paths to use *instead* of configured catalogs
        #[arg(long = "at")]
        at: Vec<String>,

        /// Output format
        #[arg(long = "output-format", short = 'f', default_value = "text", value_parser = ["text", "csv", "json", "yaml"])]
        output_format: String,

        /// Show detailed information including attributes, tags, and views
        #[arg(long)]
        verbose: bool,

        /// Group output by: source, profile, or metric (text format only)
        #[arg(long = "group-by", value_name = "KEY", value_parser = ["source", "profile", "metric"])]
        group_by: Option<String>,

        /// Filter profiles by name (substring, regex, or glob)
        #[arg(long = "matching-profile")]
        matching_profile: Option<String>,

        /// Select a single dataset:profile; fails if the filters are ambiguous
        #[arg(long)]
        select: Option<String>,

        // -- Filter predicates --

        /// Filter by dataset name (exact match, substring, regex, or glob)
        #[arg(long)]
        dataset: Option<String>,

        /// Filter by dataset name (substring, regex, or glob) — alias for --dataset
        #[arg(long = "matching-name", conflicts_with = "dataset")]
        matching_name: Option<String>,

        /// Filter: dataset must contain this facet/view
        #[arg(long = "with-facet")]
        facet: Vec<String>,

        /// Filter: dataset must use this distance metric
        #[arg(long = "with-metric")]
        metric: Option<String>,

        /// Filter by description/notes/model (substring, regex, or glob)
        #[arg(long = "matching-desc")]
        matching_desc: Option<String>,

        /// Filter: dataset has at least this many base vectors (supports K/M/B suffixes)
        #[arg(long = "with-min-size")]
        min_size: Option<String>,

        /// Filter: dataset has at most this many base vectors (supports K/M/B suffixes)
        #[arg(long = "with-max-size")]
        max_size: Option<String>,

        /// Filter: dataset has exactly this many base vectors (supports K/M/B suffixes)
        #[arg(long = "with-size")]
        size: Option<String>,

        /// Filter: minimum dimensionality of base vectors
        #[arg(long = "with-min-dim")]
        min_dim: Option<u32>,

        /// Filter: maximum dimensionality of base vectors
        #[arg(long = "with-max-dim")]
        max_dim: Option<u32>,

        /// Filter: exact dimensionality of base vectors
        #[arg(long = "with-dim")]
        dim: Option<u32>,

        /// Filter: vector data type (float32, float16, uint8, int32, numpy, hdf5)
        #[arg(long = "with-vtype")]
        vtype: Option<String>,

        /// Filter: minimum total data size in bytes (supports K/M/G/T suffixes)
        #[arg(long = "with-data-min")]
        data_min: Option<String>,

        /// Filter: maximum total data size in bytes (supports K/M/G/T suffixes)
        #[arg(long = "with-data-max")]
        data_max: Option<String>,

        /// List locally cached datasets instead of catalog entries
        #[arg(long)]
        cached: bool,

        /// Override cache directory location (used with --cached)
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Generate curl download commands for a dataset
    Curlify {
        /// Dataset directory or path to dataset.yaml
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Output directory for downloads
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Configure vectordata settings (cache directory, mounts)
    #[command(disable_help_subcommand = true)]
    Config {
        #[command(subcommand)]
        command: ConfigSubcommand,
    },
    /// Show cache status for a dataset (merkle coverage, sizes, completion)
    CacheStatus {
        /// Dataset name from catalog (omit with --all for all cached datasets)
        #[arg(long)]
        dataset: Option<String>,

        /// Show status for all cached datasets
        #[arg(long)]
        all: bool,

        /// Show per-chunk RLE coverage detail
        #[arg(long)]
        verbose: bool,

        /// Show file tree of the cache directory
        #[arg(long)]
        tree: bool,

        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir())]
        configdir: String,

        /// Additional catalog directories, file paths, or HTTP URLs
        #[arg(long)]
        catalog: Vec<String>,

        /// Catalog URLs or paths to use *instead* of configured catalogs
        #[arg(long = "at")]
        at: Vec<String>,
    },
    /// Ping a remote dataset: verify catalog access, list facets, read a sample
    #[command(alias = "probe")]
    Ping {
        /// Catalog base URL(s) or number(s) (e.g., 1, https://bucket.s3.amazonaws.com/path/)
        /// to pin the search to. If omitted, searches configured catalogs for the dataset.
        #[arg(long = "at")]
        at: Vec<String>,

        /// Dataset name within the catalog
        #[arg(long)]
        dataset: String,

        /// Profile to ping (default: "default")
        #[arg(long, default_value = "default")]
        profile: String,
    },
    /// Remove cached datasets from the local cache directory
    #[command(alias = "purge")]
    DropCache {
        /// Dataset names or glob patterns to drop (default: all cached)
        datasets: Vec<String>,

        /// Skip confirmation prompts
        #[arg(short = 'y', long)]
        yes: bool,

        /// List all files in each dataset before dropping
        #[arg(long)]
        verbose: bool,

        /// Override cache directory location
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Remove legacy `<host>` / `<host>:<port>` cache directories left
    /// behind by versions before the content-addressed layout. Safe
    /// to run repeatedly — only matches the exact legacy shapes.
    PruneLegacyCache {
        /// Print what would be removed without deleting.
        #[arg(long)]
        dry_run: bool,

        /// Override cache directory location.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Materialize a profile of an existing dataset as a new
    /// self-standing dataset (flattens windowed views into their
    /// own files).
    Derive {
        /// Source dataset: catalog name, local directory containing
        /// `dataset.yaml`, path to a `dataset.yaml` file, or HTTPS URL.
        /// Local directories take a fast path that bypasses the
        /// runtime cache + access layer entirely.
        #[arg(long)]
        dataset: String,

        /// Profile to derive. Required.
        #[arg(long)]
        profile: String,

        /// Output directory for the new dataset.
        #[arg(long)]
        output: PathBuf,

        /// Override the derived dataset's name (default: `<source>-<profile>`).
        #[arg(long)]
        name: Option<String>,

        /// Overwrite the output directory if it already exists.
        #[arg(long)]
        force: bool,

        /// Configuration directory containing catalogs.yaml
        /// (only used when `--dataset` is a catalog name).
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir())]
        configdir: String,

        /// Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long)]
        catalog: Vec<String>,

        /// Catalog URLs or paths to use *instead* of configured catalogs.
        #[arg(long = "at")]
        at: Vec<String>,
    },
    /// Download and cache dataset facets locally
    Precache {
        /// Dataset name or dataset:profile from catalog
        #[arg(long)]
        dataset: String,

        /// Profile name (overrides profile in dataset:profile)
        #[arg(long)]
        profile: Option<String>,

        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir())]
        configdir: String,

        /// Additional catalog directories, file paths, or HTTP URLs
        #[arg(long)]
        catalog: Vec<String>,

        /// Catalog URLs or paths to use *instead* of configured catalogs
        #[arg(long = "at")]
        at: Vec<String>,

        /// Override cache directory location
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

/// Subcommands under `veks datasets config` (mirrors `vectordata config`).
#[derive(veks_completion_derive::VeksCli)]
pub enum ConfigSubcommand {
    /// Read config — all settings, or one value.
    ///
    ///   veks datasets config get          # current configuration
    ///   veks datasets config get cache    # the cache_dir on one line
    Get {
        /// A single setting key (currently `cache`); omit to show everything.
        key: Option<String>,
    },
    /// Write a config value, e.g. `veks datasets config set cache <dir>`.
    Set {
        /// Setting key (currently `cache`).
        key: String,
        /// New value (for `cache`: a path, or `auto`).
        value: String,
        /// Overwrite a protected settings.yaml.
        #[arg(long)]
        force: bool,
    },
    /// Manage catalog sources (the list in catalogs.yaml).
    Catalog {
        #[command(subcommand)]
        command: ConfigCatalogSubcommand,
    },
    /// List writable mount points to help pick a cache dir.
    Mounts {
        /// Include mounts with less than 100 MiB free.
        #[arg(long)]
        all: bool,
    },
}

/// Catalog-source management under `veks datasets config catalog`.
#[derive(veks_completion_derive::VeksCli)]
pub enum ConfigCatalogSubcommand {
    /// Add a catalog source (URL or path).
    Add {
        /// Catalog URL or path (e.g., https://host/path/ or /local/path).
        source: String,
    },
    /// Remove a catalog source (by URL/path or `--index <N>`).
    #[command(alias = "rm")]
    Remove {
        /// Catalog URL or path to remove.
        source: Option<String>,
        /// Remove by 1-based index (see `config catalog list`).
        #[arg(long)]
        index: Option<usize>,
    },
    /// List configured catalog sources.
    List,
}

/// Dispatch to the appropriate datasets subcommand.
pub fn run(args: DatasetsArgs) {
    match args.command {
        DatasetsCommand::List {
            configdir,
            catalog: raw_catalog,
            at,
            output_format,
            verbose,
            group_by,
            dataset,
            matching_name,
            facet,
            metric,
            matching_desc,
            matching_profile,
            select,
            min_size,
            max_size,
            size,
            min_dim,
            max_dim,
            dim,
            vtype,
            data_min,
            data_max,
            cached,
            cache_dir,
        } => {
            if cached {
                cache::run(cache_dir.as_deref(), verbose);
                return;
            }
            // Resolve numbered catalog shortcuts
            let catalog: Vec<String> = raw_catalog.iter().map(|v| resolve_catalog_value(v)).collect();

            // Normalize glob patterns to regex (--dataset takes precedence)
            let name_filter = dataset.or(matching_name);
            let name = name_filter.map(|p| filter::normalize_match_pattern(&p, "--dataset"));
            let desc = matching_desc.map(|p| filter::normalize_match_pattern(&p, "--matching-desc"));
            let profile_pat = matching_profile.map(|p| filter::normalize_match_pattern(&p, "--matching-profile"));

            let filter = filter::DatasetFilter {
                name,
                facet,
                metric,
                desc,
                min_size: min_size.as_deref().map(filter::parse_size).transpose()
                    .unwrap_or_else(|e| { eprintln!("ERROR: --with-min-size: {}", e); std::process::exit(1); }),
                max_size: max_size.as_deref().map(filter::parse_size).transpose()
                    .unwrap_or_else(|e| { eprintln!("ERROR: --with-max-size: {}", e); std::process::exit(1); }),
                size: size.as_deref().map(filter::parse_size).transpose()
                    .unwrap_or_else(|e| { eprintln!("ERROR: --with-size: {}", e); std::process::exit(1); }),
                min_dim,
                max_dim,
                dim,
                vtype,
                data_min: data_min.as_deref().map(filter::parse_bytes).transpose()
                    .unwrap_or_else(|e| { eprintln!("ERROR: --with-data-min: {}", e); std::process::exit(1); }),
                data_max: data_max.as_deref().map(filter::parse_bytes).transpose()
                    .unwrap_or_else(|e| { eprintln!("ERROR: --with-data-max: {}", e); std::process::exit(1); }),
            };
            let profile_view = filter::ProfileView::new(profile_pat);
            list::run(&configdir, &catalog, &at, &output_format, verbose, group_by.as_deref(), &filter, &profile_view, select.as_deref());
        }
        DatasetsCommand::Curlify { path, output } => {
            curlify::run(&path, output.as_deref());
        }
        DatasetsCommand::Config { command } => {
            run_config_command(command);
        }
        DatasetsCommand::CacheStatus { dataset, all, verbose, tree, configdir, catalog: raw_catalog, at } => {
            let catalog: Vec<String> = raw_catalog.iter().map(|v| resolve_catalog_value(v)).collect();
            if all {
                cache::run_cache_status_all(verbose, &configdir, &catalog, &at);
            } else if let Some(ds) = dataset {
                cache::run_cache_status(&ds, verbose, &configdir, &catalog, &at);
                if tree {
                    let cache_dir = crate::pipeline::commands::config::configured_cache_dir_or_exit();
                    let ds_cache = cache_dir.join(&ds);
                    if ds_cache.is_dir() {
                        println!();
                        println!("  File tree:");
                        cache::print_cache_tree(&ds_cache);
                    }
                }
            } else {
                eprintln!("Specify --dataset <name> or --all");
                std::process::exit(1);
            }
        }
        DatasetsCommand::Ping { at, dataset, profile } => {
            // Ping now goes through the unified resolver; --at pins the
            // search to that catalog, otherwise every configured one is
            // in play and the dataset is looked up against the union.
            let sources = if at.is_empty() {
                crate::catalog::sources::CatalogSources::new().configure_default()
            } else {
                let resolved: Vec<String> = at.iter().map(|a| resolve_catalog_value(a)).collect();
                crate::catalog::sources::CatalogSources::new().add_catalogs(&resolved)
            };
            let catalog = crate::catalog::resolver::Catalog::of(&sources);
            let code = ping::run_via_catalog(&catalog, &dataset, &profile);
            if code != 0 { std::process::exit(code); }
        }
        DatasetsCommand::DropCache { datasets, yes, verbose, cache_dir } => {
            drop_cache::run(&datasets, cache_dir.as_deref(), yes, verbose);
        }
        DatasetsCommand::PruneLegacyCache { dry_run, cache_dir } => {
            let cache_dir = cache_dir
                .unwrap_or_else(|| crate::pipeline::commands::config::configured_cache_dir_or_exit());
            if dry_run {
                println!("Scanning {} for pre-cutover cache directories \
                    (blobs/, http/, <host>[:<port>]/)...",
                    cache_dir.display());
                let mut would_remove = Vec::new();
                if let Ok(entries) = std::fs::read_dir(&cache_dir) {
                    for e in entries.flatten() {
                        if let Some(name) = e.file_name().to_str()
                            && vectordata::cache_admin::is_legacy_layout_dir(name)
                            && e.path().is_dir()
                        {
                            would_remove.push(name.to_string());
                        }
                    }
                }
                if would_remove.is_empty() {
                    println!("No legacy cache directories found.");
                } else {
                    println!("Would remove {} legacy dir(s):", would_remove.len());
                    for name in &would_remove { println!("  {}", name); }
                    println!("\nRun without --dry-run to delete.");
                }
            } else {
                match vectordata::cache_admin::prune_legacy_layout(&cache_dir) {
                    Ok(removed) if removed.is_empty() => {
                        println!("No legacy cache directories found in {}.",
                            cache_dir.display());
                    }
                    Ok(removed) => {
                        println!("Removed {} legacy cache dir(s) from {}:",
                            removed.len(), cache_dir.display());
                        for name in &removed { println!("  {}", name); }
                    }
                    Err(e) => {
                        eprintln!("Error pruning cache: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        DatasetsCommand::Derive {
            dataset, profile, output, name, force, configdir, catalog: raw_catalog, at,
        } => {
            let catalog: Vec<String> = raw_catalog.iter().map(|v| resolve_catalog_value(v)).collect();
            let code = vectordata::datasets::derive::run(
                &dataset, &profile, &output, &configdir, &catalog, &at,
                name.as_deref(), force);
            if code != 0 { std::process::exit(code); }
        }
        DatasetsCommand::Precache { dataset, profile, configdir, catalog: raw_catalog, at, cache_dir } => {
            let catalog: Vec<String> = raw_catalog.iter().map(|v| resolve_catalog_value(v)).collect();
            let ds = match profile {
                Some(p) => format!("{}:{}", dataset.split(':').next().unwrap_or(&dataset), p),
                None => dataset,
            };
            precache::run(&ds, &configdir, &catalog, &at, cache_dir.as_deref());
        }
    }
}

/// Run a `veks config` subcommand. All operations are implemented
/// once in `vectordata::config`; this function only translates the
/// clap-derived enum into the library calls. `veks config` is, in
/// effect, an alias for `vectordata config`.
fn run_config_command(command: ConfigSubcommand) {
    let code = match command {
        ConfigSubcommand::Get { key } => match key.as_deref() {
            None => vectordata::config::show(),
            Some("cache") => vectordata::config::get_cache(),
            Some(other) => {
                eprintln!("unknown config key '{other}' (try `config get cache`, or `config get` for all)");
                2
            }
        },
        ConfigSubcommand::Set { key, value, force } => match key.as_str() {
            "cache" => vectordata::config::set_cache(std::path::Path::new(&value), force),
            other => {
                eprintln!("unknown config key '{other}' (settable keys: cache)");
                2
            }
        },
        ConfigSubcommand::Mounts { all } => vectordata::config::list_mounts(all),
        ConfigSubcommand::Catalog { command } => match command {
            ConfigCatalogSubcommand::Add { source } => vectordata::config::add_catalog(&source),
            ConfigCatalogSubcommand::Remove { source, index } => match (source, index) {
                (Some(s), _) => vectordata::config::remove_catalog(
                    vectordata::config::RemoveCatalogSpec::Source(&s)),
                (None, Some(n)) => vectordata::config::remove_catalog(
                    vectordata::config::RemoveCatalogSpec::Index(n)),
                (None, None) => {
                    eprintln!("Error: specify a catalog URL/path or --index <N>");
                    eprintln!("  Use `veks datasets config catalog list` to see available indices.");
                    1
                }
            },
            ConfigCatalogSubcommand::List => vectordata::config::list_catalogs(),
        },
    };
    if code != 0 { std::process::exit(code); }
}

/// Completer for `--catalog`: suggests configured catalog sources from catalogs.yaml.
/// Completer for `--catalog`: suggests numbered shortcuts for configured
/// Resolve a `--catalog` value: if it's a number, look up the configured
/// catalog by index (1-based). Otherwise use it as a literal URL/path.
pub fn resolve_catalog_value(value: &str) -> String {
    if let Ok(n) = value.parse::<usize>() {
        if n >= 1 {
            let config_dir = crate::catalog::sources::expand_tilde(
                crate::catalog::sources::DEFAULT_CONFIG_DIR,
            );
            let entries = crate::catalog::sources::raw_catalog_entries(&config_dir);
            if let Some(url) = entries.get(n - 1) {
                return url.clone();
            }
        }
        eprintln!("Error: catalog #{} not found. Use 'veks datasets config catalog list' to see available catalogs.", value);
        std::process::exit(1);
    }
    value.to_string()
}

