// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Root-level `veks datasets` command group.
//!
//! Provides inventory and exploration of datasets addressable by the catalog
//! system. Subcommands: list, plan, cache, curlify, prebuffer, catalog.
//!
//! These are standalone commands — no pipeline context or StreamContext required.

mod cache;
mod curlify;
pub mod filter;
mod list;
pub(crate) mod prebuffer;
mod probe;

use std::path::PathBuf;

use clap::{Args, Subcommand};

/// Browse, search, and manage datasets
#[derive(Args)]
#[command(disable_help_subcommand = true)]
pub struct DatasetsArgs {
    #[command(subcommand)]
    pub command: DatasetsCommand,
}

/// Available subcommands under `veks datasets`.
#[derive(Subcommand)]
pub enum DatasetsCommand {
    /// List available datasets from configured or specified catalogs
    #[command(alias = "ls")]
    List {
        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value = "~/.config/vectordata")]
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
        #[arg(long, short = 'v')]
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

        /// Filter by dataset name (substring, regex, or glob)
        #[arg(long = "matching-name")]
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
        #[arg(long, short = 'v')]
        verbose: bool,

        /// Show file tree of the cache directory
        #[arg(long)]
        tree: bool,

        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value = "~/.config/vectordata")]
        configdir: String,

        /// Additional catalog directories, file paths, or HTTP URLs
        #[arg(long)]
        catalog: Vec<String>,

        /// Catalog URLs or paths to use *instead* of configured catalogs
        #[arg(long = "at")]
        at: Vec<String>,
    },
    /// Probe a remote dataset: verify catalog access, list facets, read a sample
    Probe {
        /// Catalog base URL or number (e.g., 1, https://bucket.s3.amazonaws.com/path/)
        #[arg(long)]
        at: String,

        /// Dataset name within the catalog
        #[arg(long)]
        dataset: String,

        /// Profile to probe (default: "default")
        #[arg(long, default_value = "default")]
        profile: String,
    },
    /// Download and cache dataset facets locally
    Prebuffer {
        /// Dataset name or dataset:profile from catalog
        #[arg(long)]
        dataset: String,

        /// Profile name (overrides profile in dataset:profile)
        #[arg(long)]
        profile: Option<String>,

        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value = "~/.config/vectordata")]
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

/// Subcommands under `veks datasets config`.
#[derive(Subcommand)]
pub enum ConfigSubcommand {
    /// Display current vectordata configuration
    Show,
    /// Set the cache directory for downloaded datasets
    SetCache {
        /// Cache directory path (e.g., /mnt/data/vectordata-cache)
        #[arg(long = "cache-dir")]
        cache_dir: Option<String>,

        /// Overwrite existing protected settings
        #[arg(long)]
        force: bool,
    },
    /// List configured dataset mount points
    ListMounts,
    /// Add a catalog source (URL or path) to catalogs.yaml
    #[command(alias = "add")]
    AddCatalog {
        /// Catalog URL or path (e.g., https://host/path/ or /local/path)
        source: String,
    },
    /// Remove a catalog source from catalogs.yaml
    #[command(alias = "rm")]
    RemoveCatalog {
        /// Catalog URL or path to remove
        source: String,
    },
    /// List configured catalog sources
    ListCatalogs,
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

            // Normalize glob patterns to regex
            let name = matching_name.map(|p| filter::normalize_match_pattern(&p, "--matching-name"));
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
                    let cache_dir = crate::pipeline::commands::config::configured_cache_dir();
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
        DatasetsCommand::Probe { at, dataset, profile } => {
            let resolved_at = resolve_catalog_value(&at);
            probe::run(&resolved_at, &dataset, &profile);
        }
        DatasetsCommand::Prebuffer { dataset, profile, configdir, catalog: raw_catalog, at, cache_dir } => {
            let catalog: Vec<String> = raw_catalog.iter().map(|v| resolve_catalog_value(v)).collect();
            let ds = match profile {
                Some(p) => format!("{}:{}", dataset.split(':').next().unwrap_or(&dataset), p),
                None => dataset,
            };
            prebuffer::run(&ds, &configdir, &catalog, &at, cache_dir.as_deref());
        }
    }
}

/// Run a config subcommand by dispatching to the pipeline CommandOp.
fn run_config_command(command: ConfigSubcommand) {
    use crate::pipeline::command::{CommandOp, Options, StreamContext, Status};
    use crate::pipeline::progress::ProgressLog;
    use crate::pipeline::resource::ResourceGovernor;
    use crate::pipeline::commands::config;
    use indexmap::IndexMap;

    let (mut cmd, extra_opts): (Box<dyn CommandOp>, Vec<(&str, String)>) = match command {
        ConfigSubcommand::Show => (config::show_factory(), vec![]),
        ConfigSubcommand::SetCache { cache_dir, force } => {
            let mut opts = Vec::new();
            if let Some(dir) = cache_dir {
                opts.push(("cache-dir", dir));
            }
            if force {
                opts.push(("force", "true".to_string()));
            }
            (config::init_factory(), opts)
        }
        ConfigSubcommand::ListMounts => (config::list_mounts_factory(), vec![]),
        ConfigSubcommand::AddCatalog { source } => {
            run_catalog_config("add", &source);
            return;
        }
        ConfigSubcommand::RemoveCatalog { source } => {
            run_catalog_config("remove", &source);
            return;
        }
        ConfigSubcommand::ListCatalogs => {
            run_catalog_config("list", "");
            return;
        }
    };

    let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut ctx = StreamContext {
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace: workspace.clone(),
        scratch: workspace.join(".scratch"),
        cache: workspace.join(".cache"),
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads: 0,
        step_id: String::new(),
        governor: ResourceGovernor::default_governor(),
        ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::PlainSink::new())),
        status_interval: std::time::Duration::from_secs(1),
        estimated_total_steps: 0,
    };

    let mut opts = Options::new();
    for (k, v) in extra_opts {
        opts.set(k, &v);
    }
    let result = cmd.execute(&opts, &mut ctx);
    if result.status == Status::Error {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
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
        eprintln!("Error: catalog #{} not found. Use 'veks datasets config list-catalogs' to see available catalogs.", value);
        std::process::exit(1);
    }
    value.to_string()
}

/// Manage catalog sources in `~/.config/vectordata/catalogs.yaml`.
fn run_catalog_config(action: &str, source: &str) {
    let config_dir = crate::catalog::sources::expand_tilde(
        crate::catalog::sources::DEFAULT_CONFIG_DIR,
    );
    let catalogs_path = std::path::Path::new(&config_dir).join("catalogs.yaml");

    // Load existing entries (or start empty)
    let mut entries: Vec<String> = if catalogs_path.is_file() {
        let content = std::fs::read_to_string(&catalogs_path).unwrap_or_default();
        serde_yaml::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };

    match action {
        "add" => {
            if entries.iter().any(|e| e == source) {
                println!("Already configured: {}", source);
                return;
            }

            // Verify the source is reachable and contains a catalog
            let sources = crate::catalog::sources::CatalogSources::new()
                .add_catalogs(&[source.to_string()]);
            let catalog = crate::catalog::resolver::Catalog::of(&sources);
            let count = catalog.datasets().len();

            if count > 0 {
                println!("{} {} ({} dataset(s))",
                    crate::term::ok("OK"),
                    source,
                    count);
            } else {
                eprintln!("{} {} — no datasets found at this location",
                    crate::term::fail("FAIL"),
                    source);
                std::process::exit(1);
            }

            entries.push(source.to_string());
            // Ensure config directory exists
            if let Some(parent) = catalogs_path.parent() {
                std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                    eprintln!("Error creating {}: {}", parent.display(), e);
                    std::process::exit(1);
                });
            }
            let yaml = serde_yaml::to_string(&entries).unwrap();
            std::fs::write(&catalogs_path, &yaml).unwrap_or_else(|e| {
                eprintln!("Error writing {}: {}", catalogs_path.display(), e);
                std::process::exit(1);
            });
            println!("Saved to {}", catalogs_path.display());
        }
        "remove" => {
            let before = entries.len();
            entries.retain(|e| e != source);
            if entries.len() == before {
                eprintln!("Not found: {}", source);
                std::process::exit(1);
            }
            let yaml = serde_yaml::to_string(&entries).unwrap();
            std::fs::write(&catalogs_path, &yaml).unwrap_or_else(|e| {
                eprintln!("Error writing {}: {}", catalogs_path.display(), e);
                std::process::exit(1);
            });
            println!("Removed: {}", source);
        }
        "list" => {
            if entries.is_empty() {
                println!("No catalog sources configured.");
                println!();
                println!("Add one with:");
                println!("  veks datasets config add-catalog <URL-or-path>");
            } else {
                println!("Configured catalog sources ({}):", entries.len());
                for entry in &entries {
                    println!("  {}", entry);
                }
                println!();
                println!("Catalogs file: {}", catalogs_path.display());
            }
        }
        _ => unreachable!(),
    }
}
