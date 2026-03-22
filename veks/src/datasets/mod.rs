// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Root-level `veks datasets` command group.
//!
//! Provides inventory and exploration of datasets addressable by the catalog
//! system. Subcommands: list, plan, cache, curlify, prebuffer, catalog.
//!
//! These are standalone commands — no pipeline context or StreamContext required.

mod cache;
mod cache_compress;
mod curlify;
pub mod filter;
pub mod import;
mod list;
mod prebuffer;
mod wizard;

use std::path::PathBuf;

use clap::{Args, Subcommand};
use clap_complete::engine::ArgValueCompleter;

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

        /// Output format: text, csv, json, yaml
        #[arg(long = "output-format", short = 'f', default_value = "text")]
        output_format: String,

        /// Show detailed information including attributes, tags, and views
        #[arg(long, short = 'v')]
        verbose: bool,

        /// Limit output to profiles matching this name (exact, case-insensitive)
        #[arg(long, add = ArgValueCompleter::new(filter::profile_completer))]
        profile: Option<String>,

        /// Limit output to profiles matching this regex pattern
        #[arg(long, add = ArgValueCompleter::new(filter::profile_completer))]
        profile_regex: Option<String>,

        /// Select a single dataset:profile; fails if the filters are ambiguous
        #[arg(long, add = ArgValueCompleter::new(filter::select_completer))]
        select: Option<String>,

        // -- Filter predicates (--with-* prefix) --

        /// Filter by dataset name (substring match, case-insensitive)
        #[arg(long = "with-name", add = ArgValueCompleter::new(filter::name_completer))]
        name: Option<String>,

        /// Filter by dataset name (regex pattern)
        #[arg(long = "with-name-regex", add = ArgValueCompleter::new(filter::name_completer))]
        name_regex: Option<String>,

        /// Filter: dataset must contain this facet/view
        #[arg(long = "with-facet", add = ArgValueCompleter::new(filter::facet_completer))]
        facet: Vec<String>,

        /// Filter: dataset must use this distance metric
        #[arg(long = "with-metric", add = ArgValueCompleter::new(filter::metric_completer))]
        metric: Option<String>,

        /// Filter: description/notes/name contains this word (case-insensitive)
        #[arg(long = "with-desc", add = ArgValueCompleter::new(filter::desc_completer))]
        desc: Option<String>,

        /// Filter: description/notes/name matches this regex pattern
        #[arg(long = "with-desc-regex", add = ArgValueCompleter::new(filter::desc_completer))]
        desc_regex: Option<String>,

        /// Filter: dataset has at least this many base vectors (supports K/M/B suffixes)
        #[arg(long = "with-min-size", add = ArgValueCompleter::new(filter::size_completer))]
        min_size: Option<String>,

        /// Filter: dataset has at most this many base vectors (supports K/M/B suffixes)
        #[arg(long = "with-max-size", add = ArgValueCompleter::new(filter::size_completer))]
        max_size: Option<String>,

        /// Filter: dataset has exactly this many base vectors (supports K/M/B suffixes)
        #[arg(long = "with-size", add = ArgValueCompleter::new(filter::size_completer))]
        size: Option<String>,

        /// Filter: minimum dimensionality of base vectors
        #[arg(long = "with-min-dim", add = ArgValueCompleter::new(filter::dim_completer))]
        min_dim: Option<u32>,

        /// Filter: maximum dimensionality of base vectors
        #[arg(long = "with-max-dim", add = ArgValueCompleter::new(filter::dim_completer))]
        max_dim: Option<u32>,

        /// Filter: exact dimensionality of base vectors
        #[arg(long = "with-dim", add = ArgValueCompleter::new(filter::dim_completer))]
        dim: Option<u32>,

        /// Filter: vector data type (float32, float16, uint8, int32, numpy, hdf5)
        #[arg(long = "with-vtype", add = ArgValueCompleter::new(filter::vtype_completer))]
        vtype: Option<String>,

        /// Filter: minimum total data size in bytes (supports K/M/G/T suffixes)
        #[arg(long = "with-data-min", add = ArgValueCompleter::new(filter::data_size_completer))]
        data_min: Option<String>,

        /// Filter: maximum total data size in bytes (supports K/M/G/T suffixes)
        #[arg(long = "with-data-max", add = ArgValueCompleter::new(filter::data_size_completer))]
        data_max: Option<String>,
    },
    /// List locally cached datasets
    Cache {
        /// Override cache directory location
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Show per-file details
        #[arg(long)]
        verbose: bool,
    },
    /// Compress eligible cache files to save disk space
    CacheCompress {
        /// Cache directory to compress (default: .cache/ in current directory)
        #[arg(default_value = ".cache")]
        cache_dir: PathBuf,

        /// Compression level (0-9, default 9 = maximum)
        #[arg(long, default_value = "9")]
        level: u32,

        /// Dry run — show what would be compressed without changing files
        #[arg(long)]
        dry_run: bool,
    },
    /// Decompress cache files back to their original form
    CacheUncompress {
        /// Cache directory to uncompress (default: .cache/ in current directory)
        #[arg(default_value = ".cache")]
        cache_dir: PathBuf,

        /// Dry run — show what would be uncompressed without changing files
        #[arg(long)]
        dry_run: bool,
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
    /// Download and cache dataset facets locally
    Prebuffer {
        /// Dataset specifier: dataset:profile from catalog, or local path
        #[arg(long, add = ArgValueCompleter::new(filter::select_completer))]
        dataset: String,

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
    /// Bootstrap a new dataset directory from source files
    Import {
        /// Interactive wizard mode — prompts for each option
        #[arg(long, short = 'i')]
        interactive: bool,

        /// Accept all defaults without prompting (use with -i)
        #[arg(long, short = 'y')]
        yes: bool,

        /// Dataset name (required unless --interactive)
        #[arg(long)]
        name: Option<String>,

        /// Output directory for the new dataset (required unless --interactive)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        /// Path to base vectors (file or directory)
        #[arg(long)]
        base_vectors: Option<PathBuf>,

        /// Path to separate query vectors (file or directory)
        #[arg(long)]
        query_vectors: Option<PathBuf>,

        /// Extract queries from base via shuffle (default when no --query-vectors)
        #[arg(long)]
        self_search: bool,

        /// Number of query vectors in self-search mode
        #[arg(long, default_value = "10000")]
        query_count: u32,

        /// Path to metadata (file or directory)
        #[arg(long)]
        metadata: Option<PathBuf>,

        /// Pre-computed ground truth indices (ivec file)
        #[arg(long)]
        ground_truth: Option<PathBuf>,

        /// Pre-computed ground truth distances (fvec file)
        #[arg(long)]
        ground_truth_distances: Option<PathBuf>,

        /// Distance metric for KNN computation
        #[arg(long, default_value = "L2")]
        metric: String,

        /// Number of neighbors for KNN ground truth
        #[arg(long, default_value = "100")]
        neighbors: u32,

        /// Random seed for shuffle
        #[arg(long, default_value = "42")]
        seed: u32,

        /// Dataset description
        #[arg(long)]
        description: Option<String>,

        /// Skip deduplication stage
        #[arg(long)]
        no_dedup: bool,

        /// Skip zero-vector check and clean ordinals
        #[arg(long)]
        no_zero_check: bool,

        /// Skip filtered KNN even when metadata is present
        #[arg(long)]
        no_filtered: bool,

        /// L2-normalize vectors during extraction
        #[arg(long)]
        normalize: bool,

        /// Overwrite existing dataset.yaml
        #[arg(long)]
        force: bool,

        /// Start fresh — ignore existing dataset.yaml, variables.yaml,
        /// and .cache state. Equivalent to --force but also removes
        /// variables.yaml and the progress log.
        #[arg(long)]
        restart: bool,
    },
    /// Generate and manage dataset catalog index files
    #[command(disable_help_subcommand = true)]
    Catalog {
        #[command(subcommand)]
        command: CatalogSubcommand,
    },
}

/// Subcommands under `veks datasets catalog`.
#[derive(Subcommand)]
pub enum CatalogSubcommand {
    /// Generate catalog.json and catalog.yaml from dataset directories
    Generate {
        /// Root directory to scan for datasets (default: current directory)
        #[arg(default_value = ".")]
        input: PathBuf,

        /// Base filename for catalog files (without extension)
        #[arg(long, default_value = "catalog")]
        basename: String,

        /// Walk up to the .publish_url root and generate catalogs for the
        /// entire publish hierarchy. Creates catalog files at every directory
        /// level from the publish root down to each dataset.
        #[arg(long)]
        for_publish_url: bool,

        /// Update only catalog files that already exist in the hierarchy
        /// (default behavior). Skips directories that have no existing catalog.
        /// Use --no-update to generate at all hierarchy levels.
        #[arg(long, overrides_with = "no_update", default_value_t = true)]
        update: bool,

        /// Disable --update: generate catalogs at all hierarchy levels,
        /// not just where catalog files already exist.
        #[arg(long = "no-update", hide = true)]
        no_update: bool,
    },
}

/// Dispatch to the appropriate datasets subcommand.
pub fn run(args: DatasetsArgs) {
    match args.command {
        DatasetsCommand::List {
            configdir,
            catalog,
            at,
            output_format,
            verbose,
            name,
            name_regex,
            facet,
            metric,
            desc,
            desc_regex,
            profile,
            profile_regex,
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
        } => {
            // Parse size values
            let filter = filter::DatasetFilter {
                name,
                name_regex,
                facet,
                metric,
                desc,
                desc_regex,
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
            let profile_view = filter::ProfileView::new(profile, profile_regex);
            list::run(&configdir, &catalog, &at, &output_format, verbose, &filter, &profile_view, select.as_deref());
        }
        DatasetsCommand::Cache { cache_dir, verbose } => {
            cache::run(cache_dir.as_deref(), verbose);
        }
        DatasetsCommand::CacheCompress { cache_dir, level, dry_run } => {
            crate::pipeline::gz_cache::set_compression_level(level);
            cache_compress::run(&cache_dir, dry_run);
        }
        DatasetsCommand::CacheUncompress { cache_dir, dry_run } => {
            cache_compress::run_uncompress(&cache_dir, dry_run);
        }
        DatasetsCommand::Curlify { path, output } => {
            curlify::run(&path, output.as_deref());
        }
        DatasetsCommand::Prebuffer { dataset, configdir, catalog, at, cache_dir } => {
            prebuffer::run(&dataset, &configdir, &catalog, &at, cache_dir.as_deref());
        }
        DatasetsCommand::Import {
            interactive, yes, name, output, base_vectors, query_vectors,
            self_search, query_count, metadata, ground_truth,
            ground_truth_distances, metric, neighbors, seed, description,
            no_dedup, no_zero_check, no_filtered, normalize, force, restart,
        } => {
            if restart {
                // Clean up state files before proceeding
                let out_dir = output.as_deref()
                    .unwrap_or_else(|| std::path::Path::new("."));
                let _ = std::fs::remove_file(out_dir.join("dataset.yaml"));
                let _ = std::fs::remove_file(out_dir.join("variables.yaml"));
                let _ = std::fs::remove_file(out_dir.join(".cache/.upstream.progress.yaml"));
                eprintln!("Restarting: removed dataset.yaml, variables.yaml, and progress log");
            }
            if interactive {
                let args = wizard::run_wizard_with_auto_accept(yes);
                import::run(args);
            } else {
                let name = name.unwrap_or_else(|| {
                    eprintln!("Error: --name is required (or use --interactive)");
                    std::process::exit(1);
                });
                let output = output.unwrap_or_else(|| {
                    eprintln!("Error: --output is required (or use --interactive)");
                    std::process::exit(1);
                });
                import::run(import::ImportArgs {
                    name, output, base_vectors, query_vectors, self_search,
                    query_count, metadata, ground_truth, ground_truth_distances,
                    metric, neighbors, seed, description, no_dedup, no_zero_check,
                    no_filtered, normalize, force: force || restart,
                    base_convert_format: None,
                    query_convert_format: None,
                    compress_cache: true,
                    sized_profiles: None,
                });
            }
        }
        DatasetsCommand::Catalog { command } => {
            match command {
                CatalogSubcommand::Generate { input, basename, for_publish_url, update, no_update } => {
                    let effective_update = update && !no_update;
                    crate::catalog::generate::run(&input, &basename, for_publish_url, effective_update);
                }
            }
        }
    }
}
