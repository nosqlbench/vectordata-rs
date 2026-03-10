// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Root-level `veks datasets` command group.
//!
//! Provides inventory and exploration of datasets addressable by the catalog
//! system. Subcommands: list, plan, cache, curlify, prebuffer.
//!
//! These are standalone commands — no pipeline context or StreamContext required.

mod cache;
mod curlify;
pub mod filter;
mod list;
mod plan;
mod prebuffer;

use std::path::PathBuf;

use clap::{Args, Subcommand};
use clap_complete::engine::ArgValueCompleter;

/// Browse, search, and manage datasets
#[derive(Args)]
pub struct DatasetsArgs {
    #[command(subcommand)]
    pub command: DatasetsCommand,
}

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
    /// Show which facets are present or missing for a dataset
    Plan {
        /// Dataset directory or path to dataset.yaml
        #[arg(default_value = ".")]
        path: PathBuf,
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
        /// Dataset directory or path to dataset.yaml
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Override cache directory location
        #[arg(long)]
        cache_dir: Option<PathBuf>,
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
        DatasetsCommand::Plan { path } => {
            plan::run(&path);
        }
        DatasetsCommand::Cache { cache_dir, verbose } => {
            cache::run(cache_dir.as_deref(), verbose);
        }
        DatasetsCommand::Curlify { path, output } => {
            curlify::run(&path, output.as_deref());
        }
        DatasetsCommand::Prebuffer { path, cache_dir } => {
            prebuffer::run(&path, cache_dir.as_deref());
        }
    }
}
