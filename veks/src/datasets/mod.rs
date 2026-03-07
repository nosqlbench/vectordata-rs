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
mod list;
mod plan;
mod prebuffer;

use std::path::PathBuf;

use clap::{Args, Subcommand};

/// Browse, search, and manage datasets
#[derive(Args)]
pub struct DatasetsArgs {
    #[command(subcommand)]
    pub command: DatasetsCommand,
}

#[derive(Subcommand)]
pub enum DatasetsCommand {
    /// List available datasets from a directory or catalog
    List {
        /// Directory to scan for datasets (default: current directory)
        #[arg(default_value = ".")]
        catalog: PathBuf,

        /// Output format: text, json, yaml
        #[arg(long, default_value = "text")]
        format: String,

        /// Show detailed information
        #[arg(long)]
        verbose: bool,
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
        DatasetsCommand::List { catalog, format, verbose } => {
            list::run(&catalog, &format, verbose);
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
