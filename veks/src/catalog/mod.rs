// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Root-level `veks catalog` command group.
//!
//! Provides catalog generation for dataset directories. Catalogs are
//! hierarchical index files (`catalog.json`, `catalog.yaml`) that describe
//! all datasets discoverable under a directory tree.
//!
//! The output uses the layout-embedded entry format: each entry contains a
//! `layout` block with `attributes` and `profiles`, a `path` relative to the
//! catalog file, a `name`, and a `dataset_type` of `"dataset.yaml"`.

mod generate;
pub mod resolver;
pub mod sources;

use std::path::PathBuf;

use clap::{Args, Subcommand};

/// Generate and manage dataset catalog index files
#[derive(Args)]
pub struct CatalogArgs {
    #[command(subcommand)]
    pub command: CatalogCommand,
}

#[derive(Subcommand)]
pub enum CatalogCommand {
    /// Generate catalog.json and catalog.yaml from dataset directories
    Generate {
        /// Root directory to scan for datasets (default: current directory)
        #[arg(default_value = ".")]
        input: PathBuf,

        /// Base filename for catalog files (without extension)
        #[arg(long, default_value = "catalog")]
        basename: String,
    },
}

/// Dispatch to the appropriate catalog subcommand.
pub fn run(args: CatalogArgs) {
    match args.command {
        CatalogCommand::Generate { input, basename } => {
            generate::run(&input, &basename);
        }
    }
}
