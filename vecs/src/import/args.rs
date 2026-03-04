// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! CLI arguments for the import subcommand.

use std::path::PathBuf;

use clap::Args;

use super::facet::Facet;

/// Arguments for the import subcommand.
///
/// Import operates on individual facets, converting source data into the
/// preferred internal format for each facet type. It can import a single facet
/// at a time, or process a full dataset.yaml with upstream sources defined.
///
/// In single-facet mode, `source` and `--facet` are required.
/// In dataset mode (`--dataset`), upstream sources come from the YAML.
/// In scaffold mode (`--scaffold`), no source is needed.
#[derive(Args)]
pub struct ImportArgs {
    /// Source file or directory to import (required for single-facet mode)
    pub source: Option<PathBuf>,

    /// The facet type of this data
    #[arg(long, value_enum)]
    pub facet: Option<Facet>,

    /// Output file path (auto-generated from facet name if omitted)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Dataset YAML config file (for multi-facet import)
    #[arg(long)]
    pub dataset: Option<PathBuf>,

    /// Generate a scaffold dataset.yaml with the given name
    #[arg(long)]
    pub scaffold: Option<String>,

    /// Re-import even if the output file already exists and is complete
    #[arg(long)]
    pub force: bool,

    /// Source format override (auto-detected if omitted)
    #[arg(long)]
    pub from: Option<String>,

    /// Slab page size for slab outputs (bytes, must be multiple of 512).
    /// Defaults to the slabtastic library default (currently 4 MiB).
    #[arg(long)]
    pub slab_page_size: Option<u32>,

    /// Slab namespace index for slab outputs
    #[arg(long, default_value = "1")]
    pub slab_namespace: u8,

    /// Number of loader threads for parallel readers (0 = auto-detect from
    /// available CPU parallelism)
    #[arg(long, default_value = "0")]
    pub threads: usize,
}
