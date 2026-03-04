// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::Args;

use crate::formats::VecFormat;

/// Arguments for the convert subcommand
#[derive(Args)]
pub struct ConvertArgs {
    /// Source file or directory
    pub source: PathBuf,

    /// Output file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Source format (auto-detected if omitted)
    #[arg(long, value_enum)]
    pub from: Option<VecFormat>,

    /// Output format
    #[arg(long, value_enum)]
    pub to: VecFormat,

    /// Preferred page size for slab output (bytes, must be multiple of 512).
    /// Defaults to the slabtastic library default (currently 4 MiB).
    #[arg(long)]
    pub slab_page_size: Option<u32>,

    /// Namespace index for slab output
    #[arg(long, default_value = "1")]
    pub slab_namespace: u8,
}
