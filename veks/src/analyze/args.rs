// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! CLI argument types for the `analyze` command group.

use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::import::facet::Facet;

/// Analyze vector data files and datasets
#[derive(Args)]
pub struct AnalyzeArgs {
    #[command(subcommand)]
    pub command: AnalyzeCommand,
}

/// Analyze subcommands
#[derive(Subcommand)]
pub enum AnalyzeCommand {
    /// Describe a vector file or dataset facet — dimensions, record count,
    /// element type, normalization status, and dot-product compatibility
    Describe(DescribeArgs),
}

/// Arguments for `veks analyze describe`
#[derive(Args)]
pub struct DescribeArgs {
    /// File path, directory, or dataset.yaml to describe
    pub source: PathBuf,

    /// When source is a dataset.yaml, which facet to describe.
    /// If omitted, describes all facets.
    #[arg(long)]
    pub facet: Option<Facet>,

    /// Format override (auto-detected if omitted)
    #[arg(long)]
    pub from: Option<String>,
}
