// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::Args;

/// Arguments for the bulkdl subcommand
#[derive(Args)]
pub struct BulkDlArgs {
    /// Path to the YAML configuration file
    pub config: PathBuf,
}
