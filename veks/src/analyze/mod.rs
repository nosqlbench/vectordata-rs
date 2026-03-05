// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Analyze command group — inspect and describe vector data.
//!
//! Subcommands:
//! - `describe` — report dimensions, record count, element type,
//!   normalization status, and dot-product compatibility

mod args;
mod describe;

pub use args::AnalyzeArgs;
use args::AnalyzeCommand;

/// Dispatch to the appropriate analyze subcommand.
pub fn run(args: AnalyzeArgs) {
    match args.command {
        AnalyzeCommand::Describe(desc_args) => describe::run(desc_args),
    }
}
