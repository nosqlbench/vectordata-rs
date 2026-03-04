// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

mod analyze;
mod bulkdl;
mod cli;
mod convert;
mod formats;
mod import;
mod pipeline;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::CompleteEnv;

/// Vecs — umbrella CLI for vector data tools
#[derive(Parser)]
#[command(name = "vecs", version, about)]
struct Vecs {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze vector data files and datasets
    Analyze(analyze::AnalyzeArgs),
    /// Bulk file downloader driven by YAML config with token expansion
    Bulkdl(bulkdl::BulkDlArgs),
    /// Convert vector data between formats
    Convert(convert::ConvertArgs),
    /// Import data into preferred internal format by facet type
    Import(import::ImportArgs),
    /// Execute a command stream pipeline defined in dataset.yaml
    Run(pipeline::RunArgs),
    /// Execute a single pipeline command directly
    Pipeline {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Generate shell completions
    Completions(cli::CompletionsArgs),
}

/// Build the augmented CLI command tree with dynamic pipeline subcommands
/// for shell completion support.
fn build_augmented_cli() -> clap::Command {
    let mut cmd = Vecs::command();
    // Replace the derive-generated `pipeline` stub with the full dynamic tree.
    cmd = cmd.mut_subcommand("pipeline", |_| {
        pipeline::cli::build_pipeline_command()
    });
    cmd
}

#[tokio::main]
async fn main() {
    CompleteEnv::with_factory(build_augmented_cli).complete();

    let vecs = Vecs::parse();

    match vecs.command {
        Commands::Analyze(args) => analyze::run(args),
        Commands::Bulkdl(args) => bulkdl::run(args).await,
        Commands::Convert(args) => convert::run(args),
        Commands::Import(args) => import::run(args),
        Commands::Run(args) => pipeline::run_pipeline(args),
        Commands::Pipeline { args } => pipeline::cli::run_direct(args),
        Commands::Completions(args) => cli::completions(args),
    }
}
