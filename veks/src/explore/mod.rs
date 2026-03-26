// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks explore` — interactive visualization and exploration commands.
//!
//! These are user-facing TUI/interactive tools that don't belong in the
//! pipeline command registry (they're not composable pipeline steps).
//! The source argument accepts either a local file path or a
//! `dataset:profile:facet` specifier from the catalog.

pub mod shared;
mod data_shell;
mod dataset_picker;
mod unified;

use std::path::PathBuf;

use clap::{Args, Subcommand};
use clap_complete::engine::ArgValueCompleter;

pub use shared::SampleMode;
use shared::{parse_sample_mode, source_completer, dataset_completer};

/// Interactive data exploration and visualization
#[derive(Args)]
#[command(disable_help_subcommand = true)]
pub struct ExploreArgs {
    #[command(subcommand)]
    pub command: ExploreCommand,
}

#[derive(Subcommand)]
pub enum ExploreCommand {
    /// Unified vector space explorer — norms, distances, eigenvalues, PCA in one TUI
    Explore {
        /// Dataset from catalog (e.g., img-search or img-search:default)
        #[arg(long, group = "input", add = ArgValueCompleter::new(dataset_completer))]
        dataset: Option<String>,
        /// Any data source: local file path or dataset:profile:facet
        #[arg(long, group = "input", add = ArgValueCompleter::new(source_completer))]
        source: Option<String>,
        /// Profile name (used with --dataset; overrides profile in dataset:profile)
        #[arg(long, add = ArgValueCompleter::new(shared::profile_completer))]
        profile: Option<String>,
        /// Number of vectors to sample
        #[arg(long, default_value = "50000")]
        sample: usize,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Sampling mode [streaming, clumped, sparse]
        #[arg(long, default_value = "streaming", value_parser = parse_sample_mode,
              add = ArgValueCompleter::new(shared::sample_mode_completer))]
        sample_mode: SampleMode,
    },
    /// Interactive data exploration shell for vector files
    Shell {
        /// Dataset from catalog (e.g., img-search or img-search:default)
        #[arg(long, group = "input", add = ArgValueCompleter::new(dataset_completer))]
        dataset: Option<String>,
        /// Any data source: local file path or dataset:profile:facet
        #[arg(long, group = "input", add = ArgValueCompleter::new(source_completer))]
        source: Option<String>,
        /// Profile name (used with --dataset; overrides profile in dataset:profile)
        #[arg(long, add = ArgValueCompleter::new(shared::profile_completer))]
        profile: Option<String>,

        /// Trailing args passed as command options
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

/// Resolve the data source from mutually exclusive --dataset / --source options.
///
/// When `--profile` is given with `--dataset`, it's appended as `dataset:profile`.
/// If the dataset already contains a `:`, the explicit `--profile` overrides it.
fn resolve_input(dataset: Option<String>, source: Option<String>, profile: Option<String>) -> Option<String> {
    let base = match (dataset, source) {
        (Some(ds), None) => ds,
        (None, Some(src)) => src,
        (None, None) => return None,
        (Some(_), Some(_)) => unreachable!("clap group ensures mutual exclusion"),
    };

    Some(match profile {
        Some(p) => {
            let name = base.split(':').next().unwrap_or(&base);
            format!("{}:{}", name, p)
        }
        None => {
            if !base.contains(':') && !base.contains('/') && !base.contains('.') {
                format!("{}:default", base)
            } else {
                base
            }
        }
    })
}

/// Dispatch a visualize subcommand.
///
/// Visualize commands create a standalone ratatui TUI for interactive
/// display. The user presses 'q' to exit.
pub fn run(args: ExploreArgs) {
    // Install a panic hook that restores the terminal before printing the
    // panic message. Without this, panics during TUI sessions leave the
    // terminal in raw mode with the alternate screen active, hiding errors.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(
            std::io::stdout(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::cursor::Show,
        );
        original_hook(info);
    }));

    match args.command {
        ExploreCommand::Explore { dataset, source, profile, sample, seed, sample_mode } => {
            let from_picker = dataset.is_none() && source.is_none();
            let mut src = match resolve_input(dataset, source, profile) {
                Some(s) => s,
                None => match dataset_picker::run_picker() {
                    Some(s) => s,
                    None => { std::process::exit(0); }
                },
            };
            loop {
                match unified::run_interactive_explore(&src, sample, seed, sample_mode) {
                    unified::ExploreExit::Quit => break,
                    unified::ExploreExit::Back if from_picker => {
                        // Return to dataset picker
                        match dataset_picker::run_picker() {
                            Some(s) => { src = s; }
                            None => break,
                        }
                    }
                    unified::ExploreExit::Back => break, // no picker to go back to
                }
            }
        }
        ExploreCommand::Shell { dataset, source, profile, args: extra } => {
            let src = match resolve_input(dataset, source, profile) {
                Some(s) => s,
                None => match dataset_picker::run_picker() {
                    Some(s) => s,
                    None => { std::process::exit(0); }
                },
            };
            if extra.is_empty() {
                data_shell::run_data_shell_interactive(&src);
            } else {
                let commands = extra.join(" ");
                data_shell::run_data_shell_batch(&src, &commands);
            }
        }
    }
}

/// Run a pipeline command with a plain (non-TUI) sink.
#[allow(dead_code)]
fn run_pipeline_command(
    mut cmd: Box<dyn crate::pipeline::command::CommandOp>,
    source: &str,
    extra_args: &[String],
) {
    use crate::pipeline::command::{Options, StreamContext, Status};
    use crate::pipeline::progress::ProgressLog;
    use crate::pipeline::resource::ResourceGovernor;
    use indexmap::IndexMap;

    let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let mut opts = Options::new();
    opts.set("source", source);
    for arg in extra_args {
        if let Some(kv) = arg.strip_prefix("--") {
            if let Some((k, v)) = kv.split_once('=') {
                opts.set(k, v);
            } else {
                opts.set(kv, "true");
            }
        }
    }

    let mut ctx = StreamContext {
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace: workspace.clone(),
        scratch: workspace.join(".scratch"),
        cache: workspace.join(".cache"),
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads: 0,
        step_id: String::new(),
        governor: ResourceGovernor::default_governor(),
        ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::PlainSink::new())),
        status_interval: std::time::Duration::from_secs(1),
        estimated_total_steps: 0,
    };

    let result = cmd.execute(&opts, &mut ctx);
    if result.status == Status::Error {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
}
