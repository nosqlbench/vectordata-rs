// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata explore` — interactive visualization and exploration.
//!
//! Owns the ratatui-based dataset browser, raw-values grid, and REPL
//! command engine. Originally lived in `veks/src/explore/`; migrated
//! into vectordata to make `vectordata explore` self-contained — the
//! TUI never had a real pipeline-command-framework dependency (the
//! wiring that looked like one was dead code).
//!
//! The source argument accepts either a local file path or a
//! `dataset:profile:facet` specifier from the catalog.

pub mod shared;
pub mod repl;
mod data_shell;
mod dataset_picker;
mod palette;
mod unified;
mod values_grid;

/// Resolve the configured cache directory or exit the process with a
/// helpful error message. Used as the entry-point fallback when the
/// explore TUI needs to know where remote-cache blobs live and has
/// nowhere sensible to proceed without one.
pub(crate) fn cache_dir_or_exit() -> std::path::PathBuf {
    match crate::settings::cache_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: cannot resolve cache_dir from settings: {e}");
            std::process::exit(1);
        }
    }
}

/// Deterministic seeded RNG used by sampling code in the explore
/// TUI. Mirrors the `veks-pipeline::rng::seeded_rng` shape that the
/// pre-migration code called into — same xoshiro256++ generator, so
/// any test that pins a sample under a given seed continues to
/// produce identical output.
pub(crate) fn seeded_rng(seed: u64) -> rand_xoshiro::Xoshiro256PlusPlus {
    // SeedableRng is re-exported by rand_xoshiro itself so this works
    // without a separate `rand` dependency in vectordata's
    // (non-dev) deps.
    use rand_xoshiro::rand_core::SeedableRng;
    rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed)
}

use std::path::PathBuf;

use clap::{Args, Subcommand};

pub use shared::SampleMode;
use shared::parse_sample_mode;

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
        #[arg(long, group = "input")]
        dataset: Option<String>,
        /// Any data source: local file path or dataset:profile:facet
        #[arg(long, group = "input")]
        source: Option<String>,
        /// Profile name (used with --dataset; overrides profile in dataset:profile)
        #[arg(long)]
        profile: Option<String>,
        /// Number of vectors to sample
        #[arg(long, default_value = "50000")]
        sample: usize,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Sampling mode [streaming, clumped, sparse]
        #[arg(long, default_value = "streaming", value_parser = parse_sample_mode)]
        sample_mode: SampleMode,
    },
    /// Interactive data exploration shell for vector files
    Shell {
        /// Dataset from catalog (e.g., img-search or img-search:default)
        #[arg(long, group = "input")]
        dataset: Option<String>,
        /// Any data source: local file path or dataset:profile:facet
        #[arg(long, group = "input")]
        source: Option<String>,
        /// Profile name (used with --dataset; overrides profile in dataset:profile)
        #[arg(long)]
        profile: Option<String>,

        /// Trailing args passed as command options
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Scrollable raw-values grid: ordinals × dimensions, with sig-digit
    /// control and 24-bit-color heatmap.
    Values {
        /// Dataset from catalog (e.g., img-search or img-search:default)
        #[arg(long, group = "input")]
        dataset: Option<String>,
        /// Any data source: local file path or dataset:profile:facet
        #[arg(long, group = "input")]
        source: Option<String>,
        /// Profile name (used with --dataset; overrides profile in dataset:profile)
        #[arg(long)]
        profile: Option<String>,
        /// First ordinal to display
        #[arg(long, default_value = "0")]
        start: u64,
        /// Initial significant-digit count (1–6)
        #[arg(long, default_value = "4")]
        digits: u8,
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
        ExploreCommand::Values { dataset, source, profile, start, digits } => {
            let from_picker = dataset.is_none() && source.is_none();
            let mut src = match resolve_input(dataset, source, profile) {
                Some(s) => s,
                None => match dataset_picker::run_picker() {
                    Some(s) => s,
                    None => { std::process::exit(0); }
                },
            };
            loop {
                match values_grid::run(&src, start as usize, digits) {
                    values_grid::Exit::Quit => break,
                    values_grid::Exit::Back if from_picker => {
                        match dataset_picker::run_picker() {
                            Some(s) => { src = s; }
                            None => break,
                        }
                    }
                    values_grid::Exit::Back => break,
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

// `run_pipeline_command` lived here as `#[allow(dead_code)]`
// scaffolding for invoking pipeline `CommandOp` instances from the
// explore TUI. It was the only consumer of `veks-pipeline`'s
// `StreamContext` / `ProgressLog` / `ResourceGovernor` / `ui::*`
// inside this module, and nothing ever called it. It was deleted
// during the migration into vectordata — explore is now leaf-crate
// code and has no business reaching back up into the pipeline
// command framework. Re-add a vectordata-local equivalent if a
// future need surfaces.
