// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

use slabtastic::cli;
use veks_completion::cli as vcli;
use veks_completion::VeksCli;

fn main() {
    let spec = cli::Cli::veks_command_spec("slab");

    // Dynamic shell completion (COMPLETE=bash/zsh/… or _SLAB_COMPLETE=…),
    // driven by the same spec that drives parsing.
    let resolvers: std::collections::BTreeMap<String, veks_completion::ValueProvider> =
        std::collections::BTreeMap::new();
    let tree = vcli::build_completion_tree(&spec, &resolvers);
    if veks_completion::handle_complete_env("slab", &tree) {
        return;
    }
    if veks_completion::handle_diagnostic_args("slab", &tree) {
        return;
    }
    veks_completion::hint_completions_unregistered("slab");

    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("slab {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.first().is_none() || args.iter().any(|a| a == "--help" || a == "-h") {
        // Render help for the deepest subcommand named on the line (group →
        // leaf), falling back to the top-level overview.
        print!("{}", vcli::render_help_for(&spec, &args));
        return;
    }

    let parsed = vcli::parse(&spec, &args).unwrap_or_else(|e| {
        eprintln!("slab: {e}");
        std::process::exit(2);
    });
    let cli = <cli::Cli as VeksCli>::veks_from_parsed(&parsed).unwrap_or_else(|e| {
        eprintln!("slab: {e}");
        std::process::exit(2);
    });
    if let Err(e) = cli::run(cli) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
