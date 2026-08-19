// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI shell for the `vecd` binary — dynamic completions + CLI dispatch
//! over the library core. Lives in the library so wrapper binaries (the
//! workspace-root `vectordata` multiplexer) can embed the identical
//! personality; `vecd/src/main.rs` is a thin shim over [`bin_main`].

use veks_completion::VeksCli;
use veks_completion::cli as vcli;

use crate::cli::{Cli, run};

/// This crate's plain version, for umbrella-binary version listings.
pub fn version_line() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Binary entry point over explicit args (everything after the program
/// name).
pub fn bin_main(args: Vec<String>) {
    let spec = Cli::veks_command_spec("vecd");

    // Dynamic-completion entry: when invoked with `COMPLETE=<shell>` (or
    // `_VECD_COMPLETE=…`), emit candidates and exit. `vecd completions` is a
    // one-liner that re-invokes the binary with that env var set, so completion
    // logic lives in the spec, not a frozen script.
    // Dynamic value completion for admin commands: backend names, namespace
    // paths, roles, principals — read live from the control-plane DB.
    let resolvers = crate::completion::resolvers();
    let tree = vcli::build_completion_tree(&spec, &resolvers);
    if veks_completion::handle_complete_env("vecd", &tree) {
        return;
    }
    if veks_completion::handle_diagnostic_args("vecd", &tree) {
        return;
    }
    // No completion-enablement nudge for vecd: it's mostly run as a daemon /
    // admin tool in tight loops, where a per-invocation stderr note is just
    // noise. (`vectordata`/`veks` — interactive clients — still hint.)

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("vecd {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        // Render help for the deepest subcommand named on the line (group →
        // leaf), falling back to the top-level overview.
        print!("{}", vcli::render_help_for(&spec, &args));
        return;
    }

    let parsed = vcli::parse(&spec, &args).unwrap_or_else(|e| {
        eprintln!("vecd: {e}");
        std::process::exit(2);
    });
    let cli = <Cli as VeksCli>::veks_from_parsed(&parsed).unwrap_or_else(|e| {
        eprintln!("vecd: {e}");
        std::process::exit(2);
    });
    std::process::exit(run(cli));
}
