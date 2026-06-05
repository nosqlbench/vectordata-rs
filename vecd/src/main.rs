// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vecd` binary entry point — dynamic completions + CLI dispatch over the
//! library core.

use veks_completion::cli as vcli;
use veks_completion::VeksCli;

use vecd::cli::{run, Cli};

fn main() {
    let spec = Cli::veks_command_spec("vecd");

    // Dynamic-completion entry: when invoked with `COMPLETE=<shell>` (or
    // `_VECD_COMPLETE=…`), emit candidates and exit. `vecd completions` is a
    // one-liner that re-invokes the binary with that env var set, so completion
    // logic lives in the spec, not a frozen script.
    let resolvers: std::collections::BTreeMap<String, veks_completion::ValueProvider> =
        std::collections::BTreeMap::new();
    let tree = vcli::build_completion_tree(&spec, &resolvers);
    if veks_completion::handle_complete_env("vecd", &tree) {
        return;
    }

    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("vecd {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.first().is_none() || args.iter().any(|a| a == "--help" || a == "-h") {
        // Render the matching subcommand's help when the first word names one;
        // otherwise the top-level overview.
        if let Some(sub) = args.first().and_then(|f| spec.subcommands.iter().find(|c| &c.name == f)) {
            print!("{}", vcli::render_help(sub));
        } else {
            print!("{}", vcli::render_help(&spec));
        }
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
