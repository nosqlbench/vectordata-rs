// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vecd` binary entry point — dynamic completions + CLI dispatch over the
//! library core.

use clap::{CommandFactory, Parser};
use clap_complete::CompleteEnv;

use vecd::cli::{Cli, run};

fn main() {
    // Dynamic-completion entry: when invoked with `COMPLETE=<shell>`, emit
    // candidates and exit. `vecd completions` is a one-liner that
    // re-invokes the binary with that env var set, so completion logic
    // lives in the clap-derived metadata, not a frozen script.
    CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();
    std::process::exit(run(cli));
}
