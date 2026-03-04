// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

use clap::{CommandFactory, Parser};
use clap_complete::CompleteEnv;
use slabtastic::cli;

fn main() {
    CompleteEnv::with_factory(cli::Cli::command).complete();
    let cli = cli::Cli::parse();
    if let Err(e) = cli::run(cli) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
