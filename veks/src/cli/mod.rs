// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shell completion support.
//!
//! Completions are handled dynamically by `veks` itself via
//! `clap_complete::CompleteEnv`. The `completions` subcommand outputs a
//! minimal, sourceable registration snippet that wires up the shell to
//! delegate completion requests back to `veks`.
//!
//! Usage:
//! ```sh
//! source <(veks completions --shell bash)
//! source <(veks completions --shell zsh)
//! veks completions --shell fish | source
//! ```

use clap::Args;
use clap_complete::Shell;

/// Arguments for the completions subcommand
#[derive(Args)]
pub struct CompletionsArgs {
    /// Shell to generate completions for (bash, zsh, fish, elvish, powershell)
    #[arg(long, value_enum)]
    pub shell: Shell,
}

/// Print a minimal, sourceable shell snippet that registers dynamic
/// completions. The actual completion logic runs inside `veks` via
/// the `COMPLETE` env var handled by `CompleteEnv` in `main()`.
pub fn completions(args: CompletionsArgs) {
    match args.shell {
        Shell::Bash => print!(
            r#"source <(COMPLETE=bash veks)
"#
        ),
        Shell::Zsh => print!(
            r#"source <(COMPLETE=zsh veks)
"#
        ),
        Shell::Fish => print!(
            r#"COMPLETE=fish veks | source
"#
        ),
        Shell::Elvish => print!(
            r#"eval (COMPLETE=elvish veks | slurp)
"#
        ),
        Shell::PowerShell => print!(
            r#"(& {{ $env:COMPLETE="powershell"; veks }}) | Invoke-Expression
"#
        ),
        _ => eprintln!("Unsupported shell: {:?}", args.shell),
    }
}
