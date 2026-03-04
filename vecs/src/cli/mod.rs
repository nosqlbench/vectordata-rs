// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shell completion support.
//!
//! Completions are handled dynamically by `vecs` itself via
//! `clap_complete::CompleteEnv`. The `completions` subcommand outputs a
//! minimal, sourceable registration snippet that wires up the shell to
//! delegate completion requests back to `vecs`.
//!
//! Usage:
//! ```sh
//! source <(vecs completions --shell bash)
//! source <(vecs completions --shell zsh)
//! vecs completions --shell fish | source
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
/// completions. The actual completion logic runs inside `vecs` via
/// the `COMPLETE` env var handled by `CompleteEnv` in `main()`.
pub fn completions(args: CompletionsArgs) {
    match args.shell {
        Shell::Bash => print!(
            r#"source <(COMPLETE=bash vecs)
"#
        ),
        Shell::Zsh => print!(
            r#"source <(COMPLETE=zsh vecs)
"#
        ),
        Shell::Fish => print!(
            r#"COMPLETE=fish vecs | source
"#
        ),
        Shell::Elvish => print!(
            r#"eval (COMPLETE=elvish vecs | slurp)
"#
        ),
        Shell::PowerShell => print!(
            r#"(& {{ $env:COMPLETE="powershell"; vecs }}) | Invoke-Expression
"#
        ),
        _ => eprintln!("Unsupported shell: {:?}", args.shell),
    }
}
