// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! CLI argument parsing and shell completion support.
//!
//! Built on [`clap`] and [`clap_complete`], this module provides dynamic
//! shell completions for bash, zsh, fish, elvish, and PowerShell.
//! Completions are handled by `veks` itself via `clap_complete::CompleteEnv`.
//! The `completions` subcommand outputs a minimal, sourceable registration
//! snippet that wires up the shell to delegate completion requests back
//! to `veks`.
//!
//! Usage:
//! ```sh
//! source <(veks completions --shell bash)
//! source <(veks completions --shell zsh)
//! veks completions --shell fish | source
//! ```

use clap::Args;
use clap_complete::Shell;

/// Generate shell completions for veks.
///
/// Source the output in your shell profile for tab completion:
///
///   bash:  source <(veks completions --shell bash)
///   zsh:   source <(veks completions --shell zsh)
///   fish:  veks completions --shell fish | source
#[derive(Args)]
#[command(after_long_help = "\
EXAMPLES:
  # Add to ~/.bashrc for persistent completions:
  echo 'source <(veks completions --shell bash)' >> ~/.bashrc

  # Or source directly in the current session:
  source <(veks completions --shell bash)

  # For zsh (add to ~/.zshrc):
  source <(veks completions --shell zsh)

  # For fish (add to config.fish):
  veks completions --shell fish | source")]
pub struct CompletionsArgs {
    /// Shell to generate completions for (bash, zsh, fish, elvish, powershell)
    #[arg(long, value_enum)]
    pub shell: Option<Shell>,
}

/// Print a minimal, sourceable shell snippet that registers dynamic
/// completions. The actual completion logic runs inside `veks` via
/// the `COMPLETE` env var handled by `CompleteEnv` in `main()`.
///
/// For bash, we emit a custom completion function that reconstructs
/// words from `COMP_LINE` instead of relying on `COMP_WORDS`, which
/// splits on `COMP_WORDBREAKS` characters (including `:`). This ensures
/// that values like `mem:25%` are treated as a single token.
pub fn completions(args: CompletionsArgs) {
    let shell = match args.shell {
        Some(s) => s,
        None => {
            // Auto-detect shell from environment
            match detect_shell() {
                Some(s) => s,
                None => {
                    // Emit benign output that won't break eval but tells the user
                    println!("# veks: could not detect your shell.");
                    println!("# Use one of:");
                    println!("#   eval \"$(veks completions --shell bash)\"");
                    println!("#   eval \"$(veks completions --shell zsh)\"");
                    println!("#   veks completions --shell fish | source");
                    return;
                }
            }
        }
    };
    match shell {
        Shell::Bash => print_bash_completions(),
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
        _ => {
            println!("# veks: unsupported shell variant");
        }
    }
}

/// Detect the current shell from environment.
///
/// Checks `SHELL` env var first (standard on Unix), then falls back to
/// inspecting the parent process name on Linux via `/proc`.
fn detect_shell() -> Option<Shell> {
    // Try SHELL env var (e.g., /bin/bash, /usr/bin/zsh)
    if let Ok(shell_path) = std::env::var("SHELL") {
        let name = std::path::Path::new(&shell_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if let Some(s) = shell_name_to_enum(name) {
            return Some(s);
        }
    }

    // Fallback: check parent process on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(comm) = std::fs::read_to_string(format!("/proc/{}/comm", std::os::unix::process::parent_id())) {
            let name = comm.trim();
            if let Some(s) = shell_name_to_enum(name) {
                return Some(s);
            }
        }
    }

    None
}

fn shell_name_to_enum(name: &str) -> Option<Shell> {
    match name {
        "bash" => Some(Shell::Bash),
        "zsh" => Some(Shell::Zsh),
        "fish" => Some(Shell::Fish),
        "elvish" => Some(Shell::Elvish),
        "pwsh" | "powershell" => Some(Shell::PowerShell),
        _ => None,
    }
}

/// Emit a custom bash completion function that handles colon-containing
/// values correctly.
///
/// The standard clap_complete bash script uses `COMP_WORDS` which is
/// subject to `COMP_WORDBREAKS` splitting (`:` is in the default set).
/// This means `--resources mem:25%` gets split into `["--resources",
/// "mem", ":", "25%"]`, and clap sees `mem` as the entire value.
///
/// Our custom function reconstructs words by splitting `COMP_LINE` on
/// whitespace only, bypassing `COMP_WORDBREAKS` entirely.
fn print_bash_completions() {
    // Find the veks binary path at registration time
    let completer = std::env::args_os()
        .next()
        .and_then(|p| {
            let path = std::path::PathBuf::from(p);
            if path.components().count() > 1 {
                std::env::current_dir().ok().map(|cwd| cwd.join(path))
            } else {
                Some(path)
            }
        })
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "veks".to_string());

    print!(
        r#"_clap_complete_veks() {{
    # Remove : from COMP_WORDBREAKS so bash's COMPREPLY processing
    # treats values like "mem:25%" as single words, matching our
    # whitespace-only word splitting below.
    COMP_WORDBREAKS="${{COMP_WORDBREAKS//:}}"
    # Reconstruct words from COMP_LINE, splitting on whitespace only.
    # This avoids COMP_WORDBREAKS splitting on ':' which breaks
    # values like "mem:25%" into separate tokens.
    local line="${{COMP_LINE:0:$COMP_POINT}}"
    local -a words=()
    local word=""
    local in_quote=""
    local i=0
    while [ $i -lt ${{#line}} ]; do
        local ch="${{line:$i:1}}"
        if [ -n "$in_quote" ]; then
            if [ "$ch" = "$in_quote" ]; then
                in_quote=""
            else
                word+="$ch"
            fi
        elif [ "$ch" = "'" ] || [ "$ch" = '"' ]; then
            in_quote="$ch"
        elif [ "$ch" = " " ] || [ "$ch" = $'\t' ]; then
            if [ -n "$word" ]; then
                words+=("$word")
                word=""
            fi
        else
            word+="$ch"
        fi
        i=$((i + 1))
    done
    # The last word is the one being completed (may be empty).
    words+=("$word")
    local _CLAP_COMPLETE_INDEX=$((${{#words[@]}} - 1))

    if compopt +o nospace 2> /dev/null; then
        local _CLAP_COMPLETE_SPACE=false
    else
        local _CLAP_COMPLETE_SPACE=true
    fi
    # Use fish format to get value\tdescription pairs.
    local raw
    raw=$( \
        _CLAP_IFS=$'\013' \
        _CLAP_COMPLETE_INDEX="$_CLAP_COMPLETE_INDEX" \
        _CLAP_COMPLETE_COMP_TYPE="${{COMP_TYPE}}" \
        _CLAP_COMPLETE_SPACE="$_CLAP_COMPLETE_SPACE" \
        COMPLETE="fish" \
        "{completer}" -- "${{words[@]}}" \
        2>/dev/null )
    if [[ $? != 0 ]]; then
        unset COMPREPLY
        return
    fi
    COMPREPLY=()
    local _has_nospace=false
    while IFS=$'\t' read -r _val _desc; do
        [[ -z "$_val" ]] && continue
        COMPREPLY+=("$_val")
        if [[ "$_val" =~ [=/:]$ ]]; then
            _has_nospace=true
        fi
    done <<< "$raw"
    if [[ $_CLAP_COMPLETE_SPACE == false ]] && [[ "$_has_nospace" == true ]]; then
        compopt -o nospace
    fi
}}
if [[ "${{BASH_VERSINFO[0]}}" -eq 4 && "${{BASH_VERSINFO[1]}}" -ge 4 || "${{BASH_VERSINFO[0]}}" -gt 4 ]]; then
    complete -o nospace -o bashdefault -o nosort -F _clap_complete_veks veks
else
    complete -o nospace -o bashdefault -F _clap_complete_veks veks
fi
"#,
        completer = completer,
    );
}
