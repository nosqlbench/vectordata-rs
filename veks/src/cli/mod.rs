// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI argument parsing and shell completion support.
//!
//! CLI argument parsing and shell completion support.
//!
//! Dynamic completions are handled by the `dyncomp` module. The
//! `completions` subcommand outputs a sourceable shell snippet.
//!
//! Usage:
//! ```sh
//! source <(veks completions --shell bash)
//! source <(veks completions --shell zsh)
//! veks completions --shell fish | source
//! ```

pub mod dyncomp;

use clap::Args;
/// Supported shells for completion script generation.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    Elvish,
    #[value(name = "powershell")]
    PowerShell,
}

/// Generate shell completions for veks.
///
/// Auto-detects your shell when --shell is omitted:
///
///   eval "$(veks completions)"
///
/// Or specify the shell explicitly:
///
///   eval "$(veks completions --shell bash)"
///   eval "$(veks completions --shell zsh)"
///   veks completions --shell fish | source
#[derive(Args)]
#[command(after_long_help = "\
EXAMPLES:
  # Auto-detect shell and activate completions:
  eval \"$(veks completions)\"

  # Add to ~/.bashrc for persistent completions:
  echo 'eval \"$(veks completions)\"' >> ~/.bashrc

  # Or specify the shell explicitly:
  eval \"$(veks completions --shell bash)\"
  eval \"$(veks completions --shell zsh)\"
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
    match args.shell {
        Some(shell) => {
            // Explicit --shell: emit the raw script directly
            emit_completions(shell);
        }
        None => {
            // Auto-detect: emit an indirect one-liner that's safe for
            // both eval "$(veks completions)" and eval `veks completions`.
            // The indirection avoids backtick-substitution mangling the
            // raw script (which contains backslashes and special chars).
            let veks_path = std::env::current_exe()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "veks".to_string());
            match detect_shell() {
                Some(Shell::Bash) => {
                    println!("# veks tab-completion for bash");
                    println!("# To activate:  eval \"$(veks completions)\"");
                    println!("# To persist:   echo 'eval \"$(veks completions)\"' >> ~/.bashrc");
                    println!("source <(\"{veks_path}\" completions --shell bash)");
                }
                Some(Shell::Zsh) => {
                    println!("# veks tab-completion for zsh");
                    println!("# To activate:  eval \"$(veks completions)\"");
                    println!("# To persist:   echo 'eval \"$(veks completions)\"' >> ~/.zshrc");
                    println!("source <(\"{veks_path}\" completions --shell zsh)");
                }
                Some(Shell::Fish) => {
                    println!("# veks tab-completion for fish");
                    println!("# To activate:  eval (veks completions)");
                    println!("# To persist:   add to ~/.config/fish/config.fish");
                    println!("\"{veks_path}\" completions --shell fish | source");
                }
                Some(shell) => {
                    emit_completions(shell);
                }
                None => {
                    println!("# veks: could not detect your shell.");
                    println!("# Use: eval \"$(veks completions --shell bash)\"");
                }
            }
        }
    }
}

fn emit_completions(shell: Shell) {
    match shell {
        Shell::Bash => dyncomp::print_bash_script(),
        Shell::Zsh => {
            eprintln!("# zsh completions not yet implemented in dyncomp; use bash compatibility mode");
            dyncomp::print_bash_script(); // bash-compatible fallback
        }
        Shell::Fish | Shell::Elvish | Shell::PowerShell => {
            eprintln!("# {} completions not yet implemented", match shell {
                Shell::Fish => "fish", Shell::Elvish => "elvish",
                Shell::PowerShell => "powershell", _ => "unknown",
            });
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
