// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dynamic shell completion engine for veks.
//!
//! Walks the clap `Command` tree at completion time so the completion
//! candidates are always in sync with the actual CLI definition. No
//! static command map — the tree is built from `build_augmented_cli()`
//! on every invocation.

use veks_completion::{CommandTree, Node, ValueProvider};

/// Build the completion tree by walking a clap `Command` recursively.
///
/// This is the single source of truth: every subcommand, option, and
/// alias defined via clap's derive macros or `build_augmented_cli()`
/// appears automatically — nothing to keep in sync by hand.
pub fn build_tree(cmd: &clap::Command) -> CommandTree {
    let app_name = cmd.get_name();
    // Each pipeline CommandOp declares per-option value-completion
    // sets via `value_completions()`, and discovery metadata
    // (category + level) via `category()` / `level()`. Collect both
    // into flat maps keyed by full pipeline command path (e.g.,
    // "verify engine-parity"), then thread through the tree walk so
    // leaf nodes get value providers, category, and level populated.
    let pipeline_vc = veks_pipeline::pipeline::cli::pipeline_value_completions();
    let pipeline_meta = veks_pipeline::pipeline::cli::pipeline_command_metadata();
    let root = walk_clap_command(cmd, &[], &pipeline_vc, &pipeline_meta);

    // Identify hidden commands
    let mut hidden: std::collections::HashSet<String> = std::collections::HashSet::new();
    for sub in cmd.get_subcommands() {
        if sub.is_hide_set() {
            hidden.insert(sub.get_name().to_string());
        }
    }

    let mut tree = CommandTree {
        app_name: app_name.to_string(),
        root,
        hidden,
        global_value_providers: std::collections::BTreeMap::new(),
        strict_metadata: false,
    };

    // Global value providers for options that appear across many commands
    use veks_completion::fn_provider;
    tree = tree
        .global_value_provider("--dataset", fn_provider(complete_dataset_names))
        .global_value_provider("--profile", fn_provider(complete_profile_names))
        .global_value_provider("--metric", fn_provider(complete_metrics))
        .global_value_provider("--at", fn_provider(complete_catalog_urls))
        .global_value_provider("--shell", fn_provider(complete_shells));

    tree
}

/// Recursively convert a clap `Command` into a completion `Node`.
///
/// `path_segments` accumulates the words leading up to this command
/// (excluding the program name and the leading `pipeline` group name)
/// so we can look up per-leaf value providers in `pipeline_vc`.
fn walk_clap_command(
    cmd: &clap::Command,
    path_segments: &[&str],
    pipeline_vc: &std::collections::BTreeMap<
        String,
        std::collections::HashMap<String, veks_pipeline::pipeline::command::ValueCompletions>,
    >,
    pipeline_meta: &std::collections::BTreeMap<
        String,
        veks_pipeline::pipeline::cli::CommandMetadata,
    >,
) -> Node {
    let subs: Vec<_> = cmd.get_subcommands().collect();

    if subs.is_empty() {
        // Leaf command — collect its options and identify boolean flags
        let mut options: Vec<String> = Vec::new();
        let mut flags: std::collections::HashSet<String> = std::collections::HashSet::new();
        for a in cmd.get_arguments() {
            if a.get_id() == "help" || a.get_id() == "version" { continue; }
            let name = if let Some(long) = a.get_long() {
                format!("--{}", long)
            } else if let Some(short) = a.get_short() {
                format!("-{}", short)
            } else {
                continue;
            };
            // Boolean flag: takes 0 args (num_args == 0..=0 or action is SetTrue/SetFalse/Count)
            let is_flag = a.get_action().takes_values() == false;
            if is_flag {
                flags.insert(name.clone());
            }
            options.push(name);
        }

        // Look up per-option value completions for this leaf. The
        // pipeline registry is keyed by the path joined with spaces
        // (e.g., "verify engine-parity"), matching the `command_path`
        // field on each `CommandOp`. When we're walking the
        // `pipeline` subtree, `path_segments` looks like
        // `["pipeline", "verify", "engine-parity"]`; strip the
        // leading "pipeline" to match the registry key. The same
        // applies when commands are also exposed as hidden top-level
        // shortcuts (e.g., `veks verify engine-parity` with no
        // `pipeline` prefix).
        let mut value_providers: std::collections::BTreeMap<String, ValueProvider> =
            std::collections::BTreeMap::new();
        let stripped: Vec<&str> = if path_segments.first() == Some(&"pipeline") {
            path_segments[1..].to_vec()
        } else {
            path_segments.to_vec()
        };
        let key = stripped.join(" ");
        if let Some(opt_map) = pipeline_vc.get(&key) {
            for (opt_name, vc) in opt_map {
                let vc = vc.clone();
                let provider: ValueProvider = std::sync::Arc::new(
                    move |partial: &str, _ctx: &[&str]| -> Vec<String> {
                        enum_value_completer(partial, &vc)
                    });
                value_providers.insert(format!("--{}", opt_name), provider);
            }
        }

        // Look up category/level metadata for this leaf in the same
        // way as value_completions: by full pipeline command path.
        let (category, level) = pipeline_meta.get(&key)
            .map(|m| (Some(m.category.tag().to_string()), Some(m.level.rank())))
            .unwrap_or((None, None));

        Node::Leaf {
            options,
            flags,
            value_providers,
            dynamic_options: None,
            category,
            level,
        }
    } else {
        // Group command — recurse into children
        let mut children = std::collections::BTreeMap::new();
        for sub in subs {
            let name = sub.get_name();
            let mut child_path: Vec<&str> = path_segments.to_vec();
            child_path.push(name);
            let child = walk_clap_command(sub, &child_path, pipeline_vc, pipeline_meta);
            children.insert(name.to_string(), child);
        }
        Node::Group { children, category: None, level: None }
    }
}

/// Plain-string version of the enum value completer (the cli.rs
/// version returns clap_complete `CompletionCandidate`s; this one
/// returns plain `String`s for the veks-completion engine).
fn enum_value_completer(
    partial: &str,
    vc: &veks_pipeline::pipeline::command::ValueCompletions,
) -> Vec<String> {
    if !vc.comma_separated {
        let mut out: Vec<String> = vc.values.iter()
            .filter(|v| partial.is_empty() || v.starts_with(partial))
            .map(|v| v.clone())
            .collect();
        out.sort();
        return out;
    }
    let (already, rest) = match partial.rfind(',') {
        Some(i) => (&partial[..=i], &partial[i + 1..]),
        None    => ("", partial),
    };
    let chosen: std::collections::HashSet<&str> = already
        .split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();
    let mut out: Vec<String> = vc.values.iter()
        .filter(|v| !chosen.contains(v.as_str()))
        .filter(|v| rest.is_empty() || v.starts_with(rest))
        .map(|v| format!("{}{}", already, v))
        .collect();
    out.sort();
    out
}

/// Extract the value of a previously typed `--option` from the completion args.
fn extract_option(option: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let words_start = args.iter().position(|a| a == "--").map(|i| i + 1).unwrap_or(1);
    let words = &args[words_start..];
    words.windows(2)
        .find(|w| w[0] == option)
        .map(|w| w[1].clone())
}

/// Build a catalog resolver that respects `--at` and `--catalog` from the
/// command line. Falls back to the default configured catalogs.
fn resolve_catalog() -> crate::catalog::resolver::Catalog {
    let at_values: Vec<String> = {
        let args: Vec<String> = std::env::args().collect();
        let words_start = args.iter().position(|a| a == "--").map(|i| i + 1).unwrap_or(1);
        let words = &args[words_start..];
        words.windows(2)
            .filter(|w| w[0] == "--at")
            .map(|w| crate::datasets::resolve_catalog_value(&w[1]))
            .collect()
    };

    if at_values.is_empty() {
        let sources = crate::catalog::sources::CatalogSources::new().configure_default();
        crate::catalog::resolver::Catalog::of(&sources)
    } else {
        let sources = crate::catalog::sources::CatalogSources::new()
            .add_catalogs(&at_values);
        crate::catalog::resolver::Catalog::of(&sources)
    }
}

/// Suggest dataset names from catalogs (respects `--at`).
fn complete_dataset_names(partial: &str, _context: &[&str]) -> Vec<String> {
    let catalog = resolve_catalog();
    let prefix = partial.to_lowercase();
    catalog.datasets().iter()
        .map(|e| e.name.clone())
        .filter(|n| prefix.is_empty() || n.to_lowercase().starts_with(&prefix))
        .collect()
}

/// Suggest profile names (respects `--at` and `--dataset`).
fn complete_profile_names(partial: &str, _context: &[&str]) -> Vec<String> {
    let catalog = resolve_catalog();
    let prefix = partial.to_lowercase();
    let dataset_name = extract_option("--dataset");

    let mut profiles = std::collections::BTreeSet::new();
    for entry in catalog.datasets() {
        if let Some(ref ds) = dataset_name {
            if !entry.name.eq_ignore_ascii_case(ds) { continue; }
        }
        for name in entry.profile_names() {
            if prefix.is_empty() || name.to_lowercase().starts_with(&prefix) {
                profiles.insert(name.to_string());
            }
        }
    }
    profiles.into_iter().collect()
}

/// Suggest distance metrics.
fn complete_metrics(partial: &str, _context: &[&str]) -> Vec<String> {
    let metrics = ["L2", "COSINE", "DOT_PRODUCT", "L1"];
    let prefix = partial.to_uppercase();
    metrics.iter()
        .filter(|m| prefix.is_empty() || m.starts_with(&prefix))
        .map(|m| m.to_string())
        .collect()
}

/// Suggest configured catalog URLs by index.
///
/// Each configured catalog can be referenced by its 1-based index or its
/// full URL. When multiple candidates match, the index-to-URL mapping is
/// printed to stderr (which goes to /dev/tty) so the user can see what
/// each number means. Only the numbers are returned as completable values.
///
/// Bash calls the completer twice per tab press (generate + display).
/// We use a lock file keyed on the partial string to avoid printing
/// descriptions twice.
fn complete_catalog_urls(partial: &str, _context: &[&str]) -> Vec<String> {
    let config_dir = crate::catalog::sources::expand_tilde(
        crate::catalog::sources::DEFAULT_CONFIG_DIR,
    );
    let entries = crate::catalog::sources::raw_catalog_entries(&config_dir);
    let mut results: Vec<(String, String)> = Vec::new();
    for (i, url) in entries.iter().enumerate() {
        let num = format!("{}", i + 1);
        if partial.is_empty() || num.starts_with(partial) || url.starts_with(partial) {
            results.push((num, url.clone()));
        }
    }
    // Only show descriptions when there are multiple candidates.
    // Use a temp file to avoid printing twice per tab press.
    if results.len() > 1 {
        let lock = format!("/tmp/.veks_comp_at_{}", partial);
        let lock_path = std::path::Path::new(&lock);
        let stale = lock_path.metadata()
            .and_then(|m| m.modified())
            .map(|t| t.elapsed().unwrap_or_default().as_secs() > 2)
            .unwrap_or(true);
        if stale {
            let _ = std::fs::write(lock_path, "");
            eprintln!();
            for (num, url) in &results {
                eprintln!("  {} = {}", num, url);
            }
        }
    }
    results.into_iter().map(|(num, _)| num).collect()
}

/// Suggest shell names.
fn complete_shells(partial: &str, _context: &[&str]) -> Vec<String> {
    let shells = ["bash", "zsh", "fish", "elvish", "powershell"];
    let prefix = partial.to_lowercase();
    shells.iter()
        .filter(|s| prefix.is_empty() || s.starts_with(&prefix))
        .map(|s| s.to_string())
        .collect()
}

/// Print the bash completion script for veks.
pub fn print_bash_script() {
    veks_completion::print_bash_script("veks");
}

/// Handle completion env vars. Returns true if handled (caller should exit).
///
/// The caller must pass the fully augmented `clap::Command` tree so that
/// completion candidates are derived dynamically — no static map needed.
pub fn handle_complete_env(cmd: &clap::Command) -> bool {
    let tree = build_tree(cmd);
    veks_completion::handle_complete_env("veks", &tree)
}
