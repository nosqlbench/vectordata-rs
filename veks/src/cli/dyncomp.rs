// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Dynamic shell completion engine for veks.
//!
//! Walks the clap `Command` tree at completion time so the completion
//! candidates are always in sync with the actual CLI definition. No
//! static command map — the tree is built from `build_augmented_cli()`
//! on every invocation.

use veks_completion::{CommandTree, Node};

/// Build the completion tree by walking a clap `Command` recursively.
///
/// This is the single source of truth: every subcommand, option, and
/// alias defined via clap's derive macros or `build_augmented_cli()`
/// appears automatically — nothing to keep in sync by hand.
pub fn build_tree(cmd: &clap::Command) -> CommandTree {
    let app_name = cmd.get_name();
    let root = walk_clap_command(cmd);

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
    };

    // Global value providers for options that appear across many commands
    tree = tree
        .global_value_provider("--dataset", complete_dataset_names)
        .global_value_provider("--profile", complete_profile_names)
        .global_value_provider("--metric", complete_metrics)
        .global_value_provider("--at", complete_catalog_urls)
        .global_value_provider("--shell", complete_shells);

    tree
}

/// Recursively convert a clap `Command` into a completion `Node`.
fn walk_clap_command(cmd: &clap::Command) -> Node {
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
        Node::Leaf {
            options,
            flags,
            value_providers: std::collections::BTreeMap::new(),
        }
    } else {
        // Group command — recurse into children
        let mut children = std::collections::BTreeMap::new();
        for sub in subs {
            let name = sub.get_name().to_string();
            let child = walk_clap_command(sub);
            children.insert(name, child);
        }
        // If this group also has its own options (e.g., --help, --version
        // at root), they'll be picked up by the tree walk logic in
        // veks-completion when the node is a Group with options.
        Node::Group { children }
    }
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
fn complete_dataset_names(partial: &str) -> Vec<String> {
    let catalog = resolve_catalog();
    let prefix = partial.to_lowercase();
    catalog.datasets().iter()
        .map(|e| e.name.clone())
        .filter(|n| prefix.is_empty() || n.to_lowercase().starts_with(&prefix))
        .collect()
}

/// Suggest profile names (respects `--at` and `--dataset`).
fn complete_profile_names(partial: &str) -> Vec<String> {
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
fn complete_metrics(partial: &str) -> Vec<String> {
    let metrics = ["L2", "COSINE", "DOT_PRODUCT", "L1"];
    let prefix = partial.to_uppercase();
    metrics.iter()
        .filter(|m| prefix.is_empty() || m.starts_with(&prefix))
        .map(|m| m.to_string())
        .collect()
}

/// Suggest configured catalog URLs by number.
fn complete_catalog_urls(partial: &str) -> Vec<String> {
    let config_dir = crate::catalog::sources::expand_tilde(
        crate::catalog::sources::DEFAULT_CONFIG_DIR,
    );
    let entries = crate::catalog::sources::raw_catalog_entries(&config_dir);
    entries.iter()
        .enumerate()
        .filter(|(i, _)| {
            let num = format!("{}", i + 1);
            partial.is_empty() || num.starts_with(partial)
        })
        .map(|(i, _)| format!("{}", i + 1))
        .collect()
}

/// Suggest shell names.
fn complete_shells(partial: &str) -> Vec<String> {
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
