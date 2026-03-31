// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Builds the veks command tree for dynamic completion using `veks-completion`.

use veks_completion::{CommandTree, Node};

/// Build the full veks command tree for shell completion.
pub fn build_tree() -> CommandTree {
    let registry = crate::pipeline::registry::CommandRegistry::with_builtins();

    // Build pipeline groups from the command registry
    let mut pipeline_groups: std::collections::BTreeMap<String, Vec<(String, Vec<String>)>> =
        std::collections::BTreeMap::new();
    for path in registry.command_paths() {
        let mut parts = path.splitn(2, ' ');
        let group = parts.next().unwrap_or("").to_string();
        let command = parts.next().unwrap_or("").to_string();
        let options: Vec<String> = if let Some(factory) = registry.get(&path) {
            let cmd = factory();
            cmd.describe_options().iter()
                .map(|o| format!("--{}", o.name))
                .collect()
        } else {
            Vec::new()
        };
        pipeline_groups.entry(group).or_default().push((command, options));
    }

    // Build pipeline node
    let mut pipeline_node = Node::empty_group();
    for (group_name, commands) in &pipeline_groups {
        let mut group_node = Node::empty_group();
        for (cmd_name, opts) in commands {
            if cmd_name.is_empty() {
                // Direct command (no subcommand within group)
                group_node = Node::leaf(&opts.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                break;
            }
            let opt_refs: Vec<&str> = opts.iter().map(|s| s.as_str()).collect();
            group_node = group_node.with_child(cmd_name, Node::leaf(&opt_refs));
        }
        pipeline_node = pipeline_node.with_child(group_name, group_node);
    }

    // Build the tree
    let mut tree = CommandTree::new("veks")
        .command("pipeline", pipeline_node.clone())
        .command("datasets", Node::group(vec![
            ("cache-status", Node::leaf(&["--dataset", "--all", "--verbose", "--tree", "--configdir", "--catalog", "--at"])),
            ("config", Node::empty_group()),
            ("curlify", Node::leaf(&["--dataset", "--profile", "--configdir", "--catalog", "--at"])),
            ("list", Node::leaf(&["--configdir", "--catalog", "--at", "--output-format", "--verbose", "--group-by",
                "--matching-profile", "--select", "--matching-name", "--with-facet", "--with-metric",
                "--matching-desc", "--with-min-size", "--with-max-size", "--with-size",
                "--with-min-dim", "--with-max-dim", "--with-dim", "--with-vtype",
                "--with-data-min", "--with-data-max", "--cached"])),
            ("prebuffer", Node::leaf(&["--dataset", "--profile", "--configdir", "--catalog", "--at", "--cache-dir"])),
            ("probe", Node::leaf(&["--at", "--dataset", "--profile"])),
        ]))
        .command("prepare", Node::group(vec![
            ("cache-compress", Node::leaf(&[])),
            ("cache-uncompress", Node::leaf(&[])),
            ("catalog", Node::group(vec![
                ("generate", Node::leaf(&["--for-publish-url", "--update", "--no-update"])),
            ])),
            ("check", Node::leaf(&["--check-all", "--check-pipelines", "--check-publish",
                "--check-merkle", "--check-integrity", "--check-catalogs",
                "--check-extraneous", "--clean", "--clean-files", "--json", "--quiet",
                "--update-pipeline"])),
            ("import", Node::leaf(&["--source", "--metric", "--normalize", "--self-search"])),
            ("run", Node::leaf(&["--dry-run", "--clean", "--threads", "--resources", "--profile"])),
            ("stratify", Node::leaf(&[])),
            ("wizard", Node::leaf(&[])),
        ]))
        .command("interact", Node::group(vec![
            ("explore", Node::leaf(&["--dataset", "--source", "--profile", "--sample", "--seed", "--sample-mode"])),
            ("shell", Node::leaf(&["--dataset", "--source", "--profile"])),
        ]))
        .command("help", {
            // help accepts pipeline group names then command names
            let mut help_node = Node::empty_group();
            for (group_name, commands) in &pipeline_groups {
                let mut group_cmds = Node::empty_group();
                for (cmd_name, _) in commands {
                    if !cmd_name.is_empty() {
                        group_cmds = group_cmds.with_child(cmd_name, Node::leaf(&[]));
                    }
                }
                help_node = help_node.with_child(group_name, group_cmds);
            }
            // Also add static commands as help targets
            for name in &["run", "check", "publish", "explore", "datasets", "prepare"] {
                help_node = help_node.with_child(name, Node::leaf(&[]));
            }
            help_node = help_node.with_child("--list", Node::leaf(&[]));
            help_node
        })
        // Hidden shortcuts for commonly used leaf commands
        .hidden_command("run", Node::leaf(&["--dry-run", "--clean", "--threads", "--resources", "--profile"]))
        .hidden_command("check", Node::leaf(&["--check-all", "--check-pipelines", "--check-publish",
            "--check-merkle", "--check-integrity", "--check-catalogs",
            "--check-extraneous", "--clean", "--clean-files", "--json", "--quiet",
            "--update-pipeline"]))
        .hidden_command("publish", Node::leaf(&["--dry-run", "--delete", "--size-only", "--yes",
            "--profile", "--endpoint-url"]))
        .hidden_command("explore", Node::leaf(&["--dataset", "--source", "--profile", "--sample", "--seed"]))
        .hidden_command("completions", Node::leaf(&["--shell"]));

    // Also add --help and --version as root-level options
    tree = tree.command("--help", Node::leaf(&[]))
        .command("--version", Node::leaf(&[]));

    // Add pipeline groups as hidden top-level shortcuts.
    // Hidden commands don't appear in the initial `veks <TAB>` listing
    // but DO complete when the user starts typing (e.g., `veks comp<TAB>`).
    for (group_name, commands) in &pipeline_groups {
        let mut group_node = Node::empty_group();
        for (cmd_name, opts) in commands {
            if cmd_name.is_empty() { continue; }
            let opt_refs: Vec<&str> = opts.iter().map(|s| s.as_str()).collect();
            group_node = group_node.with_child(cmd_name, Node::leaf(&opt_refs));
        }
        if tree.root.child(group_name).is_none() {
            tree = tree.hidden_command(group_name, group_node);
        }
    }

    // Global value providers for options that appear across many commands
    tree = tree
        .global_value_provider("--dataset", complete_dataset_names)
        .global_value_provider("--profile", complete_profile_names)
        .global_value_provider("--metric", complete_metrics)
        .global_value_provider("--at", complete_catalog_urls)
        .global_value_provider("--shell", complete_shells);

    tree
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
pub fn handle_complete_env() -> bool {
    let tree = build_tree();
    veks_completion::handle_complete_env("veks", &tree)
}
