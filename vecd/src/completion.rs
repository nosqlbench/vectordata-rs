// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dynamic shell-completion value providers for `vecd` admin commands.
//!
//! These read the local control-plane DB (read-only, best-effort) to suggest
//! live values — backend names, namespace paths, roles, principals — for option
//! values (`--ns`, `--to`, `--role`, `--backend-config`) and positionals
//! (`backends remove <name>`, `ns remove <path>`, …). The completion engine
//! calls each provider with the partial word plus the preceding command-line
//! words, so `--conf` / `--data-dir` on the line are honored. Any failure (no
//! config, locked/missing DB, unreadable row) yields no suggestions — never a
//! hang or an error on the user's prompt.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::db::Db;
use crate::{admin, config};
use veks_completion::{fn_provider, ValueProvider};

/// Find `--flag value` or `--flag=value` among the preceding words.
fn flag_value<'a>(ctx: &'a [&str], flag: &str) -> Option<&'a str> {
    if let Some(i) = ctx.iter().position(|&w| w == flag) {
        return ctx.get(i + 1).copied();
    }
    let pfx = format!("{flag}=");
    ctx.iter().find_map(|&w| w.strip_prefix(&pfx))
}

/// Open the control-plane DB read-only for completion, honoring `--conf`,
/// `--data-dir`, and the config-preference flags when present on the line.
/// Read-only so it never contends with a running server for a write lock.
fn open_db(ctx: &[&str]) -> Option<Db> {
    let conf = flag_value(ctx, "--conf").map(PathBuf::from);
    let prefer = if ctx.contains(&"--config-is-local") {
        Some(config::Prefer::Local)
    } else if ctx.contains(&"--config-is-home") {
        Some(config::Prefer::Home)
    } else {
        None
    };
    let resolved = config::resolve(conf.as_deref(), prefer).ok()?;
    if !resolved.exists {
        return None;
    }
    let cfg = config::Config::load(&resolved.dir).ok()?;
    let data_flag = flag_value(ctx, "--data-dir").map(PathBuf::from);
    let data_dir = crate::cli::resolve_data_dir(&data_flag, &cfg, &resolved.dir);
    Db::open_readonly(&config::db_path(&data_dir)).ok()
}

fn prefixed(items: impl IntoIterator<Item = String>, partial: &str) -> Vec<String> {
    items.into_iter().filter(|s| s.starts_with(partial)).collect()
}

/// Backend names (e.g. for `backends remove|set <name>`, `ns … --backend-config`).
pub fn backends(partial: &str, ctx: &[&str]) -> Vec<String> {
    let Some(db) = open_db(ctx) else { return Vec::new() };
    prefixed(admin::list_backends(&db).unwrap_or_default().into_iter().map(|(n, ..)| n), partial)
}

/// Namespace paths (e.g. for `--ns`, `ns set|remove <path>`). The root namespace
/// (empty path) is offered as `/`.
pub fn namespaces(partial: &str, ctx: &[&str]) -> Vec<String> {
    let Some(db) = open_db(ctx) else { return Vec::new() };
    let names = admin::list_namespaces(&db)
        .unwrap_or_default()
        .into_iter()
        .map(|(path, ..)| if path.is_empty() { "/".to_string() } else { path });
    prefixed(names, partial)
}

/// Role names (e.g. for `--role`).
pub fn roles(partial: &str, ctx: &[&str]) -> Vec<String> {
    let Some(db) = open_db(ctx) else { return Vec::new() };
    prefixed(admin::list_roles(&db).unwrap_or_default().into_iter().map(|(n, ..)| n), partial)
}

/// Principals — users plus the synthetic `PUBLIC` (e.g. for `bind --to`,
/// `--principal`, `users remove <name>`).
pub fn principals(partial: &str, ctx: &[&str]) -> Vec<String> {
    let Some(db) = open_db(ctx) else { return Vec::new() };
    let mut names: Vec<String> =
        admin::list_users(&db).unwrap_or_default().into_iter().map(|(n, ..)| n).collect();
    names.push("PUBLIC".to_string());
    prefixed(names, partial)
}

/// User names only (e.g. for `--user`, `users remove <name>`).
pub fn users(partial: &str, ctx: &[&str]) -> Vec<String> {
    let Some(db) = open_db(ctx) else { return Vec::new() };
    prefixed(admin::list_users(&db).unwrap_or_default().into_iter().map(|(n, ..)| n), partial)
}

/// Namespace/backend owners — users plus the system-role owners (`@level`).
pub fn owners(partial: &str, ctx: &[&str]) -> Vec<String> {
    let Some(db) = open_db(ctx) else { return Vec::new() };
    let mut names: Vec<String> =
        admin::list_users(&db).unwrap_or_default().into_iter().map(|(n, ..)| n).collect();
    names.extend(["@superuser", "@admin", "@operator", "@user"].iter().map(|s| s.to_string()));
    prefixed(names, partial)
}

/// A provider over a fixed set of valid values (enum-like flags).
fn static_set(values: &'static [&'static str]) -> ValueProvider {
    std::sync::Arc::new(move |partial: &str, _: &[&str]| {
        values.iter().filter(|v| v.starts_with(partial)).map(|v| v.to_string()).collect()
    })
}

/// Resolvers for option values, keyed by flag name — consistent across every
/// command that uses the flag.
pub fn option_resolvers() -> BTreeMap<String, ValueProvider> {
    BTreeMap::from([
        // Dynamic (read from the DB).
        ("--ns".to_string(), fn_provider(namespaces)),
        ("--to".to_string(), fn_provider(principals)),
        ("--principal".to_string(), fn_provider(principals)),
        ("--user".to_string(), fn_provider(users)),
        ("--owner".to_string(), fn_provider(owners)),
        ("--role".to_string(), fn_provider(roles)),
        ("--backend-config".to_string(), fn_provider(backends)),
        // Static (enum-like) value sets.
        ("--kind".to_string(), static_set(&["local", "s3", "mem"])),
        ("--level".to_string(), static_set(&["user", "operator", "admin", "superuser"])),
        ("--listable".to_string(), static_set(&["public", "known", "grantees"])),
        ("--shell".to_string(), static_set(&["bash", "zsh", "fish", "elvish", "powershell"])),
    ])
}

/// Resolvers for command positionals, keyed by full command path (e.g.
/// `"backends remove"`). Space-separated keys can't collide with the `--flag`
/// keys in [`option_resolvers`], so both share the engine's one resolver map.
pub fn positional_resolvers() -> BTreeMap<String, ValueProvider> {
    BTreeMap::from([
        ("backends remove".to_string(), fn_provider(backends)),
        ("backends set".to_string(), fn_provider(backends)),
        ("ns set".to_string(), fn_provider(namespaces)),
        ("ns remove".to_string(), fn_provider(namespaces)),
        ("users remove".to_string(), fn_provider(principals)),
        ("roles remove".to_string(), fn_provider(roles)),
    ])
}

/// The combined resolver map (option flags + command-path positionals) for the
/// vecd completion tree.
pub fn resolvers() -> BTreeMap<String, ValueProvider> {
    let mut m = option_resolvers();
    m.extend(positional_resolvers());
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use veks_completion::cli::build_completion_tree;
    use veks_completion::VeksCli;

    /// The completion tree must attach dynamic providers to exactly the right
    /// commands — catches a wrong command path or flag name in the resolver map
    /// (and confirms the derive emits the positionals).
    #[test]
    fn tree_wires_dynamic_values_to_the_right_commands() {
        let spec = Cli::veks_command_spec("vecd");
        let tree = build_completion_tree(&spec, &resolvers());

        // Option-value completion, per command.
        let has_flags = |group: &str, sub: Option<&str>, flags: &[&str]| {
            let g = tree.root.child(group).unwrap_or_else(|| panic!("{group} missing"));
            let node = match sub {
                Some(s) => g.child(s).unwrap_or_else(|| panic!("{group} {s} missing")),
                None => g,
            };
            for f in flags {
                assert!(
                    node.value_providers().contains_key(*f),
                    "{group} {} should complete {f}",
                    sub.unwrap_or("")
                );
            }
        };
        has_flags("bind", None, &["--ns", "--to", "--role"]);
        has_flags("ns", Some("add"), &["--owner", "--backend-config", "--listable"]);
        has_flags("backends", Some("add"), &["--kind"]);
        has_flags("users", Some("add"), &["--level"]);

        // First-positional completion on backends / ns / users / roles.
        let cases = [
            ("backends", "remove"),
            ("backends", "set"),
            ("ns", "set"),
            ("ns", "remove"),
            ("users", "remove"),
            ("roles", "remove"),
        ];
        for (group, sub) in cases {
            let g = tree.root.child(group).unwrap_or_else(|| panic!("{group} group missing"));
            let leaf = g.child(sub).unwrap_or_else(|| panic!("{group} {sub} missing"));
            assert!(
                leaf.positional_provider().is_some(),
                "{group} {sub} should complete its positional"
            );
        }
    }
}
