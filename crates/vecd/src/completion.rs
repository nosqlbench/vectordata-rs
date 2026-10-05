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

/// The bare positional words already typed after a command path,
/// skipping flags and the value consumed by `--from` (the only
/// value-taking flag on `config set`). Used to tell the key slot
/// from the value slot for `config set`/`config get`.
fn config_positionals<'a>(ctx: &[&'a str]) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ctx.len() {
        let w = ctx[i];
        if w == "--from" {
            i += 2; // skip the flag and its value
            continue;
        }
        if w.starts_with('-') {
            i += 1; // `--force`, `--from=…`, etc.
            continue;
        }
        out.push(w);
        i += 1;
    }
    out
}

/// Positional completion for `config set <key> [value]` and
/// `config get <key>`: the recognized config keys
/// ([`config::KNOWN_KEYS`]) at the key slot, and — for `config set` —
/// the closed value set of a key that has one (`lock_config`: on/off)
/// at the value slot. No DB access; purely the static schema.
pub fn config_keys(partial: &str, ctx: &[&str]) -> Vec<String> {
    let pos = config_positionals(ctx);
    match pos.first() {
        // Key slot: offer every known config parameter.
        None => prefixed(config::KNOWN_KEYS.iter().map(|k| k.to_string()), partial),
        // Value slot for the one key with a closed value set.
        Some(&"lock_config") if pos.len() == 1 => {
            prefixed(["on", "off"].iter().map(|s| s.to_string()), partial)
        }
        // Any other key's value is free-form (addresses, paths, sizes).
        _ => Vec::new(),
    }
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
        ("store backends remove".to_string(), fn_provider(backends)),
        ("store backends set".to_string(), fn_provider(backends)),
        ("store ns set".to_string(), fn_provider(namespaces)),
        ("store ns remove".to_string(), fn_provider(namespaces)),
        ("access users remove".to_string(), fn_provider(principals)),
        ("access roles remove".to_string(), fn_provider(roles)),
        ("config set".to_string(), fn_provider(config_keys)),
        ("config get".to_string(), fn_provider(config_keys)),
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

        // Walk a full command path (e.g. ["store", "ns", "add"]) to its node.
        let node_at = |path: &[&str]| {
            let mut node = &tree.root;
            for seg in path {
                node = node
                    .child(seg)
                    .unwrap_or_else(|| panic!("path {} missing at {seg}", path.join(" ")));
            }
            node
        };

        // Option-value completion, at the new grouped paths.
        let has_flags = |path: &[&str], flags: &[&str]| {
            let node = node_at(path);
            for f in flags {
                assert!(
                    node.value_providers().contains_key(*f),
                    "{} should complete {f}",
                    path.join(" ")
                );
            }
        };
        has_flags(&["access", "bind"], &["--ns", "--to", "--role"]);
        has_flags(&["store", "ns", "add"], &["--owner", "--backend-config", "--listable"]);
        has_flags(&["store", "backends", "add"], &["--kind"]);
        has_flags(&["access", "users", "add"], &["--level"]);

        // First-positional completion at the grouped leaf paths.
        let positional_paths: &[&[&str]] = &[
            &["store", "backends", "remove"],
            &["store", "backends", "set"],
            &["store", "ns", "set"],
            &["store", "ns", "remove"],
            &["access", "users", "remove"],
            &["access", "roles", "remove"],
            &["config", "set"],
            &["config", "get"],
        ];
        for path in positional_paths {
            assert!(
                node_at(path).positional_provider().is_some(),
                "{} should complete its positional",
                path.join(" ")
            );
        }
    }

    /// `config set`/`config get` complete the key slot with the known
    /// config parameters; the value slot completes a closed set only
    /// for `lock_config`.
    #[test]
    fn config_keys_complete_param_then_value() {
        // Key slot (no positional typed yet): every known key, prefix-filtered.
        let all = config_keys("", &[]);
        assert_eq!(all.len(), config::KNOWN_KEYS.len());
        assert!(all.contains(&"bind".to_string()));
        assert!(all.contains(&"lock_config".to_string()));
        assert_eq!(config_keys("rate", &[]),
            config::KNOWN_KEYS.iter().filter(|k| k.starts_with("rate"))
                .map(|k| k.to_string()).collect::<Vec<_>>());

        // Value slot: lock_config offers on/off; a free-form key offers nothing.
        assert_eq!(config_keys("", &["lock_config"]), vec!["on".to_string(), "off".to_string()]);
        assert!(config_keys("", &["bind"]).is_empty());

        // `--from`'s value isn't mistaken for the key positional.
        assert_eq!(config_keys("", &["--from", "cfg.json"]).len(), config::KNOWN_KEYS.len());
    }
}
