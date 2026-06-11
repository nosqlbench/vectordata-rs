// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dynamic shell completion engine for veks.
//!
//! Walks the clap `Command` tree at completion time so the completion
//! candidates are always in sync with the actual CLI definition. No
//! static command map — the tree is built from `build_augmented_cli()`
//! on every invocation.

use veks_completion::{CommandOption, OptionDef, OptionRegistry, ParseMismatch, ValueProvider};

/// One shared option: a DRY parse definition plus the value resolver to use
/// wherever it's attached. (The trait allows the resolver to vary per command;
/// these cross-cutting options use one resolver each.)
struct SharedOpt {
    def: OptionDef,
    resolver: ValueProvider,
}

impl CommandOption for SharedOpt {
    fn definition(&self) -> OptionDef {
        self.def.clone()
    }
    fn value_resolver(&self) -> Option<ValueProvider> {
        Some(self.resolver.clone())
    }
}

/// The options that appear across many commands, declared **once** here. This
/// is the only place their value resolvers live; `build_tree` registers them
/// (enforcing one consistent definition per name) and attaches each to the
/// commands that actually declare the flag.
fn shared_options() -> Vec<SharedOpt> {
    use veks_completion::fn_provider;
    let mk = |def: OptionDef, r: ValueProvider| SharedOpt { def, resolver: r };
    vec![
        mk(OptionDef::value("--dataset").value_name("DATASET"), fn_provider(complete_dataset_names)),
        mk(OptionDef::value("--profile").value_name("PROFILE"), fn_provider(complete_profile_names)),
        mk(OptionDef::value("--metric").value_name("METRIC"), fn_provider(complete_metrics)),
        mk(OptionDef::value("--at").value_name("URL").multiple(true), fn_provider(complete_catalog_urls)),
        mk(OptionDef::value("--shell").value_name("SHELL"), fn_provider(complete_shells)),
    ]
}

/// The shared per-flag value resolvers (`--at`/`--dataset`/`--profile`/
/// `--metric`/`--shell`), keyed by canonical long token, for
/// [`veks_completion::cli::build_completion_tree`]. These are the dynamic
/// completers the derive structs don't carry on their own `OptionSpec`s.
pub fn shared_resolvers() -> std::collections::BTreeMap<String, ValueProvider> {
    let mut map = std::collections::BTreeMap::new();
    for opt in &shared_options() {
        if let Some(r) = opt.value_resolver() {
            map.insert(opt.definition().name, r);
        }
    }
    map
}

/// Collect `(command_path, OptionDef)` for every option in a [`CommandSpec`]
/// tree — the spec-native counterpart of [`observed_options`].
fn observed_options_spec(
    spec: &veks_completion::CommandSpec,
    path: &[&str],
    out: &mut Vec<(String, OptionDef)>,
) {
    let here = path.join(" ");
    for o in &spec.options {
        out.push((here.clone(), o.def.clone()));
    }
    for sub in &spec.subcommands {
        let mut child_path: Vec<&str> = path.to_vec();
        child_path.push(&sub.name);
        observed_options_spec(sub, &child_path, out);
    }
}

/// Parse-consistency audit against a [`CommandSpec`] (the clap-free path).
/// Same invariant as [`audit_parse_consistency`], walking the spec instead of a
/// `clap::Command`.
pub fn audit_parse_consistency_spec(spec: &veks_completion::CommandSpec) -> Vec<ParseMismatch> {
    let mut registry = OptionRegistry::new();
    for opt in &shared_options() {
        let _ = registry.define(&opt.definition());
    }
    let mut observed = Vec::new();
    observed_options_spec(spec, &[], &mut observed);
    registry.audit(&observed)
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
        if let Some(ref ds) = dataset_name
            && !entry.name.eq_ignore_ascii_case(ds) { continue; }
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
