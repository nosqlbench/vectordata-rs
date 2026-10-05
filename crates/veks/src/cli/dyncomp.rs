// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dynamic shell completion engine for veks.
//!
//! Walks the clap `Command` tree at completion time so the completion
//! candidates are always in sync with the actual CLI definition. No
//! static command map — the tree is built from `build_augmented_cli()`
//! on every invocation.
//!
//! The dataset-domain completers (`--dataset`, `--profile`, `--at`,
//! and the datasets positionals) are NOT defined here — they live in
//! [`vectordata::datasets::dyncomp`] alongside the commands they
//! complete, and both binaries register the same resolvers. This
//! module only adds the veks-specific ones (`--metric`, `--shell`).

use vectordata::datasets::dyncomp::{
    complete_catalog_urls, complete_dataset_names, complete_profile_names, datasets_resolvers,
};
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
        mk(OptionDef::value("--dataset").short('d').value_name("DATASET"), fn_provider(complete_dataset_names)),
        mk(OptionDef::value("--profile").value_name("PROFILE"), fn_provider(complete_profile_names)),
        mk(OptionDef::value("--metric").value_name("METRIC"), fn_provider(complete_metrics)),
        mk(OptionDef::value("--at").value_name("URL").multiple(true), fn_provider(complete_catalog_urls)),
        mk(OptionDef::value("--shell").value_name("SHELL"), fn_provider(complete_shells)),
    ]
}

/// The shared value resolvers for
/// [`veks_completion::cli::build_completion_tree`]: the dataset-domain
/// map from [`vectordata::datasets::dyncomp::datasets_resolvers`]
/// (flag keys plus the `datasets …` positional path keys), extended
/// with the veks-specific per-flag completers (`--metric`/`--shell`).
/// These are the dynamic completers the derive structs don't carry on
/// their own `OptionSpec`s.
pub fn shared_resolvers() -> std::collections::BTreeMap<String, ValueProvider> {
    let mut map = datasets_resolvers();
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

/// Suggest distance metrics.
fn complete_metrics(partial: &str, _context: &[&str]) -> Vec<String> {
    let metrics = ["L2", "COSINE", "DOT_PRODUCT", "L1"];
    let prefix = partial.to_uppercase();
    metrics.iter()
        .filter(|m| prefix.is_empty() || m.starts_with(&prefix))
        .map(|m| m.to_string())
        .collect()
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
