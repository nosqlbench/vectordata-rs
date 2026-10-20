// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI support. `vectordata datasets describe` — print the full catalog descriptor
//! for a dataset + profile, the CLI rehoming of the picker's
//! `Describe` action.
//!
//! Same data the picker overlay shows — dataset name + path, catalog
//! attributes, profile metadata (`maxk`, `base_count`, `partition`),
//! every facet with its source / namespace / window — but as plain
//! text so it composes with `less`, `grep`, redirects, and
//! diff-against-an-expected-value scripts.

use crate::catalog::resolver::Catalog;
#[cfg(feature = "cli")]
use crate::dataset::selector::DatasetSpec;
use crate::dataset::CatalogEntry;

/// Shared clap-derived argument struct for `vectordata datasets
/// describe`. Mirrors the shape of `PingArgs` so the catalog-
/// resolution surface stays uniform across CLI commands.
#[cfg(feature = "cli")]
#[derive(Debug, veks_completion_derive::VeksCli)]
pub struct DescribeArgs {
    /// Pin describe to a single catalog location (URL or path). When
    /// omitted, every catalog configured under `<configdir>/
    /// catalogs.yaml` plus any `--catalog` extras is in play.
    #[arg(long = "at")]
    pub at: Option<String>,
    /// `<dataset>` or `<dataset>:<selector>` (PS-1). No selector means
    /// `default`; a selector must name exactly one profile, so
    /// `ds:size=10m,predicates=uniform-2` works where it is unique and
    /// `ds:family=uniform` is refused listing what it matched (PS-9).
    pub spec: String,
}

/// Drive `describe` from parsed [`DescribeArgs`].
#[cfg(feature = "cli")]
pub fn run_args(args: DescribeArgs, program: &crate::catalog::CatalogSources, configdir: &str, catalog: &[String]) -> i32 {
    // `--at` pins describe to a single catalog source; numbered
    // shortcuts (`--at 2`) resolve against the configured list.
    // `--at` pins the command to the catalogs it names, numbered
    // shortcuts (`--at 2`) included, by the one assembly every
    // command uses.
    let at: Vec<String> = args.at.iter().cloned().collect();
    let sources = super::build_sources(program, configdir, catalog, &at);
    if sources.is_empty() {
        eprintln!("error: no catalog sources configured");
        eprintln!();
        eprintln!("Add a catalog with `vectordata config catalog add <URL-or-path>`,");
        eprintln!("or pass `--at <URL-or-path>` for one-off use.");
        return 1;
    }
    let spec = match DatasetSpec::parse(&args.spec) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let cat = match sources.open() {
        Ok(cat) => cat,
        Err(code) => return code,
    };
    run_via_catalog(&cat, &spec.head, spec.selector.as_ref().map(|s| s.text()))
}

/// Render the descriptor using a pre-built catalog. Used by both the
/// CLI path and any future programmatic caller.
///
/// `selector` is resolved against the entry before anything is
/// rendered (PS-9): `None` is `default`, and a selector matching more
/// or fewer than one profile is an error naming what it matched.
pub fn run_via_catalog(catalog: &Catalog, dataset_name: &str, selector: Option<&str>) -> i32 {
    // `lookup`, not `find_exact`: a name two catalogs hold is ambiguous,
    // and saying so is the opposite of saying it is nowhere.
    let entry = match catalog.lookup(dataset_name) {
        Ok(entry) => entry,
        Err(e) => {
            super::report_lookup_failure(catalog, &e);
            return 1;
        }
    };
    let profile_name = match entry.select_one(selector) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: dataset '{}': {e}", entry.name());
            return 1;
        }
    };
    render(entry, &profile_name);
    0
}

/// Render an attribute value for a terminal line.
///
/// Scalars print bare; anything structured falls back to its YAML,
/// trimmed. Unknown keys and unusual shapes are reported rather than
/// dropped — the conventional key set is a convention (P-5).
fn render_attribute(value: &serde_yaml::Value) -> String {
    match value {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        serde_yaml::Value::Null => "null".to_string(),
        other => serde_yaml::to_string(other)
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

/// The attribute keys whose values differ across `name`'s family.
///
/// A family varies along one axis, so the key that differs is the
/// parameter — the thing switching profiles actually changes. Keys
/// every member agrees on are invariants and say nothing about the
/// choice.
fn varying_attributes(
    profiles: &crate::dataset::profile::DSProfileGroup,
    name: &str,
) -> Vec<String> {
    let members = profiles.family_attributes(name);
    if members.len() < 2 {
        return Vec::new();
    }
    let mut keys: Vec<String> = members
        .iter()
        .flat_map(|(_, attrs)| attrs.keys().cloned())
        .collect();
    keys.sort();
    keys.dedup();
    keys.retain(|key| {
        let mut seen = members.iter().map(|(_, attrs)| attrs.get(key));
        let first = seen.next().unwrap_or(None);
        seen.any(|v| v != first)
    });
    keys
}

fn render(entry: &CatalogEntry, profile_name: &str) {
    println!("Dataset: {}", entry.name());
    println!("Profile: {profile_name}");
    println!("Path:    {}", entry.location());
    println!("Type:    {}", entry.dataset_type());

    if let Some(attrs) = entry.attributes() {
        let mut have_any = false;
        let mut header = || {
            if !have_any { println!(); println!("Attributes:"); have_any = true; }
        };
        if let Some(v) = attrs.model.as_deref()
            { header(); kv("model", v); }
        if let Some(v) = attrs.distance_function.as_deref()
            { header(); kv("distance", v); }
        if let Some(v) = attrs.vendor.as_deref()
            { header(); kv("vendor", v); }
        if let Some(v) = attrs.license.as_deref()
            { header(); kv("license", v); }
        if let Some(v) = attrs.url.as_deref()
            { header(); kv("upstream", v); }
        if let Some(b) = attrs.is_normalized
            { header(); kv("normalized", if b { "yes" } else { "no" }); }
        if let Some(v) = attrs.notes.as_deref()
            { header(); kv("notes", v); }
    }

    let Some(profile) = entry.profiles().profile(profile_name) else {
        println!();
        eprintln!("error: profile '{profile_name}' not found in dataset '{}'", entry.name());
        eprintln!();
        eprintln!("Available profiles: {}",
            entry.profiles().profile_names().join(", "));
        // Soft-fail render — the dataset-level info above is still
        // useful to the user even when the profile is the typo.
        // Exit code is handled by the caller; we just stop here.
        std::process::exit(1);
    };

    println!();
    println!("Profile:");
    if let Some(maxk) = profile.maxk
        { kv("maxk", &maxk.to_string()); }
    if let Some(bc) = profile.base_count
        { kv("base_count", &crate::dataset::source::format_count_with_suffix(bc)); }
    if profile.partition
        { kv("partition", "yes (independent base)"); }
    if let Some(parent) = profile.inherits.as_deref()
        { kv("inherits", parent); }

    // What this profile *is*, before what it points at (P-12).
    //
    // Absent attributes report as undescribed rather than as zeroes: a
    // profile that never recorded a selectivity has not recorded one of
    // 0.0, and printing the latter invents a measurement (P-7).
    println!();
    println!("Attributes:");
    if profile.attributes.is_empty() {
        kv("(undescribed)", "no attributes recorded");
    } else {
        for (key, value) in &profile.attributes {
            kv(key, &render_attribute(value));
        }
    }

    // The family this profile belongs to, and what distinguishes it
    // from its siblings — which is what a reader actually needs when
    // deciding which profile to open (P-12).
    if let Some((spec, members)) = entry.profiles().family_of(profile_name) {
        println!();
        println!("Family:");
        kv("spec", spec);
        kv("members", &members.join(", "));
        let varying = varying_attributes(entry.profiles(), profile_name);
        if varying.is_empty() {
            kv("varies by", "(nothing recorded — members are undescribed)");
        } else {
            for key in varying {
                let here = profile
                    .attributes
                    .get(&key)
                    .map(render_attribute)
                    .unwrap_or_else(|| "(undescribed)".to_string());
                kv(&format!("varies by {key}"), &format!("this profile: {here}"));
            }
        }
    }

    println!();
    println!("Facets:");
    for (facet, view) in &profile.views {
        println!("  {facet}:");
        if view.is_series() {
            // Present the series as one facet with a shard count, and
            // report a stride **only when the lengths are uniform**
            // (SH-46) — a non-uniform series has no stride, and
            // printing one would be a number that does not exist.
            //
            // The shards are the files the shard model realizes, the
            // planner's and `FacetDescriptor::shard_count`'s count: a
            // `__NNNN` pattern is one declared source naming many files.
            let files = match series_files(facet, view) {
                Ok(files) => files,
                Err(e) => {
                    println!("    shards        (invalid declaration: {e})");
                    continue;
                }
            };
            println!("    shards        {}", files.len());
            if let Some(stride) = view.shard_stride {
                println!("    shard_stride  {stride}");
            }
            if let Some(n) = view.record_count {
                println!("    records       {}",
                    crate::dataset::source::format_count_with_suffix(n));
            }
            if let [pattern] = view.declaration_sources().as_slice()
                && crate::dataset::shards::has_shard_field(pattern)
            {
                println!("    pattern       {pattern}");
            }
            for (i, file) in files.iter().enumerate() {
                println!("    [{i:>4}]        {file}");
            }
        } else {
            println!("    source        {}", view.source.path);
        }
        if let Some(ns) = view.source.namespace.as_deref() {
            println!("    namespace     {ns}");
        }
        // Both window fields the catalog YAML can populate. The
        // view-level override (sibling of `source`) shadows the
        // source-level one when both are set; the CLI surfaces both
        // so users can tell which one is winning.
        if let Some(view_w) = view.window.as_ref()
            && let Some(iv) = view_w.0.first()
        {
            println!("    window        [{}..{}) (view override)",
                iv.min_incl, iv.max_excl);
        }
        if let Some(iv) = view.source.window.0.first() {
            println!("    source.window [{}..{})", iv.min_incl, iv.max_excl);
        }
    }
}

/// The files a series view names, in ordinal order, as the shard
/// model realizes its declaration: a uniform `__NNNN` series expands to
/// its `shard_count` files with no file read. An explicit series is one
/// file per listed entry; the model would size an entry with no declared
/// count by opening it, which describing never does, so such a series is
/// reported as listed. A uniform declaration the model refuses is the
/// error.
fn series_files(facet: &str, view: &crate::dataset::profile::DSView) -> Result<Vec<String>, String> {
    let sources = view.declaration_sources();
    let no_read = |_: &crate::dataset::source::DSSource| Err("describe opens no file".to_string());
    match crate::dataset::shards::realize(facet, &view.declaration(&sources), &no_read) {
        Ok(shards) => Ok(shards.entries().iter().map(|e| e.source.to_string()).collect()),
        Err(e) if sources.first().is_some_and(|s| crate::dataset::shards::has_shard_field(s)) => Err(e.to_string()),
        Err(_) => Ok(sources),
    }
}

fn kv(key: &str, value: &str) {
    println!("  {key:<14} {value}");
}
