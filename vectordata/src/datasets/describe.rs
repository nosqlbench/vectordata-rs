// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata datasets describe` — print the full catalog descriptor
//! for a dataset + profile, the CLI rehoming of the picker's
//! `Describe` action.
//!
//! Same data the picker overlay shows — dataset name + path, catalog
//! attributes, profile metadata (`maxk`, `base_count`, `partition`),
//! every facet with its source / namespace / window — but as plain
//! text so it composes with `less`, `grep`, redirects, and
//! diff-against-an-expected-value scripts.

use crate::catalog::resolver::Catalog;
use crate::catalog::sources::CatalogSources;
use crate::dataset::CatalogEntry;

/// Shared clap-derived argument struct for `vectordata datasets
/// describe`. Mirrors the shape of `PingArgs` so the catalog-
/// resolution surface stays uniform across CLI commands.
#[cfg(feature = "cli")]
#[derive(Debug, clap::Args)]
pub struct DescribeArgs {
    /// Pin describe to a single catalog location (URL or path). When
    /// omitted, every catalog configured under `<configdir>/
    /// catalogs.yaml` plus any `--catalog` extras is in play.
    #[arg(long = "at")]
    pub at: Option<String>,
    /// `<dataset>` or `<dataset>:<profile>`. Profile defaults to
    /// `default` when omitted.
    pub spec: String,
}

/// Drive `describe` from parsed [`DescribeArgs`].
#[cfg(feature = "cli")]
pub fn run_args(args: DescribeArgs, configdir: &str, catalog: &[String], at_extra: &[String]) -> i32 {
    let sources = if let Some(at) = args.at {
        CatalogSources::new().add_catalogs(&[at])
    } else {
        CatalogSources::new()
            .configure(configdir)
            .add_catalogs(catalog)
            .add_catalogs(at_extra)
    };
    if sources.is_empty() {
        eprintln!("error: no catalog sources configured");
        eprintln!();
        eprintln!("Add a catalog with `vectordata config add-catalog <URL-or-path>`,");
        eprintln!("or pass `--at <URL-or-path>` for one-off use.");
        return 1;
    }
    let (dataset, profile) = split_spec(&args.spec);
    let cat = Catalog::of(&sources);
    run_via_catalog(&cat, &dataset, &profile)
}

/// Render the descriptor using a pre-built catalog. Used by both the
/// CLI path and any future programmatic caller.
pub fn run_via_catalog(catalog: &Catalog, dataset_name: &str, profile_name: &str) -> i32 {
    let Some(entry) = catalog.find_exact(dataset_name) else {
        eprintln!("error: dataset '{dataset_name}' not found in any configured catalog");
        eprintln!();
        eprintln!("Try `vectordata datasets list` to see what's reachable.");
        return 1;
    };
    render(entry, profile_name);
    0
}

/// Split `<dataset>[:<profile>]`. Profile defaults to `default`,
/// including the bare-trailing-colon case (`sift1m:` → `sift1m` +
/// `default`) so a stray keystroke doesn't produce a dataset name
/// with a useless trailing `:` that no catalog lookup resolves.
fn split_spec(spec: &str) -> (String, String) {
    match spec.split_once(':') {
        Some((d, p)) if !p.is_empty() => (d.to_string(), p.to_string()),
        Some((d, _))                  => (d.to_string(), "default".to_string()),
        None                          => (spec.to_string(), "default".to_string()),
    }
}

fn render(entry: &CatalogEntry, profile_name: &str) {
    println!("Dataset: {}", entry.name);
    println!("Profile: {profile_name}");
    println!("Path:    {}", entry.path);
    println!("Type:    {}", entry.dataset_type);

    if let Some(attrs) = entry.layout.attributes.as_ref() {
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

    let Some(profile) = entry.layout.profiles.profile(profile_name) else {
        println!();
        eprintln!("error: profile '{profile_name}' not found in dataset '{}'", entry.name);
        eprintln!();
        eprintln!("Available profiles: {}",
            entry.layout.profiles.profile_names().join(", "));
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

    println!();
    println!("Facets:");
    for (facet, view) in &profile.views {
        println!("  {facet}:");
        println!("    source        {}", view.source.path);
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

fn kv(key: &str, value: &str) {
    println!("  {key:<14} {value}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_spec_defaults_to_default_profile() {
        assert_eq!(split_spec("sift1m"), ("sift1m".to_string(), "default".to_string()));
    }

    #[test]
    fn split_spec_honours_explicit_profile() {
        assert_eq!(split_spec("sift1m:1m"), ("sift1m".to_string(), "1m".to_string()));
    }

    /// Bare trailing colon falls back to the default profile rather
    /// than producing an empty profile name that no lookup would
    /// resolve.
    #[test]
    fn split_spec_empty_profile_after_colon_is_default() {
        assert_eq!(split_spec("sift1m:"), ("sift1m".to_string(), "default".to_string()));
    }
}
