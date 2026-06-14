// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `<binary> datasets ping` — verify catalog + dataset access through
//! the unified resolver.
//!
//! Ping opens the dataset via [`crate::catalog::Catalog::open`] — the
//! same path the runtime uses — so the verification chain covers
//! every quirk of the access layer: canonical `catalog.json` /
//! `catalog.yaml` lookup, `knn_entries.yaml`-shape synthesis, s3 →
//! https scheme translation, mref-backed cache routing, and chunked
//! HTTP fallback. Then for every facet it opens a [`FacetStorage`]
//! (which probes `.mref` / does the HEAD that confirms reachability)
//! and reports per-facet OK / FAIL plus byte count.
//!
//! The previous implementation rolled its own `reqwest::blocking`
//! client, hand-composed `<base>/<dataset>/dataset.yaml` URLs, and
//! had no idea what `s3://` was. That bypassed every piece of the
//! unified access layer the runtime relies on — verification by
//! coincidence at best, false negatives on `s3://` catalogs always.

use crate::catalog::resolver::Catalog;
#[cfg(feature = "cli")]
use crate::catalog::sources::CatalogSources;

/// Shared clap-derived argument struct for `<binary> datasets
/// ping`. Both the `vectordata` and `veks` binaries import this.
#[cfg(feature = "cli")]
#[derive(Debug, veks_completion_derive::VeksCli)]
pub struct PingArgs {
    /// Pin ping to a single catalog location (URL or path). When
    /// omitted, every catalog configured under `<configdir>/
    /// catalogs.yaml` plus any `--catalog` extras is in play, and
    /// ping searches them all.
    #[arg(long = "at")]
    pub at: Option<String>,
    /// Dataset name in the catalog.
    pub dataset: String,
    /// Profile to ping.
    #[arg(long, default_value = "default")]
    pub profile: String,
}

/// Drive `ping` from a parsed [`PingArgs`].
#[cfg(feature = "cli")]
pub fn run_args(args: PingArgs, configdir: &str, catalog: &[String], at_extra: &[String]) -> i32 {
    let sources = if let Some(at) = args.at {
        // --at pins ping to a single catalog source. The user is
        // saying "verify the dataset *here*", so don't expand back
        // out to every configured catalog. Numbered shortcuts
        // (`--at 2`) resolve against the configured list.
        CatalogSources::new().add_catalogs(&[crate::catalog::sources::resolve_catalog_value(&at)])
    } else {
        CatalogSources::new()
            .configure(configdir)
            .add_catalogs(&crate::catalog::sources::resolve_catalog_values(catalog))
            .add_catalogs(&crate::catalog::sources::resolve_catalog_values(at_extra))
    };
    if sources.is_empty() {
        eprintln!("error: no catalog sources configured");
        eprintln!();
        eprintln!("Add a catalog with `vectordata config catalog add <URL-or-path>`,");
        eprintln!("or pass `--at <URL-or-path>` for one-off use.");
        return 1;
    }
    let cat = Catalog::of(&sources);
    run_via_catalog(&cat, &args.dataset, &args.profile)
}

/// Ping using a pre-built catalog. Used by the picker's Ping action
/// and by `run_args` after it has built the catalog from its CLI
/// inputs.
pub fn run_via_catalog(catalog: &Catalog, dataset_name: &str, profile_name: &str) -> i32 {
    // Pre-flight: every facet probe below opens through the cache
    // layer, which needs a resolvable cache directory. Without this
    // check, a missing cache_dir surfaced as N cryptic per-facet
    // "storage open" failures instead of the actual problem.
    if let Err(e) = crate::settings::cache_dir() {
        eprintln!("error: no usable cache directory is configured: {e}");
        eprintln!();
        eprintln!("Facet probes open through the cache layer, so ping cannot run without one.");
        eprintln!("Fix with:  vectordata config set cache auto");
        eprintln!("      or:  vectordata config set cache <directory>");
        return 1;
    }
    if catalog.find_exact(dataset_name).is_none() {
        eprintln!("error: dataset '{dataset_name}' not found in any configured catalog");
        eprintln!();
        eprintln!("Try `vectordata datasets list` to see what's reachable.");
        return 1;
    }
    println!("Pinging dataset '{dataset_name}' (profile '{profile_name}')");
    println!();

    let group = match catalog.open(dataset_name) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: failed to open dataset '{dataset_name}': {e}");
            return 1;
        }
    };

    println!("  Profiles:");
    let profile_names = group.profile_names();
    for name in &profile_names {
        if let Some(view) = group.profile(name) {
            let manifest = view.facet_manifest();
            let mut facets: Vec<&String> = manifest.keys().collect();
            facets.sort();
            let bc = view.base_count()
                .map(|n| format!("base_count={n}"))
                .unwrap_or_else(|| "full".into());
            println!("    {} ({}): {}", name, bc,
                facets.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
        }
    }
    println!();

    let view = match group.profile(profile_name) {
        Some(v) => v,
        None => {
            eprintln!("error: profile '{profile_name}' not found");
            eprintln!("  available: {:?}", profile_names);
            return 1;
        }
    };

    println!("  Probing facets for profile '{profile_name}':");
    let manifest = view.facet_manifest();
    let mut facets: Vec<&String> = manifest.keys().collect();
    facets.sort();
    let mut pass = 0;
    let mut fail = 0;
    for facet_name in &facets {
        let source = view.facet_source(facet_name).unwrap_or_else(|| "<unresolved>".to_string());
        print!("    {facet_name} ({source})... ");

        // open_facet_storage drives the canonical access layer:
        // s3:// is translated to https, .mref is probed, Content-
        // Length lands on the Storage handle. If this succeeds the
        // facet is reachable through every code path the runtime
        // uses for actual reads — no separate URL composition to
        // get out of sync.
        match view.open_facet_storage(facet_name) {
            Ok(storage) => {
                let size = storage.total_size();
                let locality = if storage.is_local() { " local" } else { " remote" };
                // Read the first record's header — proving the bytes
                // are actually servable (an open alone only proves
                // metadata), and reporting basic shape for uniform
                // facets. Side effect on remote storage: the covering
                // chunk lands in the cache, so the picker's survey
                // can show this dataset's size from now on.
                let shape = match storage.read_prefix(4) {
                    Ok(hdr) if hdr.len() == 4 => {
                        let dim = i32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
                        let ext = source.rsplit('.').next().unwrap_or("");
                        let elem = crate::io::infer_elem_size(ext);
                        if dim > 0 && dim <= 1_000_000 && elem > 0 && !crate::io::is_vvec_ext(ext) {
                            let bpr = 4 + dim as u64 * elem as u64;
                            format!(", dim={dim}, ~{} records", size / bpr)
                        } else if dim > 0 && dim <= 1_000_000 {
                            format!(", dim={dim}")
                        } else {
                            String::new()
                        }
                    }
                    Ok(_) => String::new(),
                    Err(e) => {
                        println!("FAILED reading first record: {e}");
                        fail += 1;
                        continue;
                    }
                };
                println!("OK{} ({}{})", locality, format_size(size), shape);
                pass += 1;
            }
            Err(e) => {
                println!("FAILED: {e}");
                fail += 1;
            }
        }
    }
    println!();
    println!("  Summary: {pass} facets OK, {fail} failed");
    if fail > 0 { 1 } else { 0 }
}

fn format_size(bytes: u64) -> String {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    const KIB: u64 = 1 << 10;
    if bytes >= GIB { format!("{:.1} GiB", bytes as f64 / GIB as f64) }
    else if bytes >= MIB { format!("{:.1} MiB", bytes as f64 / MIB as f64) }
    else if bytes >= KIB { format!("{:.1} KiB", bytes as f64 / KIB as f64) }
    else { format!("{bytes} B") }
}
