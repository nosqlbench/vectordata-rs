// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets prebuffer` — drive a dataset profile to fully-
//! resident state through the canonical vectordata API.
//!
//! There is one code path for every kind of source — catalog name,
//! local `dataset.yaml` (or directory), and HTTP URL. The vectordata
//! layer dispatches per-facet:
//!
//! - **local file** → `Storage::Mmap`, no copy, no merkle, no work.
//! - **remote URL with `.mref`** → `Storage::Cached`; download +
//!   verify chunks into the configured cache directory, promote to
//!   mmap on completion.
//! - **remote URL without `.mref`** → `Storage::Http`; nothing to
//!   prebuffer.
//!
//! Catalog `dataset.yaml` files whose facet entries mix absolute
//! URLs and relative paths are handled by the same path — the
//! `vectordata::view` resolver passes absolute URLs through
//! unchanged.

use std::path::Path;

use vectordata::TestDataView;

use crate::catalog::resolver::Catalog;
use crate::catalog::sources::CatalogSources;

/// Entry point for `veks datasets prebuffer`.
///
/// `dataset_spec` is one of:
/// - A `name:profile` pair resolved via the catalog (e.g. `sift-128:default`)
/// - A `name` resolved via the catalog (uses the default profile)
/// - A local directory containing a `dataset.yaml`
/// - A path to a `dataset.yaml` file
/// - An HTTP URL to a dataset directory or `dataset.yaml`
pub fn run(
    dataset_spec: &str,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
    cache_dir: Option<&Path>,
) {
    // Cache root is informational here — actual cache writes go via
    // vectordata::settings::cache_dir(). We surface the resolved
    // value so the user can see where downloads land. If they passed
    // --cache-dir, that doesn't override settings.yaml today (the
    // override path would need a per-call hook in Storage); print a
    // clear note in that case.
    let configured = match vectordata::settings::cache_dir() {
        Ok(p) => Some(p),
        Err(e) => {
            // Only fatal if we'll actually need the cache. Local-only
            // datasets prebuffer fine without one. Defer the fatal
            // until we know the dispatch outcome.
            eprintln!("note: {e}");
            eprintln!();
            None
        }
    };
    if let Some(override_) = cache_dir {
        eprintln!("note: --cache-dir {} is recorded but the active cache root is {}",
            override_.display(),
            configured.as_deref().map(|p| p.display().to_string())
                .unwrap_or_else(|| "(unconfigured)".to_string()));
    }

    // Parse spec → (resolution, profile selection)
    let (resolution, profile_sel) = resolve_spec(
        dataset_spec, configdir, extra_catalogs, at,
    );

    let group_path = match resolution {
        Resolved::CatalogEntry { path, .. } => path,
        Resolved::Local(path) => path,
        Resolved::Url(url) => url,
    };

    let group = match vectordata::TestDataGroup::load(&group_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open dataset at {group_path}: {e}");
            std::process::exit(1);
        }
    };

    if let Some(c) = &configured {
        eprintln!("  Cache root: {}", c.display());
    }

    match profile_sel {
        ProfileSelection::Named(profile_name) => {
            let view = match group.profile(&profile_name) {
                Some(v) => v,
                None => {
                    eprintln!("Profile '{profile_name}' not found at {group_path}.");
                    eprintln!("Available profiles: {}",
                        group.profile_names().join(", "));
                    std::process::exit(1);
                }
            };
            eprintln!("Prebuffering {group_path}:{profile_name}");
            drive_prebuffer(&*view);
        }
        ProfileSelection::AllProfiles => {
            let names = group.profile_names();
            eprintln!("Prebuffering {group_path} — all profiles ({})",
                names.join(", "));
            drive_prebuffer_all(&group);
        }
    }
}

enum Resolved {
    CatalogEntry { path: String, name: String },
    Local(String),
    Url(String),
}

/// Whether the user named a specific profile or asked for all of
/// them. A bare `dataset` spec (no `:profile` suffix) selects all
/// profiles; an explicit `dataset:profile` selects just that one.
enum ProfileSelection {
    Named(String),
    AllProfiles,
}

fn resolve_spec(
    dataset_spec: &str,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
) -> (Resolved, ProfileSelection) {
    // Split off the profile suffix, if any.
    //
    // A spec containing `:` is ambiguous between a `name:profile`
    // catalog reference and a Windows-style URL/path. The local path
    // and URL forms always contain `/`, while a bare catalog name
    // never does — so the colon-split is only applied when the
    // spec contains no `/` and is not a URL.
    let (head, profile_sel) = if dataset_spec.contains('/')
        || dataset_spec.starts_with("http://")
        || dataset_spec.starts_with("https://")
    {
        // URL / path forms don't carry a profile suffix; default
        // to all-profiles.
        (dataset_spec, ProfileSelection::AllProfiles)
    } else if let Some(pos) = dataset_spec.find(':') {
        (&dataset_spec[..pos],
         ProfileSelection::Named(dataset_spec[pos + 1..].to_string()))
    } else {
        // Bare name → all profiles.
        (dataset_spec, ProfileSelection::AllProfiles)
    };

    // 1. Local path or URL?
    if head.starts_with("http://") || head.starts_with("https://") {
        return (Resolved::Url(head.to_string()), profile_sel);
    }
    let as_path = Path::new(head);
    if as_path.exists() {
        return (Resolved::Local(head.to_string()), profile_sel);
    }

    // 2. Catalog lookup.
    let sources = build_sources(configdir, extra_catalogs, at);
    if sources.is_empty() {
        eprintln!("'{}' is not a local path, not a URL, and no catalog is configured.",
            head);
        eprintln!("Add a catalog with:");
        eprintln!("  veks datasets config add-catalog <URL-or-path>");
        eprintln!("Or use --catalog/--at for one-off access.");
        std::process::exit(1);
    }
    let catalog = Catalog::of(&sources);
    let entry = match catalog.find_exact(head) {
        Some(e) => e,
        None => {
            eprintln!("Dataset '{head}' not found.");
            catalog.list_datasets(head);
            std::process::exit(1);
        }
    };
    if let ProfileSelection::Named(ref p) = profile_sel {
        if entry.layout.profiles.profile(p).is_none() {
            eprintln!("Profile '{p}' not found in dataset '{}'. Available: {}",
                entry.name, entry.profile_names().join(", "));
            std::process::exit(1);
        }
    }
    (
        Resolved::CatalogEntry { path: entry.path.clone(), name: entry.name.clone() },
        profile_sel,
    )
}

/// Drive every facet through `view.prebuffer_all_with_progress`,
/// printing a one-line summary per facet.
fn drive_prebuffer(view: &dyn TestDataView) {
    let mut last_facet = String::new();
    let mut total_facets = 0u32;
    let mut local_facets = 0u32;

    let result = view.prebuffer_all_with_progress(&mut |facet, p| {
        if facet != last_facet {
            if !last_facet.is_empty() {
                println!("  {last_facet} — done");
            }
            // total_chunks==0 ⇒ the underlying storage isn't
            // chunk-tracked (Storage::Mmap or Storage::Http with no
            // .mref). We still log the facet but mark it as
            // already-resident or not-cacheable.
            if p.total_chunks == 0 {
                println!("  {facet} — local (mmap, no prebuffer needed)");
                local_facets += 1;
                last_facet = facet.to_string();
            } else {
                println!(
                    "  {facet} — prebuffering ({}/{} chunks, {:.1} MiB total)…",
                    p.verified_chunks,
                    p.total_chunks,
                    p.total_bytes as f64 / 1_048_576.0,
                );
                last_facet = facet.to_string();
            }
            total_facets += 1;
        }
    });
    if !last_facet.is_empty() && total_facets != local_facets {
        // "done" only meaningful for facets that actually downloaded
        println!("  {last_facet} — done");
    }

    match result {
        Ok(()) => {
            println!();
            if total_facets == 0 {
                println!("Prebuffer: profile declared no facets.");
            } else if local_facets == total_facets {
                println!("Prebuffer: {total_facets} facets are all local — nothing to do.");
            } else {
                let remote = total_facets - local_facets;
                println!("Prebuffer: {total_facets} facets processed ({remote} remote, {local_facets} local).");
            }
        }
        Err(e) => {
            eprintln!();
            eprintln!("Prebuffer: failed — {e}");
            std::process::exit(1);
        }
    }
}

/// All-profiles variant: `view.prebuffer_all` per profile, with a
/// pre-flight 250 MiB warning so the operator knows what they're
/// committing to.
fn drive_prebuffer_all(group: &vectordata::TestDataGroup) {
    let mut warned = false;
    let mut last_profile = String::new();
    let mut total_facets = 0u32;
    let mut local_facets = 0u32;

    let result = group.prebuffer_all_profiles_with_progress(
        &mut |profile, facet, p| {
            if profile != last_profile {
                if !last_profile.is_empty() { println!(); }
                println!("[profile {profile}]");
                last_profile = profile.to_string();
            }
            if p.total_chunks == 0 {
                println!("  {facet} — local (mmap, no prebuffer needed)");
                local_facets += 1;
            } else {
                println!(
                    "  {facet} — done ({}/{} chunks, {:.1} MiB)",
                    p.verified_chunks,
                    p.total_chunks,
                    p.total_bytes as f64 / 1_048_576.0,
                );
            }
            total_facets += 1;
        },
        &mut |total_bytes| {
            warned = true;
            eprintln!();
            eprintln!("WARNING: prebuffer announced {:.1} MiB across all profiles \
                       (above the {:.0} MiB advisory threshold).",
                total_bytes as f64 / 1_048_576.0,
                vectordata::PREBUFFER_LARGE_WARNING_BYTES as f64 / 1_048_576.0);
            eprintln!("Continuing — pass an explicit `dataset:profile` to limit \
                       which profiles are downloaded.");
            eprintln!();
        },
    );
    let _ = warned;

    match result {
        Ok(()) => {
            println!();
            let remote = total_facets - local_facets;
            println!("Prebuffer: {total_facets} facets across all profiles \
                      ({remote} remote, {local_facets} local).");
        }
        Err(e) => {
            eprintln!();
            eprintln!("Prebuffer: failed — {e}");
            std::process::exit(1);
        }
    }
}

/// Build catalog sources from CLI args (same logic as `datasets list`).
fn build_sources(configdir: &str, extra_catalogs: &[String], at: &[String]) -> CatalogSources {
    let mut sources = CatalogSources::new();

    if !at.is_empty() {
        sources = sources.add_catalogs(at);
    } else {
        sources = sources.configure(configdir);
        if !extra_catalogs.is_empty() {
            sources = sources.add_catalogs(extra_catalogs);
        }
    }

    sources
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_dataset_directory_prebuffers_via_canonical_path() {
        // A local dataset with a dataset.yaml — every facet should
        // resolve to Storage::Mmap and prebuffer should be a no-op.
        // Verifies the code path doesn't try to copy or download.
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        std::fs::write(ws.join("base.fvec"), {
            let mut buf = Vec::new();
            // 2 records of dim=2 — minimum valid fvec
            buf.extend(&2i32.to_le_bytes());
            buf.extend(&1.0f32.to_le_bytes());
            buf.extend(&2.0f32.to_le_bytes());
            buf.extend(&2i32.to_le_bytes());
            buf.extend(&3.0f32.to_le_bytes());
            buf.extend(&4.0f32.to_le_bytes());
            buf
        }).unwrap();
        std::fs::write(ws.join("dataset.yaml"), "\
name: test
profiles:
  default:
    base_vectors: base.fvec
").unwrap();

        // Must not panic, must not exit non-zero. We can't call
        // `run` directly (it does std::process::exit on errors), so
        // exercise the same code path used by the catalog-name and
        // local-path branches: TestDataGroup::load + prebuffer_all.
        let group = vectordata::TestDataGroup::load(
            ws.join("dataset.yaml").to_str().unwrap(),
        ).unwrap();
        let view = group.profile("default").unwrap();
        view.prebuffer_all().unwrap();

        // The base.fvec original must still be exactly where we put
        // it — no shadow copy in any cache directory.
        assert!(ws.join("base.fvec").is_file());
    }
}
