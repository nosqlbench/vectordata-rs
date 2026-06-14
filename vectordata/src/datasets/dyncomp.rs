// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dynamic tab-completion resolvers for the `datasets` command family.
//!
//! This is the canonical home of the dataset-domain completers —
//! catalog dataset names, profile names, `name[:profile]` specs, and
//! numbered catalog shortcuts for `--at`. Both the `vectordata` and
//! `veks` binaries register the same resolvers via
//! [`datasets_resolvers`], so the completion surface lives with the
//! commands it completes rather than in any one binary.
//!
//! Layering follows the house rule for env/config-adjacent code:
//! every decision is a pure function over explicit inputs (the typed
//! partial, the engine-provided `context` words, the configured
//! catalog list), and the IO wrappers registered with the completion
//! engine are one-call-deep shims over those cores. Tests target the
//! pure layer only.

use std::collections::BTreeMap;

use crate::catalog::resolver::Catalog;
use crate::catalog::sources::{self, CatalogSources};
use veks_completion::{ValueProvider, fn_provider};

/// The dataset-domain resolvers, keyed for
/// [`veks_completion::cli::build_completion_tree`]: `--flag` keys
/// attach to every command that declares the flag; space-joined path
/// keys attach a first-positional completer to that command.
///
/// Merge this map into a binary's own resolver set:
///
/// ```ignore
/// let mut resolvers = vectordata::datasets::dyncomp::datasets_resolvers();
/// resolvers.insert("--to".into(), fn_provider(complete_push_to));
/// ```
pub fn datasets_resolvers() -> BTreeMap<String, ValueProvider> {
    let mut map = BTreeMap::new();
    map.insert("--dataset".to_string(), fn_provider(complete_dataset_names));
    map.insert("--profile".to_string(), fn_provider(complete_profile_names));
    map.insert("--at".to_string(), fn_provider(complete_catalog_urls));
    // The `list` filter values, completed from what's actually in
    // the catalog (respecting `--at` and any filters already typed).
    map.insert("--matching-name".to_string(), fn_provider(complete_dataset_names));
    map.insert("--matching-profile".to_string(), fn_provider(complete_profile_names));
    map.insert("--with-metric".to_string(), fn_provider(complete_with_metric));
    map.insert("--with-vtype".to_string(), fn_provider(complete_with_vtype));
    map.insert("--with-facet".to_string(), fn_provider(complete_with_facet));
    // Exact flags stay present-only; min/max thresholds add a ladder.
    map.insert("--with-dim".to_string(), fn_provider(complete_with_dim));
    map.insert("--with-min-dim".to_string(), fn_provider(complete_with_dim_threshold));
    map.insert("--with-max-dim".to_string(), fn_provider(complete_with_dim_threshold));
    map.insert("--with-count".to_string(), fn_provider(complete_with_count));
    map.insert("--with-min-count".to_string(), fn_provider(complete_with_count_threshold));
    map.insert("--with-max-count".to_string(), fn_provider(complete_with_count_threshold));
    map.insert("--with-data".to_string(), fn_provider(complete_with_data));
    map.insert("--with-min-data".to_string(), fn_provider(complete_with_data_threshold));
    map.insert("--with-max-data".to_string(), fn_provider(complete_with_data_threshold));
    map.insert("--select".to_string(), fn_provider(complete_select_specs));
    // Positional slots: `ping <dataset>`, `describe <dataset[:profile]>`,
    // `precache <dataset[:profile]>`. Keys that don't match a command
    // path in a given binary's spec are inert.
    map.insert("datasets ping".to_string(), fn_provider(complete_dataset_names));
    map.insert("datasets describe".to_string(), fn_provider(complete_dataset_specs));
    map.insert("datasets precache".to_string(), fn_provider(complete_dataset_specs));
    map
}

// ---------------------------------------------------------------------------
// Numbered catalog shortcuts (`--at 2` → second configured catalog)
// ---------------------------------------------------------------------------
//
// The shortcut semantics live in [`crate::catalog::sources`]
// (`lookup_catalog_index` / `resolve_catalog_value`) so the shared,
// non-`cli` run paths can resolve them too. Completion only needs
// the forgiving variant below.

/// Completion-time variant of [`sources::resolve_catalog_value`]: an
/// out-of-range index yields `None` (the candidate set just shrinks)
/// instead of exiting — aborting a tab-completion callback over a
/// typo'd index would help no one.
fn resolve_catalog_value_opt(value: &str, configured: &[String]) -> Option<String> {
    match sources::lookup_catalog_index(value, configured) {
        None => Some(value.to_string()),
        Some(Ok(url)) => Some(url),
        Some(Err(_)) => None,
    }
}

// ---------------------------------------------------------------------------
// Context-word extraction (pure)
// ---------------------------------------------------------------------------
//
// The completion engine hands every value provider the `context`
// slice — the words already typed after the resolved command. That
// slice is the ONLY correct source of cross-flag context (`--at`,
// `--dataset`) during completion: the process argv at completion
// time is `[binary, $COMP_LINE, $COMP_POINT]`, so scanning
// `std::env::args()` for flags finds nothing.

/// The value following the last `option` occurrence in `context`.
fn option_value(context: &[&str], option: &str) -> Option<String> {
    context
        .windows(2)
        .rev()
        .find(|w| w[0] == option)
        .map(|w| w[1].to_string())
}

/// Every value following an `option` occurrence in `context`.
fn option_values(context: &[&str], option: &str) -> Vec<String> {
    context
        .windows(2)
        .filter(|w| w[0] == option)
        .map(|w| w[1].to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Catalog access (IO) + the (name, profiles) view the pure cores use
// ---------------------------------------------------------------------------

/// Build a catalog resolver honoring any `--at` values already typed
/// on the line; falls back to the configured catalogs.
fn resolve_catalog(context: &[&str]) -> Catalog {
    let configured = sources::raw_catalog_entries(&sources::config_dir());
    let at_values: Vec<String> = option_values(context, "--at")
        .iter()
        .filter_map(|v| resolve_catalog_value_opt(v, &configured))
        .collect();

    let catalog_sources = if at_values.is_empty() {
        CatalogSources::new().configure_default()
    } else {
        CatalogSources::new().add_catalogs(&at_values)
    };
    Catalog::of(&catalog_sources)
}

/// Flatten a catalog into the `(dataset, profiles)` pairs the pure
/// completion cores operate on.
fn dataset_profile_pairs(catalog: &Catalog) -> Vec<(String, Vec<String>)> {
    catalog
        .datasets()
        .iter()
        .map(|e| {
            (
                e.name.clone(),
                e.profile_names().into_iter().map(|s| s.to_string()).collect(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Pure completion cores
// ---------------------------------------------------------------------------

/// Case-insensitive prefix filter over dataset names.
pub fn filter_dataset_names(pairs: &[(String, Vec<String>)], partial: &str) -> Vec<String> {
    let prefix = partial.to_lowercase();
    pairs
        .iter()
        .map(|(name, _)| name.clone())
        .filter(|n| prefix.is_empty() || n.to_lowercase().starts_with(&prefix))
        .collect()
}

/// Profile-name candidates. When `dataset` is given (from a typed
/// `--dataset` flag or an inferred bare positional), only that
/// dataset's profiles are offered; otherwise the union across the
/// catalog, deduplicated and sorted.
pub fn filter_profile_names(
    pairs: &[(String, Vec<String>)],
    dataset: Option<&str>,
    partial: &str,
) -> Vec<String> {
    let prefix = partial.to_lowercase();
    let mut profiles = std::collections::BTreeSet::new();
    for (name, entry_profiles) in pairs {
        if let Some(ds) = dataset
            && !name.eq_ignore_ascii_case(ds)
        {
            continue;
        }
        for p in entry_profiles {
            if prefix.is_empty() || p.to_lowercase().starts_with(&prefix) {
                profiles.insert(p.clone());
            }
        }
    }
    profiles.into_iter().collect()
}

/// `name[:profile]` spec candidates (the `describe`/`precache`
/// positional). A partial without `:` completes dataset names; once
/// the user types `name:` the candidates become that dataset's
/// `name:profile` forms — splice-ready because the bash hook's
/// `COMP_WORDBREAKS` keeps `name:prof` one shell word.
pub fn filter_spec_candidates(pairs: &[(String, Vec<String>)], partial: &str) -> Vec<String> {
    let Some((name_part, profile_part)) = partial.split_once(':') else {
        return filter_dataset_names(pairs, partial);
    };
    let profile_prefix = profile_part.to_lowercase();
    pairs
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case(name_part))
        .flat_map(|(name, profiles)| {
            profiles
                .iter()
                .filter(|p| profile_prefix.is_empty() || p.to_lowercase().starts_with(&profile_prefix))
                .map(|p| format!("{name}:{p}"))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Infer the dataset a profile completion should scope to when no
/// `--dataset` flag is on the line: the first bare context word that
/// names a catalog dataset (covers `ping <dataset> --profile <TAB>`,
/// where the dataset is positional). The `name[:profile]` spec shape
/// is honored too.
pub fn infer_dataset(context: &[&str], pairs: &[(String, Vec<String>)]) -> Option<String> {
    context
        .iter()
        .filter(|w| !w.starts_with('-'))
        .map(|w| w.split_once(':').map(|(n, _)| n).unwrap_or(w))
        .find_map(|w| {
            pairs
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(w))
                .map(|(name, _)| name.clone())
        })
}

/// Numbered `--at` candidates over the configured catalog list:
/// 1-based indices (matched by index or by URL prefix), returned as
/// the index strings. The index → URL legend is the IO wrapper's
/// concern.
pub fn filter_catalog_indices(configured: &[String], partial: &str) -> Vec<(String, String)> {
    configured
        .iter()
        .enumerate()
        .filter_map(|(i, url)| {
            let num = (i + 1).to_string();
            (partial.is_empty() || num.starts_with(partial) || url.starts_with(partial))
                .then(|| (num, url.clone()))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// IO wrappers registered with the completion engine
// ---------------------------------------------------------------------------

/// Suggest dataset names from catalogs (honors `--at` on the line).
pub fn complete_dataset_names(partial: &str, context: &[&str]) -> Vec<String> {
    let pairs = dataset_profile_pairs(&resolve_catalog(context));
    filter_dataset_names(&pairs, partial)
}

/// Suggest profile names (honors `--at`, `--dataset`, and a bare
/// positional dataset already on the line).
pub fn complete_profile_names(partial: &str, context: &[&str]) -> Vec<String> {
    let pairs = dataset_profile_pairs(&resolve_catalog(context));
    let dataset = option_value(context, "--dataset").or_else(|| infer_dataset(context, &pairs));
    filter_profile_names(&pairs, dataset.as_deref(), partial)
}

/// Suggest `name[:profile]` specs for the `describe`/`precache`
/// positional (honors `--at` on the line).
pub fn complete_dataset_specs(partial: &str, context: &[&str]) -> Vec<String> {
    let pairs = dataset_profile_pairs(&resolve_catalog(context));
    filter_spec_candidates(&pairs, partial)
}

/// Run `f` over the context catalog's entries, narrowed by the
/// filters already typed on the line — so each filter completer only
/// offers values consistent with the narrowing so far.
///
/// `completer_needs_probe` says whether the completer being served
/// derives candidates from facet probes (count / dim / data). The
/// probe cache is warmed only when that is true OR an active filter
/// is itself probe-dependent ([`DatasetFilter::needs_probe`]) —
/// otherwise this is pure-offline and does no facet I/O at all, so
/// completing `--with-metric` / `--with-vtype` / `--dataset` never
/// touches the network.
///
/// When warmed, narrowing is probe-CONSISTENT with a real run:
/// dimension / count / data predicates the catalog metadata can't
/// answer are matched against a [`FacetProbeCache`] warmed once per
/// TAB (bounded parallel reads, persisted memo), so an entry the
/// command would match via probing is not silently dropped here. `f`
/// receives the resolved [`ProbeMode`] so candidate extraction sees
/// the same values. A TAB never does unbounded blocking I/O: warm-up
/// is deadline-bounded and a cache miss reads as "unknown".
fn with_filtered_entries<T>(
    context: &[&str],
    completer_needs_probe: bool,
    f: impl FnOnce(&[&crate::dataset::CatalogEntry], crate::datasets::filter::ProbeMode) -> T,
) -> T {
    use crate::datasets::filter::{FacetProbeCache, ProbeMode};
    let catalog = resolve_catalog(context);
    let all: Vec<&crate::dataset::CatalogEntry> = catalog.datasets().iter().collect();
    let (filter, pv) = crate::datasets::filter::parse_active_filters(context);

    // Warm the cache only when probe values are actually needed —
    // either by this completer or by a probe-dependent active filter
    // (whose narrowing must stay consistent with the real run).
    let needs_probe = completer_needs_probe || filter.needs_probe();
    let cache: FacetProbeCache = if needs_probe { warm_probe_cache(&all) } else { Default::default() };
    let mode = if needs_probe { ProbeMode::Cached(&cache) } else { ProbeMode::Off };

    let refs: Vec<&crate::dataset::CatalogEntry> = all
        .into_iter()
        .filter(|e| filter.matches_with(e, mode))
        .filter(|e| !pv.is_active() || !pv.matching_profiles(e).is_empty())
        .collect();
    f(&refs, mode)
}

/// Case-insensitive prefix filter over completed values.
fn ci_prefix_filter(values: Vec<String>, partial: &str) -> Vec<String> {
    let prefix = partial.to_lowercase();
    values
        .into_iter()
        .filter(|v| prefix.is_empty() || v.to_lowercase().starts_with(&prefix))
        .collect()
}

// Every `--with-*` value completer below offers ONLY values actually
// present in the (already-narrowed) catalog — never a hardcoded
// vocabulary. If no entry has the attribute (or no entry matches the
// filters typed so far), the candidate set is empty: showing values
// no dataset has would misrepresent the catalog, which is not what
// completion is for.

/// Suggest `--with-metric` values: the canonical metrics present
/// among the already-narrowed catalog entries.
pub fn complete_with_metric(partial: &str, context: &[&str]) -> Vec<String> {
    let values =
        with_filtered_entries(context, false, |e, _| crate::datasets::filter::distinct_metrics(e));
    ci_prefix_filter(values, partial)
}

/// Suggest `--with-vtype` values: the vector data types present among
/// the already-narrowed entries.
pub fn complete_with_vtype(partial: &str, context: &[&str]) -> Vec<String> {
    let values =
        with_filtered_entries(context, false, |e, _| crate::datasets::filter::distinct_vtypes(e));
    ci_prefix_filter(values, partial)
}

/// Suggest `--with-facet` values: the facet names present among the
/// already-narrowed entries, each paired with its capital code letter
/// (`B`, `Q`, `G`, …) since the filter accepts codes as facet
/// vocabulary. Codes derive from the present facets — no standard-set
/// fallback, so a code only appears when a dataset actually has that
/// facet.
pub fn complete_with_facet(partial: &str, context: &[&str]) -> Vec<String> {
    use crate::dataset::facet::StandardFacet;
    let mut values =
        with_filtered_entries(context, false, |e, _| crate::datasets::filter::distinct_facets(e));
    let codes: Vec<String> = values
        .iter()
        .filter_map(|key| StandardFacet::from_key(key))
        .filter_map(|f| f.code())
        .map(|c| c.to_string())
        .collect();
    for code in codes {
        if !values.contains(&code) {
            values.push(code);
        }
    }
    // Bare capital codes must survive the filter when the partial is
    // a capital letter, and the lowercase prefix rule covers keys.
    ci_prefix_filter(values, partial)
}

/// Round-number ladders merged into the min/max THRESHOLD completers
/// (`--with-min-*` / `--with-max-*`) on top of the present values, so
/// a user can pick a convenient boundary (e.g. `--with-min-data 1GB`)
/// even when no dataset sits exactly there. The EXACT flags
/// (`--with-count` / `--with-data` / `--with-dim`) stay present-only —
/// an exact value should name a dataset that actually has it.
const COUNT_LADDER: &[&str] = &["1k", "10k", "100k", "1m", "10m", "100m"];
const DATA_LADDER: &[&str] = &["1MB", "10MB", "100MB", "1GB", "10GB", "100GB", "1TB"];
const DIM_LADDER: &[&str] =
    &["64", "128", "256", "384", "512", "768", "1024", "1536", "2048", "3072", "4096"];

/// Merge a round-number `ladder` into `present` candidate values,
/// then sort by MAGNITUDE (via `key`) and drop duplicate display
/// strings. Sorting on the parsed magnitude, not the string, is what
/// keeps `2m` before `10m` and `118MB` before `1GB` — lexical order
/// would scramble them.
fn merge_threshold_ladder(
    mut present: Vec<String>,
    ladder: &[&str],
    key: impl Fn(&str) -> Option<u64>,
) -> Vec<String> {
    for l in ladder {
        present.push((*l).to_string());
    }
    present.sort_by_key(|s| key(s).unwrap_or(u64::MAX));
    present.dedup();
    present
}

/// The record-count candidates present among the narrowed entries,
/// in compact suffix form (`100k`, `1m`), magnitude-sorted. Counts
/// that exist only via probing (no `base_count` metadata) come from
/// the warm probe cache, so they appear here too.
fn present_count_candidates(context: &[&str]) -> Vec<String> {
    with_filtered_entries(context, true, crate::datasets::filter::distinct_base_counts)
}

/// The data-size candidates present among the narrowed entries, in
/// byte-styled form (`118MB`, `12GB`) so the data axis never reads
/// like the count axis, magnitude-sorted and de-duplicated.
fn present_data_candidates(context: &[&str]) -> Vec<String> {
    let mut sizes = with_filtered_entries(context, true, crate::datasets::filter::distinct_data_sizes);
    sizes.sort_unstable();
    let mut values: Vec<String> = sizes
        .into_iter()
        .map(crate::datasets::filter::format_bytes_approx)
        .collect();
    values.dedup();
    values
}

/// Suggest `--with-count` (EXACT): record counts present among the
/// narrowed entries, no ladder.
pub fn complete_with_count(partial: &str, context: &[&str]) -> Vec<String> {
    ci_prefix_filter(present_count_candidates(context), partial)
}

/// Suggest `--with-min-count` / `--with-max-count` (THRESHOLD):
/// present counts plus a round-number ladder, magnitude-sorted.
pub fn complete_with_count_threshold(partial: &str, context: &[&str]) -> Vec<String> {
    let merged = merge_threshold_ladder(present_count_candidates(context), COUNT_LADDER, |s| {
        crate::datasets::filter::parse_size(s).ok()
    });
    ci_prefix_filter(merged, partial)
}

/// Suggest `--with-data` (EXACT): data sizes present among the
/// narrowed entries, no ladder.
pub fn complete_with_data(partial: &str, context: &[&str]) -> Vec<String> {
    ci_prefix_filter(present_data_candidates(context), partial)
}

/// Suggest `--with-min-data` / `--with-max-data` (THRESHOLD): present
/// sizes plus a round-number byte ladder, magnitude-sorted.
pub fn complete_with_data_threshold(partial: &str, context: &[&str]) -> Vec<String> {
    let merged = merge_threshold_ladder(present_data_candidates(context), DATA_LADDER, |s| {
        crate::datasets::filter::parse_bytes(s).ok()
    });
    ci_prefix_filter(merged, partial)
}

/// Probe deadline per completion callback. One TAB spends at most
/// this long resolving facet probes; whatever lands afterwards is
/// picked up from the memo by the next TAB.
const PROBE_DEADLINE: std::time::Duration = std::time::Duration::from_millis(1500);

/// How long memoized facet probes stay trusted. Long enough that a
/// completion session never re-probes, short enough that a freshly
/// published facet shows up within a couple of minutes.
const PROBE_MEMO_TTL: std::time::Duration = std::time::Duration::from_secs(120);

/// Warm and return a [`FacetProbeCache`] covering `entries`' facets:
/// byte length for every facet, plus the dim header for each entry's
/// base-vectors facet. Values come from a persisted memo when fresh,
/// else from bounded parallel probes (local stat/read or authed
/// HEAD/range via the unified transport) capped by [`PROBE_DEADLINE`].
/// What lands is merged back into the memo — the first TAB warms it,
/// later TABs are instant. Whatever the deadline didn't reach is
/// simply absent (read as "unknown"), so completion degrades
/// gracefully rather than blocking.
fn warm_probe_cache(
    entries: &[&crate::dataset::CatalogEntry],
) -> crate::datasets::filter::FacetProbeCache {
    use crate::datasets::filter::{base_vectors_path, entry_facet_paths, probe_facet_live, FacetProbe};

    // What each path needs: bytes always; dim only for base facets.
    let mut want_dim: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut all_paths: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for e in entries {
        for p in entry_facet_paths(e) {
            all_paths.insert(p);
        }
        if let Some(base) = base_vectors_path(e) {
            want_dim.insert(base);
        }
    }

    let mut known = load_probe_memo();
    // A memo entry is sufficient only if it already carries the dim
    // for a path that needs one — else we must (re)probe that path.
    let missing: Vec<String> = all_paths
        .iter()
        .filter(|p| match known.get(*p) {
            None => true,
            Some(fp) => want_dim.contains(*p) && fp.dim.is_none() && fp.bytes.is_some(),
        })
        .cloned()
        .collect();

    if !missing.is_empty() {
        // Each job carries whether the dim header is wanted, so
        // bytes-only facets skip the extra 4-byte read.
        let jobs: std::collections::VecDeque<(String, bool)> =
            missing.into_iter().map(|p| { let d = want_dim.contains(&p); (p, d) }).collect();
        let queue = std::sync::Arc::new(std::sync::Mutex::new(jobs));
        let (tx, rx) = std::sync::mpsc::channel::<(String, FacetProbe)>();
        let workers = crate::cache::download_concurrency().clamp(1, 8);
        for _ in 0..workers {
            let queue = queue.clone();
            let tx = tx.clone();
            std::thread::spawn(move || {
                loop {
                    let job = match queue.lock() {
                        Ok(mut q) => q.pop_front(),
                        Err(_) => None,
                    };
                    let Some((path, want_dim)) = job else { break };
                    let probe = if want_dim {
                        probe_facet_live(&path)
                    } else {
                        FacetProbe { bytes: crate::datasets::filter::facet_len(&path), dim: None }
                    };
                    if tx.send((path, probe)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let deadline = std::time::Instant::now() + PROBE_DEADLINE;
        while let Some(left) = deadline.checked_duration_since(std::time::Instant::now())
            && let Ok((path, probe)) = rx.recv_timeout(left)
        {
            known.insert(path, probe);
        }
        save_probe_memo(&known);
    }

    known
}

/// Directory for completion scratch (the facet-probe memo, the
/// `--at` legend debounce marker). Anchored under the
/// VECTORDATA_HOME-isolated config tree (via [`sources::config_dir`]),
/// not raw system temp — so a test or tutorial that points
/// VECTORDATA_HOME at `target/` keeps its completion scratch there,
/// and two homes never collide. Best-effort creation; every caller
/// tolerates IO failure (completion must never fail over scratch).
fn completion_scratch_dir() -> std::path::PathBuf {
    let dir = std::path::Path::new(&sources::config_dir()).join(".completion-cache");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Path of the facet-probe memo. See [`completion_scratch_dir`].
fn probe_memo_path() -> std::path::PathBuf {
    completion_scratch_dir().join("facet_probes.json")
}

/// Load the memo when fresher than [`PROBE_MEMO_TTL`]; else empty.
fn load_probe_memo() -> crate::datasets::filter::FacetProbeCache {
    let path = probe_memo_path();
    let fresh = path
        .metadata()
        .and_then(|m| m.modified())
        .map(|t| t.elapsed().unwrap_or_default() < PROBE_MEMO_TTL)
        .unwrap_or(false);
    if !fresh {
        return Default::default();
    }
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persist the memo, best-effort — completion must never fail over a
/// temp-file hiccup.
fn save_probe_memo(known: &crate::datasets::filter::FacetProbeCache) {
    if let Ok(json) = serde_json::to_string(known) {
        let _ = std::fs::write(probe_memo_path(), json);
    }
}

/// The dimensionalities present among the narrowed entries
/// (magnitude-sorted; the probe cache fills in dims the metadata/name
/// can't supply).
fn present_dim_candidates(context: &[&str]) -> Vec<String> {
    with_filtered_entries(context, true, crate::datasets::filter::distinct_dims)
}

/// Numeric-prefix filter: a digit partial keeps candidates that start
/// with it. (Dims are bare integers — no suffix/case folding.)
fn dim_prefix_filter(values: Vec<String>, partial: &str) -> Vec<String> {
    values
        .into_iter()
        .filter(|v| partial.is_empty() || v.starts_with(partial))
        .collect()
}

/// Suggest `--with-dim` (EXACT): dimensionalities present among the
/// narrowed entries, no ladder.
pub fn complete_with_dim(partial: &str, context: &[&str]) -> Vec<String> {
    dim_prefix_filter(present_dim_candidates(context), partial)
}

/// Suggest `--with-min-dim` / `--with-max-dim` (THRESHOLD): present
/// dims plus a ladder of common dimensions, magnitude-sorted, so a
/// boundary like `--with-min-dim 512` is offered even if no dataset
/// is exactly 512.
pub fn complete_with_dim_threshold(partial: &str, context: &[&str]) -> Vec<String> {
    let merged =
        merge_threshold_ladder(present_dim_candidates(context), DIM_LADDER, |s| s.parse().ok());
    dim_prefix_filter(merged, partial)
}

/// Suggest `--select` values: `dataset[:profile]` specs across the
/// already-narrowed entries.
pub fn complete_select_specs(partial: &str, context: &[&str]) -> Vec<String> {
    let pairs = with_filtered_entries(context, false, |entries, _| {
        entries
            .iter()
            .map(|e| {
                (
                    e.name.clone(),
                    e.profile_names().into_iter().map(|s| s.to_string()).collect(),
                )
            })
            .collect::<Vec<(String, Vec<String>)>>()
    });
    filter_spec_candidates(&pairs, partial)
}

/// Suggest configured catalog shortcuts for `--at` by 1-based index.
///
/// Only the index numbers are returned as candidates; when more than
/// one matches, the index → URL legend is printed to stderr (the
/// terminal) so the numbers mean something. Bash invokes the
/// completer twice per tab press (generate + display), so the legend
/// is debounced through a short-lived marker file in the
/// VECTORDATA_HOME-isolated completion scratch dir (see
/// [`completion_scratch_dir`]).
pub fn complete_catalog_urls(partial: &str, _context: &[&str]) -> Vec<String> {
    let configured = sources::raw_catalog_entries(&sources::config_dir());
    let results = filter_catalog_indices(&configured, partial);
    if results.len() > 1 {
        let sanitized: String = partial
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let lock_path = completion_scratch_dir().join(format!("at_legend_{sanitized}"));
        let stale = lock_path
            .metadata()
            .and_then(|m| m.modified())
            .map(|t| t.elapsed().unwrap_or_default().as_secs() > 2)
            .unwrap_or(true);
        if stale {
            let _ = std::fs::write(&lock_path, "");
            eprintln!();
            for (num, url) in &results {
                eprintln!("  {} = {}", num, url);
            }
        }
    }
    results.into_iter().map(|(num, _)| num).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs() -> Vec<(String, Vec<String>)> {
        vec![
            ("alpha".to_string(), vec!["default".to_string(), "d100".to_string()]),
            ("Beta".to_string(), vec!["default".to_string(), "wide".to_string()]),
            ("gamma".to_string(), vec!["narrow".to_string()]),
        ]
    }

    #[test]
    fn threshold_ladder_merges_and_sorts_by_magnitude() {
        // Present values (out of order) + ladder must come out sorted
        // by MAGNITUDE — `2m` before `10m`, never lexical order — and
        // a value present in both is shown once.
        let present = vec!["10m".to_string(), "2m".to_string(), "100k".to_string()];
        let out = merge_threshold_ladder(present, &["1k", "100k", "1m"], |s| {
            crate::datasets::filter::parse_size(s).ok()
        });
        assert_eq!(
            out,
            ["1k", "100k", "1m", "2m", "10m"].map(String::from).to_vec()
        );

        // Bytes: an actual `118MB` size sorts between the `100MB` and
        // `1GB` ladder rungs.
        let out = merge_threshold_ladder(vec!["118MB".to_string()], &["100MB", "1GB"], |s| {
            crate::datasets::filter::parse_bytes(s).ok()
        });
        assert_eq!(out, ["100MB", "118MB", "1GB"].map(String::from).to_vec());

        // Dims sort numerically too (512 between 384 and 768).
        let out = merge_threshold_ladder(vec!["384".to_string(), "768".to_string()], &["512"], |s| {
            s.parse().ok()
        });
        assert_eq!(out, ["384", "512", "768"].map(String::from).to_vec());
    }

    #[test]
    fn dataset_names_filter_case_insensitively() {
        assert_eq!(filter_dataset_names(&pairs(), "be"), vec!["Beta".to_string()]);
        assert_eq!(filter_dataset_names(&pairs(), "").len(), 3);
        assert!(filter_dataset_names(&pairs(), "zz").is_empty());
    }

    #[test]
    fn profile_names_union_without_dataset_scope() {
        assert_eq!(
            filter_profile_names(&pairs(), None, ""),
            vec!["d100".to_string(), "default".to_string(), "narrow".to_string(), "wide".to_string()]
        );
    }

    #[test]
    fn profile_names_scope_to_dataset_case_insensitively() {
        assert_eq!(
            filter_profile_names(&pairs(), Some("beta"), ""),
            vec!["default".to_string(), "wide".to_string()]
        );
        assert_eq!(
            filter_profile_names(&pairs(), Some("alpha"), "d1"),
            vec!["d100".to_string()]
        );
    }

    #[test]
    fn spec_candidates_complete_names_then_profiles() {
        assert_eq!(filter_spec_candidates(&pairs(), "ga"), vec!["gamma".to_string()]);
        assert_eq!(
            filter_spec_candidates(&pairs(), "alpha:"),
            vec!["alpha:default".to_string(), "alpha:d100".to_string()]
        );
        assert_eq!(
            filter_spec_candidates(&pairs(), "beta:w"),
            vec!["Beta:wide".to_string()]
        );
        assert!(filter_spec_candidates(&pairs(), "nope:x").is_empty());
    }

    #[test]
    fn infer_dataset_finds_bare_positional_and_spec_forms() {
        let p = pairs();
        assert_eq!(infer_dataset(&["gamma", "--profile"], &p), Some("gamma".to_string()));
        assert_eq!(infer_dataset(&["BETA:wide"], &p), Some("Beta".to_string()));
        assert_eq!(infer_dataset(&["--at", "2"], &p), None);
        assert_eq!(infer_dataset(&["unknown"], &p), None);
    }

    #[test]
    fn catalog_index_candidates_match_by_number_or_url_prefix() {
        let configured = vec!["https://a/".to_string(), "file:///data/".to_string()];
        let all = filter_catalog_indices(&configured, "");
        assert_eq!(all.len(), 2);
        assert_eq!(filter_catalog_indices(&configured, "2"),
            vec![("2".to_string(), "file:///data/".to_string())]);
        assert_eq!(filter_catalog_indices(&configured, "file"),
            vec![("2".to_string(), "file:///data/".to_string())]);
    }

    #[test]
    fn context_extraction_reads_engine_words() {
        let ctx = ["--at", "2", "--dataset", "alpha", "--at", "https://c/"];
        assert_eq!(option_value(&ctx, "--dataset"), Some("alpha".to_string()));
        assert_eq!(option_values(&ctx, "--at"),
            vec!["2".to_string(), "https://c/".to_string()]);
        assert_eq!(option_value(&ctx, "--profile"), None);
    }

    #[test]
    fn resolver_map_covers_the_datasets_surface() {
        let keys: Vec<String> = datasets_resolvers().into_keys().collect();
        assert_eq!(keys, vec![
            "--at".to_string(),
            "--dataset".to_string(),
            "--matching-name".to_string(),
            "--matching-profile".to_string(),
            "--profile".to_string(),
            "--select".to_string(),
            "--with-count".to_string(),
            "--with-data".to_string(),
            "--with-dim".to_string(),
            "--with-facet".to_string(),
            "--with-max-count".to_string(),
            "--with-max-data".to_string(),
            "--with-max-dim".to_string(),
            "--with-metric".to_string(),
            "--with-min-count".to_string(),
            "--with-min-data".to_string(),
            "--with-min-dim".to_string(),
            "--with-vtype".to_string(),
            "datasets describe".to_string(),
            "datasets ping".to_string(),
            "datasets precache".to_string(),
        ]);
    }
}
