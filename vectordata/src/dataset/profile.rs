// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dataset profiles — named configurations of data views with inheritance.
//!
//! A profile groups data views (keyed by canonical facet names) together with
//! optional metadata like `maxk`. Profiles support inheritance: when a
//! `"default"` profile exists, all other profiles automatically inherit its
//! views and `maxk` unless explicitly overridden.
//!
//! ## Reserved profile names
//!
//! Two profile names are reserved:
//!
//! - **`default`** — The baseline profile. All other profiles inherit its
//!   `maxk` and views unless explicitly overridden.
//! - **`sized`** — Not a profile itself, but a list of size specifications
//!   that expand into named profiles with `base_count` set. See below.
//!
//! All other profile names are available for custom profiles.
//!
//! ## Custom facets
//!
//! Profile view keys that do not match a recognized standard facet are
//! preserved as-is (custom facets). They are not rejected during
//! deserialization and participate identically to standard facets.
//!
//! ## YAML layout
//!
//! ```yaml
//! profiles:
//!   default:
//!     maxk: 100
//!     base: base_vectors.fvec          # sugar: bare string, alias "base"
//!     query: query_vectors.fvec        # sugar: alias "query"
//!     indices: gnd/idx.ivecs           # sugar: alias "indices"
//!     distances: gnd/dis.fvecs
//!     model_profile: model.json        # custom facet
//!   1M:
//!     base:
//!       source: base_vectors.fvec      # map form
//!       window: [0..1000000]
//!     indices: gnd/idx_1M.ivecs
//!     distances: gnd/dis_1M.fvecs
//! ```
//!
//! ## Sized profile sugar
//!
//! Instead of declaring each sized profile individually, use the `sized` key
//! with a list of size specifications:
//!
//! ```yaml
//! profiles:
//!   default:
//!     maxk: 100
//!     query_vectors: query_vectors.mvec
//!   sized: [10m, 20m, 100m..400m/100m]
//! ```
//!
//! This expands to profiles named `10m`, `20m`, `100m`, `200m`, `300m`, `400m`,
//! each with `base_count` set and inheriting from `default`. Profiles are sorted
//! smallest to largest.
//!
//! ### Sized entry forms
//!
//! - **Simple value**: `10m` — one profile with base_count 10,000,000
//! - **Linear range (step)**: `100m..400m/100m` — profiles at each absolute step
//! - **Linear range (count)**: `0m..400m/10` — 10 equal divisions (bare number = count)
//! - **Fibonacci**: `fib:1m..400m` — fibonacci multiples of start within range
//! - **Geometric**: `mul:1m..400m/2` — compound by factor (doubling, tripling, etc.)
//! - **Decade**: `decade:100k` (or bare `decade`) — engineering-decade
//!   sweep: 100k, 200k, …, 900k, 1m, 2m, …, 9m, 10m, … step rescales
//!   each order of magnitude. Always emits decimal (M) units.

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde::de::{self, Visitor};

use super::facet::resolve_standard_key;
use super::source::{DSInterval, DSSource, DSWindow, format_count_with_suffix, parse_number_with_suffix, parse_source_string};

/// A view of a data facet within a profile.
///
/// Wraps a `DSSource` with an optional view-level window override.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DSView {
    /// The data source for this view.
    pub source: DSSource,
    /// Optional view-level window override (takes precedence over source window).
    pub window: Option<DSWindow>,
}

impl Serialize for DSView {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as bare string when possible (simple path, no namespace, no windows)
        if self.window.is_none()
            && self.source.namespace.is_none()
            && self.source.window.is_empty()
        {
            serializer.serialize_str(&self.source.path)
        } else {
            use serde::ser::SerializeMap;
            let mut map = serializer.serialize_map(None)?;
            map.serialize_entry("source", &self.source.path)?;
            if let Some(ref ns) = self.source.namespace {
                map.serialize_entry("namespace", ns)?;
            }
            if !self.source.window.is_empty() {
                map.serialize_entry("window", &self.source.window)?;
            }
            if let Some(ref w) = self.window {
                map.serialize_entry("window", w)?;
            }
            map.end()
        }
    }
}

impl DSView {
    /// Returns the effective window: view-level override if present, else source window.
    pub fn effective_window(&self) -> &DSWindow {
        self.window.as_ref().unwrap_or(&self.source.window)
    }

    /// Returns the source path.
    pub fn path(&self) -> &str {
        &self.source.path
    }
}

/// A named profile: map of canonical view names to views, plus optional maxk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DSProfile {
    /// Maximum k for KNN queries in this profile.
    pub maxk: Option<u32>,
    /// Base vector count for sized subset profiles. When set, `per_profile`
    /// pipeline steps are expanded for this profile with `${base_count}` and
    /// `${base_end}` variables.
    pub base_count: Option<u64>,
    /// When true, this is an oracle partition profile with independent
    /// base vectors (not a windowed subset of the default profile).
    /// Partition profiles are created by `compute partition-profiles`
    /// and have their own KNN computation + verification steps.
    /// They are excluded from consolidated verification and do not
    /// inherit metadata/filtered-KNN views from the default profile.
    pub partition: bool,
    /// Views keyed by canonical facet name (standard facets) or custom name.
    pub views: IndexMap<String, DSView>,
}

impl Serialize for DSProfile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut len = self.views.len();
        if self.maxk.is_some() { len += 1; }
        if self.base_count.is_some() { len += 1; }
        // partition is NOT serialized inside the profile body — it's
        // listed in the top-level partition_profiles key instead, so
        // external clients don't see unknown keys among facet entries.
        let mut map = serializer.serialize_map(Some(len))?;
        if let Some(maxk) = self.maxk {
            map.serialize_entry("maxk", &maxk)?;
        }
        if let Some(base_count) = self.base_count {
            map.serialize_entry("base_count", &base_count)?;
        }
        for (key, view) in &self.views {
            map.serialize_entry(key, view)?;
        }
        map.end()
    }
}

impl DSProfile {
    /// Get a view by canonical facet name or custom name.
    pub fn view(&self, name: &str) -> Option<&DSView> {
        self.views.get(name)
    }

    /// Iterate over all (name, view) pairs in this profile.
    pub fn views(&self) -> impl Iterator<Item = (&str, &DSView)> {
        self.views.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns the view names in this profile.
    pub fn view_names(&self) -> Vec<&str> {
        self.views.keys().map(|k| k.as_str()).collect()
    }
}

/// Collection of named profiles with default-inheritance.
///
/// When deserialized, profiles named other than `"default"` automatically
/// inherit views and `maxk` from the default profile.
///
/// Sized entries containing `${variable}` references are stored in
/// `deferred_sized` and expanded later when variables become available
/// (after core pipeline stages produce actual counts).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DSProfileGroup {
    /// Resolved profiles keyed by name.
    pub profiles: IndexMap<String, DSProfile>,
    /// Raw sized entries that contain `${...}` variable references.
    /// These are expanded by `expand_deferred_sized()` once variables
    /// are available.
    pub deferred_sized: Vec<String>,
    /// Facet templates from the `sized:` structured form, stored for
    /// deferred expansion.
    pub deferred_facet_templates: Option<IndexMap<String, String>>,
    /// All raw sized entries from the original YAML, preserved for
    /// round-trip serialization. Includes both immediately expanded
    /// entries and deferred entries.
    pub raw_sized: Vec<String>,
    /// Profile names that were generated from `sized:` expansion (not
    /// explicitly defined). These are omitted from `save()` output
    /// since they'll be re-generated from the `sized:` entries on reload.
    pub sized_profile_names: Vec<String>,
}

impl Serialize for DSProfileGroup {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.profiles.serialize(serializer)
    }
}

impl DSProfileGroup {
    /// Create a profile group from a map of profiles with no deferred entries.
    pub fn from_profiles(profiles: IndexMap<String, DSProfile>) -> Self {
        DSProfileGroup {
            profiles,
            deferred_sized: Vec::new(),
            deferred_facet_templates: None,
            raw_sized: Vec::new(),
            sized_profile_names: Vec::new(),
        }
    }

    /// Returns the default profile, if one exists.
    pub fn default_profile(&self) -> Option<&DSProfile> {
        self.profiles.get("default")
    }

    /// Look up a profile by name.
    pub fn profile(&self, name: &str) -> Option<&DSProfile> {
        self.profiles.get(name)
    }

    /// List all profile names.
    /// Profile names sorted by size ascending, with `default` first.
    pub fn profile_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.profiles.keys().map(|k| k.as_str()).collect();
        names.sort_by(|a, b| profile_sort_by_size(
            a, self.profiles.get(*a).and_then(|p| p.base_count),
            b, self.profiles.get(*b).and_then(|p| p.base_count),
        ));
        names
    }

    /// Returns view names from the default profile (for display).
    pub fn view_names(&self) -> Vec<&str> {
        self.default_profile()
            .map(|p| p.view_names())
            .unwrap_or_default()
    }

    /// Returns `true` if there are no profiles.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Returns `true` if there are deferred sized entries awaiting expansion.
    pub fn has_deferred(&self) -> bool {
        !self.deferred_sized.is_empty()
    }

    /// Expand deferred sized entries using the provided variable map.
    ///
    /// Expand deferred sized profile entries into concrete profiles.
    ///
    /// `base_count` is the actual number of clean base vectors in the
    /// dataset. This is **required** — profile expansion cannot produce
    /// valid profiles without knowing the dataset size. Every generator
    /// stops producing candidates as soon as a value would equal or exceed
    /// `base_count` (such profiles are redundant with the default profile).
    ///
    /// Called by the pipeline runner after core stages have produced actual
    /// counts (e.g., `base_count` in `variables.yaml`). Entries that still
    /// contain unresolved variables after interpolation are skipped with a
    /// warning.
    ///
    /// Returns the number of new profiles added.
    pub fn expand_deferred_sized(&mut self, vars: &IndexMap<String, String>, base_count: u64) -> usize {
        if self.deferred_sized.is_empty() {
            return 0;
        }

        let default_profile = self.profiles.get("default").cloned();
        let templates = self.deferred_facet_templates.take();
        let entries = std::mem::take(&mut self.deferred_sized);
        let mut added = 0;
        // Collect (count, name, profile) across all strategies first so we
        // can insert them in ascending-size order. Otherwise the resulting
        // profile list is grouped by strategy ("all of mul:, then fib:,
        // then decade:") which interleaves sizes — e.g. 100k after 8mi —
        // and that's what the user sees in `veks explore` and YAML.
        let mut pending: Vec<(u64, String, DSProfile)> = Vec::new();

        for entry_str in &entries {
            // Interpolate variables
            let mut resolved = entry_str.clone();
            for (key, val) in vars {
                resolved = resolved.replace(&format!("${{{}}}", key), val);
            }

            // Check if any variables remain unresolved
            if resolved.contains("${") {
                log::warn!(
                    "sized entry '{}' still has unresolved variables after interpolation: '{}'",
                    entry_str, resolved
                );
                // Re-defer for next attempt
                self.deferred_sized.push(entry_str.clone());
                continue;
            }

            // Parse the resolved entry. Every generator stops at base_count.
            let pairs = match parse_sized_entry_impl(&resolved, base_count) {
                Ok(p) => p,
                Err(e) => {
                    log::warn!("failed to parse sized entry '{}' (resolved: '{}'): {}", entry_str, resolved, e);
                    continue;
                }
            };

            for (prof_name, count) in pairs {
                let scaffold_views = if let Some(ref tmpl) = templates {
                    let range_spec = format!("[0..{}]", count);
                    let mut views = IndexMap::new();
                    for (facet_key, template) in tmpl {
                        let interpolated = template
                            .replace("${profile}", &prof_name)
                            .replace("${range}", &range_spec);
                        let canonical = resolve_standard_key(facet_key)
                            .unwrap_or_else(|| facet_key.clone());
                        if let Ok(source) = parse_source_string(&interpolated) {
                            views.insert(canonical, DSView { source, window: None });
                        }
                    }
                    Some(views)
                } else {
                    None
                };

                let base_views = scaffold_views.unwrap_or_default();

                let merged = if let Some(ref dp) = default_profile {
                    let mut merged_views: IndexMap<String, DSView> = dp.views.iter()
                        .filter(|(k, _)| !is_per_profile_output_view(k))
                        .map(|(k, v)| {
                            // For per-vector shared facets (base_vectors,
                            // metadata_content), the sized profile is
                            // genuinely the first `count` rows of the
                            // shared file — record that as a windowed
                            // reference into the inherited path. This is
                            // the documented sub-ordinal suffix model:
                            // the source path stays the same as default,
                            // but `source.window` carries `[0..count)` so
                            // the `vectordata` reader API can clip
                            // accesses without the consumer having to
                            // remember `base_count`.
                            //
                            // Other shared facets (query_vectors, etc.)
                            // are NOT windowed — query sets are the
                            // same across profiles.
                            if is_windowed_shared_view(k) && v.source.window.is_empty() {
                                let mut windowed = v.clone();
                                windowed.source.window = DSWindow(vec![DSInterval {
                                    min_incl: 0,
                                    max_excl: count,
                                }]);
                                (k.clone(), windowed)
                            } else {
                                (k.clone(), v.clone())
                            }
                        })
                        .collect();
                    for (k, v) in base_views {
                        merged_views.insert(k, v);
                    }
                    DSProfile {
                        maxk: dp.maxk,
                        base_count: Some(count),
                        partition: false,
                        views: merged_views,
                    }
                } else {
                    DSProfile {
                        maxk: None,
                        base_count: Some(count),
                        partition: false,
                        views: base_views,
                    }
                };
                pending.push((count, prof_name, merged));
            }
        }

        // Sort ascending by base_count, then by name as a tiebreaker for
        // stability when two strategies emit the same size. Insert in
        // sorted order; later inserts of the same name overwrite earlier
        // ones (preserving prior duplicate-handling semantics — the
        // larger-named or alphabetically-later one wins, but since they
        // have the same count this is moot for display).
        pending.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (_count, prof_name, merged) in pending {
            self.profiles.insert(prof_name, merged);
            added += 1;
        }

        added
    }

    /// Derive views for sized profiles from per-profile template step outputs.
    ///
    /// For each `per_profile` template step, collects the output filename and
    /// creates views in every profile:
    ///
    /// - **default**: a direct view pointing to `profiles/default/{filename}`
    /// - **sized profiles** (with `base_count`): a windowed view referencing
    ///   `profiles/default/{filename}` with window `[0, base_count)`
    ///
    /// Explicitly declared views are never overridden. This mirrors the
    /// expansion rules used by the upstream pipeline so that consumers of
    /// the dataset crate see the same profile structure without needing the
    /// pipeline runtime.
    pub fn derive_views_from_templates(&mut self, templates: &[crate::dataset::pipeline::StepDef]) {
        use super::source::{DSSource, DSWindow};

        // Collect bare output filenames from per_profile templates,
        // paired with the template's command name for facet-scope filtering.
        // Skip cache outputs and variable-interpolated names — these are
        // intermediate artifacts, not dataset views.
        let template_outputs: Vec<(String, String)> = templates
            .iter()
            .filter(|s| s.per_profile)
            .flat_map(|s| {
                let cmd = s.run.clone();
                s.output_paths().into_iter().map(move |p| (p, cmd.clone()))
            })
            .filter(|(p, _)| !p.contains("${cache}") && !p.contains("${profile_name}"))
            .map(|(p, cmd)| {
                let cleaned = p.strip_prefix("${profile_dir}")
                    .unwrap_or(&p)
                    .to_string();
                (cleaned, cmd)
            })
            .collect();

        if template_outputs.is_empty() {
            return;
        }

        for (name, profile) in self.profiles.iter_mut() {
            // Skip profiles without base_count (except default)
            if name != "default" && profile.base_count.is_none() {
                continue;
            }

            let profile_dir = format!("profiles/{}/", name);

            for (filename, _cmd) in &template_outputs {
                // Derive the view key from the filename stem
                let path = std::path::Path::new(filename);
                let stem = match path.file_stem().and_then(|s| s.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };

                // Don't override explicitly declared views
                if profile.views.contains_key(&stem) {
                    continue;
                }

                // Non-default profiles with a `base_count` (sized or
                // partition) always reference their own profile dir for
                // per-profile outputs. `bc` is consulted only to gate
                // the branch — the exact count doesn't drive path
                // construction any more.
                if name != "default" {
                    if profile.base_count.is_some() {
                        if profile.partition {
                            // Partition profiles only get views for facets
                            // in their scope (default BQG). Skip metadata,
                            // filtered KNN, and predicate views.
                            let is_knn_view = stem == "neighbor_indices"
                                || stem == "neighbor_distances";
                            if !is_knn_view {
                                continue;
                            }

                            // Partition profile: KNN outputs are in the
                            // profile's own directory
                            let resolved_path = format!("{}{}", profile_dir, filename);
                            profile.views.insert(
                                stem,
                                DSView {
                                    source: DSSource {
                                        path: resolved_path,
                                        namespace: None,
                                        window: DSWindow::default(),
                                    },
                                    window: None,
                                },
                            );
                        } else {
                            // Sized subset profile: outputs live in the
                            // profile's own directory. The earlier
                            // "window into default's file" model was
                            // incorrect — you can't derive a sized-range
                            // top-K by windowing a full-range top-K (the
                            // top-N nearest neighbours of the full base
                            // generally aren't inside the [0..N) prefix).
                            // compute-knn actually writes per-profile
                            // files at `profiles/{name}/{filename}`, and
                            // verify-knn reads them from the same place.
                            let resolved_path = format!("{}{}", profile_dir, filename);
                            profile.views.insert(
                                stem,
                                DSView {
                                    source: DSSource {
                                        path: resolved_path,
                                        namespace: None,
                                        window: DSWindow::default(),
                                    },
                                    window: None,
                                },
                            );
                        }
                        continue;
                    }
                }

                // Default profile or profiles without base_count: use direct path
                let resolved_path = format!("{}{}", profile_dir, filename);

                profile.views.insert(
                    stem,
                    DSView {
                        source: DSSource {
                            path: resolved_path,
                            namespace: None,
                            window: Default::default(),
                        },
                        window: None,
                    },
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Serde: DSView
// ---------------------------------------------------------------------------

impl<'de> Deserialize<'de> for DSView {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct ViewVisitor;

        impl<'de> Visitor<'de> for ViewVisitor {
            type Value = DSView;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a source string or map with 'source' + optional 'window'")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<DSView, E> {
                let source = parse_source_string(v).map_err(de::Error::custom)?;
                Ok(DSView {
                    source,
                    window: None,
                })
            }

            fn visit_map<M>(self, mut map: M) -> Result<DSView, M::Error>
            where
                M: de::MapAccess<'de>,
            {
                let mut window: Option<DSWindow> = None;

                let mut source_path: Option<String> = None;
                let mut source_namespace: Option<String> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "source" | "path" => {
                            source_path = Some(map.next_value()?);
                        }
                        "namespace" | "ns" => {
                            source_namespace = Some(map.next_value()?);
                        }
                        "window" => {
                            window = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                let source = source_path
                    .map(|path| DSSource {
                        path,
                        namespace: source_namespace,
                        window: Default::default(),
                    })
                    .ok_or_else(|| {
                        de::Error::custom("view map must have 'source' or 'path' key")
                    })?;

                Ok(DSView { source, window })
            }
        }

        deserializer.deserialize_any(ViewVisitor)
    }
}

// ---------------------------------------------------------------------------
// Serde: DSProfile
// ---------------------------------------------------------------------------

impl<'de> Deserialize<'de> for DSProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        // Deserialize as a raw map first, then extract maxk and normalize view keys
        let raw: IndexMap<String, serde_yaml::Value> = IndexMap::deserialize(deserializer)?;

        let mut maxk: Option<u32> = None;
        let mut base_count: Option<u64> = None;
        let mut partition = false;
        let mut views = IndexMap::new();

        for (key, value) in raw {
            if key == "maxk" {
                maxk = match &value {
                    serde_yaml::Value::Number(n) => n.as_u64().map(|v| v as u32),
                    serde_yaml::Value::String(s) => s.parse().ok(),
                    _ => None,
                };
                continue;
            }
            if key == "base_count" {
                base_count = match &value {
                    serde_yaml::Value::Number(n) => n.as_u64(),
                    serde_yaml::Value::String(s) => super::source::parse_number_with_suffix(s).ok(),
                    _ => None,
                };
                continue;
            }
            if key == "partition" {
                partition = match &value {
                    serde_yaml::Value::Bool(b) => *b,
                    serde_yaml::Value::String(s) => s == "true",
                    _ => false,
                };
                continue;
            }

            // Resolve alias to canonical name; custom facets pass through as-is
            let canonical = resolve_standard_key(&key).unwrap_or_else(|| key.clone());

            let view: DSView = serde_yaml::from_value(value).map_err(de::Error::custom)?;
            views.insert(canonical, view);
        }

        Ok(DSProfile { maxk, base_count, partition, views })
    }
}

// ---------------------------------------------------------------------------
// Sized profile expansion
// ---------------------------------------------------------------------------

/// Parse a sized entry into `(name, base_count)` pairs.
///
/// Accepts these forms:
/// - Simple value: `"10m"` → one pair `("10m", 10_000_000)`
/// - Linear range (step): `"100m..400m/100m"` → profiles at each step from start to end
/// - Linear range (count): `"0m..400m/10"` → 10 equal divisions (suffix-less divisor = count)
/// - Fibonacci: `"fib:1m..400m"` → fibonacci-progression values within [start, end]
/// - Geometric: `"mul:1m..400m/2"` → compound by factor: start, start×2, start×4, ... (factor can be fractional)
///
/// The profile name is generated from the count using `format_count_with_suffix`.
/// Parse a sized entry into `(name, base_count)` pairs.
///
/// Supported forms:
/// - `"10m"` → single profile with 10M vectors
/// - `"100m..400m/100m"` → range with step
/// - `"mul:1m..400m/2"` → multiplicative range
/// - `"fib:1m..400m"` → fibonacci range
///
/// `max_count` caps unbounded series (e.g., `mul:1mi/2` without an explicit
/// end). When provided, no profile will have a `base_count` exceeding this
/// value. This is typically the default profile's base_count (the actual
/// dataset size). Pass `None` at parse time when the dataset size is unknown.
/// View keys whose value is logically the first `base_count` rows of
/// a shared per-vector file. When inherited from the default profile,
/// these views must carry a `[0..base_count)` window so the
/// `vectordata` reader API clamps reads automatically.
///
/// Limited to facets that are physically *one row per base vector*:
///   - `base_vectors` — the actual vectors.
///   - `metadata_content` — per-vector metadata records.
///
/// `query_vectors` is NOT in this list — query sets are the same
/// across all profiles, not a prefix of anything. Per-profile output
/// views (`neighbor_indices`, etc.) are also not in this list — they
/// have their own windowing handled separately.
fn is_windowed_shared_view(key: &str) -> bool {
    matches!(key, "base_vectors" | "metadata_content")
}

/// View keys whose value is per-profile and must NOT be inherited from
/// the default profile when expanding sized profiles.
///
/// A sized profile (`1m`, `40m`, ...) computes its own KNN against a
/// subset of the base, writing to `profiles/{name}/neighbor_indices.ivecs`.
/// If we let it inherit `neighbor_indices: profiles/default/...` from the
/// default profile, every consumer (verify, benchmarks, external tools)
/// would silently load default's full-base GT instead of the profile's
/// own — and a verify against default's GT for the 1m boundary's
/// cumulative top-100 fails 100% of queries by construction.
///
/// Filtered KNN outputs follow the same per-profile pattern.
fn is_per_profile_output_view(key: &str) -> bool {
    matches!(key,
        "neighbor_indices" | "neighbor_distances"
        | "filtered_neighbor_indices" | "filtered_neighbor_distances"
    )
}

/// Compute a numeric sort key for a profile.
///
/// Uses `base_count` if available, otherwise parses the profile name as a
/// number with SI/IEC suffix (e.g., `10m` → 10,000,000, `1gi` → 1,073,741,824).
/// The `default` profile always sorts to the front.
///
/// This is the single source of truth for profile ordering across the
/// entire codebase. All profile list views should use this function.
pub fn profile_sort_key(name: &str, base_count: Option<u64>) -> u64 {
    if name == "default" {
        return 0; // always first
    }
    base_count
        .or_else(|| crate::dataset::source::parse_number_with_suffix(name).ok())
        .unwrap_or(u64::MAX - 1)
}

/// Compare two profiles by size for sorting.
///
/// `default` first, then ascending by `base_count` (or name-derived size).
/// Tiebreak by name for profiles with identical sizes.
pub fn profile_sort_by_size(
    a_name: &str, a_base_count: Option<u64>,
    b_name: &str, b_base_count: Option<u64>,
) -> std::cmp::Ordering {
    let a_key = profile_sort_key(a_name, a_base_count);
    let b_key = profile_sort_key(b_name, b_base_count);
    a_key.cmp(&b_key).then_with(|| natural_cmp(a_name, b_name))
}

/// Natural comparison: splits strings into alphabetic and numeric segments,
/// comparing numbers by value (so "label_2" < "label_10") and text
/// lexicographically.
pub fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();

    loop {
        match (ai.peek(), bi.peek()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(&ac), Some(&bc)) => {
                if ac.is_ascii_digit() && bc.is_ascii_digit() {
                    // Both at a digit run — extract full number and compare by value
                    let an = take_number(&mut ai);
                    let bn = take_number(&mut bi);
                    match an.cmp(&bn) {
                        std::cmp::Ordering::Equal => continue,
                        other => return other,
                    }
                } else {
                    match ac.cmp(&bc) {
                        std::cmp::Ordering::Equal => {
                            ai.next();
                            bi.next();
                        }
                        other => return other,
                    }
                }
            }
        }
    }
}

/// Consume consecutive digits from a peekable char iterator, returning
/// the numeric value.
fn take_number(iter: &mut std::iter::Peekable<std::str::Chars<'_>>) -> u64 {
    let mut n: u64 = 0;
    while let Some(&c) = iter.peek() {
        if c.is_ascii_digit() {
            n = n.saturating_mul(10).saturating_add((c as u64) - ('0' as u64));
            iter.next();
        } else {
            break;
        }
    }
    n
}

/// Parse a sized entry without a base count bound. Only valid for forms
/// with explicit ranges (simple values, linear ranges with explicit ends).
/// Unbounded series (mul:start/factor, fib: without explicit end) will
/// use a 0 cap and produce no results.
/// Returns true if a sized entry uses an implicit upper bound form that
/// requires `base_count` to be known before expansion. These entries
/// must be deferred during YAML deserialization.
///
/// Three open-ended forms exist:
///   - `mul:start/factor`        — no `..end`
///   - `fib:start`               — no `..end` (just the start scalar)
///   - `linear:start/step`       — no `..end` in the range portion
/// All three are resolved at runtime once the base count is materialized.
pub fn needs_base_count(entry: &str) -> bool {
    let entry = entry.trim();
    if let Some(rest) = entry.strip_prefix("mul:") {
        // mul:start/factor (no ..) needs base_count
        // mul:start..end/factor has explicit end, does not need it
        return !rest.contains("..");
    }
    if let Some(rest) = entry.strip_prefix("fib:") {
        // fib:start (no ..) needs base_count
        // fib:start..end has explicit end, does not need it
        return !rest.contains("..");
    }
    if let Some(rest) = entry.strip_prefix("linear:") {
        // linear:start/step (no ..) needs base_count
        // linear:start..end/step has explicit end, does not need it
        // The `..` only appears in the range part (left of `/`).
        let range_part = rest.split_once('/').map(|(r, _)| r).unwrap_or(rest);
        return !range_part.contains("..");
    }
    if entry == "decade" || entry.starts_with("decade:") {
        // `decade` (no body) and `decade:start` use the implicit
        // upper bound; `decade:start..end` has an explicit end.
        let body = entry.strip_prefix("decade:").unwrap_or("");
        return !body.contains("..");
    }
    false
}

/// Parse a sized entry without a base count bound. Only valid for forms
/// with explicit ranges. Implicit upper bound forms (like `mul:start/factor`)
/// will return an error — use `parse_sized_entry_impl` with a known
/// `max_count` instead, or defer the entry.
pub fn parse_sized_entry(entry: &str) -> Result<Vec<(String, u64)>, String> {
    parse_sized_entry_impl(entry, 0)
}

/// Parse a sized entry, producing only profiles with `count < max_count`.
///
/// `max_count` is the actual base vector count of the dataset. Every
/// generator checks each candidate against this bound before emitting it.
/// A candidate that equals or exceeds `max_count` is not emitted (it
/// would be redundant with the default profile).
fn parse_sized_entry_impl(entry: &str, max_count: u64) -> Result<Vec<(String, u64)>, String> {
    let entry = entry.trim();

    // Check for series generator prefixes
    if let Some(rest) = entry.strip_prefix("fib:") {
        return parse_fib_range(rest, max_count);
    }
    if let Some(rest) = entry.strip_prefix("mul:") {
        return parse_mul_range(rest, max_count);
    }
    if let Some(rest) = entry.strip_prefix("linear:") {
        return parse_linear_range(rest, max_count);
    }
    if entry == "decade" {
        return parse_decade_range("", max_count);
    }
    if let Some(rest) = entry.strip_prefix("decade:") {
        return parse_decade_range(rest, max_count);
    }

    // Explicit-prefix range forms:
    //   step:<lo>..<hi>/<step>  — emit lo, lo+step, lo+2*step, …, ≤ hi
    //   parts:<lo>..<hi>/<n>    — divide [lo, hi] into n equal segments
    // These are the recommended spelling — they make the generator
    // intent unambiguous. The bare `<lo>..<hi>/<x>` form (where the
    // divisor's alpha suffix decides between step- and parts-mode) is
    // still accepted for backward compatibility.
    if let Some(rest) = entry.strip_prefix("step:") {
        return parse_bounded_range(rest, max_count, RangeMode::Step);
    }
    if let Some(rest) = entry.strip_prefix("parts:") {
        return parse_bounded_range(rest, max_count, RangeMode::Parts);
    }

    if let Some((_, divisor_str)) = entry.split_once('/') {
        // Bare range: infer step vs parts from the divisor's suffix.
        // (Legacy form — prefer `step:` / `parts:` in new entries.)
        let mode = if divisor_str.trim().bytes().any(|b| b.is_ascii_alphabetic()) {
            RangeMode::Step
        } else {
            RangeMode::Parts
        };
        return parse_bounded_range(entry, max_count, mode);
    }

    // Simple value
    let count = parse_number_with_suffix(entry)?;
    if max_count > 0 && count >= max_count {
        return Ok(Vec::new());
    }
    Ok(vec![(format_count_with_suffix(count), count)])
}

#[derive(Clone, Copy)]
enum RangeMode {
    Step,
    Parts,
}

fn parse_bounded_range(
    body: &str,
    max_count: u64,
    mode: RangeMode,
) -> Result<Vec<(String, u64)>, String> {
    let (range_part, divisor_str) = body
        .split_once('/')
        .ok_or_else(|| format!("invalid range '{}': expected start..end/divisor", body))?;
    let (start_str, end_str) = range_part
        .split_once("..")
        .ok_or_else(|| format!("invalid range '{}': expected start..end/divisor", body))?;
    let start = parse_number_with_suffix(start_str.trim())?;
    let end_raw = parse_number_with_suffix(end_str.trim())?;
    let end = if max_count > 0 { end_raw.min(max_count - 1) } else { end_raw };
    if start > end {
        return Ok(Vec::new());
    }

    match mode {
        RangeMode::Step => {
            let step = parse_number_with_suffix(divisor_str.trim())?;
            if step == 0 {
                return Err(format!("step must be > 0 in '{}'", body));
            }
            let mut result = Vec::new();
            let mut v = start;
            while v <= end {
                if v > 0 && (max_count == 0 || v < max_count) {
                    result.push((format_count_with_suffix(v), v));
                }
                v += step;
            }
            Ok(result)
        }
        RangeMode::Parts => {
            let count: u64 = divisor_str.trim().parse()
                .map_err(|e| format!("invalid parts count '{}': {}", divisor_str.trim(), e))?;
            if count == 0 {
                return Err(format!("parts count must be > 0 in '{}'", body));
            }
            let range = end - start;
            let mut result = Vec::new();
            for i in 1..=count {
                let val = start + (range * i) / count;
                if val > 0 && (max_count == 0 || val < max_count) {
                    result.push((format_count_with_suffix(val), val));
                }
            }
            result.dedup_by(|a, b| a.1 == b.1);
            Ok(result)
        }
    }
}

/// Generate fibonacci-progression values within [start, end], capped at
/// `max_count`. No value `>= max_count` is emitted.
///
/// `fib:1m..400m` produces all fibonacci multiples of the base unit that
/// fall within the range. The base unit is the GCD-friendly smallest value;
/// the series starts at 1×base and grows by fibonacci steps.
fn parse_fib_range(range_str: &str, max_count: u64) -> Result<Vec<(String, u64)>, String> {
    // Accept both `fib:start..end` (explicit upper bound) and `fib:start`
    // (open-ended, capped at runtime by `max_count`). The open form is
    // what the wizard emits so the spec stored in dataset.yaml stays
    // independent of the dataset's exact size.
    let (start, explicit_end) = if let Some((start_str, end_str)) = range_str.split_once("..") {
        (
            parse_number_with_suffix(start_str.trim())?,
            Some(parse_number_with_suffix(end_str.trim())?),
        )
    } else {
        (parse_number_with_suffix(range_str.trim())?, None)
    };
    let end = match (explicit_end, max_count) {
        (Some(e), mc) if mc > 0 => e.min(mc - 1),
        (Some(e), _) => e,
        (None, mc) if mc > 0 => mc - 1,
        (None, _) => return Err(format!(
            "implicit upper bound form 'fib:{}' requires a known base count", range_str)),
    };
    if start == 0 {
        return Err(format!("fib range requires start > 0, got {}..{}", start, end));
    }
    if start > end {
        return Ok(Vec::new());
    }

    // Use start as the base unit; generate fib(n) * start within range
    let mut result = Vec::new();
    let (mut a, mut b): (u64, u64) = (1, 1);
    loop {
        let val = a.checked_mul(start).unwrap_or(u64::MAX);
        if val > end {
            break;
        }
        if val >= start && (max_count == 0 || val < max_count) {
            result.push((format_count_with_suffix(val), val));
        }
        let next = a.saturating_add(b);
        a = b;
        b = next;
    }
    Ok(result)
}

/// Generate geometric (compound-by-factor) values within [start, end],
/// capped at `max_count`. No value `>= max_count` is emitted.
///
/// `mul:1m..400m/2` produces: 1m, 2m, 4m, 8m, 16m, 32m, 64m, 128m, 256m
/// `mul:1m..100m/1.5` produces: 1m, 1500k (rounded), etc.
///
/// The factor must be > 1.0. Each successive value is `floor(prev × factor)`.
fn parse_mul_range(spec: &str, max_count: u64) -> Result<Vec<(String, u64)>, String> {
    let (range_part, factor_str) = spec
        .split_once('/')
        .ok_or_else(|| format!("invalid mul spec '{}': expected mul:start/factor or mul:start..end/factor", spec))?;
    // Support both mul:start..end/factor and mul:start/factor (no end)
    let (start, explicit_end) = if let Some((start_str, end_str)) = range_part.split_once("..") {
        (parse_number_with_suffix(start_str.trim())?, Some(parse_number_with_suffix(end_str.trim())?))
    } else {
        (parse_number_with_suffix(range_part.trim())?, None)
    };
    // The effective end is the minimum of the explicit end (if any) and
    // max_count - 1. Profiles at or above max_count are redundant with
    // the default profile and must never be generated.
    let end = match (explicit_end, max_count) {
        (Some(e), mc) if mc > 0 => e.min(mc - 1),
        (Some(e), _) => e,
        (None, mc) if mc > 0 => mc - 1,
        (None, _) => return Err(format!(
            "implicit upper bound form 'mul:{}' requires a known base count", range_part)),
    };
    let factor: f64 = factor_str.trim().parse()
        .map_err(|e| format!("invalid mul factor '{}': {}", factor_str, e))?;
    if factor <= 1.0 {
        return Err(format!("mul factor must be > 1.0, got {}", factor));
    }
    if start == 0 {
        return Err(format!("mul range requires start > 0, got {}..{}", start, end));
    }
    if start > end {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let mut val_f = start as f64;
    loop {
        let val = val_f as u64;
        // Stop as soon as a candidate reaches or exceeds max_count
        if val > end || (max_count > 0 && val >= max_count) || result.len() >= 100 {
            break;
        }
        // Deduplicate: skip if same as previous
        if result.last().map_or(true, |&(_, prev): &(String, u64)| prev != val) {
            result.push((format_count_with_suffix(val), val));
        }
        val_f *= factor;
        // Guard against stalling (factor near 1.0 with small values)
        if val_f as u64 == val {
            val_f = (val + 1) as f64;
        }
    }
    Ok(result)
}

/// Generate a linear (additive-step) sequence within `[start, end]`,
/// capped at `max_count`. Accepts both `linear:start..end/step` and
/// `linear:start/step` (open-ended, end resolved to `max_count - 1`).
///
/// Mirrors the open-ended-by-default style of `mul:start/factor` so a
/// spec stored in dataset.yaml stays independent of the dataset's exact
/// cardinality and resolves at runtime.
fn parse_linear_range(spec: &str, max_count: u64) -> Result<Vec<(String, u64)>, String> {
    let (range_part, step_str) = spec.split_once('/').ok_or_else(|| {
        format!(
            "invalid linear spec '{}': expected linear:start/step or linear:start..end/step",
            spec,
        )
    })?;
    let (start, explicit_end) = if let Some((start_str, end_str)) = range_part.split_once("..") {
        (
            parse_number_with_suffix(start_str.trim())?,
            Some(parse_number_with_suffix(end_str.trim())?),
        )
    } else {
        (parse_number_with_suffix(range_part.trim())?, None)
    };
    let end = match (explicit_end, max_count) {
        (Some(e), mc) if mc > 0 => e.min(mc - 1),
        (Some(e), _) => e,
        (None, mc) if mc > 0 => mc - 1,
        (None, _) => return Err(format!(
            "implicit upper bound form 'linear:{}' requires a known base count", spec)),
    };
    let step = parse_number_with_suffix(step_str.trim())?;
    if step == 0 {
        return Err(format!("linear step must be > 0 in '{}'", spec));
    }
    if start == 0 {
        return Err(format!("linear start must be > 0 in '{}'", spec));
    }
    if start > end {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    let mut v = start;
    while v <= end {
        if max_count == 0 || v < max_count {
            result.push((format_count_with_suffix(v), v));
        }
        v = v.saturating_add(step);
        if v == 0 { break; }
    }
    Ok(result)
}

/// Default starting value for `decade` when no explicit start is given.
/// Pinned to 100k because billion-vector benchmarks rarely care about
/// thousands-scale profiles; this default sweeps ~30 useful profiles
/// across four orders of magnitude (100k…900k, 1m…9m, 10m…90m,
/// 100m…900m, 1g+) without producing noise at the small end.
const DECADE_DEFAULT_START: u64 = 100_000;

/// Generate decade-stepped values within `[start, end]`, capped at
/// `max_count`. The step size is the largest power of 10 that's
/// `<= current value`; once the count crosses into the next decade
/// (e.g., 90 → 100), the step rescales accordingly.
///
/// Named after engineering "decade" plots / decade resistors / decade
/// scales — the same 1, 2, …, 9, 10, 20, …, 90, 100, 200, … layout.
/// Always emits SI/decimal units (`k`, `m`, `g`) — never binary
/// (`ki`, `mi`, `gi`) — because every value is a clean decimal
/// multiple by construction.
///
/// Examples:
///   - `decade:5` → 5, 6, 7, 8, 9, 10, 20, 30, …, 90, 100, 200, … (capped)
///   - `decade:1` → 1, 2, …, 9, 10, 20, …, 90, 100, …
///   - `decade:100k` → 100k, 200k, …, 900k, 1m, 2m, …, 9m, 10m, …
///   - `decade:5..100` → 5, 6, …, 9, 10, 20, …, 90, 100
///   - `decade` (no body) → starts at 100k
///
/// Form: `decade:<start>` or `decade:<start>..<end>` or bare `decade`.
fn parse_decade_range(spec: &str, max_count: u64) -> Result<Vec<(String, u64)>, String> {
    let spec = spec.trim();
    let (start, explicit_end) = if spec.is_empty() {
        // Bare `decade` — apply the documented default start.
        (DECADE_DEFAULT_START, None)
    } else if let Some((start_str, end_str)) = spec.split_once("..") {
        // `decade:..N` (empty start) → use default start
        let start_trimmed = start_str.trim();
        let s = if start_trimmed.is_empty() {
            DECADE_DEFAULT_START
        } else {
            parse_number_with_suffix(start_trimmed)?
        };
        (s, Some(parse_number_with_suffix(end_str.trim())?))
    } else {
        (parse_number_with_suffix(spec)?, None)
    };
    let end = match (explicit_end, max_count) {
        (Some(e), mc) if mc > 0 => e.min(mc - 1),
        (Some(e), _) => e,
        (None, mc) if mc > 0 => mc - 1,
        (None, _) => return Err(format!(
            "implicit upper bound form 'decade{}{}' requires a known base count",
            if spec.is_empty() { "" } else { ":" }, spec)),
    };
    if start == 0 {
        return Err(format!("decade start must be > 0, got 'decade:{}'", spec));
    }
    if start > end {
        return Ok(Vec::new());
    }

    // Step is the largest power of 10 that's <= start. So `decade:5`
    // starts with step 1 (giving 5,6,7,8,9 before the rescale), and
    // `decade:100k` starts with step 100k (giving 100k,200k,…,900k).
    let mut step = pow10_floor(start);
    let mut v = start;
    let mut result = Vec::new();
    let mut guard = 0usize;
    while v <= end {
        if max_count == 0 || v < max_count {
            result.push((format_count_with_suffix(v), v));
        }
        // Crossing 10× the current step jumps us into the next decade,
        // and the new step IS the new floor (so 90+10=100, then step 100).
        let next = v.saturating_add(step);
        if next == 0 { break; }
        v = next;
        if v >= step.saturating_mul(10) {
            step = pow10_floor(v);
        }
        // Defensive: every iteration must make forward progress; cap
        // total iterations at a generous upper bound so a malformed
        // input can't spin forever.
        guard += 1;
        if guard > 10_000 {
            return Err(format!("decade generator runaway for 'decade:{}'", spec));
        }
    }
    Ok(result)
}

/// Largest power of 10 that's `<= n`. For n=0 returns 1 to avoid a
/// degenerate step in the decade generator (which guards n>0 anyway).
fn pow10_floor(n: u64) -> u64 {
    if n == 0 { return 1; }
    let mut p: u64 = 1;
    while let Some(next) = p.checked_mul(10) {
        if next > n { break; }
        p = next;
    }
    p
}

// ---------------------------------------------------------------------------
// Serde: DSProfileGroup
// ---------------------------------------------------------------------------

impl<'de> Deserialize<'de> for DSProfileGroup {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        // Two-pass: parse default first, then inherit into child profiles
        let raw: IndexMap<String, serde_yaml::Value> = IndexMap::deserialize(deserializer)?;

        // Pass 1: parse default profile
        let default_profile = if let Some(default_val) = raw.get("default") {
            let p: DSProfile =
                serde_yaml::from_value(default_val.clone()).map_err(de::Error::custom)?;
            Some(p)
        } else {
            None
        };

        let mut profiles = IndexMap::new();
        let mut deferred_sized: Vec<String> = Vec::new();
        let mut deferred_facet_templates: Option<IndexMap<String, String>> = None;
        let mut sized_profile_names: Vec<String> = Vec::new();

        for (name, value) in &raw {
            if name == "default" {
                // Already parsed
                if let Some(ref dp) = default_profile {
                    profiles.insert(name.clone(), dp.clone());
                }
                continue;
            }

            if name == "sized" {
                // Sized profile sugar — two forms:
                //
                // 1. Simple list: `sized: [10m, 20m, 100m..400m/100m]`
                // 2. Structured map with ranges + facets scaffold:
                //    ```yaml
                //    sized:
                //      ranges: ["0m..400m/10m"]
                //      facets:
                //        base_vectors: "base_vectors.mvec:${range}"
                //        query_vectors: query_vectors.mvec
                //    ```
                //
                // In form 2, facet templates are interpolated per profile:
                //   ${profile} → profile name (e.g., "10m")
                //   ${range}   → window spec "[0..base_count]"
                let (entries, facet_templates) = match value {
                    serde_yaml::Value::Sequence(_) => {
                        let entries: Vec<String> = serde_yaml::from_value(value.clone())
                            .map_err(de::Error::custom)?;
                        (entries, None)
                    }
                    serde_yaml::Value::Mapping(_) => {
                        let map: IndexMap<String, serde_yaml::Value> =
                            serde_yaml::from_value(value.clone()).map_err(de::Error::custom)?;
                        let ranges_val = map.get("ranges").ok_or_else(|| {
                            de::Error::custom("sized map must have 'ranges' key")
                        })?;
                        let entries: Vec<String> = serde_yaml::from_value(ranges_val.clone())
                            .map_err(de::Error::custom)?;
                        let facets: Option<IndexMap<String, String>> = map.get("facets")
                            .map(|v| serde_yaml::from_value(v.clone()))
                            .transpose()
                            .map_err(de::Error::custom)?;
                        (entries, facets)
                    }
                    _ => return Err(de::Error::custom(
                        "sized must be a list of ranges or a map with 'ranges' + optional 'facets'"
                    )),
                };

                // Separate entries with variable references (deferred)
                // from those that can be resolved immediately.
                let mut deferred: Vec<String> = Vec::new();
                let mut all_pairs: Vec<(String, u64)> = Vec::new();
                for entry_str in &entries {
                    if entry_str.contains("${") || needs_base_count(entry_str) {
                        // Entries with unresolved variables or implicit upper
                        // bounds must be deferred until base_count is known.
                        deferred.push(entry_str.clone());
                    } else {
                        let pairs = parse_sized_entry(entry_str)
                            .map_err(de::Error::custom)?;
                        all_pairs.extend(pairs);
                    }
                }
                all_pairs.sort_by_key(|(_, count)| *count);
                for (prof_name, count) in all_pairs {
                    // Build views from facet templates if provided
                    let scaffold_views = if let Some(ref templates) = facet_templates {
                        let range_spec = format!("[0..{}]", count);
                        let mut views = IndexMap::new();
                        for (facet_key, template) in templates {
                            let interpolated = template
                                .replace("${profile}", &prof_name)
                                .replace("${range}", &range_spec);
                            let canonical = resolve_standard_key(facet_key)
                                .unwrap_or_else(|| facet_key.clone());
                            let source = parse_source_string(&interpolated)
                                .map_err(de::Error::custom)?;
                            views.insert(canonical, DSView { source, window: None });
                        }
                        Some(views)
                    } else {
                        None
                    };

                    let base_views = scaffold_views.unwrap_or_default();

                    // Inherit from default, then overlay scaffold views.
                    // Per-profile output views (neighbor_indices, etc.) are
                    // excluded — sized profiles compute their own KNN
                    // outputs into `profiles/{name}/`, so inheriting
                    // default's path would mis-route every verify and
                    // every external consumer to default's full-base GT.
                    let merged = if let Some(ref dp) = default_profile {
                        let mut merged_views: IndexMap<String, DSView> = dp.views.iter()
                            .filter(|(k, _)| !is_per_profile_output_view(k))
                            .map(|(k, v)| {
                                // Same windowed-share treatment as
                                // `expand_deferred_sized` — see comments
                                // there. base_vectors / metadata_content
                                // get a `[0..count)` window when
                                // inherited; everything else copies
                                // through unchanged.
                                if is_windowed_shared_view(k) && v.source.window.is_empty() {
                                    let mut windowed = v.clone();
                                    windowed.source.window = DSWindow(vec![DSInterval {
                                        min_incl: 0,
                                        max_excl: count,
                                    }]);
                                    (k.clone(), windowed)
                                } else {
                                    (k.clone(), v.clone())
                                }
                            })
                            .collect();
                        for (k, v) in base_views {
                            merged_views.insert(k, v);
                        }
                        DSProfile {
                            maxk: dp.maxk,
                            base_count: Some(count),
                            partition: false,
                            views: merged_views,
                        }
                    } else {
                        DSProfile {
                            maxk: None,
                            base_count: Some(count),
                            partition: false,
                            views: base_views,
                        }
                    };
                    if profiles.contains_key(&prof_name) {
                        return Err(de::Error::custom(format!(
                            "sized profile '{}' conflicts with explicitly defined profile of the same name",
                            prof_name,
                        )));
                    }
                    sized_profile_names.push(prof_name.clone());
                    profiles.insert(prof_name, merged);
                }
                if !deferred.is_empty() {
                    deferred_sized = deferred;
                    deferred_facet_templates = facet_templates;
                }
                continue;
            }

            // Parse child profile
            let child: DSProfile =
                serde_yaml::from_value(value.clone()).map_err(de::Error::custom)?;

            // Inherit views from default only if the child doesn't declare
            // its own base_vectors. Profiles with their own base_vectors are
            // self-contained (partition profiles, externally provided data)
            // and should NOT inherit metadata, filtered KNN, etc.
            // Self-contained: profile has base_vectors in its own directory
            // (e.g., profiles/label_0/base_vectors.fvec). Sized profiles that
            // reference the default's file with a window are NOT self-contained.
            let own_dir = format!("profiles/{}/", name);
            let is_self_contained = name != "default" && child.views.get("base_vectors")
                .map(|v| v.source.path.starts_with(&own_dir))
                .unwrap_or(false);
            let merged = if is_self_contained {
                // Self-contained: inherit only maxk
                DSProfile {
                    maxk: child.maxk.or(default_profile.as_ref().and_then(|dp| dp.maxk)),
                    base_count: child.base_count,
                    partition: is_self_contained,
                    views: child.views,
                }
            } else if let Some(ref dp) = default_profile {
                // Inherit shared views (base_vectors, query_vectors, ...)
                // from default. Per-profile output views are excluded —
                // every sized/derived profile owns its own
                // `profiles/{name}/neighbor_*` files; inheriting default's
                // path silently routes verifies and downstream consumers
                // to the wrong GT.
                //
                // Per-vector shared views (`base_vectors`,
                // `metadata_content`) get a `[0..base_count)` window
                // applied on inherit so the reader API can clip
                // automatically. The child can still override with an
                // explicit view (handled by the loop below). Without
                // a `base_count`, the window is meaningless — copy
                // through unchanged.
                let child_bc = child.base_count;
                let mut merged_views: IndexMap<String, DSView> = dp.views.iter()
                    .filter(|(k, _)| !is_per_profile_output_view(k))
                    .map(|(k, v)| {
                        if let Some(bc) = child_bc {
                            if is_windowed_shared_view(k) && v.source.window.is_empty() {
                                let mut windowed = v.clone();
                                windowed.source.window = DSWindow(vec![DSInterval {
                                    min_incl: 0,
                                    max_excl: bc,
                                }]);
                                return (k.clone(), windowed);
                            }
                        }
                        (k.clone(), v.clone())
                    })
                    .collect();
                for (k, v) in child.views {
                    merged_views.insert(k, v);
                }
                DSProfile {
                    maxk: child.maxk.or(dp.maxk),
                    base_count: child.base_count,
                    partition: child.partition,
                    views: merged_views,
                }
            } else {
                child
            };

            profiles.insert(name.clone(), merged);
        }

        // Collect all raw sized entries for round-trip serialization
        let raw_sized: Vec<String> = raw.get("sized").map(|value| {
            match value {
                serde_yaml::Value::Sequence(_) => {
                    serde_yaml::from_value::<Vec<String>>(value.clone()).unwrap_or_default()
                }
                serde_yaml::Value::Mapping(_) => {
                    let map: IndexMap<String, serde_yaml::Value> =
                        serde_yaml::from_value(value.clone()).unwrap_or_default();
                    map.get("ranges")
                        .and_then(|v| serde_yaml::from_value::<Vec<String>>(v.clone()).ok())
                        .unwrap_or_default()
                }
                _ => Vec::new(),
            }
        }).unwrap_or_default();

        Ok(DSProfileGroup {
            profiles,
            deferred_sized,
            deferred_facet_templates,
            raw_sized,
            sized_profile_names,
        })
    }
}

impl fmt::Display for DSView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.source)?;
        if let Some(ref w) = self.window {
            write!(f, " window={}", w)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_from_bare_string() {
        let v: DSView = serde_yaml::from_str("\"file.fvec\"").unwrap();
        assert_eq!(v.source.path, "file.fvec");
        assert!(v.window.is_none());
    }

    #[test]
    fn test_view_from_map() {
        let yaml = r#"
source: file.fvec
window: "0..1000"
"#;
        let v: DSView = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(v.source.path, "file.fvec");
        assert!(v.window.is_some());
        assert_eq!(v.window.unwrap().0[0].max_excl, 1000);
    }

    #[test]
    fn test_profile_basic() {
        let yaml = r#"
maxk: 100
base_vectors: base.fvec
query_vectors: query.fvec
"#;
        let p: DSProfile = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(p.maxk, Some(100));
        assert_eq!(p.views.len(), 2);
        assert!(p.views.contains_key("base_vectors"));
        assert!(p.views.contains_key("query_vectors"));
    }

    #[test]
    fn test_profile_alias_resolution() {
        let yaml = r#"
base: base.fvec
query: query.fvec
indices: gnd/idx.ivecs
distances: gnd/dis.fvecs
"#;
        let p: DSProfile = serde_yaml::from_str(yaml).unwrap();
        assert!(p.views.contains_key("base_vectors"));
        assert!(p.views.contains_key("query_vectors"));
        assert!(p.views.contains_key("neighbor_indices"));
        assert!(p.views.contains_key("neighbor_distances"));
    }

    #[test]
    fn test_profile_custom_facets_preserved() {
        let yaml = r#"
base_vectors: base.fvec
model_profile: model.json
sketch_vectors: sketch.mvec
"#;
        let p: DSProfile = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(p.views.len(), 3);
        assert!(p.views.contains_key("base_vectors"));
        assert!(p.views.contains_key("model_profile"));
        assert!(p.views.contains_key("sketch_vectors"));
    }

    #[test]
    fn test_profile_group_single_default() {
        let yaml = r#"
default:
  maxk: 100
  base_vectors: base.fvec
  query_vectors: query.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(g.profiles.len(), 1);
        let dp = g.default_profile().unwrap();
        assert_eq!(dp.maxk, Some(100));
        assert_eq!(dp.views.len(), 2);
    }

    #[test]
    fn test_profile_group_inheritance() {
        let yaml = r#"
default:
  maxk: 100
  base_vectors: base.fvec
  query_vectors: query.fvec
  neighbor_indices: gnd/idx.ivecs
  neighbor_distances: gnd/dis.fvecs
1M:
  base_vectors:
    source: base.fvec
    window: "0..1000000"
  neighbor_indices: gnd/idx_1M.ivecs
  neighbor_distances: gnd/dis_1M.fvecs
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(g.profiles.len(), 2);

        let child = g.profile("1M").unwrap();
        // Should inherit maxk from default
        assert_eq!(child.maxk, Some(100));
        // Should have all 4 views (2 inherited, 2 overridden + 1 from base inherited)
        assert_eq!(child.views.len(), 4);
        // query_vectors inherited from default
        assert_eq!(child.view("query_vectors").unwrap().path(), "query.fvec");
        // base_vectors overridden with window
        let base_view = child.view("base_vectors").unwrap();
        assert_eq!(base_view.source.path, "base.fvec");
        assert!(base_view.window.is_some());
    }

    #[test]
    fn test_profile_group_custom_facets_inherited() {
        let yaml = r#"
default:
  base_vectors: base.fvec
  model_profile: model.json
child:
  query_vectors: query.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let child = g.profile("child").unwrap();
        // Custom facet inherited from default
        assert!(child.views.contains_key("model_profile"));
        assert!(child.views.contains_key("base_vectors"));
        assert!(child.views.contains_key("query_vectors"));
    }

    #[test]
    fn test_profile_group_child_maxk_override() {
        let yaml = r#"
default:
  maxk: 100
  base_vectors: base.fvec
child:
  maxk: 50
  base_vectors: base_small.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let child = g.profile("child").unwrap();
        assert_eq!(child.maxk, Some(50));
    }

    #[test]
    fn test_profile_group_no_default() {
        let yaml = r#"
only:
  base_vectors: base.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        assert!(g.default_profile().is_none());
        assert_eq!(g.profiles.len(), 1);
    }

    #[test]
    fn test_view_names() {
        let yaml = r#"
default:
  base_vectors: base.fvec
  query_vectors: query.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let names = g.view_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"base_vectors"));
        assert!(names.contains(&"query_vectors"));
    }

    #[test]
    fn test_profile_names() {
        let yaml = r#"
default:
  base_vectors: base.fvec
1M:
  base_vectors: base_1M.fvec
10M:
  base_vectors: base_10M.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let names = g.profile_names();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn test_view_sugar_with_window_string() {
        let v: DSView = serde_yaml::from_str("\"file.fvec[0..1M]\"").unwrap();
        assert_eq!(v.source.path, "file.fvec");
        assert_eq!(v.source.window.0.len(), 1);
        assert_eq!(v.source.window.0[0].max_excl, 1_000_000);
    }

    #[test]
    fn test_profile_with_mixed_sugar() {
        let yaml = r#"
base: "base.fvec[0..1M]"
query: query.fvec
indices:
  source: gnd/idx.ivecs
  window: "0..1M"
"#;
        let p: DSProfile = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(p.views.len(), 3);
        // base has inline window
        let base = p.view("base_vectors").unwrap();
        assert_eq!(base.source.window.0[0].max_excl, 1_000_000);
        // indices has map-level window
        let idx = p.view("neighbor_indices").unwrap();
        assert!(idx.window.is_some());
    }

    #[test]
    fn test_sized_simple_list() {
        let yaml = r#"
default:
  maxk: 100
  query_vectors: query.fvec
sized: [10m, 20m, 50m]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // default + 3 sized profiles
        assert_eq!(g.profiles.len(), 4);

        let p10 = g.profile("10m").unwrap();
        assert_eq!(p10.base_count, Some(10_000_000));
        assert_eq!(p10.maxk, Some(100)); // inherited
        assert!(p10.views.contains_key("query_vectors")); // inherited

        let p50 = g.profile("50m").unwrap();
        assert_eq!(p50.base_count, Some(50_000_000));
    }

    #[test]
    fn test_sized_range_expansion() {
        let yaml = r#"
default:
  maxk: 100
sized: [100m..400m/100m]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // default + 4 expanded profiles (100m, 200m, 300m, 400m)
        assert_eq!(g.profiles.len(), 5);
        assert_eq!(g.profile("100m").unwrap().base_count, Some(100_000_000));
        assert_eq!(g.profile("200m").unwrap().base_count, Some(200_000_000));
        assert_eq!(g.profile("300m").unwrap().base_count, Some(300_000_000));
        assert_eq!(g.profile("400m").unwrap().base_count, Some(400_000_000));
    }

    #[test]
    fn test_sized_mixed_simple_and_range() {
        let yaml = r#"
default:
  maxk: 100
sized: [10m, 20m, 100m..300m/100m]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // default + 10m + 20m + 100m + 200m + 300m = 6
        assert_eq!(g.profiles.len(), 6);
        assert!(g.profile("10m").is_some());
        assert!(g.profile("20m").is_some());
        assert!(g.profile("100m").is_some());
        assert!(g.profile("200m").is_some());
        assert!(g.profile("300m").is_some());
    }

    #[test]
    fn test_sized_with_k_suffix() {
        let yaml = r#"
sized: [100k, 500k, 1m]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(g.profiles.len(), 3);
        assert_eq!(g.profile("100k").unwrap().base_count, Some(100_000));
        assert_eq!(g.profile("500k").unwrap().base_count, Some(500_000));
        assert_eq!(g.profile("1m").unwrap().base_count, Some(1_000_000));
    }

    #[test]
    fn test_sized_no_default() {
        let yaml = r#"
sized: [10m, 20m]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(g.profiles.len(), 2);
        let p = g.profile("10m").unwrap();
        assert_eq!(p.base_count, Some(10_000_000));
        assert!(p.maxk.is_none());
        assert!(p.views.is_empty());
    }

    #[test]
    fn test_sized_coexists_with_explicit_profiles() {
        let yaml = r#"
default:
  maxk: 100
  query_vectors: query.fvec
sized: [10m, 20m]
custom:
  maxk: 50
  base_vectors: custom_base.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // default + 10m + 20m + custom = 4
        assert_eq!(g.profiles.len(), 4);
        assert!(g.profile("10m").is_some());
        assert!(g.profile("custom").is_some());
        assert_eq!(g.profile("custom").unwrap().maxk, Some(50));
    }

    #[test]
    fn test_sized_sorted_by_count() {
        let yaml = r#"
default:
  maxk: 100
sized: [50m, 10m, 100m..300m/100m, 20m]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let names = g.profile_names();
        // default first, then sized sorted: 10m, 20m, 50m, 100m, 200m, 300m
        assert_eq!(names, vec!["default", "10m", "20m", "50m", "100m", "200m", "300m"]);
    }

    #[test]
    fn test_parse_sized_entry_simple() {
        let pairs = parse_sized_entry("10m").unwrap();
        assert_eq!(pairs, vec![("10m".to_string(), 10_000_000)]);
    }

    #[test]
    fn test_parse_sized_entry_range() {
        let pairs = parse_sized_entry("1m..3m/1m").unwrap();
        assert_eq!(pairs, vec![
            ("1m".to_string(), 1_000_000),
            ("2m".to_string(), 2_000_000),
            ("3m".to_string(), 3_000_000),
        ]);
    }

    #[test]
    fn test_parse_sized_entry_count_mode() {
        // 0m..400m/10 → 10 equal divisions: 40m, 80m, 120m, ..., 400m
        let pairs = parse_sized_entry("0m..400m/10").unwrap();
        assert_eq!(pairs.len(), 10);
        assert_eq!(pairs[0], ("40m".to_string(), 40_000_000));
        assert_eq!(pairs[9], ("400m".to_string(), 400_000_000));
        // All evenly spaced
        for i in 0..10 {
            assert_eq!(pairs[i].1, (i as u64 + 1) * 40_000_000);
        }
    }

    #[test]
    fn test_parse_step_prefix_matches_bare_form() {
        // The new explicit `step:` prefix is the recommended spelling
        // for arithmetic ranges with an alpha-suffixed step. The bare
        // form must continue to produce the same result.
        let bare = parse_sized_entry("1m..3m/1m").unwrap();
        let prefixed = parse_sized_entry("step:1m..3m/1m").unwrap();
        assert_eq!(bare, prefixed);
        assert_eq!(prefixed, vec![
            ("1m".to_string(), 1_000_000),
            ("2m".to_string(), 2_000_000),
            ("3m".to_string(), 3_000_000),
        ]);
    }

    #[test]
    fn test_parse_parts_prefix_matches_bare_form() {
        // Likewise for `parts:` — explicit-divisions-of-N spelling.
        let bare = parse_sized_entry("0m..400m/10").unwrap();
        let prefixed = parse_sized_entry("parts:0m..400m/10").unwrap();
        assert_eq!(bare, prefixed);
        assert_eq!(prefixed.len(), 10);
        assert_eq!(prefixed[0],  ("40m".to_string(),  40_000_000));
        assert_eq!(prefixed[9],  ("400m".to_string(), 400_000_000));
    }

    #[test]
    fn test_parse_step_prefix_overrides_inference() {
        // `step:` forces step-mode even when the divisor has no suffix
        // (which the bare-form heuristic would treat as parts-mode).
        let pairs = parse_sized_entry("step:1..5/2").unwrap();
        // Step of 2 from 1 to 5 → 1, 3, 5.
        let counts: Vec<u64> = pairs.iter().map(|(_, c)| *c).collect();
        assert_eq!(counts, vec![1, 3, 5]);
    }

    #[test]
    fn test_parse_parts_prefix_overrides_inference() {
        // `parts:` forces parts-mode even when the divisor has an
        // alpha suffix (which the bare-form heuristic would treat as
        // step-mode).
        let pairs = parse_sized_entry("parts:0m..400m/10").unwrap();
        assert_eq!(pairs.len(), 10);
    }

    #[test]
    fn test_parse_fib_open_resolves_at_runtime() {
        // No max → error (cannot resolve open-ended upper bound)
        assert!(parse_sized_entry("fib:1m").is_err());
        // With a max → caps at max - 1
        let pairs = parse_sized_entry_impl("fib:1m", 100_000_000).unwrap();
        assert!(!pairs.is_empty());
        assert!(pairs.iter().all(|(_, v)| *v < 100_000_000));
        assert_eq!(pairs[0].1, 1_000_000);
    }

    #[test]
    fn test_parse_linear_open_resolves_at_runtime() {
        assert!(parse_sized_entry("linear:10m/10m").is_err());
        let pairs = parse_sized_entry_impl("linear:10m/10m", 100_000_000).unwrap();
        // 10m, 20m, ..., 90m (100m excluded since v < max_count)
        assert_eq!(pairs.len(), 9);
        assert_eq!(pairs[0].1, 10_000_000);
        assert_eq!(pairs[8].1, 90_000_000);
    }

    #[test]
    fn test_parse_linear_explicit_range() {
        let pairs = parse_sized_entry("linear:5m..20m/5m").unwrap();
        assert_eq!(pairs.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                   vec![5_000_000, 10_000_000, 15_000_000, 20_000_000]);
    }

    #[test]
    fn test_composed_spec_with_open_ended_parts() {
        // The exact spec the wizard now emits — must parse under a runtime
        // cardinality and yield a non-empty mix of all three families.
        let spec = "mul:1mi/2,fib:1m,linear:10m/10m";
        let mut count = 0;
        for entry in spec.split(',') {
            let pairs = parse_sized_entry_impl(entry.trim(), 50_000_000).unwrap();
            assert!(!pairs.is_empty(), "empty pairs for entry {}", entry);
            count += pairs.len();
        }
        assert!(count > 10, "expected >10 total profiles, got {}", count);
    }

    #[test]
    fn test_parse_sized_entry_count_mode_nonzero_start() {
        // 100m..400m/3 → 3 divisions: 200m, 300m, 400m
        let pairs = parse_sized_entry("100m..400m/3").unwrap();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0], ("200m".to_string(), 200_000_000));
        assert_eq!(pairs[1], ("300m".to_string(), 300_000_000));
        assert_eq!(pairs[2], ("400m".to_string(), 400_000_000));
    }

    #[test]
    fn test_parse_sized_entry_range_not_aligned() {
        // 10m..25m/10m → 10m, 20m (25m is not reached because 20+10=30 > 25)
        let pairs = parse_sized_entry("10m..25m/10m").unwrap();
        assert_eq!(pairs, vec![
            ("10m".to_string(), 10_000_000),
            ("20m".to_string(), 20_000_000),
        ]);
    }

    #[test]
    fn test_fib_range() {
        let pairs = parse_sized_entry("fib:1m..400m").unwrap();
        let counts: Vec<u64> = pairs.iter().map(|(_, c)| *c).collect();
        // fib multiples of 1m: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377
        // deduplicated start: 1m, 2m, 3m, 5m, 8m, 13m, 21m, 34m, 55m, 89m, 144m, 233m, 377m
        assert_eq!(counts[0], 1_000_000);
        assert_eq!(counts[1], 1_000_000); // fib(2)=1 again, that's fine
        assert_eq!(counts[2], 2_000_000);
        assert_eq!(counts[3], 3_000_000);
        assert_eq!(counts[4], 5_000_000);
        assert!(*counts.last().unwrap() <= 400_000_000);
    }

    #[test]
    fn test_mul_doubling() {
        let pairs = parse_sized_entry("mul:1m..100m/2").unwrap();
        let counts: Vec<u64> = pairs.iter().map(|(_, c)| *c).collect();
        // 1m, 2m, 4m, 8m, 16m, 32m, 64m
        assert_eq!(counts, vec![
            1_000_000, 2_000_000, 4_000_000, 8_000_000,
            16_000_000, 32_000_000, 64_000_000,
        ]);
    }

    #[test]
    fn test_mul_factor_1_5() {
        let pairs = parse_sized_entry("mul:10m..100m/1.5").unwrap();
        let counts: Vec<u64> = pairs.iter().map(|(_, c)| *c).collect();
        // 10m, 15m, 22.5m→22500000, 33.75m→33750000, 50.625m→50625000, 75.9375m→75937500
        assert_eq!(counts[0], 10_000_000);
        assert_eq!(counts[1], 15_000_000);
        assert!(counts.len() >= 4);
        assert!(*counts.last().unwrap() <= 100_000_000);
    }

    #[test]
    fn test_sized_structured_with_facets() {
        let yaml = r#"
default:
  maxk: 100
  query_vectors: query_vectors.mvec
sized:
  ranges: ["10m", "20m"]
  facets:
    base_vectors: "base_vectors.mvec:${range}"
    metadata_content: "metadata.slab:${range}"
    neighbor_indices: "profiles/${profile}/neighbor_indices.ivec"
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // default + 2 sized profiles
        assert_eq!(g.profiles.len(), 3);

        let p10 = g.profile("10m").unwrap();
        assert_eq!(p10.base_count, Some(10_000_000));
        assert_eq!(p10.maxk, Some(100)); // inherited

        // base_vectors should have path + window from ${range}
        let bv = p10.view("base_vectors").unwrap();
        assert_eq!(bv.source.path, "base_vectors.mvec");
        assert_eq!(bv.source.window.0.len(), 1);
        assert_eq!(bv.source.window.0[0].min_incl, 0);
        assert_eq!(bv.source.window.0[0].max_excl, 10_000_000);

        // metadata_content should have path + window
        let mc = p10.view("metadata_content").unwrap();
        assert_eq!(mc.source.path, "metadata.slab");
        assert_eq!(mc.source.window.0[0].max_excl, 10_000_000);

        // neighbor_indices should have ${profile} interpolated
        let ni = p10.view("neighbor_indices").unwrap();
        assert_eq!(ni.source.path, "profiles/10m/neighbor_indices.ivec");
        assert!(ni.source.window.is_empty());

        // query_vectors inherited from default
        assert!(p10.views.contains_key("query_vectors"));

        // Check 20m profile too
        let p20 = g.profile("20m").unwrap();
        let bv20 = p20.view("base_vectors").unwrap();
        assert_eq!(bv20.source.window.0[0].max_excl, 20_000_000);
        let ni20 = p20.view("neighbor_indices").unwrap();
        assert_eq!(ni20.source.path, "profiles/20m/neighbor_indices.ivec");
    }

    #[test]
    fn test_sized_structured_backward_compat() {
        // Old list format should still work
        let yaml = r#"
default:
  maxk: 100
sized: [10m, 20m]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(g.profiles.len(), 3);
        assert_eq!(g.profile("10m").unwrap().base_count, Some(10_000_000));
    }

    #[test]
    fn test_mul_in_sized_yaml() {
        let yaml = r#"
default:
  maxk: 100
sized: ["mul:1m..16m/2"]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let names = g.profile_names();
        // default + 1m, 2m, 4m, 8m, 16m
        assert_eq!(names, vec!["default", "1m", "2m", "4m", "8m", "16m"]);
    }

    #[test]
    fn test_mul_implicit_end_in_yaml() {
        // The implicit-end form `mul:1m/2` is deferred at parse time since
        // base_count is not yet known. It expands when expand_deferred_sized
        // is called with the actual base_count.
        let yaml = r#"
default:
  maxk: 100
  base_vectors: base.mvec
sized: ["mul:1m/2"]
"#;
        let mut g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // Only default is present before expansion
        assert_eq!(g.profile_names(), vec!["default"]);
        assert!(g.has_deferred());

        // Expand with a known base_count of 200m
        let vars = indexmap::IndexMap::new();
        let added = g.expand_deferred_sized(&vars, 200_000_000);
        assert!(added > 5, "expected multiple profiles, got {}", added);
        let names = g.profile_names();
        assert_eq!(names[0], "default");
        assert_eq!(names[1], "1m");
        assert_eq!(names[2], "2m");
        assert_eq!(names[3], "4m");
        // Last sized profile must be < 200m
        let last_sized = g.profiles.iter()
            .filter(|(n, _)| *n != "default")
            .max_by_key(|(_, p)| p.base_count.unwrap_or(0))
            .unwrap();
        assert!(last_sized.1.base_count.unwrap() < 200_000_000,
            "last profile {} has base_count {} >= 200m",
            last_sized.0, last_sized.1.base_count.unwrap());
        // Sized profiles should inherit base_vectors from default
        let p1m = g.profile("1m").unwrap();
        assert!(p1m.base_count.is_some());
        assert_eq!(p1m.base_count.unwrap(), 1_000_000);
        assert!(p1m.view("base_vectors").is_some(),
            "sized profile should inherit base_vectors from default");
    }

    #[test]
    fn test_deferred_expansion_sorts_across_strategies() {
        // Multiple deferred entries from different generator strategies
        // (decade, fib, mul, linear) must produce a single ordering by
        // count, not "all decade then all fib then all mul". Otherwise
        // the on-disk YAML and the TUI profile picker show 100k *after*
        // 8m, which is what triggered this regression.
        let yaml = r#"
default:
  maxk: 100
  base_vectors: base.mvec
sized: ["decade", "mul:1m/2", "linear:10m/10m"]
"#;
        let mut g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let vars = indexmap::IndexMap::new();
        let added = g.expand_deferred_sized(&vars, 50_000_000);
        assert!(added > 5, "expected multiple profiles, got {}", added);

        // Inspect the IndexMap insertion order directly (not
        // profile_names(), which sorts on read) — the goal is that the
        // *stored* order is already sorted, so YAML emission and any
        // direct iteration match.
        let counts: Vec<u64> = g.profiles.iter()
            .filter(|(n, _)| *n != "default")
            .map(|(_, p)| p.base_count.unwrap_or(0))
            .collect();
        assert!(counts.windows(2).all(|w| w[0] <= w[1]),
            "expanded sized profiles must be inserted in ascending count order, got {:?}",
            counts);
        // Sanity: 100k from decade really did precede 1m from mul.
        let names: Vec<&str> = g.profiles.keys()
            .map(|s| s.as_str())
            .filter(|n| *n != "default")
            .collect();
        let pos_100k = names.iter().position(|n| *n == "100k");
        let pos_1m = names.iter().position(|n| *n == "1m");
        assert!(pos_100k.is_some() && pos_1m.is_some(),
            "expected both 100k (decade) and 1m (mul) in {:?}", names);
        assert!(pos_100k.unwrap() < pos_1m.unwrap(),
            "100k must come before 1m in {:?}", names);
    }

    #[test]
    fn test_derive_views_from_templates() {
        use crate::dataset::pipeline::StepDef;
        use indexmap::IndexMap;

        let yaml = r#"
default:
  query_vectors: query.mvec
10m:
  base_count: 10000000
"#;
        let mut profiles: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();

        // Build template steps with per_profile: true
        let mut opts1 = IndexMap::new();
        opts1.insert(
            "output".to_string(),
            serde_yaml::Value::String("base_vectors.mvec".to_string()),
        );
        let mut opts2 = IndexMap::new();
        opts2.insert(
            "output".to_string(),
            serde_yaml::Value::String("neighbor_indices.ivec".to_string()),
        );
        let templates = vec![
            StepDef {
                id: Some("extract".to_string()),
                run: "generate mvec-extract".to_string(),
                description: None,
                after: vec![],
                profiles: vec![],
                per_profile: true,
                phase: 0,
                finalize: false,
                on_partial: crate::dataset::pipeline::OnPartial::default(),
                options: opts1,
            },
            StepDef {
                id: Some("knn".to_string()),
                run: "compute knn".to_string(),
                description: None,
                after: vec![],
                profiles: vec![],
                per_profile: true,
                phase: 0,
                finalize: false,
                on_partial: crate::dataset::pipeline::OnPartial::default(),
                options: opts2,
            },
        ];
        profiles.derive_views_from_templates(&templates);

        // Default gets auto-derived views
        let pdef = profiles.profile("default").unwrap();
        assert_eq!(
            pdef.view("base_vectors").unwrap().path(),
            "profiles/default/base_vectors.mvec"
        );
        assert_eq!(
            pdef.view("neighbor_indices").unwrap().path(),
            "profiles/default/neighbor_indices.ivec"
        );
        // Explicit view preserved
        assert_eq!(pdef.view("query_vectors").unwrap().path(), "query.mvec");

        // Sized profile: per-profile-output views (base_vectors
        // treated as a per_profile template output here) point at the
        // profile's own directory — the earlier "window into default"
        // model was wrong for neighbor_indices (can't window a
        // top-K by prefix) and we use a single rule for all
        // per_profile outputs.
        let p10 = profiles.profile("10m").unwrap();
        assert_eq!(
            p10.view("base_vectors").unwrap().path(),
            "profiles/10m/base_vectors.mvec"
        );
        assert!(p10.view("base_vectors").unwrap().source.window.is_empty());
        assert_eq!(
            p10.view("neighbor_indices").unwrap().path(),
            "profiles/10m/neighbor_indices.ivec"
        );
        assert!(p10.view("neighbor_indices").unwrap().source.window.is_empty());
        // Inherited shared view unchanged
        assert_eq!(p10.view("query_vectors").unwrap().path(), "query.mvec");
    }

    #[test]
    fn test_parse_mul_implicit_end() {
        // mul:1m/2 (no ..end) requires base_count to be known.
        // With max_count=0 (unknown), this is an error.
        assert!(parse_sized_entry("mul:1m/2").is_err());

        // With a known base_count, it generates profiles up to (not including) that count.
        let pairs = parse_sized_entry_impl("mul:1m/2", 200_000_000).unwrap();
        let counts: Vec<u64> = pairs.iter().map(|(_, c)| *c).collect();
        // Should start at 1m and double: 1m, 2m, 4m, 8m, 16m, 32m, 64m, 128m
        // 256m would be >= 200m so it's not generated.
        assert_eq!(counts[0], 1_000_000);
        assert_eq!(counts[1], 2_000_000);
        assert_eq!(counts[2], 4_000_000);
        // Each successive value should be double the previous
        for i in 1..counts.len() {
            assert_eq!(counts[i], counts[i - 1] * 2,
                "expected doubling at index {}: {} vs {}", i, counts[i], counts[i - 1] * 2);
        }
        // Last value must be < 200m
        assert!(*counts.last().unwrap() < 200_000_000);
        assert_eq!(*counts.last().unwrap(), 128_000_000);
    }

    #[test]
    fn test_parse_mul_implicit_end_caps_at_100() {
        // mul:1/1.01 with a tiny factor and large base_count generates many
        // profiles. The 100-profile safety cap should stop at exactly 100.
        let pairs = parse_sized_entry_impl("mul:1/1.01", 1_000_000_000).unwrap();
        assert_eq!(pairs.len(), 100,
            "expected exactly 100 profiles (safety cap), got {}", pairs.len());
        // First profile should be 1
        assert_eq!(pairs[0].1, 1);
    }

    #[test]
    fn test_sized_name_conflict_error() {
        // A sized profile conflicting with an explicitly defined profile should
        // produce a deserialization error. The explicit profile must appear
        // before the sized key so it is already in the profile map when sized
        // expansion tries to insert the same name.
        let yaml = r#"
default:
  maxk: 100
  base_vectors: base.fvec
10m:
  base_count: 10000000
  base_vectors: custom_base.fvec
sized: [10m, 20m]
"#;
        let result: Result<DSProfileGroup, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "expected error when sized profile conflicts with explicit profile");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("conflicts"),
            "error message should mention conflict, got: {}", err_msg);
    }

    #[test]
    fn test_sized_and_explicit_complementary() {
        // Non-conflicting sized + explicit profiles should coexist
        let yaml = r#"
default:
  maxk: 100
  base_vectors: base.fvec
  query_vectors: query.fvec
sized: [10m, 20m]
custom-queries:
  query_vectors: alt_queries.fvec
  neighbor_indices: alt_indices.ivec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // default + 10m + 20m + custom-queries = 4
        assert_eq!(g.profiles.len(), 4);

        // Sized profiles have base_count set
        let p10 = g.profile("10m").unwrap();
        assert_eq!(p10.base_count, Some(10_000_000));
        assert_eq!(p10.maxk, Some(100)); // inherited from default

        let p20 = g.profile("20m").unwrap();
        assert_eq!(p20.base_count, Some(20_000_000));

        // Explicit custom-queries profile is unaffected by sized expansion
        let custom = g.profile("custom-queries").unwrap();
        assert!(custom.base_count.is_none());
        assert_eq!(custom.view("query_vectors").unwrap().path(), "alt_queries.fvec");
        // custom-queries should inherit base_vectors from default
        assert_eq!(custom.view("base_vectors").unwrap().path(), "base.fvec");
    }

    #[test]
    fn test_natural_sort_partition_profiles() {
        // Verify that partition profiles sort in natural numeric order,
        // not lexicographic ASCII order.
        assert_eq!(super::natural_cmp("label_0", "label_1"), std::cmp::Ordering::Less);
        assert_eq!(super::natural_cmp("label_2", "label_10"), std::cmp::Ordering::Less);
        assert_eq!(super::natural_cmp("label_9", "label_10"), std::cmp::Ordering::Less);
        assert_eq!(super::natural_cmp("label_10", "label_10"), std::cmp::Ordering::Equal);
        assert_eq!(super::natural_cmp("label_100", "label_20"), std::cmp::Ordering::Greater);
        assert_eq!(super::natural_cmp("abc", "def"), std::cmp::Ordering::Less);
        assert_eq!(super::natural_cmp("a1b", "a2b"), std::cmp::Ordering::Less);
        assert_eq!(super::natural_cmp("a10b", "a2b"), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_profile_names_natural_order() {
        // Profile names with numeric suffixes should sort naturally.
        let yaml = r#"
default:
  base_vectors: base.fvec
label_00:
  base_count: 100
  base_vectors: profiles/label_00/base_vectors.fvec
label_01:
  base_count: 100
  base_vectors: profiles/label_01/base_vectors.fvec
label_10:
  base_count: 100
  base_vectors: profiles/label_10/base_vectors.fvec
label_02:
  base_count: 100
  base_vectors: profiles/label_02/base_vectors.fvec
label_09:
  base_count: 100
  base_vectors: profiles/label_09/base_vectors.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let names = g.profile_names();
        // default first, then label_00, 01, 02, 09, 10 in natural order
        // (all have same base_count, so tiebreak is by natural name sort)
        assert_eq!(names, vec!["default", "label_00", "label_01", "label_02", "label_09", "label_10"],
            "profile names should be in natural numeric order: {:?}", names);
    }

    // ── Sized-profile windowed-share inheritance ──────────────────────
    //
    // These tests pin down the documented sub-ordinal suffix model so
    // we don't quietly regress it again. The two invariants:
    //
    //   1. `base_vectors` (and `metadata_content`) inherited from
    //      default by a sized profile MUST carry a `[0..base_count)`
    //      window. Verifying via `view.base_vectors().count()` is
    //      worth a separate integration test (in
    //      veks/tests/e2e_http_sized_profiles.rs); these unit tests
    //      pin the data-model side.
    //
    //   2. Per-profile output views (`neighbor_indices`,
    //      `neighbor_distances`) MUST NOT be inherited at all — sized
    //      profiles own those files in their own directories. A
    //      windowed reference into default's GT would silently
    //      misroute every benchmark.

    #[test]
    fn sized_profile_inherits_base_vectors_with_window() {
        // Concrete sized profile entry in the YAML; deserializer
        // should attach a `[0..base_count)` window to the inherited
        // base_vectors view automatically.
        let yaml = r#"
default:
  base_vectors: profiles/base/base_vectors.fvecs
  query_vectors: profiles/base/query_vectors.fvecs
1m:
  base_count: 1000000
  neighbor_indices: profiles/1m/neighbor_indices.ivecs
  neighbor_distances: profiles/1m/neighbor_distances.fvecs
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let p = g.profile("1m").expect("1m exists");
        let bv = p.view("base_vectors").expect("base_vectors inherited");
        assert_eq!(bv.source.path, "profiles/base/base_vectors.fvecs",
            "path must come from default");
        assert_eq!(bv.source.window.0.len(), 1, "single window applied");
        assert_eq!(bv.source.window.0[0].min_incl, 0);
        assert_eq!(bv.source.window.0[0].max_excl, 1_000_000,
            "window end must equal base_count");
    }

    #[test]
    fn sized_profile_query_vectors_inherited_unwindowed() {
        // query_vectors is shared but NOT windowed — query sets are
        // identical across profiles, not a prefix of anything.
        let yaml = r#"
default:
  base_vectors: profiles/base/base_vectors.fvecs
  query_vectors: profiles/base/query_vectors.fvecs
1m:
  base_count: 1000000
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let p = g.profile("1m").unwrap();
        let qv = p.view("query_vectors").expect("query_vectors inherited");
        assert!(qv.source.window.is_empty(),
            "query_vectors must NOT carry a window — same query set as default");
    }

    #[test]
    fn sized_profile_does_not_inherit_per_profile_outputs() {
        // If default declares `neighbor_indices`, the sized profile
        // must NOT inherit it (would mis-route to default's full GT).
        // The sized profile's own GT lives at profiles/{name}/
        // — see e2e tests for the path-on-disk invariant.
        let yaml = r#"
default:
  base_vectors: profiles/base/base_vectors.fvecs
  neighbor_indices: profiles/default/neighbor_indices.ivecs
  neighbor_distances: profiles/default/neighbor_distances.fvecs
1m:
  base_count: 1000000
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let p = g.profile("1m").unwrap();
        assert!(p.view("neighbor_indices").is_none(),
            "sized profile must not inherit default's neighbor_indices");
        assert!(p.view("neighbor_distances").is_none(),
            "sized profile must not inherit default's neighbor_distances");
    }

    #[test]
    fn sized_profile_metadata_content_gets_window() {
        // metadata_content is per-vector — same windowing rule as
        // base_vectors.
        let yaml = r#"
default:
  base_vectors: profiles/base/base_vectors.fvecs
  metadata_content: profiles/base/metadata_content.slab
1m:
  base_count: 1000000
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let p = g.profile("1m").unwrap();
        let mc = p.view("metadata_content").expect("metadata_content inherited");
        assert_eq!(mc.source.window.0.first().map(|i| i.max_excl), Some(1_000_000));
    }

    #[test]
    fn explicit_view_on_sized_profile_overrides_inherited_window() {
        // Child profile with an explicit `base_vectors` declaration
        // overrides the auto-windowed inherited one — useful when the
        // sized profile actually has its own physical file.
        let yaml = r#"
default:
  base_vectors: profiles/base/base_vectors.fvecs
1m:
  base_count: 1000000
  base_vectors: profiles/1m/own_base.fvec
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        let p = g.profile("1m").unwrap();
        let bv = p.view("base_vectors").unwrap();
        assert_eq!(bv.source.path, "profiles/1m/own_base.fvec",
            "explicit override wins over inherited windowed view");
        assert!(bv.source.window.is_empty(),
            "explicit override carries no window unless the YAML says so");
    }

    // ── decade: strategy ──────────────────────────────────────────────
    //
    // Pin the documented behaviour:
    //   - Step is the largest power of 10 ≤ current value.
    //   - Step rescales every time the value crosses into the next decade.
    //   - Bare `decade` defaults to start=100k.
    //   - Always emits SI/decimal units (k/m/g) — never binary (ki/mi/gi).
    //   - Caps at max_count (exclusive).
    //   - Both `decade:start` (open-ended) and `decade:start..end`
    //     (explicit) forms work.
    //   - `decade:..N` (empty start) uses the default start.

    #[test]
    fn decade_starts_at_5_through_explicit_end() {
        // The motivating example from the spec.
        let r = parse_sized_entry_impl("decade:5..1000", 0).unwrap();
        let counts: Vec<u64> = r.iter().map(|(_, c)| *c).collect();
        assert_eq!(counts,
            vec![5,6,7,8,9, 10,20,30,40,50,60,70,80,90, 100,200,300,400,500,600,700,800,900, 1000]);
    }

    #[test]
    fn decade_starts_at_1_full_sweep() {
        let r = parse_sized_entry_impl("decade:1..100", 0).unwrap();
        let counts: Vec<u64> = r.iter().map(|(_, c)| *c).collect();
        assert_eq!(counts,
            vec![1,2,3,4,5,6,7,8,9, 10,20,30,40,50,60,70,80,90, 100]);
    }

    #[test]
    fn decade_default_start_is_100k() {
        // Bare `decade` with a known max → must start at 100k.
        let r = parse_sized_entry_impl("decade", 1_000_000).unwrap();
        let counts: Vec<u64> = r.iter().map(|(_, c)| *c).collect();
        // 100k, 200k, …, 900k. Cap excludes 1m (== max).
        assert_eq!(counts, vec![100_000, 200_000, 300_000, 400_000, 500_000,
                                600_000, 700_000, 800_000, 900_000]);
    }

    #[test]
    fn decade_default_start_via_explicit_end() {
        // `decade:..N` (empty start) also means default start.
        let r = parse_sized_entry_impl("decade:..1m", 0).unwrap();
        let counts: Vec<u64> = r.iter().map(|(_, c)| *c).collect();
        assert_eq!(counts, vec![100_000, 200_000, 300_000, 400_000, 500_000,
                                600_000, 700_000, 800_000, 900_000, 1_000_000]);
    }

    #[test]
    fn decade_with_si_suffix_start() {
        // `decade:100k` should produce the same as default start.
        let r = parse_sized_entry_impl("decade:100k", 1_000_000).unwrap();
        let counts: Vec<u64> = r.iter().map(|(_, c)| *c).collect();
        assert_eq!(counts, vec![100_000, 200_000, 300_000, 400_000, 500_000,
                                600_000, 700_000, 800_000, 900_000]);
    }

    #[test]
    fn decade_max_count_caps_strictly_below() {
        // `<` not `<=` — generated values must never equal max_count
        // (would conflict with the default profile).
        let r = parse_sized_entry_impl("decade:5..100", 50).unwrap();
        let counts: Vec<u64> = r.iter().map(|(_, c)| *c).collect();
        assert_eq!(counts, vec![5,6,7,8,9, 10,20,30,40]); // 50, 60, … excluded
    }

    #[test]
    fn decade_implicit_end_requires_max_count() {
        // `decade:5` with no max_count is an error — open-ended forms
        // must defer until base_count is known.
        assert!(parse_sized_entry_impl("decade:5", 0).is_err());
        assert!(parse_sized_entry_impl("decade", 0).is_err());
    }

    #[test]
    fn decade_implicit_end_resolves_at_runtime() {
        // With a known max, the open-ended form caps automatically.
        let r = parse_sized_entry_impl("decade:5", 100).unwrap();
        let counts: Vec<u64> = r.iter().map(|(_, c)| *c).collect();
        assert_eq!(counts, vec![5,6,7,8,9, 10,20,30,40,50,60,70,80,90]); // max=100 → <100
    }

    #[test]
    fn decade_uses_decimal_units_in_names() {
        // `format_count_with_suffix` is the naming source — for clean
        // decimal values it always picks the SI/decimal shorthand
        // (k/m/g) over binary (ki/mi/gi). Pin that here so a future
        // change to format_count_with_suffix doesn't sneak Mi back in.
        let r = parse_sized_entry_impl("decade:100k..10m", 0).unwrap();
        let names: Vec<&str> = r.iter().map(|(n, _)| n.as_str()).collect();
        for n in &names {
            assert!(
                !n.contains("ki") && !n.contains("mi") && !n.contains("gi"),
                "decade strategy must emit decimal units, got '{}'", n,
            );
        }
        // Spot-check expected names.
        assert!(names.contains(&"100k"));
        assert!(names.contains(&"1m"));
        assert!(names.contains(&"10m"));
    }

    #[test]
    fn decade_zero_start_rejected() {
        assert!(parse_sized_entry_impl("decade:0..100", 0).is_err());
    }

    #[test]
    fn decade_start_above_end_yields_empty() {
        let r = parse_sized_entry_impl("decade:1000..100", 0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn decade_needs_base_count_classification() {
        // Open-ended forms must defer until base_count is known.
        assert!(needs_base_count("decade"));
        assert!(needs_base_count("decade:5"));
        assert!(needs_base_count("decade:100k"));
        // Explicit-end forms resolve immediately.
        assert!(!needs_base_count("decade:5..100"));
        assert!(!needs_base_count("decade:100k..10m"));
        assert!(!needs_base_count("decade:..1m"));
    }

    #[test]
    fn decade_full_sweep_billion_scale() {
        // Realistic scenario: 1G base. Default-start decade should
        // cover 100k → 900m without producing 1g (== max) and without
        // any binary-unit naming.
        let r = parse_sized_entry_impl("decade", 1_000_000_000).unwrap();
        let counts: Vec<u64> = r.iter().map(|(_, c)| *c).collect();
        let names: Vec<&str> = r.iter().map(|(n, _)| n.as_str()).collect();
        // First few and last few — full list is 36 entries.
        assert_eq!(counts.first(), Some(&100_000));
        assert_eq!(counts.last(), Some(&900_000_000));
        assert_eq!(counts.len(), 36); // 9 per decade × 4 decades (100k, 1m, 10m, 100m)
        for n in &names {
            assert!(!n.contains("ki") && !n.contains("mi") && !n.contains("gi"),
                "all names must be decimal: '{}'", n);
        }
    }

    #[test]
    fn decade_via_root_strata_in_full_config() {
        // This mirrors the user-observed bug: yaml has root-level
        // `strata: [...]` containing "decade", plus a flock of
        // explicit child profile entries from a prior expansion.
        // After load+expand, the profile map MUST contain the new
        // decade-specific entries (100k, 4m, 6m, 7m, 9m, ...) that
        // didn't exist in the previous expansion.
        use crate::dataset::DatasetConfig;
        use std::io::Write;
        let yaml = r#"name: test

attributes:
  distance_function: DOT_PRODUCT

strata: ["mul:1mi/2", "fib:1m", "linear:10m/10m", "decade"]

profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.fvecs
    query_vectors: profiles/base/query_vectors.fvecs
  1m:
    maxk: 100
    base_count: 1000000
"#;
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("dataset.yaml");
        std::fs::File::create(&p).unwrap().write_all(yaml.as_bytes()).unwrap();

        let mut cfg = DatasetConfig::load(&p).expect("load full config");
        // Sanity: strata moved into deferred_sized via the migration.
        assert_eq!(cfg.strata.len(), 4, "root strata parsed");
        assert!(cfg.profiles.has_deferred(),
            "root strata must populate deferred_sized via the migration; \
             saw raw_sized={:?}, deferred_sized={:?}",
            cfg.profiles.raw_sized, cfg.profiles.deferred_sized);

        let vars = IndexMap::new();
        let added = cfg.profiles.expand_deferred_sized(&vars, 963_689_881);
        assert!(added > 0, "expansion should add profiles");

        // The decade strategy contributes these UNIQUE names (not
        // produced by mul/fib/linear at all):
        //   100k, 200k, …, 900k          (9 entries)
        //   4m, 6m, 7m, 9m                (4 entries — fib has 1,2,3,5,8 only)
        //   plus 100m, 200m, …, 900m share with linear/fib but pass
        for name in &[
            "100k", "200k", "300k", "400k", "500k",
            "600k", "700k", "800k", "900k",
            "4m", "6m", "7m", "9m",
        ] {
            assert!(
                cfg.profiles.profile(name).is_some(),
                "decade-specific profile '{}' missing after expansion. \
                 deferred entries were: {:?}, profile names present: {:?}",
                name,
                cfg.strata,
                cfg.profiles.profiles.keys().collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn decade_via_strata_spec_in_yaml() {
        // The deserializer must accept `decade` alongside the legacy
        // strategies in a `strata:` (or `sized:`) list and produce
        // sized profiles with windowed `base_vectors`.
        let yaml = r#"
default:
  base_vectors: profiles/base/base_vectors.fvecs
sized: ["decade:5..50"]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // First sized profile should be "5".
        let p5 = g.profile("5").expect("decade emits profile '5'");
        assert_eq!(p5.base_count, Some(5));
        let bv = p5.view("base_vectors").unwrap();
        assert_eq!(bv.source.window.0.first().map(|i| i.max_excl), Some(5));
        // And the highest in the sweep should be 50 (since explicit end is inclusive of last decade detent).
        assert!(g.profile("50").is_some());
    }

    #[test]
    fn sized_profile_via_strata_spec_inherits_with_window() {
        // The other code path: profiles generated by `sized: [...]`
        // (now `strata: [...]` at root, but the deserializer accepts
        // both placements). Same windowing rule must apply.
        let yaml = r#"
default:
  base_vectors: profiles/base/base_vectors.fvecs
sized: ["1m", "2m"]
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        for (name, expected_bc) in [("1m", 1_000_000u64), ("2m", 2_000_000u64)] {
            let p = g.profile(name).unwrap_or_else(|| panic!("expected sized profile '{}'", name));
            let bv = p.view("base_vectors")
                .unwrap_or_else(|| panic!("'{}' must inherit base_vectors with a window", name));
            assert_eq!(
                bv.source.window.0.first().map(|i| i.max_excl),
                Some(expected_bc),
                "profile '{}' base_vectors window should end at {}",
                name, expected_bc,
            );
        }
    }
}
