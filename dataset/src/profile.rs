// Copyright (c) DataStax, Inc.
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
//!     query_vectors: query_vectors.hvec
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

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde::de::{self, Visitor};

use crate::facet::resolve_standard_key;
use crate::source::{DSSource, DSWindow, format_count_with_suffix, parse_number_with_suffix, parse_source_string};

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
        if self.maxk.is_some() {
            len += 1;
        }
        if self.base_count.is_some() {
            len += 1;
        }
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

    /// Returns the view names in this profile.
    pub fn view_names(&self) -> Vec<&str> {
        self.views.keys().map(|k| k.as_str()).collect()
    }
}

/// Collection of named profiles with default-inheritance.
///
/// When deserialized, profiles named other than `"default"` automatically
/// inherit views and `maxk` from the default profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct DSProfileGroup(pub IndexMap<String, DSProfile>);

impl DSProfileGroup {
    /// Returns the default profile, if one exists.
    pub fn default_profile(&self) -> Option<&DSProfile> {
        self.0.get("default")
    }

    /// Look up a profile by name.
    pub fn profile(&self, name: &str) -> Option<&DSProfile> {
        self.0.get(name)
    }

    /// List all profile names.
    pub fn profile_names(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }

    /// Returns view names from the default profile (for display).
    pub fn view_names(&self) -> Vec<&str> {
        self.default_profile()
            .map(|p| p.view_names())
            .unwrap_or_default()
    }

    /// Returns `true` if there are no profiles.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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
                    serde_yaml::Value::String(s) => crate::source::parse_number_with_suffix(s).ok(),
                    _ => None,
                };
                continue;
            }

            // Resolve alias to canonical name; custom facets pass through as-is
            let canonical = resolve_standard_key(&key).unwrap_or_else(|| key.clone());

            let view: DSView = serde_yaml::from_value(value).map_err(de::Error::custom)?;
            views.insert(canonical, view);
        }

        Ok(DSProfile { maxk, base_count, views })
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
fn parse_sized_entry(entry: &str) -> Result<Vec<(String, u64)>, String> {
    let entry = entry.trim();

    // Check for series generator prefixes
    if let Some(rest) = entry.strip_prefix("fib:") {
        return parse_fib_range(rest);
    }
    if let Some(rest) = entry.strip_prefix("mul:") {
        return parse_mul_range(rest);
    }

    if let Some((range_part, step_str)) = entry.split_once('/') {
        let (start_str, end_str) = range_part
            .split_once("..")
            .ok_or_else(|| format!("invalid sized range '{}': expected start..end/step", entry))?;
        let start = parse_number_with_suffix(start_str.trim())?;
        let end = parse_number_with_suffix(end_str.trim())?;
        if start > end {
            return Err(format!("sized range start > end in '{}'", entry));
        }

        let step_trimmed = step_str.trim();
        let has_suffix = step_trimmed.bytes().any(|b| b.is_ascii_alphabetic());

        if has_suffix {
            // Step-size mode: start..end/step_size
            let step = parse_number_with_suffix(step_trimmed)?;
            if step == 0 {
                return Err(format!("sized range step must be > 0 in '{}'", entry));
            }
            let mut result = Vec::new();
            let mut v = start;
            while v <= end {
                if v > 0 {
                    result.push((format_count_with_suffix(v), v));
                }
                v += step;
            }
            Ok(result)
        } else {
            // Count mode: start..end/N — divide into N equal parts
            let count: u64 = step_trimmed.parse()
                .map_err(|e| format!("invalid division count '{}': {}", step_trimmed, e))?;
            if count == 0 {
                return Err(format!("sized range division count must be > 0 in '{}'", entry));
            }
            let range = end - start;
            let mut result = Vec::new();
            for i in 1..=count {
                let val = start + (range * i) / count;
                if val > 0 {
                    result.push((format_count_with_suffix(val), val));
                }
            }
            // Deduplicate (possible with small ranges and large counts)
            result.dedup_by(|a, b| a.1 == b.1);
            Ok(result)
        }
    } else {
        // Simple value
        let count = parse_number_with_suffix(entry)?;
        Ok(vec![(format_count_with_suffix(count), count)])
    }
}

/// Generate fibonacci-progression values within [start, end].
///
/// `fib:1m..400m` produces all fibonacci multiples of the base unit that
/// fall within the range. The base unit is the GCD-friendly smallest value;
/// the series starts at 1×base and grows by fibonacci steps.
fn parse_fib_range(range_str: &str) -> Result<Vec<(String, u64)>, String> {
    let (start_str, end_str) = range_str
        .split_once("..")
        .ok_or_else(|| format!("invalid fib range '{}': expected fib:start..end", range_str))?;
    let start = parse_number_with_suffix(start_str.trim())?;
    let end = parse_number_with_suffix(end_str.trim())?;
    if start == 0 || start > end {
        return Err(format!("fib range requires 0 < start <= end, got {}..{}", start, end));
    }

    // Use start as the base unit; generate fib(n) * start within range
    let mut result = Vec::new();
    let (mut a, mut b): (u64, u64) = (1, 1);
    loop {
        let val = a.checked_mul(start).unwrap_or(u64::MAX);
        if val > end {
            break;
        }
        if val >= start {
            result.push((format_count_with_suffix(val), val));
        }
        let next = a.saturating_add(b);
        a = b;
        b = next;
    }
    Ok(result)
}

/// Generate geometric (compound-by-factor) values within [start, end].
///
/// `mul:1m..400m/2` produces: 1m, 2m, 4m, 8m, 16m, 32m, 64m, 128m, 256m
/// `mul:1m..100m/1.5` produces: 1m, 1500k (rounded), etc.
///
/// The factor must be > 1.0. Each successive value is `floor(prev × factor)`.
fn parse_mul_range(spec: &str) -> Result<Vec<(String, u64)>, String> {
    let (range_part, factor_str) = spec
        .split_once('/')
        .ok_or_else(|| format!("invalid mul spec '{}': expected mul:start..end/factor", spec))?;
    let (start_str, end_str) = range_part
        .split_once("..")
        .ok_or_else(|| format!("invalid mul range '{}': expected start..end", range_part))?;
    let start = parse_number_with_suffix(start_str.trim())?;
    let end = parse_number_with_suffix(end_str.trim())?;
    let factor: f64 = factor_str.trim().parse()
        .map_err(|e| format!("invalid mul factor '{}': {}", factor_str, e))?;
    if factor <= 1.0 {
        return Err(format!("mul factor must be > 1.0, got {}", factor));
    }
    if start == 0 || start > end {
        return Err(format!("mul range requires 0 < start <= end, got {}..{}", start, end));
    }

    let mut result = Vec::new();
    let mut val_f = start as f64;
    loop {
        let val = val_f as u64;
        if val > end {
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
                //        base_vectors: "base_vectors.hvec:${range}"
                //        query_vectors: query_vectors.hvec
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

                let mut all_pairs: Vec<(String, u64)> = Vec::new();
                for entry_str in &entries {
                    let pairs = parse_sized_entry(entry_str)
                        .map_err(de::Error::custom)?;
                    all_pairs.extend(pairs);
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

                    // Inherit from default, then overlay scaffold views
                    let merged = if let Some(ref dp) = default_profile {
                        let mut merged_views = dp.views.clone();
                        for (k, v) in base_views {
                            merged_views.insert(k, v);
                        }
                        DSProfile {
                            maxk: dp.maxk,
                            base_count: Some(count),
                            views: merged_views,
                        }
                    } else {
                        DSProfile {
                            maxk: None,
                            base_count: Some(count),
                            views: base_views,
                        }
                    };
                    profiles.insert(prof_name, merged);
                }
                continue;
            }

            // Parse child profile
            let child: DSProfile =
                serde_yaml::from_value(value.clone()).map_err(de::Error::custom)?;

            // Inherit from default
            let merged = if let Some(ref dp) = default_profile {
                let mut merged_views = dp.views.clone();
                // Overlay child views on top of default
                for (k, v) in child.views {
                    merged_views.insert(k, v);
                }
                DSProfile {
                    maxk: child.maxk.or(dp.maxk),
                    base_count: child.base_count, // never inherit base_count
                    views: merged_views,
                }
            } else {
                child
            };

            profiles.insert(name.clone(), merged);
        }

        Ok(DSProfileGroup(profiles))
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
sketch_vectors: sketch.hvec
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
        assert_eq!(g.0.len(), 1);
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
        assert_eq!(g.0.len(), 2);

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
        assert_eq!(g.0.len(), 1);
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
        assert_eq!(g.0.len(), 4);

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
        assert_eq!(g.0.len(), 5);
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
        assert_eq!(g.0.len(), 6);
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
        assert_eq!(g.0.len(), 3);
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
        assert_eq!(g.0.len(), 2);
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
        assert_eq!(g.0.len(), 4);
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
  query_vectors: query_vectors.hvec
sized:
  ranges: ["10m", "20m"]
  facets:
    base_vectors: "base_vectors.hvec:${range}"
    metadata_content: "metadata.slab:${range}"
    neighbor_indices: "profiles/${profile}/neighbor_indices.ivec"
"#;
        let g: DSProfileGroup = serde_yaml::from_str(yaml).unwrap();
        // default + 2 sized profiles
        assert_eq!(g.0.len(), 3);

        let p10 = g.profile("10m").unwrap();
        assert_eq!(p10.base_count, Some(10_000_000));
        assert_eq!(p10.maxk, Some(100)); // inherited

        // base_vectors should have path + window from ${range}
        let bv = p10.view("base_vectors").unwrap();
        assert_eq!(bv.source.path, "base_vectors.hvec");
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
        assert_eq!(g.0.len(), 3);
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
}
