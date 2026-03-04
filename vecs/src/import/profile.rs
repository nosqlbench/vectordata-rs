// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset profiles — named configurations of data views with inheritance.
//!
//! A profile groups data views (keyed by canonical facet names) together with
//! optional metadata like `maxk`. Profiles support inheritance: when a
//! `"default"` profile exists, all other profiles automatically inherit its
//! views and `maxk` unless explicitly overridden.
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
//!   1M:
//!     base:
//!       source: base_vectors.fvec      # map form
//!       window: [0..1000000]
//!     indices: gnd/idx_1M.ivecs
//!     distances: gnd/dis_1M.fvecs
//! ```

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde::de::{self, Visitor};

use super::facet::Facet;
use super::source::{DSSource, DSWindow};

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
    /// Views keyed by canonical facet name.
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
        let mut map = serializer.serialize_map(Some(len))?;
        if let Some(maxk) = self.maxk {
            map.serialize_entry("maxk", &maxk)?;
        }
        for (key, view) in &self.views {
            map.serialize_entry(key, view)?;
        }
        map.end()
    }
}

impl DSProfile {
    /// Get a view by canonical facet name.
    pub fn view(&self, name: &str) -> Option<&DSView> {
        self.views.get(name)
    }

    /// Returns the canonical view names in this profile.
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

    /// Returns canonical view names from the default profile (for display).
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
// Alias resolution
// ---------------------------------------------------------------------------

/// Resolve a view key to its canonical facet name.
///
/// Tries `Facet::from_key` first (exact match), then `Facet::from_alias`
/// (shorthand names). Returns the canonical key string or `None`.
fn resolve_view_key(key: &str) -> Option<String> {
    if let Some(f) = Facet::from_key(key) {
        return Some(f.key().to_string());
    }
    if let Some(f) = Facet::from_alias(key) {
        return Some(f.key().to_string());
    }
    None
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
                let source = super::source::parse_source_string(v).map_err(de::Error::custom)?;
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

            // Resolve alias to canonical name
            let canonical = resolve_view_key(&key).unwrap_or_else(|| key.clone());

            let view: DSView = serde_yaml::from_value(value).map_err(de::Error::custom)?;
            views.insert(canonical, view);
        }

        Ok(DSProfile { maxk, views })
    }
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
}
