// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Named strata — sized-profile generator specs keyed by name.
//!
//! A *stratum* is one sized-profile generator spec (the
//! `mul:/fib:/linear:/decade` grammar parsed by
//! [`crate::dataset::profile::parse_sized_entry`]) together with the
//! *series* of profile names it produces once expanded against the
//! dataset's base count:
//!
//! ```yaml
//! strata:
//!   mul:
//!     spec: "mul:1mi/2"
//!     series: ["1mi", "2mi", "4mi", "8mi"]
//!   fib:
//!     spec: "fib:1m"
//!     series: ["1m", "2m", "3m", "5m", "8m"]
//! ```
//!
//! A profile may belong to multiple strata — overlapping series are
//! valid and expected (e.g. `mul:1m/2` and `fib:1m` both produce
//! `1m` and `2m`).
//!
//! Two input forms are accepted:
//!
//! 1. **Compact list** (authoring shorthand): `strata: ["mul:1mi/2", "fib:1m"]`.
//!    Stratum names are synthesized from each spec's generator strategy
//!    (`mul`, `fib`, `linear`, `decade`, `step`, `parts`), with `-2`,
//!    `-3`, … suffixes on collision.
//! 2. **Structured map** (canonical): names as keys, `spec` + optional
//!    `series` values. A bare string value is shorthand for `spec:`.
//!
//! Rendering (both `dataset.yaml` and `dataset.json`) always emits the
//! structured map form — the published artifact is self-describing.

use std::fmt;

use indexmap::IndexMap;
use serde::de::{self, Deserialize};
use serde::ser::{Serialize, SerializeMap};

/// One named sized-profile generator: the spec string plus the series
/// of profile names it produced (empty until expansion has run).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Stratum {
    /// Generator spec — `mul:1mi/2`, `fib:1m`, `decade`, `100m..400m/100m`, …
    pub spec: String,
    /// Profile names generated from `spec`, in generator order
    /// (ascending size). Empty until sized-profile expansion has run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub series: Vec<String>,
}

/// Ordered collection of named strata, keyed by stratum name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Strata(pub IndexMap<String, Stratum>);

impl Strata {
    /// Returns `true` when no strata are defined.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of strata.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterate `(name, stratum)` pairs in definition order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Stratum)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Look up a stratum by name.
    pub fn get(&self, name: &str) -> Option<&Stratum> {
        self.0.get(name)
    }

    /// Insert (or replace) a stratum under `name`.
    pub fn insert(&mut self, name: &str, stratum: Stratum) {
        self.0.insert(name.to_string(), stratum);
    }

    /// Iterate the generator specs in definition order.
    pub fn specs(&self) -> impl Iterator<Item = &str> {
        self.0.values().map(|s| s.spec.as_str())
    }

    /// Build named strata from bare generator specs (the compact list
    /// form). Names are synthesized via [`stratum_name_for_spec`], with
    /// `-2`, `-3`, … suffixes to keep them unique.
    pub fn from_specs<I>(specs: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        let mut strata = Strata::default();
        for spec in specs {
            strata.insert_spec(&spec.into(), Vec::new());
        }
        strata
    }

    /// Insert a stratum for `spec` under a synthesized, collision-free
    /// name (see [`stratum_name_for_spec`]); returns the chosen name.
    /// If a stratum with this exact spec already exists, its series is
    /// refreshed instead and its existing name returned.
    pub fn insert_spec(&mut self, spec: &str, series: Vec<String>) -> String {
        if let Some((name, stratum)) = self.0.iter_mut().find(|(_, s)| s.spec == spec) {
            if !series.is_empty() {
                stratum.series = series;
            }
            return name.clone();
        }
        let base = stratum_name_for_spec(spec);
        let mut name = base.clone();
        let mut n = 1usize;
        while self.0.contains_key(&name) {
            n += 1;
            name = format!("{}-{}", base, n);
        }
        self.0.insert(name.clone(), Stratum { spec: spec.to_string(), series });
        name
    }

    /// Refresh each stratum's `series` from the per-spec expansion
    /// record kept by the profile group
    /// ([`crate::dataset::profile::DSProfileGroup::series_by_spec`]).
    /// Strata whose spec has no expansion record keep their current
    /// series (e.g. values loaded from a published dataset.yaml).
    pub fn sync_series(&mut self, series_by_spec: &IndexMap<String, Vec<String>>) {
        for stratum in self.0.values_mut() {
            if let Some(names) = series_by_spec.get(&stratum.spec) {
                stratum.series = names.clone();
            }
        }
    }
}

/// Synthesize a stratum name from a generator spec.
///
/// Recognized strategy prefixes name the stratum after the strategy
/// (`mul`, `fib`, `linear`, `decade`, `step`, `parts`). Bare bounded
/// ranges (`100m..400m/100m`) resolve to `step` or `parts` by the same
/// divisor-suffix rule the parser uses. Anything else (e.g. a simple
/// value like `10m`) is sanitized into a key-safe slug of itself.
pub fn stratum_name_for_spec(spec: &str) -> String {
    let s = spec.trim();
    if let Some((prefix, _)) = s.split_once(':') {
        match prefix {
            "mul" | "fib" | "linear" | "decade" | "step" | "parts" => {
                return prefix.to_string();
            }
            _ => {}
        }
    }
    if s == "decade" {
        return "decade".to_string();
    }
    if s.contains("..") && let Some((_, divisor)) = s.split_once('/') {
        // Bare bounded range — mirror `parse_sized_entry_impl`'s
        // step-vs-parts inference from the divisor's suffix.
        return if divisor.trim().bytes().any(|b| b.is_ascii_alphabetic()) {
            "step".to_string()
        } else {
            "parts".to_string()
        };
    }
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

impl Serialize for Strata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (name, stratum) in &self.0 {
            map.serialize_entry(name, stratum)?;
        }
        map.end()
    }
}

/// A string that also tolerates numeric scalars — series entries like
/// `5` (from `decade:5..50`) parse as YAML integers when unquoted.
struct StringLike(String);

impl<'de> Deserialize<'de> for StringLike {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct V;
        impl de::Visitor<'_> for V {
            type Value = StringLike;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a string or number")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<StringLike, E> {
                Ok(StringLike(v.to_string()))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<StringLike, E> {
                Ok(StringLike(v.to_string()))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<StringLike, E> {
                Ok(StringLike(v.to_string()))
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl<'de> Deserialize<'de> for Stratum {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct V;
        impl<'de> de::Visitor<'de> for V {
            type Value = Stratum;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a spec string or a map with 'spec' and optional 'series'")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Stratum, E> {
                Ok(Stratum { spec: v.to_string(), series: Vec::new() })
            }

            fn visit_map<M>(self, mut map: M) -> Result<Stratum, M::Error>
            where
                M: de::MapAccess<'de>,
            {
                let mut spec: Option<String> = None;
                let mut series: Option<Vec<String>> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "spec" => spec = Some(map.next_value()?),
                        "series" => {
                            let names: Vec<StringLike> = map.next_value()?;
                            series = Some(names.into_iter().map(|s| s.0).collect());
                        }
                        other => {
                            return Err(de::Error::unknown_field(other, &["spec", "series"]));
                        }
                    }
                }
                Ok(Stratum {
                    spec: spec.ok_or_else(|| de::Error::missing_field("spec"))?,
                    series: series.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl<'de> Deserialize<'de> for Strata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct V;
        impl<'de> de::Visitor<'de> for V {
            type Value = Strata;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a list of generator specs or a map of named strata")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Strata, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut specs: Vec<String> = Vec::new();
                while let Some(spec) = seq.next_element::<String>()? {
                    specs.push(spec);
                }
                Ok(Strata::from_specs(specs))
            }

            fn visit_map<M>(self, mut map: M) -> Result<Strata, M::Error>
            where
                M: de::MapAccess<'de>,
            {
                let mut strata = Strata::default();
                while let Some((name, stratum)) = map.next_entry::<String, Stratum>()? {
                    strata.0.insert(name, stratum);
                }
                Ok(strata)
            }

            fn visit_unit<E: de::Error>(self) -> Result<Strata, E> {
                Ok(Strata::default())
            }

            fn visit_none<E: de::Error>(self) -> Result<Strata, E> {
                Ok(Strata::default())
            }
        }
        deserializer.deserialize_any(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_form_synthesizes_strategy_names() {
        let strata: Strata =
            serde_yaml::from_str(r#"["mul:1mi/2", "fib:1m", "linear:10m/10m", "decade"]"#)
                .unwrap();
        let names: Vec<&str> = strata.iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["mul", "fib", "linear", "decade"]);
        assert_eq!(strata.get("mul").unwrap().spec, "mul:1mi/2");
        assert!(strata.get("mul").unwrap().series.is_empty());
    }

    #[test]
    fn list_form_disambiguates_name_collisions() {
        let strata: Strata =
            serde_yaml::from_str(r#"["mul:1mi/2", "mul:1k..4k/2"]"#).unwrap();
        let names: Vec<&str> = strata.iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["mul", "mul-2"]);
        assert_eq!(strata.get("mul-2").unwrap().spec, "mul:1k..4k/2");
    }

    #[test]
    fn list_form_names_bare_ranges_and_values() {
        let strata: Strata =
            serde_yaml::from_str(r#"["100m..400m/100m", "1m..8m/4", "10m"]"#).unwrap();
        let names: Vec<&str> = strata.iter().map(|(n, _)| n).collect();
        // 100m step (alpha divisor suffix), 4 parts (numeric divisor),
        // simple value slug.
        assert_eq!(names, vec!["step", "parts", "10m"]);
    }

    #[test]
    fn map_form_parses_spec_and_series() {
        let strata: Strata = serde_yaml::from_str(
            r#"
binomial:
  spec: "mul:1mi/2"
  series: ["1mi", "2mi", "4mi"]
fib:
  spec: "fib:1m"
"#,
        )
        .unwrap();
        assert_eq!(strata.len(), 2);
        let binomial = strata.get("binomial").unwrap();
        assert_eq!(binomial.spec, "mul:1mi/2");
        assert_eq!(binomial.series, vec!["1mi", "2mi", "4mi"]);
        assert!(strata.get("fib").unwrap().series.is_empty());
    }

    #[test]
    fn map_form_accepts_bare_spec_shorthand_and_numeric_series() {
        let strata: Strata = serde_yaml::from_str(
            r#"
decade:
  spec: "decade:5..50"
  series: [5, 10, 20, 30, 40, 50]
mul: "mul:1mi/2"
"#,
        )
        .unwrap();
        assert_eq!(strata.get("mul").unwrap().spec, "mul:1mi/2");
        assert_eq!(
            strata.get("decade").unwrap().series,
            vec!["5", "10", "20", "30", "40", "50"],
        );
    }

    #[test]
    fn map_form_rejects_unknown_keys() {
        let err = serde_yaml::from_str::<Strata>(
            r#"
binomial:
  spec: "mul:1mi/2"
  serie: ["1mi"]
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("serie"), "got: {}", err);
    }

    #[test]
    fn serializes_to_structured_map_form() {
        let mut strata = Strata::default();
        strata.insert("mul", Stratum {
            spec: "mul:1mi/2".into(),
            series: vec!["1mi".into(), "2mi".into()],
        });
        strata.insert("fib", Stratum { spec: "fib:1m".into(), series: Vec::new() });

        let json = serde_json::to_string(&strata).unwrap();
        assert_eq!(
            json,
            r#"{"mul":{"spec":"mul:1mi/2","series":["1mi","2mi"]},"fib":{"spec":"fib:1m"}}"#,
        );

        // Round-trip through YAML preserves the structure.
        let yaml = serde_yaml::to_string(&strata).unwrap();
        let back: Strata = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back, strata);
    }

    #[test]
    fn sync_series_updates_only_recorded_specs() {
        let mut strata = Strata::from_specs(["mul:1mi/2".to_string(), "fib:1m".to_string()]);
        strata.0.get_mut("fib").unwrap().series = vec!["stale".into()];

        let mut by_spec: IndexMap<String, Vec<String>> = IndexMap::new();
        by_spec.insert("mul:1mi/2".into(), vec!["1mi".into(), "2mi".into()]);
        strata.sync_series(&by_spec);

        assert_eq!(strata.get("mul").unwrap().series, vec!["1mi", "2mi"]);
        // No record for fib's spec — loaded value is kept.
        assert_eq!(strata.get("fib").unwrap().series, vec!["stale"]);
    }
}
