// Copyright 2020-2025 The original authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Conformance enforcement for `dataset.yaml`.
//!
//! The facet spec in [`crate::dataset::facet`] is the single authority for
//! *which resources a facet may own* (basenames, formats/extensions, and
//! namespaces). This module checks that a loaded [`DatasetConfig`] adheres
//! to that spec: every profile/facet/resource is expressed consistently, so
//! that — given a facet — one can always tell which files or namespaces it
//! may contain, and given a resource, which facet it belongs to.
//!
//! ## Enforcement posture
//! Conformance is a **check-time** gate (the `check` / pipeline-build path),
//! not a **load-time** one. Loading stays lenient so a partially-built
//! dataset mid-pipeline (a facet declared but not yet produced) does not
//! error; the strict gate runs once the dataset is meant to be complete.
//!
//! ## What is validated
//! For every view whose key resolves to a [`StandardFacet`], the declared
//! resource's **format** (derived from its file extension) must be one the
//! facet accepts. Views whose key is not a standard facet are left alone —
//! they are custom/forward-compatible and outside the spec's authority.
//! Templated or synthetic locators (containing `${…}`, or with no file
//! extension) are skipped because they cannot be classified statically.

use crate::dataset::config::DatasetConfig;
use crate::dataset::facet::{FacetFormat, StandardFacet};

/// A single way in which a `dataset.yaml` deviates from the facet spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetViolation {
    /// Profile in which the offending view was declared.
    pub profile: String,
    /// The view key (as written in the YAML).
    pub key: String,
    /// The declared resource locator (`path` or `path#namespace`).
    pub path: String,
    /// Human-readable explanation of the deviation.
    pub detail: String,
}

impl std::fmt::Display for FacetViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "profile '{}': view '{}' ('{}') — {}",
            self.profile, self.key, self.path, self.detail
        )
    }
}

/// Extract the classifiable file extension from a view locator, stripping a
/// `#namespace` suffix and any `[..]` window notation. Returns `None` when
/// the locator is templated/synthetic or carries no extension to classify.
fn classifiable_extension(locator: &str) -> Option<&str> {
    let file = locator.split('#').next().unwrap_or(locator);
    let file = file.split('[').next().unwrap_or(file);
    if file.contains('$') || file.contains(':') {
        // `${var}` templates and `kind:spec` synthetic sources are resolved
        // elsewhere; they have no statically classifiable on-disk format.
        return None;
    }
    let (_base, ext) = file.rsplit_once('.')?;
    if ext.is_empty() || ext.contains('/') {
        return None;
    }
    Some(ext)
}

/// Validate that every profile/facet/resource in `cfg` conforms to the
/// standardized facet spec.
///
/// Returns `Ok(())` when the dataset is fully conformant, or `Err` with one
/// [`FacetViolation`] per deviation (collected, not fail-fast, so a single
/// pass surfaces every problem).
pub fn validate_conformance(cfg: &DatasetConfig) -> Result<(), Vec<FacetViolation>> {
    let mut violations = Vec::new();

    for (profile_name, profile) in &cfg.profiles.profiles {
        for (key, view) in profile.views() {
            // Only standard facets fall under the spec's authority. A view
            // key that is neither a canonical name nor a known alias is a
            // custom view and is not constrained here.
            let Some(facet) = StandardFacet::from_key(key).or_else(|| StandardFacet::from_alias(key))
            else {
                continue;
            };

            // A series of one shard must be spelled as a single file,
            // so that readers predating multi-file facets can open it
            // (SH-4, SH-72). Readers accept the non-canonical form;
            // reporting it is this function's job.
            if view.is_one_shard_series() {
                violations.push(FacetViolation {
                    profile: profile_name.clone(),
                    key: key.to_string(),
                    path: view.path().to_string(),
                    detail: "a series of one shard must be spelled as a single file, so \
                             that readers predating multi-file facets can open it"
                        .to_string(),
                });
            }

            // Every shard is checked, not just the first: a series whose
            // third file is the wrong format is as non-conformant as one
            // whose first is (SH-42, SH-43).
            for locator in view.sources().iter().map(|s| s.path.as_str()) {
            let Some(ext) = classifiable_extension(locator) else {
                continue;
            };

            match FacetFormat::from_extension(ext) {
                // Recognized format that the facet accepts: conformant.
                Some(fmt) if facet.accepts_format(fmt) => {}
                // Recognized format the facet does *not* permit.
                Some(fmt) => violations.push(FacetViolation {
                    profile: profile_name.clone(),
                    key: key.to_string(),
                    path: locator.to_string(),
                    detail: format!(
                        "format {:?} (.{}) is not valid for facet '{}'; permitted formats: {:?}",
                        fmt,
                        ext,
                        facet.key(),
                        facet.formats(),
                    ),
                }),
                // Extension not recognized as any facet format at all.
                None => violations.push(FacetViolation {
                    profile: profile_name.clone(),
                    key: key.to_string(),
                    path: locator.to_string(),
                    detail: format!(
                        "extension '.{}' is not a recognized facet format for facet '{}'",
                        ext,
                        facet.key(),
                    ),
                }),
            }
            }
        }
    }

    violations.extend(family_violations(cfg));
    violations.extend(inheritance_violations(cfg));

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Ways in which the members of one family disagree about what they are
/// parameterizations *of* (P-11).
///
/// A family is one corpus at several parameter values. Members must
/// therefore agree on their invariants — the same `base_count`, the
/// same shared facets — and differ only in what the generator varies. A
/// member that quietly points at a different `base_vectors` is not a
/// parameterization of the same corpus, and a benchmark comparing
/// results across it is comparing two datasets while reporting one
/// number.
///
/// Reported against the **first** member, which is the generator's own
/// order, so a sweep that drifted partway through names the point it
/// drifted rather than the whole family.
fn family_violations(cfg: &DatasetConfig) -> Vec<FacetViolation> {
    let mut out = Vec::new();
    for (spec, members) in cfg.profiles.families() {
        let Some(first) = members.first() else { continue };
        let Some(anchor) = cfg.profiles.profiles.get(first) else {
            continue;
        };
        for name in members.iter().skip(1) {
            let Some(profile) = cfg.profiles.profiles.get(name) else {
                out.push(FacetViolation {
                    profile: name.clone(),
                    key: "<family>".to_string(),
                    path: spec.to_string(),
                    detail: format!(
                        "spec '{spec}' names this profile as a family member, but the                          dataset declares no such profile"
                    ),
                });
                continue;
            };
            // A size family varies `base_count` by construction, so it
            // is an invariant only where the generator holds it fixed.
            // Comparing the *shared* facets is what holds in either
            // case: a member reading different base bytes is a
            // different corpus whatever the axis.
            for key in ["base_vectors", "query_vectors", "metadata_content"] {
                let (Some(a), Some(b)) = (anchor.view(key), profile.view(key)) else {
                    continue;
                };
                // Windows differ legitimately across a size family —
                // that *is* the parameter. The file must not.
                if a.sources().iter().map(|s| &s.path).collect::<Vec<_>>()
                    != b.sources().iter().map(|s| &s.path).collect::<Vec<_>>()
                {
                    out.push(FacetViolation {
                        profile: name.clone(),
                        key: key.to_string(),
                        path: b.path().to_string(),
                        detail: format!(
                            "family '{spec}' member differs from '{first}' in '{key}'                              ('{}'): members of one family are one corpus at different                              parameter values, so comparing results across them is only                              meaningful if they read the same base data",
                            a.path()
                        ),
                    });
                }
            }
        }
    }
    out
}

/// Profiles whose `inherits:` names something that cannot be resolved
/// (P-2).
///
/// The loader leaves such a profile with what it declared rather than
/// failing — the facets it does declare are still readable, and a load
/// failure would take a whole dataset out of reach over one profile.
/// That makes reporting it here the only place the mistake surfaces.
fn inheritance_violations(cfg: &DatasetConfig) -> Vec<FacetViolation> {
    let mut out = Vec::new();
    for (name, profile) in &cfg.profiles.profiles {
        let Some(parent) = profile.inherits.as_deref() else {
            continue;
        };
        let detail = if parent == name {
            Some("a profile cannot inherit from itself".to_string())
        } else if !cfg.profiles.profiles.contains_key(parent) {
            Some(format!(
                "inherits from '{parent}', which this dataset does not declare"
            ))
        } else if inherits_cycle(cfg, name) {
            Some(format!(
                "inherits from '{parent}' through a cycle, so neither profile                  resolves and both keep only what they declared"
            ))
        } else {
            None
        };
        if let Some(detail) = detail {
            out.push(FacetViolation {
                profile: name.clone(),
                key: "inherits".to_string(),
                path: parent.to_string(),
                detail,
            });
        }
    }
    out
}

/// Whether following `inherits:` from `start` returns to `start`.
fn inherits_cycle(cfg: &DatasetConfig, start: &str) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut at = start.to_string();
    while seen.insert(at.clone()) {
        let Some(profile) = cfg.profiles.profiles.get(&at) else {
            return false;
        };
        let Some(parent) = profile.inherits.clone() else {
            return false;
        };
        if parent == start {
            return true;
        }
        at = parent;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::config::DatasetConfig;

    /// A dataset whose facets all declare spec-valid formats passes.
    #[test]
    fn conformant_dataset_validates() {
        let yaml = r#"
name: ok
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvecs
    query_vectors: profiles/default/query_vectors.fvecs
    neighbor_indices: profiles/default/neighbor_indices.ivecs
    neighbor_distances: profiles/default/neighbor_distances.fvecs
    metadata_content: profiles/default/metadata_content.slab
    metadata_predicates: profiles/default/predicates.u8
    metadata_indices: profiles/default/metadata_indices.ivvecs
"#;
        let cfg: DatasetConfig = serde_yaml::from_str(yaml).expect("parse");
        assert_eq!(validate_conformance(&cfg), Ok(()));
    }

    /// A float-xvec resource declared under an integer-index facet is a
    /// violation; an unknown extension under a known facet is a violation;
    /// custom keys and templated locators are ignored.
    #[test]
    fn nonconformant_formats_are_reported() {
        let yaml = r#"
name: bad
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvecs
    neighbor_indices: profiles/default/neighbor_indices.fvecs
    metadata_content: profiles/default/metadata_content.zzz
    my_custom_view: profiles/default/whatever.bin
    query_vectors: ${query_path}
"#;
        let cfg: DatasetConfig = serde_yaml::from_str(yaml).expect("parse");
        let err = validate_conformance(&cfg).expect_err("should report violations");
        // Exactly two violations: the float-under-indices and the bad ext.
        assert_eq!(err.len(), 2, "violations: {err:?}");
        assert!(err.iter().any(|v| v.key == "neighbor_indices"));
        assert!(err.iter().any(|v| v.key == "metadata_content"));
        // Custom and templated views were skipped.
        assert!(!err.iter().any(|v| v.key == "my_custom_view"));
        assert!(!err.iter().any(|v| v.key == "query_vectors"));
    }

    /// The legacy `metadata_indices` alias resolves to `metadata_results`
    /// and validates against that facet's permitted formats.
    #[test]
    fn legacy_alias_validates_against_canonical_facet() {
        let yaml = r#"
name: alias
profiles:
  default:
    metadata_indices: profiles/default/metadata_indices.ivvecs
"#;
        let cfg: DatasetConfig = serde_yaml::from_str(yaml).expect("parse");
        assert_eq!(validate_conformance(&cfg), Ok(()));
    }
}

#[cfg(test)]
mod series_conformance {
    use super::*;
    use crate::dataset::DatasetConfig;

    fn check(yaml: &str) -> Result<(), Vec<FacetViolation>> {
        // This `DatasetConfig` is the catalog-side model, whose views are
        // `DSView` — the same shape the catalog path realizes through.
        let cfg: DatasetConfig =
            serde_yaml::from_str(&format!("name: t\n{yaml}")).expect("loads");
        validate_conformance(&cfg)
    }

    /// A conformant series draws no complaint.
    #[test]
    fn a_conformant_series_validates() {
        assert!(
            check(
                "profiles:\n  default:\n    base_vectors:\n      source: base__NNNN.fvec\n\
                 \x20     shard_stride: 100\n      shard_count: 3\n      record_count: 250\n"
            )
            .is_ok()
        );
    }

    /// **Every shard is checked, not just the first** (SH-42, SH-43).
    /// A series whose third file is the wrong format is as
    /// non-conformant as one whose first is.
    #[test]
    fn a_wrong_format_in_a_later_shard_is_reported() {
        let v = check(
            "profiles:\n  default:\n    base_vectors:\n      source:\n        \
             - a.fvec=10\n        - b.fvec=10\n        - c.parquet=10\n      record_count: 30\n",
        )
        .expect_err("a non-conformant shard must be reported");
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].path.contains("c.parquet"), "names the shard: {:?}", v[0]);
    }

    /// **A one-shard series is reported** — readers accept it, but it
    /// must be spelled as a single file so pre-sharding readers can open
    /// it (SH-4, SH-72).
    #[test]
    fn a_one_shard_series_is_reported_as_non_canonical() {
        let v = check(
            "profiles:\n  default:\n    base_vectors:\n      source:\n        \
             - only.fvec=10\n      record_count: 10\n",
        )
        .expect_err("a one-shard series is not canonical");
        assert!(v.iter().any(|x| x.detail.contains("single file")), "{v:?}");
    }

    /// A sharded filename classifies to its facet, so a conformant
    /// series is not reported merely for being sharded (SH-6).
    #[test]
    fn a_sharded_filename_is_not_mistaken_for_an_unknown_facet() {
        assert!(
            check(
                "profiles:\n  default:\n    metadata_results:\n      source:\n        \
                 - metadata_results__0000.ivvec=5\n        - metadata_results__0001.ivvec=5\n\
                 \x20     record_count: 10\n"
            )
            .is_ok()
        );
    }
}
