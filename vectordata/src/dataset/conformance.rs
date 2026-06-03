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

            let locator = view.path();
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

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
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
