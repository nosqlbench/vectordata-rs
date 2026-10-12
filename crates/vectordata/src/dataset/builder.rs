// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Building a dataset description in code.
//!
//! [`DatasetConfig::builder`] is how a program, or one of the crate's
//! own generators, makes a `dataset.yaml` without writing YAML: name the
//! dataset, state its attributes, declare its profiles and their facets,
//! and [`build`](DatasetBuilder::build). The result passes the checks a
//! parsed file passes (version gate, stated parents, shard declarations,
//! inheritance) and is equal to the description parsed from the file a
//! [`save`](DatasetConfig::save) of it writes.
//!
//! ```
//! use vectordata::dataset::{DatasetConfig, StandardFacet};
//!
//! let config = DatasetConfig::builder("demo-10k")
//!     .distance_function("COSINE")
//!     .profile("default", |p| p
//!         .maxk(100)
//!         .facet(StandardFacet::BaseVectors, "base.fvec")
//!         .facet(StandardFacet::QueryVectors, "query.fvec")
//!         .facet(StandardFacet::NeighborIndices, "gt.ivec"))
//!     .profile("1k", |p| p.base_count(1000).facet(StandardFacet::NeighborIndices, "profiles/1k/gt.ivec"))
//!     .build()
//!     .unwrap();
//! // The sized profile inherits the base, cut to its count, and the queries.
//! let small = &config.profiles.profiles["1k"];
//! assert_eq!(small.views["base_vectors"].effective_window().to_string(), "[0..1000]");
//! assert_eq!(small.views["query_vectors"].source.path, "query.fvec");
//! ```

use indexmap::IndexMap;

use super::config::{DatasetAttributes, DatasetConfig};
use super::facet::StandardFacet;
use super::pipeline::PipelineConfig;
use super::profile::{DSProfile, DSProfileGroup, DSView};
use super::source::parse_source_string;
use super::strata::Strata;

impl DatasetConfig {
    /// Start building a description of the dataset called `name`.
    pub fn builder(name: impl Into<String>) -> DatasetBuilder {
        DatasetBuilder {
            name: name.into(),
            description: None,
            attributes: None,
            stated_version: None,
            profile_tags: IndexMap::new(),
            profiles: IndexMap::new(),
            strata: Vec::new(),
            variables: IndexMap::new(),
            upstream: None,
            header: Vec::new(),
            error: None,
        }
    }
}

/// A dataset description being built. See the [module docs](self).
///
/// Problems (a source string that does not parse, a duplicate profile)
/// are held until [`build`](Self::build), which reports the first, so a
/// chain of calls stays a chain.
#[derive(Debug, Clone)]
pub struct DatasetBuilder {
    name: String,
    description: Option<String>,
    attributes: Option<DatasetAttributes>,
    stated_version: Option<u32>,
    profile_tags: IndexMap<String, serde_yaml::Value>,
    profiles: IndexMap<String, DSProfile>,
    strata: Vec<String>,
    variables: IndexMap<String, String>,
    upstream: Option<PipelineConfig>,
    header: Vec<String>,
    error: Option<String>,
}

impl DatasetBuilder {
    /// A one-paragraph description of the dataset.
    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.description = Some(text.into());
        self
    }

    /// A dataset attribute. The keys the attributes type knows
    /// (`distance_function`, `model`, `license`, `is_normalized`, …) are
    /// stored in their fields; any other key is kept as given, as a
    /// parsed file keeps it.
    pub fn attribute(mut self, key: &str, value: impl Into<serde_yaml::Value>) -> Self {
        let value = value.into();
        let attributes = self.attributes.get_or_insert_with(DatasetAttributes::default);
        if !attributes.set_known(key, &value) {
            attributes.other.insert(key.to_string(), value);
        }
        self
    }

    /// The distance function (`COSINE`, `L2`, `DOT_PRODUCT`, …).
    pub fn distance_function(self, name: &str) -> Self {
        self.attribute("distance_function", name)
    }

    /// The embedding model that produced the vectors.
    pub fn model(self, name: &str) -> Self {
        self.attribute("model", name)
    }

    /// The dataset's license.
    pub fn license(self, name: &str) -> Self {
        self.attribute("license", name)
    }

    /// The format version to state. Without it the description states
    /// the lowest version its content needs (V-4); stating less than that
    /// is refused at build, as it is in a file (V-14).
    pub fn format_version(mut self, version: u32) -> Self {
        self.stated_version = Some(version);
        self
    }

    /// A naming tag of the profile tag schema (PS-19), in order, with its
    /// default, or `serde_yaml::Value::Null` (`~`) for none.
    pub fn profile_tag(mut self, key: &str, default: impl Into<serde_yaml::Value>) -> Self {
        self.profile_tags.insert(key.to_string(), default.into());
        self
    }

    /// A profile, declared by `declare`. What it does not declare it
    /// inherits from its parent at build, by the rule a parsed file
    /// follows.
    pub fn profile(mut self, name: &str, declare: impl FnOnce(ProfileBuilder) -> ProfileBuilder) -> Self {
        if self.profiles.contains_key(name) {
            self.error.get_or_insert_with(|| format!("profile '{name}' is declared twice"));
            return self;
        }
        let built = declare(ProfileBuilder { name: name.to_string(), profile: DSProfile::default(), error: None });
        if let Some(e) = built.error {
            self.error.get_or_insert(e);
        }
        self.profiles.insert(name.to_string(), built.profile);
        self
    }

    /// A sized-profile generator spec (`mul:1m/2`, `fib:1m..10m`, …),
    /// expanded against the base count like a file's `strata:`.
    pub fn stratum(mut self, spec: &str) -> Self {
        self.strata.push(spec.to_string());
        self
    }

    /// An entry of the `variables:` block.
    pub fn variable(mut self, key: &str, value: impl Into<String>) -> Self {
        self.variables.insert(key.to_string(), value.into());
        self
    }

    /// The pipeline that builds the dataset (`upstream:`).
    pub fn upstream(mut self, pipeline: PipelineConfig) -> Self {
        self.upstream = Some(pipeline);
        self
    }

    /// A comment line a new file starts with, without its `# `.
    pub fn header_line(mut self, line: impl Into<String>) -> Self {
        self.header.push(line.into());
        self
    }

    /// The description, checked as a parsed file is checked: the
    /// version gate, stated parents, shard declarations, and
    /// inheritance resolved.
    pub fn build(self) -> Result<DatasetConfig, String> {
        if let Some(e) = self.error {
            return Err(e);
        }
        let mut profiles = DSProfileGroup::from_profiles(self.profiles);
        profiles.settle()?;
        let required = super::config::required_format_version(&profiles, &self.profile_tags);
        let stated = self.stated_version.unwrap_or(required);
        let format_version = super::config::check_description(Some(stated), &self.profile_tags, &profiles)?;
        profiles.layered = format_version >= crate::model::FORMAT_VERSION_TAGGED;
        let mut config = DatasetConfig {
            format_version,
            profile_tags: self.profile_tags,
            name: self.name,
            description: self.description,
            attributes: self.attributes,
            upstream: self.upstream,
            strata: Strata::from_specs(self.strata),
            profiles,
            variables: self.variables,
            header: self.header,
        };
        config.unify_strata();
        Ok(config)
    }
}

/// One profile being declared. See [`DatasetBuilder::profile`].
#[derive(Debug, Clone)]
pub struct ProfileBuilder {
    name: String,
    profile: DSProfile,
    error: Option<String>,
}

impl ProfileBuilder {
    /// A standard facet, from a source string as a file writes it: a
    /// path or URL, optionally with a namespace and a window
    /// (`base.fvec[0..1000)`).
    pub fn facet(self, facet: StandardFacet, source: &str) -> Self {
        self.custom_facet(facet.key(), source)
    }

    /// A facet under a key of the dataset's own, from a source string.
    pub fn custom_facet(mut self, key: &str, source: &str) -> Self {
        match parse_source_string(source) {
            Ok(source) => {
                // A standard facet's alias is stored under its canonical
                // key, as a parse stores it.
                let key = super::facet::resolve_standard_key(key).unwrap_or_else(|| key.to_string());
                self.profile.views.insert(key, DSView { source, ..Default::default() });
            }
            Err(e) => {
                let profile = &self.name;
                self.error.get_or_insert_with(|| format!("profile '{profile}': {key}: {e}"));
            }
        }
        self
    }

    /// A facet given as a whole view: a series, or a view-level window.
    pub fn view(mut self, key: &str, view: DSView) -> Self {
        let key = super::facet::resolve_standard_key(key).unwrap_or_else(|| key.to_string());
        self.profile.views.insert(key, view);
        self
    }

    /// The number of base vectors this profile covers.
    pub fn base_count(mut self, count: u64) -> Self {
        self.profile.base_count = Some(count);
        self
    }

    /// The largest `k` its ground truth answers.
    pub fn maxk(mut self, k: u32) -> Self {
        self.profile.maxk = Some(k);
        self
    }

    /// The profile it builds on, other than `default` (PL-5).
    pub fn inherits(mut self, parent: &str) -> Self {
        self.profile.inherits = Some(parent.to_string());
        self
    }

    /// An oracle partition: its own base rows, inheriting nothing.
    pub fn partition(mut self) -> Self {
        self.profile.partition = true;
        self
    }

    /// A tag or attribute describing what the profile is (P-4).
    pub fn attribute(mut self, key: &str, value: impl Into<serde_yaml::Value>) -> Self {
        self.profile.attributes.insert(key.to_string(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A built description equals the one parsed from its save.** The
    /// builder and the parser are two ways into one type, so a program
    /// building in code gets exactly what the file would have given it.
    #[test]
    fn a_built_description_equals_the_parse_of_its_save() {
        let built = DatasetConfig::builder("demo")
            .description("A demo.")
            .distance_function("L2")
            .attribute("dimension", 128)
            .attribute("is_normalized", true)
            .format_version(3)
            .profile_tag("size", serde_yaml::Value::Null)
            .profile("default", |p| {
                p.maxk(10)
                    .facet(StandardFacet::BaseVectors, "profiles/base/base.fvec")
                    .facet(StandardFacet::QueryVectors, "profiles/base/q.fvec")
                    .facet(StandardFacet::NeighborIndices, "profiles/default/gt.ivec")
            })
            .profile("100", |p| p.inherits("default").base_count(100).attribute("size", "100")
                .facet(StandardFacet::NeighborIndices, "profiles/100/gt.ivec"))
            .profile("100-q2", |p| p.inherits("100").facet(StandardFacet::QueryVectors, "profiles/base/q2.fvec"))
            .stratum("mul:1k/2")
            .header_line("made by a test")
            .build()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dataset.yaml");
        built.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# made by a test\n"), "{text}");
        let parsed = DatasetConfig::load(&path).unwrap();
        assert_eq!(parsed.name, built.name);
        assert_eq!(parsed.format_version, built.format_version);
        assert_eq!(parsed.attributes, built.attributes);
        assert_eq!(parsed.profile_tags, built.profile_tags);
        assert_eq!(parsed.strata.specs().collect::<Vec<_>>(), built.strata.specs().collect::<Vec<_>>());
        for (name, p) in &built.profiles.profiles {
            let q = &parsed.profiles.profiles[name];
            assert_eq!((&q.views, q.base_count, q.maxk, &q.inherits), (&p.views, p.base_count, p.maxk, &p.inherits), "{name}:\n{text}");
        }
        // `100-q2` names `100` as its parent at one size, so the
        // unfiltered ground truth crosses (P-2).
        assert_eq!(built.profiles.profiles["100-q2"].views["neighbor_indices"].source.path, "profiles/100/gt.ivec");
    }

    /// **A builder refuses what a parse refuses**, with the same words.
    #[test]
    fn a_builder_refuses_what_a_parse_refuses() {
        let err = DatasetConfig::builder("ds")
            .format_version(1)
            .profile_tag("size", serde_yaml::Value::Null)
            .build()
            .unwrap_err();
        assert!(err.contains("declares format_version 1"), "{err}");
        let err = DatasetConfig::builder("ds")
            .profile("default", |p| p.facet(StandardFacet::BaseVectors, "b.fvec[0..x)"))
            .build()
            .unwrap_err();
        assert!(err.contains("profile 'default': base_vectors"), "{err}");
        let err = DatasetConfig::builder("ds")
            .profile("a", |p| p)
            .profile("a", |p| p)
            .build()
            .unwrap_err();
        assert!(err.contains("declared twice"), "{err}");
    }
}
