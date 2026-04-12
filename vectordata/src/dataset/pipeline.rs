// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline configuration for the `upstream` block in `dataset.yaml`.
//!
//! Defines [`PipelineConfig`], [`StepDef`], and [`OnPartial`] — the schema
//! types for multi-step dataset build pipelines. These describe step
//! definitions, shared defaults, ordering constraints, and partial-output
//! behavior without knowing anything about the commands that execute those
//! steps.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Top-level pipeline configuration, embedded as `upstream` in the dataset YAML.
///
/// ```yaml
/// upstream:
///   defaults:
///     seed: 42
///     threads: 8
///   steps:
///     - id: shuffle
///       run: generate ivec-shuffle
///       interval: 0..1000000
///       output: shuffle.ivec
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Shared default variables for `${name}` interpolation in step options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defaults: Option<IndexMap<String, String>>,

    /// Ordered list of shared pipeline steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<StepDef>>,
}

/// A single step in a command stream pipeline.
///
/// ```yaml
/// - id: shuffle
///   run: generate ivec-shuffle
///   description: Create a random permutation for base vector extraction
///   after:
///     - download
///   interval: 0..1000000
///   seed: ${seed}
///   output: shuffle.ivec
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDef {
    /// Optional explicit step identifier. Auto-derived from `run` if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Command to execute, e.g. `"generate ivec-shuffle"` or `"import"`.
    pub run: String,

    /// Optional human-readable description of what this step does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Steps that must complete before this one (explicit ordering).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub after: Vec<String>,

    /// Optional profile gate: when present, this step only runs if the active
    /// profile name is in the list. Steps with no `profiles` field are shared
    /// and always run regardless of the active profile.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub profiles: Vec<String>,

    /// When true, this step is a template that gets expanded once per sized
    /// profile (profiles with `base_count`). The expansion produces concrete
    /// steps with profile-suffixed IDs, profile-gated `profiles` fields, and
    /// resolved `${profile_dir}`, `${base_count}`, `${base_end}` variables.
    /// Template steps are removed from the final step list after expansion.
    #[serde(default, skip_serializing_if = "is_false")]
    pub per_profile: bool,

    /// Execution phase for profile ordering. Per-profile steps are grouped
    /// by phase: all steps in phase 0 (default) run across all profiles
    /// before any phase 1 steps begin. This prevents I/O thrashing between
    /// compute-heavy steps (phase 0) and verification steps (phase 1).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub phase: u32,

    /// When true, this step runs in the finalization pass — after all
    /// compute phases (core, deferred sized, partition expansion) have
    /// completed. Finalization steps are held out of the Phase 1/2/3
    /// DAGs and executed once at the end, ensuring they see the full
    /// set of profiles and artifacts. Typical finalization steps:
    /// generate-dataset-json, generate-variables-json, generate-merkle,
    /// generate-catalog.
    #[serde(default, skip_serializing_if = "is_false")]
    pub finalize: bool,

    /// What to do when the output artifact is partially complete.
    #[serde(default, skip_serializing_if = "OnPartial::is_restart")]
    pub on_partial: OnPartial,

    /// All other key-value pairs are treated as command options.
    /// These are passed to the `CommandOp` after variable interpolation.
    #[serde(flatten)]
    pub options: IndexMap<String, serde_yaml::Value>,
}

impl StepDef {
    /// Resolve the effective step ID: explicit `id` field, or derived from `run`.
    ///
    /// Derivation replaces spaces with hyphens and lowercases:
    /// `"generate ivec-shuffle"` -> `"generate-ivec-shuffle"`.
    pub fn effective_id(&self) -> String {
        if let Some(ref id) = self.id {
            id.clone()
        } else {
            self.run.to_lowercase().replace(' ', "-")
        }
    }

    /// Extract the `output` option value if present.
    pub fn output_path(&self) -> Option<String> {
        self.options.get("output").and_then(|v| match v {
            serde_yaml::Value::String(s) => Some(s.clone()),
            _ => Some(format!("{:?}", v)),
        })
    }

    /// Return all option values that look like output file paths.
    ///
    /// Checks `output`, `indices`, and `distances` keys — the known set of
    /// option names that commands use for files they write. Used by
    /// `expand_per_profile_steps()` to auto-prefix bare filenames with the
    /// profile directory.
    pub fn output_paths(&self) -> Vec<String> {
        const OUTPUT_KEYS: &[&str] = &["output", "indices", "distances"];
        OUTPUT_KEYS
            .iter()
            .filter_map(|key| {
                self.options.get(*key).and_then(|v| match v {
                    serde_yaml::Value::String(s) => Some(s.clone()),
                    _ => None,
                })
            })
            .collect()
    }
}

/// Behavior when an output artifact is partially complete.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnPartial {
    /// Delete partial output and restart from scratch.
    #[default]
    Restart,
    /// Attempt to resume from where the previous run left off.
    Resume,
}

impl OnPartial {
    fn is_restart(&self) -> bool {
        *self == OnPartial::Restart
    }
}

fn is_false(v: &bool) -> bool {
    !v
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_effective_id_explicit() {
        let step = StepDef {
            id: Some("my-step".to_string()),
            run: "generate vectors".to_string(),
            description: None,
            after: vec![],
            profiles: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            on_partial: OnPartial::default(),
            options: IndexMap::new(),
        };
        assert_eq!(step.effective_id(), "my-step");
    }

    #[test]
    fn test_step_effective_id_derived() {
        let step = StepDef {
            id: None,
            run: "generate ivec-shuffle".to_string(),
            description: None,
            after: vec![],
            profiles: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            on_partial: OnPartial::default(),
            options: IndexMap::new(),
        };
        assert_eq!(step.effective_id(), "generate-ivec-shuffle");
    }

    #[test]
    fn test_on_partial_default() {
        assert_eq!(OnPartial::default(), OnPartial::Restart);
    }

    #[test]
    fn test_pipeline_config_roundtrip() {
        let yaml = r#"
defaults:
  seed: "42"
  threads: "8"
steps:
  - run: generate ivec-shuffle
    output: shuffle.ivec
"#;
        let config: PipelineConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.defaults.is_some());
        let defaults = config.defaults.as_ref().unwrap();
        assert_eq!(defaults.get("seed").unwrap(), "42");

        let steps = config.steps.as_ref().unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].run, "generate ivec-shuffle");
        assert_eq!(steps[0].output_path(), Some("shuffle.ivec".to_string()));

        // Roundtrip
        let serialized = serde_yaml::to_string(&config).unwrap();
        let reparsed: PipelineConfig = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(
            reparsed.defaults.as_ref().unwrap().get("seed").unwrap(),
            "42"
        );
    }

    #[test]
    fn test_step_output_path() {
        let mut opts = IndexMap::new();
        opts.insert(
            "output".to_string(),
            serde_yaml::Value::String("result.fvec".to_string()),
        );
        let step = StepDef {
            id: None,
            run: "convert file".to_string(),
            description: None,
            after: vec![],
            profiles: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            on_partial: OnPartial::default(),
            options: opts,
        };
        assert_eq!(step.output_path(), Some("result.fvec".to_string()));
    }

    #[test]
    fn test_step_profiles_field_parsing() {
        let yaml = r#"
steps:
  - run: import
    output: all.mvec
  - run: compute knn
    profiles: [default]
    output: gnd.ivec
  - run: compute knn
    profiles: [10M, 100M]
    output: 10M/gnd.ivec
"#;
        let config: PipelineConfig = serde_yaml::from_str(yaml).unwrap();
        let steps = config.steps.unwrap();
        assert!(steps[0].profiles.is_empty());
        assert_eq!(steps[1].profiles, vec!["default"]);
        assert_eq!(steps[2].profiles, vec!["10M", "100M"]);
    }

    #[test]
    fn test_step_profiles_not_serialized_when_empty() {
        let step = StepDef {
            id: Some("s1".to_string()),
            run: "import".to_string(),
            description: None,
            after: vec![],
            profiles: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            on_partial: OnPartial::default(),
            options: IndexMap::new(),
        };
        let yaml = serde_yaml::to_string(&step).unwrap();
        assert!(!yaml.contains("profiles"));
    }

    #[test]
    fn test_step_profiles_roundtrip() {
        let step = StepDef {
            id: Some("s1".to_string()),
            run: "compute knn".to_string(),
            description: None,
            after: vec![],
            profiles: vec!["10M".to_string(), "100M".to_string()],
            per_profile: false,
            phase: 0,
            finalize: false,
            on_partial: OnPartial::default(),
            options: IndexMap::new(),
        };
        let yaml = serde_yaml::to_string(&step).unwrap();
        assert!(yaml.contains("profiles"));
        let reparsed: StepDef = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(reparsed.profiles, vec!["10M", "100M"]);
    }
}
