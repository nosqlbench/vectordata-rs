// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema types for command stream pipeline configuration.
//!
//! Extends `dataset.yaml` with an optional `upstream` block at the top level
//! that can declare shared defaults and multi-step transformation pipelines.

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

    /// Command to execute, e.g. `"generate ivec-shuffle"` or `"import facet"`.
    pub run: String,

    /// Steps that must complete before this one (explicit ordering).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub after: Vec<String>,

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
    /// `"generate ivec-shuffle"` → `"generate-ivec-shuffle"`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_effective_id_explicit() {
        let step = StepDef {
            id: Some("my-step".to_string()),
            run: "generate vectors".to_string(),
            after: vec![],
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
            after: vec![],
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
            after: vec![],
            on_partial: OnPartial::default(),
            options: opts,
        };
        assert_eq!(step.output_path(), Some("result.fvec".to_string()));
    }

}
