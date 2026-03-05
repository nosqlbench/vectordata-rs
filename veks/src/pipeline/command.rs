// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Core types for the command stream pipeline.
//!
//! Defines the `CommandOp` trait that all pipeline commands implement, along
//! with result types, option containers, and the execution context shared
//! across steps.

use std::path::{Path, PathBuf};
use std::time::Duration;

use indexmap::IndexMap;

use super::bound;
use super::progress::ProgressLog;

/// A single executable operation in a command stream.
///
/// Implementations wrap specific data-processing operations (import, convert,
/// generate, compute, etc.) behind a uniform interface so the pipeline runner
/// can orchestrate them generically.
pub trait CommandOp: Send {
    /// Canonical command path, e.g. `"import facet"` or `"generate vectors"`.
    fn command_path(&self) -> &str;

    /// Execute with resolved options.
    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult;

    /// Describe accepted options for dry-run validation.
    fn describe_options(&self) -> Vec<OptionDesc>;

    /// Check whether an existing output artifact is complete.
    ///
    /// The default delegates to the format-aware bound check in [`bound`].
    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState {
        bound::check_artifact_default(output, options)
    }
}

/// Outcome of executing a single pipeline step.
#[derive(Debug, Clone)]
pub struct CommandResult {
    /// Overall status of the step execution.
    pub status: Status,
    /// Human-readable summary message.
    pub message: String,
    /// Files produced by the step.
    pub produced: Vec<PathBuf>,
    /// Wall-clock elapsed time.
    pub elapsed: Duration,
}

/// Status of a completed pipeline step.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Step completed successfully.
    Ok,
    /// Step completed with warnings.
    Warning,
    /// Step failed.
    Error,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Ok => write!(f, "OK"),
            Status::Warning => write!(f, "WARNING"),
            Status::Error => write!(f, "ERROR"),
        }
    }
}

/// State of an output artifact on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactState {
    /// Output exists and passes bound checks — no work needed.
    Complete,
    /// Output exists but is incomplete or corrupt.
    Partial,
    /// Output does not exist.
    Absent,
}

/// Resolved key-value options for a pipeline step.
///
/// All values are strings — individual `CommandOp` implementations parse
/// them into the types they need.
#[derive(Debug, Clone, Default)]
pub struct Options(pub IndexMap<String, String>);

impl Options {
    /// Create an empty options map.
    pub fn new() -> Self {
        Options(IndexMap::new())
    }

    /// Get an option value by name.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(|s| s.as_str())
    }

    /// Get an option value, returning an error string if missing.
    pub fn require(&self, name: &str) -> Result<&str, String> {
        self.get(name)
            .ok_or_else(|| format!("required option '{}' not set", name))
    }

    /// Insert or update an option.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.0.insert(name.into(), value.into());
    }

    /// Check whether an option is present.
    pub fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

/// Execution context shared across all pipeline steps.
pub struct StreamContext {
    /// Workspace directory (usually the directory containing `dataset.yaml`).
    pub workspace: PathBuf,
    /// Scratch directory for temporary files deleted after pipeline success.
    pub scratch: PathBuf,
    /// Cache directory for reusable intermediates persisted across runs.
    pub cache: PathBuf,
    /// Shared default variables available for interpolation.
    pub defaults: IndexMap<String, String>,
    /// When `true`, steps print their plan but do not execute.
    pub dry_run: bool,
    /// Persistent progress log for skip-if-fresh semantics.
    pub progress: ProgressLog,
    /// Number of threads available for parallel work.
    pub threads: usize,
    /// Current step identifier, set by the runner before each step executes.
    pub step_id: String,
}

/// Describes a single accepted option for a `CommandOp`.
///
/// Used for dry-run validation and help output.
#[derive(Debug, Clone)]
pub struct OptionDesc {
    /// Option name (key in the YAML step definition).
    pub name: String,
    /// Type hint: `"Path"`, `"int"`, `"enum"`, `"String"`, etc.
    pub type_name: String,
    /// Whether the option must be provided.
    pub required: bool,
    /// Default value if not provided.
    pub default: Option<String>,
    /// Human-readable description.
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_get_set() {
        let mut opts = Options::new();
        assert!(!opts.has("source"));
        opts.set("source", "/data/input.fvec");
        opts.set("threads", "8");
        assert_eq!(opts.get("source"), Some("/data/input.fvec"));
        assert_eq!(opts.get("threads"), Some("8"));
        assert_eq!(opts.get("missing"), None);
        assert!(opts.has("source"));
        assert!(opts.has("threads"));
        assert!(!opts.has("missing"));
    }

    #[test]
    fn test_options_require() {
        let mut opts = Options::new();
        opts.set("key", "value");
        assert_eq!(opts.require("key").unwrap(), "value");
        assert!(opts.require("missing").is_err());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(Status::Ok.to_string(), "OK");
        assert_eq!(Status::Warning.to_string(), "WARNING");
        assert_eq!(Status::Error.to_string(), "ERROR");
    }
}
