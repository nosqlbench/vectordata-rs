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
pub use super::resource::{ResourceDesc, ResourceType};
use super::resource::ResourceGovernor;
use crate::ui::UiHandle;

/// Built-in documentation for a pipeline command.
///
/// Every command provides a one-line summary (for completions and listings)
/// and a full markdown body (for `--help` and `veks help`).
#[derive(Debug, Clone)]
pub struct CommandDoc {
    /// One-line summary (plain text, no markdown).
    /// Used in completion suggestions and command listings.
    pub summary: String,

    /// Full markdown documentation body.
    /// Rendered when the user requests detailed help.
    pub body: String,
}

/// Render an options table in markdown from `OptionDesc` entries.
///
/// Commands can call this from their `command_doc()` to keep the body
/// consistent with `describe_options()`.
pub fn render_options_table(options: &[OptionDesc]) -> String {
    if options.is_empty() {
        return String::from("_(none)_");
    }
    let mut s = String::from("| Option | Type | Required | Default | Description |\n");
    s.push_str("|--------|------|----------|---------|-------------|\n");
    for opt in options {
        let req = if opt.required { "yes" } else { "no" };
        let def = opt.default.as_deref().unwrap_or("—");
        s.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |\n",
            opt.name, opt.type_name, req, def, opt.description,
        ));
    }
    s
}

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

    /// Return built-in documentation for this command.
    ///
    /// The default generates a basic doc from the command path and options.
    /// Commands SHOULD override this with a meaningful summary and body.
    fn command_doc(&self) -> CommandDoc {
        let path = self.command_path();
        let options = self.describe_options();
        let table = render_options_table(&options);
        CommandDoc {
            summary: path.to_string(),
            body: format!("# {}\n\n## Options\n\n{}", path, table),
        }
    }

    /// Declare which resource types this command consumes.
    ///
    /// Commands that process arbitrarily large data MUST override this
    /// (see SRD §06 REQ-RM-11). The default returns an empty list.
    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

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
    /// Output exists but its format is unrecognized, so completeness
    /// cannot be verified.
    Unknown(String),
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
    /// Dataset name (from `name` field in dataset.yaml).
    pub dataset_name: String,
    /// Active profile name (`"all"` when running all profiles).
    pub profile: String,
    /// Ordered list of all profile names (for progress display).
    pub profile_names: Vec<String>,
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
    /// Resource governor for adaptive resource management.
    pub governor: ResourceGovernor,
    /// UI-agnostic event handle for progress, logging, and output.
    pub ui: UiHandle,
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
