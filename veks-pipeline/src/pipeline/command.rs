// Copyright (c) Jonathan Shook
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
use veks_core::ui::UiHandle;

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
    /// Canonical command path, e.g. `"import"` or `"generate vectors"`.
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

    /// Build version string for provenance tracking.
    ///
    /// Returns a version identifier that changes whenever the command's
    /// logic changes. The default combines the crate version and git
    /// commit hash and build number, injected at compile time. The build
    /// number (epoch seconds) changes every compilation, so fingerprints
    /// always invalidate when the binary is rebuilt — even if the git SHA
    /// is unchanged.
    ///
    /// Format: `{CARGO_PKG_VERSION}+{git_short_hash}.{build_number}`
    fn build_version(&self) -> &str {
        concat!(env!("CARGO_PKG_VERSION"), "+", env!("VEKS_BUILD_HASH"), ".", env!("VEKS_BUILD_NUMBER"))
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

    /// Project the artifact manifest for this step without executing it.
    ///
    /// Given resolved options, returns the exact list of input files consumed
    /// and output files produced. Every command **must** implement this
    /// explicitly — the default panics to ensure no command silently returns
    /// an incorrect manifest.
    ///
    /// Paths containing `${cache}` or `.cache/` should be classified as
    /// intermediates rather than final outputs.
    fn project_artifacts(&self, step_id: &str, _options: &Options) -> ArtifactManifest {
        // Commands that have not yet implemented project_artifacts return
        // an empty manifest with a warning, rather than guessing wrong.
        log::warn!(
            "command '{}' (step '{}') has no project_artifacts implementation — \
             manifest will be incomplete",
            self.command_path(), step_id,
        );
        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs: vec![],
            outputs: vec![],
            intermediates: vec![],
        }
    }
}

/// Check if a path looks like a cache/intermediate artifact.
pub fn is_cache_path(path: &str) -> bool {
    path.starts_with("${cache}") || path.starts_with(".cache/")
        || path.contains("/.cache/")
}

/// Helper to build an ArtifactManifest from explicit input/output option keys.
///
/// Reads the named options from `options`, strips window notation from paths,
/// and classifies outputs as intermediate when they target the cache directory.
pub fn manifest_from_keys(
    step_id: &str,
    command: &str,
    options: &Options,
    input_keys: &[&str],
    output_keys: &[&str],
) -> ArtifactManifest {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut intermediates = Vec::new();

    for &key in input_keys {
        if let Some(v) = options.get(key) {
            // Strip window notation (e.g., "file.mvec[0..1000)")
            let path = v.split('[').next().unwrap_or(v);
            if !path.is_empty() && !path.starts_with("count:") {
                inputs.push(path.to_string());
            }
        }
    }

    for &key in output_keys {
        if let Some(v) = options.get(key) {
            if is_cache_path(v) {
                intermediates.push(v.to_string());
            } else {
                outputs.push(v.to_string());
            }
        }
    }

    ArtifactManifest {
        step_id: step_id.to_string(),
        command: command.to_string(),
        inputs,
        outputs,
        intermediates,
    }
}

/// Projected artifact manifest from a single pipeline step.
///
/// Describes the inputs consumed and outputs produced by a step, given
/// its resolved options, without actually executing the step. Used by
/// the check system to build a complete workspace manifest.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ArtifactManifest {
    /// Step ID (from pipeline definition).
    pub step_id: String,
    /// Command path (e.g., "compute knn").
    pub command: String,
    /// Input files consumed (must exist before execution).
    pub inputs: Vec<String>,
    /// Output files produced (final artifacts).
    pub outputs: Vec<String>,
    /// Intermediate files produced (cache artifacts, not publishable).
    pub intermediates: Vec<String>,
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
    /// Output exists but is incomplete — must be deleted and restarted.
    Partial,
    /// Output exists but is incomplete — can be resumed incrementally.
    /// The command handles picking up where it left off.
    PartialResumable,
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

    /// Parse an option value, returning a default if unset, or an error
    /// string if the value is present but cannot be parsed.
    pub fn parse_or<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String>
    where
        T::Err: std::fmt::Display,
    {
        match self.get(name) {
            Some(s) => s.parse::<T>()
                .map_err(|e| format!("invalid '{}' value '{}': {}", name, s, e)),
            None => Ok(default),
        }
    }

    /// Parse an optional option value, returning `None` if unset, or an error
    /// string if the value is present but cannot be parsed.
    pub fn parse_opt<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String>
    where
        T::Err: std::fmt::Display,
    {
        match self.get(name) {
            Some(s) => s.parse::<T>()
                .map(Some)
                .map_err(|e| format!("invalid '{}' value '{}': {}", name, s, e)),
            None => Ok(None),
        }
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
    /// Interval between resource status polls and TUI redraws.
    pub status_interval: std::time::Duration,
    /// Estimated total steps including deferred per-profile expansions.
    /// When > 0 and > the current step count, the runner shows both
    /// the current batch count and the estimated total.
    pub estimated_total_steps: usize,
}

/// Describes a single accepted option for a `CommandOp`.
///
/// Used for dry-run validation and help output.

/// The role of a pipeline option — whether it's an input file, output file,
/// or a configuration parameter. Used for TUI display and manifest generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptionRole {
    /// An input file or directory consumed by this step.
    Input,
    /// An output file or directory produced by this step.
    Output,
    /// A configuration parameter (not a file path).
    #[default]
    Config,
}

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
    /// Role of this option: input file, output file, or config.
    /// Defaults to `Config`. Commands should set `Input` or `Output`
    /// for path-typed options that represent consumed/produced artifacts.
    pub role: OptionRole,
}

impl OptionDesc {
    /// Set the role of this option and return self (builder pattern).
    pub fn with_role(mut self, role: OptionRole) -> Self {
        self.role = role;
        self
    }
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
