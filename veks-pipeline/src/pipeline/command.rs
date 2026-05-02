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

    /// Discovery category — drives grouping in shell completion and
    /// help output. Required, with no default: a new command must
    /// declare where it belongs in the user's mental model.
    ///
    /// The return type is a trait object so consumers of
    /// `veks-completion` outside this crate can define their own
    /// category enum and reuse the same infrastructure. Inside
    /// `veks-pipeline`, commands return one of the static instances
    /// from this module (e.g., [`CAT_VERIFY`], [`CAT_COMPUTE`]) which
    /// are `PipelineCategory` variants exposed as `&'static dyn`.
    fn category(&self) -> &'static dyn veks_completion::CategoryTag;

    /// Discovery tier for stratified shell completion. Required, with
    /// no default: the command's author decides whether the command
    /// surfaces on the first tab tap or sits behind a deeper one.
    ///
    /// Symmetric with [`category`]: returns `&'static dyn LevelTag`
    /// so consumers of `veks-completion` can define their own tier
    /// enum. Inside `veks-pipeline`, return one of the static
    /// instances ([`LVL_PRIMARY`], [`LVL_SECONDARY`], [`LVL_ADVANCED`]).
    /// The completion engine reads `rank()` for ordering — lower =
    /// more discoverable.
    fn level(&self) -> &'static dyn veks_completion::LevelTag;

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
    /// *source* changes. The default combines the crate version and git
    /// commit hash (with a `+dirty` suffix if the working tree has any
    /// uncommitted changes at build time).
    ///
    /// Deliberately excluded:
    ///   - `VEKS_BUILD_NUMBER` (epoch seconds on each build) — would
    ///     invalidate every step's fingerprint on every `cargo install`
    ///     even when the source didn't change, forcing full pipeline
    ///     re-runs for no benefit.
    ///   - The host `rustc` version — lives in `--version` output for
    ///     human debugging, but isn't a data-provenance signal.
    ///
    /// Consequences for local development with an uncommitted tree:
    /// the short hash stays fixed at `<last-commit>+dirty` across
    /// multiple builds of the same working tree, so repeated rebuilds
    /// with local edits do NOT cascade fingerprints stale. Committing
    /// (or toggling between dirty/clean) changes the hash and
    /// invalidates once.
    ///
    /// Format: `{CARGO_PKG_VERSION}+{git_short_hash}[+dirty]`
    fn build_version(&self) -> &str {
        concat!(env!("CARGO_PKG_VERSION"), "+", env!("VEKS_BUILD_HASH"))
    }

    /// Declare which resource types this command consumes.
    ///
    /// Commands that process arbitrarily large data MUST override this
    /// (see SRD §06 REQ-RM-11). The default returns an empty list.
    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    /// Per-option value-completion declarations for shell tab-
    /// completion. Map key is the option name (matching
    /// [`OptionDesc::name`]); the value is the candidate set.
    ///
    /// Default: empty map. Commands with enum-shaped or
    /// comma-separated-list options should override and return
    /// `ValueCompletions` for those keys so users get tab-completion
    /// at the value position.
    fn value_completions(&self) -> std::collections::HashMap<String, ValueCompletions> {
        std::collections::HashMap::new()
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
    /// Provenance components consulted when deciding whether a step is
    /// stale. Defaults to [`super::provenance::ProvenanceFlags::STRICT`]
    /// (every component matters); `veks run --provenance` overrides at
    /// the CLI.
    pub provenance_selector: super::provenance::ProvenanceFlags,
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

/// `veks-pipeline`'s closed set of discovery categories. Every
/// project that uses `veks-completion` defines its own enum like this
/// and implements [`veks_completion::CategoryTag`] on it. The
/// completion engine groups by `tag()` — implementors with matching
/// tags collapse into the same group.
///
/// Variants mirror the verb-prefix structure of the command paths
/// (`analyze ...`, `verify ...`, …). For commands that don't fit a
/// single verb (cross-cutting utilities) use [`PipelineCategory::Pipeline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineCategory {
    Analyze,
    Catalog,
    Cleanup,
    Compute,
    Config,
    Download,
    Generate,
    Merkle,
    Pipeline,
    Query,
    Slab,
    State,
    Transform,
    Verify,
}

impl veks_completion::CategoryTag for PipelineCategory {
    fn tag(&self) -> &'static str {
        match self {
            Self::Analyze   => "analyze",
            Self::Catalog   => "catalog",
            Self::Cleanup   => "cleanup",
            Self::Compute   => "compute",
            Self::Config    => "config",
            Self::Download  => "download",
            Self::Generate  => "generate",
            Self::Merkle    => "merkle",
            Self::Pipeline  => "pipeline",
            Self::Query     => "query",
            Self::Slab      => "slab",
            Self::State     => "state",
            Self::Transform => "transform",
            Self::Verify    => "verify",
        }
    }
}

// Static instances per variant. `CommandOp::category()` returns
// `&'static dyn CategoryTag`, so impls reference these by name
// instead of constructing a static binding inline.
pub static CAT_ANALYZE:   PipelineCategory = PipelineCategory::Analyze;
pub static CAT_CATALOG:   PipelineCategory = PipelineCategory::Catalog;
pub static CAT_CLEANUP:   PipelineCategory = PipelineCategory::Cleanup;
pub static CAT_COMPUTE:   PipelineCategory = PipelineCategory::Compute;
pub static CAT_CONFIG:    PipelineCategory = PipelineCategory::Config;
pub static CAT_DOWNLOAD:  PipelineCategory = PipelineCategory::Download;
pub static CAT_GENERATE:  PipelineCategory = PipelineCategory::Generate;
pub static CAT_MERKLE:    PipelineCategory = PipelineCategory::Merkle;
pub static CAT_PIPELINE:  PipelineCategory = PipelineCategory::Pipeline;
pub static CAT_QUERY:     PipelineCategory = PipelineCategory::Query;
pub static CAT_SLAB:      PipelineCategory = PipelineCategory::Slab;
pub static CAT_STATE:     PipelineCategory = PipelineCategory::State;
pub static CAT_TRANSFORM: PipelineCategory = PipelineCategory::Transform;
pub static CAT_VERIFY:    PipelineCategory = PipelineCategory::Verify;

/// `veks-pipeline`'s discovery-tier enum. Mirrors the
/// [`PipelineCategory`] pattern: implement
/// [`veks_completion::LevelTag`] once, expose static instances per
/// variant, and let `CommandOp` impls reference them by name.
///
/// Tier semantics:
///   - [`PipelineLevel::Primary`] (rank 1) — common everyday
///     commands; the first tab tap reveals these. The bulk of the
///     registry.
///   - [`PipelineLevel::Secondary`] (rank 2) — less common
///     workflow-completing commands (alternate engines, deep-dive
///     analysis, finalization). Second tap.
///   - [`PipelineLevel::Advanced`] (rank 3) —
///     specialized/internal/diagnostic. Third tap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PipelineLevel {
    Primary,
    Secondary,
    Advanced,
}

impl veks_completion::LevelTag for PipelineLevel {
    fn rank(&self) -> u32 {
        match self {
            Self::Primary   => 1,
            Self::Secondary => 2,
            Self::Advanced  => 3,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Primary   => "primary",
            Self::Secondary => "secondary",
            Self::Advanced  => "advanced",
        }
    }
}

pub static LVL_PRIMARY:   PipelineLevel = PipelineLevel::Primary;
pub static LVL_SECONDARY: PipelineLevel = PipelineLevel::Secondary;
pub static LVL_ADVANCED:  PipelineLevel = PipelineLevel::Advanced;

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

/// Tab-completion value list for an option's value position. Returned
/// from [`CommandOp::value_completions`] so commands can declare
/// enumerated value sets without touching the 400+ existing
/// `OptionDesc { … }` literals.
#[derive(Debug, Clone)]
pub struct ValueCompletions {
    /// The valid values to suggest.
    pub values: Vec<String>,
    /// When true, the option accepts a comma-separated list (e.g.,
    /// `--engines metal,stdarch,blas`). Completion parses the
    /// current partial input, drops values already chosen, and
    /// returns candidates suitable to complete the in-progress
    /// segment.
    pub comma_separated: bool,
}

impl ValueCompletions {
    /// Construct from a static slice of string literals.
    pub fn enum_values(values: &[&str]) -> Self {
        Self {
            values: values.iter().map(|s| s.to_string()).collect(),
            comma_separated: false,
        }
    }

    /// Same as `enum_values` but flags the option as comma-separated
    /// so completion suggests only values not already chosen.
    pub fn comma_separated_enum(values: &[&str]) -> Self {
        Self {
            values: values.iter().map(|s| s.to_string()).collect(),
            comma_separated: true,
        }
    }
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
