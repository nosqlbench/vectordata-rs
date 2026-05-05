// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dynamic shell completion engine for CLI tools.
//!
//! Provides a generic, tree-based completion system that completes one level
//! at a time (no eager subcommand chaining). The caller defines the command
//! tree via [`CommandTree`], and this crate handles:
//!
//! - Walking the tree to find candidates for a given input
//! - Filtering out options already present on the command line
//! - Handling bare `key=value` params alongside `--flag` options
//! - Dynamic option discovery from command-line context (e.g., reading
//!   a workload file to discover its declared parameters)
//! - Generating bash completion scripts
//! - Handling the `_<APP>_COMPLETE=bash` env var callbacks
//!
//! # Usage
//!
//! ```rust,no_run
//! use veks_completion::{CommandTree, Node, complete, print_bash_script, handle_complete_env};
//!
//! let tree = CommandTree::new("myapp")
//!     .command("run", Node::leaf(&["--dry-run", "--threads"]))
//!     .command("check", Node::leaf(&["--all", "--quiet"]))
//!     .group("pipeline", Node::group(vec![
//!         ("compute", Node::group(vec![
//!             ("knn", Node::leaf(&["--base", "--query", "--metric"])),
//!         ])),
//!     ]));
//!
//! // In main():
//! if handle_complete_env("myapp", &tree) {
//!     std::process::exit(0);
//! }
//! ```

use std::collections::BTreeMap;

/// A function that provides dynamic completion values for a specific option.
///
/// Called when the user tabs after an option that has a registered provider.
/// Receives the partial word being typed and the full context of completed
/// words on the command line (excluding the program name and the partial).
/// Heap-allocated, thread-safe closure type so providers can capture
/// data (e.g., a static enum-value list discovered from a
/// `CommandOp::value_completions` map). Function pointers can be
/// promoted to this type via [`ValueProvider::from_fn`] so existing
/// `fn(&str, &[&str]) -> Vec<String>` providers keep working.
pub type ValueProvider = std::sync::Arc<dyn Fn(&str, &[&str]) -> Vec<String> + Send + Sync>;

/// Helper to wrap a `fn`-pointer provider into the closure-typed
/// [`ValueProvider`]. Most existing global providers use plain `fn`
/// pointers and call this when registering.
pub fn fn_provider(f: fn(&str, &[&str]) -> Vec<String>) -> ValueProvider {
    std::sync::Arc::new(f)
}

/// Discovery-tier abstraction symmetric with [`CategoryTag`].
///
/// `veks-completion` doesn't define how many tiers exist or how
/// they're named — each consuming crate decides. Implement
/// `LevelTag` on your own enum to declare a closed set of
/// stratified-completion tiers; commands then return `&'static dyn
/// LevelTag` and the completion engine orders by [`rank`].
///
/// `rank()` is the scalar used by stratified completion (the Nth
/// tab tap reveals everything with `rank <= N`). Lower = more
/// discoverable. Two implementors with the same `rank` are treated
/// as the same tier.
pub trait LevelTag: 'static + Send + Sync + std::fmt::Debug {
    /// Numeric tier; lower values are revealed first by the
    /// stratified-completion tab cycle.
    fn rank(&self) -> u32;

    /// Optional display name (e.g., "primary", "advanced") for
    /// help renderers. Default returns the empty string —
    /// implementors should override for human-friendly listings.
    fn name(&self) -> &'static str { "" }
}

/// Discovery-category abstraction.
///
/// `veks-completion` doesn't define WHICH categories exist — each
/// consuming crate decides. Implement `CategoryTag` on your own
/// enum to declare a closed set of categories specific to your
/// project; commands then return `&'static dyn CategoryTag`
/// references and the completion engine groups by `tag()`.
///
/// Example:
/// ```ignore
/// #[derive(Debug, Clone, Copy)]
/// enum MyCategory { Foo, Bar }
/// impl veks_completion::CategoryTag for MyCategory {
///     fn tag(&self) -> &'static str {
///         match self { Self::Foo => "foo", Self::Bar => "bar" }
///     }
/// }
/// // Static instances per variant for `&'static dyn` returns:
/// static CAT_FOO: MyCategory = MyCategory::Foo;
/// static CAT_BAR: MyCategory = MyCategory::Bar;
/// ```
///
/// `tag()` is the stable, lowercase grouping key. Two implementors
/// returning the same `tag()` are treated as the same group at
/// completion time.
pub trait CategoryTag: 'static + Send + Sync + std::fmt::Debug {
    /// Stable lowercase tag used by completion grouping and help
    /// rendering as the user-visible category name.
    fn tag(&self) -> &'static str;
}

/// A function that provides additional option candidates based on context.
///
/// Called during leaf completion to discover extra `key=` options that
/// aren't statically declared. For example, reading a workload file
/// referenced on the command line and returning its declared parameter
/// names as completable options.
///
/// Receives the partial word being typed and the full context of completed
/// words. Returns additional option names (e.g., `["keyspace=", "table="]`).
pub type DynamicOptionsProvider = fn(partial: &str, context: &[&str]) -> Vec<String>;

/// Default visibility tier when a node doesn't explicitly opt
/// into a higher tier. Tier 1 means "show on the very first
/// tab tap" — preserves the pre-stratification behavior for
/// existing apps that haven't categorized their commands.
pub const DEFAULT_LEVEL: u32 = 1;

/// Errors produced by [`CommandTree::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    /// A registered command lacks a category tag and the tree
    /// was built with [`CommandTree::require_metadata`].
    MissingCategory { command: String },
    /// A registered command lacks an explicit `with_level()`
    /// call and the tree was built with
    /// [`CommandTree::require_metadata`].
    MissingLevel { command: String },
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataError::MissingCategory { command } =>
                write!(f, "command '{command}' is missing a category — call \
                    Node::with_category(...) when registering"),
            MetadataError::MissingLevel { command } =>
                write!(f, "command '{command}' is missing an explicit level — \
                    call Node::with_level(N) when registering"),
        }
    }
}

impl std::error::Error for MetadataError {}

/// A node in the command tree.
///
/// Carries two metadata fields used by stratified
/// (multi-tap) completion:
///
/// - `category` — free-form display tag. Apps can group root
///   commands by category in expanded help / future renderers
///   (e.g. `workloads`, `documentation`, `tools`).
/// - `level` — visibility tier. The Nth tab tap reveals every
///   root-level node with `level <= N`. Default
///   [`DEFAULT_LEVEL`] (= 1) means "always shown from the
///   first tap." Use a higher level (2, 3, …) for
///   less-discoverable commands so the first tap stays focused
///   on a small set the user wants by default.
///
/// Existing callers that didn't set these fields get the
/// pre-existing behavior automatically (everything visible at
/// tap 1, no category metadata).
#[derive(Clone)]
pub enum Node {
    /// A leaf command with option names and optional value providers.
    Leaf {
        options: Vec<String>,
        /// Options that are boolean flags (no value expected).
        flags: std::collections::HashSet<String>,
        /// Dynamic value providers keyed by option name (e.g., "--dataset").
        value_providers: BTreeMap<String, ValueProvider>,
        /// Optional provider that discovers additional options from context.
        dynamic_options: Option<DynamicOptionsProvider>,
        /// Display group tag — see type-level docs for usage.
        category: Option<String>,
        /// Tap-tier visibility — see type-level docs for
        /// usage. `None` means "never explicitly set"; the
        /// effective level resolves to [`DEFAULT_LEVEL`] but
        /// strict-metadata mode treats `None` as missing.
        level: Option<u32>,
    },
    /// A group containing named child nodes.
    Group {
        children: BTreeMap<String, Node>,
        category: Option<String>,
        level: Option<u32>,
    },
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Leaf { options, flags, value_providers, dynamic_options, category, level } => {
                f.debug_struct("Leaf")
                    .field("options", options)
                    .field("flags", flags)
                    .field("value_providers", &value_providers.keys().collect::<Vec<_>>())
                    .field("has_dynamic_options", &dynamic_options.is_some())
                    .field("category", category)
                    .field("level", level)
                    .finish()
            }
            Node::Group { children, category, level } => {
                f.debug_struct("Group")
                    .field("children", children)
                    .field("category", category)
                    .field("level", level)
                    .finish()
            }
        }
    }
}

impl Node {
    /// Create a leaf node with the given option names (all assumed to take values).
    pub fn leaf(options: &[&str]) -> Self {
        Node::Leaf {
            options: options.iter().map(|s| s.to_string()).collect(),
            flags: std::collections::HashSet::new(),
            value_providers: BTreeMap::new(),
            dynamic_options: None,
            category: None,
            level: None,
        }
    }

    /// Create a leaf node with separate value-options and boolean flags.
    pub fn leaf_with_flags(options: &[&str], flags: &[&str]) -> Self {
        Node::Leaf {
            options: options.iter().chain(flags.iter()).map(|s| s.to_string()).collect(),
            flags: flags.iter().map(|s| s.to_string()).collect(),
            value_providers: BTreeMap::new(),
            dynamic_options: None,
            category: None,
            level: None,
        }
    }

    /// Attach a dynamic value provider to an option on this leaf node.
    pub fn with_value_provider(mut self, option: &str, provider: ValueProvider) -> Self {
        if let Node::Leaf { ref mut value_providers, .. } = self {
            value_providers.insert(option.to_string(), provider);
        }
        self
    }

    /// Attach a dynamic options provider to this leaf node.
    ///
    /// The provider is called during completion to discover additional
    /// `key=` options from context (e.g., workload file parameters).
    pub fn with_dynamic_options(mut self, provider: DynamicOptionsProvider) -> Self {
        if let Node::Leaf { ref mut dynamic_options, .. } = self {
            *dynamic_options = Some(provider);
        }
        self
    }

    /// Tag this node with a display category (e.g. `"workloads"`,
    /// `"documentation"`). Categories are free-form strings used by
    /// renderers to group commands; they don't affect completion
    /// candidate ordering directly.
    pub fn with_category(mut self, cat: &str) -> Self {
        match &mut self {
            Node::Leaf { category, .. } => *category = Some(cat.to_string()),
            Node::Group { category, .. } => *category = Some(cat.to_string()),
        }
        self
    }

    /// Set the tap-tier visibility for this node. The Nth tab
    /// tap reveals every root-level node with `level <= N`.
    /// Default (when `with_level` is not called) is
    /// [`DEFAULT_LEVEL`] (= 1) — but strict-metadata mode (see
    /// [`CommandTree::require_metadata`]) treats the absence
    /// of an explicit call as a registration error.
    pub fn with_level(mut self, lvl: u32) -> Self {
        match &mut self {
            Node::Leaf { level, .. } => *level = Some(lvl),
            Node::Group { level, .. } => *level = Some(lvl),
        }
        self
    }

    /// Get the node's category tag, if any.
    pub fn category(&self) -> Option<&str> {
        match self {
            Node::Leaf { category, .. } => category.as_deref(),
            Node::Group { category, .. } => category.as_deref(),
        }
    }

    /// Get the node's effective tap-tier level — explicit
    /// value if set, otherwise [`DEFAULT_LEVEL`].
    pub fn level(&self) -> u32 {
        self.level_explicit().unwrap_or(DEFAULT_LEVEL)
    }

    /// Get the node's *explicit* tap-tier level — `None` when
    /// `with_level` was never called. Used by strict-metadata
    /// validation to distinguish "user picked level 1" from
    /// "user forgot to set a level."
    pub fn level_explicit(&self) -> Option<u32> {
        match self {
            Node::Leaf { level, .. } => *level,
            Node::Group { level, .. } => *level,
        }
    }

    /// Check if an option is a boolean flag (no value expected).
    pub fn is_flag(&self, option: &str) -> bool {
        match self {
            Node::Leaf { flags, .. } => flags.contains(option),
            _ => false,
        }
    }

    /// Create a group node from a list of `(name, child)` pairs.
    pub fn group(children: Vec<(&str, Node)>) -> Self {
        Node::Group {
            children: children.into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            category: None,
            level: None,
        }
    }

    /// Create an empty group node.
    pub fn empty_group() -> Self {
        Node::Group {
            children: BTreeMap::new(),
            category: None,
            level: None,
        }
    }

    /// Add a child to a group node. Panics if called on a leaf.
    pub fn with_child(mut self, name: &str, child: Node) -> Self {
        match &mut self {
            Node::Group { children, .. } => { children.insert(name.to_string(), child); }
            Node::Leaf { .. } => panic!("cannot add child to leaf node"),
        }
        self
    }

    /// Get child names (empty for leaves).
    pub fn child_names(&self) -> Vec<&str> {
        match self {
            Node::Group { children, .. } => children.keys().map(|k| k.as_str()).collect(),
            Node::Leaf { .. } => Vec::new(),
        }
    }

    /// Get a child by name.
    pub fn child(&self, name: &str) -> Option<&Node> {
        match self {
            Node::Group { children, .. } => children.get(name),
            Node::Leaf { .. } => None,
        }
    }

    /// Get option names (empty for groups).
    pub fn options(&self) -> Vec<&str> {
        match self {
            Node::Leaf { options, .. } => options.iter().map(|s| s.as_str()).collect(),
            Node::Group { .. } => Vec::new(),
        }
    }
}

// =====================================================================
// Strict-metadata builder (compile-time enforcement)
// =====================================================================

/// Type-state wrapper around [`Node`] that tracks at the type
/// level whether the node has been given a category and an
/// explicit tap-tier level.
///
/// Used together with [`CommandTree::strict_command`] /
/// [`CommandTree::strict_group`] to force compile-time
/// enforcement of the stratified completion contract: an app
/// that opts into strict mode cannot register an uncategorized
/// or unleveled command, because the registration call itself
/// will not type-check unless both fields have been provided.
///
/// The two `bool` const generics flip from `false` to `true`
/// when the matching builder method is called:
///
/// - `with_category("…")` → `HAS_CATEGORY = true`
/// - `with_level(N)`     → `HAS_LEVEL    = true`
///
/// Apps that don't want the compile-time check can keep using
/// the regular [`Node`] API and call
/// [`CommandTree::require_metadata`] to get an equivalent
/// runtime check at registration time.
///
/// # Successful registration (compiles)
///
/// ```
/// use veks_completion::{CommandTree, StrictNode};
/// let tree = CommandTree::new("myapp")
///     .strict_command(
///         "run",
///         StrictNode::leaf(&["--cycles=", "--threads="])
///             .with_category("workloads")
///             .with_level(1),
///     );
/// # let _ = tree;
/// ```
///
/// # Missing category (compile error)
///
/// ```compile_fail
/// use veks_completion::{CommandTree, StrictNode};
/// let _tree = CommandTree::new("myapp").strict_command(
///     "bad",
///     StrictNode::leaf(&[]).with_level(1),  // missing with_category
/// );
/// ```
///
/// # Missing level (compile error)
///
/// ```compile_fail
/// use veks_completion::{CommandTree, StrictNode};
/// let _tree = CommandTree::new("myapp").strict_command(
///     "bad",
///     StrictNode::leaf(&[]).with_category("x"),  // missing with_level
/// );
/// ```
pub struct StrictNode<const HAS_CATEGORY: bool, const HAS_LEVEL: bool> {
    inner: Node,
}

impl StrictNode<false, false> {
    /// Begin building a strict leaf node. Both `with_category`
    /// and `with_level` must be called before this can be
    /// passed to [`CommandTree::strict_command`].
    pub fn leaf(options: &[&str]) -> Self {
        Self { inner: Node::leaf(options) }
    }

    /// Same as [`Node::leaf_with_flags`] but type-state-checked.
    pub fn leaf_with_flags(options: &[&str], flags: &[&str]) -> Self {
        Self { inner: Node::leaf_with_flags(options, flags) }
    }

    /// Begin building a strict group node.
    pub fn group(children: Vec<(&str, Node)>) -> Self {
        Self { inner: Node::group(children) }
    }

    /// Begin from an already-constructed [`Node`]. Useful when
    /// migrating an existing tree to strict mode incrementally.
    pub fn from_node(node: Node) -> Self {
        Self { inner: node }
    }
}

impl<const C: bool, const L: bool> StrictNode<C, L> {
    /// Tag with a category. Flips `HAS_CATEGORY` to `true`.
    pub fn with_category(self, cat: &str) -> StrictNode<true, L> {
        StrictNode { inner: self.inner.with_category(cat) }
    }

    /// Set the tap-tier level. Flips `HAS_LEVEL` to `true`.
    pub fn with_level(self, lvl: u32) -> StrictNode<C, true> {
        StrictNode { inner: self.inner.with_level(lvl) }
    }

    /// Forward through to the inner node's value-provider
    /// builder.
    pub fn with_value_provider(mut self, option: &str, provider: ValueProvider) -> Self {
        self.inner = self.inner.with_value_provider(option, provider);
        self
    }

    /// Forward through to the inner node's dynamic-options
    /// builder.
    pub fn with_dynamic_options(mut self, provider: DynamicOptionsProvider) -> Self {
        self.inner = self.inner.with_dynamic_options(provider);
        self
    }
}

impl StrictNode<true, true> {
    /// Unwrap a fully-qualified strict node into a plain
    /// [`Node`]. The compile-time guarantee carries through to
    /// the moment of unwrapping: only nodes that have set both
    /// category and level can be downgraded.
    pub fn into_node(self) -> Node { self.inner }
}

/// The top-level command tree for an application.
pub struct CommandTree {
    /// Application name (used for env var naming).
    pub app_name: String,
    /// Root node (always a group).
    pub root: Node,
    /// Commands that exist but are hidden from root-level listing.
    pub hidden: std::collections::HashSet<String>,
    /// Global value providers keyed by option name.
    pub global_value_providers: BTreeMap<String, ValueProvider>,
    /// When true, every registered command must declare both a
    /// category (via [`Node::with_category`]) and an explicit
    /// level (via [`Node::with_level`]). [`Self::command`] /
    /// [`Self::group`] / [`Self::hidden_command`] panic on
    /// registration of an undertagged node, surfacing the
    /// problem at the call site rather than producing a
    /// silently-uncategorized completion tree at runtime.
    /// Opt-in for apps that want their stratified completion
    /// UX enforced at compile-test time.
    pub strict_metadata: bool,
}

impl CommandTree {
    /// Maximum [`Node::level`] across all root-level children. Drives
    /// the rotation cycle in [`complete_rotating`]: a tap beyond
    /// `max_level()` wraps back to level 1.
    ///
    /// Returns at least 1 (since [`DEFAULT_LEVEL`] is 1) so callers
    /// can use the result directly as a modular cycle length.
    pub fn max_level(&self) -> u32 {
        let mut max = DEFAULT_LEVEL;
        if let Node::Group { children, .. } = &self.root {
            for (_, child) in children {
                if child.level() > max {
                    max = child.level();
                }
            }
        }
        max
    }

    /// Create a new command tree with an empty root group.
    ///
    /// The `app_name` is used to construct the environment variable name
    /// for completion callbacks (e.g., `_MYAPP_COMPLETE=bash`).
    pub fn new(app_name: &str) -> Self {
        CommandTree {
            app_name: app_name.to_string(),
            root: Node::empty_group(),
            hidden: std::collections::HashSet::new(),
            global_value_providers: BTreeMap::new(),
            strict_metadata: false,
        }
    }

    /// Opt-in to strict-metadata mode. Every subsequent call
    /// to [`Self::command`] / [`Self::group`] /
    /// [`Self::hidden_command`] checks that the node has both
    /// a category and an explicit level — registration panics
    /// if either is missing, with a message naming the
    /// offending command.
    ///
    /// Use this in apps that have committed to a stratified
    /// completion model and want the build to break if a new
    /// command is added without categorizing it.
    pub fn require_metadata(mut self) -> Self {
        self.strict_metadata = true;
        self
    }

    /// Walk every registered command and check for missing
    /// category / level metadata. Returns `Ok(())` when every
    /// node satisfies the contract; `Err(Vec<MetadataError>)`
    /// otherwise with one entry per offending command.
    ///
    /// Always available regardless of `strict_metadata` — apps
    /// that want validation as a one-shot post-build check
    /// (CI test, debug-assert, etc.) can call this directly
    /// without enabling the panic-at-registration mode.
    pub fn validate(&self) -> Result<(), Vec<MetadataError>> {
        let mut errors = Vec::new();
        if let Node::Group { children, .. } = &self.root {
            for (name, node) in children {
                if node.category().is_none() {
                    errors.push(MetadataError::MissingCategory {
                        command: name.clone(),
                    });
                }
                if node.level_explicit().is_none() {
                    errors.push(MetadataError::MissingLevel {
                        command: name.clone(),
                    });
                }
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Internal: panic if `strict_metadata` is set and `node`
    /// is missing required metadata. Called from every
    /// `command`-style registration helper so the error fires
    /// at the source line that registered the bad node.
    fn check_strict(&self, name: &str, node: &Node) {
        if !self.strict_metadata { return; }
        if node.category().is_none() {
            panic!("veks-completion: app '{}' has require_metadata() set, \
                    but command '{name}' was registered without \
                    Node::with_category(...). Add a category tag.",
                self.app_name);
        }
        if node.level_explicit().is_none() {
            panic!("veks-completion: app '{}' has require_metadata() set, \
                    but command '{name}' was registered without \
                    Node::with_level(...). Pick a tap-tier level (1, 2, 3, ...).",
                self.app_name);
        }
    }

    /// Add a top-level command (leaf or group) to the tree.
    ///
    /// This is a builder method — it consumes and returns `self` for chaining.
    pub fn command(mut self, name: &str, node: Node) -> Self {
        self.check_strict(name, &node);
        self.root = self.root.with_child(name, node);
        self
    }

    /// Add a top-level command using the type-state-checked
    /// [`StrictNode`] API. The signature requires
    /// `StrictNode<true, true>`, so calling this with a node
    /// missing either `with_category(...)` or `with_level(...)`
    /// is a **compile-time** error — no runtime panic, no
    /// silent skip. Recommended entry point for apps that want
    /// the stratified completion model strictly enforced.
    pub fn strict_command(
        mut self,
        name: &str,
        node: StrictNode<true, true>,
    ) -> Self {
        self.root = self.root.with_child(name, node.into_node());
        self
    }

    /// Type-state-checked alias for grouping. Same compile-time
    /// guarantee as [`Self::strict_command`].
    pub fn strict_group(self, name: &str, node: StrictNode<true, true>) -> Self {
        self.strict_command(name, node)
    }

    /// Type-state-checked variant of [`Self::hidden_command`].
    pub fn strict_hidden_command(
        mut self,
        name: &str,
        node: StrictNode<true, true>,
    ) -> Self {
        self.hidden.insert(name.to_string());
        self.root = self.root.with_child(name, node.into_node());
        self
    }

    /// Add a top-level group to the tree. Alias for [`command`](Self::command).
    pub fn group(self, name: &str, node: Node) -> Self {
        self.command(name, node)
    }

    /// Add a command that is registered but hidden from root-level listing.
    ///
    /// Hidden commands are still completable if the user types the name
    /// prefix directly — they are just excluded from the initial empty-prefix
    /// candidate list. Useful for aliases and shorthands.
    pub fn hidden_command(mut self, name: &str, node: Node) -> Self {
        self.check_strict(name, &node);
        self.hidden.insert(name.to_string());
        self.command(name, node)
    }

    /// Register a value provider that applies to an option name across all
    /// leaf commands in the tree.
    ///
    /// When the user types `--dataset <TAB>`, the provider is called regardless
    /// of which leaf command is active. Per-leaf providers registered via
    /// [`Node::with_value_provider`] take precedence over global providers.
    pub fn global_value_provider(mut self, option: &str, provider: ValueProvider) -> Self {
        self.global_value_providers.insert(option.to_string(), provider);
        self
    }
}

/// Check if a word on the command line matches (and thus consumes) a
/// defined option. Handles exact flags, `key=value`, `--key=value`,
/// and cross-style equivalence.
fn word_matches_option(word: &str, option: &str) -> bool {
    if word == option { return true; }

    if let Some(key) = option.strip_suffix('=') {
        if word.starts_with(key) && word[key.len()..].starts_with('=') {
            return true;
        }
        let dashed = format!("--{key}");
        if word.starts_with(&dashed) && word[dashed.len()..].starts_with('=') {
            return true;
        }
    }

    if option.starts_with("--") && !option.ends_with('=') {
        if word.starts_with(option) && word[option.len()..].starts_with('=') {
            return true;
        }
        let bare = &option[2..];
        if word.starts_with(bare) && word[bare.len()..].starts_with('=') {
            return true;
        }
    }

    false
}

/// Collect canonical keys for options already present on the command line.
fn consumed_keys(words: &[&str], options: &[String]) -> std::collections::HashSet<String> {
    let mut consumed = std::collections::HashSet::new();
    for &word in words {
        for opt in options {
            if word_matches_option(word, opt) {
                let key = opt.trim_start_matches('-').trim_end_matches('=');
                consumed.insert(key.to_string());
            }
        }
    }
    consumed
}

/// Check if an option's canonical key is in the consumed set.
fn is_consumed(option: &str, consumed: &std::collections::HashSet<String>) -> bool {
    let key = option.trim_start_matches('-').trim_end_matches('=');
    consumed.contains(key)
}

/// Compute completion candidates for the given input words.
///
/// Options already present on the command line are excluded. Both
/// `--flag` and bare `key=` styles are supported and deduplicated.
/// Dynamic options from context providers are included.
///
/// Always operates at tap level 1. For stratified completion
/// where successive tabs reveal more candidates, use
/// [`complete_at_tap`].
pub fn complete(tree: &CommandTree, words: &[&str]) -> Vec<String> {
    complete_at_tap(tree, words, 1)
}

/// Rotating-tier completion. Returns root-level candidates whose
/// `Node::level() == only_level` (NOT the cumulative `<=` set), so
/// successive tab taps cycle through tiers one at a time and wrap
/// around after the highest. Once the user starts typing a
/// specific name, behaves identically to [`complete`] — the level
/// filter applies only at the root with an empty partial.
///
/// Use [`complete_rotating`] (or [`handle_complete_env`] which
/// already wires this up) for the recommended UX:
///
///   tap 1 → only level 1   (Primary)
///   tap 2 → only level 2   (Secondary)
///   tap 3 → only level 3   (Advanced)
///   tap 4 → wraps back to level 1
///   ...
///
/// Cycle length is the tree's [`max_level`].
pub fn complete_at_level_only(tree: &CommandTree, words: &[&str], only_level: u32) -> Vec<String> {
    // Words shape: [binary, completed..., partial]. At absolute root
    // with no input at all, words may have just [binary], so treat
    // missing partial as empty.
    let partial = if words.len() > 1 { *words.last().unwrap_or(&"") } else { "" };
    let completed: &[&str] = if words.len() > 1 { &words[1..words.len() - 1] } else { &[] };

    // Rotation only filters when the user is at a group prompt with
    // no partial typed. Once they start typing a name, we want
    // anything matching it (regardless of tier) so half-typed
    // higher-tier commands still complete on the first tap.
    if !partial.is_empty() {
        return complete(tree, words);
    }

    // Walk the tree following completed words to find the current
    // group node. If we hit a non-existent child or a leaf, fall
    // back to the standard completion engine.
    let mut node = &tree.root;
    let at_root = completed.is_empty();
    for &word in completed {
        match node.child(word) {
            Some(child) => node = child,
            None => return complete(tree, words),
        }
    }

    // Apply the rotation filter to whichever group we landed on.
    // Cumulative semantics: tap N reveals every child whose level is
    // <= N. So a single tap shows layer 1; a rapid double-tap shows
    // layers 1 + 2 together; etc. The result is always sorted in
    // *layer order* (layer 1 candidates first, then layer 2, …) with
    // the standard `--`-flags-last + alphabetical ordering applied
    // within each layer.
    if let Node::Group { children, .. } = node {
        let mut candidates: Vec<(u32, String)> = children.iter()
            .filter(|(k, _)| !at_root || !tree.hidden.contains(k.as_str()))
            .filter(|(_, child)| child.level() <= only_level)
            .map(|(k, child)| (child.level(), k.to_string()))
            .collect();
        candidates.sort_by(|(la, a), (lb, b)| {
            la.cmp(lb)
                .then_with(|| a.starts_with('-').cmp(&b.starts_with('-')))
                .then_with(|| a.cmp(b))
        });
        return candidates.into_iter().map(|(_, k)| k).collect();
    }

    // Landed on a leaf — no children to rotate, defer to standard
    // completion (which handles option/value completion).
    complete(tree, words)
}

/// Convenience wrapper over [`complete_at_level_only`] that maps
/// the raw tap counter to the rotating level. Cycle length is
/// computed relative to whichever group node the user has
/// descended into — a subgroup with only level-1 children has a
/// cycle length of 1 (every tap shows the same set), while a
/// subgroup with Primary+Secondary+Advanced children has a cycle
/// length of 3. A tap beyond the cycle wraps back to level 1.
pub fn complete_rotating(tree: &CommandTree, words: &[&str], tap_count: u32) -> Vec<String> {
    // Determine the max level among the children of whichever
    // group the cursor is currently inside.
    let completed: &[&str] = if words.len() > 1 { &words[1..words.len() - 1] } else { &[] };
    let mut node = &tree.root;
    for &word in completed {
        match node.child(word) {
            Some(child) => node = child,
            None => break,
        }
    }
    let max = max_level_of_children(node).max(1);
    let only = ((tap_count.saturating_sub(1)) % max) + 1;
    complete_at_level_only(tree, words, only)
}

/// Maximum [`Node::level`] across the immediate children of `node`,
/// or [`DEFAULT_LEVEL`] if `node` is a leaf or has no children.
pub(crate) fn max_level_of_children(node: &Node) -> u32 {
    let mut max = DEFAULT_LEVEL;
    if let Node::Group { children, .. } = node {
        for (_, child) in children {
            if child.level() > max {
                max = child.level();
            }
        }
    }
    max
}

/// Stratified completion: returns root-level candidates with
/// `Node::level() <= tap_count`, so the Nth tab tap reveals
/// progressively more commands. Inside a subcommand or with a
/// non-empty partial, behaves identically to [`complete`] —
/// the level filter applies only when the user is at the
/// root prompt with no prefix typed.
///
/// Default Node level is [`DEFAULT_LEVEL`] (= 1), so apps that
/// haven't categorized their commands see the same single-tap
/// behavior they did before stratification.
pub fn complete_at_tap(tree: &CommandTree, words: &[&str], tap_count: u32) -> Vec<String> {
    if words.len() <= 1 {
        let mut cmds: Vec<String> = tree.root.child_names().iter()
            .filter(|s| !tree.hidden.contains(**s))
            .filter(|s| {
                tree.root.child(s)
                    .map(|n| n.level() <= tap_count)
                    .unwrap_or(true)
            })
            .map(|s| s.to_string())
            .collect();
        cmds.sort_by(|a, b| {
            a.starts_with('-').cmp(&b.starts_with('-')).then_with(|| a.cmp(b))
        });
        return cmds;
    }

    let partial = words.last().unwrap_or(&"");
    let completed = &words[1..words.len() - 1];
    let at_root = completed.is_empty();

    // Walk the tree following completed words.
    let mut node = &tree.root;
    let mut remaining_start = 0;
    for (i, &word) in completed.iter().enumerate() {
        match node.child(word) {
            Some(child) => { node = child; remaining_start = i + 1; }
            None => break,
        }
    }
    let remaining = &completed[remaining_start..];

    // Check global value providers for the previous word.
    if let Some(&prev_word) = completed.last()
        && let Some(provider) = tree.global_value_providers.get(prev_word) {
        return provider(partial, remaining);
    }

    match node {
        Node::Group { children, .. } => {
            let mut candidates: Vec<String> = children.iter()
                .filter(|(k, _)| k.starts_with(partial))
                .filter(|(k, _)| !at_root || !partial.is_empty() || !tree.hidden.contains(k.as_str()))
                // Level filter only applies at root with an
                // empty partial — once the user has started
                // typing a specific name, return matching
                // commands regardless of tap tier so a
                // partially-typed level-2 command (e.g.
                // `--ins<TAB>`) still completes on tap 1.
                .filter(|(_, child)| {
                    !at_root || !partial.is_empty() || child.level() <= tap_count
                })
                .map(|(k, _)| k.to_string())
                .collect();
            candidates.sort_by(|a, b| {
                a.starts_with('-').cmp(&b.starts_with('-')).then_with(|| a.cmp(b))
            });
            candidates
        }
        Node::Leaf { options, flags, value_providers, dynamic_options, .. } => {
            // Check if the previous word is a --option expecting a separate value.
            if let Some(&prev_word) = remaining.last()
                && prev_word.starts_with("--") && !prev_word.contains('=') && !flags.contains(prev_word) {
                if let Some(provider) = value_providers.get(prev_word) {
                    return provider(partial, remaining);
                }
                if let Some(provider) = tree.global_value_providers.get(prev_word) {
                    return provider(partial, remaining);
                }
                return Vec::new();
            }

            // Partial of the form `key=value_prefix` — the user
            // has committed to the key, they want value
            // candidates for it. Bash's default
            // `COMP_WORDBREAKS` contains `=`, which means
            // readline already treats the current word as just
            // the segment AFTER the `=`. So we must return BARE
            // values here, not `key=value` strings — emitting
            // the key would replicate it on the line and yield
            // `key=key=value` ("stutter"). When no provider
            // matches, return an empty candidate set: the user
            // has signaled "I'm typing a value", so falling
            // through to the option-name list (which can
            // re-emit `key=` and stutter the same way) would be
            // wrong.
            if let Some(eq_pos) = partial.find('=') {
                let key = &partial[..eq_pos];
                let value_partial = &partial[eq_pos + 1..];
                let key_eq = format!("{key}=");
                let dashed_key = format!("--{key}");
                if let Some(provider) = value_providers.get(&key_eq)
                    .or_else(|| value_providers.get(&dashed_key))
                    .or_else(|| tree.global_value_providers.get(&key_eq))
                    .or_else(|| tree.global_value_providers.get(&dashed_key))
                {
                    return provider(value_partial, remaining);
                }
                return Vec::new();
            }

            // Collect all available options: static + dynamic from context.
            let mut all_options: Vec<String> = options.clone();
            if let Some(provider) = dynamic_options {
                let dynamic = provider(partial, remaining);
                for opt in dynamic {
                    if !all_options.contains(&opt) {
                        all_options.push(opt);
                    }
                }
            }

            // Filter out already-consumed options.
            let consumed = consumed_keys(remaining, &all_options);

            let mut candidates: Vec<String> = all_options.iter()
                .filter(|o| o.starts_with(partial) && !is_consumed(o, &consumed))
                .map(|o| o.to_string())
                .collect();

            // Also offer global provider options.
            for global_opt in tree.global_value_providers.keys() {
                if global_opt.starts_with(partial) && !candidates.contains(global_opt) {
                    candidates.push(global_opt.clone());
                }
            }

            // Sort: bare params first, then --flags.
            candidates.sort_by(|a, b| {
                a.starts_with('-').cmp(&b.starts_with('-')).then_with(|| a.cmp(b))
            });
            candidates
        }
    }
}

/// Supported shells for completion-script generation.
///
/// `Bash` and `Zsh` (via bash-compatible mode) are fully implemented;
/// `Fish`, `Elvish`, and `PowerShell` placeholders are accepted but
/// emit a stub-with-warning so callers can register them in a CLI
/// `--shell` flag without separate dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    Elvish,
    PowerShell,
}

impl Shell {
    /// Parse a shell name (case-insensitive) — `"bash"`, `"zsh"`,
    /// `"fish"`, `"elvish"`, or `"pwsh"` / `"powershell"`. Returns
    /// `None` for unrecognized names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "bash" => Some(Self::Bash),
            "zsh" => Some(Self::Zsh),
            "fish" => Some(Self::Fish),
            "elvish" => Some(Self::Elvish),
            "pwsh" | "powershell" => Some(Self::PowerShell),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::Zsh => "zsh",
            Self::Fish => "fish",
            Self::Elvish => "elvish",
            Self::PowerShell => "powershell",
        }
    }
}

/// Detect the user's interactive shell.
///
/// Tries `$SHELL` first (the standard Unix mechanism), then falls
/// back to inspecting the parent process's `/proc/PID/comm` on Linux
/// for the case where `$SHELL` is set to something other than the
/// actual interactive shell (e.g., when running under a wrapper).
/// Returns `None` if no recognized shell can be determined.
pub fn detect_shell() -> Option<Shell> {
    if let Ok(shell_path) = std::env::var("SHELL") {
        let name = std::path::Path::new(&shell_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if let Some(s) = Shell::from_name(name) {
            return Some(s);
        }
    }
    #[cfg(target_os = "linux")]
    {
        // Read PPid from /proc/self/status — avoids a libc dep on
        // getppid(2). Format line: `PPid:\t<pid>\n`.
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            if let Some(ppid_line) = status.lines().find(|l| l.starts_with("PPid:")) {
                if let Some(ppid) = ppid_line.split_whitespace().nth(1) {
                    if let Ok(comm) = std::fs::read_to_string(format!("/proc/{}/comm", ppid)) {
                        let name = comm.trim();
                        if let Some(s) = Shell::from_name(name) {
                            return Some(s);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Print a completions snippet for `<app> completions` (no `--shell`)
/// — the convenience entry point users typically call once at shell
/// startup. Auto-detects the user's shell and emits a comment header
/// plus an indirect-`source <(...)` line that pulls the actual script
/// from `<app> completions --shell <shell>`.
///
/// The indirect form is what makes `eval "$(myapp completions)"`
/// safe regardless of which shell the user is in: the heredoc content
/// (which contains backslashes, `$`, etc.) is sourced from a
/// subshell rather than substituted into the caller's `eval` argument.
///
/// Falls back to a help message if shell detection fails.
pub fn print_indirect_wrapper(app_name: &str) {
    // Use argv[0] exactly as the user invoked us. If they typed
    // `veks completions`, emit `source <(veks completions ...)`.
    // If they typed `./target/release/veks completions`, emit that
    // path. The point: the snippet they paste into ~/.bashrc should
    // re-invoke the binary the *same way* they just did, not via a
    // canonicalised absolute path that may not exist on a different
    // machine, in a different toolchain, or after a `cargo install`.
    // We deliberately do NOT call `std::env::current_exe()` here —
    // that resolves symlinks and ignores how the user actually ran
    // the binary.
    let app_path = std::env::args_os()
        .next()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| app_name.to_string());

    match detect_shell() {
        Some(Shell::Bash) => {
            println!("# {} tab-completion for bash", app_name);
            println!("# To activate:  eval \"$({} completions)\"", app_name);
            println!("# To persist:   echo 'eval \"$({} completions)\"' >> ~/.bashrc", app_name);
            println!("source <(\"{}\" completions --shell bash)", app_path);
        }
        Some(Shell::Zsh) => {
            println!("# {} tab-completion for zsh", app_name);
            println!("# To activate:  eval \"$({} completions)\"", app_name);
            println!("# To persist:   echo 'eval \"$({} completions)\"' >> ~/.zshrc", app_name);
            println!("source <(\"{}\" completions --shell zsh)", app_path);
        }
        Some(Shell::Fish) => {
            println!("# {} tab-completion for fish", app_name);
            println!("# To activate:  eval ({} completions)", app_name);
            println!("# To persist:   add to ~/.config/fish/config.fish");
            println!("\"{}\" completions --shell fish | source", app_path);
        }
        Some(other) => {
            // Recognized shell but no auto-wrapper format defined;
            // emit the direct script.
            print_completions(app_name, other);
        }
        None => {
            println!("# {0}: could not detect your shell.", app_name);
            println!("# Use: eval \"$({0} completions --shell bash)\"", app_name);
        }
    }
}

/// Print a direct completion script for `<app> completions --shell
/// <shell>`. This is what the indirect wrapper sources at shell
/// startup; callers can invoke it directly when they want the raw
/// script in stdout.
///
/// `Bash` is fully implemented. `Zsh` reuses the bash script via
/// bash-compatible mode (with a stderr note). `Fish`, `Elvish`, and
/// `PowerShell` print a stderr stub indicating they're not yet
/// implemented but accept the shell choice without panicking.
pub fn print_completions(app_name: &str, shell: Shell) {
    match shell {
        Shell::Bash => print_bash_script(app_name),
        Shell::Zsh => {
            eprintln!("# zsh completions: using bash-compatible mode");
            print_bash_script(app_name);
        }
        Shell::Fish | Shell::Elvish | Shell::PowerShell => {
            eprintln!(
                "# {} completions for `{}` are not yet implemented",
                shell.name(), app_name,
            );
        }
    }
}

/// Generate a bash completion script that calls back into the app.
///
/// The emitted script is intentionally minimal — a single
/// `complete -F` registration plus a body that hands the raw
/// `$COMP_LINE` and `$COMP_POINT` to the binary. All
/// word-splitting and candidate logic runs in Rust inside
/// [`handle_complete_env`]; the bash side never sees the
/// completion rules. That keeps user-facing behavior from
/// drifting between shell and binary on upgrades — the only
/// thing that can break is the (trivial) handoff itself.
pub fn print_bash_script(app_name: &str) {
    let env_var = format!("_{}_COMPLETE", app_name.to_uppercase().replace('-', "_"));

    // Echo argv[0] verbatim — whatever the user typed when invoking
    // the binary (`veks`, `./target/release/veks`,
    // `/usr/local/bin/veks`, etc.). Resist the temptation to
    // canonicalise via `current_dir().join(...)` or
    // `std::env::current_exe()` — both produce paths that survive
    // `cargo install`/symlinks/PATH-rebinding worse than the bare
    // argv[0] does, and both surprise users who explicitly chose to
    // call the binary by short name.
    let completer = std::env::args_os()
        .next()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| app_name.to_string());

    // `-o nosort` is critical: the Rust completion engine returns
    // candidates in *layer order* (level-1 entries first, then
    // level-2, …, alphabetical within each layer). Without
    // `nosort`, readline re-sorts the whole list alphabetically and
    // visually scrambles the layers — making rapid-tap stratification
    // pointless.
    print!(r#"_{app}_complete() {{
    local IFS=$'\n'
    COMPREPLY=($({env_var}=bash _COMP_SHELL_PID=$$ "{completer}" "$COMP_LINE" "$COMP_POINT" 2>/dev/null))
    if [[ ${{#COMPREPLY[@]}} -ge 1 ]] \
        && [[ "${{COMPREPLY[0]}}" == *= || "${{COMPREPLY[0]}}" == */ ]]; then
        compopt -o nospace 2>/dev/null
    fi
}}
complete -o nosort -F _{app}_complete {app}
"#,
        app = app_name,
        env_var = env_var,
        completer = completer,
    );
}

/// Tokenize a shell line up to `point`, returning the prior
/// completed words and the current (in-progress) token. Honors
/// single and double quotes and `\` escapes; preserves `=` as
/// part of a token (so `key=value` stays one word). The first
/// token (the binary name) is dropped — callers already know it.
fn split_line(line: &str, point: usize) -> (Vec<String>, String) {
    let point = point.min(line.len());
    let head = &line[..point];
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_quote: Option<char> = None;
    let mut chars = head.chars().peekable();
    while let Some(ch) = chars.next() {
        match in_quote {
            Some(q) if ch == q => { in_quote = None; }
            Some(_) => cur.push(ch),
            None => match ch {
                '\'' | '"' => { in_quote = Some(ch); }
                '\\' => {
                    if let Some(next) = chars.next() { cur.push(next); }
                }
                ' ' | '\t' => {
                    if !cur.is_empty() {
                        words.push(std::mem::take(&mut cur));
                    }
                }
                _ => cur.push(ch),
            }
        }
    }
    if !words.is_empty() { words.remove(0); }
    (words, cur)
}

/// Check for completion env vars and handle them.
///
/// Expects the bash shim emitted by [`print_bash_script`] —
/// `<binary> "$COMP_LINE" "$COMP_POINT"` — and tokenizes the
/// line in Rust before walking the [`CommandTree`].
pub fn handle_complete_env(app_name: &str, tree: &CommandTree) -> bool {
    let env_var = format!("_{}_COMPLETE", app_name.to_uppercase().replace('-', "_"));
    let is_ours = std::env::var(&env_var).ok().as_deref() == Some("bash");
    let is_legacy = std::env::var("COMPLETE").ok().as_deref() == Some("bash");
    if !is_ours && !is_legacy {
        return false;
    }

    let argv: Vec<String> = std::env::args().collect();
    let line = argv.get(1).cloned().unwrap_or_default();
    let point: usize = argv.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(line.len());

    let (prior, cur) = split_line(&line, point);
    // The downstream API takes a `words: &[&str]` shape where
    // index 0 is the binary name and the last entry is the
    // (possibly empty) word under the cursor.
    let mut words_owned: Vec<String> = vec![app_name.to_string()];
    words_owned.extend(prior);
    words_owned.push(cur);
    let words: Vec<&str> = words_owned.iter().map(|s| s.as_str()).collect();

    let input_key = words[1..].join(" ");

    // Determine the max-level of the group the cursor is currently
    // inside. `tap_detect` needs this to know when to reset the
    // persistent state (after the user has tapped through to the
    // last layer).
    let completed_for_max: &[&str] = if words.len() > 1 {
        &words[1..words.len() - 1]
    } else {
        &[]
    };
    let mut max_node = &tree.root;
    for &word in completed_for_max {
        if let Some(child) = max_node.child(word) {
            max_node = child;
        } else {
            break;
        }
    }
    let max_level = max_level_of_children(max_node).max(1);
    let tap_count = tap_detect(app_name, &input_key, max_level);

    // Rotating-tier completion with cumulative supersets:
    //
    //   tap 1 (cold)         → layer 1
    //   tap 2 (within 200ms) → layers 1 + 2 (cumulative)
    //   tap 3 (within 200ms) → layers 1 + 2 + 3 (cumulative, max)
    //                          ↑ persistent state resets here
    //   tap 4 (within 200ms) → layer 1 (fresh, because state was reset)
    //   …
    //
    // Within each cumulative result, candidates are sorted in layer
    // order (layer 1 first, then layer 2, …). Trees that haven't
    // categorized their commands have max_level == 1, so every tap
    // returns the same layer-1 set (no stratification visible).
    let candidates = complete_rotating(tree, &words, tap_count);

    for candidate in candidates {
        println!("{}", candidate);
    }

    true
}

fn tap_detect(app_name: &str, input_key: &str, max_level: u32) -> u32 {
    use std::io::Write;

    // Cadence:
    //   • A tap that comes within ADVANCE_MS of the previous tap on
    //     the same input key advances to the next layer.
    //   • Any other tap (cold, idle past ADVANCE_MS, or a key change)
    //     starts fresh at layer 1.
    //   • Once the advance reaches `max_level` (the deepest layer the
    //     current group has), the persisted count is reset to 0. The
    //     next tap therefore starts fresh at layer 1, even if it
    //     comes within ADVANCE_MS — closing the cycle.
    //
    // Concretely, with max_level = 3 and rapid tapping:
    //   tap → layer 1   (persist 1)
    //   tap → layer 2   (persist 2)
    //   tap → layer 3   (persist 0  ← reset on reaching max)
    //   tap → layer 1   (persist 1)  ← cycle restarts
    const ADVANCE_MS: u128 = 200;
    let max = max_level.max(1);

    let ppid = std::env::var("_COMP_SHELL_PID")
        .or_else(|_| std::env::var("PPID"))
        .unwrap_or_else(|_| "0".to_string());
    let tap_file = std::path::PathBuf::from(format!("/tmp/.{}_tap_{}", app_name, ppid));
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    // Compare key normalized to the same shape on read AND write so
    // a trailing space (common when the user just typed a separator
    // before pressing TAB) doesn't trip the same-key check.
    let cur_key = input_key.trim_end();

    let mut tap_count = 1u32;
    if let Ok(content) = std::fs::read_to_string(&tap_file) {
        let mut parts = content.splitn(3, ' ');
        let prev_time: u128 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let prev_count: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let prev_key = parts.next().unwrap_or("").trim_end();

        if prev_key == cur_key && now_ms.saturating_sub(prev_time) < ADVANCE_MS {
            // Rapid follow-up on the same input — advance one layer,
            // capped at max. A persisted prev_count of 0 means the
            // previous tap reset the cycle, so we advance from 0 → 1
            // (a fresh start at layer 1).
            tap_count = prev_count.saturating_add(1).min(max);
        }
    }

    // Persist for the NEXT tap. When we just hit the deepest layer,
    // reset to 0 so the next tap (rapid or otherwise) starts fresh
    // at layer 1.
    let to_persist = if tap_count >= max { 0 } else { tap_count };
    if let Ok(mut f) = std::fs::File::create(&tap_file) {
        let _ = write!(f, "{} {} {}", now_ms, to_persist, cur_key);
    }

    tap_count
}

/// Expanded completion: show all `group command` pairs.
pub fn complete_expanded(tree: &CommandTree, words: &[&str]) -> Vec<String> {
    let partial = if words.len() > 1 { words.last().unwrap_or(&"") } else { &"" };
    let completed = if words.len() > 2 { &words[1..words.len() - 1] } else { &[] };

    if !completed.is_empty() || !partial.is_empty() {
        return complete(tree, words);
    }

    let mut results = Vec::new();
    if let Node::Group { children, .. } = &tree.root {
        for (name, node) in children {
            if name == "help" || name.starts_with('-') {
                continue;
            }
            match node {
                Node::Group { children: sub, .. } if !sub.is_empty() => {
                    for sub_name in sub.keys() {
                        results.push(format!("{} {}", name, sub_name));
                    }
                }
                _ => {
                    if !tree.hidden.contains(name.as_str()) {
                        results.push(name.to_string());
                    }
                }
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tree() -> CommandTree {
        CommandTree::new("testapp")
            .command("run", Node::leaf_with_flags(
                &["cycles=", "threads=", "adapter=", "workload="],
                &["--strict", "--tui"],
            ).with_dynamic_options(dynamic_workload_params))
            .command("bench", Node::group(vec![
                ("gk", Node::leaf_with_flags(
                    &["cycles=", "threads=", "--cycles", "--threads"],
                    &["--explain"],
                )),
            ]))
    }

    /// Test dynamic options provider: if workload=X is on the line,
    /// return extra params that the workload declares.
    fn dynamic_workload_params(_partial: &str, context: &[&str]) -> Vec<String> {
        // Find workload= on the context
        for word in context {
            if let Some(path) = word.strip_prefix("workload=") {
                if path == "test_keyvalue.yaml" {
                    return vec!["keyspace=".into(), "table=".into(), "keycount=".into()];
                }
            }
        }
        Vec::new()
    }

    #[test]
    fn root_completions() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", ""]);
        assert!(candidates.contains(&"bench".to_string()));
        assert!(candidates.contains(&"run".to_string()));
    }

    #[test]
    fn run_shows_all_options() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", "run", ""]);
        assert!(candidates.contains(&"cycles=".to_string()));
        assert!(candidates.contains(&"--strict".to_string()));
        assert!(candidates.contains(&"adapter=".to_string()));
    }

    #[test]
    fn run_filters_consumed_bare_param() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", "run", "cycles=1000", ""]);
        assert!(!candidates.contains(&"cycles=".to_string()));
        assert!(candidates.contains(&"threads=".to_string()));
        assert!(candidates.contains(&"--strict".to_string()));
    }

    #[test]
    fn run_filters_consumed_flag() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", "run", "--strict", ""]);
        assert!(!candidates.contains(&"--strict".to_string()));
        assert!(candidates.contains(&"cycles=".to_string()));
    }

    #[test]
    fn bench_gk_filters_consumed() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", "bench", "gk", "expr", "--cycles=1000000", "--threads=20", ""]);
        assert!(!candidates.contains(&"--cycles".to_string()));
        assert!(!candidates.contains(&"cycles=".to_string()));
        assert!(!candidates.contains(&"--threads".to_string()));
        assert!(!candidates.contains(&"threads=".to_string()));
        assert!(candidates.contains(&"--explain".to_string()));
    }

    #[test]
    fn partial_match_bare_param() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", "run", "cy"]);
        assert!(candidates.contains(&"cycles=".to_string()));
        assert!(!candidates.contains(&"--strict".to_string()));
    }

    #[test]
    fn dynamic_options_from_workload() {
        let tree = test_tree();
        // When workload=test_keyvalue.yaml is on the line, dynamic params appear
        let candidates = complete(&tree, &["testapp", "run", "workload=test_keyvalue.yaml", ""]);
        assert!(candidates.contains(&"keyspace=".to_string()), "dynamic param 'keyspace=' should appear");
        assert!(candidates.contains(&"table=".to_string()), "dynamic param 'table=' should appear");
        assert!(candidates.contains(&"keycount=".to_string()), "dynamic param 'keycount=' should appear");
        // Static options should still be present
        assert!(candidates.contains(&"--strict".to_string()));
        // workload= should be consumed
        assert!(!candidates.contains(&"workload=".to_string()));
    }

    #[test]
    fn dynamic_options_filtered_when_consumed() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", "run", "workload=test_keyvalue.yaml", "keyspace=mykeyspace", ""]);
        assert!(!candidates.contains(&"keyspace=".to_string()), "keyspace= already used");
        assert!(candidates.contains(&"table=".to_string()), "table= still available");
    }

    #[test]
    fn dynamic_options_partial_match() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", "run", "workload=test_keyvalue.yaml", "key"]);
        assert!(candidates.contains(&"keyspace=".to_string()));
        assert!(candidates.contains(&"keycount=".to_string()));
        assert!(!candidates.contains(&"table=".to_string()), "table= doesn't start with 'key'");
    }

    #[test]
    fn no_dynamic_options_without_workload() {
        let tree = test_tree();
        let candidates = complete(&tree, &["testapp", "run", ""]);
        assert!(!candidates.contains(&"keyspace=".to_string()), "no workload= means no dynamic params");
    }

    #[test]
    fn word_matches_exact_flag() {
        assert!(word_matches_option("--strict", "--strict"));
        assert!(!word_matches_option("--strict", "--tui"));
    }

    #[test]
    fn word_matches_bare_key_value() {
        assert!(word_matches_option("cycles=1000", "cycles="));
        assert!(!word_matches_option("threads=4", "cycles="));
    }

    #[test]
    fn word_matches_dashed_to_bare_equivalence() {
        assert!(word_matches_option("--cycles=1000", "cycles="));
        assert!(word_matches_option("cycles=1000", "--cycles"));
    }

    // ---- stratified tap completion ----

    fn stratified_tree() -> CommandTree {
        CommandTree::new("nbrs")
            .command("run",
                Node::leaf(&["--cycles="])
                    .with_category("workloads").with_level(1))
            .command("--inspector",
                Node::leaf(&[])
                    .with_category("tools").with_level(2))
            .command("--summary",
                Node::leaf(&[])
                    .with_category("tools").with_level(2))
            .command("describe",
                Node::leaf(&[])
                    .with_category("documentation").with_level(3))
            .command("bench",
                Node::leaf(&[])
                    .with_category("benchmark").with_level(3))
    }

    #[test]
    fn tap1_shows_only_level1_commands() {
        let tree = stratified_tree();
        let cands = complete_at_tap(&tree, &["nbrs"], 1);
        assert_eq!(cands, vec!["run".to_string()]);
    }

    #[test]
    fn tap2_adds_level2_commands() {
        let tree = stratified_tree();
        let cands = complete_at_tap(&tree, &["nbrs"], 2);
        assert!(cands.contains(&"run".to_string()));
        assert!(cands.contains(&"--inspector".to_string()));
        assert!(cands.contains(&"--summary".to_string()));
        assert!(!cands.contains(&"describe".to_string()),
            "level-3 'describe' should not appear at tap 2");
    }

    #[test]
    fn tap3_shows_everything() {
        let tree = stratified_tree();
        let cands = complete_at_tap(&tree, &["nbrs"], 3);
        assert!(cands.contains(&"run".to_string()));
        assert!(cands.contains(&"--inspector".to_string()));
        assert!(cands.contains(&"describe".to_string()));
        assert!(cands.contains(&"bench".to_string()));
    }

    #[test]
    fn level_filter_does_not_block_partial_match() {
        // Typing `--ins` should still complete to `--inspector`
        // even at tap 1, where level-2 commands aren't shown
        // empty-prefix. Once the user has typed a prefix,
        // they've signaled intent for that specific command.
        let tree = stratified_tree();
        let cands = complete_at_tap(&tree, &["nbrs", "--ins"], 1);
        assert!(cands.contains(&"--inspector".to_string()),
            "partial-prefix matches should bypass the tap-tier filter");
    }

    #[test]
    fn rotating_tap1_baseline_is_layer1_only() {
        // The production path is `complete_rotating` (used by
        // `handle_complete_env`). At tap=1 it shows only layer-1
        // commands, in alphabetical order.
        let tree = stratified_tree();
        let cands = complete_rotating(&tree, &["nbrs"], 1);
        assert_eq!(cands, vec!["run".to_string()]);
    }

    #[test]
    fn rotating_tap2_is_cumulative_superset() {
        // At tap=2 (rapid double-tap), the result is the *cumulative
        // superset* — every layer-1 command plus every layer-2
        // command, never just layer 2 alone.
        let tree = stratified_tree();
        let cands = complete_rotating(&tree, &["nbrs"], 2);
        assert!(cands.contains(&"run".to_string()),
            "layer-1 'run' must remain visible at tap 2");
        assert!(cands.contains(&"--inspector".to_string()),
            "layer-2 '--inspector' must appear at tap 2");
        assert!(cands.contains(&"--summary".to_string()),
            "layer-2 '--summary' must appear at tap 2");
    }

    #[test]
    fn rotating_tap2_orders_by_layer() {
        // Within the cumulative result, candidates must be sorted
        // *layer-first* — every layer-1 entry precedes every layer-2
        // entry — and within a layer, `--`-flags last + alphabetical.
        let tree = stratified_tree();
        let cands = complete_rotating(&tree, &["nbrs"], 2);
        let pos = |name: &str| cands.iter().position(|s| s == name).unwrap();
        assert!(pos("run") < pos("--inspector"),
            "layer-1 'run' must precede layer-2 '--inspector'");
        assert!(pos("run") < pos("--summary"),
            "layer-1 'run' must precede layer-2 '--summary'");
        assert!(pos("--inspector") < pos("--summary"),
            "within a layer, alphabetical: '--inspector' before '--summary'");
    }

    #[test]
    fn rotating_tap3_at_max_includes_all_layers() {
        // At tap=3 (third rapid tap on a 3-layer tree), the
        // cumulative result must include every layer 1, 2, and 3
        // candidate, in layer order.
        let tree = stratified_tree();
        let cands = complete_rotating(&tree, &["nbrs"], 3);
        assert!(cands.contains(&"run".to_string()));
        assert!(cands.contains(&"--inspector".to_string()));
        assert!(cands.contains(&"--summary".to_string()));
        assert!(cands.contains(&"describe".to_string()));
        assert!(cands.contains(&"bench".to_string()));

        let pos = |name: &str| cands.iter().position(|s| s == name).unwrap();
        // Layer 1 (run) before layer 2 (inspector/summary) before
        // layer 3 (bench, describe). `bench` and `describe` are both
        // layer 3; alphabetical within the layer keeps `bench` first.
        assert!(pos("run") < pos("--inspector"));
        assert!(pos("--summary") < pos("bench"));
        assert!(pos("--summary") < pos("describe"));
        assert!(pos("bench") < pos("describe"));
    }

    #[test]
    fn nodes_without_level_default_to_visible() {
        // Backward-compat: a node that never called
        // `with_level` resolves to DEFAULT_LEVEL = 1 and is
        // visible from tap 1. Apps that haven't migrated to
        // categorized completion see no behavior change.
        let tree = CommandTree::new("legacy")
            .command("run", Node::leaf(&[]))
            .command("describe", Node::leaf(&[]));
        let cands = complete_at_tap(&tree, &["legacy"], 1);
        assert!(cands.contains(&"run".to_string()));
        assert!(cands.contains(&"describe".to_string()));
    }

    // ---- strict-metadata: type-state enforcement ----

    #[test]
    fn strict_node_with_full_metadata_compiles() {
        // The success case — adding a fully-tagged node
        // through `strict_command` is the canonical strict
        // mode usage. The compiler doesn't reject this.
        let _tree = CommandTree::new("app")
            .strict_command(
                "run",
                StrictNode::leaf(&["--cycles="])
                    .with_category("workloads")
                    .with_level(1),
            );
    }

    // The next two are intentionally `#[ignore]`d compile-fail
    // demonstrations. They live here as documentation rather
    // than tests, since `cargo test` won't try to build them
    // unless explicitly invoked, but a reader can uncomment to
    // verify the gate fires.
    //
    // ```compile_fail,ignore
    // CommandTree::new("app").strict_command(
    //     "bad",
    //     StrictNode::leaf(&[]).with_category("x"),  // missing with_level
    // );
    // ```
    //
    // ```compile_fail,ignore
    // CommandTree::new("app").strict_command(
    //     "bad",
    //     StrictNode::leaf(&[]).with_level(1),  // missing with_category
    // );
    // ```

    // ---- runtime-validation path ----

    #[test]
    fn runtime_validate_reports_missing_metadata() {
        let tree = CommandTree::new("app")
            .command("run",
                Node::leaf(&[]).with_category("workloads").with_level(1))
            .command("undertagged", Node::leaf(&[]));
        let errors = tree.validate().unwrap_err();
        assert!(errors.iter().any(|e| matches!(e,
            MetadataError::MissingCategory { command } if command == "undertagged")));
        assert!(errors.iter().any(|e| matches!(e,
            MetadataError::MissingLevel { command } if command == "undertagged")));
        // The properly-tagged 'run' should not appear in errors.
        assert!(!errors.iter().any(|e| matches!(e,
            MetadataError::MissingCategory { command } if command == "run")));
    }

    #[test]
    fn runtime_validate_passes_when_all_tagged() {
        let tree = CommandTree::new("app")
            .command("run",
                Node::leaf(&[]).with_category("workloads").with_level(1))
            .command("describe",
                Node::leaf(&[]).with_category("docs").with_level(3));
        assert!(tree.validate().is_ok());
    }

    #[test]
    #[should_panic(expected = "without Node::with_category")]
    fn require_metadata_panics_on_undertagged_command() {
        let _tree = CommandTree::new("app")
            .require_metadata()
            .command("bad", Node::leaf(&[])); // missing both
    }
}
