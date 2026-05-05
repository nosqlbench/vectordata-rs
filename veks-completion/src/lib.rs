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

pub mod providers;

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

/// A closed set of valid values for a flag. Both static (`&'static
/// [&'static str]`) and runtime-owned (`Vec<String>`) variants are
/// supported via the same query API.
///
/// Solves two TODO gaps in one type:
///
/// - **Item 1** (no per-set glue functions): callers no longer need
///   to write `fn palette_provider(...) -> Vec<String> { palette.iter()
///   .filter(...).collect() }` boilerplate per closed set. Just
///   construct a [`ClosedValues`] and convert to a [`ValueProvider`]
///   via [`ClosedValues::into_provider`].
/// - **Item 5** (validation surface): the same declaration that
///   drives completion can drive parser-side validation —
///   [`ClosedValues::validate`] returns `true` for any value the
///   completer would have offered.
///
/// ```
/// use veks_completion::ClosedValues;
///
/// let metrics = ClosedValues::Static(&["L2", "IP", "COSINE"]);
///
/// // Completion: prefix-filtered.
/// assert_eq!(metrics.complete(""), vec!["L2", "IP", "COSINE"]);
/// assert_eq!(metrics.complete("CO"), vec!["COSINE"]);
///
/// // Validation: exact set membership.
/// assert!(metrics.validate("L2"));
/// assert!(!metrics.validate("bogus"));
///
/// // Convert to a ValueProvider for tree registration.
/// let provider = metrics.clone().into_provider();
/// assert_eq!(provider("CO", &[]), vec!["COSINE"]);
/// ```
#[derive(Debug, Clone)]
pub enum ClosedValues {
    /// Borrowed `&'static` slice — preferred when the set is known
    /// at compile time (the common case).
    Static(&'static [&'static str]),
    /// Heap-owned values — for runtime-built specs whose closed set
    /// isn't known until the binary inspects its environment.
    Owned(Vec<String>),
}

impl ClosedValues {
    /// Iterate over the values as `&str` regardless of variant.
    pub fn values(&self) -> Vec<&str> {
        match self {
            ClosedValues::Static(s) => s.to_vec(),
            ClosedValues::Owned(v) => v.iter().map(|s| s.as_str()).collect(),
        }
    }

    /// Prefix-filtered completion candidates.
    pub fn complete(&self, partial: &str) -> Vec<String> {
        match self {
            ClosedValues::Static(s) => s
                .iter()
                .filter(|v| v.starts_with(partial))
                .map(|v| (*v).to_string())
                .collect(),
            ClosedValues::Owned(v) => v
                .iter()
                .filter(|val| val.starts_with(partial))
                .cloned()
                .collect(),
        }
    }

    /// Membership check — `true` iff `value` is in the set.
    pub fn validate(&self, value: &str) -> bool {
        match self {
            ClosedValues::Static(s) => s.iter().any(|v| *v == value),
            ClosedValues::Owned(v) => v.iter().any(|val| val == value),
        }
    }

    /// Wrap as a [`ValueProvider`] for use with
    /// [`Node::with_value_provider`] /
    /// [`CommandTree::global_value_provider`]. The set is moved into
    /// the closure; clone the [`ClosedValues`] first if you also
    /// need to keep it for validation.
    pub fn into_provider(self) -> ValueProvider {
        std::sync::Arc::new(move |partial: &str, _ctx: &[&str]| self.complete(partial))
    }
}

/// Convenience: hand any [`ClosedValues`] to APIs that take a
/// [`ValueProvider`] without explicit `.into_provider()`.
impl From<ClosedValues> for ValueProvider {
    fn from(cv: ClosedValues) -> Self {
        cv.into_provider()
    }
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
///
/// A node in the command tree. Carries everything a command-tree
/// node *can* have: subcommand children, flags, providers,
/// discovery metadata, help text, and a free-form attachment slot.
///
/// "Leaf" and "Group" are no longer separate variants. A node with
/// no children is leaf-shaped; a node with children is group-shaped;
/// a node with both is hybrid (e.g., `report --workload x base`,
/// where `report` accepts `--workload` *and* has a `base` subcommand).
/// Walkers branch on `children.is_empty()` only when the distinction
/// actually matters.
///
/// All builder methods (`with_*`) return `self` so they chain.
/// Methods that operate on children (e.g. `with_child`) work on
/// any node — calling them just adds the child, regardless of
/// whether the node was previously leaf-shaped or not.
#[derive(Clone)]
pub struct Node {
    // ---- discovery / display ----
    /// Display group tag — see [`CategoryTag`] for usage.
    category: Option<String>,
    /// Tap-tier visibility. `None` ⇒ "never explicitly set"; the
    /// effective level resolves to [`DEFAULT_LEVEL`], but
    /// strict-metadata mode treats `None` as missing.
    level: Option<u32>,
    /// One-line `--help` summary. Set via [`Node::with_help`].
    help: Option<String>,

    // ---- subcommand children ----
    /// Named children. Empty ⇒ leaf-shaped.
    children: BTreeMap<String, Node>,

    // ---- flags this node accepts ----
    /// All flag names (value-taking + boolean), in declared order.
    flags: Vec<String>,
    /// Subset of `flags` that don't take a value.
    boolean_flags: std::collections::HashSet<String>,
    /// Per-flag help text. Used by [`render_usage`].
    flag_help: BTreeMap<String, String>,
    /// Dynamic value providers keyed by flag name.
    value_providers: BTreeMap<String, ValueProvider>,

    // ---- discovery extras ----
    /// Optional provider that discovers additional `key=` options
    /// from context (e.g., workload-file parameters).
    dynamic_options: Option<DynamicOptionsProvider>,
    /// Per-node staged tree-globals — promoted to the
    /// [`CommandTree`]'s global registry by
    /// [`CommandTree::lift_promoted_globals`].
    promoted_globals: Vec<(String, ValueProvider)>,
    /// Context-aware completion override that fires whenever the
    /// cursor sits inside this subtree.
    subtree_provider: Option<SubtreeProvider>,
    /// Free-form attachment slot. Downstream crates use this to
    /// carry handler payloads, parser state, dispatch rules, etc.,
    /// without forcing this crate to grow generics.
    extras: Option<Extras>,
}

/// Type alias for group-level context-aware completion providers
/// (TODO item 7). Receives a structured [`PartialParse`] of the
/// command line state and returns candidates to merge into the
/// completer's output.
pub type SubtreeProvider =
    std::sync::Arc<dyn Fn(&PartialParse) -> Vec<String> + Send + Sync>;

/// Free-form payload slot on a Node (TODO item 8). Wraps an
/// `Arc<dyn Any + Send + Sync>` so embedders can attach handler
/// types, parser state, or anything else without forcing
/// veks-completion to grow generic parameters or hard dependencies.
///
/// Recover the payload via `Arc::downcast` on the inner Arc.
#[derive(Clone)]
pub struct Extras(pub std::sync::Arc<dyn std::any::Any + Send + Sync>);

impl std::fmt::Debug for Extras {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extras").field("type_id", &self.0.type_id()).finish()
    }
}

impl Extras {
    /// Wrap any `Send + Sync + 'static` value as an extras payload.
    pub fn new<T: std::any::Any + Send + Sync + 'static>(value: T) -> Self {
        Extras(std::sync::Arc::new(value))
    }

    /// Try to downcast to a concrete type. Returns `None` if the
    /// payload was attached as a different type.
    pub fn downcast<T: std::any::Any + Send + Sync + 'static>(
        &self,
    ) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

/// Structured snapshot of the partial command-line state at the
/// cursor position. Passed to subtree providers so they can offer
/// context-aware completions without re-tokenising `COMP_LINE`.
///
/// Carries both the **whitespace-tokenised** view (`completed`,
/// `partial`, `tree_path` — the same shape every veks-completion
/// flow uses) AND the **raw line + cursor offset** that grammar-aware
/// providers (e.g. for embedded query DSLs like MetricsQL or PromQL)
/// need to resolve quote / bracket / operator state. Callers that
/// don't have raw context populate `raw_line` with an empty string
/// and `cursor_offset` with `0` — grammar helpers fall back to the
/// tokenised view in that case.
#[derive(Debug, Clone)]
pub struct PartialParse<'a> {
    /// Words the user has already completed (whitespace-separated,
    /// program name excluded).
    pub completed: Vec<&'a str>,
    /// The partial word currently under the cursor (may be empty).
    pub partial: &'a str,
    /// Path through the command tree that resolved against
    /// `completed`. Same shape as `completed` but only the prefix
    /// that maps to actual nodes.
    pub tree_path: Vec<&'a str>,
    /// Raw `COMP_LINE` (or equivalent) — the full command line as
    /// the user typed it, before any tokenisation. Empty when the
    /// caller didn't have it.
    pub raw_line: &'a str,
    /// Byte offset of the cursor within `raw_line`. `0` when
    /// `raw_line` is empty.
    pub cursor_offset: usize,
}

/// Bracket / quote depth at the cursor, computed by
/// [`PartialParse::bracket_state`]. Lets a grammar-aware provider
/// answer "am I inside a `{...}`, `(...)`, `[...]`, or quoted
/// string?" without re-implementing the scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BracketState {
    /// Net `(` minus `)` count up to the cursor. Negative ⇒ extra
    /// closes (likely user error).
    pub paren: i32,
    /// Net `{` minus `}` count up to the cursor.
    pub brace: i32,
    /// Net `[` minus `]` count up to the cursor.
    pub bracket: i32,
    /// `Some(quote)` when the cursor sits inside an unclosed
    /// quote of the indicated kind (`"` or `'`); `None` otherwise.
    pub inside_quote: Option<char>,
}

impl<'a> PartialParse<'a> {
    /// Slice of `raw_line` strictly before the cursor. Empty when
    /// `raw_line` is empty.
    pub fn before_cursor(&self) -> &'a str {
        if self.raw_line.is_empty() { return ""; }
        &self.raw_line[..self.cursor_offset.min(self.raw_line.len())]
    }

    /// Slice of `raw_line` from the cursor to the end.
    pub fn after_cursor(&self) -> &'a str {
        if self.raw_line.is_empty() { return ""; }
        &self.raw_line[self.cursor_offset.min(self.raw_line.len())..]
    }

    /// Compute the bracket / quote state at the cursor by linearly
    /// scanning `before_cursor`. Honors quotes (everything inside
    /// `"…"` or `'…'` is counted as string content, brackets within
    /// don't shift the depth) and supports backslash-escapes inside
    /// quotes.
    ///
    /// When `raw_line` is empty (caller didn't supply it), returns
    /// the default zero-depth state.
    pub fn bracket_state(&self) -> BracketState {
        let s = self.before_cursor();
        let mut state = BracketState::default();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if let Some(q) = state.inside_quote {
                if c == '\\' {
                    // Skip the next character (escaped).
                    chars.next();
                    continue;
                }
                if c == q {
                    state.inside_quote = None;
                }
                continue;
            }
            match c {
                '(' => state.paren += 1,
                ')' => state.paren -= 1,
                '{' => state.brace += 1,
                '}' => state.brace -= 1,
                '[' => state.bracket += 1,
                ']' => state.bracket -= 1,
                '"' | '\'' => state.inside_quote = Some(c),
                _ => {}
            }
        }
        state
    }

    /// Last non-whitespace, non-identifier character before the
    /// cursor, scanning back over identifier characters first.
    /// Useful for "what symbol triggered this completion?" — e.g.,
    /// `=` after `label` means we're in a label-value position.
    /// Returns `None` if the only thing before the cursor is
    /// identifier characters or whitespace.
    pub fn trigger_char(&self) -> Option<char> {
        let s = self.before_cursor();
        let mut chars = s.chars().rev();
        // Skip current identifier-ish run.
        while let Some(c) = chars.clone().next() {
            if is_ident_char(c) {
                chars.next();
            } else {
                break;
            }
        }
        chars.next()
    }

    /// Identifier (or partial identifier) immediately to the left
    /// of the cursor. For input `up{job=`, returns `""` (cursor is
    /// right after `=`, so the partial-ident before the cursor is
    /// empty). For input `up{jo`, returns `"jo"`.
    pub fn ident_before_cursor(&self) -> &'a str {
        let s = self.before_cursor();
        let bytes = s.as_bytes();
        let mut i = bytes.len();
        while i > 0 {
            let c = bytes[i - 1] as char;
            if is_ident_char(c) {
                i -= 1;
            } else {
                break;
            }
        }
        &s[i..]
    }
}

#[inline]
fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == ':'
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("category", &self.category)
            .field("level", &self.level)
            .field("help", &self.help)
            .field("children", &self.children)
            .field("flags", &self.flags)
            .field("boolean_flags", &self.boolean_flags)
            .field("flag_help", &self.flag_help.keys().collect::<Vec<_>>())
            .field("value_providers", &self.value_providers.keys().collect::<Vec<_>>())
            .field("has_dynamic_options", &self.dynamic_options.is_some())
            .field("promoted_globals", &self.promoted_globals.iter().map(|(k, _)| k).collect::<Vec<_>>())
            .field("has_subtree_provider", &self.subtree_provider.is_some())
            .field("has_extras", &self.extras.is_some())
            .finish()
    }
}

impl Default for Node {
    fn default() -> Self {
        Node {
            category: None,
            level: None,
            help: None,
            children: BTreeMap::new(),
            flags: Vec::new(),
            boolean_flags: std::collections::HashSet::new(),
            flag_help: BTreeMap::new(),
            value_providers: BTreeMap::new(),
            dynamic_options: None,
            promoted_globals: Vec::new(),
            subtree_provider: None,
            extras: None,
        }
    }
}

impl Node {
    /// Empty node — no flags, no children, no metadata. Build up
    /// from here using the `with_*` builders.
    pub fn new() -> Self { Self::default() }

    /// Convenience: a leaf-shaped node carrying the supplied
    /// value-taking flags (none of them booleans).
    pub fn leaf(flags: &[&str]) -> Self {
        Node {
            flags: flags.iter().map(|s| s.to_string()).collect(),
            ..Self::default()
        }
    }

    /// Convenience: a leaf-shaped node with separate value-taking
    /// and boolean flag lists.
    pub fn leaf_with_flags(value_flags: &[&str], boolean_flags: &[&str]) -> Self {
        let all: Vec<String> = value_flags.iter()
            .chain(boolean_flags.iter())
            .map(|s| s.to_string())
            .collect();
        Node {
            flags: all,
            boolean_flags: boolean_flags.iter().map(|s| s.to_string()).collect(),
            ..Self::default()
        }
    }

    /// Convenience: a group-shaped node from a list of `(name, child)`
    /// pairs. Add flags to the group separately via [`Node::with_flags`]
    /// / [`Node::with_boolean_flags`].
    pub fn group(children: Vec<(&str, Node)>) -> Self {
        Node {
            children: children.into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            ..Self::default()
        }
    }

    /// Empty group — no children, no flags. Add via [`Node::with_child`].
    pub fn empty_group() -> Self { Self::default() }

    // ---- shape predicates -----------------------------------------

    /// Leaf-shaped if it has no children. A node with both flags AND
    /// children is *not* leaf-shaped — it's hybrid.
    pub fn is_leaf(&self) -> bool { self.children.is_empty() }

    /// Group-shaped if it has at least one child.
    pub fn is_group(&self) -> bool { !self.children.is_empty() }

    // ---- children -------------------------------------------------

    /// Add a child to this node. Works on any node; if the node was
    /// previously leaf-shaped, this turns it into a hybrid (flags +
    /// children).
    pub fn with_child(mut self, name: &str, child: Node) -> Self {
        self.children.insert(name.to_string(), child);
        self
    }

    /// Direct access to children (empty for leaf-shaped nodes).
    pub fn children(&self) -> &BTreeMap<String, Node> { &self.children }

    /// Mutable access to children — used by internal walkers.
    pub fn children_mut(&mut self) -> &mut BTreeMap<String, Node> { &mut self.children }

    /// Names of this node's children, in `BTreeMap` order.
    pub fn child_names(&self) -> Vec<&str> {
        self.children.keys().map(|k| k.as_str()).collect()
    }

    /// Look up a child by name.
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.get(name)
    }

    // ---- flags ----------------------------------------------------

    /// Add value-taking flags to this node. Idempotent — duplicates
    /// are skipped.
    pub fn with_flags(mut self, flags: &[&str]) -> Self {
        for f in flags {
            if !self.flags.iter().any(|x| x == f) {
                self.flags.push((*f).to_string());
            }
        }
        self
    }

    /// Add boolean flags (no value expected) to this node. Idempotent.
    pub fn with_boolean_flags(mut self, flags: &[&str]) -> Self {
        for f in flags {
            if !self.flags.iter().any(|x| x == f) {
                self.flags.push((*f).to_string());
            }
            self.boolean_flags.insert((*f).to_string());
        }
        self
    }

    /// All flag names this node accepts (value-taking + boolean), in
    /// declared order.
    pub fn flags(&self) -> &[String] { &self.flags }

    /// Returns `true` if `flag` is a boolean flag on this node.
    pub fn is_flag(&self, flag: &str) -> bool {
        self.boolean_flags.contains(flag)
    }

    /// Convenience accessor — same as [`Node::flags`] but as a `Vec<&str>`.
    /// Kept for callers that prefer the `&str` view.
    pub fn options(&self) -> Vec<&str> {
        self.flags.iter().map(|s| s.as_str()).collect()
    }

    /// Attach a value provider to one of this node's flags.
    pub fn with_value_provider(mut self, flag: &str, provider: ValueProvider) -> Self {
        self.value_providers.insert(flag.to_string(), provider);
        self
    }

    /// Attach the same value provider to every name in `aliases` —
    /// e.g. `--tofile` and `--to-file`.
    pub fn with_value_provider_aliases(
        mut self,
        aliases: &[&str],
        provider: ValueProvider,
    ) -> Self {
        for name in aliases {
            self.value_providers.insert((*name).to_string(), provider.clone());
        }
        self
    }

    /// Direct access to the value-provider map (used by walkers).
    pub fn value_providers(&self) -> &BTreeMap<String, ValueProvider> {
        &self.value_providers
    }

    /// Attach a dynamic options provider.
    ///
    /// The provider is called during completion to discover
    /// additional `key=` options from context (e.g., workload-file
    /// parameters).
    pub fn with_dynamic_options(mut self, provider: DynamicOptionsProvider) -> Self {
        self.dynamic_options = Some(provider);
        self
    }

    /// The attached dynamic options provider, if any.
    pub fn dynamic_options(&self) -> Option<DynamicOptionsProvider> {
        self.dynamic_options
    }

    // ---- discovery / display --------------------------------------

    /// Tag this node with a display category.
    pub fn with_category(mut self, cat: &str) -> Self {
        self.category = Some(cat.to_string());
        self
    }

    /// Get the node's category tag, if any.
    pub fn category(&self) -> Option<&str> { self.category.as_deref() }

    /// Set the tap-tier visibility for this node.
    pub fn with_level(mut self, lvl: u32) -> Self {
        self.level = Some(lvl);
        self
    }

    /// Effective tap-tier level — explicit value if set, otherwise
    /// [`DEFAULT_LEVEL`].
    pub fn level(&self) -> u32 {
        self.level.unwrap_or(DEFAULT_LEVEL)
    }

    /// Explicit tap-tier level — `None` when `with_level` was never
    /// called. Used by strict-metadata validation.
    pub fn level_explicit(&self) -> Option<u32> { self.level }

    // ---- help -----------------------------------------------------

    /// Attach a one-line help summary.
    pub fn with_help(mut self, text: &str) -> Self {
        self.help = Some(text.to_string());
        self
    }

    /// Get the node's help text, if any.
    pub fn help(&self) -> Option<&str> { self.help.as_deref() }

    /// Attach help text for one of this node's flags.
    pub fn with_flag_help(mut self, flag: &str, help: &str) -> Self {
        self.flag_help.insert(flag.to_string(), help.to_string());
        self
    }

    /// Get help text for one of this node's flags, if any.
    pub fn flag_help_for(&self, flag: &str) -> Option<&str> {
        self.flag_help.get(flag).map(|s| s.as_str())
    }

    // ---- per-node tree-global staging -----------------------------

    /// Stage a tree-global value provider on this node. Promoted to
    /// the [`CommandTree`]'s global registry when
    /// [`CommandTree::lift_promoted_globals`] runs.
    pub fn with_promoted_global(mut self, token: &str, provider: ValueProvider) -> Self {
        self.promoted_globals.push((token.to_string(), provider));
        self
    }

    /// Drain staged globals (recursively) into the supplied
    /// registry. Used by [`CommandTree::lift_promoted_globals`].
    fn drain_promoted_globals_into(
        &mut self,
        out: &mut std::collections::BTreeMap<String, ValueProvider>,
    ) {
        for (token, provider) in std::mem::take(&mut self.promoted_globals) {
            out.insert(token, provider);
        }
        for child in self.children.values_mut() {
            child.drain_promoted_globals_into(out);
        }
    }

    // ---- subtree provider -----------------------------------------

    /// Attach a context-aware completion override for this subtree.
    pub fn with_subtree_provider(mut self, provider: SubtreeProvider) -> Self {
        self.subtree_provider = Some(provider);
        self
    }

    /// The subtree provider, if any.
    pub fn subtree_provider(&self) -> Option<&SubtreeProvider> {
        self.subtree_provider.as_ref()
    }

    // ---- extras ---------------------------------------------------

    /// Attach a free-form payload (handler, parser state, etc.).
    pub fn with_extras(mut self, extras: Extras) -> Self {
        self.extras = Some(extras);
        self
    }

    /// The attached extras payload, if any.
    pub fn extras(&self) -> Option<&Extras> { self.extras.as_ref() }
}

/// Render a `--help`-style usage block for a node at the given path
/// (TODO item 6). The same model that drives tab completion drives
/// help, so the two surfaces can't drift.
///
/// Output format:
///
/// ```text
/// USAGE: <path>
///
/// <help text>
///
/// FLAGS:
///   --foo    Help text for --foo
///   --bar    (no help)
///
/// SUBCOMMANDS:
///   sub-a    Help text for sub-a
///   sub-b    Help text for sub-b
/// ```
///
/// Sections are omitted when their content is empty. Children are
/// listed in `BTreeMap` order (alphabetical).
pub fn render_usage(node: &Node, path: &[&str]) -> String {
    let mut out = String::new();
    out.push_str(&format!("USAGE: {}\n", path.join(" ")));
    if let Some(help) = node.help() {
        out.push('\n');
        out.push_str(help);
        out.push('\n');
    }

    // Flags section.
    if !node.flags.is_empty() {
        out.push_str("\nFLAGS:\n");
        let width = node.flags.iter().map(|f| f.len()).max().unwrap_or(0);
        for f in &node.flags {
            let h = node.flag_help_for(f).unwrap_or("");
            out.push_str(&format!("  {:width$}  {}\n", f, h, width = width));
        }
    }

    // Subcommands section.
    if !node.children.is_empty() {
        out.push_str("\nSUBCOMMANDS:\n");
        let width = node.children.keys().map(|k| k.len()).max().unwrap_or(0);
        for (name, child) in &node.children {
            let h = child.help().unwrap_or("");
            out.push_str(&format!("  {:width$}  {}\n", name, h, width = width));
        }
    }

    out
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
        for child in self.root.children.values() {
            if child.level() > max {
                max = child.level();
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
        for (name, node) in &self.root.children {
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

    /// Walk the tree and promote any per-node staged globals (set via
    /// [`Node::with_promoted_global`]) into the tree-level
    /// `global_value_providers` map. Solves TODO item 2: lets a spec
    /// declare globals as part of node construction without a
    /// post-build patching step.
    ///
    /// Idempotent: drains the staged entries, so re-calling is safe
    /// (and a no-op).
    pub fn lift_promoted_globals(mut self) -> Self {
        self.root.drain_promoted_globals_into(&mut self.global_value_providers);
        self
    }

    /// Built-in option: enable `--help` everywhere. Walks the whole
    /// tree and adds `--help` (boolean) to every node that doesn't
    /// already declare it. Embedders use this to opt into uniform
    /// help support without writing per-node `with_boolean_flags(&[
    /// "--help"])` boilerplate. The same `--help` shows up in tab
    /// completion at every level and is recognised by [`parse_argv`]
    /// as a known flag.
    ///
    /// Pair with [`render_usage`] in your handler:
    ///
    /// ```ignore
    /// let parsed = parse_argv(&tree, &argv)?;
    /// if parsed.flags.contains_key("--help") {
    ///     // walk parsed.path to find the node, then:
    ///     println!("{}", render_usage(node, &parsed.path));
    ///     return Ok(());
    /// }
    /// ```
    pub fn with_auto_help(mut self) -> Self {
        attach_auto_help(&mut self.root);
        self
    }

    /// Built-in option: attach a [`crate::providers::metricsql_provider`]
    /// at the supplied subcommand path. The path is `["sub1",
    /// "sub2", …]` — the chain of children to descend through from
    /// the root before the provider takes over completion.
    ///
    /// Equivalent to manually navigating to the node and calling
    /// `with_subtree_provider(metricsql_provider(catalog))`, but
    /// surfaces the intent at tree-construction time.
    pub fn with_metricsql_at(
        mut self,
        path: &[&str],
        catalog: std::sync::Arc<dyn crate::providers::MetricsqlCatalog>,
    ) -> Self {
        if let Some(node) = walk_path_mut(&mut self.root, path) {
            *node = std::mem::take(node)
                .with_subtree_provider(crate::providers::metricsql_provider(catalog));
        }
        self
    }
}

fn attach_auto_help(node: &mut Node) {
    if !node.flags.iter().any(|f| f == "--help") {
        node.flags.push("--help".to_string());
        node.boolean_flags.insert("--help".to_string());
        if !node.flag_help.contains_key("--help") {
            node.flag_help.insert("--help".to_string(),
                "Show usage information for this command.".to_string());
        }
    }
    for child in node.children.values_mut() {
        attach_auto_help(child);
    }
}

fn walk_path_mut<'a>(root: &'a mut Node, path: &[&str]) -> Option<&'a mut Node> {
    let mut node = root;
    for segment in path {
        node = node.children.get_mut(*segment)?;
    }
    Some(node)
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
    if !node.children.is_empty() {
        let mut candidates: Vec<(u32, String)> = node.children.iter()
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
    complete_rotating_with_raw(tree, words, tap_count, "", 0)
}

/// Same as [`complete_rotating`] but additionally threads the raw
/// `COMP_LINE` and cursor offset through to subtree providers via
/// [`PartialParse::raw_line`] / [`PartialParse::cursor_offset`].
/// Used by [`handle_complete_env`] so grammar-aware providers can
/// inspect raw text + cursor position.
pub fn complete_rotating_with_raw(
    tree: &CommandTree,
    words: &[&str],
    tap_count: u32,
    raw_line: &str,
    cursor_offset: usize,
) -> Vec<String> {
    // Determine the max level among the children of whichever
    // group the cursor is currently inside.
    let completed: &[&str] = if words.len() > 1 { &words[1..words.len() - 1] } else { &[] };
    let partial: &str = if words.len() > 1 { *words.last().unwrap_or(&"") } else { "" };
    let mut node = &tree.root;
    for &word in completed {
        match node.child(word) {
            Some(child) => node = child,
            None => break,
        }
    }
    let max = max_level_of_children(node).max(1);
    let only = ((tap_count.saturating_sub(1)) % max) + 1;
    // Empty partial at a group boundary → apply the level filter
    // via complete_at_level_only (which also handles the no-partial
    // edge case where words = [binary] only). Otherwise dispatch
    // through complete_at_tap_with_raw so subtree providers still
    // see raw context.
    if partial.is_empty() {
        return complete_at_level_only(tree, words, only);
    }
    complete_at_tap_with_raw(tree, words, only, raw_line, cursor_offset)
}

/// Maximum [`Node::level`] across the immediate children of `node`,
/// or [`DEFAULT_LEVEL`] if `node` is a leaf or has no children.
pub(crate) fn max_level_of_children(node: &Node) -> u32 {
    let mut max = DEFAULT_LEVEL;
    for child in node.children.values() {
        if child.level() > max {
            max = child.level();
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
    complete_at_tap_with_raw(tree, words, tap_count, "", 0)
}

/// Same as [`complete_at_tap`] but additionally accepts the raw
/// `COMP_LINE` and the cursor's byte offset. Subtree providers
/// receive these in [`PartialParse::raw_line`] /
/// [`PartialParse::cursor_offset`], enabling grammar-aware
/// completion (e.g. for embedded query DSLs like MetricsQL or
/// PromQL where bracket / quote / operator state at the cursor
/// matters).
///
/// Pass empty `raw_line` and `0` for `cursor_offset` if the caller
/// doesn't have raw context — grammar helpers in [`PartialParse`]
/// fall back to the tokenised view in that case.
pub fn complete_at_tap_with_raw(
    tree: &CommandTree,
    words: &[&str],
    tap_count: u32,
    raw_line: &str,
    cursor_offset: usize,
) -> Vec<String> {
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

    // Walk the tree following completed words. Track the deepest node
    // with a subtree_provider attached — that provider takes
    // precedence over the regular completion path (TODO item 7),
    // letting embedders register context-aware completions inside any
    // subtree without a pre-walker hook.
    let mut node = &tree.root;
    let mut remaining_start = 0;
    let mut tree_path: Vec<&str> = Vec::new();
    let mut deepest_subtree: Option<&SubtreeProvider> = node.subtree_provider();
    for (i, &word) in completed.iter().enumerate() {
        match node.child(word) {
            Some(child) => {
                node = child;
                remaining_start = i + 1;
                tree_path.push(word);
                if let Some(p) = node.subtree_provider() {
                    deepest_subtree = Some(p);
                }
            }
            None => break,
        }
    }
    let remaining = &completed[remaining_start..];

    if let Some(provider) = deepest_subtree {
        let pp = PartialParse {
            completed: completed.to_vec(),
            partial,
            tree_path,
            raw_line,
            cursor_offset,
        };
        return provider(&pp);
    }

    // Check global value providers for the previous word.
    if let Some(&prev_word) = completed.last()
        && let Some(provider) = tree.global_value_providers.get(prev_word) {
        return provider(partial, remaining);
    }

    // Unified node — a node may carry children, flags, or both.
    // Order of candidate sourcing:
    //   1. If the previous word is a value-taking flag, defer
    //      entirely to its value provider.
    //   2. If the partial is `key=…`, defer to the key's provider.
    //   3. Otherwise, collect children (subject to level filter at
    //      root with empty partial) + flags (static + dynamic) +
    //      global flag tokens, prefix-filtered.

    // (1) Value-completion for the previous flag.
    if let Some(&prev_word) = remaining.last()
        && prev_word.starts_with("--")
        && !prev_word.contains('=')
        && !node.boolean_flags.contains(prev_word)
    {
        if let Some(provider) = node.value_providers.get(prev_word) {
            return provider(partial, remaining);
        }
        if let Some(provider) = tree.global_value_providers.get(prev_word) {
            return provider(partial, remaining);
        }
        return Vec::new();
    }

    // (2) `key=value_prefix` form. Bash's default COMP_WORDBREAKS
    // contains `=`, so readline already treats the current word as
    // just the post-`=` segment. We must return BARE values to
    // avoid `key=key=value` stutter.
    if let Some(eq_pos) = partial.find('=') {
        let key = &partial[..eq_pos];
        let value_partial = &partial[eq_pos + 1..];
        let key_eq = format!("{key}=");
        let dashed_key = format!("--{key}");
        if let Some(provider) = node.value_providers.get(&key_eq)
            .or_else(|| node.value_providers.get(&dashed_key))
            .or_else(|| tree.global_value_providers.get(&key_eq))
            .or_else(|| tree.global_value_providers.get(&dashed_key))
        {
            return provider(value_partial, remaining);
        }
        return Vec::new();
    }

    // (3) Children (subcommands).
    let mut child_candidates: Vec<(u32, String)> = node.children.iter()
        .filter(|(k, _)| k.starts_with(partial))
        .filter(|(k, _)| !at_root || !partial.is_empty() || !tree.hidden.contains(k.as_str()))
        .filter(|(_, child)| {
            // Level filter only applies at root with an empty
            // partial — once the user starts typing a name,
            // return matching commands regardless of tap tier.
            !at_root || !partial.is_empty() || child.level() <= tap_count
        })
        .map(|(k, child)| (child.level(), k.to_string()))
        .collect();
    child_candidates.sort_by(|(la, a), (lb, b)| {
        la.cmp(lb)
            .then_with(|| a.starts_with('-').cmp(&b.starts_with('-')))
            .then_with(|| a.cmp(b))
    });

    // (3) Flags on this node — static + dynamic.
    let mut flag_candidates: Vec<String> = Vec::new();
    if !node.flags.is_empty() || node.dynamic_options.is_some() {
        let mut all_flags: Vec<String> = node.flags.clone();
        if let Some(provider) = node.dynamic_options {
            for opt in provider(partial, remaining) {
                if !all_flags.contains(&opt) {
                    all_flags.push(opt);
                }
            }
        }
        let consumed = consumed_keys(remaining, &all_flags);
        for f in &all_flags {
            if f.starts_with(partial) && !is_consumed(f, &consumed) {
                flag_candidates.push(f.clone());
            }
        }
    }

    // (3) Global flag tokens.
    for global_opt in tree.global_value_providers.keys() {
        if global_opt.starts_with(partial) && !flag_candidates.contains(global_opt) {
            flag_candidates.push(global_opt.clone());
        }
    }
    flag_candidates.sort_by(|a, b| {
        a.starts_with('-').cmp(&b.starts_with('-')).then_with(|| a.cmp(b))
    });

    // Children precede flags so a hybrid node lists subcommands
    // first, then flags.
    let mut out: Vec<String> = child_candidates.into_iter().map(|(_, k)| k).collect();
    out.extend(flag_candidates);
    out
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
    let candidates = complete_rotating_with_raw(tree, &words, tap_count, &line, point);

    for candidate in candidates {
        println!("{}", candidate);
    }

    true
}

/// Window inside which a same-key tap counts as a rapid follow-up
/// and advances to the next layer. Outside this window, the next
/// tap starts fresh at layer 1. Exposed so embedders can mention or
/// match the same value in their own UX.
pub const TAP_ADVANCE_MS: u128 = 200;

/// Persistable tap state — the bytes a driver would write between
/// tap events. Two fields: the wall-clock ms at which the tap
/// happened, and the count to *persist* (which is the layer just
/// shown, or 0 if we just closed the cycle by hitting `max_level`).
///
/// Embedders can store a `TapState` in any backing they like (file,
/// memory, an in-process map keyed by shell PID, a test fixture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TapState {
    /// Wall-clock time of the previous tap, in milliseconds since
    /// the UNIX epoch (or any monotonic source the embedder picks —
    /// the rule only inspects differences).
    pub time_ms: u128,
    /// Persisted count from the previous tap. `0` means "cycle was
    /// just closed; next tap starts fresh"; non-zero means "previous
    /// tap showed layer N, advance to N+1 if rapid".
    pub count: u32,
}

/// Pure cadence rule. Given the previous persisted state (and the
/// input key it was recorded against), the current time, the current
/// input key, and the max layer count for the current group, returns:
///
///   - `tap_count` — the layer to show on this invocation (the
///     value the caller passes to [`complete_rotating`]); always in
///     `1..=max_level`.
///   - `next` — the [`TapState`] to persist for the *next* tap.
///
/// Cadence:
///
///   - A same-key tap within [`TAP_ADVANCE_MS`] of the previous tap
///     advances one layer (`prev.count + 1`, capped at `max_level`).
///   - Any other tap (cold, idle past the window, or a key change)
///     starts fresh at layer 1.
///   - Reaching `max_level` resets the persisted count to 0 so the
///     next tap (even rapid) starts fresh at layer 1 — closing the
///     rotation cycle.
///
/// Stateless. No file I/O, no clock reads. This is what tests and
/// embedders should call directly to script arbitrary timing
/// scenarios:
///
/// ```
/// use veks_completion::{TapState, next_tap_state};
///
/// // Cold start — no previous state.
/// let (tap, st) = next_tap_state(None, 1_000, "veks", 2);
/// assert_eq!(tap, 1);
///
/// // Rapid follow-up 100ms later — advances to layer 2.
/// let (tap, st) = next_tap_state(Some((st, "veks")), 1_100, "veks", 2);
/// assert_eq!(tap, 2);
/// // We hit max (2), so persisted count is 0 — cycle closed.
/// assert_eq!(st.count, 0);
///
/// // Third rapid tap — advances from 0+1 = 1 (fresh start).
/// let (tap, _) = next_tap_state(Some((st, "veks")), 1_200, "veks", 2);
/// assert_eq!(tap, 1);
/// ```
pub fn next_tap_state(
    prev: Option<(TapState, &str)>,
    now_ms: u128,
    cur_key: &str,
    max_level: u32,
) -> (u32, TapState) {
    let max = max_level.max(1);
    let mut tap_count = 1u32;
    if let Some((prev_state, prev_key)) = prev {
        if prev_key == cur_key
            && now_ms.saturating_sub(prev_state.time_ms) < TAP_ADVANCE_MS
        {
            tap_count = prev_state.count.saturating_add(1).min(max);
        }
    }
    let to_persist = if tap_count >= max { 0 } else { tap_count };
    let next = TapState {
        time_ms: now_ms,
        count: to_persist,
    };
    (tap_count, next)
}

/// File-backed driver around [`next_tap_state`]. Reads previous
/// state from `/tmp/.<app>_tap_<ppid>`, runs the rule, writes the
/// new state back. Used by [`handle_complete_env`] for the standard
/// shell-completion flow.
///
/// Embedders that want different storage (in-memory map, custom
/// path, sandboxed tempdir for tests) should call [`next_tap_state`]
/// directly and persist the returned [`TapState`] themselves.
fn tap_detect(app_name: &str, input_key: &str, max_level: u32) -> u32 {
    use std::io::Write;

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

    let prev_owned: Option<(TapState, String)> = std::fs::read_to_string(&tap_file)
        .ok()
        .and_then(|content| {
            let mut parts = content.splitn(3, ' ');
            let time_ms: u128 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let count: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let key = parts.next().unwrap_or("").trim_end().to_string();
            Some((TapState { time_ms, count }, key))
        });
    let prev = prev_owned.as_ref().map(|(s, k)| (*s, k.as_str()));

    let (tap_count, next) = next_tap_state(prev, now_ms, cur_key, max_level);

    if let Ok(mut f) = std::fs::File::create(&tap_file) {
        let _ = write!(f, "{} {} {}", next.time_ms, next.count, cur_key);
    }

    tap_count
}

// =====================================================================
// Directive set adapter (TODO item 10)
// =====================================================================

/// A richer flag descriptor that bundles the CLI form, optional
/// YAML mirror, value semantics, and repeatability into one
/// declaration. Adapters can expand a `&[Directive]` into per-flag
/// completion + parse rules without the caller writing a parallel
/// translation layer.
///
/// Maps roughly to nbrs's `vocab::Directive` shape — a directive is
/// "the canonical statement of what a flag IS, in every surface
/// it appears." Use [`apply_directives`] to register a slice of
/// directives onto a [`Node`] in one call.
#[derive(Debug, Clone)]
pub struct Directive {
    /// CLI form, e.g. `"--metric"`. Required.
    pub cli_flag: &'static str,
    /// One-line help text. Optional.
    pub help: Option<&'static str>,
    /// Closed value set for both completion and validation. `None`
    /// means free-form value (or boolean).
    pub values: Option<ClosedValues>,
    /// `true` for boolean flags (no value expected).
    pub boolean: bool,
    /// `true` when this flag may appear multiple times. Currently
    /// informational; reserved for downstream parsers/validators.
    pub repeatable: bool,
    /// Optional YAML directive mirror. Currently informational; lets
    /// downstream YAML parsers cross-reference the same Directive.
    pub yaml_directive: Option<&'static str>,
}

impl Directive {
    /// Construct a value-taking directive with a closed set.
    pub const fn closed(
        cli_flag: &'static str,
        values: &'static [&'static str],
    ) -> Self {
        Directive {
            cli_flag,
            help: None,
            values: Some(ClosedValues::Static(values)),
            boolean: false,
            repeatable: false,
            yaml_directive: None,
        }
    }

    /// Construct a free-form value-taking directive.
    pub const fn value(cli_flag: &'static str) -> Self {
        Directive {
            cli_flag,
            help: None,
            values: None,
            boolean: false,
            repeatable: false,
            yaml_directive: None,
        }
    }

    /// Construct a boolean-flag directive.
    pub const fn boolean(cli_flag: &'static str) -> Self {
        Directive {
            cli_flag,
            help: None,
            values: None,
            boolean: true,
            repeatable: false,
            yaml_directive: None,
        }
    }

    /// Builder: attach help text.
    pub const fn with_help(mut self, help: &'static str) -> Self {
        self.help = Some(help);
        self
    }

    /// Builder: mark as repeatable.
    pub const fn repeatable(mut self) -> Self {
        self.repeatable = true;
        self
    }

    /// Builder: attach the YAML mirror directive name.
    pub const fn with_yaml(mut self, name: &'static str) -> Self {
        self.yaml_directive = Some(name);
        self
    }
}

/// Apply a slice of [`Directive`]s to a [`Node`] in one call. Each
/// directive becomes:
///   - an entry in the node's options/flags list,
///   - a `flag_help` entry if `help` is set,
///   - a value provider if `values` is set (also feeding validation).
///
/// Vocab-driven CLIs become a one-liner: declare the directive list
/// once, hand it to `apply_directives`, get tab + help + (with
/// [`parse_argv`]) parsing all from the same source.
pub fn apply_directives(mut node: Node, directives: &[Directive]) -> Node {
    let value_flags: Vec<&str> = directives.iter()
        .filter(|d| !d.boolean)
        .map(|d| d.cli_flag)
        .collect();
    let bool_flags: Vec<&str> = directives.iter()
        .filter(|d| d.boolean)
        .map(|d| d.cli_flag)
        .collect();

    // Add the flags via the unified builders (idempotent: skip
    // duplicates).
    let value_refs: Vec<&str> = value_flags.iter().copied().collect();
    let bool_refs: Vec<&str> = bool_flags.iter().copied().collect();
    node = node.with_flags(&value_refs).with_boolean_flags(&bool_refs);

    // Help + value providers via the existing builder methods.
    for d in directives {
        if let Some(h) = d.help {
            node = node.with_flag_help(d.cli_flag, h);
        }
        if let Some(values) = &d.values {
            let provider: ValueProvider = values.clone().into_provider();
            node = node.with_value_provider(d.cli_flag, provider);
        }
    }

    node
}

// =====================================================================
// Argv parser companion (TODO item 9)
// =====================================================================

/// Result of [`parse_argv`] — a structured view of an argv vector
/// against a [`CommandTree`]. Embedders use this to dispatch handlers
/// without writing a parallel walker.
///
/// Single source of truth: the same tree that drives tab completion
/// drives argv parsing. Add a flag → both completion and parsing
/// pick it up. Add a subcommand → both reach it.
#[derive(Debug, Clone)]
pub struct ParsedCommand<'a> {
    /// Path through the tree resolved by argv. `["compute", "knn"]`
    /// for `myapp compute knn ...`. Excludes the program name.
    pub path: Vec<&'a str>,
    /// Flags collected along the way. Multiple values per key when
    /// the flag was repeated. Boolean flags map to a single empty
    /// string entry.
    pub flags: std::collections::BTreeMap<String, Vec<String>>,
    /// Positional arguments — anything that wasn't consumed as a
    /// flag, flag value, or subcommand name.
    pub positionals: Vec<&'a str>,
}

/// Errors returned by [`parse_argv`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// `--flag` was given but the tree expects a value to follow,
    /// and argv ended.
    MissingValue {
        flag: String,
    },
    /// A flag appeared that no leaf or ancestor declares.
    UnknownFlag {
        flag: String,
        path: Vec<String>,
    },
    /// A `--flag=value` was given for a closed-set flag whose
    /// validator rejected the value. (Caller-driven; this variant is
    /// reserved for downstream validators that walk the parse
    /// result.)
    InvalidValue {
        flag: String,
        value: String,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingValue { flag } =>
                write!(f, "flag '{}' expects a value but none was given", flag),
            ParseError::UnknownFlag { flag, path } =>
                write!(f, "unknown flag '{}' at '{}'", flag, path.join(" ")),
            ParseError::InvalidValue { flag, value } =>
                write!(f, "invalid value '{}' for flag '{}'", value, flag),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse `argv` (excluding the program name) against the supplied
/// [`CommandTree`]. Walks subcommands, collects flags, separates
/// positionals.
///
/// Strict-ish: rejects flags that aren't declared anywhere along the
/// resolved path. Use [`parse_argv_lenient`] when you want unknown
/// flags treated as positionals.
///
/// ```
/// use veks_completion::{CommandTree, Node, parse_argv};
///
/// let tree = CommandTree::new("myapp")
///     .command("compute", Node::group(vec![
///         ("knn", Node::leaf_with_flags(&["--metric"], &["--verbose"])),
///     ]));
///
/// let parsed = parse_argv(&tree, &[
///     "compute", "knn", "--metric", "L2", "--verbose", "input.fvec",
/// ]).unwrap();
/// assert_eq!(parsed.path, vec!["compute", "knn"]);
/// assert_eq!(parsed.flags["--metric"], vec!["L2".to_string()]);
/// assert_eq!(parsed.flags["--verbose"], vec!["".to_string()]);
/// assert_eq!(parsed.positionals, vec!["input.fvec"]);
/// ```
pub fn parse_argv<'a>(
    tree: &CommandTree,
    argv: &[&'a str],
) -> Result<ParsedCommand<'a>, ParseError> {
    parse_argv_inner(tree, argv, /*lenient=*/ false)
}

/// Lenient variant of [`parse_argv`] — unknown flags are treated as
/// positionals rather than rejected. Useful for "pass-through" CLIs
/// that wrap external commands.
pub fn parse_argv_lenient<'a>(
    tree: &CommandTree,
    argv: &[&'a str],
) -> Result<ParsedCommand<'a>, ParseError> {
    parse_argv_inner(tree, argv, /*lenient=*/ true)
}

fn parse_argv_inner<'a>(
    tree: &CommandTree,
    argv: &[&'a str],
    lenient: bool,
) -> Result<ParsedCommand<'a>, ParseError> {
    let mut path: Vec<&'a str> = Vec::new();
    let mut flags: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    let mut positionals: Vec<&'a str> = Vec::new();

    // Walk the tree, switching nodes when a subcommand name matches
    // a child. Flags collected along the way are credited to the
    // resolved leaf.
    let mut node: &Node = &tree.root;
    let mut i = 0usize;
    while i < argv.len() {
        let arg = argv[i];

        // Subcommand match: only when we're at a Group and the next
        // argv token is a child name AND the cursor isn't already
        // inside a flag-value position.
        if let Some(child) = node.child(arg) {
            path.push(arg);
            node = child;
            i += 1;
            continue;
        }

        // `--key=value` form.
        if let Some(stripped) = arg.strip_prefix("--") {
            if let Some(eq_pos) = stripped.find('=') {
                let key = format!("--{}", &stripped[..eq_pos]);
                let val = stripped[eq_pos + 1..].to_string();
                if flag_is_known(node, tree, &key) {
                    flags.entry(key).or_default().push(val);
                } else if lenient {
                    positionals.push(arg);
                } else {
                    return Err(ParseError::UnknownFlag {
                        flag: key,
                        path: path.iter().map(|s| s.to_string()).collect(),
                    });
                }
                i += 1;
                continue;
            }

            // Bare `--flag` form. Look up to see if it's a boolean
            // or expects a value.
            let key = arg.to_string();
            if !flag_is_known(node, tree, &key) {
                if lenient {
                    positionals.push(arg);
                    i += 1;
                    continue;
                } else {
                    return Err(ParseError::UnknownFlag {
                        flag: key,
                        path: path.iter().map(|s| s.to_string()).collect(),
                    });
                }
            }
            if flag_is_boolean(node, &key) {
                flags.entry(key).or_default().push(String::new());
                i += 1;
            } else {
                if i + 1 >= argv.len() {
                    return Err(ParseError::MissingValue { flag: key });
                }
                let val = argv[i + 1].to_string();
                flags.entry(key).or_default().push(val);
                i += 2;
            }
            continue;
        }

        // Single-char `-x` flags get treated as `--x` for now.
        // Future: explicit short-flag table on the node.
        if arg.starts_with('-') && arg.len() > 1 && !arg.starts_with("--") {
            // Treat as positional fallthrough for the additive
            // version; short-flag handling is deferred (see TODO).
            positionals.push(arg);
            i += 1;
            continue;
        }

        // Plain positional.
        positionals.push(arg);
        i += 1;
    }

    Ok(ParsedCommand { path, flags, positionals })
}

fn flag_is_known(node: &Node, tree: &CommandTree, flag: &str) -> bool {
    // A flag is "known" if the current leaf, the current group's
    // group-level flags, or the tree's global value-provider map
    // recognises it.
    if node.flags.iter().any(|o| flag_canonical_match(o, flag)) {
        return true;
    }
    if tree.global_value_providers.contains_key(flag) {
        return true;
    }
    false
}

fn flag_is_boolean(node: &Node, flag: &str) -> bool {
    node.boolean_flags.contains(flag)
}

fn flag_canonical_match(declared: &str, given: &str) -> bool {
    // Trim a trailing `=` from a declared `--flag=` form before
    // comparing — `--metric=` is the declaration shape used in some
    // pipeline commands.
    let d = declared.trim_end_matches('=');
    d == given
}

/// Expanded completion: show all `group command` pairs.
pub fn complete_expanded(tree: &CommandTree, words: &[&str]) -> Vec<String> {
    let partial = if words.len() > 1 { words.last().unwrap_or(&"") } else { &"" };
    let completed = if words.len() > 2 { &words[1..words.len() - 1] } else { &[] };

    if !completed.is_empty() || !partial.is_empty() {
        return complete(tree, words);
    }

    let mut results = Vec::new();
    for (name, node) in &tree.root.children {
        if name == "help" || name.starts_with('-') {
            continue;
        }
        if !node.children.is_empty() {
            for sub_name in node.children.keys() {
                results.push(format!("{} {}", name, sub_name));
            }
        } else if !tree.hidden.contains(name.as_str()) {
            results.push(name.to_string());
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

    // ---- pure-rule cadence scenarios (no clock, no file I/O) ----

    /// Drive `next_tap_state` through a sequence of taps the same
    /// way an embedder would: own the state in a local variable, call
    /// the pure function with simulated times. Returns the sequence
    /// of `tap_count` values shown across the script.
    fn replay_taps(
        max_level: u32,
        key: &str,
        events: &[u128], // wall-clock times of successive taps
    ) -> Vec<u32> {
        let mut state: Option<TapState> = None;
        let mut shown = Vec::with_capacity(events.len());
        for &t in events {
            let prev = state.map(|s| (s, key));
            let (tap, next) = next_tap_state(prev, t, key, max_level);
            shown.push(tap);
            state = Some(next);
        }
        shown
    }

    #[test]
    fn cadence_cold_tap_is_layer1() {
        let shown = replay_taps(3, "veks", &[1_000]);
        assert_eq!(shown, vec![1]);
    }

    #[test]
    fn cadence_rapid_advances_through_layers() {
        // Three rapid taps with a 3-layer tree should walk 1 → 2 → 3.
        let shown = replay_taps(3, "veks", &[1_000, 1_100, 1_200]);
        assert_eq!(shown, vec![1, 2, 3]);
    }

    #[test]
    fn cadence_rapid_past_max_resets_cycle() {
        // Four rapid taps with a 3-layer tree: 1 → 2 → 3 → 1
        // (the fourth tap reads the reset state and starts fresh).
        let shown = replay_taps(3, "veks", &[1_000, 1_100, 1_200, 1_300]);
        assert_eq!(shown, vec![1, 2, 3, 1]);
    }

    #[test]
    fn cadence_pause_resets_to_layer1() {
        // Two taps separated by 500ms (>200ms window) — the second
        // is treated as a fresh start, not a rapid follow-up.
        let shown = replay_taps(3, "veks", &[1_000, 1_500]);
        assert_eq!(shown, vec![1, 1]);
    }

    #[test]
    fn cadence_key_change_resets_to_layer1() {
        // Two rapid taps but the input key changed — second tap is
        // a fresh start.
        let mut state: Option<TapState> = None;
        let (t1, st1) = next_tap_state(None, 1_000, "veks", 3);
        state = Some(st1);
        assert_eq!(t1, 1);

        let prev = state.map(|s| (s, "veks"));
        let (t2, _) = next_tap_state(prev, 1_100, "veks compute", 3);
        // Different key — fresh start, layer 1.
        assert_eq!(t2, 1);
    }

    #[test]
    fn cadence_max_level_2_alternates() {
        // With max_level = 2, sustained rapid tapping should
        // alternate 1 → 2 → 1 → 2 …
        let shown = replay_taps(2, "veks", &[1_000, 1_100, 1_200, 1_300, 1_400]);
        assert_eq!(shown, vec![1, 2, 1, 2, 1]);
    }

    #[test]
    fn cadence_max_level_1_pinned() {
        // With max_level = 1 (no stratification), every tap shows
        // layer 1 — the rotation is a no-op.
        let shown = replay_taps(1, "veks", &[1_000, 1_100, 1_200, 1_300]);
        assert_eq!(shown, vec![1, 1, 1, 1]);
    }

    #[test]
    fn cadence_advance_window_boundary() {
        // Exactly TAP_ADVANCE_MS apart should NOT count as rapid
        // (strict less-than comparison). Just under should.
        let shown = replay_taps(3, "veks", &[1_000, 1_000 + TAP_ADVANCE_MS]);
        assert_eq!(shown, vec![1, 1], "tap exactly at the boundary is fresh");
        let shown = replay_taps(3, "veks", &[1_000, 1_000 + TAP_ADVANCE_MS - 1]);
        assert_eq!(shown, vec![1, 2], "tap just inside the boundary is rapid");
    }

    // ---- TODO item 1 + 5: ClosedValues -----------------------------

    #[test]
    fn closed_values_static_completes_and_validates() {
        let cv = ClosedValues::Static(&["L2", "IP", "COSINE"]);
        assert_eq!(cv.complete(""), vec!["L2", "IP", "COSINE"]);
        assert_eq!(cv.complete("CO"), vec!["COSINE"]);
        assert_eq!(cv.complete("Z"), Vec::<String>::new());
        assert!(cv.validate("L2"));
        assert!(cv.validate("COSINE"));
        assert!(!cv.validate("bogus"));
    }

    #[test]
    fn closed_values_owned_completes_and_validates() {
        let cv = ClosedValues::Owned(vec!["alpha".into(), "beta".into(), "gamma".into()]);
        assert_eq!(cv.complete("a"), vec!["alpha"]);
        assert!(cv.validate("beta"));
        assert!(!cv.validate("delta"));
    }

    #[test]
    fn closed_values_into_provider_filters() {
        let cv = ClosedValues::Static(&["a", "ab", "abc"]);
        let provider: ValueProvider = cv.into_provider();
        assert_eq!(provider("ab", &[]), vec!["ab", "abc"]);
    }

    // ---- TODO item 4: alias slice ----------------------------------

    #[test]
    fn aliases_share_one_provider() {
        let provider: ValueProvider = std::sync::Arc::new(|partial: &str, _| {
            ["red", "green", "blue"].iter()
                .filter(|s| s.starts_with(partial))
                .map(|s| s.to_string())
                .collect()
        });
        let leaf = Node::leaf(&["--color", "--colour", "--col"])
            .with_value_provider_aliases(&["--color", "--colour", "--col"], provider);
        // Each alias resolves to the same provider — we verify by
        // confirming all three names complete identically.
        let tree = CommandTree::new("paint").command("draw", leaf);
        let out_a = complete(&tree, &["paint", "draw", "--color", "g"]);
        let out_b = complete(&tree, &["paint", "draw", "--colour", "g"]);
        let out_c = complete(&tree, &["paint", "draw", "--col", "g"]);
        assert_eq!(out_a, vec!["green"]);
        assert_eq!(out_a, out_b);
        assert_eq!(out_b, out_c);
    }

    // ---- TODO item 6: help text + render_usage ---------------------

    #[test]
    fn render_usage_includes_help_flags_and_subcommands() {
        let leaf = Node::leaf_with_flags(&["--metric"], &["--verbose"])
            .with_help("Compute KNN over base vectors")
            .with_flag_help("--metric", "Distance metric: L2 / IP / COSINE")
            .with_flag_help("--verbose", "Print per-step progress");
        let tree = CommandTree::new("app")
            .command("compute", Node::group(vec![
                ("knn", leaf),
            ]).with_help("Compute commands"));

        let knn_node = tree.root.child("compute").unwrap().child("knn").unwrap();
        let out = render_usage(knn_node, &["app", "compute", "knn"]);
        assert!(out.contains("USAGE: app compute knn"), "{}", out);
        assert!(out.contains("Compute KNN over base vectors"), "{}", out);
        assert!(out.contains("--metric"), "{}", out);
        assert!(out.contains("Distance metric"), "{}", out);
        assert!(out.contains("--verbose"), "{}", out);

        let compute_node = tree.root.child("compute").unwrap();
        let out = render_usage(compute_node, &["app", "compute"]);
        assert!(out.contains("Compute commands"));
        assert!(out.contains("SUBCOMMANDS:"));
        assert!(out.contains("knn"));
    }

    // ---- TODO item 2: per-node global registration -----------------

    #[test]
    fn promoted_global_lifts_to_tree_globals() {
        let provider = fn_provider(|p: &str, _: &[&str]| {
            ["alpha", "beta"].iter()
                .filter(|s| s.starts_with(p))
                .map(|s| s.to_string())
                .collect()
        });
        let tree = CommandTree::new("app")
            .command("run", Node::leaf(&["--name"])
                .with_promoted_global("--name", provider))
            .lift_promoted_globals();
        // The provider should now live in the tree-level globals
        // map and fire wherever `--name` appears.
        assert!(tree.global_value_providers.contains_key("--name"));
        let out = complete(&tree, &["app", "run", "--name", "a"]);
        assert_eq!(out, vec!["alpha"]);
    }

    // ---- TODO item 3 (additive): group flags -----------------------

    // ---- built-in option: with_auto_help --------------------------

    #[test]
    fn with_auto_help_attaches_help_flag_recursively() {
        let tree = CommandTree::new("app")
            .command("compute", Node::group(vec![
                ("knn", Node::leaf(&["--metric"])),
            ]))
            .command("run", Node::leaf(&["--input"]))
            .with_auto_help();

        let knn = tree.root.child("compute").unwrap().child("knn").unwrap();
        assert!(knn.flags().iter().any(|f| f == "--help"),
            "leaf 'knn' should have --help auto-attached");
        assert!(knn.is_flag("--help"), "--help should be boolean");
        assert!(knn.flag_help_for("--help").is_some());

        let compute = tree.root.child("compute").unwrap();
        assert!(compute.flags().iter().any(|f| f == "--help"),
            "group 'compute' should have --help auto-attached");

        let run = tree.root.child("run").unwrap();
        assert!(run.flags().iter().any(|f| f == "--help"));

        // --help is now a known flag for the parser too.
        let parsed = parse_argv(&tree, &["compute", "knn", "--help"]).unwrap();
        assert_eq!(parsed.path, vec!["compute", "knn"]);
        assert!(parsed.flags.contains_key("--help"));
    }

    #[test]
    fn with_auto_help_doesnt_double_register() {
        let tree = CommandTree::new("app")
            .command("run", Node::leaf_with_flags(&[], &["--help"]))
            .with_auto_help();
        let run = tree.root.child("run").unwrap();
        let count = run.flags().iter().filter(|f| **f == "--help").count();
        assert_eq!(count, 1, "auto-help must be idempotent");
    }

    // ---- built-in option: with_metricsql_at -----------------------

    #[test]
    fn with_metricsql_at_attaches_provider() {
        use std::sync::Arc;
        struct EmptyCatalog;
        impl crate::providers::MetricsqlCatalog for EmptyCatalog {
            fn metric_names(&self, p: &str) -> Vec<String> {
                ["up", "node_cpu"].iter()
                    .filter(|n| n.starts_with(p))
                    .map(|s| s.to_string())
                    .collect()
            }
            fn label_keys(&self, _: &str, p: &str) -> Vec<String> {
                ["job", "instance"].iter()
                    .filter(|n| n.starts_with(p))
                    .map(|s| s.to_string())
                    .collect()
            }
            fn label_values(&self, _: &str, _: &str, p: &str) -> Vec<String> {
                ["prometheus"].iter()
                    .filter(|n| n.starts_with(p))
                    .map(|s| s.to_string())
                    .collect()
            }
        }

        let tree = CommandTree::new("nbrs")
            .command("query", Node::leaf(&[]))
            .with_metricsql_at(&["query"], Arc::new(EmptyCatalog));

        // Verify the subtree provider was attached.
        let query = tree.root.child("query").unwrap();
        assert!(query.subtree_provider().is_some(),
            "with_metricsql_at must attach a subtree provider");
    }

    #[test]
    fn hybrid_node_completes_children_and_flags_together() {
        // A node that has BOTH children AND flags — the central
        // capability the unified Node was built for. `report` here
        // accepts `--workload` (a group-level flag) AND has a
        // `base` subcommand. Tab at root-after-`report` should
        // offer both kinds of candidates.
        let report = Node::group(vec![
            ("base", Node::leaf(&[])),
            ("filtered", Node::leaf(&[])),
        ])
        .with_flags(&["--workload"])
        .with_boolean_flags(&["--dry-run"]);
        let tree = CommandTree::new("app").command("report", report);

        // Empty partial: subcommands first, then flags.
        let cands = complete(&tree, &["app", "report", ""]);
        let pos = |name: &str| cands.iter().position(|s| s == name);
        assert!(pos("base").is_some(), "expected 'base' subcommand: {:?}", cands);
        assert!(pos("filtered").is_some(), "expected 'filtered' subcommand: {:?}", cands);
        assert!(pos("--workload").is_some(), "expected '--workload' flag: {:?}", cands);
        assert!(pos("--dry-run").is_some(), "expected '--dry-run' flag: {:?}", cands);
        // Subcommands precede flags.
        assert!(pos("base").unwrap() < pos("--workload").unwrap());
        assert!(pos("filtered").unwrap() < pos("--dry-run").unwrap());

        // `--workload <TAB>` defers to the parser's general value
        // path (no provider attached → empty) — the important
        // part is that it doesn't fall back to listing children.
        let cands = complete(&tree, &["app", "report", "--workload", ""]);
        assert!(cands.is_empty() || !cands.iter().any(|c| c == "base"),
            "value position must not list children: {:?}", cands);
    }

    #[test]
    fn group_flags_appear_via_options_accessor() {
        let group = Node::group(vec![
            ("sub", Node::leaf(&[])),
        ])
        .with_flags(&["--workload"])
        .with_boolean_flags(&["--dry-run"]);
        assert!(group.options().iter().any(|o| *o == "--workload"));
        assert!(group.options().iter().any(|o| *o == "--dry-run"));
        // Hybrid node — has both children and flags. is_group()
        // returns true; is_leaf() returns false.
        assert!(group.is_group());
        assert!(!group.is_leaf());
    }

    // ---- TODO item 7: subtree provider -----------------------------

    #[test]
    fn subtree_provider_takes_over_completion() {
        // Register a context-aware provider on the `metrics`
        // subtree. The provider sees a structured PartialParse and
        // returns its own candidates.
        let provider: SubtreeProvider = std::sync::Arc::new(|pp: &PartialParse| {
            // Return the partial echoed plus the path joined with `:`.
            vec![format!("{}:{}", pp.tree_path.join("/"), pp.partial)]
        });
        let tree = CommandTree::new("app")
            .command("metrics",
                Node::group(vec![("match", Node::leaf(&[]))])
                    .with_subtree_provider(provider));

        // Cursor inside the metrics subtree.
        let out = complete(&tree, &["app", "metrics", "match", "foo"]);
        assert_eq!(out, vec!["metrics/match:foo".to_string()]);
    }

    // ---- TODO item 8: extras attachment ----------------------------

    #[test]
    fn extras_round_trip_via_downcast() {
        #[derive(Debug, PartialEq, Eq)]
        struct Handler(u32);
        let leaf = Node::leaf(&[])
            .with_extras(Extras::new(Handler(42)));
        let h: &Handler = leaf.extras().unwrap().downcast::<Handler>().unwrap();
        assert_eq!(h.0, 42);
        // Downcast to a different type returns None.
        assert!(leaf.extras().unwrap().downcast::<u8>().is_none());
    }

    // ---- TODO item 9: argv parser ----------------------------------

    #[test]
    fn parse_argv_walks_subcommands_and_collects_flags() {
        let tree = CommandTree::new("app")
            .command("compute", Node::group(vec![
                ("knn", Node::leaf_with_flags(&["--metric"], &["--verbose"])),
            ]));
        let parsed = parse_argv(&tree, &[
            "compute", "knn", "--metric", "L2", "--verbose", "data.fvec",
        ]).unwrap();
        assert_eq!(parsed.path, vec!["compute", "knn"]);
        assert_eq!(parsed.flags["--metric"], vec!["L2".to_string()]);
        assert_eq!(parsed.flags["--verbose"], vec!["".to_string()]);
        assert_eq!(parsed.positionals, vec!["data.fvec"]);
    }

    #[test]
    fn parse_argv_handles_eq_form() {
        let tree = CommandTree::new("app")
            .command("run", Node::leaf(&["--name"]));
        let parsed = parse_argv(&tree, &["run", "--name=foo"]).unwrap();
        assert_eq!(parsed.flags["--name"], vec!["foo".to_string()]);
    }

    #[test]
    fn parse_argv_repeats_collect_into_vec() {
        let tree = CommandTree::new("app")
            .command("run", Node::leaf(&["--set"]));
        let parsed = parse_argv(&tree, &[
            "run", "--set", "a=1", "--set", "b=2",
        ]).unwrap();
        assert_eq!(parsed.flags["--set"], vec!["a=1".to_string(), "b=2".to_string()]);
    }

    #[test]
    fn parse_argv_unknown_flag_strict_errors() {
        let tree = CommandTree::new("app")
            .command("run", Node::leaf(&["--known"]));
        let err = parse_argv(&tree, &["run", "--bogus", "x"]).unwrap_err();
        assert!(matches!(err, ParseError::UnknownFlag { .. }));
    }

    #[test]
    fn parse_argv_unknown_flag_lenient_falls_through() {
        let tree = CommandTree::new("app")
            .command("run", Node::leaf(&["--known"]));
        let parsed = parse_argv_lenient(&tree, &["run", "--bogus", "x"]).unwrap();
        // `--bogus` was treated as positional; `x` followed.
        assert_eq!(parsed.positionals, vec!["--bogus", "x"]);
    }

    #[test]
    fn parse_argv_missing_value_errors() {
        let tree = CommandTree::new("app")
            .command("run", Node::leaf(&["--name"]));
        let err = parse_argv(&tree, &["run", "--name"]).unwrap_err();
        assert!(matches!(err, ParseError::MissingValue { .. }));
    }

    // ---- TODO item 10: directive set adapter -----------------------

    #[test]
    fn apply_directives_registers_flags_help_and_providers() {
        const DIRS: &[Directive] = &[
            Directive::closed("--metric", &["L2", "IP", "COSINE"])
                .with_help("Distance metric"),
            Directive::value("--name").with_help("Name of the run"),
            Directive::boolean("--verbose").with_help("Verbose output"),
        ];
        let leaf = apply_directives(Node::leaf(&[]), DIRS);
        let tree = CommandTree::new("app").command("run", leaf);

        // Tab completion picks up the closed-set values.
        let out = complete(&tree, &["app", "run", "--metric", "C"]);
        assert_eq!(out, vec!["COSINE"]);

        // Help text is reachable via flag_help_for.
        let leaf = tree.root.child("run").unwrap();
        assert_eq!(leaf.flag_help_for("--metric"), Some("Distance metric"));
        assert_eq!(leaf.flag_help_for("--verbose"), Some("Verbose output"));

        // The boolean flag is recognized as such by the parser.
        let parsed = parse_argv(&tree, &[
            "run", "--metric", "L2", "--verbose", "--name", "test",
        ]).unwrap();
        assert_eq!(parsed.flags["--metric"], vec!["L2".to_string()]);
        assert_eq!(parsed.flags["--verbose"], vec!["".to_string()]);
        assert_eq!(parsed.flags["--name"], vec!["test".to_string()]);
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
