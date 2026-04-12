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
pub type ValueProvider = fn(partial: &str, context: &[&str]) -> Vec<String>;

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

/// A node in the command tree.
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
    },
    /// A group containing named child nodes.
    Group { children: BTreeMap<String, Node> },
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Leaf { options, flags, value_providers, dynamic_options } => {
                f.debug_struct("Leaf")
                    .field("options", options)
                    .field("flags", flags)
                    .field("value_providers", &value_providers.keys().collect::<Vec<_>>())
                    .field("has_dynamic_options", &dynamic_options.is_some())
                    .finish()
            }
            Node::Group { children } => {
                f.debug_struct("Group").field("children", children).finish()
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
        }
    }

    /// Create a leaf node with separate value-options and boolean flags.
    pub fn leaf_with_flags(options: &[&str], flags: &[&str]) -> Self {
        Node::Leaf {
            options: options.iter().chain(flags.iter()).map(|s| s.to_string()).collect(),
            flags: flags.iter().map(|s| s.to_string()).collect(),
            value_providers: BTreeMap::new(),
            dynamic_options: None,
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
        }
    }

    /// Create an empty group node.
    pub fn empty_group() -> Self {
        Node::Group { children: BTreeMap::new() }
    }

    /// Add a child to a group node. Panics if called on a leaf.
    pub fn with_child(mut self, name: &str, child: Node) -> Self {
        match &mut self {
            Node::Group { children } => { children.insert(name.to_string(), child); }
            Node::Leaf { .. } => panic!("cannot add child to leaf node"),
        }
        self
    }

    /// Get child names (empty for leaves).
    pub fn child_names(&self) -> Vec<&str> {
        match self {
            Node::Group { children } => children.keys().map(|k| k.as_str()).collect(),
            Node::Leaf { .. } => Vec::new(),
        }
    }

    /// Get a child by name.
    pub fn child(&self, name: &str) -> Option<&Node> {
        match self {
            Node::Group { children } => children.get(name),
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
}

impl CommandTree {
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
        }
    }

    /// Add a top-level command (leaf or group) to the tree.
    ///
    /// This is a builder method — it consumes and returns `self` for chaining.
    pub fn command(mut self, name: &str, node: Node) -> Self {
        self.root = self.root.with_child(name, node);
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
pub fn complete(tree: &CommandTree, words: &[&str]) -> Vec<String> {
    if words.len() <= 1 {
        let mut cmds: Vec<String> = tree.root.child_names().iter()
            .filter(|s| !tree.hidden.contains(**s))
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
        Node::Group { children } => {
            let mut candidates: Vec<String> = children.keys()
                .filter(|k| k.starts_with(partial))
                .filter(|k| !at_root || !partial.is_empty() || !tree.hidden.contains(k.as_str()))
                .map(|k| k.to_string())
                .collect();
            candidates.sort_by(|a, b| {
                a.starts_with('-').cmp(&b.starts_with('-')).then_with(|| a.cmp(b))
            });
            candidates
        }
        Node::Leaf { options, flags, value_providers, dynamic_options } => {
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

/// Generate a bash completion script that calls back into the app.
pub fn print_bash_script(app_name: &str) {
    let env_var = format!("_{}_COMPLETE", app_name.to_uppercase().replace('-', "_"));

    let completer = std::env::args_os()
        .next()
        .and_then(|p| {
            let path = std::path::PathBuf::from(&p);
            if path.components().count() > 1 {
                std::env::current_dir().ok().map(|cwd| cwd.join(path))
            } else {
                Some(path)
            }
        })
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| app_name.to_string());

    print!(r#"_{app}_complete() {{
    COMP_WORDBREAKS="${{COMP_WORDBREAKS//:}}"
    local line="${{COMP_LINE:0:$COMP_POINT}}"
    local -a words=()
    local word=""
    local in_quote=""
    local i=0
    while [ $i -lt ${{#line}} ]; do
        local ch="${{line:$i:1}}"
        if [ -n "$in_quote" ]; then
            if [ "$ch" = "$in_quote" ]; then
                in_quote=""
            else
                word+="$ch"
            fi
        elif [ "$ch" = "'" ] || [ "$ch" = '"' ]; then
            in_quote="$ch"
        elif [ "$ch" = " " ] || [ "$ch" = $'\t' ]; then
            if [ -n "$word" ]; then
                words+=("$word")
                word=""
            fi
        else
            word+="$ch"
        fi
        i=$((i + 1))
    done
    words+=("$word")

    local IFS=$'\n'
    COMPREPLY=($({env_var}=bash _COMP_SHELL_PID=$$ "{completer}" -- "${{words[@]}}" 2>/dev/tty))
}}
if [[ "${{BASH_VERSINFO[0]}}" -eq 4 && "${{BASH_VERSINFO[1]}}" -ge 4 || "${{BASH_VERSINFO[0]}}" -gt 4 ]]; then
    complete -o default -o bashdefault -o nosort -F _{app}_complete {app}
else
    complete -o default -o bashdefault -F _{app}_complete {app}
fi
"#,
        app = app_name,
        env_var = env_var,
        completer = completer,
    );
}

/// Check for completion env vars and handle them.
pub fn handle_complete_env(app_name: &str, tree: &CommandTree) -> bool {
    let env_var = format!("_{}_COMPLETE", app_name.to_uppercase().replace('-', "_"));
    let is_ours = std::env::var(&env_var).ok().as_deref() == Some("bash");
    let is_legacy = std::env::var("COMPLETE").ok().as_deref() == Some("bash");
    if !is_ours && !is_legacy {
        return false;
    }

    let args: Vec<String> = std::env::args().collect();
    let words_start = args.iter().position(|a| a == "--").map(|i| i + 1).unwrap_or(1);
    let words: Vec<&str> = args[words_start..].iter().map(|s| s.as_str()).collect();

    let input_key = words[1..].join(" ");
    let tap_count = tap_detect(app_name, &input_key);
    let expanded = tap_count >= 3 && tap_count % 2 == 1;

    let candidates = if expanded {
        complete_expanded(tree, &words)
    } else {
        complete(tree, &words)
    };

    for candidate in candidates {
        println!("{}", candidate);
    }

    true
}

fn tap_detect(app_name: &str, input_key: &str) -> u32 {
    use std::io::Write;

    let ppid = std::env::var("_COMP_SHELL_PID")
        .or_else(|_| std::env::var("PPID"))
        .unwrap_or_else(|_| "0".to_string());
    let tap_file = std::path::PathBuf::from(format!("/tmp/.{}_tap_{}", app_name, ppid));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut count = 1u32;

    if let Ok(content) = std::fs::read_to_string(&tap_file) {
        let mut parts = content.splitn(3, ' ');
        let prev_time: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let prev_count: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let prev_key = parts.next().unwrap_or("").trim();

        if prev_key == input_key && now.saturating_sub(prev_time) < 5 {
            count = prev_count + 1;
        }
    }

    if let Ok(mut f) = std::fs::File::create(&tap_file) {
        let _ = write!(f, "{} {} {}", now, count, input_key);
    }

    count
}

/// Expanded completion: show all `group command` pairs.
pub fn complete_expanded(tree: &CommandTree, words: &[&str]) -> Vec<String> {
    let partial = if words.len() > 1 { words.last().unwrap_or(&"") } else { &"" };
    let completed = if words.len() > 2 { &words[1..words.len() - 1] } else { &[] };

    if !completed.is_empty() || !partial.is_empty() {
        return complete(tree, words);
    }

    let mut results = Vec::new();
    if let Node::Group { children } = &tree.root {
        for (name, node) in children {
            if name == "help" || name.starts_with('-') {
                continue;
            }
            match node {
                Node::Group { children: sub } if !sub.is_empty() => {
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
}
