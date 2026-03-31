// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Dynamic shell completion engine for CLI tools.
//!
//! Provides a generic, tree-based completion system that completes one level
//! at a time (no eager subcommand chaining). The caller defines the command
//! tree via [`CommandTree`], and this crate handles:
//!
//! - Walking the tree to find candidates for a given input
//! - Generating bash completion scripts
//! - Handling the `_<APP>_COMPLETE=bash` and `COMPLETE=bash` env var callbacks
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

/// A function that provides dynamic completion values for an option.
pub type ValueProvider = fn(partial: &str) -> Vec<String>;

/// A node in the command tree.
#[derive(Clone)]
pub enum Node {
    /// A leaf command with option names and optional value providers.
    Leaf {
        options: Vec<String>,
        /// Dynamic value providers keyed by option name (e.g., "--dataset").
        value_providers: BTreeMap<String, ValueProvider>,
    },
    /// A group containing named child nodes.
    Group { children: BTreeMap<String, Node> },
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Leaf { options, value_providers } => {
                f.debug_struct("Leaf")
                    .field("options", options)
                    .field("value_providers", &value_providers.keys().collect::<Vec<_>>())
                    .finish()
            }
            Node::Group { children } => {
                f.debug_struct("Group").field("children", children).finish()
            }
        }
    }
}

impl Node {
    /// Create a leaf node with the given option names.
    pub fn leaf(options: &[&str]) -> Self {
        Node::Leaf {
            options: options.iter().map(|s| s.to_string()).collect(),
            value_providers: BTreeMap::new(),
        }
    }

    /// Attach a dynamic value provider to an option on this leaf node.
    /// The provider is called when the user tabs after `--option `.
    pub fn with_value_provider(mut self, option: &str, provider: ValueProvider) -> Self {
        if let Node::Leaf { ref mut value_providers, .. } = self {
            value_providers.insert(option.to_string(), provider);
        }
        self
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
    /// Global value providers keyed by option name. These fire for ANY
    /// command when the previous word matches the option. Used for
    /// ubiquitous options like `--dataset` that appear across many commands.
    pub global_value_providers: BTreeMap<String, ValueProvider>,
}

impl CommandTree {
    /// Create a new command tree with the given app name.
    pub fn new(app_name: &str) -> Self {
        CommandTree {
            app_name: app_name.to_string(),
            root: Node::empty_group(),
            hidden: std::collections::HashSet::new(),
            global_value_providers: BTreeMap::new(),
        }
    }

    /// Add a top-level command.
    pub fn command(mut self, name: &str, node: Node) -> Self {
        self.root = self.root.with_child(name, node);
        self
    }

    /// Add a top-level group with children.
    pub fn group(self, name: &str, node: Node) -> Self {
        self.command(name, node)
    }

    /// Add a hidden top-level command (shortcut).
    ///
    /// Hidden commands don't appear in root-level listings but still
    /// complete when the user types a matching prefix. This is used for
    /// pipeline group shortcuts like `compute`, `analyze`, etc.
    pub fn hidden_command(mut self, name: &str, node: Node) -> Self {
        self.hidden.insert(name.to_string());
        self.command(name, node)
    }

    /// Register a global value provider for an option name.
    ///
    /// Fires for ANY command when the user tabs after this option.
    /// Use for ubiquitous options like `--dataset` that appear across
    /// many commands.
    pub fn global_value_provider(mut self, option: &str, provider: ValueProvider) -> Self {
        self.global_value_providers.insert(option.to_string(), provider);
        self
    }
}

/// Compute completion candidates for the given input words.
///
/// `words[0]` is the program name, `words[1..]` are subcommands and the
/// partial word being completed (last element, may be empty).
///
/// Returns one candidate per line, suitable for bash `COMPREPLY`.
pub fn complete(tree: &CommandTree, words: &[&str]) -> Vec<String> {
    if words.len() <= 1 {
        // Show only non-hidden root commands, flags last
        let mut cmds: Vec<String> = tree.root.child_names().iter()
            .filter(|s| !tree.hidden.contains(&s.to_string()))
            .map(|s| s.to_string())
            .collect();
        cmds.sort_by(|a, b| {
            let a_flag = a.starts_with('-');
            let b_flag = b.starts_with('-');
            a_flag.cmp(&b_flag).then_with(|| a.cmp(b))
        });
        return cmds;
    }

    let partial = words.last().unwrap_or(&"");
    let completed = &words[1..words.len() - 1];

    // At root level with a partial prefix: include hidden commands
    // (shortcuts) so they complete when the user starts typing them
    let at_root = completed.is_empty();

    // Walk the tree following completed words, stopping at leaves or
    // unknown words (options like --dataset aren't tree children).
    let mut node = &tree.root;
    let mut remaining_start = 0;
    for (i, &word) in completed.iter().enumerate() {
        match node.child(word) {
            Some(child) => { node = child; remaining_start = i + 1; }
            None => break, // hit an option or unknown word
        }
    }
    let remaining = &completed[remaining_start..];

    // Check global value providers: if the previous word (before the
    // partial) is an option with a global provider, use it.
    if let Some(&prev_word) = completed.last() {
        if let Some(provider) = tree.global_value_providers.get(prev_word) {
            return provider(partial);
        }
    }

    // At this node, offer completions based on type.
    match node {
        Node::Group { children } => {
            let mut candidates: Vec<String> = children.keys()
                .filter(|k| k.starts_with(partial))
                .filter(|k| !at_root || !partial.is_empty() || !tree.hidden.contains(k.as_str()))
                .map(|k| k.to_string())
                .collect();
            // Sort flags (--help, --version) after subcommands
            candidates.sort_by(|a, b| {
                a.starts_with('-').cmp(&b.starts_with('-')).then_with(|| a.cmp(b))
            });
            candidates
        }
        Node::Leaf { options, value_providers } => {
            // Check if the previous word is an option with a per-command
            // or global value provider (e.g., user typed `--dataset <TAB>`)
            if let Some(&prev_word) = remaining.last() {
                if let Some(provider) = value_providers.get(prev_word) {
                    return provider(partial);
                }
            }
            if partial.starts_with('-') || partial.is_empty() {
                // Include both the command's own options AND global provider
                // option names (like --dataset) so they complete everywhere.
                let mut candidates: Vec<String> = options.iter()
                    .filter(|o| o.starts_with(partial))
                    .map(|o| o.to_string())
                    .collect();
                for global_opt in tree.global_value_providers.keys() {
                    if global_opt.starts_with(partial) && !candidates.contains(global_opt) {
                        candidates.push(global_opt.clone());
                    }
                }
                candidates.sort();
                candidates
            } else {
                Vec::new()
            }
        }
    }
}

/// Generate a bash completion script that calls back into the app.
///
/// The script registers a completion function that invokes the app with
/// `_<APP>_COMPLETE=bash` (and also handles the legacy `COMPLETE=bash`).
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
    COMPREPLY=($({env_var}=bash _COMP_SHELL_PID=$$ "{completer}" -- "${{words[@]}}" 2>/dev/null))
}}
if [[ "${{BASH_VERSINFO[0]}}" -eq 4 && "${{BASH_VERSINFO[1]}}" -ge 4 || "${{BASH_VERSINFO[0]}}" -gt 4 ]]; then
    complete -o bashdefault -o nosort -F _{app}_complete {app}
else
    complete -o bashdefault -F _{app}_complete {app}
fi
"#,
        app = app_name,
        env_var = env_var,
        completer = completer,
    );
}

/// Check for completion env vars and handle them.
///
/// Checks both `_<APP>_COMPLETE=bash` (our format) and `COMPLETE=bash`
/// (legacy clap format) so stale shell sessions work.
///
/// Returns `true` if a completion request was handled (caller should exit).
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

    // Detect repeated taps at the same prompt via a temp file.
    // On the 3rd consecutive tap with the same input, show expanded listing.
    let input_key = words[1..].join(" ");
    let tap_count = tap_detect(app_name, &input_key);
    // Alternate: 1-2 = short, 3 = expanded, 4 = short, 5 = expanded, ...
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

/// Track consecutive tab presses via a temp file. Returns the current
/// tap count (1 on first press, incrementing on rapid repeats).
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

/// Expanded completion: show all `group command` pairs for the full
/// command tree. Used on the third consecutive tab press at root level.
pub fn complete_expanded(tree: &CommandTree, words: &[&str]) -> Vec<String> {
    let partial = if words.len() > 1 { words.last().unwrap_or(&"") } else { &"" };
    let completed = if words.len() > 2 { &words[1..words.len() - 1] } else { &[] };

    // Only expand at root level with empty prefix
    if !completed.is_empty() || !partial.is_empty() {
        return complete(tree, words);
    }

    // Generate "group command" pairs for all children.
    // Skip "help" sub-entries (help is a navigation aid, not a command group)
    // and hidden leaf commands (shortcuts shown via their parent group).
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
