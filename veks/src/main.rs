// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)] // legacy completion functions pending removal

use veks::{cli, datasets, explore, pipeline, prepare};

use clap::{Arg, Command, CommandFactory, Parser, Subcommand};


/// Veks — umbrella CLI for vector data tools
#[derive(Parser)]
#[command(
    name = "veks",
    version,
    about,
    disable_help_subcommand = true,
    after_help = "\
Most commands are organized under one of the top-level categories above.\n\
If you know a command, you don't have to use the full `veks <category> ...` format.\n\
For example, `veks run` is shorthand for `veks prepare run`.\n\n\
Hit TAB twice to see the full expanded command list."
)]
struct Veks {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Browse, search, and manage datasets and catalogs
    Datasets(datasets::DatasetsArgs),
    /// Interactive data visualization and exploration
    Interact(explore::ExploreArgs),
    /// Import, stratify, and prepare datasets for benchmarking
    Prepare(prepare::PrepareArgs),
    /// Execute a single pipeline command directly
    #[command(disable_help_flag = true)]
    Pipeline {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Generate shell completions — eval "$(veks completions)"
    #[command(hide = true)]
    Completions(cli::CompletionsArgs),
    /// Show detailed help for a pipeline command or command group
    Help(HelpArgs),
    /// Catch-all for shorthand subcommands (not shown in help/completions)
    #[command(external_subcommand)]
    Shorthand(Vec<String>),
}

/// Arguments for `veks help`.
#[derive(clap::Args)]
struct HelpArgs {
    /// Command path to look up (e.g., "compute knn" or "compute").
    /// Omit to list all commands.
    #[arg(trailing_var_arg = true)]
    command_path: Vec<String>,

    /// List all commands with summaries, grouped by category
    #[arg(long)]
    list: bool,

    /// Emit raw markdown to stdout instead of rendering for the terminal
    #[arg(long)]
    markdown: bool,
}

/// Build the augmented CLI command tree with dynamic pipeline subcommands
/// for shell completion support.
fn build_augmented_cli() -> clap::Command {
    let mut cmd = Veks::command();
    // Replace the derive-generated `pipeline` stub with the full dynamic tree.
    cmd = cmd.mut_subcommand("pipeline", |_| {
        pipeline::cli::build_pipeline_command()
    });
    // Replace the derive-generated `help` stub with a tree that supports
    // tab-completing pipeline group names and command names.
    cmd = cmd.mut_subcommand("help", |_| {
        build_help_completion_command()
    });
    // Add pipeline groups as hidden top-level subcommands so that
    // `veks merkle <TAB>` offers merkle subcommands with full argument
    // completion. Reuse the full pipeline command tree (which includes
    // args, value hints, etc.) rather than building stubs.
    {
        let pipeline_cmd = pipeline::cli::build_pipeline_command();
        for group_sub in pipeline_cmd.get_subcommands() {
            let group_name = group_sub.get_name();
            // Skip groups that collide with derive-based subcommands
            if cmd.get_subcommands().any(|c| c.get_name() == group_name) {
                continue;
            }
            // Clone the full group command (with all child args) and hide it
            // from --help while keeping it visible to the completion engine.
            let hidden_group = group_sub.clone().hide(true);
            cmd = cmd.subcommand(hidden_group);
        }
    }

    // Dynamically hide `datasets list` filter args that can't narrow the
    // current result set any further. This must happen BEFORE shorthand
    // cloning so the hidden args propagate to the root-level aliases.
    let hidden_args = datasets::filter::hidden_list_args();
    cmd = cmd.mut_subcommand("datasets", |datasets_cmd| {
        datasets_cmd.mut_subcommand("list", |mut list_cmd| {
            if hidden_args.contains(&"__disable_help") {
                return clap::Command::new("list")
                    .alias("ls")
                    .disable_help_flag(true);
            }
            for id in &hidden_args {
                list_cmd = list_cmd.mut_arg(*id, |arg| arg.hide(true));
            }
            list_cmd
        })
    });

    // Add hidden root-level aliases for leaf subcommands of groups so that
    // shorthand completions work: e.g., `veks config <TAB>` resolves
    // `config` to `datasets config` and offers its subcommands.
    // This runs AFTER arg hiding so clones inherit the hidden state.
    {
        let shorthand_map = build_shorthand_map();
        for (leaf_name, group_name) in &shorthand_map {
            if cmd.get_subcommands().any(|c| c.get_name() == leaf_name) {
                continue;
            }
            let group_cmd = cmd.get_subcommands().find(|c| c.get_name() == group_name.as_str());
            if let Some(group_cmd) = group_cmd {
                let child = group_cmd.get_subcommands().find(|c| c.get_name() == leaf_name.as_str());
                if let Some(child) = child {
                    let hidden = child.clone().hide(true);
                    cmd = cmd.subcommand(hidden);
                }
            }
        }
    }
    cmd
}

/// Build a `clap::Command` for `veks help` with nested subcommands
/// matching all valid help targets.
///
/// This tree is only used for shell completion — runtime parsing still
/// goes through the derive-generated `HelpArgs`. The structure mirrors
/// what `run_help()` accepts:
///
/// ```text
/// veks help                          # top-level overview
/// veks help --list                   # full catalog
/// veks help run                      # root command help
/// veks help compute                  # pipeline group listing
/// veks help compute knn              # pipeline command docs
/// veks help pipeline compute knn     # same, with explicit prefix
/// ```
fn build_help_completion_command() -> clap::Command {
    use pipeline::registry::CommandRegistry;
    use std::collections::BTreeMap;

    let registry = CommandRegistry::with_builtins();
    let paths = registry.command_paths();

    // Group pipeline command paths by first word.
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for path in &paths {
        let mut parts = path.splitn(2, ' ');
        let group = parts.next().unwrap().to_string();
        let subname = parts.next().unwrap_or("").to_string();
        let factory = registry.get(path).unwrap();
        let summary = factory().command_doc().summary;
        groups.entry(group).or_default().push((subname, summary));
    }

    let mut help_cmd = Command::new("help")
        .about("Show detailed help for a pipeline command or command group")
        .arg(Arg::new("list").long("list").num_args(0)
            .help("List all commands with summaries, grouped by category"))
        .arg(Arg::new("markdown").long("markdown").num_args(0)
            .help("Emit raw markdown to stdout instead of rendering for the terminal"));

    // Add root-level commands as help subcommands.
    // All subcommands in the help tree disable their own --help flag since
    // they exist purely for tab-completion routing, not execution.
    for (name, desc) in root_commands() {
        if name == "help" || name == "pipeline" {
            continue; // handled separately
        }
        help_cmd = help_cmd.subcommand(
            Command::new(name).about(desc).disable_help_flag(true),
        );
    }

    // Add pipeline groups as direct help subcommands (for `veks help compute knn`).
    // If a root command and pipeline group share a name, replace the existing
    // entry with one that includes the pipeline subcommands. Otherwise add a
    // new subcommand for the group.
    for (group, commands) in &groups {
        let group_about = format!("{} pipeline commands", group);
        let exists = help_cmd.get_subcommands().any(|c| c.get_name() == group.as_str());

        let mut group_cmd = if exists {
            help_cmd.get_subcommands()
                .find(|c| c.get_name() == group.as_str())
                .cloned()
                .unwrap()
        } else {
            Command::new(group.clone()).about(group_about).disable_help_flag(true)
        };

        for (subname, summary) in commands {
            if !subname.is_empty() {
                group_cmd = group_cmd.subcommand(
                    Command::new(subname.clone()).about(summary.clone())
                        .disable_help_flag(true),
                );
            }
        }

        if exists {
            help_cmd = help_cmd.mut_subcommand(group.as_str(), |_| group_cmd);
        } else {
            help_cmd = help_cmd.subcommand(group_cmd);
        }
    }

    // Add "pipeline" as a help subcommand with the full group/command tree
    // for `veks help pipeline analyze stats`.
    let mut pipeline_sub = Command::new("pipeline")
        .about("Execute a single pipeline command directly")
        .disable_help_flag(true);
    for (group, commands) in &groups {
        let mut group_cmd = Command::new(group.clone())
            .about(format!("{} pipeline commands", group))
            .disable_help_flag(true);
        for (subname, summary) in commands {
            if !subname.is_empty() {
                group_cmd = group_cmd.subcommand(
                    Command::new(subname.clone()).about(summary.clone())
                        .disable_help_flag(true),
                );
            }
        }
        pipeline_sub = pipeline_sub.subcommand(group_cmd);
    }
    help_cmd = help_cmd.subcommand(pipeline_sub);

    help_cmd
}

fn main() {
    // Dynamic completion engine — handles all shell completion.
    // Handles _VEKS_COMPLETE=bash (from our dyncomp script) and also
    // intercepts COMPLETE=bash (from any stale clap-generated scripts)
    // to prevent clap's eager subcommand chaining.
    if cli::dyncomp::handle_complete_env() {
        std::process::exit(0);
    }

    let veks = Veks::parse();

    match veks.command {
        Commands::Datasets(args) => datasets::run(args),
        Commands::Interact(args) => explore::run(args),
        Commands::Prepare(args) => prepare::run(args),
        Commands::Pipeline { args } => pipeline::cli::run_direct(args),
        Commands::Completions(args) => cli::completions(args),
        Commands::Help(args) => run_help(args),
        Commands::Shorthand(args) => dispatch_shorthand(args),
    }
}

// ---------------------------------------------------------------------------
// Triple-tap detection
// ---------------------------------------------------------------------------

/// Detect if we're completing inside a known subcommand group.
///
// Legacy completion functions — retained for reference but replaced by dyncomp.
#[allow(dead_code)]
fn subcommand_completion_context() -> Option<(String, String)> {
    let args: Vec<String> = std::env::args().collect();
    let words = if let Some(pos) = args.iter().position(|a| a == "--") {
        args[pos + 1..].to_vec()
    } else {
        args[1..].to_vec()
    };
    // Need exactly: program-context group [partial-child]
    // words[0] = group name, words[1] = partial child (may be empty)
    if words.len() < 2 || words.len() > 3 {
        return None;
    }
    let group = &words[0];
    let child_prefix = if words.len() == 3 {
        words[2].clone()
    } else {
        words.get(1).cloned().unwrap_or_default()
    };
    // Verify this is a known group with children
    let cmd = Veks::command();
    let sub = cmd.get_subcommands().find(|c| c.get_name() == group.as_str())?;
    if sub.get_subcommands().next().is_some() {
        Some((group.clone(), child_prefix))
    } else {
        None
    }
}

/// Filter completion output to show only children of a specific group,
/// with the group prefix stripped so they complete naturally.
fn filter_subcommand_children(group: &str, prefix: &str, _full_list: &str) -> String {
    let mut result = String::new();

    if group == "pipeline" {
        // Pipeline commands: build from registry (they use "group child" paths)
        let registry = pipeline::registry::CommandRegistry::with_builtins();
        for path in registry.command_paths() {
            if prefix.is_empty() || path.starts_with(prefix) {
                result.push_str(&format!("{}\t{}\n", path, path));
            }
        }
    } else {
        // Derive-based groups: walk the clap tree
        let cmd = Veks::command();
        if let Some(group_cmd) = cmd.get_subcommands().find(|c| c.get_name() == group) {
            for child in group_cmd.get_subcommands() {
                let name = child.get_name();
                if prefix.is_empty() || name.starts_with(prefix) {
                    let about = child.get_about().map(|s| s.to_string()).unwrap_or_default();
                    result.push_str(&format!("{}\t{}\n", name, about));
                }
            }
        }
    }
    result
}

/// If this is a root-level completion, return the partial prefix the user
/// has typed. Returns `Some("")` for `veks <TAB>`, `Some("se")` for
/// `veks se<TAB>`, and `None` if we're inside a subcommand context.
fn root_completion_prefix() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let after_separator = if let Some(pos) = args.iter().position(|a| a == "--") {
        args[pos + 1..].to_vec()
    } else {
        if args.len() <= 2 {
            return Some(args.get(1).cloned().unwrap_or_default());
        }
        return None;
    };
    // Root level: program name + optional partial word
    if after_separator.len() <= 2 {
        Some(after_separator.get(1).cloned().unwrap_or_default())
    } else {
        None
    }
}

/// Filter the full command list to entries matching a prefix.
/// Returns fish-format completion output, or empty string if no matches.
fn filter_commands_by_prefix(prefix: &str) -> String {
    let full = build_full_command_list();
    let mut matches = String::new();
    for line in full.lines() {
        // Each line is "group child\tdescription" or "leaf\tdescription"
        let cmd = line.split('\t').next().unwrap_or("");
        // Match against the full command string or the first word (group name).
        // Do NOT match interior/trailing words — "datas" should not match
        // "generate dataset". Shorthand completion (e.g., "config" → "datasets
        // config") is handled by hidden root subcommands in build_augmented_cli.
        let first_word = cmd.split(' ').next().unwrap_or("");
        if cmd.starts_with(prefix) || first_word.starts_with(prefix) {
            matches.push_str(line);
            matches.push('\n');
        }
    }
    matches
}

const TAP_FILE: &str = "/tmp/.veks_tab";
/// Number of identical completion invocations before showing the expanded
/// command list. Shells typically invoke completion twice per Tab press
/// (once to compute, once to display), so 3 means: first Tab shows normal
/// completions, second Tab triggers the expanded list.
const TAP_COUNT: usize = 3;

/// Maximum age (in seconds) for a tap file entry to be considered valid.
/// Prevents stale state from a previous session triggering the expanded list.
const TAP_MAX_AGE_SECS: u64 = 10;

/// Check if the user has tapped Tab repeatedly in the same completion state.
///
/// Tracks the completion args (cursor position + partial input) in a file.
/// If the same state is seen `TAP_COUNT` times in a row within
/// `TAP_MAX_AGE_SECS`, the user is repeatedly hitting Tab without
/// changing their input — trigger the expanded command list.
///
/// If /tmp is not writable or accessible, silently returns `None`.
fn check_triple_tap() -> Option<String> {
    use std::time::SystemTime;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?
        .as_secs();

    // Build a fingerprint of the current completion state from env/args.
    let state_key: String = std::env::args().skip(1).collect::<Vec<_>>().join("\x00");

    // Read previous state (best-effort)
    // Format: count\ntimestamp\nstate_key
    let (prev_key, prev_count, prev_time) = std::fs::read_to_string(TAP_FILE)
        .ok()
        .and_then(|content| {
            let mut lines = content.lines();
            let count: usize = lines.next()?.parse().ok()?;
            let timestamp: u64 = lines.next()?.parse().ok()?;
            let key = lines.next().unwrap_or("").to_string();
            Some((key, count, timestamp))
        })
        .unwrap_or_default();

    let is_stale = now.saturating_sub(prev_time) > TAP_MAX_AGE_SECS;

    let count = if state_key == prev_key && !is_stale {
        prev_count + 1
    } else {
        1
    };

    // Write back (best-effort, silently ignore errors)
    let _ = std::fs::write(TAP_FILE, format!("{}\n{}\n{}", count, now, state_key));

    if count >= TAP_COUNT {
        // Reset so next tap starts fresh
        let _ = std::fs::write(TAP_FILE, "");
        Some(build_full_command_list())
    } else {
        None
    }
}

/// Build a fish-format completion list of all available commands.
/// Build the full command list dynamically from the clap tree + pipeline registry.
///
/// Used by the double-tap completion and prefix-filtered completion.
/// Walks the derive-generated command tree so it never goes stale.
fn build_full_command_list() -> String {
    let cmd = Veks::command();
    let mut lines: Vec<String> = Vec::new();

    // Groups whose children should NOT appear as shorthands in the flat list
    let internal_groups = ["pipeline", "help", "completions"];

    for sub in cmd.get_subcommands() {
        let name = sub.get_name();
        let about = sub.get_about().map(|s| s.to_string()).unwrap_or_default();

        let has_children = sub.get_subcommands().next().is_some();

        if has_children && !internal_groups.contains(&name) {
            // Only show children with "group child" prefix — not the group
            // itself. Including the bare group name prevents the shell from
            // advancing past it (it's a prefix of every child).
            for child in sub.get_subcommands() {
                let child_name = child.get_name();
                let child_about = child.get_about()
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                lines.push(format!("{} {}\t{}", name, child_name, child_about));
            }
        } else if !internal_groups.contains(&name) {
            // Leaf command at root level (no subcommands)
            lines.push(format!("{}\t{}", name, about));
        }
    }

    // Pipeline commands from the registry (shorthand form only)
    let registry = pipeline::registry::CommandRegistry::with_builtins();
    for path in registry.command_paths() {
        lines.push(format!("{}\tpipeline: {}", path, path));
    }

    lines.join("\n") + "\n"
}

// ---------------------------------------------------------------------------
// Shorthand subcommand dispatch
// ---------------------------------------------------------------------------

/// Build the shorthand map dynamically by walking the clap command tree.
///
/// Enumerates all subcommands of groups that have nested subcommands
/// (like `prepare`, `datasets`, `visualize`) and maps each leaf
/// subcommand name → group name. This eliminates the static table that
/// was prone to going stale.
///
/// All subcommand names MUST be globally unique across all groups
/// (SRD §1.4.1). If a collision is detected at build time, this will
/// produce ambiguous results — but the design invariant prevents that.
fn build_shorthand_map() -> Vec<(String, String)> {
    let cmd = Veks::command();
    let mut map = Vec::new();

    // Groups that should NOT have their subcommands exposed as shorthands
    let skip_groups = ["pipeline", "help", "completions"];

    for sub in cmd.get_subcommands() {
        let group = sub.get_name().to_string();
        if skip_groups.contains(&group.as_str()) {
            continue;
        }

        let children: Vec<_> = sub.get_subcommands().collect();
        if children.is_empty() {
            // No subcommands — this is a leaf root command, not a group
            continue;
        }

        for child in children {
            let name = child.get_name().to_string();
            map.push((name.clone(), group.clone()));

            // Also register aliases
            for alias in child.get_all_aliases() {
                map.push((alias.to_string(), group.clone()));
            }
        }
    }

    map
}

/// Dispatch an unrecognized root-level subcommand.
///
/// Walks the clap command tree to find which group owns this subcommand
/// name, then re-parses with the group prefix inserted. Falls through
/// to the pipeline command registry if no match is found.
fn dispatch_shorthand(args: Vec<String>) {
    if args.is_empty() {
        eprintln!("Error: no command specified. Run `veks --help` for usage.");
        std::process::exit(1);
    }

    let name = &args[0];
    let shorthand_map = build_shorthand_map();

    // Look up in the dynamic shorthand map
    if let Some((_, group)) = shorthand_map.iter().find(|(cmd, _)| cmd == name) {
        let mut full_args = vec!["veks".to_string(), group.clone()];
        full_args.extend(args);
        let veks = Veks::parse_from(full_args);
        match veks.command {
            Commands::Datasets(a) => datasets::run(a),
            Commands::Interact(a) => explore::run(a),
            Commands::Prepare(a) => prepare::run(a),
            _ => unreachable!(),
        }
        return;
    }

    // Not a known shorthand — try as a pipeline command
    pipeline::cli::run_direct(args);
}

/// Root-level command descriptions for `veks help`.
fn root_commands() -> Vec<(&'static str, &'static str)> {
    vec![
        ("datasets", "Browse, search, and manage datasets and catalogs"),
        ("run", "Execute a command stream pipeline defined in dataset.yaml"),
        ("script", "Emit a dataset pipeline as an equivalent shell script"),
        ("pipeline", "Execute a single pipeline command directly"),
        ("check", "Pre-flight checks for dataset readiness"),
        ("publish", "Publish dataset to S3"),
        ("completions", "Generate shell completions"),
        ("help", "Show detailed help for a pipeline command or command group"),
    ]
}

/// Execute `veks help` — display documentation for commands and groups.
///
/// With no arguments, shows the same concise overview as `veks --help`.
/// With `--list`, shows all commands including the full pipeline catalog.
/// With a command path, shows detailed help for that command or group.
///
/// The `pipeline` prefix is optional: `veks help analyze stats` and
/// `veks help pipeline analyze stats` are equivalent.
fn run_help(args: HelpArgs) {
    use pipeline::registry::CommandRegistry;
    use std::collections::BTreeMap;

    let registry = CommandRegistry::with_builtins();
    let paths = registry.command_paths();

    // Group pipeline command paths by first word.
    let mut groups: BTreeMap<String, Vec<(&str, String)>> = BTreeMap::new();
    for path in &paths {
        let mut parts = path.splitn(2, ' ');
        let group = parts.next().unwrap().to_string();
        let factory = registry.get(path).unwrap();
        let cmd = factory();
        let summary = cmd.command_doc().summary;
        groups.entry(group).or_default().push((path, summary));
    }

    // --list mode: show all commands grouped (full pipeline catalog)
    if args.list {
        print_command_list(&groups, args.markdown);
        return;
    }

    // No arguments: show the same concise output as `veks --help`.
    if args.command_path.is_empty() {
        let mut cmd = build_augmented_cli();
        let help = cmd.render_help();
        println!("{}", help);
        return;
    }

    // Strip optional "pipeline" prefix so `veks help pipeline analyze stats`
    // works the same as `veks help analyze stats`.
    let had_pipeline_prefix = args.command_path.first().map(|s| s.as_str()) == Some("pipeline");
    let command_path = if had_pipeline_prefix {
        &args.command_path[1..]
    } else {
        &args.command_path[..]
    };

    // If stripping "pipeline" left nothing, show the pipeline subcommand help.
    if command_path.is_empty() {
        if let Some(help_text) = root_command_help("pipeline") {
            if args.markdown {
                println!("{}", help_text);
            } else {
                render_markdown_to_terminal(&help_text);
            }
        }
        return;
    }

    let query = command_path.join(" ");

    // When the user did NOT say "pipeline", check root-level commands first
    // (e.g., `veks help run` shows the root `run` command).
    // When they DID say "pipeline", skip root commands so
    // `veks help pipeline analyze` shows the pipeline group, not root analyze.
    if !had_pipeline_prefix {
        if let Some(help_text) = root_command_help(&query) {
            if args.markdown {
                println!("{}", help_text);
            } else {
                render_markdown_to_terminal(&help_text);
            }
            return;
        }
    }

    // Check if query matches a pipeline group name (DOC-07)
    if groups.contains_key(&query) {
        print_group_help(&query, &groups, &registry, args.markdown);
        return;
    }

    // Check if query matches a specific pipeline command
    if let Some(factory) = registry.get(&query) {
        let cmd = factory();
        let doc = cmd.command_doc();
        if args.markdown {
            println!("{}", doc.body);
        } else {
            render_markdown_to_terminal(&doc.body);
        }
        return;
    }

    // Fall through: when user didn't say "pipeline" and it's not a pipeline
    // group/command either, try root commands as a last resort.
    if had_pipeline_prefix {
        if let Some(help_text) = root_command_help(&query) {
            if args.markdown {
                println!("{}", help_text);
            } else {
                render_markdown_to_terminal(&help_text);
            }
            return;
        }
    }

    eprintln!("Unknown command or group: '{}'", query);
    eprintln!("Use 'veks help --list' to see all available commands.");
    std::process::exit(1);
}

/// Generate help text for a root-level command by rendering its clap help.
fn root_command_help(name: &str) -> Option<String> {
    let cmd = build_augmented_cli();
    let sub = cmd.get_subcommands().find(|c| c.get_name() == name)?;
    let mut sub = sub.clone();
    let help = sub.render_help();
    Some(help.to_string())
}

/// Print all commands grouped by category (DOC-05 --list mode).
fn print_command_list(
    pipeline_groups: &std::collections::BTreeMap<String, Vec<(&str, String)>>,
    markdown: bool,
) {
    use std::collections::BTreeMap;

    // Build a unified group map: derive-based groups + pipeline groups.
    // Groups that are parent-only (have children) are shown as groups,
    // not as standalone root commands.
    let mut all_groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();

    // Collect derive-based groups (datasets, visualize, prepare) and their children.
    let cmd = Veks::command();
    let internal = ["pipeline", "help", "completions"];
    let mut leaf_commands: Vec<(String, String)> = Vec::new();

    for sub in cmd.get_subcommands() {
        let name = sub.get_name().to_string();
        let about = sub.get_about().map(|s| s.to_string()).unwrap_or_default();

        if internal.contains(&name.as_str()) {
            continue;
        }

        let has_children = sub.get_subcommands().next().is_some();
        if has_children {
            let children: Vec<(String, String)> = sub.get_subcommands()
                .map(|child| {
                    let child_name = child.get_name().to_string();
                    let child_about = child.get_about().map(|s| s.to_string()).unwrap_or_default();
                    (child_name, child_about)
                })
                .collect();
            all_groups.insert(name, children);
        } else {
            leaf_commands.push((name, about));
        }
    }

    // Add pipeline groups
    for (group, commands) in pipeline_groups {
        let entries: Vec<(String, String)> = commands.iter()
            .map(|(path, summary)| {
                let subname = path.splitn(2, ' ').nth(1).unwrap_or(path).to_string();
                (subname, summary.clone())
            })
            .collect();
        all_groups.entry(group.clone())
            .or_default()
            .extend(entries);
    }

    if markdown {
        println!("# veks commands\n");
        if !leaf_commands.is_empty() {
            println!("## standalone\n");
            println!("| Command | Summary |");
            println!("|---------|---------|");
            for (name, summary) in &leaf_commands {
                println!("| `{}` | {} |", name, summary);
            }
            println!();
        }
        for (group, commands) in &all_groups {
            println!("## {}\n", group);
            println!("| Command | Summary |");
            println!("|---------|---------|");
            for (name, summary) in commands {
                println!("| `{} {}` | {} |", group, name, summary);
            }
            println!();
        }
    } else {
        if !leaf_commands.is_empty() {
            let max_len = leaf_commands.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
            for (name, summary) in &leaf_commands {
                println!("  {:<width$}  {}", name, summary, width = max_len);
            }
            println!();
        }
        for (group, commands) in &all_groups {
            println!("  {}:", group);
            let max_len = commands.iter()
                .map(|(n, _)| n.len())
                .max()
                .unwrap_or(0);
            for (name, summary) in commands {
                println!("    {:<width$}  {}", name, summary, width = max_len);
            }
            println!();
        }
    }
}

/// Print group-level documentation (DOC-07).
fn print_group_help(
    group: &str,
    groups: &std::collections::BTreeMap<String, Vec<(&str, String)>>,
    _registry: &pipeline::registry::CommandRegistry,
    markdown: bool,
) {
    let commands = &groups[group];

    if markdown {
        println!("# {} — pipeline command group\n", group);
        println!("| Command | Summary |");
        println!("|---------|---------|");
        for (path, summary) in commands {
            let subname = path.splitn(2, ' ').nth(1).unwrap_or(path);
            println!("| `{}` | {} |", subname, summary);
        }
        println!("\nUse `veks help {} <command>` for detailed documentation.", group);
    } else {
        println!("{} — pipeline command group\n", group);
        let max_len = commands.iter()
            .map(|(p, _)| p.splitn(2, ' ').nth(1).unwrap_or(p).len())
            .max()
            .unwrap_or(0);
        for (path, summary) in commands {
            let subname = path.splitn(2, ' ').nth(1).unwrap_or(path);
            println!("  {:<width$}  {}", subname, summary, width = max_len);
        }
        println!("\nUse 'veks help {} <command>' for detailed documentation.", group);
    }
}

/// Render markdown to the terminal with lightweight formatting.
///
/// Applies basic transformations: headings get bold/underline treatment,
/// code blocks are indented, and tables are passed through as-is.
/// No external dependency required.
fn render_markdown_to_terminal(body: &str) {
    let bold = "\x1b[1m";
    let dim = "\x1b[2m";
    let reset = "\x1b[0m";
    let underline = "\x1b[4m";

    let use_ansi = veks::term::use_color();
    let mut in_code_block = false;

    for line in body.lines() {
        if line.starts_with("```") {
            in_code_block = !in_code_block;
            if in_code_block {
                if use_ansi {
                    println!("{}", dim);
                }
            } else if use_ansi {
                print!("{}", reset);
            }
            continue;
        }

        if in_code_block {
            println!("    {}", line);
            continue;
        }

        if let Some(heading) = line.strip_prefix("# ") {
            if use_ansi {
                println!("{}{}{}{}", bold, underline, heading, reset);
            } else {
                println!("{}", heading);
                println!("{}", "=".repeat(heading.len()));
            }
        } else if let Some(heading) = line.strip_prefix("## ") {
            if use_ansi {
                println!("\n{}{}{}", bold, heading, reset);
            } else {
                println!("\n{}", heading);
                println!("{}", "-".repeat(heading.len()));
            }
        } else if let Some(heading) = line.strip_prefix("### ") {
            if use_ansi {
                println!("\n{}{}{}", bold, heading, reset);
            } else {
                println!("\n{}", heading);
            }
        } else {
            println!("{}", line);
        }
    }
}
