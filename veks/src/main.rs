// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use veks::{check, cli, datasets, pipeline, prepare, publish};

use clap::{Arg, Command, CommandFactory, Parser, Subcommand};
use clap_complete::CompleteEnv;

/// Veks — umbrella CLI for vector data tools
#[derive(Parser)]
#[command(name = "veks", version, about, disable_help_subcommand = true)]
struct Veks {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Browse, search, and manage datasets and catalogs
    Datasets(datasets::DatasetsArgs),
    /// Import, stratify, and prepare datasets for benchmarking
    Prepare(prepare::PrepareArgs),
    /// Execute a command stream pipeline defined in dataset.yaml
    Run(pipeline::RunArgs),
    /// Emit a dataset pipeline as an equivalent shell script
    Script(pipeline::ScriptArgs),
    /// Execute a single pipeline command directly
    #[command(disable_help_flag = true)]
    Pipeline {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Pre-flight checks for dataset readiness
    Check(check::CheckArgs),
    /// Publish dataset to S3
    Publish(publish::PublishArgs),
    /// Generate shell completions
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
    // Dynamically hide `datasets list` filter args that can't narrow the
    // current result set any further.
    cmd = cmd.mut_subcommand("datasets", |datasets_cmd| {
        datasets_cmd.mut_subcommand("list", |mut list_cmd| {
            let hidden = datasets::filter::hidden_list_args();
            if hidden.contains(&"__disable_help") {
                // Fully resolved — replace with an empty command so the
                // completion engine offers nothing at all (not even `--`).
                return clap::Command::new("list")
                    .alias("ls")
                    .disable_help_flag(true);
            }
            for id in &hidden {
                list_cmd = list_cmd.mut_arg(*id, |arg| arg.hide(true));
            }
            list_cmd
        })
    });
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

#[tokio::main]
async fn main() {
    // Double-tap detection: if the user taps Tab repeatedly at the root
    // level (no subcommand context yet), show the full command list.
    // Only triggers for root-level completions like `veks <TAB>`, not
    // when already inside a subcommand like `veks prepare <TAB>`.
    if std::env::var_os("COMPLETE").is_some() && is_root_completion() {
        if let Some(output) = check_triple_tap() {
            print!("{}", output);
            std::process::exit(0);
        }
    }

    CompleteEnv::with_factory(build_augmented_cli).complete();

    let veks = Veks::parse();

    match veks.command {
        Commands::Datasets(args) => datasets::run(args),
        Commands::Prepare(args) => prepare::run(args),
        Commands::Run(args) => pipeline::run_pipeline(args),
        Commands::Script(args) => pipeline::run_script(args),
        Commands::Pipeline { args } => pipeline::cli::run_direct(args),
        Commands::Check(args) => check::run(args),
        Commands::Publish(args) => publish::run(args),
        Commands::Completions(args) => cli::completions(args),
        Commands::Help(args) => run_help(args),
        Commands::Shorthand(args) => dispatch_shorthand(args),
    }
}

// ---------------------------------------------------------------------------
// Triple-tap detection
// ---------------------------------------------------------------------------

/// Check if the current completion invocation is at the root level.
///
/// The shell passes args like `-- veks ""` for `veks <TAB>` (root) vs
/// `-- veks prepare ""` for `veks prepare <TAB>` (subcommand context).
/// We only want the expanded list at the root level — when there are at
/// most 2 real words after `--` (the program name and a partial/empty word).
fn is_root_completion() -> bool {
    let args: Vec<String> = std::env::args().collect();
    // Find `--` separator; everything after it is the completion words.
    let after_separator = if let Some(pos) = args.iter().position(|a| a == "--") {
        &args[pos + 1..]
    } else {
        // No separator — check raw arg count (program + maybe partial word)
        return args.len() <= 2;
    };
    // Root level: just the program name, or program name + partial/empty word
    after_separator.len() <= 2
}

const TAP_FILE: &str = "/tmp/.veks_tab";
const TAP_COUNT: usize = 2;

/// Check if the user has tapped Tab repeatedly in the same completion state.
///
/// Tracks the completion args (cursor position + partial input) in a file.
/// If the same state is seen `TAP_COUNT` times in a row, the user is
/// repeatedly hitting Tab without changing their input — trigger the
/// expanded command list. No timing constraint; purely state-based.
///
/// If /tmp is not writable or accessible, silently returns `None`.
fn check_triple_tap() -> Option<String> {
    // Build a fingerprint of the current completion state from env/args.
    // The shell passes the words and cursor index via env vars or args.
    let state_key: String = std::env::args().skip(1).collect::<Vec<_>>().join("\x00");

    // Read previous state (best-effort)
    let (prev_key, prev_count) = std::fs::read_to_string(TAP_FILE)
        .ok()
        .and_then(|content| {
            let mut lines = content.lines();
            let count: usize = lines.next()?.parse().ok()?;
            let key = lines.next().unwrap_or("").to_string();
            Some((key, count))
        })
        .unwrap_or_default();

    let count = if state_key == prev_key {
        prev_count + 1
    } else {
        1
    };

    // Write back (best-effort, silently ignore errors)
    let _ = std::fs::write(TAP_FILE, format!("{}\n{}", count, state_key));

    if count >= TAP_COUNT {
        // Reset so next tap starts fresh
        let _ = std::fs::write(TAP_FILE, "");
        Some(build_full_command_list())
    } else {
        None
    }
}

/// Build a fish-format completion list of all available commands.
fn build_full_command_list() -> String {
    let mut lines: Vec<String> = Vec::new();

    // Root commands
    lines.push("datasets\tBrowse, search, and manage datasets".into());
    lines.push("prepare\tImport, stratify, and prepare datasets".into());
    lines.push("run\tExecute pipeline from dataset.yaml".into());
    lines.push("script\tEmit pipeline as shell script".into());
    lines.push("pipeline\tExecute a single pipeline command".into());
    lines.push("check\tPre-flight checks for dataset readiness".into());
    lines.push("publish\tPublish dataset to S3".into());
    lines.push("help\tShow help for a command".into());

    // Prepare subcommands (usable directly via shorthand)
    lines.push("import\tBootstrap dataset from source files".into());
    lines.push("stratify\tAdd sized profiles to dataset".into());
    lines.push("catalog\tGenerate dataset catalog index".into());
    lines.push("cache-compress\tCompress eligible cache files".into());
    lines.push("cache-uncompress\tDecompress cache files".into());

    // Datasets subcommands (usable directly via shorthand)
    lines.push("list\tList available datasets from catalogs".into());
    lines.push("cache\tList locally cached datasets".into());
    lines.push("curlify\tGenerate curl download commands".into());
    lines.push("prebuffer\tDownload and cache dataset facets".into());

    // Pipeline commands from the registry
    let registry = pipeline::registry::CommandRegistry::with_builtins();
    for path in registry.command_paths() {
        lines.push(format!("{}\tpipeline: {}", path, path));
    }

    lines.join("\n") + "\n"
}

// ---------------------------------------------------------------------------
// Shorthand subcommand dispatch
// ---------------------------------------------------------------------------

/// Subcommand-to-group mapping for shorthand dispatch.
///
/// All subcommand names MUST be globally unique across all groups.
/// This is a design invariant — new subcommands must not collide with
/// existing names in any group. If a collision is introduced, it must
/// be resolved by renaming one of the conflicting commands.
const SHORTHAND_COMMANDS: &[(&str, &str)] = &[
    // prepare subcommands
    ("import", "prepare"),
    ("stratify", "prepare"),
    ("catalog", "prepare"),
    ("cache-compress", "prepare"),
    ("cache-uncompress", "prepare"),
    // datasets subcommands
    ("list", "datasets"),
    ("ls", "datasets"),
    ("cache", "datasets"),
    ("curlify", "datasets"),
    ("prebuffer", "datasets"),
];

/// Dispatch an unrecognized root-level subcommand.
///
/// Looks up the name in the shorthand table. If found, re-parses with
/// the group prefix inserted. If not found, falls through to the
/// pipeline command registry as a last resort.
fn dispatch_shorthand(args: Vec<String>) {
    if args.is_empty() {
        eprintln!("Error: no command specified. Run `veks --help` for usage.");
        std::process::exit(1);
    }

    let name = &args[0];

    // Look up in the shorthand table
    if let Some((_, group)) = SHORTHAND_COMMANDS.iter().find(|(cmd, _)| *cmd == name.as_str()) {
        let mut full_args = vec!["veks".to_string(), group.to_string()];
        full_args.extend(args);
        let veks = Veks::parse_from(full_args);
        match veks.command {
            Commands::Datasets(a) => datasets::run(a),
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
    groups: &std::collections::BTreeMap<String, Vec<(&str, String)>>,
    markdown: bool,
) {
    let root_cmds = root_commands();

    if markdown {
        println!("# veks commands\n");
        println!("## root\n");
        println!("| Command | Summary |");
        println!("|---------|---------|");
        for (name, summary) in &root_cmds {
            println!("| `{}` | {} |", name, summary);
        }
        println!();
        println!("## pipeline commands\n");
        for (group, commands) in groups {
            println!("### {}\n", group);
            println!("| Command | Summary |");
            println!("|---------|---------|");
            for (path, summary) in commands {
                let subname = path.splitn(2, ' ').nth(1).unwrap_or(path);
                println!("| `{}` | {} |", subname, summary);
            }
            println!();
        }
    } else {
        let max_root = root_cmds.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
        println!("root:");
        for (name, summary) in &root_cmds {
            println!("  {:<width$}  {}", name, summary, width = max_root);
        }
        println!();
        println!("pipeline:");
        for (group, commands) in groups {
            println!("  {}:", group);
            let max_len = commands.iter()
                .map(|(p, _)| p.splitn(2, ' ').nth(1).unwrap_or(p).len())
                .max()
                .unwrap_or(0);
            for (path, summary) in commands {
                let subname = path.splitn(2, ' ').nth(1).unwrap_or(path);
                println!("    {:<width$}  {}", subname, summary, width = max_len);
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
