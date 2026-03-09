// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use veks::{analyze, bulkdl, cli, convert, datasets, import, pipeline};

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
    /// Analyze vector data files and datasets
    Analyze(analyze::AnalyzeArgs),
    /// Bulk file downloader driven by YAML config with token expansion
    Bulkdl(bulkdl::BulkDlArgs),
    /// Browse, search, and manage datasets
    Datasets(datasets::DatasetsArgs),
    /// Convert vector data between formats
    Convert(convert::ConvertArgs),
    /// Import data into preferred internal format by facet type
    Import(import::ImportArgs),
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
    /// Generate shell completions
    Completions(cli::CompletionsArgs),
    /// Show detailed help for a pipeline command or command group
    Help(HelpArgs),
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
    CompleteEnv::with_factory(build_augmented_cli).complete();

    let veks = Veks::parse();

    match veks.command {
        Commands::Analyze(args) => analyze::run(args),
        Commands::Bulkdl(args) => bulkdl::run(args).await,
        Commands::Convert(args) => convert::run(args),
        Commands::Datasets(args) => datasets::run(args),
        Commands::Import(args) => import::run(args),
        Commands::Run(args) => pipeline::run_pipeline(args),
        Commands::Script(args) => pipeline::run_script(args),
        Commands::Pipeline { args } => pipeline::cli::run_direct(args),
        Commands::Completions(args) => cli::completions(args),
        Commands::Help(args) => run_help(args),
    }
}

/// Root-level command descriptions for `veks help`.
fn root_commands() -> Vec<(&'static str, &'static str)> {
    vec![
        ("analyze", "Analyze vector data files and datasets"),
        ("bulkdl", "Bulk file downloader driven by YAML config with token expansion"),
        ("datasets", "Browse, search, and manage datasets"),
        ("convert", "Convert vector data between formats"),
        ("import", "Import data into preferred internal format by facet type"),
        ("run", "Execute a command stream pipeline defined in dataset.yaml"),
        ("script", "Emit a dataset pipeline as an equivalent shell script"),
        ("pipeline", "Execute a single pipeline command directly"),
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

    let use_ansi = atty::is(atty::Stream::Stdout);
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
