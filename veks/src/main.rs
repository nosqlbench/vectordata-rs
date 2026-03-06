// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use veks::{analyze, bulkdl, cli, convert, import, pipeline};

use clap::{CommandFactory, Parser, Subcommand};
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
    /// Convert vector data between formats
    Convert(convert::ConvertArgs),
    /// Import data into preferred internal format by facet type
    Import(import::ImportArgs),
    /// Execute a command stream pipeline defined in dataset.yaml
    Run(pipeline::RunArgs),
    /// Execute a single pipeline command directly
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
    cmd
}

#[tokio::main]
async fn main() {
    CompleteEnv::with_factory(build_augmented_cli).complete();

    let veks = Veks::parse();

    match veks.command {
        Commands::Analyze(args) => analyze::run(args),
        Commands::Bulkdl(args) => bulkdl::run(args).await,
        Commands::Convert(args) => convert::run(args),
        Commands::Import(args) => import::run(args),
        Commands::Run(args) => pipeline::run_pipeline(args),
        Commands::Pipeline { args } => pipeline::cli::run_direct(args),
        Commands::Completions(args) => cli::completions(args),
        Commands::Help(args) => run_help(args),
    }
}

/// Execute `veks help` — display documentation for pipeline commands and groups.
fn run_help(args: HelpArgs) {
    use pipeline::registry::CommandRegistry;
    use std::collections::BTreeMap;

    let registry = CommandRegistry::with_builtins();
    let paths = registry.command_paths();

    // Group command paths by first word.
    let mut groups: BTreeMap<String, Vec<(&str, String)>> = BTreeMap::new();
    for path in &paths {
        let mut parts = path.splitn(2, ' ');
        let group = parts.next().unwrap().to_string();
        let factory = registry.get(path).unwrap();
        let cmd = factory();
        let summary = cmd.command_doc().summary;
        groups.entry(group).or_default().push((path, summary));
    }

    // --list mode: show all commands grouped
    if args.list || args.command_path.is_empty() {
        print_command_list(&groups, args.markdown);
        return;
    }

    let query = args.command_path.join(" ");

    // Check if query matches a group name (DOC-07)
    if groups.contains_key(&query) {
        print_group_help(&query, &groups, &registry, args.markdown);
        return;
    }

    // Check if query matches a specific command
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

    eprintln!("Unknown command or group: '{}'", query);
    eprintln!("Use 'veks help --list' to see all available commands.");
    std::process::exit(1);
}

/// Print all commands grouped by category (DOC-05 --list mode).
fn print_command_list(
    groups: &std::collections::BTreeMap<String, Vec<(&str, String)>>,
    markdown: bool,
) {
    if markdown {
        println!("# veks pipeline commands\n");
        for (group, commands) in groups {
            println!("## {}\n", group);
            println!("| Command | Summary |");
            println!("|---------|---------|");
            for (path, summary) in commands {
                let subname = path.splitn(2, ' ').nth(1).unwrap_or(path);
                println!("| `{}` | {} |", subname, summary);
            }
            println!();
        }
    } else {
        for (group, commands) in groups {
            println!("{}:", group);
            let max_len = commands.iter()
                .map(|(p, _)| p.splitn(2, ' ').nth(1).unwrap_or(p).len())
                .max()
                .unwrap_or(0);
            for (path, summary) in commands {
                let subname = path.splitn(2, ' ').nth(1).unwrap_or(path);
                println!("  {:<width$}  {}", subname, summary, width = max_len);
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
