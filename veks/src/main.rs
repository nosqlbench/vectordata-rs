// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)] // legacy completion functions pending removal

use veks::{cli, datasets, pipeline, prepare};



/// Full version string including git sha, build number, and build time.
fn long_version() -> &'static str {
    use std::sync::OnceLock;
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let epoch: u64 = env!("VEKS_BUILD_NUMBER").parse().unwrap_or(0);
        let ts = std::time::UNIX_EPOCH + std::time::Duration::from_secs(epoch);
        let build_time = ts.duration_since(std::time::UNIX_EPOCH).map(|d| {
            // UTC timestamp from epoch seconds
            let s = d.as_secs();
            let days = s / 86400;
            let time = s % 86400;
            let h = time / 3600;
            let m = (time % 3600) / 60;
            let sec = time % 60;
            // Days since epoch to Y-M-D (simplified: good through 2099)
            let mut y = 1970i64;
            let mut rem = days as i64;
            loop {
                let ydays = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
                if rem < ydays { break; }
                rem -= ydays;
                y += 1;
            }
            let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
            let mdays = [31, if leap {29} else {28}, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut mo = 0usize;
            while mo < 12 && rem >= mdays[mo] { rem -= mdays[mo]; mo += 1; }
            format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, rem + 1, h, m, sec)
        }).unwrap_or_else(|_| "unknown".to_string());
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let age_secs = now_epoch.saturating_sub(epoch);
        let age_str = if age_secs < 60 {
            format!("{} seconds ago", age_secs)
        } else if age_secs < 3600 {
            format!("{} minutes ago", age_secs / 60)
        } else if age_secs < 86400 {
            let h = age_secs / 3600;
            let m = (age_secs % 3600) / 60;
            if m == 0 { format!("{} hours ago", h) }
            else { format!("{}h {}m ago", h, m) }
        } else {
            let d = age_secs / 86400;
            let h = (age_secs % 86400) / 3600;
            if d == 1 { format!("1 day {}h ago", h) }
            else { format!("{} days {}h ago", d, h) }
        };
        format!("{}\ngit:    {}\nbuild:  {} ({}, {})\nrustc:  {}",
            env!("CARGO_PKG_VERSION"),
            env!("VEKS_BUILD_HASH"),
            env!("VEKS_BUILD_NUMBER"),
            build_time,
            age_str,
            env!("VEKS_RUSTC_VERSION"),
        )
    })
}

/// Arguments for `veks help`.
#[derive(veks_completion_derive::VeksCli)]
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

/// Build the whole veks CLI as a clap-free [`veks_completion::CommandSpec`] —
/// the replacement for [`build_augmented_cli`]. Assembles the derived
/// subcommand specs (`datasets`/`prepare`/`completions`/`help`), grafts the
/// pipeline tree (`pipeline_command_spec`) both under `pipeline` and as
/// top-level groups, and adds shorthand aliases (a group's leaf subcommands at
/// the root — `veks run` ≡ `prepare run`, `veks knn` ≡ `compute knn`), skipping
/// name collisions. One spec drives parse + help + completion.
fn build_veks_spec() -> veks_completion::CommandSpec {
    use veks_completion::{CommandSpec, VeksCli};

    let mut root = CommandSpec::new("veks")
        .about("veks — vector dataset toolkit")
        .after_help(
            "Most commands are organized under one of the top-level categories above.\n\
             If you know a command, you don't have to use the full `veks <category> ...` format.\n\
             For example, `veks run` is shorthand for `veks prepare run`.\n\n\
             Hit TAB twice to see the full expanded command list.",
        );
    root.subcommand_required = true;

    // Canonical, derive-backed subcommands.
    root = root.subcommand(datasets::DatasetsArgs::veks_command_spec("datasets"));
    root = root.subcommand(prepare::PrepareArgs::veks_command_spec("prepare"));
    root = root.subcommand(cli::CompletionsArgs::veks_command_spec("completions"));
    root = root.subcommand(HelpArgs::veks_command_spec("help"));

    // Pipeline groups grafted at the top level (e.g. `veks compute knn`),
    // skipping any that collide with a derive subcommand, plus the full tree
    // under `pipeline`.
    let pipeline = pipeline::cli::pipeline_command_spec();
    for group in &pipeline.subcommands {
        if root.subcommands.iter().all(|c| c.name != group.name) {
            root = root.subcommand(group.clone());
        }
    }
    root = root.subcommand(pipeline);

    // Shorthand aliases: graft each top-level group's leaf subcommands at the
    // root (matches `dispatch_shorthand`), first-wins on collision.
    let groups: Vec<CommandSpec> = root.subcommands.clone();
    for group in &groups {
        for leaf in &group.subcommands {
            if root.subcommands.iter().all(|c| c.name != leaf.name) {
                root = root.subcommand(leaf.clone());
            }
        }
    }

    root
}

fn main() {
    // Dynamic completion engine — walks the augmented clap tree so
    // completion candidates are always in sync with the actual CLI.
    // Handles _VEKS_COMPLETE=bash and legacy COMPLETE=bash.
    // Completion + diagnostics + the parse-consistency audit now run off the
    // clap-free `CommandSpec` (the single source). Dispatch still goes through
    // clap below until the parse cutover lands.
    let spec = build_veks_spec();
    let tree = veks_completion::cli::build_completion_tree(&spec, &cli::dyncomp::shared_resolvers());
    if veks_completion::handle_complete_env("veks", &tree) {
        std::process::exit(0);
    }
    // Engine-level diagnostics — `---dump-tree`, `---trace-completion`,
    // `---validate`, etc. Triple-dash prefix can't collide with normal `--`
    // flags. Used by tests to introspect the live tree shape.
    if veks_completion::handle_diagnostic_args("veks", &tree) {
        std::process::exit(0);
    }

    // Parse-consistency enforcement — `---audit-options`. Walks the spec and
    // fails if any option name parses differently across commands.
    if std::env::args().any(|a| a == "---audit-options") {
        let mismatches = cli::dyncomp::audit_parse_consistency_spec(&spec);
        if mismatches.is_empty() {
            println!("option parse-consistency: OK");
            std::process::exit(0);
        }
        eprintln!("option parse-consistency: {} violation(s)\n", mismatches.len());
        for m in &mismatches {
            eprint!("{m}");
        }
        std::process::exit(1);
    }

    // Parse + dispatch entirely off the clap-free spec.
    let raw: Vec<String> = std::env::args().skip(1).collect();

    // `--help`/`-h`/`--version` anywhere — clap used to handle these for us.
    if dispatch_help_version(&spec, &raw) {
        return;
    }

    let first = raw.iter().find(|a| !a.starts_with('-')).cloned();
    match first.as_deref() {
        None => print!("{}", veks_completion::cli::render_help(&spec)),
        Some("datasets") => {
            let parsed = parse_or_exit(&spec, &raw);
            datasets::run(from_parsed::<datasets::DatasetsArgs>(sub_or_exit(&spec, &parsed, "datasets")));
        }
        Some("prepare") => {
            let parsed = parse_or_exit(&spec, &raw);
            prepare::run(from_parsed::<prepare::PrepareArgs>(sub_or_exit(&spec, &parsed, "prepare")));
        }
        Some("completions") => {
            let parsed = parse_or_exit(&spec, &raw);
            cli::completions(from_parsed::<cli::CompletionsArgs>(sub_or_exit(&spec, &parsed, "completions")));
        }
        Some("help") => {
            let parsed = parse_or_exit(&spec, &raw);
            let args = parsed
                .subcommand()
                .map(|(_, sub)| from_parsed::<HelpArgs>(sub))
                .unwrap_or_else(|| from_parsed::<HelpArgs>(&Default::default()));
            run_help(args);
        }
        // `veks pipeline <group> <cmd> …` — strip the leading `pipeline` word.
        Some("pipeline") => {
            let pos = raw.iter().position(|a| a == "pipeline").unwrap();
            pipeline::cli::run_direct(raw[pos + 1..].to_vec());
        }
        // Pipeline groups + shorthands (`compute knn`, `run`, `bootstrap`, …).
        Some(_) => dispatch_shorthand(&spec, raw),
    }
}

/// Extract a typed command from already-parsed args, exiting on conversion error.
fn from_parsed<T: veks_completion::VeksCli>(parsed: &veks_completion::cli::ParsedArgs) -> T {
    T::veks_from_parsed(parsed).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    })
}

/// Parse `argv` against `spec`, printing the error and exiting on failure.
fn parse_or_exit(
    spec: &veks_completion::CommandSpec,
    argv: &[String],
) -> veks_completion::cli::ParsedArgs {
    veks_completion::cli::parse(spec, argv).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    })
}

/// The parsed args of the chosen subcommand; renders that group's help and
/// exits when no subcommand was given (clap's `arg_required_else_help`).
fn sub_or_exit<'a>(
    spec: &veks_completion::CommandSpec,
    parsed: &'a veks_completion::cli::ParsedArgs,
    group: &str,
) -> &'a veks_completion::cli::ParsedArgs {
    match parsed.subcommand() {
        Some((_, sub)) => sub,
        None => {
            if let Some(node) = spec.subcommands.iter().find(|c| c.name == group) {
                print!("{}", veks_completion::cli::render_help(node));
            }
            std::process::exit(0);
        }
    }
}

/// Handle `--help`/`-h`/`--version` like clap did: render the help for the
/// deepest matched command (or version) and report whether it was handled.
fn dispatch_help_version(spec: &veks_completion::CommandSpec, argv: &[String]) -> bool {
    if argv.iter().any(|a| a == "--version" || a == "-V") {
        println!("veks {}", long_version());
        return true;
    }
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        // Walk to the deepest subcommand named by the leading positional words.
        let mut node = spec;
        for w in argv.iter().filter(|a| !a.starts_with('-')) {
            match node.subcommands.iter().find(|c| &c.name == w || c.aliases.iter().any(|a| a == w)) {
                Some(child) => node = child,
                None => break,
            }
        }
        print!("{}", veks_completion::cli::render_help(node));
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Shorthand subcommand dispatch
// ---------------------------------------------------------------------------

/// Map each `datasets`/`prepare` leaf subcommand name (and its aliases) to its
/// owning group, so shorthands like `veks ping` / `veks bootstrap` re-dispatch
/// to `datasets ping` / `prepare bootstrap`. Built from the spec — no clap.
fn build_shorthand_map_spec(spec: &veks_completion::CommandSpec) -> Vec<(String, String)> {
    let mut map = Vec::new();
    for group in ["datasets", "prepare"] {
        if let Some(g) = spec.subcommands.iter().find(|c| c.name == group) {
            for child in &g.subcommands {
                map.push((child.name.clone(), group.to_string()));
                for alias in &child.aliases {
                    map.push((alias.clone(), group.to_string()));
                }
            }
        }
    }
    map
}

/// Dispatch a root-level word that isn't a canonical subcommand: either a
/// `datasets`/`prepare` shorthand (re-parsed with the group prefix) or a
/// pipeline command (handed to `run_direct`, which resolves it itself).
fn dispatch_shorthand(spec: &veks_completion::CommandSpec, args: Vec<String>) {
    if args.is_empty() {
        eprintln!("Error: no command specified. Run `veks --help` for usage.");
        std::process::exit(1);
    }

    let name = &args[0];
    let shorthand_map = build_shorthand_map_spec(spec);

    if let Some((_, group)) = shorthand_map.iter().find(|(cmd, _)| cmd == name) {
        let mut full = vec![group.clone()];
        full.extend(args.clone());
        let parsed = parse_or_exit(spec, &full);
        match parsed.subcommand() {
            Some(("datasets", sub)) => datasets::run(from_parsed::<datasets::DatasetsArgs>(sub)),
            Some(("prepare", sub)) => prepare::run(from_parsed::<prepare::PrepareArgs>(sub)),
            _ => unreachable!("shorthand map only contains datasets/prepare leaves"),
        }
        return;
    }

    // Not a datasets/prepare shorthand — a pipeline command.
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
        print!("{}", veks_completion::cli::render_help(&build_veks_spec()));
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

/// Generate help text for a root-level command by rendering its spec help.
fn root_command_help(name: &str) -> Option<String> {
    let spec = build_veks_spec();
    let sub = spec.subcommands.iter().find(|c| c.name == name)?;
    Some(veks_completion::cli::render_help(sub))
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
    use veks_completion::VeksCli;
    let derive_specs = [
        datasets::DatasetsArgs::veks_command_spec("datasets"),
        prepare::PrepareArgs::veks_command_spec("prepare"),
    ];
    let internal = ["pipeline", "help", "completions"];
    let mut leaf_commands: Vec<(String, String)> = Vec::new();

    for sub in &derive_specs {
        let name = sub.name.clone();
        if internal.contains(&name.as_str()) {
            continue;
        }
        if !sub.subcommands.is_empty() {
            let children: Vec<(String, String)> = sub
                .subcommands
                .iter()
                .map(|c| (c.name.clone(), c.about.clone().unwrap_or_default()))
                .collect();
            all_groups.insert(name, children);
        } else {
            leaf_commands.push((name, sub.about.clone().unwrap_or_default()));
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

#[cfg(test)]
mod tests {
    use super::*;

    // The "one parse-definition per option name" gate now runs off the spec —
    // see `veks_spec_parse_consistency_holds` below.

    /// Validates the additive `#[derive(VeksCli)]` pass on the nested datasets
    /// command tree: the derived spec must reproduce the subcommands and their
    /// options (the clap-free path).
    #[test]
    fn datasets_veks_cli_spec_is_correct() {
        use veks_completion::VeksCli;
        let spec = datasets::DatasetsArgs::veks_command_spec("datasets");
        let subs: Vec<&str> = spec.subcommands.iter().map(|c| c.name.as_str()).collect();
        assert!(subs.contains(&"ping"), "got subcommands {subs:?}");
        assert!(subs.contains(&"list"));
        assert!(subs.contains(&"config"));

        let ping = spec.subcommands.iter().find(|c| c.name == "ping").unwrap();
        let pflags: Vec<&str> = ping.options.iter().map(|o| o.flag()).collect();
        assert!(pflags.contains(&"--at"), "ping flags {pflags:?}");
        assert!(pflags.contains(&"--dataset"));
        // Aliases survive the derive (ping has #[command(alias = "probe")]).
        assert!(ping.aliases.contains(&"probe".to_string()), "ping aliases {:?}", ping.aliases);

        // And the parser resolves an alias to the canonical subcommand.
        let parsed = veks_completion::cli::parse(
            &spec,
            &["probe", "--dataset", "d"].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(parsed.subcommand().map(|(n, _)| n), Some("ping"));

        // `config` nests another subcommand enum (ConfigSubcommand → catalog → …).
        let config = spec.subcommands.iter().find(|c| c.name == "config").unwrap();
        assert!(config.subcommand_required, "config is a group");
        assert!(!config.subcommands.is_empty(), "config has nested subcommands");
    }

    /// The hand-assembled top-level spec: canonical subcommands + pipeline graft
    /// + shorthands, all reachable — the clap-free replacement for
    /// `build_augmented_cli`.
    #[test]
    fn veks_spec_assembles_full_tree() {
        let spec = build_veks_spec();
        let names: Vec<&str> = spec.subcommands.iter().map(|c| c.name.as_str()).collect();
        for expected in ["datasets", "prepare", "completions", "pipeline", "compute", "run", "knn"] {
            assert!(names.contains(&expected), "missing {expected:?}; got {names:?}");
        }

        // `compute` group still contains `knn` (graft didn't flatten it away).
        let compute = spec.subcommands.iter().find(|c| c.name == "compute").unwrap();
        assert!(compute.subcommands.iter().any(|c| c.name == "knn"));

        // Shorthand `veks run` (≡ prepare run / RunArgs, no required args) parses.
        let argv: Vec<String> = ["run", "--dry-run"].iter().map(|s| s.to_string()).collect();
        let parsed = veks_completion::cli::parse(&spec, &argv);
        assert!(parsed.is_ok(), "run shorthand parse: {:?}", parsed.err());
        assert!(parsed.unwrap().subcommand().map(|(n, _)| n) == Some("run"));

        // Completion builds from the spec and offers a grafted group.
        let tree = veks_completion::cli::build_completion_tree(&spec, &cli::dyncomp::shared_resolvers());
        let datasets_subs = veks_completion::complete(&tree, &["veks", "datasets", ""]);
        assert!(datasets_subs.iter().any(|s| s == "ping"), "datasets subs: {datasets_subs:?}");
        // Shared resolver fires: `datasets ping --at <TAB>` → live catalog values.
        let at_vals = veks_completion::complete(&tree, &["veks", "datasets", "ping", "--at", ""]);
        let _ = at_vals; // value set depends on local config; just exercise the path
    }

    /// The assembled spec must satisfy the same "one parse-definition per name"
    /// invariant as the clap tree — the gate has to pass on the clap-free path
    /// before the cutover can remove clap.
    #[test]
    fn veks_spec_parse_consistency_holds() {
        let spec = build_veks_spec();
        let mismatches = cli::dyncomp::audit_parse_consistency_spec(&spec);
        assert!(
            mismatches.is_empty(),
            "spec parse-consistency violations ({}):\n{}",
            mismatches.len(),
            mismatches.iter().map(|m| m.to_string()).collect::<String>()
        );
    }
}
