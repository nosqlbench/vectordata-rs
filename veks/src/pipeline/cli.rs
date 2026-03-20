// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Direct CLI invocation of pipeline commands.
//!
//! Builds a clap `Command` tree from the `CommandRegistry` so that pipeline
//! commands can be invoked directly (with shell tab-completion) instead of
//! only through YAML pipeline files:
//!
//! ```text
//! veks pipeline analyze stats --source=test.fvec
//! veks pipeline generate predicated --input-dir=./dataset --selectivity=0.1
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use std::ffi::OsStr;

use clap::{Arg, Command, ValueHint};
use clap_complete::engine::{ArgValueCompleter, CompletionCandidate};
use indexmap::IndexMap;

use super::command::{Options, StreamContext};
use super::progress::ProgressLog;
use super::registry::CommandRegistry;
use super::resource::ResourceType;

/// Build a `clap::Command` representing all registered pipeline commands.
///
/// Commands are grouped by their first word (e.g. `"analyze"`, `"generate"`)
/// and each group becomes a subcommand containing its individual commands as
/// nested subcommands with `Arg` entries derived from `describe_options()`.
pub fn build_pipeline_command() -> Command {
    let registry = CommandRegistry::with_builtins();
    let paths = registry.command_paths();

    // Group command paths by first word.
    let mut groups: BTreeMap<String, Vec<(String, Vec<super::command::OptionDesc>, super::command::CommandDoc, Vec<super::command::ResourceDesc>)>> =
        BTreeMap::new();

    for path in &paths {
        let mut parts = path.splitn(2, ' ');
        let group = parts.next().unwrap().to_string();
        let subname = parts.next().unwrap_or("").to_string();

        // Get option descriptions, documentation, and resource declarations from a fresh command instance.
        let factory = registry.get(path).unwrap();
        let cmd = factory();
        let opts = cmd.describe_options();
        let doc = cmd.command_doc();
        let resources = cmd.describe_resources();

        groups.entry(group).or_default().push((subname, opts, doc, resources));
    }

    let mut pipeline_cmd = Command::new("pipeline")
        .about("Execute a single pipeline command directly")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .disable_help_subcommand(true);

    for (group, commands) in groups {
        let mut group_cmd = Command::new(group.clone())
            .about(format!("{} commands", group))
            .subcommand_required(true)
            .arg_required_else_help(true)
            .disable_help_subcommand(true);

        for (subname, opts, doc, resources) in commands {
            // Single-word commands (e.g. "survey") have no subname — attach
            // their options directly to the group command.
            let is_direct = subname.is_empty();
            let mut sub_cmd = if is_direct {
                group_cmd.clone()
                    .about(doc.summary.clone())
                    .long_about(doc.body.clone())
                    .subcommand_required(false)
                    .arg_required_else_help(false)
            } else {
                Command::new(subname)
                    .about(doc.summary.clone())
                    .long_about(doc.body.clone())
            }
                .arg(
                    Arg::new("emit-yaml")
                        .long("emit-yaml")
                        .num_args(0)
                        .help("Emit a YAML step block for dataset.yaml instead of executing"),
                )
                .arg(
                    Arg::new("id")
                        .long("id")
                        .help("Step identifier for the YAML block"),
                )
                .arg(
                    Arg::new("after")
                        .long("after")
                        .help("Comma-separated step IDs this step depends on"),
                );

            // Collect option names before consuming opts, for resource alias conflict check.
            let opt_names: std::collections::HashSet<String> = opts.iter()
                .map(|o| o.name.clone())
                .collect();

            for opt in opts {
                // DOC-04a: Append type/default info to help text so it
                // appears as a shell-rendered description at both
                // --option-name and value positions, without injecting
                // anything into the command line.
                let help_text = build_option_help(&opt.description, &opt.type_name, opt.default.as_deref());

                let mut arg = Arg::new(opt.name.clone())
                    .long(opt.name.clone())
                    .required(opt.required)
                    .help(help_text);

                if let Some(ref default) = opt.default {
                    arg = arg.default_value(default.clone());
                }

                if opt.type_name == "Path" {
                    arg = arg.value_hint(ValueHint::AnyPath);
                    // Path options get file completion (data files) plus variable completion.
                    arg = arg.add(ArgValueCompleter::new(datafile_completer));
                }

                // Add variable completion so tab-completion suggests
                // '${variables:name}' from variables.yaml.
                arg = arg.add(ArgValueCompleter::new(variable_completer));

                sub_cmd = sub_cmd.arg(arg);
            }

            // Add --resources arg with completion candidates listing all resource
            // types. Every command gets the full list so users can always
            // discover what's available; commands that declare specific resources
            // also get per-command help text noting which are applicable.
            {
                let help_text = if !resources.is_empty() {
                    let resource_names: Vec<&str> = resources.iter()
                        .map(|r| r.name.as_str())
                        .collect();
                    format!("Resource limits (applicable: {})", resource_names.join(", "))
                } else {
                    "Resource limits (this command declares no resource requirements)".to_string()
                };

                sub_cmd = sub_cmd.arg(
                    Arg::new("resources")
                        .long("resources")
                        .help(help_text)
                        .num_args(1)
                        .add(ArgValueCompleter::new(resource_completer)),
                );
            }

            // Add long-form resource aliases (e.g., --mem, --readahead) for each
            // declared resource. Skip aliases that conflict with existing option names.
            if !resources.is_empty() {
                for res in &resources {
                    if opt_names.contains(&res.name) {
                        continue; // Skip — option already provides this flag
                    }
                    if let Some(rt) = ResourceType::from_name(&res.name) {
                        sub_cmd = sub_cmd.arg(
                            Arg::new(format!("resource-{}", res.name))
                                .long(res.name.clone())
                                .help(format!("{} (resource alias)", rt.description())),
                        );
                    }
                }
            }

            // Add --governor strategy selection with value completions
            sub_cmd = sub_cmd.arg(
                Arg::new("governor")
                    .long("governor")
                    .default_value("maximize")
                    .help("Governor strategy: maximize, conservative, fixed")
                    .add(ArgValueCompleter::new(governor_completer)),
            );

            if is_direct {
                // Single-word command: replace the group command entirely
                group_cmd = sub_cmd;
            } else {
                group_cmd = group_cmd.subcommand(sub_cmd);
            }
        }

        pipeline_cmd = pipeline_cmd.subcommand(group_cmd);
    }

    pipeline_cmd
}

/// Print pipeline help — either all commands or a specific group.
///
/// When `group` is `None`, lists all pipeline commands grouped by category.
/// When `group` is `Some(name)`, lists commands in that group with summaries,
/// or reports that the group is unknown.
fn print_pipeline_help(registry: &CommandRegistry, group: Option<&str>) {
    let paths = registry.command_paths();

    // Group command paths by first word.
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for path in &paths {
        let mut parts = path.splitn(2, ' ');
        let g = parts.next().unwrap().to_string();
        let subname = parts.next().unwrap_or("").to_string();
        let factory = registry.get(path).unwrap();
        let summary = factory().command_doc().summary;
        groups.entry(g).or_default().push((subname, summary));
    }

    match group {
        None => {
            println!("Execute a single pipeline command directly.\n");
            println!("Usage: veks pipeline <group> <command> [--option=value ...]\n");
            for (g, commands) in &groups {
                println!("  {}:", g);
                let max_len = commands.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
                for (subname, summary) in commands {
                    println!("    {:<width$}  {}", subname, summary, width = max_len);
                }
                println!();
            }
            println!("Use 'veks pipeline <group> <command> --help' for detailed command documentation.");
        }
        Some(g) => {
            if let Some(commands) = groups.get(g) {
                println!("{} — pipeline command group\n", g);
                let max_len = commands.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
                for (subname, summary) in commands {
                    println!("  {:<width$}  {}", subname, summary, width = max_len);
                }
                println!("\nUse 'veks pipeline {} <command> --help' for detailed documentation.", g);
            } else {
                eprintln!("Unknown pipeline group: '{}'\n", g);
                eprintln!("Available groups:");
                for g in groups.keys() {
                    eprintln!("  {}", g);
                }
            }
        }
    }
}

/// Completion candidates for `--governor`: lists strategy names with descriptions.
pub fn governor_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let prefix = current.to_string_lossy().to_lowercase();
    ["maximize", "conservative", "fixed"]
        .iter()
        .filter(|v| prefix.is_empty() || v.starts_with(&*prefix))
        .map(|v| CompletionCandidate::new(*v))
        .collect()
}

/// Recognized data file extensions for pipeline Path-type options.
///
/// These cover the vector, slab, and metadata formats that pipeline commands
/// typically operate on.
const DATA_EXTENSIONS: &[&str] = &[
    "fvec", "ivec", "mvec", "bvec", "dvec", "svec",
    "slab", "npy", "parquet", "json", "yaml",
];

/// Completion candidates for Path-type options: lists files in the current
/// directory (recursively one level) whose extension matches a recognized
/// data format.
///
/// When the user types a partial value, only files whose name starts with
/// that prefix are returned. Directories that contain at least one matching
/// file are also suggested (with a trailing `/`) so the user can drill down.
pub fn datafile_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let partial = current.to_string_lossy();

    // Split into directory prefix and filename prefix.
    // e.g. "data/tes" -> dir="data/", file_prefix="tes"
    //       "tes"      -> dir=".",    file_prefix="tes"
    let (dir_str, file_prefix) = match partial.rfind('/') {
        Some(pos) => (&partial[..=pos], &partial[pos + 1..]),
        None => ("", partial.as_ref()),
    };
    let search_dir = if dir_str.is_empty() {
        std::env::current_dir().unwrap_or_default()
    } else {
        std::path::PathBuf::from(dir_str)
    };

    let entries = match std::fs::read_dir(&search_dir) {
        Ok(rd) => rd,
        Err(_) => return vec![],
    };

    let mut candidates = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Filter by the typed filename prefix.
        if !file_prefix.is_empty() && !name_str.starts_with(file_prefix) {
            continue;
        }

        let path = entry.path();
        if path.is_dir() {
            // Suggest directories so the user can navigate deeper.
            let display = format!("{}{}/", dir_str, name_str);
            candidates.push(CompletionCandidate::new(display)
                .help(Some("directory".into())));
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if DATA_EXTENSIONS.iter().any(|de| de.eq_ignore_ascii_case(ext)) {
                let display = format!("{}{}", dir_str, name_str);
                let help = format!(".{} file", ext);
                candidates.push(CompletionCandidate::new(display)
                    .help(Some(help.into())));
            }
        }
    }

    candidates.sort_by(|a, b| a.get_value().cmp(b.get_value()));
    candidates
}

/// Context-sensitive completer for `--resources` values.
///
/// Handles the `name:value,name:value,...` syntax:
///
/// - Empty or after a comma: suggests all resource type names not yet used
/// - After `name:`: shows the value format help for that resource type
/// - After `name:value,`: suggests remaining resource types
pub fn resource_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let full = current.to_string_lossy();

    // Split at the last comma to isolate the segment being typed.
    let (prefix, segment) = match full.rfind(',') {
        Some(pos) => (&full[..=pos], &full[pos + 1..]),
        None => ("", full.as_ref()),
    };

    // If the segment contains a colon, the user is typing a freeform value.
    // Return multiple comment candidates so bash displays them as a list
    // (avoiding auto-insertion) and the user sees the format help.
    if let Some((name, _partial)) = segment.split_once(':') {
        if let Some(rt) = ResourceType::from_name(name) {
            let (hint, example) = match rt.value_kind() {
                super::resource::ValueKind::Memory => (
                    "Accepts: absolute (4GiB), percent (25%), or range (1GiB-4GiB, 25%-50%)",
                    format!("{}4GiB", full),
                ),
                super::resource::ValueKind::Count => (
                    "Accepts: integer (8) or range (4-16)",
                    format!("{}8", full),
                ),
            };
            return vec![
                CompletionCandidate::new(format!("# {}: {}", rt.name(), rt.description())),
                CompletionCandidate::new(format!("# {}", hint)),
                CompletionCandidate::new(example),
            ];
        }
        return vec![];
    }

    // Collect resource names already present before the current segment.
    let used: std::collections::HashSet<&str> = full
        .split(',')
        .filter_map(|s| {
            let name = s.split(':').next().unwrap_or("");
            if !name.is_empty() && name != segment { Some(name) } else { None }
        })
        .collect();

    // Suggest resource types not yet used, filtered by the typed prefix.
    ResourceType::all()
        .iter()
        .filter(|rt| !used.contains(rt.name()))
        .filter(|rt| rt.name().starts_with(segment))
        .map(|rt| {
            CompletionCandidate::new(format!("{}{}:", prefix, rt.name()))
                .help(Some(rt.description().into()))
        })
        .collect()
}

/// Context-sensitive completer for option values that may reference pipeline
/// variables. Suggests `'${variables:name}'` candidates from `variables.yaml`
/// in the current working directory (or `--workspace` if set).
///
/// The suggestions are single-quoted so shells pass them through verbatim.
pub fn variable_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let partial = current.to_string_lossy();

    // Only trigger when the user has started typing a variable reference
    // or the value is empty (show all available variables as hints).
    let workspace = std::env::current_dir().unwrap_or_default();
    let vars = match super::variables::load(&workspace) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    if vars.is_empty() {
        return vec![];
    }

    // Only suggest variables when the user has started typing a variable
    // reference. Otherwise return empty so bash falls back to filesystem
    // completion via -o bashdefault.
    let filter = if let Some(rest) = partial.strip_prefix("'${variables:") {
        rest.trim_end_matches(['\'', '}'].as_ref())
    } else if let Some(rest) = partial.strip_prefix("${variables:") {
        rest.trim_end_matches('}')
    } else if partial.starts_with('$') || partial.starts_with("'$") {
        // Starting a variable reference but hasn't typed the full prefix yet
        ""
    } else {
        return vec![];
    };

    vars.iter()
        .filter(|(name, _)| filter.is_empty() || name.starts_with(filter))
        .map(|(name, value)| {
            CompletionCandidate::new(format!("'${{variables:{}}}'", name))
                .help(Some(format!("= {}", value).into()))
        })
        .collect()
}

/// Build the help string for an option arg (DOC-04a).
///
/// Appends type and default info to the description so shells render it
/// as a description annotation, never inserting it into the command line.
///
/// Format: `description (type: type_name[, default: default])`
fn build_option_help(description: &str, type_name: &str, default: Option<&str>) -> String {
    match default {
        Some(def) => format!("{} (type: {}, default: {})", description, type_name, def),
        None => format!("{} (type: {})", description, type_name),
    }
}

/// Execute a pipeline command directly from CLI arguments.
///
/// Parses the trailing argument list to identify the command path and its
/// options, then creates a `StreamContext` and executes the command.
///
/// Help is handled at three levels:
/// - `veks pipeline --help` / `veks pipeline` — list all pipeline commands
/// - `veks pipeline <group> --help` — list commands in that group
/// - `veks pipeline <group> <command> --help` — show command documentation
pub fn run_direct(args: Vec<String>) {
    let registry = CommandRegistry::with_builtins();

    // No args, or bare --help: show all pipeline commands grouped.
    if args.is_empty()
        || (args.len() == 1 && (args[0] == "--help" || args[0] == "-h" || args[0] == "help"))
    {
        print_pipeline_help(&registry, None);
        std::process::exit(0);
    }

    let group = &args[0];

    // Group-level help: `veks pipeline compute --help` or `veks pipeline compute`
    // when the second arg is --help/-h/help, or only one arg that is a valid group.
    let is_help_request = args.len() >= 2
        && (args[1] == "--help" || args[1] == "-h" || args[1] == "help");
    let is_bare_group = args.len() == 1
        && !group.starts_with('-')
        && registry.command_paths().iter().any(|p| p.starts_with(&format!("{} ", group)));

    if is_help_request || is_bare_group {
        let pipeline_cmd = build_pipeline_command();
        if let Some(group_cmd) = pipeline_cmd.get_subcommands().find(|c| c.get_name() == group.as_str()) {
            let mut group_cmd = group_cmd.clone();
            group_cmd.print_help().ok();
            println!();
        } else {
            print_pipeline_help(&registry, Some(group));
        }
        std::process::exit(0);
    }

    // Try single-word command first (e.g. "survey"), then "group subcommand".
    let (command_path, option_args) = if registry.get(group).is_some() {
        (group.to_string(), &args[1..])
    } else if args.len() >= 2 {
        (format!("{} {}", group, args[1]), &args[2..])
    } else {
        eprintln!("Unknown pipeline command group: '{}'", group);
        eprintln!();
        print_pipeline_help(&registry, None);
        std::process::exit(1);
    };

    let factory = match registry.get(&command_path) {
        Some(f) => f,
        None => {
            eprintln!("Unknown pipeline command: '{}'", command_path);
            eprintln!();
            print_pipeline_help(&registry, Some(group));
            std::process::exit(1);
        }
    };

    // Check for --help anywhere in args before parsing. Show the clap-generated
    // help which lists all --option flags the command accepts.
    if option_args.iter().any(|a| a == "--help" || a == "-h") {
        let pipeline_cmd = build_pipeline_command();
        // For single-word commands the options are on the group command itself;
        // for two-word commands navigate to the nested subcommand.
        let parts: Vec<&str> = command_path.splitn(2, ' ').collect();
        if let Some(group_cmd) = pipeline_cmd.get_subcommands().find(|c| c.get_name() == parts[0]) {
            if parts.len() == 1 {
                // Single-word command: group IS the command
                let mut cmd = group_cmd.clone();
                cmd.print_help().ok();
                println!();
                std::process::exit(0);
            } else if let Some(sub_cmd) = group_cmd.get_subcommands().find(|c| c.get_name() == parts[1]) {
                let mut sub_cmd = sub_cmd.clone();
                sub_cmd.print_help().ok();
                println!();
                std::process::exit(0);
            }
        }
        // Fallback to doc body if clap tree doesn't match
        let cmd = factory();
        let doc = cmd.command_doc();
        println!("{}", doc.body);
        std::process::exit(0);
    }

    // Collect resource names declared by this command for alias recognition.
    let cmd_instance = factory();
    let declared_resources = cmd_instance.describe_resources();
    let resource_names: Vec<String> = declared_resources
        .iter()
        .filter_map(|r| ResourceType::from_name(&r.name).map(|_| r.name.clone()))
        .collect();
    drop(cmd_instance);

    // Parse --key=value and --key value pairs from remaining args.
    let mut options = Options::new();
    let mut workspace: Option<PathBuf> = None;
    let mut emit_yaml = false;
    let mut step_id: Option<String> = None;
    let mut after: Vec<String> = Vec::new();
    let mut resources_str: Option<String> = None;
    let mut governor_strategy = "maximize".to_string();
    let mut resource_aliases: Vec<(String, String)> = Vec::new();
    let rest = option_args;
    let mut i = 0;
    while i < rest.len() {
        let arg = &rest[i];
        if let Some(stripped) = arg.strip_prefix("--") {
            if let Some((key, value)) = stripped.split_once('=') {
                match key {
                    "workspace" => workspace = Some(PathBuf::from(value)),
                    "emit-yaml" => emit_yaml = true,
                    "id" => step_id = Some(value.to_string()),
                    "after" => after.extend(value.split(',').map(|s| s.trim().to_string())),
                    "resources" => resources_str = Some(value.to_string()),
                    "governor" => governor_strategy = value.to_string(),
                    _ if resource_names.iter().any(|n| n == key) => {
                        resource_aliases.push((key.to_string(), value.to_string()));
                    }
                    _ => options.set(key, value),
                }
            } else if stripped == "emit-yaml" {
                emit_yaml = true;
            } else if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                let key = stripped;
                let value = &rest[i + 1];
                match key {
                    "workspace" => workspace = Some(PathBuf::from(value)),
                    "id" => step_id = Some(value.to_string()),
                    "after" => after.extend(value.split(',').map(|s| s.trim().to_string())),
                    "resources" => resources_str = Some(value.to_string()),
                    "governor" => governor_strategy = value.to_string(),
                    _ if resource_names.iter().any(|n| n == key) => {
                        resource_aliases.push((key.to_string(), value.to_string()));
                    }
                    _ => options.set(key, value),
                }
                i += 1;
            } else {
                // Boolean flag — set to "true".
                options.set(stripped, "true");
            }
        } else {
            println!("Unexpected argument: '{}'", arg);
            std::process::exit(1);
        }
        i += 1;
    }

    // Rewrite long-form resource aliases into the canonical --resources string.
    if !resource_aliases.is_empty() {
        let alias_parts: Vec<String> = resource_aliases
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect();
        let alias_str = alias_parts.join(",");
        resources_str = Some(match resources_str {
            Some(existing) => format!("{},{}", existing, alias_str),
            None => alias_str,
        });
    }

    if emit_yaml {
        emit_step_yaml(&command_path, step_id.as_deref(), &after, &options);
        return;
    }

    let workspace = workspace.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Parse resources and create governor
    let resource_budget = if let Some(ref res_str) = resources_str {
        super::resource::ResourceBudget::parse(res_str).unwrap_or_else(|e| {
            println!("Invalid --resources: {}", e);
            std::process::exit(1);
        })
    } else {
        super::resource::ResourceBudget::new()
    };
    let governor = super::resource::ResourceGovernor::new(resource_budget, Some(&workspace));

    match governor_strategy.as_str() {
        "maximize" => {} // default
        "conservative" => {
            governor.set_strategy(Box::new(super::resource::ConservativeStrategy::default()));
        }
        "fixed" => {
            governor.set_strategy(Box::new(super::resource::FixedStrategy));
        }
        other => {
            println!("Unknown governor strategy: '{}'. Use maximize, conservative, or fixed.", other);
            std::process::exit(1);
        }
    }

    let scratch = workspace.join(".scratch");
    let cache = workspace.join(".cache");
    let mut ctx = StreamContext {
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace,
        scratch,
        cache,
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads,
        step_id: String::new(),
        governor,
        ui: crate::ui::auto_ui_handle(),
        status_interval: std::time::Duration::from_secs(1),
    };

    let mut cmd = factory();

    // Start resource status monitoring
    let resource_source = ctx.governor.status_source();
    {
        let (line, metrics) = resource_source.status_line_with_metrics();
        ctx.ui.resource_status_with_metrics(line, metrics);
    }
    let resource_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let resource_stop2 = resource_stop.clone();
    let resource_ui = ctx.ui.clone();
    let poll_interval = ctx.status_interval;
    let resource_handle = std::thread::Builder::new()
        .name("resource-status".into())
        .spawn(move || {
            let mut emergency_ticks = 0u32;
            let tick_secs = poll_interval.as_secs_f64();
            while !resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(poll_interval);
                if resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                {
                    let (line, metrics) = resource_source.status_line_with_metrics();
                    resource_ui.resource_status_with_metrics(line, metrics);
                }

                if resource_source.is_emergency() {
                    emergency_ticks += 1;
                    let overage = resource_source.rss_overage_pct();
                    let base_ticks = (10.0 / tick_secs).ceil() as u32;
                    let reduction = overage.floor() as u32;
                    let grace_ticks = base_ticks.saturating_sub(reduction).max(1);
                    if emergency_ticks >= grace_ticks {
                        resource_ui.log(&format!(
                            "FATAL: resource emergency for {:.1}s, RSS {:.0}% over ceiling — aborting to prevent system lockup",
                            emergency_ticks as f64 * tick_secs,
                            overage,
                        ));
                        resource_ui.log(&format!(
                            "  last status: {}", resource_source.status_line()
                        ));
                        std::process::exit(1);
                    }
                } else {
                    emergency_ticks = 0;
                }
            }
        })
        .ok();

    let result = cmd.execute(&options, &mut ctx);

    // Stop resource status monitoring
    resource_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(h) = resource_handle {
        let _ = h.join();
    }

    // Drop ctx (including the UI handle) to restore the terminal before
    // printing any messages. Without this, output goes to the alternate
    // screen and is lost when the ratatui sink is dropped.
    drop(ctx);

    println!("[{}] {}", result.status, result.message);
    if !result.produced.is_empty() {
        println!("Produced:");
        for p in &result.produced {
            println!("  {}", p.display());
        }
    }
    println!("Elapsed: {:.2?}", result.elapsed);

    if result.status == super::command::Status::Error {
        std::process::exit(1);
    }
}

/// Emit a YAML step block suitable for inclusion in a `dataset.yaml`
/// upstream pipeline section.
///
/// Prints a `StepDef`-compatible YAML block to stdout. The `--id` and
/// `--after` meta-options control the step identity and ordering fields;
/// all other options become step option entries.
fn emit_step_yaml(
    command_path: &str,
    step_id: Option<&str>,
    after: &[String],
    options: &Options,
) {
    use super::schema::StepDef;

    let mut step_options = IndexMap::new();
    for (k, v) in &options.0 {
        step_options.insert(k.clone(), serde_yaml::Value::String(v.clone()));
    }

    let step = StepDef {
        id: step_id.map(|s| s.to_string()),
        run: command_path.to_string(),
        description: None,
        after: after.to_vec(),
        profiles: vec![],
        per_profile: false,
        on_partial: Default::default(),
        options: step_options,
    };

    let yaml = serde_yaml::to_string(&[&step]).unwrap_or_else(|e| {
        println!("Failed to serialize step: {}", e);
        std::process::exit(1);
    });
    print!("{}", yaml);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_pipeline_command_has_all_commands() {
        let cmd = build_pipeline_command();
        let registry = CommandRegistry::with_builtins();
        let paths = registry.command_paths();

        // Every registered command should be reachable as a subcommand.
        for path in &paths {
            let mut parts = path.splitn(2, ' ');
            let group = parts.next().unwrap();
            let subname = parts.next().unwrap_or("");

            let group_cmd = cmd
                .get_subcommands()
                .find(|c| c.get_name() == group);
            assert!(
                group_cmd.is_some(),
                "Missing group subcommand: '{}'",
                group
            );

            if subname.is_empty() {
                // Single-word command (e.g. "barrier", "survey", "import"):
                // the group command itself IS the command — no subcommand needed.
                continue;
            }

            let sub_cmd = group_cmd
                .unwrap()
                .get_subcommands()
                .find(|c| c.get_name() == subname);
            assert!(
                sub_cmd.is_some(),
                "Missing subcommand: '{}' under group '{}'",
                subname, group
            );
        }
    }

    #[test]
    fn test_build_pipeline_command_has_args_from_describe_options() {
        let cmd = build_pipeline_command();
        let registry = CommandRegistry::with_builtins();

        // Spot-check a few commands for their expected args.
        let factory = registry.get("analyze stats").unwrap();
        let expected_opts = factory().describe_options();

        let analyze = cmd
            .get_subcommands()
            .find(|c| c.get_name() == "analyze")
            .unwrap();
        let stats = analyze
            .get_subcommands()
            .find(|c| c.get_name() == "stats")
            .unwrap();

        for opt in &expected_opts {
            let arg = stats.get_arguments().find(|a| a.get_id() == opt.name.as_str());
            assert!(
                arg.is_some(),
                "analyze stats missing arg: '{}'",
                opt.name
            );
        }
    }

    #[test]
    fn test_emit_step_yaml_basic() {
        let mut options = Options::new();
        options.set("source", "test.fvec");
        options.set("dimension", "3");

        let mut step_options = IndexMap::new();
        for (k, v) in &options.0 {
            step_options.insert(k.clone(), serde_yaml::Value::String(v.clone()));
        }

        let step = super::super::schema::StepDef {
            id: Some("my-stats".to_string()),
            run: "analyze stats".to_string(),
            description: None,
            after: vec!["download".to_string()],
            profiles: vec![],
            per_profile: false,
            on_partial: Default::default(),
            options: step_options,
        };

        let yaml = serde_yaml::to_string(&[&step]).unwrap();

        // Verify the YAML contains the expected fields.
        assert!(yaml.contains("run: analyze stats"));
        assert!(yaml.contains("id: my-stats"));
        assert!(yaml.contains("source: test.fvec"));
        assert!(yaml.contains("dimension: '3'"));
        assert!(yaml.contains("- download"));

        // Verify it round-trips as a valid StepDef.
        let parsed: Vec<super::super::schema::StepDef> =
            serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].run, "analyze stats");
        assert_eq!(parsed[0].id.as_deref(), Some("my-stats"));
        assert_eq!(parsed[0].after, vec!["download"]);
    }

    #[test]
    fn test_emit_step_yaml_no_id_no_after() {
        let mut options = Options::new();
        options.set("source", "input.fvec");

        let mut step_options = IndexMap::new();
        for (k, v) in &options.0 {
            step_options.insert(k.clone(), serde_yaml::Value::String(v.clone()));
        }

        let step = super::super::schema::StepDef {
            id: None,
            run: "analyze stats".to_string(),
            description: None,
            after: vec![],
            profiles: vec![],
            per_profile: false,
            on_partial: Default::default(),
            options: step_options,
        };

        let yaml = serde_yaml::to_string(&[&step]).unwrap();
        assert!(yaml.contains("run: analyze stats"));
        assert!(yaml.contains("source: input.fvec"));
        // id and after should be omitted when empty.
        assert!(!yaml.contains("id:"));
        assert!(!yaml.contains("after:"));
    }

    #[test]
    fn test_completion_tree_has_emit_yaml_arg() {
        let cmd = build_pipeline_command();
        let analyze = cmd
            .get_subcommands()
            .find(|c| c.get_name() == "analyze")
            .unwrap();
        let stats = analyze
            .get_subcommands()
            .find(|c| c.get_name() == "stats")
            .unwrap();
        assert!(
            stats.get_arguments().any(|a| a.get_id() == "emit-yaml"),
            "analyze stats should have --emit-yaml arg"
        );
        assert!(
            stats.get_arguments().any(|a| a.get_id() == "id"),
            "analyze stats should have --id arg"
        );
        assert!(
            stats.get_arguments().any(|a| a.get_id() == "after"),
            "analyze stats should have --after arg"
        );
    }

    #[test]
    fn test_resource_args_per_command() {
        let cmd = build_pipeline_command();
        let registry = CommandRegistry::with_builtins();

        // Every command should have --resources and --governor args.
        for path in registry.command_paths() {
            let mut parts = path.splitn(2, ' ');
            let group = parts.next().unwrap();
            let subname = parts.next().unwrap_or("");

            let group_cmd = cmd
                .get_subcommands()
                .find(|c| c.get_name() == group)
                .unwrap();
            // Single-word commands (e.g. "barrier") are direct — the group
            // command itself carries the args.
            let sub_cmd = if subname.is_empty() {
                group_cmd
            } else {
                group_cmd
                    .get_subcommands()
                    .find(|c| c.get_name() == subname)
                    .unwrap()
            };

            assert!(
                sub_cmd.get_arguments().any(|a| a.get_id() == "resources"),
                "Command '{}' missing --resources arg",
                path,
            );
            assert!(
                sub_cmd.get_arguments().any(|a| a.get_id() == "governor"),
                "Command '{}' missing --governor arg",
                path,
            );
        }

        // Commands that declare resources should have long-form aliases
        // (except when the resource name conflicts with an existing option).
        let factory = registry.get("compute knn").unwrap();
        let cmd_instance = factory();
        let resources = cmd_instance.describe_resources();
        let option_names: std::collections::HashSet<String> = cmd_instance
            .describe_options()
            .iter()
            .map(|o| o.name.clone())
            .collect();
        assert!(!resources.is_empty(), "compute knn should declare resources");

        let compute = cmd
            .get_subcommands()
            .find(|c| c.get_name() == "compute")
            .unwrap();
        let knn = compute
            .get_subcommands()
            .find(|c| c.get_name() == "knn")
            .unwrap();

        for res in &resources {
            if option_names.contains(&res.name) {
                continue; // Resource name conflicts with option — no alias
            }
            if super::super::resource::ResourceType::from_name(&res.name).is_some() {
                let alias_id = format!("resource-{}", res.name);
                assert!(
                    knn.get_arguments().any(|a| a.get_id().as_str() == alias_id),
                    "compute knn missing resource alias --{} (id: {})",
                    res.name, alias_id,
                );
            }
        }
    }

    #[test]
    fn test_option_help_format() {
        assert_eq!(
            build_option_help("Input file path", "Path", None),
            "Input file path (type: Path)"
        );
        assert_eq!(
            build_option_help("Number of neighbors", "int", Some("100")),
            "Number of neighbors (type: int, default: 100)"
        );
    }

    #[test]
    fn test_option_args_have_type_in_help() {
        // Verify that build_pipeline_command includes type/default info in arg help text.
        let cmd = build_pipeline_command();
        let registry = CommandRegistry::with_builtins();

        let factory = registry.get("compute knn").unwrap();
        let expected_opts = factory().describe_options();

        let compute = cmd
            .get_subcommands()
            .find(|c| c.get_name() == "compute")
            .unwrap();
        let knn = compute
            .get_subcommands()
            .find(|c| c.get_name() == "knn")
            .unwrap();

        for opt in &expected_opts {
            let arg = knn.get_arguments().find(|a| a.get_id() == opt.name.as_str());
            assert!(arg.is_some(), "compute knn missing arg '{}'", opt.name);
            let help = arg.unwrap().get_help().map(|h| h.to_string()).unwrap_or_default();
            assert!(
                help.contains(&format!("type: {}", opt.type_name)),
                "compute knn arg '{}' help missing type info, got: {}",
                opt.name, help,
            );
        }
    }

    #[test]
    fn test_all_commands_have_documentation() {
        let registry = CommandRegistry::with_builtins();
        for path in registry.command_paths() {
            let factory = registry.get(&path).unwrap();
            let cmd = factory();
            let doc = cmd.command_doc();

            assert!(
                !doc.summary.is_empty(),
                "Command '{}' has empty summary",
                path,
            );
            assert!(
                !doc.body.is_empty(),
                "Command '{}' has empty body",
                path,
            );

            // Summary should not just be the command path (i.e., default impl was overridden)
            assert_ne!(
                doc.summary, path,
                "Command '{}' uses default summary (should provide a meaningful one)",
                path,
            );

            // Verify all options are mentioned in the body
            for opt in cmd.describe_options() {
                assert!(
                    doc.body.contains(&opt.name),
                    "Command '{}' doc body does not mention option '{}'",
                    path, opt.name,
                );
            }
        }
    }

    #[test]
    fn test_all_commands_reachable() {
        let registry = CommandRegistry::with_builtins();
        let count = registry.command_paths().len();
        assert!(
            count >= 50,
            "Expected at least 50 registered commands, found {}",
            count
        );

        let cmd = build_pipeline_command();
        let mut reachable = 0;
        for group_cmd in cmd.get_subcommands() {
            let sub_count = group_cmd.get_subcommands().count();
            if sub_count == 0 {
                // Direct (single-word) command — the group itself is the command
                reachable += 1;
            } else {
                reachable += sub_count;
            }
        }
        assert_eq!(
            reachable, count,
            "Subcommand count ({}) != registry count ({})",
            reachable, count
        );
    }
}
