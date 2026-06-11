// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Direct CLI invocation of pipeline commands.
//!
//! Builds a clap `Command` tree from the `CommandRegistry` so that pipeline
//! commands can be invoked directly (with shell tab-completion) instead of
//! only through YAML pipeline files:
//!
//! ```text
//! veks pipeline analyze stats --source=test.fvec
//! veks pipeline generate predicates --output=predicates.u8 --count=10000
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

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
/// Collect each pipeline command's `value_completions()` into a flat
/// map keyed by full command path (e.g., `"verify engine-parity"`).
/// Consumed by the shell-completion engine to attach per-leaf value
/// providers without round-tripping through clap's Arg metadata.
pub fn pipeline_value_completions() -> BTreeMap<String, std::collections::HashMap<String, super::command::ValueCompletions>> {
    let registry = CommandRegistry::with_builtins();
    let mut out = BTreeMap::new();
    for path in registry.command_paths() {
        let factory = registry.get(path).unwrap();
        let cmd = factory();
        let vc = cmd.value_completions();
        if !vc.is_empty() {
            out.insert(path.to_string(), vc);
        }
    }
    out
}

/// Metadata payload returned by [`pipeline_command_metadata`].
pub struct CommandMetadata {
    pub category: &'static dyn veks_completion::CategoryTag,
    pub level:    &'static dyn veks_completion::LevelTag,
}

/// Collect every pipeline command's discovery metadata
/// ([`super::command::CommandOp::category`] and `level`) into a flat
/// map keyed by the full command path. Consumed by the
/// shell-completion engine to populate `Node::Leaf::category` /
/// `level` and feed the stratified-completion tab cycle.
pub fn pipeline_command_metadata() -> BTreeMap<String, CommandMetadata> {
    let registry = CommandRegistry::with_builtins();
    let mut out = BTreeMap::new();
    for path in registry.command_paths() {
        let factory = registry.get(path).unwrap();
        let cmd = factory();
        out.insert(path.to_string(), CommandMetadata {
            category: cmd.category(),
            level:    cmd.level(),
        });
    }
    out
}

/// The full pipeline command tree as a [`veks_completion::CommandSpec`] — the
/// clap-free replacement for [`build_pipeline_command`]. Commands are grouped by
/// first word; a single-word command (empty subname) becomes the group node
/// itself (matching the clap builder's `is_direct` behavior).
pub fn pipeline_command_spec() -> veks_completion::CommandSpec {
    use veks_completion::CommandSpec;

    let registry = CommandRegistry::with_builtins();
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for path in registry.command_paths() {
        let mut parts = path.splitn(2, ' ');
        let group = parts.next().unwrap().to_string();
        let subname = parts.next().unwrap_or("").to_string();
        groups.entry(group).or_default().push((subname, path.to_string()));
    }

    let mut pipeline = CommandSpec::new("pipeline").about("Execute a single pipeline command directly");
    for (group, cmds) in groups {
        let mut group_spec = CommandSpec::new(&group).about(format!("{} commands", group));
        group_spec.subcommand_required = true;
        let mut direct: Option<CommandSpec> = None;
        for (subname, path) in cmds {
            let factory = registry.get(&path).unwrap();
            let cmd = factory();
            if subname.is_empty() {
                // Single-word command: the group node *is* the command.
                direct = Some(command_op_to_spec(&group, &*cmd));
            } else {
                group_spec = group_spec.subcommand(command_op_to_spec(&subname, &*cmd));
            }
        }
        pipeline = pipeline.subcommand(direct.unwrap_or(group_spec));
    }
    pipeline
}

/// Map a single pipeline command's options — the common ones plus its
/// `describe_options()` — into a [`veks_completion::CommandSpec`]. Fixes the
/// command's *shape* (flags, value options, required-ness, defaults, help) and
/// attaches value completers (path → file completion, enum → declared value set,
/// governor → its strategy set).
pub fn command_op_to_spec(name: &str, cmd: &dyn super::command::CommandOp) -> veks_completion::CommandSpec {
    use veks_completion::{CommandSpec, OptionDef, OptionSpec};

    let doc = cmd.command_doc();
    let value_completions = cmd.value_completions();
    let mut spec = CommandSpec::new(name)
        .about(doc.summary.clone())
        .stability(cmd.stability());

    // Common args injected onto every pipeline command (mirrors
    // build_pipeline_command's emit-yaml / id / after).
    spec = spec
        .option(OptionSpec::new(OptionDef::flag("--emit-yaml")
            .help("Emit a YAML step block for dataset.yaml instead of executing")))
        .option(OptionSpec::new(OptionDef::value("--id")
            .help("Step identifier for the YAML block")))
        .option(OptionSpec::new(OptionDef::value("--after")
            .help("Comma-separated step IDs this step depends on")));

    // The command's declared options.
    for opt in cmd.describe_options() {
        spec = spec.option(optiondesc_to_spec(&opt, value_completions.get(&opt.name)));
    }

    // Trailing common args: --resources and --governor (default "maximize",
    // completes to its strategy set).
    let governor_vc = super::command::ValueCompletions::enum_values(&["maximize", "conservative", "fixed"]);
    spec = spec
        .option(OptionSpec::new(OptionDef::value("--resources").help("Resource limits")))
        .option(OptionSpec::new(OptionDef::value("--governor")
            .help("Governor strategy: maximize, conservative, fixed"))
            .default("maximize")
            .value_completion(enum_provider(governor_vc)));

    spec
}

/// Map one `OptionDesc` to an `OptionSpec`: a `bool`/`flag` becomes a real
/// flag; everything else a value option carrying required-ness, default, help
/// (`build_option_help` reproduces the type/default suffix clap showed), and a
/// value completer (`Path` → file paths, enum → its declared `value_completions`).
fn optiondesc_to_spec(
    opt: &super::command::OptionDesc,
    vc: Option<&super::command::ValueCompletions>,
) -> veks_completion::OptionSpec {
    use veks_completion::{OptionDef, OptionSpec};
    let dashed = format!("--{}", opt.name);
    let help = build_option_help(&opt.description, &opt.type_name, opt.default.as_deref());
    if matches!(opt.type_name.as_str(), "bool" | "flag") {
        OptionSpec::new(OptionDef::flag(dashed).help(help))
    } else {
        let mut spec = OptionSpec::new(OptionDef::value(dashed).help(help)).required(opt.required);
        if let Some(d) = &opt.default {
            spec = spec.default(d.clone());
        }
        if opt.type_name == "Path" {
            spec = spec.value_completion(veks_completion::providers::fs_paths_provider());
        } else if let Some(vc) = vc {
            spec = spec.value_completion(enum_provider(vc.clone()));
        }
        spec
    }
}

/// Wrap a [`ValueCompletions`](super::command::ValueCompletions) closed set as a
/// veks [`ValueProvider`](veks_completion::ValueProvider).
fn enum_provider(vc: super::command::ValueCompletions) -> veks_completion::ValueProvider {
    std::sync::Arc::new(move |partial: &str, _ctx: &[&str]| enum_complete(partial, &vc))
}

/// Plain-`String` closed-set completion (the veks-native counterpart of the
/// clap_complete [`enum_value_completer`]). Handles comma-separated lists by
/// dropping already-chosen values.
fn enum_complete(partial: &str, vc: &super::command::ValueCompletions) -> Vec<String> {
    if !vc.comma_separated {
        let mut out: Vec<String> = vc
            .values
            .iter()
            .filter(|v| partial.is_empty() || v.starts_with(partial))
            .cloned()
            .collect();
        out.sort();
        return out;
    }
    let (already, rest) = match partial.rfind(',') {
        Some(i) => (&partial[..=i], &partial[i + 1..]),
        None => ("", partial),
    };
    let chosen: std::collections::HashSet<&str> =
        already.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()).collect();
    let mut out: Vec<String> = vc
        .values
        .iter()
        .filter(|v| !chosen.contains(v.as_str()))
        .filter(|v| rest.is_empty() || v.starts_with(rest))
        .map(|v| format!("{}{}", already, v))
        .collect();
    out.sort();
    out
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
            println!();
            println!("Tip: Distinct commands can be invoked directly, e.g., 'veks bootstrap' instead");
            println!("of 'veks prepare bootstrap'. Tap Tab twice to see the expanded command list.");
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
        let spec = pipeline_command_spec();
        if let Some(group_spec) = spec.subcommands.iter().find(|c| c.name == group.as_str()) {
            print!("{}", veks_completion::cli::render_help(group_spec));
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

    // Check for --help anywhere in args before parsing. Render the command's
    // spec help (lists all --option flags it accepts).
    if option_args.iter().any(|a| a == "--help" || a == "-h") {
        let spec = pipeline_command_spec();
        // For single-word commands the options are on the group node itself;
        // for two-word commands navigate to the nested subcommand.
        let parts: Vec<&str> = command_path.splitn(2, ' ').collect();
        if let Some(group_spec) = spec.subcommands.iter().find(|c| c.name == parts[0]) {
            let target = if parts.len() == 1 {
                Some(group_spec)
            } else {
                group_spec.subcommands.iter().find(|c| c.name == parts[1])
            };
            if let Some(cmd_spec) = target {
                print!("{}", veks_completion::cli::render_help(cmd_spec));
                std::process::exit(0);
            }
        }
        // Fallback to doc body if the spec tree doesn't match.
        let cmd = factory();
        println!("{}", cmd.command_doc().body);
        std::process::exit(0);
    }

    // Collect resource names declared by this command for alias recognition,
    // and the ordered list of required Input-role option names so we can
    // accept bare positional args as shorthand for them (in declaration order).
    let cmd_instance = factory();
    let declared_resources = cmd_instance.describe_resources();
    let resource_names: Vec<String> = declared_resources
        .iter()
        .filter_map(|r| ResourceType::from_name(&r.name).map(|_| r.name.clone()))
        .collect();
    let positional_targets: Vec<String> = cmd_instance
        .describe_options()
        .into_iter()
        .filter(|d| d.required && d.role == super::command::OptionRole::Input)
        .map(|d| d.name)
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
    let mut next_positional = 0usize;
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
        } else if next_positional < positional_targets.len() {
            // Bare positional: assign to the next unfilled required-Input option.
            // An explicit --key=value earlier still wins because Options::set
            // overwrites; we only fill slots that the user hasn't already named.
            let target = &positional_targets[next_positional];
            if options.get(target).is_none() {
                options.set(target, arg);
            }
            next_positional += 1;
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

    let workspace = workspace.unwrap_or_else(|| PathBuf::from("."));

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

    let cache = workspace.join(".cache");
    let mut ctx = StreamContext {
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace,
        cache,
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads,
        step_id: String::new(),
        governor,
        ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::PlainSink::new())),
        status_interval: std::time::Duration::from_secs(1),
        estimated_total_steps: 0,
        provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
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
                        // Build messages once, log them to the TUI
                        // (so they end up in runlog.jsonl), then shut
                        // down the TUI and re-emit on stderr so the
                        // user actually sees them at the shell prompt.
                        // Without the stderr emission, the alt-screen
                        // swallows the log and the user gets a silent
                        // exit 1.
                        let fatal = format!(
                            "FATAL: resource emergency for {:.1}s, RSS {:.0}% over ceiling — aborting to prevent system lockup",
                            emergency_ticks as f64 * tick_secs,
                            overage,
                        );
                        let status = format!(
                            "  last status: {}", resource_source.status_line(),
                        );
                        resource_ui.log(&fatal);
                        resource_ui.log(&status);
                        resource_ui.shutdown();
                        eprintln!();
                        eprintln!("{}", fatal);
                        eprintln!("{}", status);
                        eprintln!("Hint: lower the in-flight count, narrow the workload, or pass `--mem <smaller>` and re-run.");
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

    if result.status == super::command::Status::Error {
        eprintln!("ERROR: {}", result.message);
    } else {
        println!("{}", result.message);
    }
    if !result.produced.is_empty() {
        println!("Produced:");
        for p in &result.produced {
            println!("  {}", p.display());
        }
    }
    // Only show elapsed when it's non-trivial (>100ms)
    if result.elapsed.as_millis() >= 100 {
        println!("Elapsed: {:.2?}", result.elapsed);
    }

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
        phase: 0,
        finalize: false,
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
    fn test_pipeline_command_spec_has_all_commands() {
        let spec = pipeline_command_spec();
        let registry = CommandRegistry::with_builtins();

        // Every registered command should be reachable in the spec.
        for path in registry.command_paths() {
            let mut parts = path.splitn(2, ' ');
            let group = parts.next().unwrap();
            let subname = parts.next().unwrap_or("");

            let group_spec = spec.subcommands.iter().find(|c| c.name == group);
            assert!(group_spec.is_some(), "Missing group: '{}'", group);

            if subname.is_empty() {
                // Single-word command: the group node IS the command.
                continue;
            }
            let sub = group_spec.unwrap().subcommands.iter().find(|c| c.name == subname);
            assert!(sub.is_some(), "Missing subcommand '{}' under group '{}'", subname, group);
        }
    }

    #[test]
    fn test_pipeline_spec_has_args_from_describe_options() {
        let spec = pipeline_command_spec();
        let registry = CommandRegistry::with_builtins();
        let expected_opts = registry.get("analyze stats").unwrap()().describe_options();

        let analyze = spec.subcommands.iter().find(|c| c.name == "analyze").unwrap();
        let stats = analyze.subcommands.iter().find(|c| c.name == "stats").unwrap();

        for opt in &expected_opts {
            let flag = format!("--{}", opt.name);
            assert!(
                stats.options.iter().any(|o| o.flag() == flag),
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
            phase: 0,
            finalize: false,
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
            phase: 0,
            finalize: false,
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
    fn test_all_commands_have_documentation() {
        let registry = CommandRegistry::with_builtins();
        for path in registry.command_paths() {
            let factory = registry.get(path).unwrap();
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
    fn command_op_to_spec_maps_common_and_declared_options() {
        let registry = CommandRegistry::with_builtins();
        let factory = registry.get("compute knn").expect("compute knn registered");
        let cmd = factory();
        let spec = command_op_to_spec("knn", &*cmd);

        let names: Vec<&str> = spec.options.iter().map(|o| o.flag()).collect();
        assert!(names.contains(&"--emit-yaml"), "common --emit-yaml present");
        assert!(names.contains(&"--governor"), "common --governor present");

        // emit-yaml is a flag (no value); governor is a value option defaulting
        // to "maximize".
        let emit = spec.options.iter().find(|o| o.flag() == "--emit-yaml").unwrap();
        assert!(!emit.def.takes_value, "--emit-yaml must be a flag");
        let gov = spec.options.iter().find(|o| o.flag() == "--governor").unwrap();
        assert!(gov.def.takes_value, "--governor takes a value");
        assert_eq!(gov.default.as_deref(), Some("maximize"));

        // The command's own declared options are mapped too.
        assert!(spec.options.len() > 5, "declared options mapped alongside common ones");
    }

    #[test]
    fn runargs_veks_cli_spec_parse_and_extract() {
        use veks_completion::VeksCli;
        let spec = crate::pipeline::RunArgs::veks_command_spec("run");

        let names: Vec<&str> = spec.options.iter().map(|o| o.flag()).collect();
        assert!(names.contains(&"--recursive"));
        assert!(names.contains(&"--profile"));
        assert!(names.contains(&"--set")); // overrides field, renamed via #[arg(long = "set")]
        assert_eq!(spec.positionals.len(), 1, "dataset is a positional");
        let profile = spec.options.iter().find(|o| o.flag() == "--profile").unwrap();
        assert_eq!(profile.default.as_deref(), Some("all"));

        // Parse + typed extraction of a real invocation.
        let argv: Vec<String> = ["--recursive", "--threads", "8", "mydata.yaml"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let parsed = veks_completion::cli::parse(&spec, &argv).unwrap();
        let run = crate::pipeline::RunArgs::veks_from_parsed(&parsed).unwrap();
        assert!(run.recursive);
        assert_eq!(run.threads, 8);
        assert_eq!(run.dataset, Some(std::path::PathBuf::from("mydata.yaml")));
        assert_eq!(run.profile, "all", "default applied");
        assert_eq!(run.governor, "maximize", "default applied");
    }

    #[test]
    fn pipeline_command_spec_builds_tree_with_completers() {
        let spec = pipeline_command_spec();

        // Group → command structure, mirroring build_pipeline_command.
        let compute = spec.subcommands.iter().find(|c| c.name == "compute").expect("compute group");
        assert!(compute.subcommand_required);
        let knn = compute.subcommands.iter().find(|c| c.name == "knn").expect("compute knn");
        assert!(knn.options.iter().any(|o| o.flag() == "--governor"));

        // Completion via the bridge: --governor completes to its strategy set,
        // carried on the option's own value_completion.
        let resolvers = std::collections::BTreeMap::new();
        let tree = veks_completion::cli::build_completion_tree(&spec, &resolvers);
        let out = veks_completion::complete(&tree, &["veks", "compute", "knn", "--governor", ""]);
        assert_eq!(
            out,
            vec!["conservative".to_string(), "fixed".to_string(), "maximize".to_string()]
        );
    }
}
