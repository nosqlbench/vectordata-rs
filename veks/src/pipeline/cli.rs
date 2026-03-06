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
use clap_complete::engine::{ArgValueCandidates, ArgValueCompleter, CompletionCandidate};
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
        .arg_required_else_help(true);

    for (group, commands) in groups {
        let mut group_cmd = Command::new(group.clone())
            .about(format!("{} commands", group))
            .subcommand_required(true)
            .arg_required_else_help(true);

        for (subname, opts, doc, resources) in commands {
            let mut sub_cmd = Command::new(subname)
                .about(doc.summary.clone())
                .long_about(doc.body.clone())
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
                }

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
                    .add(ArgValueCandidates::new(governor_candidates)),
            );

            group_cmd = group_cmd.subcommand(sub_cmd);
        }

        pipeline_cmd = pipeline_cmd.subcommand(group_cmd);
    }

    pipeline_cmd
}

/// Completion candidates for `--governor`: lists strategy names with descriptions.
pub fn governor_candidates() -> Vec<CompletionCandidate> {
    vec![
        CompletionCandidate::new("maximize")
            .help(Some("Aggressively use resources up to ceiling".into())),
        CompletionCandidate::new("conservative")
            .help(Some("Start at floor, increase after sustained low use".into())),
        CompletionCandidate::new("fixed")
            .help(Some("Use midpoint values, never adjust".into())),
    ]
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
pub fn run_direct(args: Vec<String>) {
    if args.len() < 2 {
        eprintln!("Usage: veks pipeline <group> <command> [--option=value ...]");
        std::process::exit(1);
    }

    let group = &args[0];
    let subcommand = &args[1];
    let command_path = format!("{} {}", group, subcommand);

    let registry = CommandRegistry::with_builtins();

    let factory = match registry.get(&command_path) {
        Some(f) => f,
        None => {
            eprintln!("Unknown pipeline command: '{}'", command_path);
            eprintln!("Available commands:");
            for path in registry.command_paths() {
                eprintln!("  {}", path);
            }
            std::process::exit(1);
        }
    };

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
    let rest = &args[2..];
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
            eprintln!("Unexpected argument: '{}'", arg);
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
            eprintln!("Invalid --resources: {}", e);
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
            eprintln!("Unknown governor strategy: '{}'. Use maximize, conservative, or fixed.", other);
            std::process::exit(1);
        }
    }

    let scratch = workspace.join(".scratch");
    let cache = workspace.join(".cache");
    let display = super::display::ProgressDisplay::new();
    let mut ctx = StreamContext {
        workspace,
        scratch,
        cache,
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads,
        step_id: String::new(),
        governor,
        display,
    };

    let mut cmd = factory();

    // Start resource status bar
    let resource_bar = ctx.display.resource_bar();
    let resource_source = ctx.governor.status_source();
    resource_bar.set_message(resource_source.status_line());
    let resource_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let resource_stop2 = resource_stop.clone();
    let resource_handle = std::thread::Builder::new()
        .name("resource-status".into())
        .spawn(move || {
            while !resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(500));
                if resource_stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                resource_bar.set_message(resource_source.status_line());
            }
            resource_bar.finish_and_clear();
        })
        .ok();

    let result = cmd.execute(&options, &mut ctx);

    // Stop resource status bar
    resource_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(h) = resource_handle {
        let _ = h.join();
    }

    eprintln!("[{}] {}", result.status, result.message);
    if !result.produced.is_empty() {
        eprintln!("Produced:");
        for p in &result.produced {
            eprintln!("  {}", p.display());
        }
    }
    eprintln!("Elapsed: {:.2?}", result.elapsed);

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
        on_partial: Default::default(),
        options: step_options,
    };

    let yaml = serde_yaml::to_string(&[&step]).unwrap_or_else(|e| {
        eprintln!("Failed to serialize step: {}", e);
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
            let sub_cmd = group_cmd
                .get_subcommands()
                .find(|c| c.get_name() == subname)
                .unwrap();

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
            reachable += group_cmd.get_subcommands().count();
        }
        assert_eq!(
            reachable, count,
            "Subcommand count ({}) != registry count ({})",
            reachable, count
        );
    }
}
