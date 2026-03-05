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

use clap::{Arg, Command, ValueHint};
use indexmap::IndexMap;

use super::command::{Options, StreamContext};
use super::progress::ProgressLog;
use super::registry::CommandRegistry;

/// Build a `clap::Command` representing all registered pipeline commands.
///
/// Commands are grouped by their first word (e.g. `"analyze"`, `"generate"`)
/// and each group becomes a subcommand containing its individual commands as
/// nested subcommands with `Arg` entries derived from `describe_options()`.
pub fn build_pipeline_command() -> Command {
    let registry = CommandRegistry::with_builtins();
    let paths = registry.command_paths();

    // Group command paths by first word.
    let mut groups: BTreeMap<String, Vec<(String, Vec<super::command::OptionDesc>)>> =
        BTreeMap::new();

    for path in &paths {
        let mut parts = path.splitn(2, ' ');
        let group = parts.next().unwrap().to_string();
        let subname = parts.next().unwrap_or("").to_string();

        // Get option descriptions from a fresh command instance.
        let factory = registry.get(path).unwrap();
        let cmd = factory();
        let opts = cmd.describe_options();

        groups.entry(group).or_default().push((subname, opts));
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

        for (subname, opts) in commands {
            let about = format!("{} {}", group, subname);
            let mut sub_cmd = Command::new(subname)
                .about(about)
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

            for opt in opts {
                let mut arg = Arg::new(opt.name.clone())
                    .long(opt.name.clone())
                    .required(opt.required)
                    .help(opt.description);

                if let Some(default) = opt.default {
                    arg = arg.default_value(default);
                }

                if opt.type_name == "Path" {
                    arg = arg.value_hint(ValueHint::AnyPath);
                }

                sub_cmd = sub_cmd.arg(arg);
            }

            group_cmd = group_cmd.subcommand(sub_cmd);
        }

        pipeline_cmd = pipeline_cmd.subcommand(group_cmd);
    }

    pipeline_cmd
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

    // Parse --key=value and --key value pairs from remaining args.
    let mut options = Options::new();
    let mut workspace: Option<PathBuf> = None;
    let mut emit_yaml = false;
    let mut step_id: Option<String> = None;
    let mut after: Vec<String> = Vec::new();
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

    if emit_yaml {
        emit_step_yaml(&command_path, step_id.as_deref(), &after, &options);
        return;
    }

    let workspace = workspace.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let scratch = workspace.join(".scratch");
    let cache = workspace.join(".cache");
    let mut ctx = StreamContext {
        workspace,
        scratch,
        cache,
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads,
        step_id: String::new(),
    };

    let mut cmd = factory();
    let result = cmd.execute(&options, &mut ctx);

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
