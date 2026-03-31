// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: set a variable in `variables.yaml`.
//!
//! Evaluates an expression and persists the result to `variables.yaml` next to
//! the dataset file. The variable is also injected into the pipeline defaults
//! so that downstream steps in the same run can reference it via `${name}`.
//!
//! Expressions:
//! - `count:<path>` — record count of a vector file (fvec/mvec/ivec) or slab
//! - `dim:<path>` — dimension of a vector file
//! - `<literal>` — stored as-is

use std::time::Instant;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use crate::pipeline::variables;

/// Pipeline command: set a variable from an expression.
pub struct SetVariableOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(SetVariableOp)
}

impl CommandOp for SetVariableOp {
    fn command_path(&self) -> &str {
        "state set"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Set a pipeline variable from an expression".into(),
            body: format!(
                r#"# state set

Set a pipeline variable from an expression.

## Description

Evaluates the `value` expression and stores the result under `name` in
`variables.yaml` next to the dataset file. The variable becomes available
to all subsequent steps via `${{name}}` interpolation.

## Expressions

- `count:<path>` — record count of a vector file (fvec/mvec/ivec) or slab
- `dim:<path>` — dimension of a vector file (fvec/mvec/ivec)
- Any other value is stored as a literal string.

## Options

{}

## Data Preparation Role

Variables are the mechanism for passing computed values between pipeline
steps. Because pipeline steps can run in any order (subject to dependency
constraints) and may be skipped if their outputs are already up to date,
direct data flow between steps is not possible. Instead, a step like
`set variable` evaluates an expression (typically `count:<path>` to
determine how many records a file contains), persists the result to
`variables.yaml`, and injects it into the pipeline defaults map. All
downstream steps can then reference the variable via `${{name}}`
interpolation in their option values. This pattern is used extensively to
pass dynamic values like vector counts, dimension sizes, and computed
thresholds through the pipeline without hard-coding them.

## Examples

```yaml
- id: set-vector-count
  run: set variable
  after:
    - import-all
  name: vector_count
  value: "count:all_vectors.mvec"
```

Then reference in a downstream step:

```yaml
- id: generate-shuffle
  run: generate ivec-shuffle
  interval: "${{vector_count}}"
```
"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let name = match options.require("name") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let expr = match options.require("value") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        // Evaluate expression
        let value = match variables::evaluate_expr(expr, &ctx.workspace) {
            Ok(v) => v,
            Err(e) => return error_result(
                format!("failed to evaluate '{}': {}", expr, e),
                start,
            ),
        };

        // Persist to variables.yaml
        if let Err(e) = variables::set_and_save(&ctx.workspace, name, &value) {
            return error_result(
                format!("failed to save variable '{}': {}", name, e),
                start,
            );
        }

        // Inject into current pipeline defaults so downstream steps see it
        ctx.defaults.insert(name.to_string(), value.clone());

        ctx.ui.log(&format!("  {} = {}", name, value));

        CommandResult {
            status: Status::Ok,
            message: format!("set {} = {}", name, value),
            // Don't list variables.yaml as a produced artifact — it's a shared
            // append-only resource modified by multiple set-variable steps.
            // Tracking its size would cause false staleness when a later step
            // adds another variable.
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "name".to_string(),
                type_name: "String".to_string(),
                required: true,
                default: None,
                description: "Variable name (used as ${name} in downstream steps)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "value".to_string(),
                type_name: "String".to_string(),
                required: true,
                default: None,
                description: "Expression: count:<path>, dim:<path>, or a literal value".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, _options: &Options) -> ArtifactManifest {
        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs: vec![],
            outputs: vec![],
            intermediates: vec![],
        }
    }
}

/// Pipeline command: clear all variables in `variables.yaml`.
pub struct ClearVariablesOp;

pub fn clear_factory() -> Box<dyn CommandOp> {
    Box::new(ClearVariablesOp)
}

impl CommandOp for ClearVariablesOp {
    fn command_path(&self) -> &str {
        "state clear"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Clear all pipeline variables (reset variables.yaml)".into(),
            body: format!(
                r#"# state clear

Clear all pipeline variables, resetting `variables.yaml`.

## Description

Removes the `variables.yaml` file next to the dataset file, clearing all
stored pipeline variables. This acts as a context reset step — downstream
steps that depend on variables will be forced to re-evaluate them.

Because this command produces `variables.yaml` as an output artifact,
cascade invalidation ensures all downstream steps that were skipped based
on stale variable values will re-execute.

## Options

{}

## Data Preparation Role

Variables are the mechanism for passing computed values between pipeline
steps. When a pipeline is re-run after changes to source data, stale
variable values from the previous run can cause downstream steps to use
incorrect parameters (for example, an outdated vector count). Placing
`clear variables` at the beginning of the pipeline ensures a clean slate,
forcing all `set variable` steps to re-evaluate their expressions. The
cascade invalidation ensures that any step depending on a cleared variable
will re-execute rather than using cached results from a prior run.

## Examples

Place as the first step to ensure a clean variable context:

```yaml
- id: reset-vars
  run: clear variables

- id: set-vector-count
  run: set variable
  after:
    - import-all
  name: vector_count
  value: "count:all_vectors.mvec"
```

CLI equivalent:

```bash
veks pipeline clear variables --workspace=/path/to/dataset
```
"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, _options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let vars_path = variables::variables_path(&ctx.workspace);

        // Load existing variables to know what to remove from defaults
        let existing = variables::load(&ctx.workspace).unwrap_or_default();

        // Remove the file
        if vars_path.exists() {
            if let Err(e) = std::fs::remove_file(&vars_path) {
                return error_result(
                    format!("failed to remove {}: {}", vars_path.display(), e),
                    start,
                );
            }
        }

        // Remove any variables.yaml keys from the current defaults
        let removed = existing.len();
        for key in existing.keys() {
            ctx.defaults.swap_remove(key);
        }

        ctx.ui.log(&format!("  cleared {} variable(s)", removed));

        CommandResult {
            status: Status::Ok,
            message: format!("cleared {} variable(s)", removed),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![]
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}
