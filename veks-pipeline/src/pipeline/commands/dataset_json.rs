// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate `dataset.json` from `dataset.yaml`.
//!
//! Writes a JSON-format copy of the dataset configuration alongside
//! the YAML source. This provides a JSON-accessible copy for clients
//! that prefer JSON over YAML. The file is only written when missing
//! or when `dataset.yaml` is newer.

use std::time::Instant;

use vectordata::dataset::DatasetConfig;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, Options,
    Status, StreamContext, render_options_table,
};

pub struct DatasetJsonOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(DatasetJsonOp)
}

impl CommandOp for DatasetJsonOp {
    fn command_path(&self) -> &str {
        "generate dataset-json"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_GENERATE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate dataset.json from dataset.yaml".into(),
            body: format!(
                "# generate dataset-json\n\n\
                 Generate a JSON copy of the dataset configuration.\n\n\
                 ## Description\n\n\
                 Reads `dataset.yaml` from the workspace and writes `dataset.json` \
                 alongside it. The JSON file is only written when missing or when \
                 `dataset.yaml` has a newer modification time. This provides a \
                 JSON-accessible copy for clients that prefer JSON over YAML.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, _options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let yaml_path = ctx.workspace.join("dataset.yaml");
        let json_path = ctx.workspace.join("dataset.json");

        if !yaml_path.exists() {
            return CommandResult {
                status: Status::Error,
                message: format!("dataset.yaml not found in {}", ctx.workspace.display()),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let needs_update = if !json_path.exists() {
            true
        } else {
            match (std::fs::metadata(&yaml_path), std::fs::metadata(&json_path)) {
                (Ok(ym), Ok(jm)) => ym.modified().ok() > jm.modified().ok(),
                _ => true,
            }
        };

        if !needs_update {
            return CommandResult {
                status: Status::Ok,
                message: "dataset.json is up to date".into(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let mut config = match DatasetConfig::load(&yaml_path) {
            Ok(c) => c,
            Err(e) => return CommandResult {
                status: Status::Error,
                message: format!("failed to load dataset.yaml: {}", e),
                produced: vec![],
                elapsed: start.elapsed(),
            },
        };

        // Materialize deferred sized profiles before re-emitting, so the
        // published dataset.yaml/json carries every concrete profile that
        // a consumer would see — not the compact `sized: [...]` spec
        // that requires the veks parser to interpret. Use the same
        // variable-fallthrough as the runtime so this works whether
        // pipeline state is in `vector_count`, `base_count`, etc.
        if config.profiles.has_deferred() {
            let mut vars: indexmap::IndexMap<String, String> = config.variables.clone();
            if let Ok(file_vars) = crate::pipeline::variables::load(&ctx.workspace) {
                for (k, v) in file_vars {
                    vars.insert(k, v);
                }
            }
            let base_count: u64 = ["base_count", "clean_count", "vector_count", "source_base_count"]
                .iter()
                .find_map(|k| vars.get(*k).and_then(|v| v.parse().ok()).filter(|&n: &u64| n > 0))
                .unwrap_or(0);
            if base_count > 0 {
                let added = config.profiles.expand_deferred_sized(&vars, base_count);
                if added > 0 {
                    ctx.ui.log(&format!(
                        "  expanded {} sized profile(s) for publish", added));
                }
            }
        }

        // Derive per-profile output views (neighbor_indices,
        // neighbor_distances, etc.) from the per_profile templates so
        // each expanded sized profile carries the concrete paths
        // clients need: `profiles/{name}/neighbor_indices.ivecs` and
        // friends. Without this, sized profiles serialize with only
        // `maxk` + `base_count` because the views they inherit from
        // default get suppressed (default's paths match default's
        // own paths), and their per-profile output views don't
        // exist yet to be surfaced.
        let templates: Vec<_> = vectordata::dataset::collect_all_steps(&config)
            .into_iter()
            .filter(|s| s.per_profile)
            .collect();
        config.profiles.derive_views_from_templates(&templates);

        // NOTE: the expanded yaml itself is persisted by
        // `update_dataset_attributes` at the very end of every
        // `veks run` — it syncs variables + attributes from
        // `variables.yaml` then calls `save_expanded`. Doing that
        // write here too would be redundant and cause run-2
        // freshness failures: this step's `resolved_options`
        // snapshot would record yaml.size as S1, but the subsequent
        // `update_dataset_attributes` would bump it to S2 (with the
        // synced variables), so run 2's freshness check sees
        // `output 'dataset.yaml' size changed (S1 → S2)` and
        // pointlessly re-runs every finalize step. Single writer
        // for dataset.yaml → finalize pass is idempotent.

        let json = match serde_json::to_string_pretty(&config) {
            Ok(j) => j,
            Err(e) => return CommandResult {
                status: Status::Error,
                message: format!("JSON serialization failed: {}", e),
                produced: vec![],
                elapsed: start.elapsed(),
            },
        };

        if let Err(e) = std::fs::write(&json_path, &json) {
            return CommandResult {
                status: Status::Error,
                message: format!("failed to write {}: {}", json_path.display(), e),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        ctx.ui.log(&format!("  wrote {}", json_path.display()));

        CommandResult {
            status: Status::Ok,
            message: "wrote dataset.json".into(),
            produced: vec![json_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![]
    }

    fn project_artifacts(&self, step_id: &str, _options: &Options) -> ArtifactManifest {
        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs: vec![],
            outputs: vec!["dataset.json".to_string()],
            intermediates: vec![],
        }
    }
}
