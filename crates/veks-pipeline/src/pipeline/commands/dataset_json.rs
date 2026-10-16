// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate `dataset.json` from `dataset.yaml`.
//!
//! Writes a JSON-format copy of the dataset configuration alongside
//! the YAML source. This provides a JSON-accessible copy for clients
//! that prefer JSON over YAML. The file is rewritten whenever what
//! this build renders differs from it, so a fix to the rendering
//! reaches a dataset whose `dataset.yaml` has not changed.

use std::time::Instant;

use vectordata::dataset::DatasetConfig;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, Options,
    Status, StreamContext, render_options_table,
};

/// Pipeline command: write `dataset.json` as a JSON copy of `dataset.yaml` when it differs from what is there.
pub struct DatasetJsonOp;

/// Creates a boxed [`DatasetJsonOp`]; registered as the `generate dataset-json` command.
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
                 alongside it. The JSON file is rendered every time and written when \
                 it differs from the file there, so a new veks rewrites a \
                 `dataset.json` an older one rendered. This provides a \
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
            let added = config.expand_sized_profiles(&vars);
            if added > 0 {
                ctx.ui.log(&format!(
                    "  expanded {} sized profile(s) for publish", added));
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

        // Compared by content, not by modification time: a `dataset.json`
        // newer than its `dataset.yaml` may still be an older build's
        // rendering of it.
        if std::fs::read(&json_path).is_ok_and(|current| current == json.as_bytes()) {
            return CommandResult {
                status: Status::Ok,
                message: "dataset.json is up to date".into(),
                produced: vec![json_path],
                elapsed: start.elapsed(),
            };
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::progress::ProgressLog;
    use std::path::Path;

    const YAML: &str = "name: t\nprofiles:\n  default:\n    maxk: 10\n    base_vectors: base.fvecs\n";

    fn ctx_at(workspace: &Path) -> StreamContext {
        StreamContext {
            attributes: Vec::new(),
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            cache: workspace.join(".cache"),
            defaults: indexmap::IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    /// **An older build's `dataset.json` is rewritten** even when it is
    /// newer than `dataset.yaml`: a modification time says nothing about
    /// which build rendered the file, so a rendering fix would otherwise
    /// never reach a dataset whose `dataset.yaml` is unchanged.
    #[test]
    fn a_newer_but_different_dataset_json_is_rewritten() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("dataset.yaml"), YAML).unwrap();
        let json_path = tmp.path().join("dataset.json");
        std::fs::write(&json_path, "{\"rendered_by\": \"an older veks\"}").unwrap();
        let older = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        std::fs::File::options().write(true).open(tmp.path().join("dataset.yaml")).unwrap()
            .set_modified(older).unwrap();

        let result = DatasetJsonOp.execute(&Options::new(), &mut ctx_at(tmp.path()));

        assert_eq!(result.status, Status::Ok, "{}", result.message);
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(written["name"], "t", "dataset.json was not re-rendered: {written}");
        assert_eq!(result.produced, vec![json_path]);
    }

    /// **A `dataset.json` this build would render is left alone**, so a
    /// re-run does not touch its modification time and send the steps
    /// that read it (its merkle tree, the catalogs) round again.
    #[test]
    fn an_identical_dataset_json_is_not_rewritten() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("dataset.yaml"), YAML).unwrap();
        let json_path = tmp.path().join("dataset.json");
        let first = DatasetJsonOp.execute(&Options::new(), &mut ctx_at(tmp.path()));
        assert_eq!(first.status, Status::Ok, "{}", first.message);
        let older = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        std::fs::File::options().write(true).open(&json_path).unwrap().set_modified(older).unwrap();

        let second = DatasetJsonOp.execute(&Options::new(), &mut ctx_at(tmp.path()));

        assert_eq!(second.status, Status::Ok, "{}", second.message);
        assert_eq!(second.message, "dataset.json is up to date");
        assert_eq!(std::fs::metadata(&json_path).unwrap().modified().unwrap(), older);
        assert_eq!(second.produced, vec![json_path]);
    }
}
