// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate catalog index files.
//!
//! Scans the workspace directory for `dataset.yaml` files and writes
//! `catalog.json` / `catalog.yaml` at each directory level. This is
//! the pipeline-integrated form of `veks prepare catalog generate`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::catalog::generate::{self, CatalogTree, GenerateEvent};

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status,
    StreamContext, render_options_table,
};

/// Pipeline command: write `catalog.json` / `catalog.yaml` index files for every directory level containing datasets.
pub struct CatalogGenerateOp;

/// Creates a boxed [`CatalogGenerateOp`]; registered as the `catalog generate` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(CatalogGenerateOp)
}

impl CommandOp for CatalogGenerateOp {
    fn command_path(&self) -> &str {
        "catalog generate"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_CATALOG
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate catalog index for the dataset directory".into(),
            body: format!(
                "# catalog generate\n\n\
                Generate catalog.json and catalog.yaml index files.\n\n\
                ## Description\n\n\
                Scans the input directory for dataset.yaml files and writes hierarchical \
                catalog index files (catalog.json and catalog.yaml) at each directory level. \
                Each catalog entry contains the dataset name, relative path, and embedded \
                layout (profiles and attributes).\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = options.get("source").unwrap_or(".");
        let basename = options.get("basename").unwrap_or("catalog");

        let input_path = if Path::new(input_str).is_absolute() {
            PathBuf::from(input_str)
        } else {
            ctx.workspace.join(input_str)
        };

        if !input_path.is_dir() {
            return error_result(format!("{} is not a directory", input_path.display()), start);
        }

        // The catalog boundary is the closest directory at or above the
        // input marked `.catalog_root` or bound by `.publish_url`: the
        // top of the catalog hierarchy. Paths stay logical, so a
        // workspace reached through a symlink keeps its publish root.
        let input_abs = generate::logical_path(&input_path);
        let boundary = generate::find_catalog_boundary(&input_abs);
        let scan_root = boundary.clone().unwrap_or(input_abs.clone());

        // If a publish tree exists, ensure the current dataset has a .publish
        // sentinel so it is included in the catalog and ready for publishing.
        if boundary.is_some() {
            let publish_sentinel = input_abs.join(generate::PUBLISH_SENTINEL);
            if !publish_sentinel.exists() {
                if let Err(e) = std::fs::write(&publish_sentinel, "") {
                    ctx.ui.log(&format!("  warning: could not create .publish: {}", e));
                } else {
                    ctx.ui.log("  created .publish sentinel");
                }
            }
        }

        let ui = ctx.ui.clone();
        let mut progress = |event: GenerateEvent| log_event(&ui, event);
        let tree = CatalogTree::scan(&scan_root, &mut progress);
        if tree.datasets().is_empty() {
            return CommandResult {
                status: Status::Ok,
                message: "no datasets found — no catalogs written".into(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let catalog_dirs = tree.catalog_dirs();
        let produced = match tree.write(&catalog_dirs, basename, &mut progress) {
            Ok(produced) => produced,
            Err(e) => return error_result(e, start),
        };

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} datasets cataloged across {} directories ({} files written)",
                tree.datasets().len(), catalog_dirs.len(), produced.len(),
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".into(),
                type_name: "Path".into(),
                required: false,
                default: Some(".".into()),
                description: "Root directory to scan for datasets".into(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "basename".into(),
                type_name: "String".into(),
                required: false,
                default: Some("catalog".into()),
                description: "Base filename for catalog files (without extension)".into(),
                extended_description: None,
                role: OptionRole::Config,
            },
        ]
    }
}

/// One line of the run's log for a generator event.
fn log_event(ui: &veks_core::ui::UiHandle, event: GenerateEvent) {
    match event {
        GenerateEvent::Scanning(dir) => ui.log(&format!("  scanning {} for datasets...", dir.display())),
        GenerateEvent::Found(ds) => {
            ui.log(&format!("  dataset: {} [{}]", ds.config().name, ds.yaml_path().display()))
        }
        GenerateEvent::Unreadable { path, error } => {
            ui.log(&format!("  warning: failed to load {}: {error}", path.display()))
        }
        GenerateEvent::Marked(dir) => ui.log(&format!("  skipping {} (.do_not_catalog)", dir.display())),
        GenerateEvent::Wrote { dir, entries, files } => {
            let names: Vec<String> = files
                .iter()
                .filter_map(|f| f.file_name().map(|n| n.to_string_lossy().into_owned()))
                .collect();
            ui.log(&format!("  wrote {entries} entries to {} ({})", dir.display(), names.join(" + ")))
        }
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
