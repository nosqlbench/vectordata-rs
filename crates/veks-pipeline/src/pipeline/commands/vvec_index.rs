// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: write the `IDXFOR__` offset index of every
//! variable-length file, so each ships with the index a remote reader
//! opens it by. The pipeline form of what `vectordata push` and
//! `catalog generate` do before they publish ([`vectordata::vvec_index`]).

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status,
    StreamContext, render_options_table,
};

/// Pipeline command: bring the offset index of every variable-length file under `source` current.
pub struct VvecIndexOp;

/// Creates a boxed [`VvecIndexOp`]; registered as the `generate vvec-index` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VvecIndexOp)
}

impl CommandOp for VvecIndexOp {
    fn command_path(&self) -> &str {
        "generate vvec-index"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_GENERATE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Write the offset index of every variable-length file".into(),
            body: format!(
                "# generate vvec-index\n\n\
                Write the `IDXFOR__` offset index of every variable-length file.\n\n\
                ## Description\n\n\
                A variable-length file (`.ivvec`, `.fvvecs`, …) is read by ordinal through \
                its `IDXFOR__<name>.<i32|i64>` sidecar, the start offset of each record. A \
                remote reader without one has to download the file whole to find its \
                records, and vectordata refuses to do that behind an open, so a dataset \
                ships an index beside each such file.\n\n\
                This step writes the index of every variable-length file under `source` \
                (a directory, searched with the publish filters, or one file) that has \
                none, or one older than the file. It is idempotent: a current index is \
                left as it is. Where the file has a `.mref`, the index gets one too, at \
                the file's chunk size. A file that is empty, or does not end at a record \
                boundary, fails the step.\n\n\
                The bare `.ivec` is not indexed: it is also the uniform ground-truth \
                format.\n\n\
                ## Data Preparation Role\n\n\
                Run it before `merkle create`, so the indexes are merkled with the rest \
                of the dataset, and before `catalog generate`. `vectordata push`, \
                `veks publish` and `catalog generate` write a missing index themselves; \
                this step makes it part of the pipeline's own output.\n\n\
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
        let source = options.get("source").unwrap_or(".");
        let source = if Path::new(source).is_absolute() { PathBuf::from(source) } else { ctx.workspace.join(source) };
        let files = if source.is_dir() {
            vectordata::vvec_index::indexed_files_under(&source)
        } else if source.is_file() {
            let name = source.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            if !vectordata::vvec_index::is_indexed_file(name) {
                return error_result(format!("{} is not a variable-length file", source.display()), start);
            }
            vec![source.clone()]
        } else {
            return error_result(format!("{} does not exist", source.display()), start);
        };

        let mut produced = Vec::new();
        let mut current = 0usize;
        for file in &files {
            let planned = match vectordata::vvec_index::prepare(file, true, &mut |_| {}) {
                Ok(p) => p,
                Err(e) => return error_result(e, start),
            };
            if !planned.index_written && !planned.mref_written {
                current += 1;
                continue;
            }
            let rel = file.strip_prefix(&ctx.workspace).unwrap_or(file).display().to_string();
            if ctx.dry_run {
                ctx.ui.log(&format!("  would index {rel}"));
                continue;
            }
            let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
            let bar = ctx.ui.bar_with_unit(size, format!("indexing {rel}"), "B");
            let done = vectordata::vvec_index::prepare(file, false, &mut |at| bar.set_position(at));
            bar.finish();
            let done = match done {
                Ok(d) => d,
                Err(e) => return error_result(e, start),
            };
            if done.index_written {
                produced.push(done.index.clone());
            }
            if let (Some(mref), true) = (done.mref, done.mref_written) {
                produced.push(mref);
            }
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} variable-length file(s): {} {}, {current} already current",
                files.len(),
                files.len() - current,
                if ctx.dry_run { "to index" } else { "indexed" },
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        self.project_artifacts_in(step_id, options, Path::new("."))
    }

    /// The indexes follow from the variable-length files the source
    /// holds, so the projection reads it under the dataset directory the
    /// caller names, as `merkle create`'s does: each file in, its index
    /// (and the index's `.mref`, when the file has one) out.
    fn project_artifacts_in(&self, step_id: &str, options: &Options, workspace: &Path) -> ArtifactManifest {
        let source_str = options.get("source").unwrap_or(".");
        let source = PathBuf::from(source_str);
        let source = if source.is_absolute() { source } else { workspace.join(source) };
        let files = if source.is_dir() { vectordata::vvec_index::indexed_files_under(&source) } else { vec![source.clone()] };
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for file in &files {
            let rel = |p: &Path| p.strip_prefix(&source).ok().filter(|r| !r.as_os_str().is_empty()).unwrap_or(p).to_string_lossy().into_owned();
            // A file not yet written has no size to choose the width by;
            // the narrow one is what all but a >2 GiB file get.
            let index = veks_io::reader::vvec_index_path(file).unwrap_or_else(|_| {
                let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                file.with_file_name(format!("IDXFOR__{name}.i32"))
            });
            inputs.push(if source.is_dir() { rel(file) } else { source_str.to_string() });
            let index_rel = if source.is_dir() { rel(&index) } else { index.to_string_lossy().into_owned() };
            let merkled = file.with_file_name(format!("{}.mref", file.file_name().unwrap_or_default().to_string_lossy())).is_file();
            outputs.push(index_rel.clone());
            if merkled {
                outputs.push(format!("{index_rel}.mref"));
            }
        }
        ArtifactManifest { step_id: step_id.to_string(), command: self.command_path().to_string(), inputs, outputs, intermediates: vec![] }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![OptionDesc {
            name: "source".into(),
            type_name: "Path".into(),
            required: false,
            default: Some(".".into()),
            description: "Directory to search for variable-length files, or one such file".into(),
            extended_description: None,
            role: OptionRole::Input,
        }]
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult { status: Status::Error, message, produced: vec![], elapsed: start.elapsed() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use std::sync::Arc;

    fn ctx(dir: &Path, sink: Arc<veks_core::ui::TestSink>, dry_run: bool) -> StreamContext {
        StreamContext {
            attributes: Vec::new(),
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(sink),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    fn write_ivvec(path: &Path, records: &[&[i32]]) {
        let mut bytes = Vec::new();
        for r in records {
            bytes.extend_from_slice(&(r.len() as i32).to_le_bytes());
            r.iter().for_each(|x| bytes.extend_from_slice(&x.to_le_bytes()));
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn run(dir: &Path, dry_run: bool) -> (CommandResult, Arc<veks_core::ui::TestSink>) {
        let sink = Arc::new(veks_core::ui::TestSink::new());
        let mut c = ctx(dir, sink.clone(), dry_run);
        let mut options = Options::new();
        options.set("source", ".");
        (VvecIndexOp.execute(&options, &mut c), sink)
    }

    /// **The step indexes every variable-length file once, and says
    /// so**: the second run finds them current and writes nothing; its
    /// projection names the indexes it produces. The uniform `.ivec`
    /// is not indexed.
    #[test]
    fn the_step_indexes_every_vvec_once() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("profiles/default")).unwrap();
        write_ivvec(&dir.path().join("profiles/default/metadata_results.ivvec"), &[&[1, 2], &[]]);
        write_ivvec(&dir.path().join("neighbor_indices.ivec"), &[&[1]]);
        let index = dir.path().join("profiles/default/IDXFOR__metadata_results.ivvec.i32");

        let (first, sink) = run(dir.path(), false);
        assert_eq!(first.status, Status::Ok, "{}", first.message);
        assert_eq!(first.produced, vec![index.clone()]);
        assert!(sink.events().iter().any(|e| matches!(e, veks_core::ui::UiEvent::ProgressCreate { label, .. } if label.contains("metadata_results.ivvec"))));
        assert!(!dir.path().join("IDXFOR__neighbor_indices.ivec.i32").exists());

        let (second, _) = run(dir.path(), false);
        assert_eq!(second.status, Status::Ok);
        assert!(second.produced.is_empty(), "{:?}", second.produced);
        assert!(second.message.contains("1 already current"), "{}", second.message);

        let mut options = Options::new();
        options.set("source", ".");
        let manifest = VvecIndexOp.project_artifacts_in("index", &options, dir.path());
        assert_eq!(manifest.inputs, vec!["profiles/default/metadata_results.ivvec".to_string()]);
        assert_eq!(manifest.outputs, vec!["profiles/default/IDXFOR__metadata_results.ivvec.i32".to_string()]);
    }

    /// **A dry run writes no index**, and names the file it would.
    #[test]
    fn a_dry_run_writes_no_index() {
        let dir = tempfile::tempdir().unwrap();
        write_ivvec(&dir.path().join("m.ivvec"), &[&[1]]);
        let (result, sink) = run(dir.path(), true);
        assert_eq!(result.status, Status::Ok);
        assert!(!dir.path().join("IDXFOR__m.ivvec.i32").exists());
        assert!(sink.log_messages().iter().any(|m| m.contains("would index m.ivvec")), "{:?}", sink.log_messages());
    }

    /// **An empty variable-length file fails the step**, naming it.
    #[test]
    fn an_empty_vvec_fails_the_step() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("m.ivvec"), b"").unwrap();
        let (result, _) = run(dir.path(), false);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("m.ivvec") && result.message.contains("is empty"), "{}", result.message);
    }
}
