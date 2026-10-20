// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Shared fixtures for the element type matrix tests.

#![allow(dead_code)]

pub mod samples;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use veks_core::ui::{TestSink, UiHandle};
use veks_datatypes::{Container, Element, ElementType};
use veks_pipeline::pipeline::command::{CommandOp, CommandResult, Options, StreamContext};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::registry::CommandRegistry;
use veks_pipeline::pipeline::resource::ResourceGovernor;

/// Write uniform vectors of `T`: a little-endian `i32` dimension header
/// per record, then each element through [`Element::write_le_bytes`].
pub fn write_vec_file<T: Element>(path: &Path, records: &[Vec<T>]) {
    let mut bytes = Vec::new();
    for record in records {
        bytes.extend_from_slice(&(record.len() as i32).to_le_bytes());
        let mut buf = vec![0u8; T::SIZE];
        for &v in record {
            v.write_le_bytes(&mut buf);
            bytes.extend_from_slice(&buf);
        }
    }
    std::fs::write(path, bytes).unwrap();
}

/// The path `dir/<stem>.<ext>` where `ext` is the canonical uniform
/// vector spelling of `ty`.
pub fn vec_path(dir: &Path, stem: &str, ty: ElementType) -> PathBuf {
    dir.join(format!("{stem}.{}", ty.extension(Container::Vec)))
}

/// Read a uniform vector file back as `f64` records, through the
/// pipeline's own widening reader.
pub fn read_as_f64(path: &Path) -> Vec<Vec<f64>> {
    let src = veks_pipeline::pipeline::element_type::F64Vectors::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    (0..src.count()).map(|i| src.get(i)).collect()
}

/// A `StreamContext` rooted at `dir` whose UI events are captured in
/// the returned sink.
pub fn test_ctx(dir: &Path) -> (StreamContext, Arc<TestSink>) {
    let sink = Arc::new(TestSink::new());
    let ctx = StreamContext {
        attributes: Vec::new(),
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace: dir.to_path_buf(),
        cache: dir.join(".cache"),
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads: 1,
        step_id: String::new(),
        governor: ResourceGovernor::default_governor(),
        ui: UiHandle::new(sink.clone()),
        status_interval: Duration::from_secs(1),
        estimated_total_steps: 0,
        provenance_selector: veks_pipeline::pipeline::provenance::ProvenanceFlags::STRICT,
    };
    (ctx, sink)
}

/// What a command run left behind: its result and the UI log lines.
pub struct Run {
    pub result: CommandResult,
    pub logs: Vec<String>,
}

/// Run the registered command `command` with `opts`, in a context
/// rooted at `dir`.
pub fn run_command(command: &str, dir: &Path, opts: &Options) -> Run {
    let registry = CommandRegistry::with_builtins();
    let factory = registry.get(command).unwrap_or_else(|| panic!("command {command:?} is not registered"));
    let mut op: Box<dyn CommandOp> = factory();
    let (mut ctx, sink) = test_ctx(dir);
    let result = op.execute(opts, &mut ctx);
    Run { result, logs: sink.log_messages() }
}

/// `opts` built from `(key, value)` pairs.
pub fn options(pairs: &[(&str, &str)]) -> Options {
    let mut opts = Options::new();
    for (k, v) in pairs {
        opts.set(*k, v.to_string());
    }
    opts
}

/// `text` with every occurrence of `path` replaced by `<path>`, so that
/// a message naming the input file compares equal across input files.
pub fn hide_path(text: &str, path: &Path) -> String {
    text.replace(&path.display().to_string(), "<path>")
}
