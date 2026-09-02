// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `state set` with `attribute: true` records a dataset attribute in
//! `dataset.yaml` rather than a pipeline variable — the back-fill route
//! for embedding provenance on a dataset whose embed step predates
//! recording it.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use vectordata::dataset::DatasetConfig;
use veks_core::ui::{TestSink, UiHandle};
use veks_pipeline::pipeline::command::{CommandOp, Options, Status, StreamContext};
use veks_pipeline::pipeline::commands::set_variable::SetVariableOp;
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::resource::ResourceGovernor;

fn tmp_dir() -> tempfile::TempDir {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

fn test_ctx(dir: &Path) -> StreamContext {
    StreamContext {
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
        ui: UiHandle::new(Arc::new(TestSink::new())),
        status_interval: Duration::from_secs(1),
        estimated_total_steps: 0,
        provenance_selector: veks_pipeline::pipeline::provenance::ProvenanceFlags::STRICT,
    }
}

fn set(dir: &Path, name: &str, value: &str, attribute: bool) -> (Status, String, StreamContext) {
    let mut op = SetVariableOp;
    let mut ctx = test_ctx(dir);
    let mut o = Options::new();
    o.set("name", name);
    o.set("value", value);
    if attribute {
        o.set("attribute", "true");
    }
    let r = op.execute(&o, &mut ctx);
    (r.status, r.message, ctx)
}

#[test]
fn state_set_attribute_writes_dataset_yaml_not_variables() {
    let dir = tmp_dir();
    std::fs::write(
        dir.path().join("dataset.yaml"),
        "name: tessera\nattributes:\n  model: Qwen/Qwen3-Embedding-0.6B\n  is_normalized: true\n",
    )
    .unwrap();

    let (status, message, ctx) = set(
        dir.path(),
        "model_revision",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        true,
    );
    assert_eq!(status, Status::Ok, "{}", message);
    assert!(message.contains("attribute"), "{}", message);
    let (status, _, _) = set(dir.path(), "model_revision_requested", "main", true);
    assert_eq!(status, Status::Ok);

    let config = DatasetConfig::load(&dir.path().join("dataset.yaml")).unwrap();
    let attrs = config.attributes.as_ref().unwrap();
    assert_eq!(
        attrs.model.as_deref(),
        Some("Qwen/Qwen3-Embedding-0.6B"),
        "existing attributes kept"
    );
    assert_eq!(attrs.is_normalized, Some(true));
    assert_eq!(
        attrs.model_revision.as_deref(),
        Some("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
    );
    assert_eq!(attrs.model_revision_requested.as_deref(), Some("main"));
    assert!(
        !dir.path().join("variables.yaml").exists(),
        "an attribute is not a variable"
    );
    assert!(ctx.defaults.get("model_revision").is_none());

    // An unknown attribute key is refused, not silently dropped.
    let (status, message, _) = set(dir.path(), "not_an_attribute", "x", true);
    assert_eq!(status, Status::Error, "{}", message);
    assert!(message.contains("not_an_attribute"), "{}", message);

    // Without the flag, the same name is an ordinary variable.
    let (status, _, ctx) = set(dir.path(), "model_revision", "other", false);
    assert_eq!(status, Status::Ok);
    assert_eq!(
        ctx.defaults.get("model_revision").map(String::as_str),
        Some("other")
    );
    assert!(dir.path().join("variables.yaml").exists());
    let config = DatasetConfig::load(&dir.path().join("dataset.yaml")).unwrap();
    assert_eq!(
        config
            .attributes
            .as_ref()
            .unwrap()
            .model_revision
            .as_deref(),
        Some("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
        "the attribute is untouched by the variable of the same name"
    );
}
