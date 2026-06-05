// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: `generate example-dataset`.
//!
//! Scaffolds a dataset's **source files** in one step, so a newcomer can get
//! something publishable with nothing but:
//!
//! ```text
//! veks generate example-dataset --target mynewdatasetdir
//! ```
//!
//! It only lays down the *generated* facets — base vectors, query vectors, and
//! optionally metadata — by reusing the existing [`generate vectors`] and
//! [`generate metadata`] commands, then writes a minimal `dataset.yaml`. The
//! *derived* facets (exact + filtered KNN ground truth, predicates, …) are not
//! produced here: point `veks prepare bootstrap` + `veks run` at these source
//! files to compute them.
//!
//! [`generate vectors`]: super::gen_vectors
//! [`generate metadata`]: super::gen_metadata

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
    render_options_table,
};
use super::{gen_metadata, gen_vectors};

const DEFAULT_BASE_COUNT: u64 = 10_000;
const DEFAULT_QUERY_COUNT: u64 = 100;
const DEFAULT_DIMENSION: u32 = 128;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_FACETS: &str = "BQ";
const DEFAULT_METADATA_FIELDS: u32 = 1;
const DEFAULT_DISTANCE: &str = "L2";

// Facet file names, relative to the dataset directory.
const BASE_FILE: &str = "base_vectors.fvec";
const QUERY_FILE: &str = "query_vectors.fvec";
const METADATA_FILE: &str = "metadata_content.u8";

/// Pipeline command: scaffold an example dataset directory (source files only).
pub struct GenerateExampleDatasetOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateExampleDatasetOp)
}

impl CommandOp for GenerateExampleDatasetOp {
    fn command_path(&self) -> &str {
        "generate example-dataset"
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
            summary: "Scaffold a dataset's source files (base/query vectors, optional metadata)"
                .into(),
            body: format!(
                r#"# generate example-dataset

Scaffold a dataset's source files in one step.

## Description

Creates `<target>/` (if needed) and lays down the *generated* facets selected
by `--facets`, reusing the existing generators:

- `B` — `base_vectors.fvec` (`generate vectors`, `--base-count` × `--dimension`)
- `Q` — `query_vectors.fvec` (`generate vectors`, `--query-count` × `--dimension`)
- `M` — `metadata_content.u8` (`generate metadata`, `--metadata-fields` per record)

…then writes a minimal `dataset.yaml` with a single `default` profile naming
them, plus the `--distance` function. The default `--facets BQ` produces base
and query vectors.

Only source facets are generated here. The *derived* facets — exact and
filtered KNN ground truth, predicates, filtered neighbors — are left to the
pipeline: point `veks prepare bootstrap --base-vectors <target>/base_vectors.fvec`
+ `veks run` at these files to compute the rest.

## Reproducibility

Generation is deterministic in `--seed` (the query set uses a derived seed so
it differs from the base set).

## Examples

```text
# Base + query vectors, defaults (10,000 base, 100 query, dim 128, L2).
veks generate example-dataset --target mynewdataset

# Base vectors only — just a big object to serve.
veks generate example-dataset --target big --facets B --base-count 200000

# Add a metadata facet.
veks generate example-dataset --target d --facets BQM --metadata-fields 3
```

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let target_str = match options.require("target") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let target = resolve_path(&target_str, &ctx.workspace);

        let base_count = match parse_u64(options.get("base-count"), DEFAULT_BASE_COUNT, "base-count") {
            Ok(n) if n > 0 => n,
            Ok(_) => return error_result("base-count must be > 0".into(), start),
            Err(e) => return error_result(e, start),
        };
        let query_count = match parse_u64(options.get("query-count"), DEFAULT_QUERY_COUNT, "query-count") {
            Ok(n) if n > 0 => n,
            Ok(_) => return error_result("query-count must be > 0".into(), start),
            Err(e) => return error_result(e, start),
        };
        let dimension = match parse_u64(options.get("dimension"), DEFAULT_DIMENSION as u64, "dimension") {
            Ok(d) if d > 0 && d <= u32::MAX as u64 => d,
            Ok(_) => return error_result("dimension out of range".into(), start),
            Err(e) => return error_result(e, start),
        };
        let metadata_fields =
            match parse_u64(options.get("metadata-fields"), DEFAULT_METADATA_FIELDS as u64, "metadata-fields") {
                Ok(f) if f > 0 => f,
                Ok(_) => return error_result("metadata-fields must be > 0".into(), start),
                Err(e) => return error_result(e, start),
            };
        let seed = match parse_u64(options.get("seed"), DEFAULT_SEED, "seed") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let distance = options.get("distance").unwrap_or(DEFAULT_DISTANCE).to_string();
        let name = match options.get("name") {
            Some(n) => n.to_string(),
            None => target
                .file_name()
                .and_then(|n| n.to_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("example")
                .to_string(),
        };
        let force = options.get("force").map(|v| v != "false").unwrap_or(false);

        // Facet selection: B is always produced; Q and M are opt-in.
        let facets = options.get("facets").unwrap_or(DEFAULT_FACETS).to_uppercase();
        for ch in facets.chars() {
            if !"BQM".contains(ch) {
                return error_result(
                    format!("unknown facet '{ch}' in --facets '{facets}' (supported: B, Q, M)"),
                    start,
                );
            }
        }
        if !facets.contains('B') {
            return error_result(
                format!("--facets '{facets}' must include B (base vectors)"),
                start,
            );
        }
        let want_query = facets.contains('Q');
        let want_metadata = facets.contains('M');

        let yaml_path = target.join("dataset.yaml");
        if yaml_path.exists() && !force {
            return error_result(
                format!("{} already exists — pass --force to overwrite", yaml_path.display()),
                start,
            );
        }
        if let Err(e) = std::fs::create_dir_all(&target) {
            return error_result(format!("creating {}: {e}", target.display()), start);
        }

        ctx.ui.log(&format!(
            "generate example-dataset '{name}' [{facets}] → {}",
            target.display()
        ));

        // Reuse `generate vectors` / `generate metadata`, writing into the
        // dataset directory (so their outputs and bookkeeping land there).
        ctx.workspace = target.clone();
        let dim = dimension.to_string();

        // B — base vectors.
        if let Err(r) = run_sub(
            ctx,
            gen_vectors::factory(),
            &[
                ("output", BASE_FILE),
                ("type", "float[]"),
                ("dimension", &dim),
                ("count", &base_count.to_string()),
                ("seed", &seed.to_string()),
            ],
            start,
        ) {
            return r;
        }

        // Q — query vectors (a derived seed keeps them distinct from the base).
        if want_query
            && let Err(r) = run_sub(
                ctx,
                gen_vectors::factory(),
                &[
                    ("output", QUERY_FILE),
                    ("type", "float[]"),
                    ("dimension", &dim),
                    ("count", &query_count.to_string()),
                    ("seed", &seed.wrapping_add(1).to_string()),
                ],
                start,
            )
        {
            return r;
        }

        // M — metadata content (single-byte integer fields).
        if want_metadata
            && let Err(r) = run_sub(
                ctx,
                gen_metadata::factory(),
                &[
                    ("output", METADATA_FILE),
                    ("format", "u8"),
                    ("count", &base_count.to_string()),
                    ("fields", &metadata_fields.to_string()),
                    ("range-min", "0"),
                    ("range-max", "12"),
                    ("seed", &seed.to_string()),
                ],
                start,
            )
        {
            return r;
        }

        if let Err(e) = write_dataset_yaml(&yaml_path, &name, &distance, want_query, want_metadata) {
            return error_result(e, start);
        }

        let mut produced = vec![target.join(BASE_FILE)];
        if want_query {
            produced.push(target.join(QUERY_FILE));
        }
        if want_metadata {
            produced.push(target.join(METADATA_FILE));
        }
        produced.push(yaml_path);

        CommandResult {
            status: Status::Ok,
            message: format!(
                "wrote example dataset '{name}' [{facets}] to {} — bootstrap + run to add derived facets",
                target.display()
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        let cfg = |name: &str, default: Option<String>, description: &str| OptionDesc {
            name: name.to_string(),
            type_name: "string".to_string(),
            required: false,
            default,
            description: description.to_string(),
            extended_description: None,
            role: OptionRole::Config,
        };
        vec![
            OptionDesc {
                name: "target".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Directory to create the dataset in".to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            cfg("facets", Some(DEFAULT_FACETS.to_string()),
                "Source facets to generate: any of B (base), Q (query), M (metadata)"),
            cfg("base-count", Some(DEFAULT_BASE_COUNT.to_string()), "Number of base vectors"),
            cfg("query-count", Some(DEFAULT_QUERY_COUNT.to_string()), "Number of query vectors (facet Q)"),
            cfg("dimension", Some(DEFAULT_DIMENSION.to_string()), "Vector dimensionality"),
            cfg("metadata-fields", Some(DEFAULT_METADATA_FIELDS.to_string()),
                "Integer metadata fields per record (facet M)"),
            cfg("seed", Some(DEFAULT_SEED.to_string()), "Random seed (same seed → identical output)"),
            cfg("distance", Some(DEFAULT_DISTANCE.to_string()),
                "Distance function recorded in dataset.yaml (e.g. L2, IP, COSINE)"),
            cfg("name", None, "Dataset name (defaults to the target directory's name)"),
            OptionDesc {
                name: "force".to_string(),
                type_name: "flag".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Overwrite an existing dataset.yaml in the target".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
        ]
    }
}

/// Run one reused sub-command into the dataset directory. On failure, returns
/// its `CommandResult` (with the sub-command's message) for the caller to
/// propagate.
fn run_sub(
    ctx: &mut StreamContext,
    mut op: Box<dyn CommandOp>,
    opts: &[(&str, &str)],
    start: Instant,
) -> Result<(), CommandResult> {
    let mut o = Options::new();
    for (k, v) in opts {
        o.set(*k, *v);
    }
    let r = op.execute(&o, ctx);
    if r.status == Status::Ok {
        Ok(())
    } else {
        Err(error_result(
            format!("{} ({})", r.message, op.command_path()),
            start,
        ))
    }
}

/// Write the minimal `dataset.yaml` naming the generated source facets.
fn write_dataset_yaml(
    path: &Path,
    name: &str,
    distance: &str,
    query: bool,
    metadata: bool,
) -> Result<(), String> {
    let mut profile = format!("    base_vectors: {BASE_FILE}\n");
    if query {
        profile.push_str(&format!("    query_vectors: {QUERY_FILE}\n"));
    }
    if metadata {
        profile.push_str(&format!("    metadata_content: {METADATA_FILE}\n"));
    }
    let body = format!(
        "# Generated by `veks generate example-dataset` (source facets only).\n\
         # Run `veks prepare bootstrap` + `veks run` against these to add the\n\
         # derived facets (KNN ground truth, predicates, filtered neighbors).\n\
         name: {name}\n\
         description: Synthetic example dataset\n\
         attributes:\n\
         \x20 distance_function: {distance}\n\
         profiles:\n\
         \x20 default:\n\
         {profile}",
    );
    std::fs::write(path, body).map_err(|e| format!("writing {}: {e}", path.display()))
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

/// Parse an optional decimal `u64` option, falling back to `default`.
fn parse_u64(value: Option<&str>, default: u64, what: &str) -> Result<u64, String> {
    match value {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("invalid {what} '{s}' (expected a non-negative integer)")),
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult { status: Status::Error, message, produced: vec![], elapsed: start.elapsed() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

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
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    fn run(dir: &Path, target: &str, extra: &[(&str, &str)]) -> CommandResult {
        let mut opts = Options::new();
        opts.set("target", target);
        for (k, v) in extra {
            opts.set(*k, *v);
        }
        GenerateExampleDatasetOp.execute(&opts, &mut test_ctx(dir))
    }

    #[test]
    fn defaults_produce_base_and_query_plus_loadable_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        let result = run(tmp.path(), "mydataset", &[("base-count", "32"), ("query-count", "8"), ("dimension", "8")]);
        assert_eq!(result.status, Status::Ok, "{}", result.message);

        let dir = tmp.path().join("mydataset");
        // B and Q present; fvec record = 4 (dim header) + 8 * 4 (f32) = 36 bytes.
        assert_eq!(std::fs::metadata(dir.join(BASE_FILE)).unwrap().len(), 32 * (4 + 8 * 4));
        assert_eq!(std::fs::metadata(dir.join(QUERY_FILE)).unwrap().len(), 8 * (4 + 8 * 4));
        assert!(!dir.join(METADATA_FILE).exists(), "no M unless requested");

        let cfg = vectordata::dataset::DatasetConfig::load(&dir.join("dataset.yaml"))
            .expect("dataset.yaml should load");
        assert_eq!(cfg.name, "mydataset");
    }

    #[test]
    fn facets_b_is_base_only() {
        let tmp = tempfile::tempdir().unwrap();
        run(tmp.path(), "b", &[("facets", "B"), ("base-count", "10"), ("dimension", "4")]);
        let dir = tmp.path().join("b");
        assert!(dir.join(BASE_FILE).exists());
        assert!(!dir.join(QUERY_FILE).exists());
    }

    #[test]
    fn facets_bqm_adds_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let r = run(tmp.path(), "m", &[("facets", "BQM"), ("base-count", "20"), ("query-count", "5"), ("dimension", "4")]);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        let dir = tmp.path().join("m");
        assert!(dir.join(BASE_FILE).exists());
        assert!(dir.join(QUERY_FILE).exists());
        assert!(dir.join(METADATA_FILE).exists());
    }

    #[test]
    fn same_seed_is_byte_identical() {
        let tmp = tempfile::tempdir().unwrap();
        run(tmp.path(), "a", &[("base-count", "64"), ("query-count", "16"), ("dimension", "16"), ("seed", "7")]);
        run(tmp.path(), "c", &[("base-count", "64"), ("query-count", "16"), ("dimension", "16"), ("seed", "7")]);
        assert_eq!(
            std::fs::read(tmp.path().join("a").join(BASE_FILE)).unwrap(),
            std::fs::read(tmp.path().join("c").join(BASE_FILE)).unwrap()
        );
        // Query set differs from the base set (derived seed).
        assert_ne!(
            std::fs::read(tmp.path().join("a").join(BASE_FILE)).unwrap(),
            std::fs::read(tmp.path().join("a").join(QUERY_FILE)).unwrap()
        );
    }

    #[test]
    fn bad_facet_and_missing_b_are_errors() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(run(tmp.path(), "x", &[("facets", "BZ")]).status, Status::Error);
        assert_eq!(run(tmp.path(), "y", &[("facets", "Q")]).status, Status::Error);
    }

    #[test]
    fn refuses_to_clobber_without_force() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(run(tmp.path(), "d", &[("facets", "B"), ("base-count", "8"), ("dimension", "4")]).status, Status::Ok);
        assert_eq!(run(tmp.path(), "d", &[("facets", "B"), ("base-count", "8"), ("dimension", "4")]).status, Status::Error);
        assert_eq!(
            run(tmp.path(), "d", &[("facets", "B"), ("base-count", "8"), ("dimension", "4"), ("force", "true")]).status,
            Status::Ok
        );
    }
}
