// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: synthesize metadata records.
//!
//! Creates a slab file of MNode records with synthetic attribute fields.
//! Used when metadata is required for predicate/filtered-KNN testing but
//! no source metadata file is available.
//!
//! Field values are drawn from uniform distributions within configurable
//! ranges. The output can be used directly by the `generate predicates`
//! and `compute evaluate-predicates` pipeline steps.

use std::path::Path;
use std::time::Instant;

use rand::SeedableRng;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use vectordata::formats::mnode::{MNode, MValue};

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};

fn error_result(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: msg.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

/// Canonical name of the slab namespace carrying the metadata layout
/// (schema) alongside the metadata content. Addressed in a view as
/// `…/metadata_content.slab#layout`.
pub const LAYOUT_NAMESPACE: &str = "layout";

/// The metadata layout (schema) bytes for a set of integer metadata
/// fields: the structural fingerprint (field names + types, values
/// defaulted) of one record, serialized as anode `MNode` bytes. Opaque to
/// the pipeline — callers decode with an `anode` implementation. Two
/// metadata artifacts are schema-compatible iff these bytes match
/// byte-for-byte.
pub fn metadata_layout_bytes(field_names: &[String]) -> Vec<u8> {
    let mut schema = MNode::new();
    for name in field_names {
        schema.insert(name.clone(), MValue::Int32(0));
    }
    schema.fingerprint().to_bytes()
}

fn resolve_path(s: &str, workspace: &Path) -> std::path::PathBuf {
    let p = std::path::Path::new(s);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

/// Pipeline command: generate synthetic metadata.
pub struct GenerateMetadataOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateMetadataOp)
}

impl CommandOp for GenerateMetadataOp {
    fn command_path(&self) -> &str {
        "generate metadata"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_GENERATE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Synthesize metadata records with random attribute fields".into(),
            body: format!(r#"# generate metadata

Create a slab file of MNode metadata records with synthetic fields.

Each record contains the configured number of integer fields, each drawn
from a uniform distribution in `[range_min, range_max)`. The output is
compatible with `generate predicates` and `compute evaluate-predicates`.

## Strategies

- **uniform-int** (default): Each field is an i32 drawn uniformly from
  the configured range. Field names are `field_0`, `field_1`, etc.

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let output_str = match options.require("output") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let count: usize = match options.require("count") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid count: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let fields: usize = match options.parse_or("fields", 3usize) {
            Ok(v) => v.max(1), Err(e) => return error_result(e, start),
        };
        let range_min: i32 = match options.parse_or("range-min", 0i32) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };
        let range_max: i32 = match options.parse_or("range-max", 1000i32) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };
        if range_max <= range_min {
            return error_result(
                format!("range-max ({}) must be greater than range-min ({})", range_max, range_min),
                start,
            );
        }
        let seed: u64 = match options.parse_or("seed", 42u64) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };

        let output_path = resolve_path(output_str, &ctx.workspace);
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        let format = options.get("format").unwrap_or("slab");

        ctx.ui.log(&format!(
            "  generate metadata: {} records, {} fields, range [{}..{}), seed={}, format={}",
            count, fields, range_min, range_max, seed, format,
        ));

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let range = range_max - range_min;

        // Detect scalar formats (u8, i8, u16, i16, u32, i32, u64, i64)
        let is_scalar = matches!(format, "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64");
        let scalar_elem_size: usize = match format {
            "u8" | "i8" => 1, "u16" | "i16" => 2, "u32" | "i32" => 4, "u64" | "i64" => 8, _ => 0,
        };

        // The metadata field schema. The same `field_N` names back every
        // content format (scalar/ivec/slab), so the layout is identical
        // regardless of how the content is encoded.
        let field_names: Vec<String> = (0..fields).map(|i| format!("field_{}", i)).collect();

        if is_scalar {
            // Write as flat packed scalar: each record is `fields` values of the scalar type
            use std::io::Write;
            let mut f = match std::fs::File::create(&output_path) {
                Ok(f) => std::io::BufWriter::new(f),
                Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
            };
            let pb = ctx.ui.bar_with_unit(count as u64, "generating metadata", "records");
            for ordinal in 0..count {
                for _ in 0..fields {
                    let value = range_min + rng.random_range(0..range);
                    let write_ok = match scalar_elem_size {
                        1 => f.write_all(&[value as u8]).is_ok(),
                        2 => f.write_all(&(value as i16).to_le_bytes()).is_ok(),
                        4 => f.write_all(&value.to_le_bytes()).is_ok(),
                        8 => f.write_all(&(value as i64).to_le_bytes()).is_ok(),
                        _ => false,
                    };
                    if !write_ok {
                        return error_result(format!("write record {}", ordinal), start);
                    }
                }
                if (ordinal + 1) % 10_000 == 0 {
                    pb.set_position((ordinal + 1) as u64);
                }
            }
            pb.finish();
        } else if format == "ivec" {
            // Write as ivec: each record is `fields` i32 values
            use std::io::Write;
            let mut f = match std::fs::File::create(&output_path) {
                Ok(f) => std::io::BufWriter::new(f),
                Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
            };
            let pb = ctx.ui.bar_with_unit(count as u64, "generating metadata", "records");
            for ordinal in 0..count {
                // Write dimension header
                if f.write_all(&(fields as i32).to_le_bytes()).is_err() {
                    return error_result(format!("write record {}", ordinal), start);
                }
                for _ in 0..fields {
                    let value = range_min + rng.random_range(0..range);
                    if f.write_all(&value.to_le_bytes()).is_err() {
                        return error_result(format!("write record {}", ordinal), start);
                    }
                }
                if (ordinal + 1) % 10_000 == 0 {
                    pb.set_position((ordinal + 1) as u64);
                }
            }
            pb.finish();
        } else {
            // Write as slab of MNode records
            let config = match slabtastic::WriterConfig::new(512, 4096, u32::MAX, false) {
                Ok(c) => c,
                Err(e) => return error_result(format!("slab config: {}", e), start),
            };
            let mut writer = match slabtastic::SlabWriter::new(&output_path, config) {
                Ok(w) => w,
                Err(e) => return error_result(format!("create slab: {}", e), start),
            };

            let pb = ctx.ui.bar_with_unit(count as u64, "generating metadata", "records");
            for ordinal in 0..count {
                let mut node = MNode::new();
                for name in &field_names {
                    let value = range_min + rng.random_range(0..range);
                    node.fields.insert(name.clone(), MValue::Int32(value));
                }
                let bytes = node.to_bytes();
                if let Err(e) = writer.add_record(&bytes) {
                    return error_result(format!("write record {}: {}", ordinal, e), start);
                }
                if (ordinal + 1) % 10_000 == 0 {
                    pb.set_position((ordinal + 1) as u64);
                }
            }
            pb.finish();

            // Co-locate the metadata layout (schema) as a `layout` namespace
            // in the same slab. The layout is the records' structural
            // fingerprint (field names + types, values defaulted) — opaque
            // anode bytes here; callers decode with an `anode` impl. Two
            // metadata slabs are schema-compatible iff their `layout`
            // namespaces are byte-identical (see `metadata_layout_bytes`).
            if let Err(e) = writer.start_namespace("layout") {
                return error_result(format!("open layout namespace: {}", e), start);
            }
            if let Err(e) = writer.add_record(&metadata_layout_bytes(&field_names)) {
                return error_result(format!("write layout: {}", e), start);
            }

            if let Err(e) = writer.finish() {
                return error_result(format!("finalize slab: {}", e), start);
            }
        }

        // Write the authoritative metadata layout (schema) to a standalone
        // `metadata_layout.slab` when requested. Its *default* namespace
        // holds the single schema record. Unlike the in-content `layout`
        // namespace (a convenience copy that slab slicing/extract may drop),
        // this standalone file is the slicing-proof home of the schema and
        // backs the optional `metadata_layout` facet. Identical bytes to the
        // embedded copy, so `vectordata`'s byte-for-byte layout compatibility
        // check holds across both.
        let mut produced = vec![output_path.clone()];
        if let Some(layout_str) = options.get("layout-output") {
            let layout_path = resolve_path(layout_str, &ctx.workspace);
            if let Some(parent) = layout_path.parent() {
                if !parent.exists() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            let cfg = match slabtastic::WriterConfig::new(512, 4096, u32::MAX, false) {
                Ok(c) => c,
                Err(e) => return error_result(format!("layout slab config: {}", e), start),
            };
            let mut lw = match slabtastic::SlabWriter::new(&layout_path, cfg) {
                Ok(w) => w,
                Err(e) => return error_result(format!("create layout slab: {}", e), start),
            };
            if let Err(e) = lw.add_record(&metadata_layout_bytes(&field_names)) {
                return error_result(format!("write layout record: {}", e), start);
            }
            if let Err(e) = lw.finish() {
                return error_result(format!("finalize layout slab: {}", e), start);
            }
            ctx.ui.log(&format!("  wrote metadata layout schema to {}", layout_path.display()));
            produced.push(layout_path);
        }

        // Set verified count variable
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &count.to_string());
        ctx.defaults.insert(var_name, count.to_string());

        ctx.ui.log(&format!(
            "  wrote {} metadata records ({} fields each) to {}",
            count, fields, output_path.display(),
        ));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} records, {} fields, range [{}..{}), seed={}",
                count, fields, range_min, range_max, seed,
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "output".into(), type_name: "Path".into(), required: true,
                default: None, description: "Output slab file".into(),
            extended_description: None,
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "layout-output".into(), type_name: "Path".into(), required: false,
                default: None,
                description: "Standalone slab to receive the metadata layout (schema)".into(),
                extended_description: Some(
                    "When set, writes the field schema as a single record in this \
                     slab's default namespace — the authoritative, slicing-proof home \
                     of the metadata layout, backing the optional `metadata_layout` \
                     facet. Byte-identical to the `layout` namespace embedded in slab \
                     content.".into(),
                ),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "count".into(), type_name: "int".into(), required: true,
                default: None, description: "Number of metadata records to generate".into(),
            extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "fields".into(), type_name: "int".into(), required: false,
                default: Some("3".into()),
                description: "Number of integer fields per record".into(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "range-min".into(), type_name: "int".into(), required: false,
                default: Some("0".into()),
                description: "Minimum value (inclusive) for uniform integer distribution".into(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "range-max".into(), type_name: "int".into(), required: false,
                default: Some("1000".into()),
                description: "Maximum value (exclusive) for uniform integer distribution".into(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "seed".into(), type_name: "int".into(), required: false,
                default: Some("42".into()),
                description: "Random seed for reproducibility".into(),
                extended_description: None,
                role: OptionRole::Config,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &[],
            &["output", "layout-output"],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use crate::pipeline::command::{Options, Status, StreamContext};
    use crate::pipeline::progress::ProgressLog;

    fn make_ctx(workspace: &std::path::Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            cache: workspace.join(".cache"),
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

    #[test]
    fn test_generate_metadata_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let out = tmp.path().join("meta.slab");
        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("count", "100");
        opts.set("fields", "3");
        opts.set("range-min", "0");
        opts.set("range-max", "50");
        opts.set("seed", "42");

        let mut op = GenerateMetadataOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "generate metadata: {}", r.message);
        assert!(out.exists());

        // Verify we can read back the slab and decode MNode records
        let reader = slabtastic::SlabReader::open(&out).unwrap();
        let mut count = 0;
        for ordinal in 0..100i64 {
            let data = reader.get(ordinal).unwrap();
            let node = MNode::from_bytes(&data).unwrap();
            assert_eq!(node.fields.len(), 3, "should have 3 fields");
            assert!(node.fields.contains_key("field_0"));
            assert!(node.fields.contains_key("field_1"));
            assert!(node.fields.contains_key("field_2"));
            for (_name, value) in &node.fields {
                match value {
                    MValue::Int32(v) => {
                        assert!(*v >= 0 && *v < 50, "value {} out of range [0, 50)", v);
                    }
                    other => panic!("expected Int32, got {:?}", other),
                }
            }
            count += 1;
        }
        assert_eq!(count, 100, "should have 100 records");
    }

    #[test]
    fn test_generate_metadata_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let out1 = tmp.path().join("meta1.slab");
        let out2 = tmp.path().join("meta2.slab");

        for out in [&out1, &out2] {
            let mut opts = Options::new();
            opts.set("output", out.to_string_lossy().to_string());
            opts.set("count", "50");
            opts.set("seed", "99");
            let mut op = GenerateMetadataOp;
            op.execute(&opts, &mut ctx);
        }

        let data1 = std::fs::read(&out1).unwrap();
        let data2 = std::fs::read(&out2).unwrap();
        assert_eq!(data1, data2, "same seed should produce identical output");
    }

    #[test]
    fn metadata_layout_namespace_and_schema_compat() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());
        let out = tmp.path().join("metadata_content.slab");
        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("count", "20");
        opts.set("fields", "3");
        opts.set("range-min", "0");
        opts.set("range-max", "10");
        let mut op = GenerateMetadataOp;
        assert_eq!(op.execute(&opts, &mut ctx).status, Status::Ok);

        // The content lives in the default namespace; the schema is a
        // co-located `layout` namespace, byte-equal to the canonical bytes.
        let field_names: Vec<String> = (0..3).map(|i| format!("field_{i}")).collect();
        let layout = slabtastic::SlabReader::open_namespace(&out, Some(LAYOUT_NAMESPACE)).unwrap();
        assert_eq!(
            layout.get(0).unwrap(),
            metadata_layout_bytes(&field_names),
            "the `layout` namespace must hold the canonical schema bytes",
        );

        // Compatibility = byte-for-byte schema match: same field structure
        // → identical bytes; a different field count → different bytes.
        let four: Vec<String> = (0..4).map(|i| format!("field_{i}")).collect();
        assert_eq!(metadata_layout_bytes(&field_names), metadata_layout_bytes(&field_names.clone()));
        assert_ne!(metadata_layout_bytes(&field_names), metadata_layout_bytes(&four));

        // The default namespace still holds the content records, intact.
        let content = slabtastic::SlabReader::open(&out).unwrap();
        let node = MNode::from_bytes(&content.get(0).unwrap()).unwrap();
        assert_eq!(node.fields.len(), 3);
    }

    /// `layout-output` writes a standalone `metadata_layout.slab` whose
    /// default namespace carries the schema, byte-identical to the embedded
    /// `layout` namespace copy — so a content slab and the standalone are
    /// compatible under `vectordata`'s byte-for-byte layout check.
    #[test]
    fn standalone_layout_output_is_byte_identical_to_embedded() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());
        let content = tmp.path().join("metadata_content.slab");
        let layout = tmp.path().join("metadata_layout.slab");
        let mut opts = Options::new();
        opts.set("output", content.to_string_lossy().to_string());
        opts.set("layout-output", layout.to_string_lossy().to_string());
        opts.set("count", "20");
        opts.set("fields", "3");
        let mut op = GenerateMetadataOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        // Both outputs are reported as produced.
        assert!(result.produced.contains(&layout), "layout slab must be a produced artifact");

        // Standalone: schema in the *default* namespace, record 0.
        let standalone = slabtastic::SlabReader::open(&layout).unwrap();
        let field_names: Vec<String> = (0..3).map(|i| format!("field_{i}")).collect();
        assert_eq!(standalone.get(0).unwrap(), metadata_layout_bytes(&field_names));

        // Byte-identical to the embedded `layout` namespace copy.
        let embedded = slabtastic::SlabReader::open_namespace(&content, Some(LAYOUT_NAMESPACE)).unwrap();
        assert_eq!(standalone.get(0).unwrap(), embedded.get(0).unwrap());
    }

    /// The standalone layout is written even when the content format is a
    /// flat scalar (no embedded namespace) — the schema is format-agnostic.
    #[test]
    fn standalone_layout_output_for_scalar_content() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());
        let content = tmp.path().join("metadata_content.u8");
        let layout = tmp.path().join("metadata_layout.slab");
        let mut opts = Options::new();
        opts.set("output", content.to_string_lossy().to_string());
        opts.set("layout-output", layout.to_string_lossy().to_string());
        opts.set("count", "16");
        opts.set("fields", "2");
        opts.set("format", "u8");
        let mut op = GenerateMetadataOp;
        assert_eq!(op.execute(&opts, &mut ctx).status, Status::Ok);

        let standalone = slabtastic::SlabReader::open(&layout).unwrap();
        let field_names: Vec<String> = (0..2).map(|i| format!("field_{i}")).collect();
        assert_eq!(standalone.get(0).unwrap(), metadata_layout_bytes(&field_names));
    }

    #[test]
    fn test_generate_metadata_invalid_range() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let mut opts = Options::new();
        opts.set("output", tmp.path().join("bad.slab").to_string_lossy().to_string());
        opts.set("count", "10");
        opts.set("range-min", "100");
        opts.set("range-max", "50");

        let mut op = GenerateMetadataOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Error, "invalid range should fail");
    }

    #[test]
    fn test_generate_metadata_single_field() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let out = tmp.path().join("single.slab");
        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("count", "5");
        opts.set("fields", "1");

        let mut op = GenerateMetadataOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "single field: {}", r.message);
    }
}
