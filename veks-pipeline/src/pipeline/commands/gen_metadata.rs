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
            let field_names: Vec<String> = (0..fields)
                .map(|i| format!("field_{}", i))
                .collect();

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

            if let Err(e) = writer.finish() {
                return error_result(format!("finalize slab: {}", e), start);
            }
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
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "output".into(), type_name: "Path".into(), required: true,
                default: None, description: "Output slab file".into(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "count".into(), type_name: "int".into(), required: true,
                default: None, description: "Number of metadata records to generate".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "fields".into(), type_name: "int".into(), required: false,
                default: Some("3".into()),
                description: "Number of integer fields per record".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "range-min".into(), type_name: "int".into(), required: false,
                default: Some("0".into()),
                description: "Minimum value (inclusive) for uniform integer distribution".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "range-max".into(), type_name: "int".into(), required: false,
                default: Some("1000".into()),
                description: "Maximum value (exclusive) for uniform integer distribution".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "seed".into(), type_name: "int".into(), required: false,
                default: Some("42".into()),
                description: "Random seed for reproducibility".into(),
                role: OptionRole::Config,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &[],
            &["output"],
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
            scratch: workspace.join(".scratch"),
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
