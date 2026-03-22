// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: select and display a single vector by ordinal.
//!
//! Retrieves a vector at a specific 0-based index from a vector file and
//! prints its type and values.
//!
//! Equivalent to the Java `CMD_analyze_select` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::element_type::ElementType;

/// Pipeline command: select a vector by ordinal.
pub struct AnalyzeSelectOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeSelectOp)
}

impl CommandOp for AnalyzeSelectOp {
    fn command_path(&self) -> &str {
        "analyze select"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Select and print specific vectors by ordinal".into(),
            body: format!(
                "# analyze select\n\n\
                Select and print specific vectors by ordinal.\n\n\
                ## Description\n\n\
                Retrieves a single vector at a specific 0-based index from a vector \
                file and prints its type, dimension count, and component values. The \
                command uses memory-mapped I/O for random access, so retrieving a vector \
                from the middle of a multi-gigabyte file is instantaneous -- no sequential \
                scan is required.\n\n\
                ## Output Formats\n\n\
                - **text** (default): Human-readable format showing type, ordinal, \
                dimension, and bracketed value list.\n\
                - **csv**: Comma-separated values suitable for spreadsheet import.\n\
                - **json**: JSON object with ordinal, type, dim, and values fields.\n\n\
                ## Role in Dataset Preparation\n\n\
                This command is used for spot-checking specific records during debugging. \
                For example, after a shuffle operation, you might select the vector that \
                was originally at index 0 to verify it was moved to the expected new \
                position. It is also useful for inspecting the query vectors or ground-truth \
                neighbor indices at particular ordinals when investigating KNN verification \
                failures. Supports both fvec (float) and ivec (integer) file formats.\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Vector data buffers".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let ordinal: usize = match options.require("ordinal") {
            Ok(s) => match s.parse() {
                Ok(n) => n,
                _ => return error_result(format!("invalid ordinal: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let format = options.get("format").unwrap_or("text");
        let input_path = resolve_path(input_str, &ctx.workspace);

        let etype = match ElementType::from_path(&input_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        // Open reader and extract (count, dim, get_f64_fn) based on element type.
        let (count, dim, get_f64): (usize, usize, Box<dyn Fn(usize) -> Vec<f64> + Sync>) = match etype {
            ElementType::F32 => {
                let r = match MmapVectorReader::<f32>::open_fvec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f32>::count(&r);
                let d = VectorReader::<f32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::F16 => {
                let r = match MmapVectorReader::<half::f16>::open_mvec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<half::f16>::count(&r);
                let d = VectorReader::<half::f16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|v| v.to_f64()).collect()))
            }
            ElementType::F64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default()))
            }
            ElementType::I32 => {
                let r = match MmapVectorReader::<i32>::open_ivec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i32>::count(&r);
                let d = VectorReader::<i32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::I16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                let d = VectorReader::<i16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U8 | ElementType::I8 => {
                let r = match MmapVectorReader::<u8>::open_bvec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<u8>::count(&r);
                let d = VectorReader::<u8>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
        };

        if ordinal >= count {
            return error_result(
                format!("ordinal {} out of range (file has {} vectors)", ordinal, count),
                start,
            );
        }

        let vec = get_f64(ordinal);
        let type_name = format!("{}[{}]", etype, dim);
        let output = format_vector(&type_name, &vec, format, ordinal);
        log::info!("{}", output);

        CommandResult {
            status: Status::Ok,
            message: format!("selected {} vector at ordinal {}", etype, ordinal),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "input".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Vector file to read from".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "ordinal".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "0-based index of the vector to retrieve".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "format".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("text".to_string()),
                description: "Output format: text, csv, json".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

fn format_vector<T: std::fmt::Display>(type_name: &str, values: &[T], format: &str, ordinal: usize) -> String {
    match format {
        "json" => {
            let vals: Vec<String> = values.iter().map(|v| format!("{}", v)).collect();
            format!(
                "{{\"ordinal\":{},\"type\":\"{}\",\"dim\":{},\"values\":[{}]}}",
                ordinal,
                type_name,
                values.len(),
                vals.join(",")
            )
        }
        "csv" => {
            let vals: Vec<String> = values.iter().map(|v| format!("{}", v)).collect();
            vals.join(",")
        }
        _ => {
            // text format
            let vals: Vec<String> = values.iter().map(|v| format!("{}", v)).collect();
            format!(
                "Vector Data:\n  type: {}\n  ordinal: {}\n  dim: {}\n  values: [{}]",
                type_name,
                ordinal,
                values.len(),
                vals.join(", ")
            )
        }
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    #[test]
    fn test_select_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate vectors
        let path = ws.join("test.fvec");
        let mut opts = Options::new();
        opts.set("output", path.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "10");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        // Select vector at ordinal 5
        let mut opts = Options::new();
        opts.set("input", path.to_string_lossy().to_string());
        opts.set("ordinal", "5");
        let mut op = AnalyzeSelectOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_select_out_of_range() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let path = ws.join("test.fvec");
        let mut opts = Options::new();
        opts.set("output", path.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "5");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("input", path.to_string_lossy().to_string());
        opts.set("ordinal", "10");
        let mut op = AnalyzeSelectOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("out of range"));
    }

    #[test]
    fn test_select_json_format() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let path = ws.join("test.fvec");
        let mut opts = Options::new();
        opts.set("output", path.to_string_lossy().to_string());
        opts.set("dimension", "3");
        opts.set("count", "5");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("input", path.to_string_lossy().to_string());
        opts.set("ordinal", "0");
        opts.set("format", "json");
        let mut op = AnalyzeSelectOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_format_vector_text() {
        let values = vec![1.0f32, 2.5, 3.7];
        let output = format_vector("float[]", &values, "text", 0);
        assert!(output.contains("float[]"));
        assert!(output.contains("dim: 3"));
    }

    #[test]
    fn test_format_vector_csv() {
        let values = vec![1, 2, 3];
        let output = format_vector("int[]", &values, "csv", 0);
        assert_eq!(output, "1,2,3");
    }
}
