// Copyright (c) Jonathan Shook
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

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let ordinal_str = match options.require("range") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };

        let format = options.get("format").unwrap_or("text");
        let input_path = resolve_path(input_str, &ctx.workspace);

        let etype = match ElementType::from_path(&input_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        // Detect scalar format (no xvec header — flat packed values)
        let ext = input_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let is_scalar = veks_core::formats::VecFormat::from_extension(ext)
            .map(|f| f.is_scalar())
            .unwrap_or(false);

        // Open reader and extract (count, dim, get_f64_fn) based on element type.
        let (count, _dim, get_f64): (usize, usize, Box<dyn Fn(usize) -> Vec<f64> + Sync>) = if is_scalar {
            // Scalar file: each record is a single element, no dim header
            let file_size = std::fs::metadata(&input_path).map(|m| m.len()).unwrap_or(0);
            let elem_size = etype.element_size();
            let rc = if elem_size > 0 { file_size as usize / elem_size } else { 0 };
            let path_clone = input_path.clone();
            (rc, 1, Box::new(move |i: usize| {
                use std::io::{Read, Seek, SeekFrom};
                let elem_size = match ext {
                    "u8" | "i8" => 1, "u16" | "i16" => 2,
                    "u32" | "i32" => 4, "u64" | "i64" => 8,
                    _ => 1,
                };
                let mut f = match std::fs::File::open(&path_clone) {
                    Ok(f) => f, Err(_) => return vec![],
                };
                let _ = f.seek(SeekFrom::Start(i as u64 * elem_size as u64));
                let mut buf = [0u8; 8];
                let _ = f.read_exact(&mut buf[..elem_size]);
                let val: f64 = match elem_size {
                    1 => buf[0] as f64,
                    2 => i16::from_le_bytes([buf[0], buf[1]]) as f64,
                    4 => i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f64,
                    8 => i64::from_le_bytes(buf) as f64,
                    _ => 0.0,
                };
                vec![val]
            }) as Box<dyn Fn(usize) -> Vec<f64> + Sync>)
        } else { match etype {
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
                match MmapVectorReader::<i32>::open_ivec(&input_path) {
                    Ok(r) => {
                        // Uniform-dimension ivec
                        let fc = VectorReader::<i32>::count(&r);
                        let d = VectorReader::<i32>::dim(&r);
                        (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
                    }
                    Err(vectordata::io::IoError::VariableLengthRecords(_)) => {
                        // Variable-length ivec — use indexed reader
                        let r = match vectordata::io::IndexedXvecReader::open_ivec(&input_path) {
                            Ok(r) => r,
                            Err(e) => return error_result(format!("open indexed ivec: {}", e), start),
                        };
                        let rc = r.count();
                        (rc, 0, Box::new(move |i: usize| {
                            r.get_i32(i).unwrap_or_default().iter().map(|&v| v as f64).collect()
                        }) as Box<dyn Fn(usize) -> Vec<f64> + Sync>)
                    }
                    Err(e) => return error_result(format!("open: {}", e), start),
                }
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
            ElementType::U16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                let d = VectorReader::<i16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U32 => {
                let r = match MmapVectorReader::<i32>::open_ivec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i32>::count(&r);
                let d = VectorReader::<i32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U64 | ElementType::I64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&input_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
        } }; // close match etype + else

        // Parse ordinal spec: single number, comma-separated, or range [start,end)
        let ordinals = parse_ordinal_spec(&ordinal_str, count);
        if ordinals.is_err() {
            return error_result(ordinals.unwrap_err(), start);
        }
        let ordinals = ordinals.unwrap();

        if ordinals.is_empty() {
            return error_result("no ordinals in range".into(), start);
        }

        let is_integer = matches!(etype, ElementType::I32 | ElementType::I16 | ElementType::U8 | ElementType::I8
            | ElementType::U16 | ElementType::U32 | ElementType::U64 | ElementType::I64);

        for &ordinal in &ordinals {
            let vec = get_f64(ordinal);
            // Use actual record length for type label (matters for variable-length)
            let type_name = format!("{}[{}]", etype, vec.len());
            let output = format_vector(&type_name, &vec, format, ordinal, is_integer);
            ctx.ui.log(&output);
        }

        CommandResult {
            status: Status::Ok,
            message: format!("selected {} {} vectors", ordinals.len(), etype),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Vector file to read from".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "range".to_string(),
                type_name: "string".to_string(),
                required: true,
                default: None,
                description: "Ordinal(s): single (42), range ([0,10) or 0..10 or 0-9), or comma-separated (0,1,2)".to_string(),
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

/// Parse an ordinal spec into a list of ordinals.
///
/// Supports: single number (`42`), comma-separated (`0,1,2`),
/// range syntax (`[0,10)`, `0..10`, `5-9`).
fn parse_ordinal_spec(spec: &str, count: usize) -> Result<Vec<usize>, String> {
    let spec = spec.trim();

    // Range: [start,end) or [start,end]
    if spec.starts_with('[') {
        let inner = spec.trim_start_matches('[').trim_end_matches(')').trim_end_matches(']');
        let exclusive = spec.ends_with(')');
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 2 {
            let start: usize = parts[0].trim().parse()
                .map_err(|_| format!("invalid range start: '{}'", parts[0]))?;
            let end: usize = parts[1].trim().parse()
                .map_err(|_| format!("invalid range end: '{}'", parts[1]))?;
            let end = if exclusive { end } else { end + 1 };
            let end = end.min(count);
            if start >= end { return Err(format!("empty range [{}, {})", start, end)); }
            return Ok((start..end).collect());
        }
    }

    // Range: start..end
    if spec.contains("..") {
        let parts: Vec<&str> = spec.split("..").collect();
        if parts.len() == 2 {
            let start: usize = parts[0].trim().parse()
                .map_err(|_| format!("invalid range start: '{}'", parts[0]))?;
            let end: usize = parts[1].trim().parse()
                .map_err(|_| format!("invalid range end: '{}'", parts[1]))?;
            let end = end.min(count);
            if start >= end { return Err(format!("empty range {}..{}", start, end)); }
            return Ok((start..end).collect());
        }
    }

    // Range: start-end (only if both parts are numeric)
    if spec.contains('-') && !spec.starts_with('-') {
        let parts: Vec<&str> = spec.splitn(2, '-').collect();
        if parts.len() == 2 {
            if let (Ok(start), Ok(end)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                let end = (end + 1).min(count);
                if start >= end { return Err(format!("empty range {}-{}", start, end - 1)); }
                return Ok((start..end).collect());
            }
        }
    }

    // Comma-separated
    if spec.contains(',') {
        let mut ordinals = Vec::new();
        for part in spec.split(',') {
            let ord: usize = part.trim().parse()
                .map_err(|_| format!("invalid ordinal: '{}'", part.trim()))?;
            if ord >= count {
                return Err(format!("ordinal {} out of range (file has {} vectors)", ord, count));
            }
            ordinals.push(ord);
        }
        return Ok(ordinals);
    }

    // Single ordinal
    let ord: usize = spec.parse()
        .map_err(|_| format!("invalid ordinal: '{}'", spec))?;
    if ord >= count {
        return Err(format!("ordinal {} out of range (file has {} vectors)", ord, count));
    }
    Ok(vec![ord])
}

fn format_vector(type_name: &str, values: &[f64], format: &str, ordinal: usize, is_integer: bool) -> String {
    let fmt_val = |v: &f64| -> String {
        if is_integer { format!("{}", *v as i64) } else { format!("{}", v) }
    };

    match format {
        "json" => {
            let vals: Vec<String> = values.iter().map(fmt_val).collect();
            format!(
                "{{\"ordinal\":{},\"type\":\"{}\",\"dim\":{},\"values\":[{}]}}",
                ordinal, type_name, values.len(), vals.join(",")
            )
        }
        "csv" => {
            let vals: Vec<String> = values.iter().map(fmt_val).collect();
            format!("{},{}", ordinal, vals.join(","))
        }
        _ => {
            let vals: Vec<String> = values.iter().map(fmt_val).collect();
            format!("[{}] {}", ordinal, vals.join(", "))
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
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
        opts.set("source", path.to_string_lossy().to_string());
        opts.set("range", "5");
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
        opts.set("source", path.to_string_lossy().to_string());
        opts.set("range", "10");
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
        opts.set("source", path.to_string_lossy().to_string());
        opts.set("range", "0");
        opts.set("format", "json");
        let mut op = AnalyzeSelectOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_format_vector_text() {
        let values = vec![1.0, 2.5, 3.7];
        let output = format_vector("float[]", &values, "text", 0, false);
        assert!(output.contains("1, 2.5, 3.7"));
    }

    #[test]
    fn test_format_vector_csv() {
        let values = vec![1.0, 2.0, 3.0];
        let output = format_vector("int[]", &values, "csv", 0, true);
        assert_eq!(output, "0,1,2,3");
    }

    #[test]
    fn test_format_vector_json() {
        let values = vec![42.0];
        let output = format_vector("i32[1]", &values, "json", 5, true);
        assert!(output.contains("\"ordinal\":5"));
        assert!(output.contains("[42]"));
    }
}
