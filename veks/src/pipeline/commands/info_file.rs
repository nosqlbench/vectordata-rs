// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: inspect vector file metadata.
//!
//! Reads a vector file header and reports format, dimensions, vector count,
//! file size, element type, and optional sample vectors.
//!
//! Supports xvec formats: fvec (f32), ivec (i32), bvec (u8), dvec (f64), mvec (f16).
//!
//! Equivalent to the Java `CMD_info_file` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: display vector file info.
pub struct InfoFileOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(InfoFileOp)
}

/// Detected file format info.
struct FileInfo {
    format: String,
    data_type: String,
    bytes_per_element: usize,
    dimensions: usize,
    vector_count: usize,
    bytes_per_vector: usize,
    file_size: u64,
    trailing_bytes: u64,
}

/// Detect format from extension and read header.
fn probe_file(path: &Path) -> Result<FileInfo, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let (data_type, bytes_per_element) = match ext.as_str() {
        "fvec" => ("float32", 4),
        "ivec" => ("int32", 4),
        "bvec" => ("uint8", 1),
        "dvec" => ("float64", 8),
        "mvec" => ("float16", 2),
        _ => return Err(format!("unknown vector format: '.{}'", ext)),
    };

    let file_size = std::fs::metadata(path)
        .map_err(|e| format!("failed to stat {}: {}", path.display(), e))?
        .len();

    if file_size < 4 {
        return Err(format!("file too small ({} bytes)", file_size));
    }

    // Read first 4 bytes to get dimension count
    let header_bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

    let dimensions = i32::from_le_bytes([
        header_bytes[0],
        header_bytes[1],
        header_bytes[2],
        header_bytes[3],
    ]) as usize;

    if dimensions == 0 {
        return Err("dimension count is 0".to_string());
    }

    let bytes_per_vector = 4 + dimensions * bytes_per_element;
    let vector_count = file_size as usize / bytes_per_vector;
    let trailing_bytes = file_size - (vector_count * bytes_per_vector) as u64;

    Ok(FileInfo {
        format: ext,
        data_type: data_type.to_string(),
        bytes_per_element,
        dimensions,
        vector_count,
        bytes_per_vector,
        file_size,
        trailing_bytes,
    })
}

/// Format bytes as human-readable string.
fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    for unit in UNITS {
        if size < 1024.0 {
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1} PB", size)
}

/// Read sample vectors as formatted strings.
fn read_samples(
    data: &[u8],
    info: &FileInfo,
    count: usize,
    max_dims_shown: usize,
) -> Vec<String> {
    let mut samples = Vec::new();
    let actual_count = count.min(info.vector_count);

    for i in 0..actual_count {
        let offset = i * info.bytes_per_vector + 4; // skip dim header
        let dims_to_show = info.dimensions.min(max_dims_shown);
        let mut values = Vec::new();

        for d in 0..dims_to_show {
            let pos = offset + d * info.bytes_per_element;
            let val = match info.format.as_str() {
                "fvec" => {
                    let v = f32::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                    ]);
                    format!("{:.4}", v)
                }
                "ivec" => {
                    let v = i32::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                    ]);
                    format!("{}", v)
                }
                "bvec" => format!("{}", data[pos]),
                "dvec" => {
                    let v = f64::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                        data[pos + 4],
                        data[pos + 5],
                        data[pos + 6],
                        data[pos + 7],
                    ]);
                    format!("{:.6}", v)
                }
                "mvec" => {
                    let bits = u16::from_le_bytes([data[pos], data[pos + 1]]);
                    format!("0x{:04x}", bits)
                }
                _ => "?".to_string(),
            };
            values.push(val);
        }

        let suffix = if info.dimensions > max_dims_shown {
            format!(", ... ({} more)", info.dimensions - max_dims_shown)
        } else {
            String::new()
        };

        samples.push(format!("  [{}]: [{}{}]", i, values.join(", "), suffix));
    }

    samples
}

impl CommandOp for InfoFileOp {
    fn command_path(&self) -> &str {
        "info file"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Display file format, dimensions, and record count".into(),
            body: format!(
                r#"# info file

Display file format, dimensions, and record count.

## Description

Reads a vector file header and reports format, dimensions, vector count,
file size, element type, per-vector byte overhead, and optional sample
vectors. Supports all xvec formats: fvec (float32), ivec (int32),
bvec (uint8), dvec (float64), and mvec (float16).

## How It Works

The command detects the file format from the extension, reads the first
four bytes to obtain the dimension count, then computes the per-vector
byte size (4-byte header + dimensions * element size). The total vector
count is derived by dividing the file size by the per-vector size.
Trailing bytes (file size not evenly divisible) are reported as a
warning indicating possible truncation or corruption. When samples are
requested, the command reads the first N vectors and displays their
component values (up to 8 dimensions shown, with an ellipsis for
higher-dimensional vectors).

## Data Preparation Role

`info file` provides quick validation of vector file integrity and
is often the first command run on a newly downloaded or generated
vector file. It confirms that the file has the expected format,
dimension, and record count before expensive downstream operations
like index building or KNN computation begin. The trailing-bytes
check catches truncated downloads or failed writes that would cause
subtle errors in later pipeline steps.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let sample_count: usize = options
            .get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let source_path = resolve_path(source_str, &ctx.workspace);

        let info = match probe_file(&source_path) {
            Ok(i) => i,
            Err(e) => return error_result(e, start),
        };

        let header_overhead =
            (4 * info.vector_count) as f64 / info.file_size as f64 * 100.0;

        ctx.ui.log(&format!("File: {}", source_path.display()));
        ctx.ui.log(&format!("  Size:       {} ({} bytes)", human_bytes(info.file_size), info.file_size));
        ctx.ui.log(&format!("  Format:     .{} ({})", info.format, info.data_type));
        ctx.ui.log(&format!(
            "  Element:    {} bytes per value",
            info.bytes_per_element
        ));
        ctx.ui.log(&format!("  Dimensions: {}", info.dimensions));
        ctx.ui.log(&format!("  Vectors:    {}", info.vector_count));
        ctx.ui.log(&format!(
            "  Per-vector: {} bytes (4 header + {} data)",
            info.bytes_per_vector,
            info.dimensions * info.bytes_per_element
        ));
        ctx.ui.log(&format!("  Header overhead: {:.1}%", header_overhead));

        if info.trailing_bytes > 0 {
            ctx.ui.log(&format!(
                "  WARNING: {} trailing bytes (file may be truncated or corrupted)",
                info.trailing_bytes
            ));
        }

        if sample_count > 0 {
            let data = std::fs::read(&source_path).unwrap_or_default();
            let samples = read_samples(&data, &info, sample_count, 8);
            ctx.ui.log("  Sample vectors:");
            for s in &samples {
                ctx.ui.log(&format!("{}", s));
            }
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{}: {} .{} vectors (dim={})",
                source_path.display(),
                info.vector_count,
                info.format,
                info.dimensions
            ),
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
                description: "Vector file to inspect".to_string(),
            },
            OptionDesc {
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Number of sample vectors to display".to_string(),
            },
        ]
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
    fn test_info_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("test.fvec");
        let mut opts = Options::new();
        opts.set("output", source.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("count", "100");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("sample", "3");

        let mut info_op = InfoFileOp;
        let result = info_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("100"));
        assert!(result.message.contains("dim=8"));
    }

    #[test]
    fn test_info_ivec() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Write a manual ivec file: 5 records of dim=3
        let source = ws.join("test.ivec");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&source).unwrap();
            for i in 0..5 {
                f.write_all(&3i32.to_le_bytes()).unwrap();
                for j in 0..3 {
                    f.write_all(&(i * 3 + j as i32).to_le_bytes()).unwrap();
                }
            }
        }

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());

        let mut info_op = InfoFileOp;
        let result = info_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("ivec"));
        assert!(result.message.contains("5"));
    }

    #[test]
    fn test_probe_file_format() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        // Write a minimal fvec: 2 vectors of dim 3
        let path = ws.join("test.fvec");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).unwrap();
            for _ in 0..2 {
                f.write_all(&3i32.to_le_bytes()).unwrap();
                for _ in 0..3 {
                    f.write_all(&1.0f32.to_le_bytes()).unwrap();
                }
            }
        }

        let info = probe_file(&path).unwrap();
        assert_eq!(info.dimensions, 3);
        assert_eq!(info.vector_count, 2);
        assert_eq!(info.bytes_per_element, 4);
        assert_eq!(info.trailing_bytes, 0);
    }

    #[test]
    fn test_probe_truncated_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("trunc.fvec");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).unwrap();
            // Write 1 complete vector (dim=2) + 3 extra bytes
            f.write_all(&2i32.to_le_bytes()).unwrap();
            f.write_all(&1.0f32.to_le_bytes()).unwrap();
            f.write_all(&2.0f32.to_le_bytes()).unwrap();
            f.write_all(&[0, 0, 0]).unwrap(); // trailing
        }

        let info = probe_file(&path).unwrap();
        assert_eq!(info.vector_count, 1);
        assert_eq!(info.trailing_bytes, 3);
    }

    #[test]
    fn test_human_bytes() {
        assert_eq!(human_bytes(0), "0.0 B");
        assert_eq!(human_bytes(1023), "1023.0 B");
        assert_eq!(human_bytes(1024), "1.0 KB");
        assert_eq!(human_bytes(1048576), "1.0 MB");
    }

    #[test]
    fn test_unknown_format() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.xyz");
        std::fs::write(&path, b"hello").unwrap();
        assert!(probe_file(&path).is_err());
    }
}
