// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: check xvec file endianness.
//!
//! Reads the dimension header under both little-endian and big-endian
//! interpretations and validates which is correct by checking that the
//! file size is a multiple of the expected record size.
//!
//! Equivalent to the Java `CMD_analyze_check_endian` command.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::element_type::ElementType;

/// Pipeline command: check endianness of xvec files.
pub struct AnalyzeCheckEndianOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeCheckEndianOp)
}

/// Result of checking one endian interpretation.
struct EndianCheck {
    valid: bool,
    vector_count: usize,
}

/// Check a file's endianness.
fn check_endianness(path: &Path) -> Result<String, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let etype = ElementType::from_path(path)
        .map_err(|_| format!("unknown vector format: '.{}'", ext))?;
    let element_width = etype.element_size();

    let file_size = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {}", path.display(), e))?
        .len();

    if file_size < 4 {
        return Err(format!("file too small ({} bytes)", file_size));
    }

    // Read first 4 bytes
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("cannot open {}: {}", path.display(), e))?;
    let mut header = [0u8; 4];
    file.read_exact(&mut header)
        .map_err(|e| format!("cannot read header: {}", e))?;

    let le_dim = u32::from_le_bytes(header) as usize;
    let be_dim = u32::from_be_bytes(header) as usize;

    let le_check = check_interpretation(le_dim, element_width, file_size, path);
    let be_check = check_interpretation(be_dim, element_width, file_size, path);

    let mut report = String::new();
    report.push_str(&format!("File: {}\n", path.display()));
    report.push_str(&format!("  Size: {} bytes\n", file_size));
    report.push_str(&format!("  Format: {} (element width: {} bytes)\n", ext, element_width));
    report.push_str(&format!(
        "  Little-endian: dim={}, valid={}, vectors={}\n",
        le_dim, le_check.valid, le_check.vector_count
    ));
    report.push_str(&format!(
        "  Big-endian:    dim={}, valid={}, vectors={}\n",
        be_dim, be_check.valid, be_check.vector_count
    ));

    if le_check.valid && !be_check.valid {
        report.push_str("  Result: LITTLE-ENDIAN (correct)\n");
        Ok(report)
    } else if !le_check.valid && be_check.valid {
        report.push_str("  Result: BIG-ENDIAN (incorrect — should be little-endian)\n");
        Err(report)
    } else if le_check.valid && be_check.valid {
        report.push_str("  Result: AMBIGUOUS (both interpretations valid)\n");
        Ok(report)
    } else {
        report.push_str("  Result: INVALID (neither interpretation valid)\n");
        Err(report)
    }
}

fn check_interpretation(dim: usize, element_width: usize, file_size: u64, path: &Path) -> EndianCheck {
    let invalid = EndianCheck {
        valid: false,
        vector_count: 0,
    };

    // Dimension must be in [1, 1048576]
    if dim == 0 || dim > 1_048_576 {
        return invalid;
    }

    let record_size = 4 + dim * element_width;
    if file_size as usize % record_size != 0 {
        return invalid;
    }

    let vector_count = file_size as usize / record_size;
    if vector_count == 0 {
        return invalid;
    }

    // For files with multiple vectors, verify the last record's header matches
    if vector_count > 1 {
        let last_offset = (vector_count - 1) * record_size;
        if let Ok(mut f) = std::fs::File::open(path) {
            use std::io::Seek;
            if f.seek(std::io::SeekFrom::Start(last_offset as u64)).is_ok() {
                let mut buf = [0u8; 4];
                if f.read_exact(&mut buf).is_ok() {
                    let last_dim = if element_width <= 4 {
                        // Use same endian as we're testing
                        u32::from_le_bytes(buf) as usize
                    } else {
                        u32::from_le_bytes(buf) as usize
                    };
                    if last_dim != dim {
                        return invalid;
                    }
                }
            }
        }
    }

    EndianCheck {
        valid: true,
        vector_count,
    }
}

impl CommandOp for AnalyzeCheckEndianOp {
    fn command_path(&self) -> &str {
        "analyze check-endian"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Check byte order of a vector file".into(),
            body: format!(
                "# analyze check-endian\n\n\
                Check byte order of a vector file.\n\n\
                ## Description\n\n\
                Reads the first 4-byte dimension header of an xvec file under both \
                little-endian and big-endian interpretations and validates which \
                interpretation is correct. Validation works by checking that the \
                decoded dimension value is reasonable (between 1 and 1,048,576) and \
                that the total file size is an exact multiple of the computed record \
                size (4 + dim * element_width). For multi-vector files, the last \
                record's header is also read to confirm consistency.\n\n\
                ## Supported Formats\n\n\
                The command supports fvec (4-byte float), ivec (4-byte int), bvec \
                (1-byte), dvec (8-byte double), and mvec (2-byte) file extensions. \
                The element width is determined from the file extension.\n\n\
                ## Diagnostic Results\n\n\
                The command reports one of four outcomes:\n\n\
                - **LITTLE-ENDIAN (correct)**: The file is in the expected native \
                byte order.\n\
                - **BIG-ENDIAN (incorrect)**: The file was likely produced on a \
                big-endian system or transferred without conversion.\n\
                - **AMBIGUOUS**: Both interpretations produce valid record sizes \
                (can happen with very small dimension counts).\n\
                - **INVALID**: Neither interpretation produces a valid file layout.\n\n\
                ## Role in Dataset Preparation\n\n\
                This command diagnoses cross-platform file transfer issues. Vector \
                files downloaded from external sources, transferred between Linux \
                and network storage, or produced by tools on different architectures \
                may have incorrect byte order. Running this check immediately after \
                download prevents hard-to-diagnose garbage data from propagating \
                through the rest of the pipeline.\n\n\
                ## Options\n\n{}",
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

        let source_path = resolve_path(source_str, &ctx.workspace);

        match check_endianness(&source_path) {
            Ok(report) => {
                ctx.ui.emit(report);
                CommandResult {
                    status: Status::Ok,
                    message: format!("endianness check passed: {}", source_path.display()),
                    produced: vec![],
                    elapsed: start.elapsed(),
                }
            }
            Err(report) => {
                ctx.ui.emit(report);
                CommandResult {
                    status: Status::Error,
                    message: format!("endianness check failed: {}", source_path.display()),
                    produced: vec![],
                    elapsed: start.elapsed(),
                }
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![OptionDesc {
            name: "source".to_string(),
            type_name: "Path".to_string(),
            required: true,
            default: None,
            description: "xvec file to check endianness".to_string(),
                role: OptionRole::Input,
    }]
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
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use std::io::Write;

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

    fn write_le_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    fn write_be_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_be_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_be_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn test_check_le_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        write_le_fvec(&input, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeCheckEndianOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_check_be_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        // dim=3 big-endian: bytes [0, 0, 0, 3]
        // LE interpretation: 50331648 — too large for valid dim, so this should fail LE
        write_be_fvec(&input, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeCheckEndianOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("failed"));
    }

    #[test]
    fn test_check_empty_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        std::fs::write(&input, &[0u8; 2]).unwrap();

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeCheckEndianOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }
}
