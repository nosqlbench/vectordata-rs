// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: count zero vectors.
//!
//! Scans an fvec or ivec file and counts vectors where all components are zero.
//! Reports the count and percentage.
//!
//! Equivalent to the Java `CMD_count_zeros` / `CMD_analyze_zeros` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::io::MmapVectorReader;
use vectordata::io::VectorReader;

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

/// Pipeline command: count all-zero vectors.
pub struct AnalyzeZerosOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeZerosOp)
}

impl CommandOp for AnalyzeZerosOp {
    fn command_path(&self) -> &str {
        "analyze zeros"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);

        let ext = source_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "fvec" => count_zeros_fvec(&source_path, start),
            "ivec" => count_zeros_ivec(&source_path, start),
            _ => error_result(
                format!(
                    "unsupported format for zero counting: '.{}' (supported: fvec, ivec)",
                    ext
                ),
                start,
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![OptionDesc {
            name: "source".to_string(),
            type_name: "Path".to_string(),
            required: true,
            default: None,
            description: "Input vector file (fvec or ivec)".to_string(),
        }]
    }
}

fn count_zeros_fvec(path: &Path, start: Instant) -> CommandResult {
    let reader = match MmapVectorReader::<f32>::open_fvec(path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("failed to open {}: {}", path.display(), e), start),
    };

    let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
    let mut zero_count = 0usize;

    for i in 0..count {
        if let Ok(vec) = reader.get(i) {
            if vec.iter().all(|&v| v == 0.0) {
                zero_count += 1;
            }
        }
    }

    let pct = if count > 0 {
        (zero_count as f64 / count as f64) * 100.0
    } else {
        0.0
    };

    eprintln!(
        "{}: {} zero vectors out of {} total ({:.2}%)",
        path.display(),
        zero_count,
        count,
        pct
    );

    let status = if zero_count > 0 {
        Status::Warning
    } else {
        Status::Ok
    };

    CommandResult {
        status,
        message: format!(
            "{} zero vectors out of {} ({:.2}%)",
            zero_count, count, pct
        ),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn count_zeros_ivec(path: &Path, start: Instant) -> CommandResult {
    let reader = match MmapVectorReader::<i32>::open_ivec(path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("failed to open {}: {}", path.display(), e), start),
    };

    let count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
    let mut zero_count = 0usize;

    for i in 0..count {
        if let Ok(vec) = reader.get(i) {
            if vec.iter().all(|&v| v == 0) {
                zero_count += 1;
            }
        }
    }

    let pct = if count > 0 {
        (zero_count as f64 / count as f64) * 100.0
    } else {
        0.0
    };

    eprintln!(
        "{}: {} zero vectors out of {} total ({:.2}%)",
        path.display(),
        zero_count,
        count,
        pct
    );

    let status = if zero_count > 0 {
        Status::Warning
    } else {
        Status::Ok
    };

    CommandResult {
        status,
        message: format!(
            "{} zero vectors out of {} ({:.2}%)",
            zero_count, count, pct
        ),
        produced: vec![],
        elapsed: start.elapsed(),
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
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
        }
    }

    fn write_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    fn write_ivec(path: &Path, vectors: &[Vec<i32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn test_zeros_fvec_no_zeros() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        write_fvec(&input, &[vec![1.0, 2.0], vec![3.0, 4.0]]);

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeZerosOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("0 zero vectors"));
    }

    #[test]
    fn test_zeros_fvec_with_zeros() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        write_fvec(
            &input,
            &[vec![1.0, 2.0], vec![0.0, 0.0], vec![3.0, 0.0], vec![0.0, 0.0]],
        );

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeZerosOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("2 zero vectors"));
    }

    #[test]
    fn test_zeros_ivec() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.ivec");
        write_ivec(&input, &[vec![0, 0, 0], vec![1, 2, 3]]);

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeZerosOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("1 zero vectors"));
    }
}
