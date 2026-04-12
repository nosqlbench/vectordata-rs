// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `slab export` subcommand — export content from a slab file.
//!
//! Supports raw (bytes as stored), text (newline-delimited), cstrings
//! (null-terminated), hex (hex-encoded), and slab format output. Output
//! goes to a file or stdout.

use std::io::{self, Write};
use std::path::Path;

use clap::ValueEnum;

use crate::{SlabReader, SlabWriter};

use super::ordinal_range;
use super::{make_writer_config, write_with_buffer_rename, ProgressReporter};

/// Export output format selectable via `--format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ExportFormatArg {
    /// Write bytes exactly as stored — no added delimiters.
    Raw,
    /// Newline-delimited text.
    Text,
    /// Null-terminated binary.
    Cstrings,
    /// Hex-encoded output, one record per line.
    Hex,
    /// Slabtastic slab format.
    Slab,
}

/// Internal resolved export format (mirrors `ExportFormatArg` but also
/// used when auto-detected from file extension).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    /// Write bytes exactly as stored — no added delimiters.
    Raw,
    /// Newline-delimited text.
    Text,
    /// Null-terminated binary.
    Cstrings,
    /// Hex-encoded output, one record per line.
    Hex,
    /// Slabtastic slab format.
    Slab,
}

impl From<ExportFormatArg> for ExportFormat {
    fn from(arg: ExportFormatArg) -> Self {
        match arg {
            ExportFormatArg::Raw => ExportFormat::Raw,
            ExportFormatArg::Text => ExportFormat::Text,
            ExportFormatArg::Cstrings => ExportFormat::Cstrings,
            ExportFormatArg::Hex => ExportFormat::Hex,
            ExportFormatArg::Slab => ExportFormat::Slab,
        }
    }
}

/// Configuration for the `export` subcommand.
pub struct ExportConfig<'a> {
    /// Input slab file path.
    pub file: &'a str,
    /// Output path; `None` writes to stdout.
    pub output: Option<&'a str>,
    /// Explicitly requested output format, or `None` for auto-detection.
    pub format: Option<ExportFormat>,
    /// Preferred page size for slab output.
    pub preferred_page_size: Option<u32>,
    /// Minimum page size for slab output.
    pub min_page_size: Option<u32>,
    /// Whether to align pages in slab output.
    pub page_alignment: bool,
    /// Whether to report progress on stderr.
    pub progress: bool,
    /// Optional namespace filter.
    pub namespace: &'a Option<String>,
    /// Optional ordinal range filter (e.g., "100", "[5,10)", "0..99").
    pub range: Option<&'a str>,
}

/// Return `true` if `byte` is printable ASCII (0x20–0x7E) or one of
/// `\t`, `\n`, `\r`.
fn is_printable(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\n' | b'\r' | 0x20..=0x7E)
}

/// Return `true` if any byte in `data` is non-printable.
fn has_unprintable(data: &[u8]) -> bool {
    data.iter().any(|&b| !is_printable(b))
}

/// Run the `export` subcommand.
///
/// Exports records from the input file in the specified format. When
/// `range` is set, only records within the specified ordinal range are
/// exported; otherwise all records are included. When `output` is
/// `None`, records are written to stdout.
///
/// Format resolution priority:
/// 1. Explicit `--format` value.
/// 2. Output file extension (`.slab` → Slab).
/// 3. Default: Raw (with a warning when writing to stdout if the data
///    contains non-printable bytes and no explicit format was given).
pub fn run(cfg: &ExportConfig) -> Result<(), Box<dyn std::error::Error>> {
    let reader = SlabReader::open_namespace(cfg.file, cfg.namespace.as_deref())?;
    let records = if let Some(range_str) = cfg.range {
        let (start, end) = ordinal_range::parse_ordinal_range(range_str)?;
        reader.iter_range(start, end)?
    } else {
        reader.iter()?
    };
    let reporter = ProgressReporter::new(cfg.progress);

    let explicit = cfg.format.is_some();

    // Determine output format
    let format = if let Some(f) = cfg.format {
        f
    } else if let Some(out) = cfg.output {
        detect_format_from_extension(out)
    } else {
        ExportFormat::Raw
    };

    // When writing raw to stdout without an explicit format, warn if
    // the data contains non-printable bytes.
    if format == ExportFormat::Raw && cfg.output.is_none() && !explicit {
        let sample_count = records.len().min(10);
        let mut found_unprintable = false;
        for i in 0..sample_count {
            let (_ordinal, data) = &records[i];
            if has_unprintable(data) {
                found_unprintable = true;
                break;
            }
        }
        if found_unprintable {
            eprintln!(
                "Warning: Output contains non-printable bytes. Use --format=raw to suppress this warning,\n\
                 or --format=hex for hex-encoded output, or --format=text for text output."
            );
        }
    }

    match format {
        ExportFormat::Slab => {
            let out_path = cfg.output.ok_or("slab export format requires --output")?;
            let config =
                make_writer_config(cfg.preferred_page_size, cfg.min_page_size, cfg.page_alignment)?;
            let record_count = records.len();
            write_with_buffer_rename(out_path, |buf_path| {
                let mut writer = SlabWriter::new(buf_path, config)?;
                for (_ordinal, data) in &records {
                    writer.add_record(data)?;
                    reporter.inc();
                }
                writer.finish()?;
                Ok(())
            })?;
            eprintln!("Exported {} records to {out_path} (slab)", record_count);
        }
        ExportFormat::Raw => {
            let mut sink: Box<dyn Write> = match cfg.output {
                Some(path) => Box::new(std::fs::File::create(path)?),
                None => Box::new(io::stdout().lock()),
            };
            for (_ordinal, data) in &records {
                sink.write_all(data)?;
                reporter.inc();
            }
            sink.flush()?;
            if let Some(path) = cfg.output {
                eprintln!("Exported {} records to {path} (raw)", records.len());
            }
        }
        ExportFormat::Text => {
            let mut sink: Box<dyn Write> = match cfg.output {
                Some(path) => Box::new(std::fs::File::create(path)?),
                None => Box::new(io::stdout().lock()),
            };
            for (_ordinal, data) in &records {
                sink.write_all(data)?;
                if !data.ends_with(b"\n") {
                    sink.write_all(b"\n")?;
                }
                reporter.inc();
            }
            sink.flush()?;
            if let Some(path) = cfg.output {
                eprintln!("Exported {} records to {path} (text)", records.len());
            }
        }
        ExportFormat::Cstrings => {
            let mut sink: Box<dyn Write> = match cfg.output {
                Some(path) => Box::new(std::fs::File::create(path)?),
                None => Box::new(io::stdout().lock()),
            };
            for (_ordinal, data) in &records {
                sink.write_all(data)?;
                if !data.ends_with(b"\0") {
                    sink.write_all(b"\0")?;
                }
                reporter.inc();
            }
            sink.flush()?;
            if let Some(path) = cfg.output {
                eprintln!("Exported {} records to {path} (cstrings)", records.len());
            }
        }
        ExportFormat::Hex => {
            let mut sink: Box<dyn Write> = match cfg.output {
                Some(path) => Box::new(std::fs::File::create(path)?),
                None => Box::new(io::stdout().lock()),
            };
            for (_ordinal, data) in &records {
                for &byte in data.iter() {
                    write!(sink, "{:02x}", byte)?;
                }
                sink.write_all(b"\n")?;
                reporter.inc();
            }
            sink.flush()?;
            if let Some(path) = cfg.output {
                eprintln!("Exported {} records to {path} (hex)", records.len());
            }
        }
    }

    reporter.finish();
    Ok(())
}

/// Detect export format from the output file extension.
fn detect_format_from_extension(path: &str) -> ExportFormat {
    match Path::new(path).extension().and_then(|e| e.to_str()) {
        Some("slab") => ExportFormat::Slab,
        _ => ExportFormat::Raw,
    }
}
