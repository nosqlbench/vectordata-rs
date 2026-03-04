// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! `slab import` subcommand — import data from external file formats into a slab file.
//!
//! Supports newline-terminated text, null-terminated binary, slab, json,
//! jsonl, csv, tsv, and yaml sources. Delimiters are preserved in the
//! record data so that concatenating exported records reproduces the
//! original file.

use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::{SlabReader, SlabWriter};

use super::{is_new_file, make_writer_config, write_with_buffer_rename, ProgressReporter};

/// Source file format for import.
enum SourceFormat {
    /// Newline-delimited text; each record includes its trailing `\n`.
    NewlineTerminated,
    /// Null-terminated binary; each record includes its trailing `\0`.
    NullTerminated,
    /// Slabtastic `.slab` format.
    Slab,
    /// JSON — stream of objects with whitespace between them.
    Json,
    /// JSONL — newline-delimited JSON, one object per line.
    Jsonl,
    /// CSV — comma-separated values.
    Csv,
    /// TSV — tab-separated values.
    Tsv,
    /// YAML — documents separated by `---`.
    Yaml,
}

/// Detect the source format from the file extension, falling back to
/// binary content scanning.
///
/// Extension mapping: `.slab` → slab, `.json` → json, `.jsonl`/`.ndjson`
/// → jsonl, `.csv` → csv, `.tsv` → tsv, `.yaml`/`.yml` → yaml.
/// Unknown extensions trigger a content scan (up to 5 seconds).
fn auto_detect_format(path: &str) -> Result<SourceFormat, Box<dyn std::error::Error>> {
    let extension = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match extension.as_deref() {
        Some("slab") => return Ok(SourceFormat::Slab),
        Some("json") => return Ok(SourceFormat::Json),
        Some("jsonl") | Some("ndjson") => return Ok(SourceFormat::Jsonl),
        Some("csv") => return Ok(SourceFormat::Csv),
        Some("tsv") => return Ok(SourceFormat::Tsv),
        Some("yaml") | Some("yml") => return Ok(SourceFormat::Yaml),
        _ => {}
    }

    // Fallback: scan file content
    let mut file = std::fs::File::open(path)?;
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut buf = [0u8; 8192];

    loop {
        if Instant::now() >= deadline {
            break;
        }
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            if !b.is_ascii() {
                return Ok(SourceFormat::NullTerminated);
            }
        }
    }

    Ok(SourceFormat::NewlineTerminated)
}

/// Split `data` on `delimiter`, including the delimiter in each record.
///
/// If the data does not end with the delimiter, the trailing chunk becomes
/// a record without a delimiter suffix.
pub fn split_with_delimiter(data: &[u8], delimiter: u8) -> Vec<Vec<u8>> {
    let mut records = Vec::new();
    let mut start = 0;
    for (i, &b) in data.iter().enumerate() {
        if b == delimiter {
            records.push(data[start..=i].to_vec());
            start = i + 1;
        }
    }
    if start < data.len() {
        records.push(data[start..].to_vec());
    }
    records
}

/// Run the `import` subcommand.
///
/// Imports records from `source` into the slab file at `file`. When no
/// format flag is given, the source format is auto-detected. Delimiters
/// are preserved in the record data unless `strip_newline` is set.
///
/// When `skip_malformed` is `true`, records that fail to parse in
/// structured formats (json, jsonl, csv, tsv, yaml) are silently
/// skipped and the count of skipped records is printed at the end.
#[allow(clippy::too_many_arguments)]
pub fn run(
    file: &str,
    source: &str,
    newline_terminated_records: bool,
    null_terminated_records: bool,
    slab_format: bool,
    json: bool,
    jsonl: bool,
    csv: bool,
    tsv: bool,
    yaml: bool,
    skip_malformed: bool,
    strip_newline: bool,
    preferred_page_size: Option<u32>,
    min_page_size: Option<u32>,
    page_alignment: bool,
    progress: bool,
    namespace: &Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let format = if slab_format {
        SourceFormat::Slab
    } else if newline_terminated_records {
        SourceFormat::NewlineTerminated
    } else if null_terminated_records {
        SourceFormat::NullTerminated
    } else if json {
        SourceFormat::Json
    } else if jsonl {
        SourceFormat::Jsonl
    } else if csv {
        SourceFormat::Csv
    } else if tsv {
        SourceFormat::Tsv
    } else if yaml {
        SourceFormat::Yaml
    } else {
        auto_detect_format(source)?
    };

    let config = make_writer_config(preferred_page_size, min_page_size, page_alignment)?;
    let creating_new = is_new_file(file);
    let reporter = ProgressReporter::new(progress);

    let do_import = |target: &str| -> std::result::Result<(u64, u64), Box<dyn std::error::Error>> {
        let mut writer = if Path::new(target).exists() {
            SlabWriter::append_namespace(target, config, namespace.as_deref())?
        } else {
            SlabWriter::new(target, config)?
        };

        let (count, skipped): (u64, u64) = match format {
            SourceFormat::Slab => {
                let reader = SlabReader::open(source)?;
                let records = reader.iter()?;
                let n = records.len() as u64;
                for (_ordinal, data) in records {
                    let rec = maybe_strip_newline(&data, strip_newline);
                    writer.add_record(&rec)?;
                    reporter.inc();
                }
                (n, 0)
            }
            SourceFormat::NewlineTerminated => {
                let data = std::fs::read(source)?;
                let records = split_with_delimiter(&data, b'\n');
                let n = records.len() as u64;
                for rec in &records {
                    let rec = maybe_strip_newline(rec, strip_newline);
                    writer.add_record(&rec)?;
                    reporter.inc();
                }
                (n, 0)
            }
            SourceFormat::NullTerminated => {
                let data = std::fs::read(source)?;
                let records = split_with_delimiter(&data, b'\0');
                let n = records.len() as u64;
                for rec in &records {
                    writer.add_record(rec)?;
                    reporter.inc();
                }
                (n, 0)
            }
            SourceFormat::Json => import_json(source, &mut writer, skip_malformed, strip_newline)?,
            SourceFormat::Jsonl => import_jsonl(source, &mut writer, skip_malformed, strip_newline)?,
            SourceFormat::Csv => import_csv(source, &mut writer, b',', skip_malformed, strip_newline)?,
            SourceFormat::Tsv => import_csv(source, &mut writer, b'\t', skip_malformed, strip_newline)?,
            SourceFormat::Yaml => import_yaml(source, &mut writer, skip_malformed, strip_newline)?,
        };

        writer.finish()?;
        Ok((count, skipped))
    };

    let (count, skipped) = if creating_new {
        let mut result = (0u64, 0u64);
        write_with_buffer_rename(file, |buf_path| {
            result = do_import(buf_path)?;
            Ok(())
        })?;
        result
    } else {
        do_import(file)?
    };

    reporter.finish();
    println!("Imported {count} records from {source} into {file}");
    if skipped > 0 {
        println!("Skipped {skipped} malformed records");
    }
    Ok(())
}

/// Strip a trailing newline from `data` when `strip` is `true`.
fn maybe_strip_newline(data: &[u8], strip: bool) -> Vec<u8> {
    if strip && data.last() == Some(&b'\n') {
        data[..data.len() - 1].to_vec()
    } else {
        data.to_vec()
    }
}

/// Import JSON: stream of JSON objects with whitespace between them.
/// Each object is serialized back to a compact JSON string as a record.
///
/// Returns `(imported_count, skipped_count)`.
fn import_json(
    source: &str,
    writer: &mut SlabWriter,
    skip_malformed: bool,
    strip_newline: bool,
) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(source)?;
    let reader = std::io::BufReader::new(file);
    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<serde_json::Value>();

    let mut count: u64 = 0;
    let mut skipped: u64 = 0;
    for value in stream {
        match value {
            Ok(value) => {
                let mut serialized = serde_json::to_string(&value)?;
                if !strip_newline {
                    serialized.push('\n');
                }
                writer.add_record(serialized.as_bytes())?;
                count += 1;
            }
            Err(e) => {
                if skip_malformed {
                    skipped += 1;
                } else {
                    return Err(e.into());
                }
            }
        }
    }
    Ok((count, skipped))
}

/// Import JSONL: newline-delimited JSON, one object per line.
/// Each line (including trailing newline) is validated as JSON and stored.
///
/// Returns `(imported_count, skipped_count)`.
fn import_jsonl(
    source: &str,
    writer: &mut SlabWriter,
    skip_malformed: bool,
    strip_newline: bool,
) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(source)?;
    let mut count: u64 = 0;
    let mut skipped: u64 = 0;
    for line in data.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Validate it parses as JSON
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(_) => {
                let mut rec = line.to_string();
                if !strip_newline {
                    rec.push('\n');
                }
                writer.add_record(rec.as_bytes())?;
                count += 1;
            }
            Err(e) => {
                if skip_malformed {
                    skipped += 1;
                } else {
                    return Err(e.into());
                }
            }
        }
    }
    Ok((count, skipped))
}

/// Import CSV or TSV. Each row (including header) becomes a record
/// with a trailing newline.
///
/// Returns `(imported_count, skipped_count)`.
fn import_csv(
    source: &str,
    writer: &mut SlabWriter,
    delimiter: u8,
    skip_malformed: bool,
    strip_newline: bool,
) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(false)
        .from_path(source)?;

    let mut count: u64 = 0;
    let mut skipped: u64 = 0;
    for result in rdr.records() {
        match result {
            Ok(record) => {
                let fields: Vec<&str> = record.iter().collect();
                let delim_char = delimiter as char;
                let mut row = fields.join(&delim_char.to_string());
                if !strip_newline {
                    row.push('\n');
                }
                writer.add_record(row.as_bytes())?;
                count += 1;
            }
            Err(e) => {
                if skip_malformed {
                    skipped += 1;
                } else {
                    return Err(e.into());
                }
            }
        }
    }
    Ok((count, skipped))
}

/// Import YAML: documents separated by `---`. Each document is a record.
///
/// Returns `(imported_count, skipped_count)`.
fn import_yaml(
    source: &str,
    writer: &mut SlabWriter,
    skip_malformed: bool,
    strip_newline: bool,
) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(source)?;
    let mut count: u64 = 0;
    let mut skipped: u64 = 0;

    let store_doc = |doc: &str, writer: &mut SlabWriter| -> Result<bool, Box<dyn std::error::Error>> {
        if doc.trim().is_empty() {
            return Ok(false);
        }
        match serde_yaml::from_str::<serde_yaml::Value>(doc) {
            Ok(_) => {
                let mut rec = doc.to_string();
                if !strip_newline {
                    if !rec.ends_with('\n') {
                        rec.push('\n');
                    }
                } else {
                    // Strip trailing newline if present
                    if rec.ends_with('\n') {
                        rec.pop();
                    }
                }
                writer.add_record(rec.as_bytes())?;
                Ok(true)
            }
            Err(e) => {
                if skip_malformed {
                    Ok(false)
                } else {
                    Err(e.into())
                }
            }
        }
    };

    // Split on document separators
    let mut current_doc = String::new();
    for line in data.lines() {
        if line.trim() == "---" {
            match store_doc(&current_doc, writer)? {
                true => count += 1,
                false if !current_doc.trim().is_empty() => skipped += 1,
                _ => {}
            }
            current_doc.clear();
        } else {
            current_doc.push_str(line);
            current_doc.push('\n');
        }
    }
    // Handle trailing document without final ---
    match store_doc(&current_doc, writer)? {
        true => count += 1,
        false if !current_doc.trim().is_empty() => skipped += 1,
        _ => {}
    }

    Ok((count, skipped))
}
