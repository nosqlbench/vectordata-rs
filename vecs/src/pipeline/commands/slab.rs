// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline commands: slabtastic file operations.
//!
//! Provides import, export, append, rewrite, check, get, analyze, explain,
//! and namespaces commands for `.slab` files using the `slabtastic` crate.
//!
//! Equivalent to the Java `CMD_slab_*` commands.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use slabtastic::{SlabReader, SlabWriter, WriterConfig};

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

// -- Slab Import --------------------------------------------------------------

/// Pipeline command: import records into a slab file.
pub struct SlabImportOp;

pub fn import_factory() -> Box<dyn CommandOp> {
    Box::new(SlabImportOp)
}

impl CommandOp for SlabImportOp {
    fn command_path(&self) -> &str {
        "slab import"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let from_str = match options.require("from") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let to_str = match options.require("to") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let from_path = resolve_path(from_str, &ctx.workspace);
        let to_path = resolve_path(to_str, &ctx.workspace);
        let force = options.get("force").map_or(false, |s| s == "true");
        let page_size: u32 = options
            .get("page-size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536);
        let format = options.get("format").unwrap_or("text");

        if to_path.exists() && !force {
            return error_result(
                format!("{} already exists. Use force=true to overwrite.", to_path.display()),
                start,
            );
        }

        // Read source records based on format
        let records = match read_source_records(&from_path, format) {
            Ok(r) => r,
            Err(e) => return error_result(e, start),
        };

        // Write to slab
        let config = match WriterConfig::new(512, page_size, u32::MAX, false) {
            Ok(c) => c,
            Err(e) => return error_result(format!("invalid config: {}", e), start),
        };
        let mut writer = match SlabWriter::new(&to_path, config) {
            Ok(w) => w,
            Err(e) => return error_result(format!("failed to create slab: {}", e), start),
        };

        for record in &records {
            if let Err(e) = writer.add_record(record) {
                return error_result(format!("write error: {}", e), start);
            }
        }

        if let Err(e) = writer.finish() {
            return error_result(format!("finish error: {}", e), start);
        }

        let count = records.len();
        eprintln!("Imported {} records from {} to {}", count, from_path.display(), to_path.display());

        CommandResult {
            status: Status::Ok,
            message: format!("imported {} records", count),
            produced: vec![to_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("from", "Path", true, None, "Source file path"),
            opt("to", "Path", true, None, "Target slab file path"),
            opt("format", "enum", false, Some("text"), "Source format: text, cstrings"),
            opt("page-size", "int", false, Some("65536"), "Preferred page size"),
            opt("force", "bool", false, Some("false"), "Overwrite existing target"),
        ]
    }
}

/// Read source records from a file based on format.
fn read_source_records(path: &Path, format: &str) -> Result<Vec<Vec<u8>>, String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

    match format {
        "text" => {
            let text = String::from_utf8_lossy(&data);
            Ok(text.lines().map(|l| l.as_bytes().to_vec()).collect())
        }
        "cstrings" => {
            Ok(data.split(|&b| b == 0).filter(|s| !s.is_empty()).map(|s| s.to_vec()).collect())
        }
        "slab" => {
            let reader = SlabReader::open(path)
                .map_err(|e| format!("failed to open slab: {}", e))?;
            let mut records = Vec::new();
            let mut iter = reader.batch_iter(4096);
            loop {
                let batch = iter.next_batch()
                    .map_err(|e| format!("read error: {}", e))?;
                if batch.is_empty() {
                    break;
                }
                for (_ord, data) in batch {
                    records.push(data);
                }
            }
            Ok(records)
        }
        _ => {
            // Default: treat entire file as raw records split by newlines
            let text = String::from_utf8_lossy(&data);
            Ok(text.lines().map(|l| l.as_bytes().to_vec()).collect())
        }
    }
}

// -- Slab Export ---------------------------------------------------------------

/// Pipeline command: export records from a slab file.
pub struct SlabExportOp;

pub fn export_factory() -> Box<dyn CommandOp> {
    Box::new(SlabExportOp)
}

impl CommandOp for SlabExportOp {
    fn command_path(&self) -> &str {
        "slab export"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);
        let format = options.get("format").unwrap_or("text");

        let reader = match SlabReader::open(&input_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open: {}", e), start),
        };

        let output_path = options.get("to").map(|s| resolve_path(s, &ctx.workspace));

        let mut output: Box<dyn Write> = if let Some(ref path) = output_path {
            Box::new(
                std::fs::File::create(path)
                    .map_err(|e| format!("create error: {}", e))
                    .unwrap(),
            )
        } else {
            Box::new(std::io::stdout())
        };

        let mut count = 0u64;
        let mut iter = reader.batch_iter(4096);
        loop {
            let batch = match iter.next_batch() {
                Ok(b) => b,
                Err(e) => return error_result(format!("read error: {}", e), start),
            };
            if batch.is_empty() {
                break;
            }
            for (ordinal, data) in &batch {
                match format {
                    "hex" => {
                        let hex: Vec<String> = data.iter().map(|b| format!("{:02x}", b)).collect();
                        writeln!(output, "{}: {}", ordinal, hex.join(" ")).ok();
                    }
                    "raw" => {
                        output.write_all(data).ok();
                    }
                    "json" => {
                        let escaped = String::from_utf8_lossy(data);
                        writeln!(output, "{{\"ordinal\":{},\"data\":\"{}\"}}", ordinal, escaped).ok();
                    }
                    _ => {
                        // text
                        let text = String::from_utf8_lossy(data);
                        writeln!(output, "{}", text).ok();
                    }
                }
                count += 1;
            }
        }

        output.flush().ok();

        let mut produced = Vec::new();
        if let Some(path) = output_path {
            produced.push(path);
        }

        CommandResult {
            status: Status::Ok,
            message: format!("exported {} records", count),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Source slab file"),
            opt("to", "Path", false, None, "Output file (stdout if omitted)"),
            opt("format", "enum", false, Some("text"), "Output format: text, hex, raw, json"),
        ]
    }
}

// -- Slab Append ---------------------------------------------------------------

/// Pipeline command: append records from one slab to another.
pub struct SlabAppendOp;

pub fn append_factory() -> Box<dyn CommandOp> {
    Box::new(SlabAppendOp)
}

impl CommandOp for SlabAppendOp {
    fn command_path(&self) -> &str {
        "slab append"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let target_str = match options.require("target") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let from_str = match options.require("from") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let target_path = resolve_path(target_str, &ctx.workspace);
        let from_path = resolve_path(from_str, &ctx.workspace);

        let page_size: u32 = options
            .get("page-size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536);

        // Read source records
        let source = match SlabReader::open(&from_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open source: {}", e), start),
        };

        let config = match WriterConfig::new(512, page_size, u32::MAX, false) {
            Ok(c) => c,
            Err(e) => return error_result(format!("invalid config: {}", e), start),
        };
        let mut writer = match SlabWriter::append(&target_path, config) {
            Ok(w) => w,
            Err(e) => return error_result(format!("failed to open target for append: {}", e), start),
        };

        let mut count = 0u64;
        let mut iter = source.batch_iter(4096);
        loop {
            let batch = match iter.next_batch() {
                Ok(b) => b,
                Err(e) => return error_result(format!("read error: {}", e), start),
            };
            if batch.is_empty() {
                break;
            }
            for (_ord, data) in &batch {
                if let Err(e) = writer.add_record(data) {
                    return error_result(format!("write error: {}", e), start);
                }
                count += 1;
            }
        }

        if let Err(e) = writer.finish() {
            return error_result(format!("finish error: {}", e), start);
        }

        eprintln!("Appended {} records from {} to {}", count, from_path.display(), target_path.display());

        CommandResult {
            status: Status::Ok,
            message: format!("appended {} records", count),
            produced: vec![target_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("target", "Path", true, None, "Target slab file to append to"),
            opt("from", "Path", true, None, "Source slab file"),
            opt("page-size", "int", false, Some("65536"), "Preferred page size"),
        ]
    }
}

// -- Slab Rewrite --------------------------------------------------------------

/// Pipeline command: rewrite a slab file with clean page alignment.
pub struct SlabRewriteOp;

pub fn rewrite_factory() -> Box<dyn CommandOp> {
    Box::new(SlabRewriteOp)
}

impl CommandOp for SlabRewriteOp {
    fn command_path(&self) -> &str {
        "slab rewrite"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let dest_str = match options.require("dest") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let dest_path = resolve_path(dest_str, &ctx.workspace);
        let force = options.get("force").map_or(false, |s| s == "true");
        let page_size: u32 = options
            .get("page-size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536);

        if dest_path.exists() && !force {
            return error_result(
                format!("{} already exists. Use force=true to overwrite.", dest_path.display()),
                start,
            );
        }

        let reader = match SlabReader::open(&source_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open source: {}", e), start),
        };

        let config = match WriterConfig::new(512, page_size, u32::MAX, false) {
            Ok(c) => c,
            Err(e) => return error_result(format!("invalid config: {}", e), start),
        };
        let mut writer = match SlabWriter::new(&dest_path, config) {
            Ok(w) => w,
            Err(e) => return error_result(format!("failed to create dest: {}", e), start),
        };

        let mut count = 0u64;
        let mut iter = reader.batch_iter(4096);
        loop {
            let batch = match iter.next_batch() {
                Ok(b) => b,
                Err(e) => return error_result(format!("read error: {}", e), start),
            };
            if batch.is_empty() {
                break;
            }
            for (_ord, data) in &batch {
                if let Err(e) = writer.add_record(data) {
                    return error_result(format!("write error: {}", e), start);
                }
                count += 1;
            }
        }

        if let Err(e) = writer.finish() {
            return error_result(format!("finish error: {}", e), start);
        }

        eprintln!("Rewrote {} records from {} to {}", count, source_path.display(), dest_path.display());

        CommandResult {
            status: Status::Ok,
            message: format!("rewrote {} records", count),
            produced: vec![dest_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("source", "Path", true, None, "Source slab file"),
            opt("dest", "Path", true, None, "Destination slab file"),
            opt("page-size", "int", false, Some("65536"), "Preferred page size"),
            opt("force", "bool", false, Some("false"), "Overwrite existing dest"),
        ]
    }
}

// -- Slab Check ----------------------------------------------------------------

/// Pipeline command: validate a slab file for structural integrity.
pub struct SlabCheckOp;

pub fn check_factory() -> Box<dyn CommandOp> {
    Box::new(SlabCheckOp)
}

impl CommandOp for SlabCheckOp {
    fn command_path(&self) -> &str {
        "slab check"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);
        let verbose = options.get("verbose").map_or(false, |s| s == "true");

        let reader = match SlabReader::open(&input_path) {
            Ok(r) => r,
            Err(e) => {
                return CommandResult {
                    status: Status::Error,
                    message: format!("cannot read {}: {}", input_path.display(), e),
                    produced: vec![],
                    elapsed: start.elapsed(),
                };
            }
        };

        let page_entries = reader.page_entries();
        let page_count = page_entries.len();

        // Validate each page by reading it
        let mut errors = 0;
        let mut record_count = 0u64;

        for (i, entry) in page_entries.iter().enumerate() {
            match reader.read_data_page(entry) {
                Ok(page) => {
                    let recs = page.record_count() as u64;
                    record_count += recs;
                    if verbose {
                        eprintln!(
                            "  Page {}: offset={}, ordinal={}, records={}",
                            i,
                            entry.file_offset,
                            entry.start_ordinal,
                            recs
                        );
                    }
                }
                Err(e) => {
                    errors += 1;
                    eprintln!("  Page {}: ERROR: {}", i, e);
                }
            }
        }

        if errors == 0 {
            eprintln!(
                "OK: {} pages, {} records in {}",
                page_count,
                record_count,
                input_path.display()
            );
            CommandResult {
                status: Status::Ok,
                message: format!("{} pages, {} records, no errors", page_count, record_count),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        } else {
            CommandResult {
                status: Status::Error,
                message: format!("{} errors in {} pages", errors, page_count),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Slab file to check"),
            opt("verbose", "bool", false, Some("false"), "Show per-page details"),
        ]
    }
}

// -- Slab Get ------------------------------------------------------------------

/// Pipeline command: extract specific records by ordinal.
pub struct SlabGetOp;

pub fn get_factory() -> Box<dyn CommandOp> {
    Box::new(SlabGetOp)
}

impl CommandOp for SlabGetOp {
    fn command_path(&self) -> &str {
        "slab get"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let ordinals_str = match options.require("ordinals") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let input_path = resolve_path(input_str, &ctx.workspace);
        let format = options.get("format").unwrap_or("text");

        let ordinals = match parse_ordinals(ordinals_str) {
            Ok(o) => o,
            Err(e) => return error_result(e, start),
        };

        let reader = match SlabReader::open(&input_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open: {}", e), start),
        };

        let mut found = 0;
        let mut missing = 0;

        for ordinal in &ordinals {
            match reader.get(*ordinal) {
                Ok(data) => {
                    match format {
                        "hex" => {
                            let hex: Vec<String> =
                                data.iter().map(|b| format!("{:02x}", b)).collect();
                            eprintln!("[{}]: {}", ordinal, hex.join(" "));
                        }
                        "raw" => {
                            std::io::stdout().write_all(&data).ok();
                        }
                        _ => {
                            let text = String::from_utf8_lossy(&data);
                            eprintln!("[{}]: {}", ordinal, text);
                        }
                    }
                    found += 1;
                }
                Err(e) => {
                    eprintln!("[{}]: NOT FOUND ({})", ordinal, e);
                    missing += 1;
                }
            }
        }

        let status = if missing > 0 {
            Status::Warning
        } else {
            Status::Ok
        };

        CommandResult {
            status,
            message: format!("{} found, {} missing", found, missing),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Slab file"),
            opt("ordinals", "String", true, None, "Comma-separated ordinals or ranges (e.g. 0,1,5..10)"),
            opt("format", "enum", false, Some("text"), "Output format: text, hex, raw"),
        ]
    }
}

/// Parse ordinal specification: comma-separated numbers or ranges.
pub(crate) fn parse_ordinals(spec: &str) -> Result<Vec<i64>, String> {
    let mut result = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some(range_sep) = part.find("..") {
            let start: i64 = part[..range_sep]
                .trim()
                .parse()
                .map_err(|_| format!("invalid range start: '{}'", &part[..range_sep]))?;
            let end: i64 = part[range_sep + 2..]
                .trim()
                .parse()
                .map_err(|_| format!("invalid range end: '{}'", &part[range_sep + 2..]))?;
            for i in start..end {
                result.push(i);
            }
        } else {
            let ord: i64 = part
                .parse()
                .map_err(|_| format!("invalid ordinal: '{}'", part))?;
            result.push(ord);
        }
    }
    Ok(result)
}

// -- Slab Analyze --------------------------------------------------------------

/// Pipeline command: analyze a slab file and report statistics.
pub struct SlabAnalyzeOp;

pub fn analyze_factory() -> Box<dyn CommandOp> {
    Box::new(SlabAnalyzeOp)
}

impl CommandOp for SlabAnalyzeOp {
    fn command_path(&self) -> &str {
        "slab analyze"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);

        let reader = match SlabReader::open(&input_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open: {}", e), start),
        };

        let page_entries = reader.page_entries();
        let page_count = page_entries.len();

        let file_size = reader.file_len().unwrap_or(0);

        // Collect statistics
        let mut total_records = 0u64;
        let mut record_sizes: Vec<usize> = Vec::new();
        let mut page_sizes: Vec<u64> = Vec::new();
        let mut min_ordinal = i64::MAX;
        let mut max_ordinal = i64::MIN;

        for entry in &page_entries {
            match reader.read_data_page(entry) {
                Ok(page) => {
                    let recs = page.record_count() as u64;
                    total_records += recs;
                    page_sizes.push(page.serialized_size() as u64);

                    let start_ord = entry.start_ordinal;
                    if start_ord < min_ordinal {
                        min_ordinal = start_ord;
                    }
                    let end_ord = start_ord + recs as i64;
                    if end_ord > max_ordinal {
                        max_ordinal = end_ord;
                    }

                    // Sample record sizes from this page using get_record
                    for i in 0..(recs as usize).min(100) {
                        if let Some(data) = page.get_record(i) {
                            record_sizes.push(data.len());
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        eprintln!("Slab Analysis: {}", input_path.display());
        eprintln!("  File size: {} bytes", file_size);
        eprintln!("  Pages: {}", page_count);
        eprintln!("  Records: {}", total_records);

        if min_ordinal <= max_ordinal {
            eprintln!("  Ordinal range: {} .. {}", min_ordinal, max_ordinal);
        }

        if !record_sizes.is_empty() {
            record_sizes.sort();
            let min_rs = record_sizes[0];
            let max_rs = record_sizes[record_sizes.len() - 1];
            let avg_rs: f64 = record_sizes.iter().sum::<usize>() as f64 / record_sizes.len() as f64;
            let median_rs = record_sizes[record_sizes.len() / 2];
            eprintln!("  Record sizes: min={}, max={}, avg={:.1}, median={}", min_rs, max_rs, avg_rs, median_rs);

            // Content type detection from first page
            if let Some(first_entry) = page_entries.first() {
                if let Ok(first_page) = reader.read_data_page(first_entry) {
                    let sample_count = first_page.record_count().min(10);
                    let mut text_count = 0;
                    for i in 0..sample_count {
                        if let Some(data) = first_page.get_record(i) {
                            if data.iter().all(|&b| b.is_ascii()) {
                                text_count += 1;
                            }
                        }
                    }
                    let content_type = if text_count == sample_count {
                        "text/ascii"
                    } else if text_count > sample_count / 2 {
                        "mixed (mostly text)"
                    } else {
                        "binary"
                    };
                    eprintln!("  Content type: {} (sampled {} records)", content_type, sample_count);
                }
            }
        }

        if !page_sizes.is_empty() {
            let min_ps = page_sizes.iter().min().unwrap();
            let max_ps = page_sizes.iter().max().unwrap();
            let avg_ps: f64 = page_sizes.iter().sum::<u64>() as f64 / page_sizes.len() as f64;
            eprintln!("  Page sizes: min={}, max={}, avg={:.0}", min_ps, max_ps, avg_ps);

            // Utilization
            let total_page_bytes: u64 = page_sizes.iter().sum();
            if file_size > 0 {
                let utilization = total_page_bytes as f64 / file_size as f64 * 100.0;
                eprintln!("  Page utilization: {:.1}%", utilization);
            }
        }

        CommandResult {
            status: Status::Ok,
            message: format!("{} pages, {} records", page_count, total_records),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Slab file to analyze"),
        ]
    }
}

// -- Slab Explain --------------------------------------------------------------

/// Pipeline command: display slab page layout diagrams.
pub struct SlabExplainOp;

pub fn explain_factory() -> Box<dyn CommandOp> {
    Box::new(SlabExplainOp)
}

impl CommandOp for SlabExplainOp {
    fn command_path(&self) -> &str {
        "slab explain"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);

        let pages_filter: Option<Vec<usize>> = options.get("pages").map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect()
        });

        let reader = match SlabReader::open(&input_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open: {}", e), start),
        };

        let page_entries = reader.page_entries();

        for (i, entry) in page_entries.iter().enumerate() {
            if let Some(ref filter) = pages_filter {
                if !filter.contains(&i) {
                    continue;
                }
            }

            match reader.read_data_page(entry) {
                Ok(page) => {
                    let recs = page.record_count();
                    let page_size = page.serialized_size();

                    eprintln!("┌─── Page {} ───────────────────────────┐", i);
                    eprintln!("│ Offset:    {:<28}│", entry.file_offset);
                    eprintln!("│ Page size: {:<28}│", page_size);
                    eprintln!("│ Ordinals:  {:<28}│",
                        format!("{}..{}", entry.start_ordinal, entry.start_ordinal + recs as i64));
                    eprintln!("│ Records:   {:<28}│", recs);

                    // Show first few record sizes
                    let preview_count = recs.min(5) as i64;
                    if preview_count > 0 {
                        eprintln!("│ Record preview:                       │");
                        for j in 0..preview_count {
                            let ord = entry.start_ordinal + j;
                            if let Ok(data) = reader.get(ord) {
                                let hex_preview: String = data.iter().take(16)
                                    .map(|b| format!("{:02x}", b))
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                let suffix = if data.len() > 16 { "..." } else { "" };
                                eprintln!("│   [{}]: {} bytes  {}{:<2}│",
                                    ord, data.len(), hex_preview, suffix);
                            }
                        }
                    }
                    eprintln!("└───────────────────────────────────────┘");
                    eprintln!();
                }
                Err(e) => {
                    eprintln!("Page {}: ERROR reading: {}", i, e);
                }
            }
        }

        // Show pages page summary
        eprintln!("Pages Page: {} entries", page_entries.len());
        for (i, entry) in page_entries.iter().enumerate() {
            eprintln!(
                "  [{}] ordinal={}, offset={}",
                i,
                entry.start_ordinal,
                entry.file_offset
            );
        }

        CommandResult {
            status: Status::Ok,
            message: format!("explained {} pages", page_entries.len()),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Slab file"),
            opt("pages", "String", false, None, "Comma-separated page indices to display"),
        ]
    }
}

// -- Slab Namespaces -----------------------------------------------------------

/// Pipeline command: list namespaces in a slab file.
pub struct SlabNamespacesOp;

pub fn namespaces_factory() -> Box<dyn CommandOp> {
    Box::new(SlabNamespacesOp)
}

impl CommandOp for SlabNamespacesOp {
    fn command_path(&self) -> &str {
        "slab namespaces"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);

        let namespaces = match SlabReader::list_namespaces(&input_path) {
            Ok(ns) => ns,
            Err(e) => return error_result(format!("failed to list namespaces: {}", e), start),
        };

        eprintln!("{:<6} {:<30} {:>14}", "Index", "Name", "PagesOffset");
        eprintln!("{}", "-".repeat(54));

        for ns in &namespaces {
            eprintln!(
                "{:<6} {:<30} {:>14}",
                ns.namespace_index,
                ns.name,
                ns.pages_page_offset,
            );
        }

        eprintln!();
        eprintln!("{} namespace(s)", namespaces.len());

        CommandResult {
            status: Status::Ok,
            message: format!("{} namespaces", namespaces.len()),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Slab file"),
        ]
    }
}

// -- Slab Inspect --------------------------------------------------------------

/// Pipeline command: decode and render slab records as ANode vernacular text.
pub struct SlabInspectOp;

pub fn inspect_factory() -> Box<dyn CommandOp> {
    Box::new(SlabInspectOp)
}

impl CommandOp for SlabInspectOp {
    fn command_path(&self) -> &str {
        "slab inspect"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let ordinals_str = match options.require("ordinals") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let input_path = resolve_path(input_str, &ctx.workspace);
        let codec = options.get("codec").unwrap_or("auto");
        let format_str = options.get("format").unwrap_or("cddl");

        let vernacular = match crate::formats::anode_vernacular::Vernacular::from_str(format_str) {
            Some(v) => v,
            None => return error_result(format!("unknown format: '{}'", format_str), start),
        };

        let ordinals = match parse_ordinals(ordinals_str) {
            Ok(o) => o,
            Err(e) => return error_result(e, start),
        };

        let reader = match SlabReader::open(&input_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open: {}", e), start),
        };

        let mut found = 0;
        let mut errors = 0;

        for ordinal in &ordinals {
            match reader.get(*ordinal) {
                Ok(data) => {
                    let decode_result = match codec {
                        "mnode" => crate::formats::anode::decode_mnode(&data),
                        "pnode" => crate::formats::anode::decode_pnode(&data),
                        _ => crate::formats::anode::decode(&data),
                    };
                    match decode_result {
                        Ok(anode) => {
                            let rendered = crate::formats::anode_vernacular::render(&anode, vernacular);
                            eprintln!("[{}]: {}", ordinal, rendered);
                            found += 1;
                        }
                        Err(e) => {
                            eprintln!("[{}]: DECODE ERROR: {}", ordinal, e);
                            errors += 1;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[{}]: NOT FOUND ({})", ordinal, e);
                    errors += 1;
                }
            }
        }

        let status = if errors > 0 { Status::Warning } else { Status::Ok };

        CommandResult {
            status,
            message: format!("{} rendered, {} errors", found, errors),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Slab file"),
            opt("ordinals", "String", true, None, "Comma-separated ordinals or ranges (e.g. 0,1,5..10)"),
            opt("codec", "enum", false, Some("auto"), "Codec: auto, mnode, pnode"),
            opt("format", "enum", false, Some("cddl"), "Vernacular format: cddl, sql, cql, json, jsonl, yaml, readout, display"),
        ]
    }
}

// -- Helpers -------------------------------------------------------------------

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

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
    }
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

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

    fn create_test_slab(dir: &Path, name: &str, records: &[&[u8]]) -> PathBuf {
        let path = dir.join(name);
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut writer = SlabWriter::new(&path, config).unwrap();
        for rec in records {
            writer.add_record(rec).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    #[test]
    fn test_slab_import_text() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create source text file
        let src = ws.join("source.txt");
        std::fs::write(&src, "hello\nworld\nfoo\nbar\n").unwrap();

        let mut opts = Options::new();
        opts.set("from", src.to_string_lossy().to_string());
        opts.set("to", ws.join("output.slab").to_string_lossy().to_string());
        let mut op = SlabImportOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "import failed: {}", result.message);

        // Verify
        let reader = SlabReader::open(&ws.join("output.slab")).unwrap();
        let data = reader.get(0).unwrap();
        assert_eq!(&data, b"hello");
        let data = reader.get(3).unwrap();
        assert_eq!(&data, b"bar");
    }

    #[test]
    fn test_slab_check() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let slab_path = create_test_slab(ws, "test.slab", &[b"rec1", b"rec2", b"rec3"]);

        let mut opts = Options::new();
        opts.set("input", slab_path.to_string_lossy().to_string());
        let mut op = SlabCheckOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "check failed: {}", result.message);
    }

    #[test]
    fn test_slab_get() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let slab_path = create_test_slab(ws, "test.slab", &[b"alpha", b"beta", b"gamma"]);

        let mut opts = Options::new();
        opts.set("input", slab_path.to_string_lossy().to_string());
        opts.set("ordinals", "0,2");
        let mut op = SlabGetOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("2 found"));
    }

    #[test]
    fn test_slab_get_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let slab_path = create_test_slab(ws, "test.slab", &[b"only"]);

        let mut opts = Options::new();
        opts.set("input", slab_path.to_string_lossy().to_string());
        opts.set("ordinals", "0,99");
        let mut op = SlabGetOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("1 missing"));
    }

    #[test]
    fn test_slab_analyze() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let slab_path = create_test_slab(ws, "test.slab", &[b"hello world", b"test data", b"more stuff"]);

        let mut opts = Options::new();
        opts.set("input", slab_path.to_string_lossy().to_string());
        let mut op = SlabAnalyzeOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "analyze failed: {}", result.message);
    }

    #[test]
    fn test_slab_rewrite() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let slab_path = create_test_slab(ws, "source.slab", &[b"rec1", b"rec2"]);
        let dest_path = ws.join("dest.slab");

        let mut opts = Options::new();
        opts.set("source", slab_path.to_string_lossy().to_string());
        opts.set("dest", dest_path.to_string_lossy().to_string());
        let mut op = SlabRewriteOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "rewrite failed: {}", result.message);

        // Verify dest has same records
        let reader = SlabReader::open(&dest_path).unwrap();
        assert_eq!(reader.get(0).unwrap(), b"rec1");
        assert_eq!(reader.get(1).unwrap(), b"rec2");
    }

    #[test]
    fn test_slab_export_to_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let slab_path = create_test_slab(ws, "test.slab", &[b"line1", b"line2"]);
        let out_path = ws.join("output.txt");

        let mut opts = Options::new();
        opts.set("input", slab_path.to_string_lossy().to_string());
        opts.set("to", out_path.to_string_lossy().to_string());
        opts.set("format", "text");
        let mut op = SlabExportOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "export failed: {}", result.message);

        let content = std::fs::read_to_string(&out_path).unwrap();
        assert!(content.contains("line1"));
        assert!(content.contains("line2"));
    }

    #[test]
    fn test_slab_append() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let target_path = create_test_slab(ws, "target.slab", &[b"existing"]);
        let source_path = create_test_slab(ws, "source.slab", &[b"new1", b"new2"]);

        let mut opts = Options::new();
        opts.set("target", target_path.to_string_lossy().to_string());
        opts.set("from", source_path.to_string_lossy().to_string());
        let mut op = SlabAppendOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "append failed: {}", result.message);

        // Verify appended records
        let reader = SlabReader::open(&target_path).unwrap();
        assert_eq!(reader.get(0).unwrap(), b"existing");
        // New records are appended at next ordinal
        assert_eq!(reader.get(1).unwrap(), b"new1");
        assert_eq!(reader.get(2).unwrap(), b"new2");
    }

    #[test]
    fn test_parse_ordinals() {
        assert_eq!(parse_ordinals("0,1,2").unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_ordinals("0..3").unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_ordinals("5").unwrap(), vec![5]);
        assert_eq!(parse_ordinals("0,5..8,10").unwrap(), vec![0, 5, 6, 7, 10]);
        assert!(parse_ordinals("abc").is_err());
    }

    #[test]
    fn test_slab_explain() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let slab_path = create_test_slab(ws, "test.slab", &[b"hello", b"world"]);

        let mut opts = Options::new();
        opts.set("input", slab_path.to_string_lossy().to_string());
        let mut op = SlabExplainOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }
}
