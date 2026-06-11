// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline commands: slabtastic file operations.
//!
//! Provides import, export, append, rewrite, check, get, analyze, explain,
//! namespaces, inspect, and survey commands for `.slab` files using the
//! `slabtastic` crate.
//!
//! Equivalent to the Java `CMD_slab_*` commands.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use slabtastic::{OpenProgress, SlabReader, SlabWriter, WriterConfig};

use crate::pipeline::atomic_write::safe_create_file;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Import records into a slab file".into(),
            body: format!(
                r#"# slab import

Import records into a slab file.

## Description

Reads records from an input source and writes them into a slab file,
creating or overwriting the target file. The source can be a text file
(one record per line), a null-delimited binary file (cstrings format),
or another slab file.

## How It Works

The command reads the entire source file into memory, splits it into
individual records according to the selected format, then writes each
record sequentially into a new slab file using the configured page size.
Records are assigned monotonically increasing ordinals starting from zero.
The slab writer packs records into pages of the requested size, flushing
each page to disk when full.

## Data Preparation Role

`slab import` is the primary entry point for converting external metadata
into the slab binary format used throughout the dataset preparation
pipeline. Typical usage converts parquet-exported JSONL or raw text files
into `.slab` files so that downstream commands like `compute predicates`,
`survey`, and `slab inspect` can operate on them efficiently. Because slab
files support ordinal-based random access, importing metadata into slab
format is a prerequisite for any step that needs to cross-reference
metadata records with vector ordinals.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "iothreads".into(), description: "Concurrent I/O for source reading".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Read-ahead buffer size".into(), adjustable: false },
        ]
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
        let force = options.get("force") == Some("true");
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
        ctx.ui.log(&format!("Imported {} records from {} to {}", count, from_path.display(), to_path.display()));

        let var_name = format!("verified_count:{}",
            to_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &count.to_string());
        ctx.defaults.insert(var_name, count.to_string());

        CommandResult {
            status: Status::Ok,
            message: format!("imported {} records", count),
            produced: vec![to_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("from", "Path", true, None, "Source file path", OptionRole::Input),
            opt("to", "Path", true, None, "Target slab file path", OptionRole::Output),
            opt("format", "enum", false, Some("text"), "Source format: text, cstrings", OptionRole::Config),
            opt("page-size", "int", false, Some("65536"), "Preferred page size", OptionRole::Config),
            opt("force", "bool", false, Some("false"), "Overwrite existing target", OptionRole::Config),
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
            let reader = open_slab(path)
                .map_err(|e| format!("failed to open slab {}: {}", path.display(), e))?;
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Export slab records to text or binary".into(),
            body: format!(
                r#"# slab export

Export slab records to text or binary.

## Description

Reads records from a slab file and exports them in the specified output
format. Supported output formats include plain text (UTF-8 lossy), hex
dump with ordinal prefixes, raw binary, and JSON with ordinal keys.

## How It Works

The command opens the slab file, iterates over every record in page
order using batch reads, and writes each record to the output in the
chosen format. When no output path is given the results are written to
stdout, making it easy to pipe into other tools. The export streams
records without loading the entire slab into memory, so it works on
arbitrarily large files.

## Data Preparation Role

`slab export` is the inverse of `slab import` and is used to inspect
the contents of metadata slab files in human-readable form. This is
useful for verifying that an import produced the expected records,
for extracting metadata into formats consumable by external tools, and
for debugging predicate or metadata issues during pipeline development.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);
        let format = options.get("format").unwrap_or("text");

        let reader = match open_slab_display(&input_path, &ctx.ui) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", input_path.display(), e), start),
        };

        let output_path = options.get("to").map(|s| resolve_path(s, &ctx.workspace));

        let mut output: Box<dyn Write> = if let Some(ref path) = output_path {
            Box::new(
                safe_create_file(path)
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
            opt("source", "Path", true, None, "Source slab file", OptionRole::Input),
            opt("to", "Path", false, None, "Output file (stdout if omitted)", OptionRole::Output),
            opt("format", "enum", false, Some("text"), "Output format: text, hex, raw, json", OptionRole::Config),
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Append records from one slab to another".into(),
            body: format!(
                r#"# slab append

Append records from one slab to another.

## Description

Reads records from a source slab file and appends them to a destination
slab file. The appended records receive ordinals that continue from where
the target file left off. The target file is opened in append mode so
existing records are preserved.

## How It Works

The command opens the source slab for reading and the target slab in
append mode. It iterates over every record in the source in page-order
batches and writes each record into the target's append writer, which
allocates new pages as needed at the configured page size. After all
source records have been transferred the writer finalizes the target
file's page directory.

## Data Preparation Role

`slab append` supports incremental metadata updates in dataset
preparation pipelines. When new metadata arrives (for example, a new
partition of a HuggingFace dataset), it can be imported into a
temporary slab and then appended to the main metadata slab without
rebuilding from scratch. This is particularly valuable for large
datasets where a full re-import would be prohibitively expensive.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
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
        let source = match open_slab_display(&from_path, &ctx.ui) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open source {}: {}", from_path.display(), e), start),
        };

        let config = match WriterConfig::new(512, page_size, u32::MAX, false) {
            Ok(c) => c,
            Err(e) => return error_result(format!("invalid config: {}", e), start),
        };
        let mut writer = match SlabWriter::append(&target_path, config) {
            Ok(w) => w,
            Err(e) => return error_result(format!("failed to open target for append {}: {}", target_path.display(), e), start),
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

        ctx.ui.log(&format!("Appended {} records from {} to {}", count, from_path.display(), target_path.display()));

        let var_name = format!("verified_count:{}",
            target_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &count.to_string());
        ctx.defaults.insert(var_name, count.to_string());

        CommandResult {
            status: Status::Ok,
            message: format!("appended {} records", count),
            produced: vec![target_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("target", "Path", true, None, "Target slab file to append to", OptionRole::Output),
            opt("from", "Path", true, None, "Source slab file", OptionRole::Input),
            opt("page-size", "int", false, Some("65536"), "Preferred page size", OptionRole::Config),
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Rewrite a slab file with new page layout".into(),
            body: format!(
                r#"# slab rewrite

Rewrite a slab file with new page layout.

## Description

Reads every record from a source slab file and writes them into a new
destination slab file using a different page size or layout
configuration. The record data itself is unchanged; only the physical
page structure differs.

## How It Works

The command performs a full sequential scan of the source slab, reading
records in page-order batches. Each record is written to a fresh slab
writer configured with the requested page size. Because the writer
repacks records from scratch, the destination file will have optimal
page alignment with no wasted space from prior appends or deletions.

## Data Preparation Role

`slab rewrite` is used to optimize slab layout for specific access
patterns. A slab originally written with small pages (good for random
access) can be rewritten with larger pages (better for sequential
scans), or vice versa. This is typically run as a post-processing step
after all imports and appends are complete, before the slab is used in
production queries where page I/O efficiency matters.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Page buffers during rewrite".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
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
        let force = options.get("force") == Some("true");
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

        let reader = match open_slab_display(&source_path, &ctx.ui) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open source {}: {}", source_path.display(), e), start),
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

        ctx.ui.log(&format!("Rewrote {} records from {} to {}", count, source_path.display(), dest_path.display()));

        let var_name = format!("verified_count:{}",
            dest_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &count.to_string());
        ctx.defaults.insert(var_name, count.to_string());

        CommandResult {
            status: Status::Ok,
            message: format!("rewrote {} records", count),
            produced: vec![dest_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("source", "Path", true, None, "Source slab file", OptionRole::Input),
            opt("dest", "Path", true, None, "Destination slab file", OptionRole::Output),
            opt("page-size", "int", false, Some("65536"), "Preferred page size", OptionRole::Config),
            opt("force", "bool", false, Some("false"), "Overwrite existing dest", OptionRole::Config),
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Verify slab file structural integrity".into(),
            body: format!(
                r#"# slab check

Verify slab file structural integrity.

## Description

Validates the internal structure of a slab file by reading and
deserializing every page. Reports the total number of pages and records
on success, or lists specific page-level errors on failure.

## How It Works

The command opens the slab file, retrieves the page directory, then
attempts to read and decode each data page in sequence. For every page
it verifies that the binary page data can be deserialized without
errors and that the record count is consistent. In verbose mode it
prints per-page details including file offset, starting ordinal, and
record count. Any page that fails to decode is reported as an error.

## Data Preparation Role

`slab check` is a health-verification step that should be run after
importing, appending, or rewriting slab files to confirm that the
resulting file is structurally sound. It is especially important after
bulk operations or interrupted writes, where a partially written page
could leave the slab in an inconsistent state. Including `slab check`
in a pipeline provides an early warning before downstream commands
attempt to read from a corrupted file.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);
        let verbose = options.get("verbose") == Some("true");

        let reader = match open_slab_display(&input_path, &ctx.ui) {
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
                        ctx.ui.log(&format!(
                            "  Page {}: offset={}, ordinal={}, records={}",
                            i,
                            entry.file_offset,
                            entry.start_ordinal,
                            recs
                        ));
                    }
                }
                Err(e) => {
                    errors += 1;
                    ctx.ui.log(&format!("  Page {}: ERROR: {}", i, e));
                }
            }
        }

        if errors == 0 {
            ctx.ui.log(&format!(
                "OK: {} pages, {} records in {}",
                page_count,
                record_count,
                input_path.display()
            ));
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
            opt("source", "Path", true, None, "Slab file to check", OptionRole::Input),
            opt("verbose", "bool", false, Some("false"), "Show per-page details", OptionRole::Config),
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Retrieve a single record by ordinal".into(),
            body: format!(
                r#"# slab get

Retrieve a single record by ordinal.

## Description

Looks up and displays one or more records from a slab file by their
ordinal positions. Records can be specified as individual ordinals or
as ranges, and the output format can be plain text, hexadecimal, or
raw binary.

## How It Works

The command parses the ordinals specification into a list of i64
values (supporting comma-separated values and `start..end` ranges).
For each ordinal it performs a random-access lookup via the slab
reader's page directory, which binary-searches the page index to find
the page containing the requested ordinal, then reads and decodes
that single page to extract the record. This makes individual lookups
efficient even on very large slab files.

## Data Preparation Role

`slab get` provides point lookups for debugging and spot-checking
metadata contents. During pipeline development you can use it to
verify that a specific metadata record (by ordinal) contains the
expected values, or to inspect the record that corresponds to a
particular vector ordinal. It is commonly used alongside
`inspect predicate` to trace the mapping between predicates,
metadata indices, and raw metadata records.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
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

        let reader = match open_slab_display(&input_path, &ctx.ui) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", input_path.display(), e), start),
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
                            ctx.ui.log(&format!("[{}]: {}", ordinal, hex.join(" ")));
                        }
                        "raw" => {
                            std::io::stdout().write_all(&data).ok();
                        }
                        _ => {
                            let text = String::from_utf8_lossy(&data);
                            ctx.ui.log(&format!("[{}]: {}", ordinal, text));
                        }
                    }
                    found += 1;
                }
                Err(e) => {
                    ctx.ui.log(&format!("[{}]: NOT FOUND ({})", ordinal, e));
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
            opt("source", "Path", true, None, "Slab file", OptionRole::Input),
            opt("ordinals", "String", true, None, "Comma-separated ordinals or ranges (e.g. 0,1,5..10)", OptionRole::Config),
            opt("format", "enum", false, Some("text"), "Output format: text, hex, raw", OptionRole::Config),
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Analyze slab page utilization and statistics".into(),
            body: format!(
                r#"# slab analyze

Analyze slab page utilization and statistics.

## Description

Scans a slab file and reports comprehensive statistics about its
page-level utilization, record size distribution, content type, and
storage efficiency.

## How It Works

The command reads every page in the slab, collecting per-page record
counts and serialized page sizes. It also samples record sizes (up to
100 records per page) to compute min, max, mean, and median record
sizes. A content-type heuristic examines the first page's records to
classify the slab as text/ascii, mixed, or binary. Finally it reports
overall page utilization as a percentage of file size occupied by page
data versus overhead.

## Data Preparation Role

`slab analyze` helps you understand the storage efficiency of a slab
file and decide whether a `slab rewrite` with a different page size
would improve performance. For example, if the analysis reveals that
records are uniformly small and pages are only partially filled, a
smaller page size would reduce wasted I/O. Conversely, if records are
large and pages are fully packed, the current layout is already
efficient. This command is also useful for validating that an import
produced the expected number of records and ordinal range.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);

        let reader = match open_slab_display(&input_path, &ctx.ui) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", input_path.display(), e), start),
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

        ctx.ui.log(&format!("Slab Analysis: {}", input_path.display()));
        ctx.ui.log(&format!("  File size: {} bytes", file_size));
        ctx.ui.log(&format!("  Pages: {}", page_count));
        ctx.ui.log(&format!("  Records: {}", total_records));

        if min_ordinal <= max_ordinal {
            ctx.ui.log(&format!("  Ordinal range: {} .. {}", min_ordinal, max_ordinal));
        }

        if !record_sizes.is_empty() {
            record_sizes.sort();
            let min_rs = record_sizes[0];
            let max_rs = record_sizes[record_sizes.len() - 1];
            let avg_rs: f64 = record_sizes.iter().sum::<usize>() as f64 / record_sizes.len() as f64;
            let median_rs = record_sizes[record_sizes.len() / 2];
            ctx.ui.log(&format!("  Record sizes: min={}, max={}, avg={:.1}, median={}", min_rs, max_rs, avg_rs, median_rs));

            // Content type detection from first page
            if let Some(first_entry) = page_entries.first()
                && let Ok(first_page) = reader.read_data_page(first_entry) {
                    let sample_count = first_page.record_count().min(10);
                    let mut text_count = 0;
                    for i in 0..sample_count {
                        if let Some(data) = first_page.get_record(i)
                            && data.iter().all(|&b| b.is_ascii()) {
                                text_count += 1;
                            }
                    }
                    let content_type = if text_count == sample_count {
                        "text/ascii"
                    } else if text_count > sample_count / 2 {
                        "mixed (mostly text)"
                    } else {
                        "binary"
                    };
                    ctx.ui.log(&format!("  Content type: {} (sampled {} records)", content_type, sample_count));
                }
        }

        if !page_sizes.is_empty() {
            let min_ps = page_sizes.iter().min().unwrap();
            let max_ps = page_sizes.iter().max().unwrap();
            let avg_ps: f64 = page_sizes.iter().sum::<u64>() as f64 / page_sizes.len() as f64;
            ctx.ui.log(&format!("  Page sizes: min={}, max={}, avg={:.0}", min_ps, max_ps, avg_ps));

            // Utilization
            let total_page_bytes: u64 = page_sizes.iter().sum();
            if file_size > 0 {
                let utilization = total_page_bytes as f64 / file_size as f64 * 100.0;
                ctx.ui.log(&format!("  Page utilization: {:.1}%", utilization));
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
            opt("source", "Path", true, None, "Slab file to analyze", OptionRole::Input),
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Decode and display a record's wire format".into(),
            body: format!(
                r#"# slab explain

Decode and display a record's wire format.

## Description

Retrieves pages from a slab file and displays detailed structural
information for each page, including file offset, page size, ordinal
range, record count, and hex previews of the first few records. This
is a low-level diagnostic that shows the on-disk layout rather than
the logical record content.

## How It Works

The command opens the slab file and iterates over its page directory.
For each page (optionally filtered by page index), it reads the page
data and renders a box-drawing diagram showing the page's metadata
fields. It then reads the first few records from the page via direct
ordinal lookup and displays a hex preview of each record's raw bytes.
After all pages have been displayed, it prints a summary of the page
directory showing ordinal-to-offset mappings.

## Data Preparation Role

`slab explain` is a debugging tool for understanding the slab binary
format at the page level. When a slab file behaves unexpectedly --
for example, when `slab check` reports errors or when ordinal lookups
return surprising data -- `slab explain` reveals the physical layout
so you can diagnose whether pages are misaligned, overlapping, or
contain unexpected data. It is also useful for developing and testing
changes to the slabtastic writer.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);

        let pages_filter: Option<Vec<usize>> = options.get("pages").map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect()
        });

        // Schema sidecar — every typed slab produced by the
        // toolchain carries a single-record descriptor in the
        // `:schema` namespace (see `vectordata::metadata_schema`).
        // Surface it BEFORE the page-by-page hex dump so the
        // operator knows what the records actually are (e.g.
        // PNode-named predicates, MNode-encoded metadata)
        // rather than having to interpret raw bytes.
        render_schema_sidecar(&input_path, ctx);
        render_survey_sidecar(&input_path, ctx);

        let reader = match open_slab_display(&input_path, &ctx.ui) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", input_path.display(), e), start),
        };

        let page_entries = reader.page_entries();

        for (i, entry) in page_entries.iter().enumerate() {
            if let Some(ref filter) = pages_filter
                && !filter.contains(&i) {
                    continue;
                }

            match reader.read_data_page(entry) {
                Ok(page) => {
                    let recs = page.record_count();
                    let page_size = page.serialized_size();

                    let mut rows = vec![
                        format!("Offset:    {}", entry.file_offset),
                        format!("Page size: {}", page_size),
                        format!("Ordinals:  {}..{}",
                            entry.start_ordinal, entry.start_ordinal + recs as i64),
                        format!("Records:   {}", recs),
                    ];

                    // First few records — hex-preview each.
                    // Compute the hex byte budget from the
                    // terminal width: each byte costs 3 chars
                    // (`xx `) and we reserve room for the
                    // `[N]: NN bytes  ` prefix and a trailing
                    // `…` ellipsis.
                    let preview_count = recs.min(5) as i64;
                    if preview_count > 0 {
                        rows.push("Record preview:".into());
                        let inner = explain_inner_width();
                        // Header + padding + ellipsis: "  [NNN]: NNNNN bytes  " ≈ 22 chars.
                        let hex_budget = inner.saturating_sub(28).max(8);
                        let max_bytes = hex_budget / 3;
                        for j in 0..preview_count {
                            let ord = entry.start_ordinal + j;
                            if let Ok(data) = reader.get(ord) {
                                let n = data.len().min(max_bytes);
                                let hex_preview: String = data.iter().take(n)
                                    .map(|b| format!("{:02x}", b))
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                let ellipsis = if data.len() > n { "…" } else { "" };
                                rows.push(format!("  [{}]: {} bytes  {}{}",
                                    ord, data.len(), hex_preview, ellipsis));
                            }
                        }
                    }

                    render_explain_box(ctx, &format!("Page {}", i), &rows);
                    ctx.ui.log("");
                }
                Err(e) => {
                    ctx.ui.log(&format!("Page {}: ERROR reading: {}", i, e));
                }
            }
        }

        // Show pages page summary
        ctx.ui.log(&format!("Pages Page: {} entries", page_entries.len()));
        for (i, entry) in page_entries.iter().enumerate() {
            ctx.ui.log(&format!(
                "  [{}] ordinal={}, offset={}",
                i,
                entry.start_ordinal,
                entry.file_offset
            ));
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
            opt("source", "Path", true, None, "Slab file", OptionRole::Input),
            opt("pages", "String", false, None, "Comma-separated page indices to display", OptionRole::Config),
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Box renderer — terminal-width-aware drawing for slab explain.
//
// The previous renderer used hardcoded format-spec widths
// (`{:<28}`, `{:<2}`) that didn't line up with the box border
// glyphs and didn't account for content longer than the assumed
// width. Result: bottom border floated free of the right edge,
// long hex previews blew out of the right wall, and `| more`
// inherited an 80-col terminal but the boxes assumed something
// narrower. This module replaces the hand-rolled formatting with
// a width-aware helper that:
//   - reads the real terminal width via `crossterm::terminal::size()`,
//     falling back to 80 when stdout isn't a TTY (piped output);
//   - clamps the inner width into a readable range [40, 120];
//   - right-pads every content row so the right border lines up;
//   - truncates long content with `…` so the box never wraps;
//   - emits the top / bottom borders sized to match.
// ─────────────────────────────────────────────────────────────────────────────

/// Effective inner width of an explain-box content row (chars
/// between the `│ ` and ` │` markers, exclusive of the borders).
///
/// Source of truth, in priority order:
///   1. `ioctl(TIOCGWINSZ)` on stdout — accurate when stdout is
///      a TTY. Honoured even when stdin is redirected, which is
///      common (`veks slab explain --source foo | less`).
///   2. `$COLUMNS` env var — set by interactive shells; survives
///      a pipe in some shell configs but not all.
///   3. 80 — universal default; matches the user's `| more` case.
///
/// Inner width = terminal columns - 2 border glyphs. Clamped
/// into `[40, 120]` so narrow terminals still produce a usable
/// box and wide ones don't waste right-edge space.
fn explain_inner_width() -> usize {
    let cols = term_columns().unwrap_or(80);
    cols.saturating_sub(2).clamp(40, 120)
}

fn term_columns() -> Option<usize> {
    // ioctl(TIOCGWINSZ) on stdout.
    #[cfg(unix)]
    unsafe {
        let mut sz: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut sz as *mut _) == 0
            && sz.ws_col > 0
        {
            return Some(sz.ws_col as usize);
        }
    }
    // Fall through to COLUMNS env var.
    std::env::var("COLUMNS").ok().and_then(|s| s.parse().ok())
}

/// Render a box with `title` centered (or left-aligned) in the
/// top border, followed by each `row` as a `│ row │` line, then
/// the bottom border. The renderer right-pads each row to the
/// configured inner width and truncates with `…` if needed so
/// the borders always line up.
pub(crate) fn render_explain_box(ctx: &StreamContext, title: &str, rows: &[String]) {
    let inner = explain_inner_width();
    ctx.ui.log(&top_border(title, inner));
    for row in rows {
        ctx.ui.log(&middle_row(row, inner));
    }
    ctx.ui.log(&bottom_border(inner));
}

fn top_border(title: &str, inner: usize) -> String {
    // `┌─── <title> ` + dashes + `┐`
    let prefix = format!("─── {title} ");
    let prefix_chars = prefix.chars().count();
    let dashes = inner.saturating_sub(prefix_chars);
    format!("┌{}{}┐", prefix, "─".repeat(dashes))
}

fn middle_row(content: &str, inner: usize) -> String {
    // Pad/truncate to (inner - 2) so the row reads `│ <content> │`
    // with single-space gutters on both sides.
    let usable = inner.saturating_sub(2);
    let trimmed = truncate(content, usable);
    let width = trimmed.chars().count();
    let pad = usable.saturating_sub(width);
    format!("│ {}{} │", trimmed, " ".repeat(pad))
}

fn bottom_border(inner: usize) -> String {
    format!("└{}┘", "─".repeat(inner))
}

/// Render the `:schema`-namespace sidecar (if any) above the
/// page-by-page diagram. Dispatches on the top-level `kind`
/// field of the parsed JSON to pick between `MetadataSchema`
/// and `PredicateSchema` renderers — older schemas without
/// `kind` are interpreted as metadata for back-compat.
fn render_schema_sidecar(input_path: &std::path::Path, ctx: &StreamContext) {
    use vectordata::metadata_schema::{
        MetadataSchema, PredicateSchema, SCHEMA_KIND_METADATA, SCHEMA_KIND_PREDICATE,
        SCHEMA_NAMESPACE,
    };
    // Open the schema namespace. Absent → silently skip (the
    // slab is a content-only blob with no sidecar; that's a
    // valid shape, just no extra info to print).
    let schema_reader = match slabtastic::SlabReader::open_namespace(
        input_path, Some(SCHEMA_NAMESPACE),
    ) {
        Ok(r) => r,
        Err(_) => return,
    };
    if schema_reader.total_records() == 0 { return; }
    let bytes = match schema_reader.get(0) {
        Ok(b) => b,
        Err(_) => return,
    };
    // Peek at the JSON's top-level `kind` to route. Both
    // descriptors carry it; pre-discriminator records default
    // to "metadata".
    let kind = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(str::to_string))
        .unwrap_or_else(|| SCHEMA_KIND_METADATA.into());

    match kind.as_str() {
        SCHEMA_KIND_PREDICATE => match PredicateSchema::from_json_bytes(&bytes) {
            Ok(s) => {
                let rows = vec![
                    format!("kind: predicate"),
                    format!("wire_format: {}", s.wire_format),
                    format!("template:    {}", s.template),
                    format!("selectivity: {}", s.selectivity),
                    format!("count:       {}", s.count),
                    format!("seed:        {}", s.seed),
                    format!("version:     {}", s.version),
                ];
                render_explain_box(ctx, ":schema namespace", &rows);
                ctx.ui.log("");
            }
            Err(e) => ctx.ui.log(&format!("  :schema parse error (kind=predicate): {e}")),
        },
        // Metadata is the default interpretation for unknown kinds.
        _ => match MetadataSchema::from_json_bytes(&bytes) {
            Ok(s) => {
                let mut rows = vec![
                    format!("kind: metadata"),
                    format!("source: {}", s.source),
                    format!("fields: {}", s.fields.len()),
                ];
                if let Some(n) = s.record_count {
                    rows.push(format!("record_count: {n}"));
                }
                rows.push(format!("version: {}", s.version));
                render_explain_box(ctx, ":schema namespace", &rows);
                ctx.ui.log("");
                if !s.fields.is_empty() {
                    ctx.ui.log("  fields:");
                    for f in &s.fields {
                        let null = if f.nullable { " (nullable)" } else { "" };
                        ctx.ui.log(&format!("    {}: {}{}", f.name, f.type_name, null));
                    }
                    ctx.ui.log("");
                }
            }
            Err(e) => ctx.ui.log(&format!("  :schema parse error (kind=metadata): {e}")),
        },
    }
}

/// Render a one-box summary of the `:survey` namespace, if
/// present. The full SurveyReport JSON can be many MB on
/// real-world slabs; we surface only enough metadata to
/// confirm the survey is there and point the user at the right
/// tool to expand it (e.g. `slab get --namespace survey
/// --ordinal 0 | jq`).
fn render_survey_sidecar(input_path: &std::path::Path, ctx: &StreamContext) {
    use vectordata::metadata_schema::SURVEY_NAMESPACE;
    let reader = match slabtastic::SlabReader::open_namespace(
        input_path, Some(SURVEY_NAMESPACE),
    ) {
        Ok(r) => r,
        Err(_) => return,
    };
    if reader.total_records() == 0 { return; }
    let bytes = match reader.get(0) {
        Ok(b) => b,
        Err(_) => return,
    };
    // Parse just enough to surface field count + source.
    // Full report retrieval is outside `slab explain`'s
    // mandate — this is a self-describing-summary view, not a
    // pretty-printer.
    let summary = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .map(|v| {
            let fields = v.get("fields")
                .and_then(|f| f.as_object())
                .map(|m| m.len())
                .unwrap_or(0);
            let source = v.get("source")
                .and_then(|s| s.get("path"))
                .and_then(|p| p.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| "(unknown)".into());
            let total = v.get("source")
                .and_then(|s| s.get("total_records"))
                .and_then(|n| n.as_u64())
                .unwrap_or(0);
            let sampled = v.get("source")
                .and_then(|s| s.get("sampled_records"))
                .and_then(|n| n.as_u64())
                .unwrap_or(0);
            (fields, source, total, sampled)
        });
    let rows = match summary {
        Some((fields, source, total, sampled)) => vec![
            format!("source:      {}", source),
            format!("fields:      {}", fields),
            format!("records:     {} total, {} sampled", total, sampled),
            format!("payload:     {} bytes (JSON)", bytes.len()),
        ],
        None => vec![format!("payload:     {} bytes (unparseable as SurveyReport)", bytes.len())],
    };
    render_explain_box(ctx, ":survey namespace", &rows);
    ctx.ui.log("");
}

/// Truncate a string for fixed-width box rendering.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max { s.to_string() } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "List namespace IDs present in a slab file".into(),
            body: format!(
                r#"# slab namespaces

List namespace IDs present in a slab file.

## Description

Scans a slab file and lists all distinct namespace identifiers found
in its header structure. Each namespace represents an independent
collection of records within the same physical file, identified by
name and index.

## How It Works

The command calls the slabtastic library's `list_namespaces` function,
which reads the slab file header to enumerate all registered
namespaces. For each namespace it reports the namespace index, name,
and the file offset of its pages page (the page directory for that
namespace). This is a metadata-only operation that does not read any
data pages.

## Data Preparation Role

`slab namespaces` is used to discover the organization of a slab file
that contains multiple logical datasets. When a pipeline produces a
multi-namespace slab (for example, separate namespaces for different
metadata facets), this command lets you verify which namespaces are
present and confirm that the expected structure was created. It is also
useful for orienting yourself when working with an unfamiliar slab file.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);

        let namespaces = match SlabReader::list_namespaces(&input_path) {
            Ok(ns) => ns,
            Err(e) => return error_result(format!("failed to list namespaces: {}", e), start),
        };

        ctx.ui.log(&format!("{:<6} {:<30} {:>14}", "Index", "Name", "PagesOffset"));
        ctx.ui.log(&"-".repeat(54).to_string());

        for ns in &namespaces {
            ctx.ui.log(&format!(
                "{:<6} {:<30} {:>14}",
                ns.namespace_index,
                ns.name,
                ns.pages_page_offset,
            ));
        }

        ctx.ui.log("");
        ctx.ui.log(&format!("{} namespace(s)", namespaces.len()));

        CommandResult {
            status: Status::Ok,
            message: format!("{} namespaces", namespaces.len()),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("source", "Path", true, None, "Slab file", OptionRole::Input),
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_SLAB
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Inspect slab file header and page structure".into(),
            body: format!(
                r#"# slab inspect

Inspect slab file header and page structure.

## Description

Decodes and renders slab records as structured ANode vernacular text.
Given a set of ordinals, retrieves each record, decodes it using the
specified codec (auto, mnode, or pnode), and renders the decoded
structure in the chosen vernacular format.

## How It Works

The command opens the slab file and performs random-access lookups for
each requested ordinal. Each record's raw bytes are passed through the
ANode decoder, which interprets the binary data as either an MNode
(metadata node) or PNode (predicate node) depending on the codec
setting. The decoded tree structure is then rendered in the chosen
vernacular format -- options include CDDL schema notation, SQL, CQL,
JSON, YAML, a human-readable readout, and a compact display format.

## Data Preparation Role

`slab inspect` is the primary tool for examining the logical content
of metadata and predicate slab records at the field level. Unlike
`slab get` (which shows raw bytes or text), `slab inspect` understands
the ANode encoding and presents records as structured data. This makes
it essential for verifying that imported metadata has the expected
schema and values, for debugging predicate synthesis, and for
understanding the data model that downstream filtered-KNN queries
will operate on.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
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

        let vernacular = match veks_core::formats::anode_vernacular::Vernacular::parse(format_str) {
            Some(v) => v,
            None => return error_result(format!("unknown format: '{}'", format_str), start),
        };

        let ordinals = match parse_ordinals(ordinals_str) {
            Ok(o) => o,
            Err(e) => return error_result(e, start),
        };

        let reader = match open_slab_display(&input_path, &ctx.ui) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", input_path.display(), e), start),
        };

        let mut found = 0;
        let mut errors = 0;

        for ordinal in &ordinals {
            match reader.get(*ordinal) {
                Ok(data) => {
                    let decode_result = match codec {
                        "mnode" => veks_core::formats::anode::decode_mnode(&data),
                        "pnode" => veks_core::formats::anode::decode_pnode(&data),
                        _ => veks_core::formats::anode::decode(&data),
                    };
                    match decode_result {
                        Ok(anode) => {
                            let rendered = veks_core::formats::anode_vernacular::render(&anode, vernacular);
                            ctx.ui.log(&format!("[{}]: {}", ordinal, rendered));
                            found += 1;
                        }
                        Err(e) => {
                            ctx.ui.log(&format!("[{}]: DECODE ERROR: {}", ordinal, e));
                            errors += 1;
                        }
                    }
                }
                Err(e) => {
                    ctx.ui.log(&format!("[{}]: NOT FOUND ({})", ordinal, e));
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
            opt("source", "Path", true, None, "Slab file", OptionRole::Input),
            opt("ordinals", "String", true, None, "Comma-separated ordinals or ranges (e.g. 0,1,5..10)", OptionRole::Config),
            opt("codec", "enum", false, Some("auto"), "Codec: auto, mnode, pnode", OptionRole::Config),
            opt("format", "enum", false, Some("cddl"), "Vernacular format: cddl, sql, cql, json, jsonl, yaml, readout, display", OptionRole::Config),
        ]
    }
}

/// Sample `n` page indices from a `total`-page slab at evenly-
/// spaced intervals. Used by the survey orchestrator to pick the
/// pages it will read on a sparse pass.
pub(crate) fn sample_page_indices(total: usize, n: usize) -> Vec<usize> {
    if n == 0 || total == 0 {
        return vec![];
    }
    if n >= total {
        return (0..total).collect();
    }
    let step = total as f64 / n as f64;
    (0..n).map(|i| (i as f64 * step) as usize).collect()
}

/// Load a [`SurveyReport`] from a `survey.json` written by
/// `analyze survey`. Same JSON contract — no legacy projection.
pub(crate) fn survey_report_from_json(
    path: &Path,
) -> Result<crate::pipeline::commands::survey::SurveyReport, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse survey JSON: {}", e))
}

/// Inline-run the survey orchestrator and return the raw
/// [`SurveyReport`] without legacy projection. The richer shape
/// (semantic types, cardinality regimes, quantile sketches, heavy
/// hitters, reservoir samples) is what `generate predicates`
/// consumes directly.
pub(crate) fn survey_report_inline(
    path: &Path,
    samples: usize,
    ui: Option<&veks_core::ui::UiHandle>,
) -> Result<crate::pipeline::commands::survey::SurveyReport, String> {
    let cfg = crate::pipeline::commands::survey::SurveyConfig {
        samples,
        ..Default::default()
    };
    crate::pipeline::commands::survey::survey(path, &cfg, ui)
}

// -- Helpers -------------------------------------------------------------------

/// Open a slab file with progress feedback on stderr.
///
/// Wraps [`SlabReader::open_with_progress`] so that large files report
/// page index scan progress during the open.
fn open_slab(path: &Path) -> slabtastic::Result<SlabReader> {
    open_slab_with_ui(path, None)
}

fn open_slab_display(path: &Path, ui: &veks_core::ui::UiHandle) -> slabtastic::Result<SlabReader> {
    open_slab_with_ui(path, Some(ui))
}

pub(crate) fn open_slab_with_ui(path: &Path, ui: Option<&veks_core::ui::UiHandle>) -> slabtastic::Result<SlabReader> {
    let pb: std::cell::RefCell<Option<veks_core::ui::ProgressHandle>> = std::cell::RefCell::new(None);

    SlabReader::open_with_progress(path, |p| {
        match p {
            OpenProgress::PagesPageRead { page_count } => {
                if let Some(u) = ui {
                    let bar = u.bar(*page_count as u64, "slab index");
                    *pb.borrow_mut() = Some(bar);
                } else {
                    log::info!("slab index: {} pages to scan", page_count);
                }
            }
            OpenProgress::IndexBuild { done, total } => {
                if let Some(ref bar) = *pb.borrow() {
                    bar.set_position(*done as u64);
                } else {
                    log::info!("slab index: {}/{} pages scanned", done, total);
                }
            }
            OpenProgress::IndexComplete { total_records } => {
                if let Some(bar) = pb.borrow_mut().take() {
                    bar.finish();
                }
                if let Some(u) = ui {
                    u.log(&format!("    slab index: complete, {} total records", total_records));
                } else {
                    log::info!("slab index: complete, {} total records", total_records);
                }
            }
        }
    })
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

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(), extended_description: None,
        role,
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
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
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
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
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
        let reader = SlabReader::open(ws.join("output.slab")).unwrap();
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
        opts.set("source", slab_path.to_string_lossy().to_string());
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
        opts.set("source", slab_path.to_string_lossy().to_string());
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
        opts.set("source", slab_path.to_string_lossy().to_string());
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
        opts.set("source", slab_path.to_string_lossy().to_string());
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
        opts.set("source", slab_path.to_string_lossy().to_string());
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
        opts.set("source", slab_path.to_string_lossy().to_string());
        let mut op = SlabExplainOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    /// The width-aware box renderer produces lines of equal
    /// visible width so the borders line up regardless of
    /// content length. The renderer truncates rows that exceed
    /// the inner width with `…` rather than letting them
    /// overflow.
    #[test]
    fn explain_box_borders_line_up() {
        let inner = 40;
        let top = top_border("Page 0", inner);
        let bot = bottom_border(inner);
        let mid_short = middle_row("hi", inner);
        let mid_long = middle_row(&"x".repeat(200), inner);
        // All four lines must have the same display width.
        let w = |s: &str| s.chars().count();
        assert_eq!(w(&top), w(&bot), "top vs bottom border");
        assert_eq!(w(&mid_short), w(&top), "short row vs border");
        assert_eq!(w(&mid_long), w(&top), "long-truncated row vs border");
        // The truncated long row must end in `…` to signal
        // truncation, with the `│` border still in place.
        assert!(mid_long.ends_with(" │"), "right border must close: {mid_long}");
        assert!(mid_long.contains('…'), "truncated row needs ellipsis: {mid_long}");
    }

    /// `slab explain` surfaces the `:schema`-namespace sidecar
    /// above the page-by-page dump. With a predicate slab the
    /// rendered output must announce `kind: predicate` plus the
    /// template / wire_format so the operator doesn't have to
    /// interpret raw hex bytes to know what's in the file.
    #[test]
    fn slab_explain_surfaces_predicate_schema_sidecar() {
        use vectordata::metadata_schema::{PredicateSchema, SCHEMA_NAMESPACE};
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Build a slab with two record-bytes and a schema sidecar.
        let path = ws.join("preds.slab");
        let config = slabtastic::WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut w = slabtastic::SlabWriter::new(&path, config).unwrap();
        w.add_record(b"\x02\xffrecord-bytes-0").unwrap();
        w.add_record(b"\x02\xffrecord-bytes-1").unwrap();
        w.start_namespace(SCHEMA_NAMESPACE).unwrap();
        let schema = PredicateSchema::new(
            "(age >= ? AND name MATCHES ?)",
            "0.05..0.20",
            42,
            2,
        );
        w.add_record(&schema.to_json_bytes()).unwrap();
        w.finish().unwrap();

        // Capture the UI output.
        let sink = std::sync::Arc::new(veks_core::ui::TestSink::new());
        ctx.ui = veks_core::ui::UiHandle::new(sink.clone());

        let mut opts = Options::new();
        opts.set("source", path.to_string_lossy().to_string());
        let mut op = SlabExplainOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let out = sink.log_messages().join("\n");
        assert!(out.contains(":schema namespace"),
            "explain output must mention the :schema namespace, got:\n{out}");
        assert!(out.contains("kind: predicate"),
            "explain output must surface kind=predicate, got:\n{out}");
        assert!(out.contains("pnode:named"),
            "explain output must surface wire_format=pnode:named, got:\n{out}");
        // The template is longer than the truncation width so
        // the full form should appear on its own line.
        assert!(out.contains("(age >= ? AND name MATCHES ?)"),
            "explain output must include the full template:\n{out}");
    }
}
