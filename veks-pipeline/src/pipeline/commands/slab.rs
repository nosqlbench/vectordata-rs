// Copyright (c) DataStax, Inc.
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

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
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
        ctx.ui.log(&format!("Imported {} records from {} to {}", count, from_path.display(), to_path.display()));

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

        let input_str = match options.require("input") {
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
            opt("input", "Path", true, None, "Source slab file", OptionRole::Input),
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

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);
        let verbose = options.get("verbose").map_or(false, |s| s == "true");

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
            opt("input", "Path", true, None, "Slab file to check", OptionRole::Input),
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
            opt("input", "Path", true, None, "Slab file", OptionRole::Input),
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

        let input_str = match options.require("input") {
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
                    ctx.ui.log(&format!("  Content type: {} (sampled {} records)", content_type, sample_count));
                }
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
            opt("input", "Path", true, None, "Slab file to analyze", OptionRole::Input),
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

        let reader = match open_slab_display(&input_path, &ctx.ui) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", input_path.display(), e), start),
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

                    ctx.ui.log(&format!("┌─── Page {} ───────────────────────────┐", i));
                    ctx.ui.log(&format!("│ Offset:    {:<28}│", entry.file_offset));
                    ctx.ui.log(&format!("│ Page size: {:<28}│", page_size));
                    ctx.ui.log(&format!("│ Ordinals:  {:<28}│",
                        format!("{}..{}", entry.start_ordinal, entry.start_ordinal + recs as i64)));
                    ctx.ui.log(&format!("│ Records:   {:<28}│", recs));

                    // Show first few record sizes
                    let preview_count = recs.min(5) as i64;
                    if preview_count > 0 {
                        ctx.ui.log("│ Record preview:                       │");
                        for j in 0..preview_count {
                            let ord = entry.start_ordinal + j;
                            if let Ok(data) = reader.get(ord) {
                                let hex_preview: String = data.iter().take(16)
                                    .map(|b| format!("{:02x}", b))
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                let suffix = if data.len() > 16 { "..." } else { "" };
                                ctx.ui.log(&format!("│   [{}]: {} bytes  {}{:<2}│",
                                    ord, data.len(), hex_preview, suffix));
                            }
                        }
                    }
                    ctx.ui.log("└───────────────────────────────────────┘");
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
            opt("input", "Path", true, None, "Slab file", OptionRole::Input),
            opt("pages", "String", false, None, "Comma-separated page indices to display", OptionRole::Config),
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

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);

        let namespaces = match SlabReader::list_namespaces(&input_path) {
            Ok(ns) => ns,
            Err(e) => return error_result(format!("failed to list namespaces: {}", e), start),
        };

        ctx.ui.log(&format!("{:<6} {:<30} {:>14}", "Index", "Name", "PagesOffset"));
        ctx.ui.log(&format!("{}", "-".repeat(54)));

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
            opt("input", "Path", true, None, "Slab file", OptionRole::Input),
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

        let vernacular = match veks_core::formats::anode_vernacular::Vernacular::from_str(format_str) {
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
            opt("input", "Path", true, None, "Slab file", OptionRole::Input),
            opt("ordinals", "String", true, None, "Comma-separated ordinals or ranges (e.g. 0,1,5..10)", OptionRole::Config),
            opt("codec", "enum", false, Some("auto"), "Codec: auto, mnode, pnode", OptionRole::Config),
            opt("format", "enum", false, Some("cddl"), "Vernacular format: cddl, sql, cql, json, jsonl, yaml, readout, display", OptionRole::Config),
        ]
    }
}

// -- Slab Survey ---------------------------------------------------------------

/// Pipeline command: sample slab records, decode as ANodes, and report per-field
/// value distribution statistics.
pub struct SlabSurveyOp;

/// Create a boxed `SlabSurveyOp` command.
pub fn survey_factory() -> Box<dyn CommandOp> {
    Box::new(SlabSurveyOp)
}

impl CommandOp for SlabSurveyOp {
    fn command_path(&self) -> &str {
        "analyze survey"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Survey metadata field value distributions".into(),
            body: format!(
                r#"# analyze survey

Survey metadata field value distributions.

## Description

Scans slab records, decodes them as ANode metadata, and reports
per-field statistics including value type breakdown, numeric ranges,
string length distributions, and distinct value cardinality. The
survey can sample a configurable number of records to keep execution
time manageable on large slabs.

## How It Works

The command opens the slab file and reads up to `samples` records in
page order. Each record is decoded as an ANode and its fields are
examined. For every field the command tracks: the number of non-null
values, a count of values by type tag, numeric min/max/mean for
numeric fields, string length min/max/mean for string fields, and a
bounded set of distinct values (up to `max-distinct`). Results are
printed to the console and optionally written to an output file in a
structured format suitable for further processing.

## Data Preparation Role

`survey` is a prerequisite for predicate synthesis. Before the pipeline
can generate predicates for filtered KNN queries, it needs to know
which metadata fields exist, what types they use, what value ranges
they span, and how many distinct values they contain. The survey output
drives decisions about which fields are suitable for predicate
generation, what bin boundaries to use for numeric fields, and whether
a field's cardinality is low enough for enumerated predicates. Running
`survey` early in the pipeline ensures that downstream `compute
predicates` steps have the schema and cardinality information they need.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let input_path = resolve_path(input_str, &ctx.workspace);

        let max_samples: usize = options
            .get("samples")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);
        let max_distinct: usize = options
            .get("max-distinct")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);

        let output_path = options.get("output").map(|s| resolve_path(s, &ctx.workspace));

        let survey = match survey_slab(&input_path, max_samples, max_distinct, Some(&ctx.ui)) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        // Print results
        ctx.ui.log(&format!(
            "Slab Survey: {} ({} of {} records sampled)",
            input_path.display(),
            survey.sampled,
            survey.total_records,
        ));

        for (name, fs) in &survey.field_stats {
            ctx.ui.log("");
            ctx.ui.log(&format!(
                "  Field \"{}\" — {} values ({} null)",
                name, fs.count, fs.null_count,
            ));

            // Type breakdown
            let mut type_parts: Vec<String> = Vec::new();
            for (tag, cnt) in &fs.type_counts {
                type_parts.push(format!("{} ({})", tag, cnt));
            }
            if !type_parts.is_empty() {
                ctx.ui.log(&format!("    types: {}", type_parts.join(", ")));
            }

            // Numeric stats
            if fs.numeric_count > 0 {
                let mean = fs.numeric_sum / fs.numeric_count as f64;
                ctx.ui.log(&format!(
                    "    range: {} .. {}   mean: {:.1}",
                    fs.numeric_min, fs.numeric_max, mean,
                ));
            }

            // String length stats
            if fs.strlen_count > 0 {
                let mean = fs.strlen_sum as f64 / fs.strlen_count as f64;
                ctx.ui.log(&format!(
                    "    strlen: {} .. {}   mean: {:.1}",
                    fs.strlen_min, fs.strlen_max, mean,
                ));
            }

            // Bytes length stats
            if fs.byteslen_count > 0 {
                let mean = fs.byteslen_sum as f64 / fs.byteslen_count as f64;
                ctx.ui.log(&format!(
                    "    byteslen: {} .. {}   mean: {:.1}",
                    fs.byteslen_min, fs.byteslen_max, mean,
                ));
            }

            // Distinct values
            if !fs.distinct.is_empty() {
                let overflow_msg = if fs.distinct_overflow {
                    " (overflow — more exist)"
                } else {
                    ""
                };
                ctx.ui.log(&format!(
                    "    {} distinct values{}",
                    fs.distinct.len(),
                    overflow_msg,
                ));
            }
        }

        if survey.non_mnode_count > 0 || survey.decode_errors > 0 {
            ctx.ui.log("");
            if survey.non_mnode_count > 0 {
                ctx.ui.log(&format!("  Non-MNode records: {}", survey.non_mnode_count));
            }
            if survey.decode_errors > 0 {
                ctx.ui.log(&format!("  Decode errors: {}", survey.decode_errors));
            }
        }

        // Write JSON output if requested
        let mut produced = vec![];
        if let Some(ref out_path) = output_path {
            match survey_to_json(&survey, out_path) {
                Ok(()) => {
                    produced.push(out_path.clone());
                }
                Err(e) => return error_result(format!("failed to write JSON: {}", e), start),
            }
        }

        // Build field names list for message
        let field_names: Vec<&str> = survey.field_stats.keys().map(|s| s.as_str()).collect();
        let message = format!(
            "{} records sampled, {} fields: {}",
            survey.sampled,
            field_names.len(),
            field_names.join(", "),
        );

        CommandResult {
            status: Status::Ok,
            message,
            produced,
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
                description: "Slab file to survey".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Optional JSON output file for survey results".to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "samples".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1000".to_string()),
                description: "Max records to sample".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "max-distinct".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("20".to_string()),
                description: "Max distinct values tracked per field".to_string(),
                role: OptionRole::Config,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["input"],
            &["output"],
        )
    }
}

/// Per-field distribution statistics accumulated during a slab survey.
pub(crate) struct FieldStats {
    pub(crate) count: usize,
    pub(crate) null_count: usize,
    pub(crate) type_counts: indexmap::IndexMap<String, usize>,
    pub(crate) numeric_min: f64,
    pub(crate) numeric_max: f64,
    pub(crate) numeric_sum: f64,
    pub(crate) numeric_count: usize,
    pub(crate) strlen_min: usize,
    pub(crate) strlen_max: usize,
    pub(crate) strlen_sum: usize,
    pub(crate) strlen_count: usize,
    pub(crate) byteslen_min: usize,
    pub(crate) byteslen_max: usize,
    pub(crate) byteslen_sum: usize,
    pub(crate) byteslen_count: usize,
    pub(crate) distinct: indexmap::IndexMap<String, usize>,
    pub(crate) distinct_overflow: bool,
}

impl FieldStats {
    pub(crate) fn new() -> Self {
        FieldStats {
            count: 0,
            null_count: 0,
            type_counts: indexmap::IndexMap::new(),
            numeric_min: f64::INFINITY,
            numeric_max: f64::NEG_INFINITY,
            numeric_sum: 0.0,
            numeric_count: 0,
            strlen_min: usize::MAX,
            strlen_max: 0,
            strlen_sum: 0,
            strlen_count: 0,
            byteslen_min: usize::MAX,
            byteslen_max: 0,
            byteslen_sum: 0,
            byteslen_count: 0,
            distinct: indexmap::IndexMap::new(),
            distinct_overflow: false,
        }
    }

    /// Observe a single field value, updating all tracked statistics.
    pub(crate) fn observe(&mut self, value: &veks_core::formats::mnode::MValue, max_distinct: usize) {
        use veks_core::formats::mnode::MValue;

        self.count += 1;

        if matches!(value, MValue::Null) {
            self.null_count += 1;
        }

        let tag_name = value.tag().to_string();
        *self.type_counts.entry(tag_name).or_insert(0) += 1;

        // Numeric stats
        let numeric_val: Option<f64> = match value {
            MValue::Int(v) => Some(*v as f64),
            MValue::Int32(v) => Some(*v as f64),
            MValue::Short(v) => Some(*v as f64),
            MValue::Float(v) => Some(*v),
            MValue::Float32(v) => Some(*v as f64),
            MValue::Half(v) => Some(half::f16::from_bits(*v).to_f64()),
            MValue::Millis(v) => Some(*v as f64),
            _ => None,
        };
        if let Some(v) = numeric_val {
            if v < self.numeric_min { self.numeric_min = v; }
            if v > self.numeric_max { self.numeric_max = v; }
            self.numeric_sum += v;
            self.numeric_count += 1;
        }

        // String length stats
        let str_val: Option<&str> = match value {
            MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s)
            | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => Some(s.as_str()),
            _ => None,
        };
        if let Some(s) = str_val {
            let len = s.len();
            if len < self.strlen_min { self.strlen_min = len; }
            if len > self.strlen_max { self.strlen_max = len; }
            self.strlen_sum += len;
            self.strlen_count += 1;
        }

        // Bytes length stats
        if let MValue::Bytes(b) = value {
            let len = b.len();
            if len < self.byteslen_min { self.byteslen_min = len; }
            if len > self.byteslen_max { self.byteslen_max = len; }
            self.byteslen_sum += len;
            self.byteslen_count += 1;
        }

        // Distinct value tracking
        if !self.distinct_overflow {
            let repr = format!("{}", value);
            if self.distinct.contains_key(&repr) {
                *self.distinct.get_mut(&repr).unwrap() += 1;
            } else if self.distinct.len() < max_distinct {
                self.distinct.insert(repr, 1);
            } else {
                self.distinct_overflow = true;
            }
        }
    }
}

/// Select `n` evenly-spaced page indices from `total` pages.
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

/// Result of surveying a slab file's field distributions.
pub(crate) struct SurveyResult {
    /// Per-field distribution statistics, keyed by field name.
    pub(crate) field_stats: indexmap::IndexMap<String, FieldStats>,
    /// Total records sampled.
    pub(crate) sampled: usize,
    /// Total records in the slab.
    pub(crate) total_records: usize,
    /// Number of non-MNode records encountered.
    pub(crate) non_mnode_count: usize,
    /// Number of decode errors encountered.
    pub(crate) decode_errors: usize,
}

/// Survey a slab file by sampling records and gathering per-field statistics.
///
/// Opens the slab, samples up to `max_samples` records across evenly-spaced
/// pages, decodes each as an MNode, and accumulates field statistics.
pub(crate) fn survey_slab(
    path: &Path,
    max_samples: usize,
    max_distinct: usize,
    ui: Option<&veks_core::ui::UiHandle>,
) -> Result<SurveyResult, String> {
    let reader = open_slab(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;

    let page_entries = reader.page_entries();
    let total_pages = page_entries.len();

    if total_pages == 0 {
        return Ok(SurveyResult {
            field_stats: indexmap::IndexMap::new(),
            sampled: 0,
            total_records: 0,
            non_mnode_count: 0,
            decode_errors: 0,
        });
    }

    // Use the cached page index for total record count — no extra I/O
    let total_records = reader.total_records() as usize;

    let desired_pages = if total_records == 0 {
        0
    } else {
        let avg_recs_per_page = total_records as f64 / total_pages as f64;
        let needed = (max_samples as f64 / avg_recs_per_page).ceil() as usize;
        needed.max(1).min(total_pages)
    };
    let sample_indices = sample_page_indices(total_pages, desired_pages);

    log::info!(
        "survey: sampling {} of {} pages ({} total records, target {} samples)",
        sample_indices.len(), total_pages, total_records, max_samples,
    );

    use veks_core::formats::anode;
    use veks_core::formats::anode::ANode;

    let mut field_stats: indexmap::IndexMap<String, FieldStats> = indexmap::IndexMap::new();
    let mut sampled = 0usize;
    let mut non_mnode_count = 0usize;
    let mut decode_errors = 0usize;
    let sample_page_count = sample_indices.len();
    let survey_start = std::time::Instant::now();
    let mut last_report = std::time::Instant::now();

    // Create a progress bar if a UI handle was provided
    let pb = ui.map(|u| u.bar_with_unit(max_samples as u64, "surveying slab records", "records"));

    'outer: for (progress_idx, &page_idx) in sample_indices.iter().enumerate() {
        if progress_idx > 0
            && (last_report.elapsed().as_secs() >= 5 || progress_idx + 1 == sample_page_count)
        {
            last_report = std::time::Instant::now();
            let elapsed = survey_start.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { sampled as f64 / elapsed } else { 0.0 };
            let pct = (progress_idx + 1) as f64 / sample_page_count as f64 * 100.0;
            let eta = if rate > 0.0 {
                let remaining_pages = sample_page_count - progress_idx - 1;
                let avg_recs_per_page = if progress_idx > 0 { sampled as f64 / progress_idx as f64 } else { 1.0 };
                let remaining_secs = remaining_pages as f64 * avg_recs_per_page / rate;
                if remaining_secs < 60.0 { format!(", ETA {:.0}s", remaining_secs) }
                else if remaining_secs < 3600.0 { format!(", ETA {:.1}m", remaining_secs / 60.0) }
                else { format!(", ETA {:.1}h", remaining_secs / 3600.0) }
            } else {
                String::new()
            };
            log::info!(
                "survey: {:.1}% ({}/{} pages, {} records, {:.0} rec/s{})",
                pct, progress_idx + 1, sample_page_count, sampled, rate, eta,
            );
        }

        let entry = &page_entries[page_idx];
        let page = match reader.read_data_page(entry) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let rec_count = page.record_count();
        for i in 0..rec_count {
            if sampled >= max_samples {
                break 'outer;
            }
            let data = match page.get_record(i) {
                Some(d) => d,
                None => continue,
            };
            match anode::decode(data) {
                Ok(ANode::MNode(node)) => {
                    for (name, value) in &node.fields {
                        let fs = field_stats
                            .entry(name.clone())
                            .or_insert_with(FieldStats::new);
                        fs.observe(value, max_distinct);
                    }
                }
                Ok(ANode::PNode(_)) => {
                    non_mnode_count += 1;
                }
                Err(_) => {
                    decode_errors += 1;
                }
            }
            sampled += 1;
            if sampled % 100_000 == 0 {
                if let Some(ref p) = pb { p.set_position(sampled as u64); }
            }
        }
    }

    if let Some(ref p) = pb { p.finish(); }
    log::info!("survey: complete, {} records sampled", sampled);

    Ok(SurveyResult {
        field_stats,
        sampled,
        total_records,
        non_mnode_count,
        decode_errors,
    })
}

/// Write survey results to a JSON file.
///
/// Produces a JSON object with `sampled`, `total_records`, and a `fields`
/// object containing per-field statistics (types, numeric range, string
/// lengths, distinct values, etc.).
fn survey_to_json(survey: &SurveyResult, path: &Path) -> Result<(), String> {
    use serde_json::{Map, Number, Value};

    let mut root = Map::new();
    root.insert("sampled".into(), Value::Number(Number::from(survey.sampled)));
    root.insert("total_records".into(), Value::Number(Number::from(survey.total_records)));
    root.insert("non_mnode_count".into(), Value::Number(Number::from(survey.non_mnode_count)));
    root.insert("decode_errors".into(), Value::Number(Number::from(survey.decode_errors)));

    let mut fields = Map::new();
    for (name, fs) in &survey.field_stats {
        let mut field_obj = Map::new();
        field_obj.insert("count".into(), Value::Number(Number::from(fs.count)));
        field_obj.insert("null_count".into(), Value::Number(Number::from(fs.null_count)));

        let mut types = Map::new();
        for (tag, cnt) in &fs.type_counts {
            types.insert(tag.clone(), Value::Number(Number::from(*cnt)));
        }
        field_obj.insert("types".into(), Value::Object(types));

        if fs.numeric_count > 0 {
            let mut numeric = Map::new();
            if let Some(n) = Number::from_f64(fs.numeric_min) {
                numeric.insert("min".into(), Value::Number(n));
            }
            if let Some(n) = Number::from_f64(fs.numeric_max) {
                numeric.insert("max".into(), Value::Number(n));
            }
            let mean = fs.numeric_sum / fs.numeric_count as f64;
            if let Some(n) = Number::from_f64(mean) {
                numeric.insert("mean".into(), Value::Number(n));
            }
            numeric.insert("count".into(), Value::Number(Number::from(fs.numeric_count)));
            field_obj.insert("numeric".into(), Value::Object(numeric));
        }

        if fs.strlen_count > 0 {
            let mut strlen = Map::new();
            strlen.insert("min".into(), Value::Number(Number::from(fs.strlen_min)));
            strlen.insert("max".into(), Value::Number(Number::from(fs.strlen_max)));
            let mean = fs.strlen_sum as f64 / fs.strlen_count as f64;
            if let Some(n) = Number::from_f64(mean) {
                strlen.insert("mean".into(), Value::Number(n));
            }
            strlen.insert("count".into(), Value::Number(Number::from(fs.strlen_count)));
            field_obj.insert("strlen".into(), Value::Object(strlen));
        }

        if fs.byteslen_count > 0 {
            let mut byteslen = Map::new();
            byteslen.insert("min".into(), Value::Number(Number::from(fs.byteslen_min)));
            byteslen.insert("max".into(), Value::Number(Number::from(fs.byteslen_max)));
            let mean = fs.byteslen_sum as f64 / fs.byteslen_count as f64;
            if let Some(n) = Number::from_f64(mean) {
                byteslen.insert("mean".into(), Value::Number(n));
            }
            byteslen.insert("count".into(), Value::Number(Number::from(fs.byteslen_count)));
            field_obj.insert("byteslen".into(), Value::Object(byteslen));
        }

        if !fs.distinct.is_empty() {
            let mut distinct = Map::new();
            for (val, cnt) in &fs.distinct {
                distinct.insert(val.clone(), Value::Number(Number::from(*cnt)));
            }
            field_obj.insert("distinct".into(), Value::Object(distinct));
            field_obj.insert("distinct_overflow".into(), Value::Bool(fs.distinct_overflow));
        }

        fields.insert(name.clone(), Value::Object(field_obj));
    }
    root.insert("fields".into(), Value::Object(fields));

    let json = serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| format!("JSON serialization failed: {}", e))?;

    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create directory: {}", e))?;
        }
    }
    std::fs::write(path, json)
        .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;

    log::info!("Survey JSON written to {}", path.display());

    Ok(())
}

/// Load survey results from a JSON file previously written by [`survey_to_json`].
///
/// Reconstructs a [`SurveyResult`] with [`FieldStats`] populated from the JSON
/// fields used by predicate generation: `count`, `null_count`, `types`,
/// `numeric` (min/max), `distinct`, and `distinct_overflow`.
pub(crate) fn survey_from_json(path: &Path) -> Result<SurveyResult, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let root: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse JSON: {}", e))?;

    let sampled = root.get("sampled").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let total_records = root.get("total_records").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let non_mnode_count = root.get("non_mnode_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let decode_errors = root.get("decode_errors").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    let mut field_stats = indexmap::IndexMap::new();

    if let Some(fields) = root.get("fields").and_then(|v| v.as_object()) {
        for (name, fobj) in fields {
            let mut fs = FieldStats::new();

            fs.count = fobj.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            fs.null_count = fobj.get("null_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

            if let Some(types) = fobj.get("types").and_then(|v| v.as_object()) {
                for (tag, cnt) in types {
                    fs.type_counts.insert(tag.clone(), cnt.as_u64().unwrap_or(0) as usize);
                }
            }

            if let Some(numeric) = fobj.get("numeric").and_then(|v| v.as_object()) {
                fs.numeric_min = numeric.get("min").and_then(|v| v.as_f64()).unwrap_or(f64::INFINITY);
                fs.numeric_max = numeric.get("max").and_then(|v| v.as_f64()).unwrap_or(f64::NEG_INFINITY);
                let mean = numeric.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
                fs.numeric_count = numeric.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                fs.numeric_sum = mean * fs.numeric_count as f64;
            }

            if let Some(strlen) = fobj.get("strlen").and_then(|v| v.as_object()) {
                fs.strlen_min = strlen.get("min").and_then(|v| v.as_u64()).unwrap_or(usize::MAX as u64) as usize;
                fs.strlen_max = strlen.get("max").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                fs.strlen_count = strlen.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let mean = strlen.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
                fs.strlen_sum = (mean * fs.strlen_count as f64) as usize;
            }

            if let Some(distinct) = fobj.get("distinct").and_then(|v| v.as_object()) {
                for (val, cnt) in distinct {
                    fs.distinct.insert(val.clone(), cnt.as_u64().unwrap_or(0) as usize);
                }
            }
            fs.distinct_overflow = fobj.get("distinct_overflow").and_then(|v| v.as_bool()).unwrap_or(false);

            field_stats.insert(name.clone(), fs);
        }
    }

    Ok(SurveyResult {
        field_stats,
        sampled,
        total_records,
        non_mnode_count,
        decode_errors,
    })
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

fn open_slab_with_ui(path: &Path, ui: Option<&veks_core::ui::UiHandle>) -> slabtastic::Result<SlabReader> {
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
        description: desc.to_string(),
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
    fn test_slab_survey() {
        use veks_core::formats::mnode::MNode;
        use veks_core::formats::mnode::MValue;

        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Build MNode-encoded records
        let mut node1 = MNode::new();
        node1.insert("user_id".into(), MValue::Int(42));
        node1.insert("name".into(), MValue::Text("alice".into()));

        let mut node2 = MNode::new();
        node2.insert("user_id".into(), MValue::Int(99));
        node2.insert("name".into(), MValue::Text("bob".into()));

        let mut node3 = MNode::new();
        node3.insert("user_id".into(), MValue::Int(7));
        node3.insert("name".into(), MValue::Null);

        let rec1 = node1.to_bytes();
        let rec2 = node2.to_bytes();
        let rec3 = node3.to_bytes();

        let slab_path = create_test_slab(
            ws,
            "survey.slab",
            &[rec1.as_slice(), rec2.as_slice(), rec3.as_slice()],
        );

        let mut opts = Options::new();
        opts.set("input", slab_path.to_string_lossy().to_string());
        opts.set("samples", "100");
        let mut op = SlabSurveyOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "survey failed: {}", result.message);
        assert!(result.message.contains("user_id"), "message should mention user_id: {}", result.message);
        assert!(result.message.contains("name"), "message should mention name: {}", result.message);
        assert!(result.message.contains("3 records sampled"), "message: {}", result.message);
    }

    #[test]
    fn test_slab_survey_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let slab_path = create_test_slab(ws, "empty.slab", &[]);

        let mut opts = Options::new();
        opts.set("input", slab_path.to_string_lossy().to_string());
        let mut op = SlabSurveyOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "survey empty failed: {}", result.message);
        assert!(result.message.contains("0 records sampled"), "message: {}", result.message);
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
