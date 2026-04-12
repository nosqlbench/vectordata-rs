// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `slab append` subcommand — append data to an existing slabtastic file.
//!
//! Reads newline-delimited records from stdin (or a source file) and
//! appends them to an existing slab file via [`SlabWriter::append`].
//! Per the design doc, the existing file is verified to be well-formed
//! before appending (equivalent to the `check` command).

use std::io::{self, BufRead};

use crate::SlabWriter;

use super::{make_writer_config, ProgressReporter};

/// Run the `append` subcommand.
///
/// Reads newline-delimited records from `source` (or stdin if `None`)
/// and appends them to the existing slab file at `file`. The file is
/// checked for structural integrity before appending.
pub fn run(
    file: &str,
    source: Option<&str>,
    preferred_page_size: Option<u32>,
    min_page_size: Option<u32>,
    page_alignment: bool,
    progress: bool,
    namespace: &Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Verify the existing file is well-formed before appending
    println!("Verifying {file} before appending...");
    super::check::run(file, namespace)?;
    println!();

    let config = make_writer_config(preferred_page_size, min_page_size, page_alignment)?;
    let mut writer = SlabWriter::append_namespace(file, config, namespace.as_deref())?;

    let reporter = ProgressReporter::new(progress);
    let mut count: u64 = 0;

    match source {
        Some(path) => {
            let file = std::fs::File::open(path)?;
            let reader = io::BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                writer.add_record(line.as_bytes())?;
                count += 1;
                reporter.inc();
            }
        }
        None => {
            let stdin = io::stdin();
            let reader = stdin.lock();
            for line in reader.lines() {
                let line = line?;
                writer.add_record(line.as_bytes())?;
                count += 1;
                reporter.inc();
            }
        }
    }

    writer.finish()?;
    reporter.finish();
    println!("Appended {count} records to {file}");
    Ok(())
}
