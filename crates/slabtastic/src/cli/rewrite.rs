// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `slab rewrite` subcommand — rewrite a slabtastic file, reordering by
//! ordinal and repacking to new page settings in a single pass.

use crate::{SlabReader, SlabWriter};

use super::ordinal_range;
use super::{make_writer_config, write_with_buffer_rename, ProgressReporter};

/// Run the `rewrite` subcommand.
///
/// Reads records from the input file (optionally filtered by an ordinal
/// range), sorts them by ordinal for monotonicity, then writes them to a
/// new file with the requested page configuration. Dead pages and
/// alignment waste are eliminated. The output file is written to a
/// `.buffer` temporary file and renamed on success to ensure atomicity.
pub fn run(
    input: &str,
    output: &str,
    range: Option<&str>,
    preferred_page_size: Option<u32>,
    min_page_size: Option<u32>,
    page_alignment: bool,
    progress: bool,
    namespace: &Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = make_writer_config(preferred_page_size, min_page_size, page_alignment)?;

    let reader = SlabReader::open_namespace(input, namespace.as_deref())?;
    let mut records = if let Some(range_str) = range {
        let (start, end) = ordinal_range::parse_ordinal_range(range_str)?;
        reader.iter_range(start, end)?
    } else {
        reader.iter()?
    };

    // Check if already monotonic before sorting
    let already_monotonic = records.windows(2).all(|w| w[0].0 <= w[1].0);

    // Sort by ordinal to ensure monotonicity
    records.sort_by_key(|&(ordinal, _)| ordinal);

    let record_count = records.len();
    let reporter = ProgressReporter::new(progress);

    write_with_buffer_rename(output, |buf_path| {
        let mut writer = SlabWriter::new(buf_path, config)?;
        for (_ordinal, data) in &records {
            writer.add_record(data)?;
            reporter.inc();
        }
        writer.finish()?;
        Ok(())
    })?;
    reporter.finish();

    if already_monotonic {
        println!("Input was already monotonically ordered.");
    } else {
        println!("Input was NOT monotonically ordered; records have been sorted.");
    }

    println!(
        "Rewrote {} records from {} to {}",
        record_count, input, output
    );

    Ok(())
}
