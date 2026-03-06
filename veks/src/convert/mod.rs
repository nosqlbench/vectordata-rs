// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

pub mod args;
pub mod order;

pub use args::ConvertArgs;

use std::fs;

use indicatif::{ProgressBar, ProgressStyle};
use log::info;

use crate::formats::VecFormat;
use crate::formats::reader;
use crate::formats::writer::{self, SinkConfig};

/// Entry point for the convert subcommand
pub fn run(args: ConvertArgs) {
    // Determine source format
    let source_format = match &args.from {
        Some(f) => *f,
        None => VecFormat::detect(&args.source).unwrap_or_else(|| {
            eprintln!(
                "Could not auto-detect source format for '{}'. Use --from to specify.",
                args.source.display()
            );
            std::process::exit(1);
        }),
    };

    if !args.to.is_writable() {
        eprintln!("{} is not a supported output format", args.to);
        std::process::exit(1);
    }

    eprintln!("Source format: {}", source_format);

    // Detect and report sort order for directory sources
    if args.source.is_dir() {
        let entries: Vec<_> = fs::read_dir(&args.source)
            .expect("failed to read source directory")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        let order_info = order::detect_sort_order(&entries);
        eprintln!("File ordering: {}", order_info);
    }

    // Open source reader
    let mut source = reader::open_source(&args.source, source_format, 0, None).unwrap_or_else(|e| {
        eprintln!("Failed to open source: {}", e);
        std::process::exit(1);
    });

    let dimension = source.dimension();
    let record_count = source.record_count();
    eprintln!(
        "Dimension: {}, records: {}",
        dimension,
        record_count.map_or("unknown".to_string(), |n| n.to_string())
    );

    // Open sink writer
    let sink_config = SinkConfig {
        dimension,
        source_format,
        slab_page_size: args.slab_page_size,
        slab_namespace: args.slab_namespace,
    };
    let mut sink =
        writer::open_sink(&args.output, args.to, &sink_config).unwrap_or_else(|e| {
            eprintln!("Failed to open sink: {}", e);
            std::process::exit(1);
        });

    // Progress bar
    let pb = if let Some(total) = record_count {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {pos}/{len} ({percent}%) records — {per_sec} — ETA {eta}")
                .expect("invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {pos} records — {per_sec}")
                .expect("invalid spinner template"),
        );
        pb
    };

    // Convert loop
    let mut ordinal: i64 = 0;
    while let Some(data) = source.next_record() {
        sink.write_record(ordinal, &data);
        ordinal += 1;
        pb.inc(1);
    }

    pb.finish_with_message("done");

    // Finalize the sink (writes pages page for slab, etc.)
    sink.finish().unwrap_or_else(|e| {
        eprintln!("Failed to finalize output: {}", e);
        std::process::exit(1);
    });

    info!("Converted {} records", ordinal);
    eprintln!("Wrote {} records to {}", ordinal, args.output.display());
}
