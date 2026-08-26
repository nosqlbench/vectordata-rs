// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Measure what a ranged text-column read actually costs resident.
//!
//! `generate embed` holds one window's text in memory while that window
//! runs, so the window size is bounded by resident cost per row -- a
//! figure the parquet footer only gives a floor for. This runs the real
//! `read_text_column_range` path and reports peak RSS (VmHWM) against
//! the decoded byte count, so the gap between the two is visible rather
//! than assumed.
//!
//!   cargo run --release -p veks-core --example text_range_probe -- <file> <column> <start> <end>

use std::path::Path;

fn stat_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1).map(|v| v.to_string()))
        })
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("usage: text_range_probe <file.parquet> <column> <start> <end>");
        std::process::exit(2);
    }
    let (path, column) = (a[0].clone(), a[1].clone());
    let start: u64 = a[2].parse().expect("start");
    let end: u64 = a[3].parse().expect("end");

    let rss_before = stat_kb("VmRSS:");
    println!("before      : RSS {:.2} GB", rss_before as f64 / 1048576.0);
    println!("reading     : rows [{}, {}) of column '{}'", start, end, column);

    let t = std::time::Instant::now();
    let texts = veks_core::formats::passage_table::read_text_column_range(
        Path::new(&path),
        &column,
        start,
        Some(end),
    )
    .unwrap_or_else(|e| {
        eprintln!("read failed: {}", e);
        std::process::exit(1);
    });
    let elapsed = t.elapsed();

    let rows = texts.len() as u64;
    let bytes: usize = texts.iter().map(|s| s.len()).sum();
    let caps: usize = texts.iter().map(|s| s.capacity()).sum();
    let hwm = stat_kb("VmHWM:");
    let rss = stat_kb("VmRSS:");

    println!("rows        : {} in {:.1}s", rows, elapsed.as_secs_f64());
    println!(
        "text bytes  : {:.2} GB  ({} B/row)",
        bytes as f64 / 1e9,
        bytes as u64 / rows.max(1)
    );
    println!(
        "capacity    : {:.2} GB  ({:.2}x over len)",
        caps as f64 / 1e9,
        caps as f64 / bytes.max(1) as f64
    );
    println!("peak RSS    : {:.2} GB (VmHWM)", hwm as f64 / 1048576.0);
    println!("now RSS     : {:.2} GB", rss as f64 / 1048576.0);
    println!(
        "\nresident per row : {:.0} B   ({:.2}x the decoded {} B)",
        (hwm - rss_before) as f64 * 1024.0 / rows.max(1) as f64,
        (hwm - rss_before) as f64 * 1024.0 / bytes.max(1) as f64,
        bytes as u64 / rows.max(1)
    );
    // Extrapolate the window sizes that matter for planning passes.
    let per_row = (hwm - rss_before) as f64 * 1024.0 / rows.max(1) as f64;
    for w in [10_000_000u64, 20_000_000, 30_000_000, 50_000_000] {
        println!(
            "  {:>3}M row window -> ~{:.0} GB resident",
            w / 1_000_000,
            w as f64 * per_row / 1e9
        );
    }
}
