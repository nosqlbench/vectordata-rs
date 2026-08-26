// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Report parquet row-group and per-column sizing from the footer.
//!
//! Sizing a ranged read means knowing what a row costs in memory once
//! decoded, which the compressed file size does not tell you: text
//! columns routinely compress 3-4x, so a window that looks affordable
//! on disk can be several times the resident set. The footer carries
//! uncompressed sizes per column chunk, so this is a metadata-only
//! read -- no column data is decoded and the file is not scanned.
//!
//!   cargo run --release -p veks-core --example parquet_stats -- <file> [column]

use std::fs::File;
use std::path::Path;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

fn gb(bytes: i64) -> f64 {
    bytes as f64 / 1e9
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: parquet_stats <file.parquet> [column]");
            std::process::exit(2);
        }
    };
    let want_col = args.next();

    let file = File::open(Path::new(&path)).unwrap_or_else(|e| {
        eprintln!("open {}: {}", path, e);
        std::process::exit(1);
    });
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap_or_else(|e| {
        eprintln!("read parquet {}: {}", path, e);
        std::process::exit(1);
    });
    let meta = builder.metadata();

    let mut total_rows: i64 = 0;
    let mut total_comp: i64 = 0;
    let mut total_uncomp: i64 = 0;
    let mut min_rows = i64::MAX;
    let mut max_rows = 0i64;

    // Per-column totals, keyed by leaf column index.
    let schema = builder.parquet_schema();
    let ncols = schema.num_columns();
    let mut col_comp = vec![0i64; ncols];
    let mut col_uncomp = vec![0i64; ncols];

    for rg in meta.row_groups() {
        let rows = rg.num_rows();
        total_rows += rows;
        min_rows = min_rows.min(rows);
        max_rows = max_rows.max(rows);
        total_comp += rg.compressed_size();
        for (i, c) in rg.columns().iter().enumerate() {
            if i < ncols {
                col_comp[i] += c.compressed_size();
                col_uncomp[i] += c.uncompressed_size();
                total_uncomp += c.uncompressed_size();
            }
        }
    }

    let ngroups = meta.num_row_groups();
    println!("file        : {}", path);
    println!("row groups  : {}", ngroups);
    println!("rows        : {}", total_rows);
    if ngroups > 0 {
        println!(
            "rows/group  : min {}  max {}  mean {}",
            min_rows,
            max_rows,
            total_rows / ngroups as i64
        );
    }
    println!(
        "compressed  : {:.2} GB   uncompressed: {:.2} GB   ratio {:.2}x",
        gb(total_comp),
        gb(total_uncomp),
        total_uncomp as f64 / total_comp.max(1) as f64
    );

    println!("\nper column:");
    println!(
        "  {:<28} {:>12} {:>14} {:>10} {:>12}",
        "name", "comp GB", "uncomp GB", "ratio", "B/row"
    );
    for i in 0..ncols {
        let name = schema.column(i).name().to_string();
        if let Some(w) = &want_col
            && &name != w
        {
            continue;
        }
        println!(
            "  {:<28} {:>12.2} {:>14.2} {:>10.2} {:>12.0}",
            name,
            gb(col_comp[i]),
            gb(col_uncomp[i]),
            col_uncomp[i] as f64 / col_comp[i].max(1) as f64,
            col_uncomp[i] as f64 / total_rows.max(1) as f64,
        );
    }

    // What a ranged read of this column actually costs resident. The
    // decoded bytes are the floor; every row also carries a String
    // header in the collected Vec, and the allocator rounds each
    // heap block up, so the real figure runs above the raw byte count.
    if let Some(w) = &want_col {
        for i in 0..ncols {
            if schema.column(i).name() != w.as_str() {
                continue;
            }
            let per_row = col_uncomp[i] as f64 / total_rows.max(1) as f64;
            let with_overhead = per_row + 24.0;
            println!("\nresident cost of a ranged read of '{}':", w);
            println!(
                "  {:.0} B/row decoded + 24 B String header = ~{:.0} B/row",
                per_row, with_overhead
            );
            for window in [10_000_000u64, 20_000_000, 50_000_000] {
                println!(
                    "  {:>3}M rows -> ~{:.0} GB resident (floor)",
                    window / 1_000_000,
                    window as f64 * with_overhead / 1e9
                );
            }
        }
    }
}
