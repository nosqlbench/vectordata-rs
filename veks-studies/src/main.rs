// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks-study` — run the SPLAT models and print what they measured.

use veks_studies::model::{Geometry, Map};
use veks_studies::study;

fn main() {
    // A geometry whose container holds 32 records, so the amplification
    // crossover sits at P = 32 and a budget sweep crosses it.
    let geometry = Geometry::new(200_000, 4_096, 128 * 1024);
    let live = geometry.payload_bytes();

    println!("\n=== read amplification: measured against the published formula ===\n");
    let budgets: Vec<u64> = [2u64, 4, 8, 16, 32, 64, 128, 256]
        .iter()
        .map(|p| live / p)
        .collect();
    let rows = study::amplification_sweep(geometry, 20_260_824, &budgets);
    print!("{}", study::render_amplification(geometry, &rows));

    let worst = rows
        .iter()
        .map(|r| r.relative_error())
        .fold(0.0f64, f64::max);
    println!("\n  worst prediction error across the sweep: {:.1}%", worst * 100.0);

    println!("\n=== rewrites head to head at P = 8 ===");
    let map = Map::shuffled(geometry.records, 20_260_824);
    let rows = study::compare_all(geometry, &map, live / 8);
    print!("{}", study::render_comparison(&rows));

    println!(
        "\n  Both issue the same record reads. gsplat orders them, which is\n  \
         why its container touches are lower and its backward steps are zero.\n"
    );
}
