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
    println!(
        "\n  worst prediction error across the sweep: {:.1}%",
        worst * 100.0
    );

    println!("\n=== rewrites head to head at P = 8 ===");
    let map = Map::shuffled(geometry.records, 20_260_824);
    let rows = study::compare_all(geometry, &map, live / 8);
    print!("{}", study::render_comparison(&rows));

    println!(
        "\n  Both issue the same record reads. gsplat orders them, which is\n  \
         why its container touches are lower and its backward steps are zero.\n"
    );

    println!("=== device models against the measurements they claim to explain ===");
    for (model, regime) in veks_studies::device::paired() {
        let report = model.fit(regime);
        println!(
            "\n  {:<15} median error {:>5.1}%   worst at or below 128 KiB {:>5.1}%",
            report.model,
            report.median_error() * 100.0,
            report.worst_error_up_to(131_072) * 100.0
        );
        println!("  {:<15} {}", "", regime.source);
    }

    println!("\n\n=== worked examples, priced against those models ===");
    let examples = [
        study::WorkedExample {
            label: "B — 100M records of 1.5 KiB",
            records: 100_000_000,
            record_bytes: 1_536,
            container_bytes: 128 * 1024,
            budget_bytes: 8 * 1024 * 1024 * 1024,
        },
        study::WorkedExample {
            label: "C — 450M records of 4 KiB, tight budget",
            records: 450_000_000,
            record_bytes: 4_096,
            container_bytes: 128 * 1024,
            budget_bytes: 32 * 1024 * 1024 * 1024,
        },
        study::WorkedExample {
            label: "C — 450M records of 4 KiB, ample budget",
            records: 450_000_000,
            record_bytes: 4_096,
            container_bytes: 128 * 1024,
            budget_bytes: 230 * 1024 * 1024 * 1024,
        },
        study::WorkedExample {
            label: "B — same rewrite, budget raised above the NVMe line",
            records: 100_000_000,
            record_bytes: 1_536,
            container_bytes: 128 * 1024,
            budget_bytes: 32 * 1024 * 1024 * 1024,
        },
        // The vectordata-specific cases: fvec records carry a 4-byte
        // dimension header, so 384-d and 1024-d f32 are 1540 B and 4100 B.
        study::WorkedExample {
            label: "sysref B — 100M × 384-d f32 (R = 1540 B)",
            records: 100_000_000,
            record_bytes: 1_540,
            container_bytes: 128 * 1024,
            budget_bytes: 8 * 1024 * 1024 * 1024,
        },
        study::WorkedExample {
            label: "sysref C — 450M × 1024-d f32 (R = 4100 B), M = 32.7 GiB",
            records: 450_000_000,
            record_bytes: 4_100,
            container_bytes: 128 * 1024,
            budget_bytes: 35_112_000_000,
        },
        study::WorkedExample {
            label: "sysref C — 450M × 1024-d f32 (R = 4100 B), M = 230 GiB",
            records: 450_000_000,
            record_bytes: 4_100,
            container_bytes: 128 * 1024,
            budget_bytes: 230 * 1024 * 1024 * 1024,
        },
    ];
    print!("{}", study::render_worked_examples(&examples));

    println!("\n\n=== where ordering starts to pay ===");
    print!(
        "{}",
        veks_studies::device::render_crossover_table(143 * (1u64 << 30))
    );
    println!();
}
