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

    println!("\n\n=== forward simulation against the measured sweeps ===");
    println!("  No throughput formula: a clock advanced through positioning and");
    println!("  transfer against a shared bandwidth ceiling, a finite command");
    println!("  queue and a serial controller.\n");
    println!(
        "  {:<15} {:>12} {:>12} {:>12}",
        "device", "worst <=1MB", "sequential", "vs measured"
    );
    for (hardware, regime) in veks_studies::io::hw::ALL_HARDWARE
        .iter()
        .zip(veks_studies::regime::ALL.iter())
    {
        let worst = regime
            .random_read
            .iter()
            .filter(|p| p.block_bytes <= 1 << 20)
            .map(|p| {
                let n = if p.block_bytes >= 1 << 20 { 300 } else { 2_000 };
                let s = veks_studies::io::fio_like(hardware, p.block_bytes, n);
                ((s.iops() - p.iops as f64) / p.iops as f64).abs()
            })
            .fold(0.0f64, f64::max);
        let seq = veks_studies::io::fio_like_sequential(hardware, 1 << 20, 1_500);
        println!(
            "  {:<15} {:>11.1}% {:>9.0} MB/s {:>9.0} MB/s",
            hardware.name,
            worst * 100.0,
            seq.throughput() / 1e6,
            regime.seq_read.bytes_per_s() as f64 / 1e6
        );
    }

    println!("\n  Where a device's time goes, 4 KiB reads:\n");
    println!(
        "  {:<15} {:>12} {:>14} {:>14}",
        "device", "order", "positioning", "bandwidth used"
    );
    for hardware in veks_studies::io::hw::ALL_HARDWARE {
        for (label, s) in [
            (
                "scattered",
                veks_studies::io::fio_like(hardware, 4_096, 2_000),
            ),
            (
                "ascending",
                veks_studies::io::fio_like_sequential(hardware, 4_096, 20_000),
            ),
        ] {
            println!(
                "  {:<15} {:>12} {:>13.0}% {:>13.0}%",
                hardware.name,
                label,
                s.positioning_fraction() * 100.0,
                s.bandwidth_utilization() * 100.0
            );
        }
    }

    println!("\n  A concurrent writer, capped at 40 MB/s and uncapped:\n");
    println!(
        "  {:<15} {:>14} {:>14} {:>10}",
        "device", "reader capped", "reader free", "cost"
    );
    for hardware in veks_studies::io::hw::ALL_HARDWARE {
        let n = if hardware.name == "spinning-sata" {
            1_500
        } else {
            20_000
        };
        let capped = veks_studies::io::contended(hardware, 8_192, Some(40.0e6), n);
        let free = veks_studies::io::contended(hardware, 8_192, None, n);
        let c = capped.stream("reader").iops();
        let f = free.stream("reader").iops();
        println!(
            "  {:<15} {:>10.0} IOPS {:>9.0} IOPS {:>9.1}x",
            hardware.name,
            c,
            f,
            c / f.max(1e-9)
        );
    }

    println!("\n  What a modern drive changes, 4 KiB random reads:\n");
    println!(
        "  {:<15} {:>12} {:>14} {:>16}",
        "device", "IOPS", "bandwidth used", "limited by"
    );
    for (hardware, host) in [
        (
            &veks_studies::io::hw::NVME_CONSUMER_HW,
            veks_studies::io::hw::HostModel::cores(8),
        ),
        (
            &veks_studies::io::hw::NVME_MODERN_HW,
            veks_studies::io::hw::HostModel::DEFAULT,
        ),
        (
            &veks_studies::io::hw::NVME_MODERN_HW,
            veks_studies::io::hw::HostModel::cores(8),
        ),
    ] {
        const SPAN: u64 = 5 * 1024 * 1024 * 1024;
        let mut sched = veks_studies::io::sched::Noop::default();
        let mut issuer = veks_studies::io::RandomAccess::new(SPAN, 4_096, 200_000, 0xF10);
        let s = veks_studies::io::run(
            hardware,
            &mut sched,
            &mut issuer,
            veks_studies::io::RunConfig {
                host,
                ..veks_studies::io::RunConfig::direct(256, SPAN)
            },
        );
        let limit = if s.host_saturation() > 0.3 {
            "host CPU"
        } else {
            "device"
        };
        println!(
            "  {:<15} {:>12.0} {:>13.0}% {:>16}",
            format!("{} x{}", hardware.name, host.cores),
            s.iops(),
            s.bandwidth_utilization() * 100.0,
            limit
        );
    }

    println!("\n\n=== where ordering starts to pay ===");
    print!(
        "{}",
        veks_studies::device::render_crossover_table(143 * (1u64 << 30))
    );

    println!("\n\n=== validation against measurement ===");
    print!("{}", veks_studies::validate::render(&veks_studies::validate::score_all()));

    println!("\n\n=== and how that line moves with concurrency ===");
    print!(
        "{}",
        veks_studies::device::render_concurrency_crossover(143 * (1u64 << 30), 4_096)
    );
    println!();
}
