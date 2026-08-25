// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks-study` — run the models and print what they measured.
//!
//! Everything this prints is reproducible from the crate's tests; the
//! binary exists so a sweep can be asked for by name rather than by
//! editing a test.

use std::collections::HashMap;
use veks_studies::model::{Geometry, Map};
use veks_studies::sweep::{self, Axis, Baseline, Config};
use veks_studies::{study, validate};

/// A minimal `--flag value` parser. The surface here is small enough that
/// a dependency would cost more than it saves.
struct Args {
    positional: Vec<String>,
    flags: HashMap<String, String>,
}

impl Args {
    fn parse() -> Args {
        let mut positional = Vec::new();
        let mut flags = HashMap::new();
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            if let Some(name) = arg.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    flags.insert(k.to_string(), v.to_string());
                } else {
                    flags.insert(name.to_string(), it.next().unwrap_or_default());
                }
            } else {
                positional.push(arg);
            }
        }
        Args { positional, flags }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(|s| s.as_str())
    }

    fn size(&self, name: &str, fallback: u64) -> u64 {
        self.get(name).and_then(parse_size).unwrap_or(fallback)
    }

    fn count(&self, name: &str, fallback: usize) -> usize {
        self.get(name)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(fallback)
    }

    fn frac(&self, name: &str, fallback: f64) -> f64 {
        self.get(name)
            .and_then(|v| {
                let trimmed = v.trim_end_matches('%');
                trimmed
                    .parse::<f64>()
                    .ok()
                    .map(|n| if v.ends_with('%') { n / 100.0 } else { n })
            })
            .unwrap_or(fallback)
    }
}

/// Accept `4096`, `4k`, `2MiB`, `1G`. Sizes are the most-typed argument
/// here and spelling them out every time is a papercut.
fn parse_size(s: &str) -> Option<u64> {
    let t = s.trim().to_lowercase();
    let (digits, scale) = if let Some(d) = t.strip_suffix("gib").or_else(|| t.strip_suffix('g')) {
        (d.to_string(), 1u64 << 30)
    } else if let Some(d) = t.strip_suffix("mib").or_else(|| t.strip_suffix('m')) {
        (d.to_string(), 1u64 << 20)
    } else if let Some(d) = t.strip_suffix("kib").or_else(|| t.strip_suffix('k')) {
        (d.to_string(), 1u64 << 10)
    } else {
        (t.clone(), 1)
    };
    digits
        .trim()
        .parse::<f64>()
        .ok()
        .map(|n| (n * scale as f64) as u64)
}

fn usage() -> String {
    let mut s = String::from(
        "veks-study — simulate SPLAT rewrites and the storage path under them\n\
         \n\
         USAGE\n  \
           veks-study <command> [options]\n\
         \n\
         COMMANDS\n  \
           sweep <axis>   vary one parameter and show how the metrics move\n  \
           validate       score the storage model against measurement\n  \
           devices        list the modelled devices\n  \
           report         the full standing report\n  \
           help           this text\n\
         \n\
         SWEEP AXES\n",
    );
    for (name, description) in Axis::all() {
        s.push_str(&format!("  {name:<12} {description}\n"));
    }
    s.push_str(
        "\nOPTIONS\n  \
           --device <name>       which device (see `devices`)\n  \
           --block <size>        request size, e.g. 4k, 128k\n  \
           --depth <n>           offered queue depth\n  \
           --cores <n>           host cores issuing I/O\n  \
           --records <n>         records in the rewrite\n  \
           --record <size>       bytes per record\n  \
           --container <size>    container size\n  \
           --budget <pct>        memory budget as a share of payload, e.g. 25%\n  \
           --page <size>         page cache granularity\n  \
           --ram <pct>           page cache size as a share of payload\n  \
           --samples <n>         requests simulated per point\n  \
           --vs first|prev|none  what each delta is measured against\n\
         \n\
         EXAMPLES\n  \
           veks-study sweep depth --device nvme-modern --cores 8\n  \
           veks-study sweep record --device nvme-consumer --depth 128\n  \
           veks-study sweep budget --records 200000 --record 512 --vs prev\n  \
           veks-study sweep readahead --record 512\n  \
           veks-study validate\n",
    );
    s
}

fn devices() -> String {
    use std::fmt::Write as _;
    let mut s = String::from("\n  modelled devices\n");
    let _ = writeln!(
        s,
        "\n  {:<16} {:>10} {:>10} {:>7} {:>7}  grounding",
        "name", "seq MB/s", "cmd rate", "dies", "queue"
    );
    for h in veks_studies::io::hw::ALL_HARDWARE_WITH_MODERN {
        let grounding = if h.name == "nvme-modern" {
            "published figures, not swept here"
        } else {
            "perfscripts fio corpus"
        };
        let _ = writeln!(
            s,
            "  {:<16} {:>10.0} {:>10.0} {:>7} {:>7}  {}",
            h.name,
            h.sequential_bandwidth() / 1e6,
            h.max_command_rate,
            h.dies,
            h.queue_slots,
            grounding
        );
    }
    s.push_str(
        "\n  Every historical figure is a statement about iodepth=10 on 2016\n  \
         hardware. `veks-study sweep depth` shows how much that matters.\n",
    );
    s
}

fn report() {
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
    print!(
        "{}",
        study::render_comparison(&study::compare_all(geometry, &map, live / 8))
    );

    println!("\n\n=== validation against measurement ===");
    print!("{}", validate::render(&validate::score_all()));

    println!("\n\n=== where ordering starts to pay ===");
    print!(
        "{}",
        veks_studies::device::render_crossover_table(143 * (1u64 << 30))
    );

    println!("\n\n=== and how that line moves with concurrency ===");
    print!(
        "{}",
        veks_studies::device::render_concurrency_crossover(143 * (1u64 << 30), 4_096)
    );
    println!();
}

fn run_sweep(args: &Args) {
    let Some(name) = args.positional.get(1) else {
        eprintln!("sweep needs an axis. Known axes:\n");
        for (n, d) in Axis::all() {
            eprintln!("  {n:<12} {d}");
        }
        std::process::exit(2);
    };
    let Some(axis) = Axis::parse(name) else {
        eprintln!("unknown axis '{name}'. Try `veks-study help`.");
        std::process::exit(2);
    };

    let defaults = Config::default();
    let device = match args.get("device") {
        Some(requested) => {
            match veks_studies::io::hw::ALL_HARDWARE_WITH_MODERN
                .iter()
                .find(|h| h.name == requested)
            {
                Some(h) => h.name,
                None => {
                    eprintln!("unknown device '{requested}'. Try `veks-study devices`.");
                    std::process::exit(2);
                }
            }
        }
        None => defaults.device,
    };

    let config = Config {
        device,
        block_bytes: args.size("block", defaults.block_bytes),
        depth: args.count("depth", defaults.depth),
        cores: args.count("cores", defaults.cores),
        records: args.size("records", defaults.records),
        record_bytes: args.size("record", defaults.record_bytes),
        container_bytes: args.size("container", defaults.container_bytes),
        budget_fraction: args.frac("budget", defaults.budget_fraction),
        page_bytes: args.size("page", defaults.page_bytes),
        ram_fraction: args.frac("ram", defaults.ram_fraction),
        samples: args.size("samples", defaults.samples),
    };

    let baseline = match args.get("vs") {
        Some(v) => match Baseline::parse(v) {
            Some(b) => b,
            None => {
                eprintln!("--vs takes first, prev or none");
                std::process::exit(2);
            }
        },
        None => Baseline::First,
    };

    print!("{}", sweep::run(axis, &config).render(baseline));
    println!();
}

fn main() {
    let args = Args::parse();
    match args
        .positional
        .first()
        .map(|s| s.as_str())
        .unwrap_or("help")
    {
        "sweep" => run_sweep(&args),
        "validate" => {
            print!("{}", validate::render(&validate::score_all()));
            let write = validate::score_sequential_write();
            println!(
                "\n  {:<31} {:>8} {:>8.1}% {:>8.1}% {:>+8.1}%",
                "sequential write throughput",
                write.throughput.samples,
                write.throughput.mape * 100.0,
                write.throughput.worst * 100.0,
                write.throughput.bias * 100.0
            );
            println!(
                "  {:<31} {:>8} {:>8.1}% {:>8.1}% {:>+8.1}%",
                "sequential write latency",
                write.mean_latency.samples,
                write.mean_latency.mape * 100.0,
                write.mean_latency.worst * 100.0,
                write.mean_latency.bias * 100.0
            );
            println!();
        }
        "devices" => println!("{}", devices()),
        "report" => report(),
        _ => print!("{}", usage()),
    }
}
