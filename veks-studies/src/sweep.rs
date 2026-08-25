// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Parameter sweeps with differential output.
//!
//! A single simulated number is nearly useless. What a cost model is for
//! is the *shape* of a dependence — how throughput moves with block size,
//! how the case for ordering erodes with concurrency, what another core
//! buys, where a page cache stops helping. So every sweep here reports a
//! delta column beside each metric, because the second derivative is
//! usually the thing being argued about.
//!
//! Each [`Axis`] varies exactly one parameter and holds the rest fixed,
//! and every sweep prints the held-fixed values in its header — a sweep
//! whose conditions are not stated is not reproducible, and most of the
//! wrong conclusions in `docs/gsplat/` came from forgetting which
//! conditions a number was measured under.
//!
//! Grounding for the device parameters these sweeps run against is in
//! [the crate bibliography](crate#sources).

use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather};
use crate::cache::CacheConfig;
use crate::io::{self, hw};
use crate::model::{Geometry, Map};

/// How a metric should be rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// Operations per second and similar counts.
    Count,
    /// Bytes per second, shown in MB/s.
    Rate,
    /// A fraction, shown as a percentage.
    Fraction,
    /// Seconds, shown in the most legible scale.
    Seconds,
    /// Seconds, shown in microseconds.
    Micros,
    /// A bare ratio, shown with an ×.
    Ratio,
}

impl Unit {
    fn render(self, v: f64) -> String {
        match self {
            Unit::Count => {
                if v >= 1e6 {
                    format!("{:.2}M", v / 1e6)
                } else if v >= 1e3 {
                    format!("{:.1}k", v / 1e3)
                } else {
                    format!("{v:.0}")
                }
            }
            Unit::Rate => format!("{:.0}", v / 1e6),
            Unit::Fraction => format!("{:.0}%", v * 100.0),
            Unit::Seconds => {
                if v >= 86_400.0 {
                    format!("{:.1}d", v / 86_400.0)
                } else if v >= 3_600.0 {
                    format!("{:.1}h", v / 3_600.0)
                } else if v >= 60.0 {
                    format!("{:.1}m", v / 60.0)
                } else if v >= 1.0 {
                    format!("{v:.2}s")
                } else {
                    format!("{:.0}ms", v * 1e3)
                }
            }
            Unit::Micros => {
                if v >= 1e-3 {
                    format!("{:.1}ms", v * 1e3)
                } else {
                    format!("{:.0}us", v * 1e6)
                }
            }
            Unit::Ratio => format!("{v:.2}x"),
        }
    }
}

/// One metric of one sweep point.
#[derive(Debug, Clone)]
pub struct Metric {
    pub name: &'static str,
    pub value: f64,
    pub unit: Unit,
}

/// One point on a sweep.
#[derive(Debug, Clone)]
pub struct Row {
    pub label: String,
    pub metrics: Vec<Metric>,
}

/// A completed sweep.
#[derive(Debug, Clone)]
pub struct Sweep {
    pub axis: String,
    pub held: Vec<(String, String)>,
    pub rows: Vec<Row>,
}

/// What each delta is measured against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Baseline {
    /// Against the first row — how far the axis has taken you.
    First,
    /// Against the row above — the local slope.
    Previous,
    /// No deltas.
    None,
}

impl Baseline {
    pub fn parse(s: &str) -> Option<Baseline> {
        match s {
            "first" => Some(Baseline::First),
            "prev" | "previous" => Some(Baseline::Previous),
            "none" => Some(Baseline::None),
            _ => None,
        }
    }
}

impl Sweep {
    /// Render as a table with a delta column beside every metric.
    pub fn render(&self, baseline: Baseline) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();

        let _ = writeln!(s, "\n  sweep: {}", self.axis);
        if !self.held.is_empty() {
            let held: Vec<String> = self.held.iter().map(|(k, v)| format!("{k}={v}")).collect();
            let _ = writeln!(s, "  held:  {}", held.join("  "));
        }
        if self.rows.is_empty() {
            let _ = writeln!(s, "  (no points)");
            return s;
        }

        let names: Vec<&'static str> = self.rows[0].metrics.iter().map(|m| m.name).collect();
        let label_width = self
            .rows
            .iter()
            .map(|r| r.label.len())
            .chain([self.axis.len()])
            .max()
            .unwrap_or(8)
            .max(8);

        let mut header = format!("\n  {:<label_width$}", self.axis);
        for name in &names {
            let _ = write!(header, "  {name:>10}");
            if baseline != Baseline::None {
                let _ = write!(header, " {:>7}", "d");
            }
        }
        let _ = writeln!(s, "{header}");

        for (i, row) in self.rows.iter().enumerate() {
            let mut line = format!("  {:<label_width$}", row.label);
            for (j, metric) in row.metrics.iter().enumerate() {
                let _ = write!(line, "  {:>10}", metric.unit.render(metric.value));
                if baseline != Baseline::None {
                    let reference = match baseline {
                        Baseline::First => self.rows.first(),
                        Baseline::Previous => {
                            if i == 0 {
                                None
                            } else {
                                self.rows.get(i - 1)
                            }
                        }
                        Baseline::None => None,
                    }
                    .and_then(|r| r.metrics.get(j))
                    .map(|m| m.value);

                    let delta = match reference {
                        Some(base) if base.abs() > 1e-12 && i > 0 => {
                            let pct = (metric.value - base) / base * 100.0;
                            if pct.abs() >= 1000.0 {
                                format!("{:+.0}x", metric.value / base)
                            } else {
                                format!("{pct:+.1}%")
                            }
                        }
                        _ => "—".to_string(),
                    };
                    let _ = write!(line, " {delta:>7}");
                }
            }
            let _ = writeln!(s, "{line}");
        }
        s
    }
}

/// Which parameter a sweep varies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Request size, against a device.
    Block,
    /// Offered queue depth.
    Depth,
    /// Across the modelled devices.
    Device,
    /// Host cores issuing I/O.
    Cores,
    /// Page cache page size, pricing a rewrite.
    Page,
    /// Page cache size, pricing a rewrite.
    Ram,
    /// Readahead policy.
    Readahead,
    /// Local versus remote node.
    Numa,
    /// Devices sharing one upstream link.
    Fabric,
    /// Record size, at the algorithm level.
    Record,
    /// Memory budget, at the algorithm level.
    Budget,
}

impl Axis {
    pub fn parse(s: &str) -> Option<Axis> {
        Some(match s {
            "block" => Axis::Block,
            "depth" => Axis::Depth,
            "device" => Axis::Device,
            "cores" => Axis::Cores,
            "page" => Axis::Page,
            "ram" => Axis::Ram,
            "readahead" | "ra" => Axis::Readahead,
            "numa" => Axis::Numa,
            "fabric" => Axis::Fabric,
            "record" => Axis::Record,
            "budget" => Axis::Budget,
            _ => return None,
        })
    }

    pub fn all() -> &'static [(&'static str, &'static str)] {
        &[
            ("block", "request size, 512 B to 1 MiB"),
            ("depth", "offered queue depth"),
            ("device", "across every modelled device"),
            ("cores", "host cores issuing I/O"),
            ("page", "page cache granularity, pricing a rewrite"),
            ("ram", "page cache size, pricing a rewrite"),
            ("readahead", "off, kernel default, FADV_SEQUENTIAL"),
            ("numa", "issuing local or across a socket"),
            ("fabric", "devices sharing one upstream link"),
            ("record", "record size — where ordering starts to pay"),
            ("budget", "memory budget — passes and amplification"),
        ]
    }
}

/// How a sweep is configured.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub device: &'static str,
    pub block_bytes: u64,
    pub depth: usize,
    pub cores: usize,
    pub records: u64,
    pub record_bytes: u64,
    pub container_bytes: u64,
    /// Budget as a fraction of the payload.
    pub budget_fraction: f64,
    pub page_bytes: u64,
    /// Cache size as a fraction of the payload.
    pub ram_fraction: f64,
    pub samples: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            device: "nvme-consumer",
            block_bytes: 4_096,
            depth: 10,
            cores: 1,
            records: 200_000,
            record_bytes: 512,
            container_bytes: 65_536,
            budget_fraction: 0.25,
            page_bytes: 4_096,
            ram_fraction: 0.125,
            samples: 4_000,
        }
    }
}

impl Config {
    pub fn hardware(&self) -> hw::Hardware {
        *hw::ALL_HARDWARE_WITH_MODERN
            .iter()
            .find(|h| h.name == self.device)
            .unwrap_or(&hw::NVME_CONSUMER_HW)
    }

    pub fn geometry(&self) -> Geometry {
        Geometry {
            records: self.records,
            record_bytes: self.record_bytes,
            container_bytes: self.container_bytes,
        }
    }

    fn host(&self) -> hw::HostModel {
        hw::HostModel::cores(self.cores)
    }
}

fn device_row(label: String, stats: &io::IoStats) -> Row {
    Row {
        label,
        metrics: vec![
            Metric {
                name: "IOPS",
                value: stats.iops(),
                unit: Unit::Count,
            },
            Metric {
                name: "MB/s",
                value: stats.throughput(),
                unit: Unit::Rate,
            },
            Metric {
                name: "util",
                value: stats.bandwidth_utilization(),
                unit: Unit::Fraction,
            },
            Metric {
                name: "posn",
                value: stats.positioning_fraction(),
                unit: Unit::Fraction,
            },
            Metric {
                name: "lat",
                value: stats.mean_latency_s(),
                unit: Unit::Micros,
            },
        ],
    }
}

fn random_read(config: &Config, hardware: &hw::Hardware, block: u64, depth: usize) -> io::IoStats {
    const SPAN: u64 = 5 * 1024 * 1024 * 1024;
    let mut scheduler = io::sched::Noop::default();
    let mut issuer = io::RandomAccess::new(SPAN, block, config.samples, 0xF10);
    io::run(
        hardware,
        &mut scheduler,
        &mut issuer,
        io::RunConfig {
            host: config.host(),
            ..io::RunConfig::direct(depth, SPAN)
        },
    )
}

/// Price a rewrite through the full simulated path.
fn priced(config: &Config, cache: Option<CacheConfig>) -> io::IoStats {
    let geometry = config.geometry();
    let map = Map::shuffled(geometry.records, 0xC0FFEE);
    let budget = (geometry.payload_bytes() as f64 * config.budget_fraction) as u64;
    let (_, trace) = Gsplat::new().run(geometry, &map, budget.max(geometry.record_bytes));
    crate::price::simulate_io(&trace, &config.hardware(), cache, 32)
}

/// Run one sweep.
pub fn run(axis: Axis, config: &Config) -> Sweep {
    let hardware = config.hardware();
    let geometry = config.geometry();

    match axis {
        Axis::Block => Sweep {
            axis: "block".into(),
            held: vec![
                ("device".into(), config.device.into()),
                ("depth".into(), config.depth.to_string()),
                ("cores".into(), config.cores.to_string()),
            ],
            rows: [512u64, 4_096, 16_384, 65_536, 262_144, 1_048_576]
                .iter()
                .map(|&b| {
                    let n = if b >= 1 << 20 {
                        config.samples / 10
                    } else {
                        config.samples
                    };
                    let stats = random_read(
                        &Config {
                            samples: n.max(200),
                            ..*config
                        },
                        &hardware,
                        b,
                        config.depth,
                    );
                    device_row(format!("{b}"), &stats)
                })
                .collect(),
        },

        Axis::Depth => Sweep {
            axis: "depth".into(),
            held: vec![
                ("device".into(), config.device.into()),
                ("block".into(), config.block_bytes.to_string()),
                ("cores".into(), config.cores.to_string()),
            ],
            rows: [1usize, 2, 4, 8, 16, 32, 64, 128]
                .iter()
                .map(|&d| {
                    let stats = random_read(config, &hardware, config.block_bytes, d);
                    device_row(format!("{d}"), &stats)
                })
                .collect(),
        },

        Axis::Device => Sweep {
            axis: "device".into(),
            held: vec![
                ("block".into(), config.block_bytes.to_string()),
                ("depth".into(), config.depth.to_string()),
                ("cores".into(), config.cores.to_string()),
            ],
            rows: hw::ALL_HARDWARE_WITH_MODERN
                .iter()
                .map(|h| {
                    let stats = random_read(config, h, config.block_bytes, config.depth);
                    device_row(h.name.into(), &stats)
                })
                .collect(),
        },

        Axis::Cores => Sweep {
            axis: "cores".into(),
            held: vec![
                ("device".into(), config.device.into()),
                ("block".into(), config.block_bytes.to_string()),
                ("depth".into(), config.depth.to_string()),
            ],
            rows: [1usize, 2, 4, 8, 16]
                .iter()
                .map(|&c| {
                    let stats = random_read(
                        &Config {
                            cores: c,
                            ..*config
                        },
                        &hardware,
                        config.block_bytes,
                        config.depth,
                    );
                    let mut row = device_row(format!("{c}"), &stats);
                    row.metrics.push(Metric {
                        name: "hostblk",
                        value: stats.host_saturation(),
                        unit: Unit::Fraction,
                    });
                    row
                })
                .collect(),
        },

        Axis::Page => Sweep {
            axis: "page".into(),
            held: vec![
                ("device".into(), config.device.into()),
                ("record".into(), config.record_bytes.to_string()),
                (
                    "budget".into(),
                    format!("{:.0}%", config.budget_fraction * 100.0),
                ),
                ("ram".into(), format!("{:.0}%", config.ram_fraction * 100.0)),
            ],
            rows: [4_096u64, 16_384, 65_536, 262_144]
                .iter()
                .map(|&p| {
                    let ram = (geometry.payload_bytes() as f64 * config.ram_fraction) as u64;
                    let stats = priced(config, Some(CacheConfig::new(ram.max(p), p)));
                    Row {
                        label: format!("{p}"),
                        metrics: vec![
                            Metric {
                                name: "elapsed",
                                value: stats.elapsed_s,
                                unit: Unit::Seconds,
                            },
                            Metric {
                                name: "requests",
                                value: stats.requests_completed as f64,
                                unit: Unit::Count,
                            },
                            Metric {
                                name: "readahead",
                                value: stats.readahead_requests as f64,
                                unit: Unit::Count,
                            },
                            Metric {
                                name: "util",
                                value: stats.bandwidth_utilization(),
                                unit: Unit::Fraction,
                            },
                        ],
                    }
                })
                .collect(),
        },

        Axis::Ram => Sweep {
            axis: "ram".into(),
            held: vec![
                ("device".into(), config.device.into()),
                ("record".into(), config.record_bytes.to_string()),
                ("page".into(), config.page_bytes.to_string()),
                (
                    "budget".into(),
                    format!("{:.0}%", config.budget_fraction * 100.0),
                ),
            ],
            rows: [0.06f64, 0.125, 0.25, 0.5, 1.0, 2.0]
                .iter()
                .map(|&f| {
                    let ram = (geometry.payload_bytes() as f64 * f) as u64;
                    let stats = priced(
                        config,
                        Some(CacheConfig::new(
                            ram.max(config.page_bytes),
                            config.page_bytes,
                        )),
                    );
                    Row {
                        label: format!("{:.0}% payload", f * 100.0),
                        metrics: vec![
                            Metric {
                                name: "elapsed",
                                value: stats.elapsed_s,
                                unit: Unit::Seconds,
                            },
                            Metric {
                                name: "hits",
                                value: stats.cache_hits as f64,
                                unit: Unit::Count,
                            },
                            Metric {
                                name: "requests",
                                value: stats.requests_completed as f64,
                                unit: Unit::Count,
                            },
                            Metric {
                                name: "util",
                                value: stats.bandwidth_utilization(),
                                unit: Unit::Fraction,
                            },
                        ],
                    }
                })
                .collect(),
        },

        Axis::Readahead => {
            let ram = (geometry.payload_bytes() as f64 * config.ram_fraction) as u64;
            let cache = CacheConfig::new(ram.max(config.page_bytes), config.page_bytes);
            let map = Map::shuffled(geometry.records, 0xC0FFEE);
            let budget = (geometry.payload_bytes() as f64 * config.budget_fraction) as u64;
            let ordered = Gsplat::new()
                .run(geometry, &map, budget.max(geometry.record_bytes))
                .1;
            let scattered = NaiveGather
                .run(geometry, &map, budget.max(geometry.record_bytes))
                .1;

            Sweep {
                axis: "readahead".into(),
                held: vec![
                    ("device".into(), config.device.into()),
                    ("record".into(), config.record_bytes.to_string()),
                    ("page".into(), config.page_bytes.to_string()),
                ],
                rows: [
                    ("off", io::Readahead::OFF),
                    ("default", io::Readahead::DEFAULT),
                    ("fadv-seq", io::Readahead::SEQUENTIAL_ADVICE),
                ]
                .iter()
                .map(|(name, ra)| {
                    let run_one = |trace: &crate::model::Trace| {
                        let mut sched = io::sched::Noop::default();
                        let mut issuer = io::Recorded::new(crate::price::accesses_of(trace));
                        io::run(
                            &hardware,
                            &mut sched,
                            &mut issuer,
                            io::RunConfig {
                                readahead: *ra,
                                host: config.host(),
                                ..io::RunConfig::buffered(32, geometry.payload_bytes() * 2, cache)
                            },
                        )
                    };
                    let o = run_one(&ordered);
                    let s = run_one(&scattered);
                    Row {
                        label: (*name).into(),
                        metrics: vec![
                            Metric {
                                name: "ordered",
                                value: o.elapsed_s,
                                unit: Unit::Seconds,
                            },
                            Metric {
                                name: "scattered",
                                value: s.elapsed_s,
                                unit: Unit::Seconds,
                            },
                            Metric {
                                name: "gain",
                                value: s.elapsed_s / o.elapsed_s.max(1e-12),
                                unit: Unit::Ratio,
                            },
                            Metric {
                                name: "ra reqs",
                                value: o.readahead_requests as f64,
                                unit: Unit::Count,
                            },
                        ],
                    }
                })
                .collect(),
            }
        }

        Axis::Numa => Sweep {
            axis: "numa".into(),
            held: vec![
                ("device".into(), config.device.into()),
                ("block".into(), config.block_bytes.to_string()),
                ("depth".into(), config.depth.to_string()),
            ],
            rows: [("local", hw::Numa::LOCAL), ("remote", hw::Numa::REMOTE)]
                .iter()
                .map(|(name, numa)| {
                    const SPAN: u64 = 5 * 1024 * 1024 * 1024;
                    let mut sched = io::sched::Noop::default();
                    let mut issuer =
                        io::RandomAccess::new(SPAN, config.block_bytes, config.samples, 0xF10);
                    let stats = io::run(
                        &hardware,
                        &mut sched,
                        &mut issuer,
                        io::RunConfig {
                            numa: *numa,
                            host: config.host(),
                            ..io::RunConfig::direct(config.depth, SPAN)
                        },
                    );
                    let mut row = device_row((*name).into(), &stats);
                    row.metrics.push(Metric {
                        name: "headroom",
                        value: stats.platform_headroom(),
                        unit: Unit::Fraction,
                    });
                    row
                })
                .collect(),
        },

        Axis::Fabric => Sweep {
            axis: "fabric".into(),
            held: vec![
                ("device".into(), config.device.into()),
                ("link".into(), "PCIe 4.0 x16".into()),
                ("block".into(), "1 MiB sequential".into()),
            ],
            rows: [1usize, 2, 4, 8, 16]
                .iter()
                .map(|&devices| {
                    const SPAN: u64 = 5 * 1024 * 1024 * 1024;
                    let mut sched = io::sched::Noop::default();
                    let mut issuer = io::SequentialAccess::new(SPAN, 1 << 20, 1_000, false);
                    let stats = io::run(
                        &hardware,
                        &mut sched,
                        &mut issuer,
                        io::RunConfig {
                            fabric: hw::Fabric::pcie4_x16(devices),
                            host: hw::HostModel {
                                memory_bandwidth: 400.0e9,
                                ..hw::HostModel::cores(config.cores.max(8))
                            },
                            ..io::RunConfig::direct(32, SPAN)
                        },
                    );
                    let mut row = device_row(format!("{devices} sharing"), &stats);
                    row.metrics.push(Metric {
                        name: "headroom",
                        value: stats.platform_headroom(),
                        unit: Unit::Fraction,
                    });
                    row
                })
                .collect(),
        },

        Axis::Record => {
            let model = crate::device::ALL_MODELS
                .iter()
                .chain([&crate::device::NVME_MODERN_MODEL])
                .find(|m| m.name == config.device)
                .copied()
                .unwrap_or(crate::device::NVME_CONSUMER_MODEL);
            let payload = geometry.payload_bytes();
            Sweep {
                axis: "record".into(),
                held: vec![
                    ("device".into(), model.name.into()),
                    ("depth".into(), config.depth.to_string()),
                    (
                        "payload".into(),
                        format!("{:.1} GiB", payload as f64 / (1u64 << 30) as f64),
                    ),
                ],
                rows: [128u64, 512, 1_540, 4_096, 16_384, 65_536]
                    .iter()
                    .map(|&r| {
                        let penalty = model.random_penalty_at_depth(r, config.depth as f64);
                        let budget =
                            model.min_budget_for_ordering_at_depth(payload, r, config.depth as f64);
                        Row {
                            label: format!("{r}"),
                            metrics: vec![
                                Metric {
                                    name: "penalty",
                                    value: penalty,
                                    unit: Unit::Ratio,
                                },
                                Metric {
                                    name: "min budget",
                                    value: if budget == u64::MAX {
                                        payload as f64
                                    } else {
                                        budget as f64
                                    },
                                    unit: Unit::Rate,
                                },
                                Metric {
                                    name: "of payload",
                                    value: if budget == u64::MAX {
                                        1.0
                                    } else {
                                        budget as f64 / payload as f64
                                    },
                                    unit: Unit::Fraction,
                                },
                            ],
                        }
                    })
                    .collect(),
            }
        }

        Axis::Budget => {
            let map = Map::shuffled(geometry.records, 0xC0FFEE);
            let payload = geometry.payload_bytes();
            Sweep {
                axis: "budget".into(),
                held: vec![
                    ("device".into(), config.device.into()),
                    ("records".into(), geometry.records.to_string()),
                    ("record".into(), geometry.record_bytes.to_string()),
                ],
                rows: [0.02f64, 0.05, 0.1, 0.25, 0.5]
                    .iter()
                    .map(|&f| {
                        let budget = ((payload as f64 * f) as u64).max(geometry.record_bytes);
                        let g = Gsplat::new().run(geometry, &map, budget).1;
                        let n = NaiveGather.run(geometry, &map, budget).1;
                        let gm = g.metrics();
                        let cache = Some(CacheConfig::new(
                            (payload as f64 * config.ram_fraction) as u64 + config.page_bytes,
                            config.page_bytes,
                        ));
                        let go = crate::price::simulate_io(&g, &hardware, cache, 32);
                        let no = crate::price::simulate_io(&n, &hardware, cache, 32);
                        Row {
                            label: format!("{:.0}% payload", f * 100.0),
                            metrics: vec![
                                Metric {
                                    name: "passes",
                                    value: gm.passes as f64,
                                    unit: Unit::Count,
                                },
                                Metric {
                                    name: "amp",
                                    value: gm.amplification(),
                                    unit: Unit::Ratio,
                                },
                                Metric {
                                    name: "ordered",
                                    value: go.elapsed_s,
                                    unit: Unit::Seconds,
                                },
                                Metric {
                                    name: "scattered",
                                    value: no.elapsed_s,
                                    unit: Unit::Seconds,
                                },
                                Metric {
                                    name: "gain",
                                    value: no.elapsed_s / go.elapsed_s.max(1e-12),
                                    unit: Unit::Ratio,
                                },
                            ],
                        }
                    })
                    .collect(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_axis_produces_rows_with_matching_metrics() {
        let config = Config {
            samples: 400,
            records: 20_000,
            ..Config::default()
        };
        for (name, _) in Axis::all() {
            let axis = Axis::parse(name).expect("listed axes must parse");
            let sweep = run(axis, &config);
            assert!(!sweep.rows.is_empty(), "{name}: produced no rows");
            let width = sweep.rows[0].metrics.len();
            for row in &sweep.rows {
                assert_eq!(
                    row.metrics.len(),
                    width,
                    "{name}: row '{}' has a different metric count",
                    row.label
                );
            }
            assert!(
                !sweep.held.is_empty(),
                "{name}: must state what it held fixed"
            );
        }
    }

    /// A sweep is only reproducible if its output says what was held
    /// fixed. Most of the wrong conclusions this crate has corrected came
    /// from a number quoted without its conditions.
    #[test]
    fn rendering_states_the_held_conditions() {
        let config = Config {
            samples: 400,
            ..Config::default()
        };
        let text = run(Axis::Block, &config).render(Baseline::First);
        assert!(text.contains("held:"));
        assert!(text.contains("device="));
        assert!(text.contains("depth="));
    }

    #[test]
    fn deltas_are_relative_to_the_chosen_baseline() {
        let config = Config {
            samples: 400,
            ..Config::default()
        };
        let sweep = run(Axis::Depth, &config);
        let first = sweep.render(Baseline::First);
        let prev = sweep.render(Baseline::Previous);
        let none = sweep.render(Baseline::None);
        assert_ne!(first, prev, "the two baselines must differ");
        assert!(
            !none.contains('%') || !none.contains('+'),
            "none means no deltas"
        );
    }

    /// The depth axis has to show the effect concurrency actually has,
    /// since that is the parameter most conclusions here turned out to
    /// hinge on.
    #[test]
    fn the_depth_axis_shows_concurrency_scaling() {
        let config = Config {
            device: "nvme-consumer",
            samples: 2_000,
            cores: 8,
            ..Config::default()
        };
        let sweep = run(Axis::Depth, &config);
        let iops: Vec<f64> = sweep.rows.iter().map(|r| r.metrics[0].value).collect();
        assert!(
            iops.last().unwrap() > &(iops[0] * 5.0),
            "deeper queues must raise throughput: {:?}",
            iops
        );
    }
}
