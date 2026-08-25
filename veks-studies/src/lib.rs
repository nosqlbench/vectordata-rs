// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Executable models of the SPLAT family.
//!
//! The documents in `docs/gsplat/` make claims: that every mapped record
//! is read exactly once, that reads ascend within a pass, that resident
//! memory stays inside the budget, that container touches follow
//! `A(P) = P · (1 − exp(−w / P))`. Prose cannot fail a test. This crate
//! exists so those claims can.
//!
//! Nothing here performs real I/O. A [`Geometry`](model::Geometry)
//! describes a store the way the cost model describes one — a record
//! count, a record size, a container size — and every access an
//! algorithm makes is recorded as a virtual operation in a
//! [`Trace`](model::Trace). Costs are then *computed* from the trace
//! rather than measured from a device, which makes them exact,
//! reproducible, and comparable against the formulas.
//!
//! Three things can be checked this way:
//!
//! - **Correctness.** A rewrite is correct when the sink ends up holding
//!   `map` — since a record's virtual payload is its own source ordinal,
//!   `sink[i] == map[i]` is the whole of `output[i] = source[map[i]]`.
//! - **Invariants.** [`check`] turns each documented invariant into an
//!   assertion over the trace.
//! - **Cost.** [`study`] sweeps parameters and prints measured cost
//!   beside predicted cost, so a wrong formula shows up as a column that
//!   does not line up.
//!
//! Scope: flat, single-space rewrites — the [gsplat](../docs/gsplat)
//! core. Structured and multi-space variants are modelled in the same
//! terms but are not implemented here yet; [`model::Geometry`] carries
//! the container notion they will need.
//!
//! # Sources
//!
//! Nothing in this crate is realistic on its own authority. Every device
//! constant, mechanism and accuracy claim traces to one of the following,
//! and each module cites the ones it depends on. Where a parameter was
//! fitted rather than read from a source, the doc comment on that
//! parameter says so.
//!
//! ## Measurement — the ground truth
//!
//! - **[perfscripts](https://github.com/jshook/perfscripts)** — the fio
//!   corpus every device figure comes from: random-read sweeps from 512 B
//!   to 16 MiB, sequential read and write, and a mixed reader/writer
//!   contention sweep, all at `direct=1, ioengine=libaio, iodepth=10`,
//!   60 s, on three devices. Grounds [`regime`] entirely, and is what
//!   [`validate`] scores against.
//! - **[Ren et al., ICPE '24](https://dl.acm.org/doi/10.1145/3629526.3645053)**,
//!   *BFQ, Multiqueue-Deadline, or Kyber?* — modern NVMe measurements
//!   (Samsung 980 PRO: ~1M 4 KiB IOPS, 68/15 µs read/write latency,
//!   5.9M IOPS across eight devices), the finding that the CPU saturates
//!   before the device does, and the 63.4% scheduler overhead figure.
//!   Grounds [`io::hw::NVME_MODERN_HW`] and [`io::hw::HostModel`].
//!   Artifact: <https://zenodo.org/records/10599514>.
//!
//! - **[Didona et al., SYSTOR '22](https://atlarge-research.com/pdfs/2022-systor-apis.pdf)**,
//!   *Understanding Modern Storage APIs* — single-core peak throughput
//!   for each Linux storage API on Intel DC P3600 NVMe under kernel 5.13:
//!   libaio 144.9 KIOPS, io_uring 171.5, io_uring polled 173.0, SPDK
//!   305.9. Grounds the [`io::hw::HostModel`] presets, and shows the
//!   per-request cost is an API-and-kernel property spanning 3.3–6.9 µs.
//! - **Device teardowns** — the Samsung 950 PRO's UBX controller
//!   addresses eight channels with eight-way interleaving, and the 850
//!   PRO's MEX drives eight NAND packages. Both drives use MLC V-NAND.
//!   Grounds the die counts and the choice of MLC page-type timings.
//!
//! ## Models — the mechanisms
//!
//! - **[MQSSD](https://arxiv.org/abs/2507.06349)** (Ransom, Lim &
//!   Mitzenmacher, 2025), *Multi-Queue SSD I/O Modeling* — makes
//!   concurrency a first-class parameter of the external-memory model,
//!   and reports the random-to-sequential read ratio falling to 1.3–1.5×
//!   at k=128. Grounds [`io::hw::ConcurrencyScaling`] and the
//!   concurrency-dependent crossover in [`device`].
//! - **[MQSim](https://www.usenix.org/conference/fast18/presentation/tavakkol)**
//!   (Tavakkol et al., FAST '18) and
//!   **[SimpleSSD](https://arxiv.org/pdf/1705.06419)** (Jung et al.) —
//!   the established SSD simulators. Both parameterise NAND read,
//!   program and erase latencies per page type — MQSim exposes
//!   `Page_Read_Latency_LSB/CSB/MSB` and `Page_Program_Latency_*`
//!   directly — which is the precedent [`io::hw::ReadVariation`] follows.
//!   Their reported accuracies are the bars [`validate`] measures against.
//! - **[Device-Level Optimization Techniques for Solid-State Drives: A
//!   Survey](https://arxiv.org/abs/2507.10573)** (2025) — NAND timing by
//!   cell type: MLC reads 40–110 µs and programs 0.4–1.5 ms, TLC reads
//!   66–170 µs and programs 0.8–2 ms, with LSB pages needing one sensing
//!   pass and MSB pages more. Grounds [`io::hw::ReadVariation`] and
//!   corroborates the per-die program rates, which were calibrated from
//!   sequential-write throughput and land at ~712 µs for a 16 KiB page —
//!   inside the published MLC band.
//! - **[Park et al., *Reducing SSD Read Latency by Optimizing
//!   Read-Retry*](https://arxiv.org/pdf/2104.09611)** — read-retry at
//!   shifted reference voltages, the mechanism behind the tail term in
//!   [`io::hw::ReadVariation`]. Its *rate* on a given drive at a given
//!   wear level is the one part still fitted.
//! - **[Schindler & Ganger, *Automated Disk Drive Characterization*,
//!   CMU-CS-99-176](https://www.pdl.cmu.edu/PDL-FTP/DriveChar/cs-99-176.pdf)**
//!   — DIXtrac extracts over 100 performance-critical parameters per
//!   drive, including mechanical timings and command overheads. The
//!   reference for how a disk's seek and rotation profile should be
//!   obtained, against which the class-typical figures in
//!   [`io::hw::SPINNING_SATA_HW`] are an admitted shortcut.
//! - **[Lebrecht, Dingle & Knottenbelt, QEST '09](http://www.doc.ic.ac.uk/~wjk/publications/lebrecht-dingle-knottenbelt-qest-2009.pdf)**,
//!   *A Performance Model of Zoned Disk Drives with I/O Request
//!   Reordering* — reordering on rotating media as a modelled
//!   phenomenon rather than an assumed one. Grounds the separation of
//!   selection cost from service cost in [`io::hw::Hardware`].
//! - **[RAIL](https://people.ucsc.edu/~hlitz/papers/rail.pdf)** — read
//!   latency under concurrent writes on NVMe flash: die-level blocking,
//!   read-after-write serialisation up to 20× at the tail. Grounds the
//!   die-affinity and program-occupancy model in [`io`].
//! - **[Aggarwal & Vitter, CACM 31(9), 1988](https://dl.acm.org/doi/10.1145/48529.48535)**
//!   — the external-memory model the amplification analysis in
//!   [`crate::study`] extends.
//!
//! ## Method — how to judge a simulator
//!
//! - **[Performance Modeling of Data Storage Systems using Generative
//!   Models](https://arxiv.org/pdf/2307.02073)** — black-box storage
//!   performance prediction at 4–10% IOPS and 3–16% latency error;
//!   another bar in [`validate`].
//! - **[Different Perspectives of Memory System
//!   Simulation](https://arxiv.org/abs/2604.16965)** (2026) — finds
//!   application-level performance frequently decoupled from internal
//!   simulator statistics, with the interface layer the dominant error
//!   source. The reason [`validate`] scores end-to-end latency and
//!   throughput rather than the internal counters this crate also
//!   produces.
//!
//! - **[SNIA Solid State Storage Performance Test
//!   Specification](https://www.snia.org/tech_activities/standards/curr_standards/pts)**
//!   — the industry-standard methodology for device-level SSD
//!   measurement, including the preconditioning this model does not
//!   represent, plus a public results database.
//!
//! ## Workload data not yet used
//!
//! Named so the gap is visible: the
//! **[SNIA IOTTA repository](https://iotta.snia.org/)**,
//! **[Alibaba block traces](https://github.com/alibaba/block-traces)**
//! and the **[Meta CacheLib traces](https://github.com/cacheMon/cache_dataset)**
//! are real request streams this simulator could be driven by, and it
//! currently is not — every workload here is synthetic or generated from
//! an algorithm's own trace.

pub mod algo;
pub mod cache;
pub mod check;
pub mod device;
pub mod io;
pub mod model;
pub mod price;
pub mod regime;
pub mod study;
pub mod sweep;
pub mod validate;

pub use algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather, naive::NaiveScatter};
pub use model::{Geometry, Map, Metrics, Trace};
