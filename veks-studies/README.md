# veks-studies

Executable models of the SPLAT family of I/O-ordered rewrites, and a
storage simulator accurate enough to price them.

This document exists so you can decide how much to believe the numbers
this crate produces. It states what is measured, what is fitted, what is
predicted, what is asserted without evidence, and where the model is
known to be wrong.

---

## What this is

Two layers, and the distinction matters.

![Layer overview](docs/gsplat/veks-sim-layers.svg)

**A model of the algorithm.** `algo`, `model`, `check` and `study`
simulate the rewrite itself — every read, scatter and write recorded as a
virtual operation in a trace. Nothing here touches a device. It exists so
that claims in `docs/gsplat/` can fail a test: that every mapped record
is read exactly once, that reads ascend within a pass, that resident
memory stays inside its budget, that container touches follow
`A(P) = P · (1 − exp(−w / P))`.

**A model of the storage path.** `io`, `cache`, `device` and `regime`
turn a trace into a time. This is a discrete-event simulator: it advances
a clock through positioning, transfer and programming against a shared
bandwidth ceiling, a finite command queue, a serial controller, dies with
address affinity, and a page cache with kernel-style readahead. **It
contains no throughput formula.** Throughput is what falls out.

`validate` scores the second layer against measurement.

Every stage one request passes through, and what can stop it at each:

![Storage request path](docs/gsplat/veks-sim-request-path.svg)

## Running it

```
veks-study sweep <axis> [options]   # vary one parameter, see the deltas
veks-study validate                 # the scorecard below
veks-study devices                  # what is modelled
veks-study report                   # the full standing report
veks-study help                     # axes and options
```

Sweeps vary exactly one parameter, hold the rest fixed, print what they
held, and put a delta column beside every metric — because the shape of a
dependence is almost always the thing being argued about, not any single
value.

```
$ veks-study sweep depth --device nvme-modern --cores 8

  sweep: depth
  held:  device=nvme-modern  block=4096  cores=8

  depth           IOPS       d        MB/s       d        util       d        posn       d
  1               8.4k       —          34       —          0%       —         91%       —
  8              77.6k +826.4%         318 +826.4%          5% +826.4%         43%  -53.0%
  32            325.8k    +39x        1335    +39x         19%    +39x          5%  -94.9%
  128           895.0k   +107x        3666   +107x         52%   +107x          6%  -93.2%
```

Axes: `block`, `depth`, `device`, `cores`, `page`, `ram`, `readahead`,
`numa`, `fabric`, `record`, `budget`. `--vs first|prev|none` chooses what
the deltas are measured against.

## Reproducing everything here

```
cargo test -p veks-studies                 # 115 assertions
cargo run -p veks-studies --bin veks-study # every table below
```

Diagnostic dumps are `#[ignore]`d tests; run one with, for example:

```
cargo test -p veks-studies --release print_validation_report -- --ignored --nocapture
```

## What it is validated against

Every device figure comes from the
[perfscripts](https://github.com/jshook/perfscripts) fio corpus:

| | |
|---|---|
| Devices | Toshiba HDWD110 (7200 RPM SATA), Samsung 850 Pro (SATA SSD), Samsung 950 Pro (NVMe) |
| Run conditions | `direct=1, ioengine=libaio, iodepth=10, size=5G, time_based, runtime=1m` |
| Workloads | random read 512 B–16 MiB, sequential read, sequential write, and a mixed random-reader / rate-capped sequential reader+writer sweep |
| Metrics used | IOPS, bandwidth, and full `clat` percentile distributions |

**Everything derived from this corpus is a statement about `iodepth=10`
on 2016 hardware.** That is the single most important caveat in this
document. Conclusions about whether ordering pays are strongly
concurrency-dependent, and the corpus fixes concurrency at one value.

## Accuracy

![Validation flow](docs/gsplat/veks-sim-validation.svg)

Scored across three devices and every block size from 512 B to 1 MiB, as
mean absolute percentage error against measurement. Latency is compared
on fio's `lat` basis — submission to completion — because that is what
the simulator times.

| Device | Metric | MAPE | Worst | Bias |
|---|---|---|---|---|
| spinning-sata | throughput | 2.6% | 7.5% | +2.6% |
| | mean latency | 2.0% | 6.1% | −2.0% |
| | p50 / p95 / p99 | 6.4% / 5.3% / 7.4% | | |
| sata-ssd | throughput | 1.5% | 3.5% | +0.7% |
| | mean latency | 1.9% | 3.8% | −0.7% |
| | p50 / p95 / p99 | 3.3% / 10.4% / 12.0% | | |
| nvme-consumer | throughput | 7.0% | 14.6% | −3.7% |
| | mean latency | 8.4% | 17.1% | +8.0% |
| | p50 / p95 / p99 | 10.0% / 10.5% / 13.4% | | |
| **aggregate** | **throughput** | **3.7%** | 14.6% | −0.1% |
| | **mean latency** | **4.1%** | 17.1% | +1.7% |
| | p50 / p95 / p99 | 6.6% / 8.7% / 11.0% | 30.3% | |
| **sequential write** | **throughput** | **0.2%** | 0.5% | +0.2% |
| | **latency** | **0.7%** | 1.3% | −0.7% |

Against the bars the literature states, compared like for like — mean
error against mean error, worst case against worst case:

| | Reported | Here |
|---|---|---|
| [MQSim](https://www.usenix.org/conference/fast18/presentation/tavakkol) (FAST '18), throughput vs 4 real SSDs | 6–18% | 3.7% MAPE / 14.6% worst |
| [SimpleSSD](https://arxiv.org/pdf/1705.06419), worst-case throughput | 28% | 14.6% |
| SimpleSSD, worst-case latency | 36% | 30.3% (p99) |
| [Generative black-box models](https://arxiv.org/pdf/2307.02073) | 4–10% IOPS, 3–16% latency | 3.7% / 4.1% |

## Fitted, predicted, or asserted

The word "validated" does a lot of work in simulator papers. Here is the
breakdown for this one.

**Fitted — agreement proves the fit converged, not that the model is
right:**

- Device parameters (`media_rate`, `bus_rate`, `access_latency`,
  `max_command_rate`, die counts and stripe) were tuned against the
  random-read throughput curve.
- The NAND page-type spread (`ReadVariation`) and the disk's
  `rotational_awareness` were tuned against measured latency percentiles.
- The disk's `command_expiry_s` was fitted against the contention sweep.
- Write-path ceilings were fitted against measured sequential write.

**Predicted — not tuned against, so agreement is evidence:**

- Sequential read throughput. Exact on all three devices (201/201,
  568/568, 1500/1497 MB/s).
- The *shape* of the latency distribution across block sizes, once the
  variation parameters are set at one point. `ReadVariation` is
  mean-preserving by construction, so it cannot have flattered any mean.
- The contention sweep's direction and rough magnitude.
- `command_expiry_s` was fitted to contention and then found to match the
  drive's measured maximum completion latency (607.7 ms) — two
  independent routes to one number.

**Asserted — no evidence either way:**

- Random-write behaviour, garbage collection, write amplification. The
  corpus has no random-write job. These are modelled and move in the
  right direction; nothing more is claimed.
- The modern NVMe regime, calibrated to published figures from
  [Ren et al. (ICPE '24)](https://dl.acm.org/doi/10.1145/3629526.3645053)
  and [MQSSD](https://arxiv.org/abs/2507.06349) rather than to a sweep
  run here.
- NUMA and multi-device fabric sharing. Structurally reasonable,
  unmeasured. Every source measurement was single-socket with a
  dedicated link.

## Known divergences

Stated because a model that hides these is not worth using.

- **Contention on flash is understated.** Reproducing the `mixed` job,
  the simulated random reader lands 20–33% below measurement at capped
  points, and the uncapped collapse comes out several times gentler than
  the measured 178×. Die-level blocking brought this from 85× wrong to
  roughly 3× wrong; the residual is real.
- **NVMe carries a +10% median bias.** That drive's measured bandwidth
  peaks at 128 KiB and *falls* for larger blocks, exceeding its own
  single-stream sequential rate. No monotone ceiling reproduces both
  ends, and raising the ceiling to fix latency costs throughput.
- **The disk's tail ratio runs 13% light** and the NVMe's 19% light.
- **`transfer_share_spread` is not strongly determined.** Raising it
  trades scatter for bias monotonically, with no optimum. The value is a
  judgement.

## What is not modelled at all

- Filesystem geometry — extents, fragmentation, journal traffic,
  metadata. The address space is flat.
- The I/O scheduler's own CPU cost. Ren et al. measure up to 63.4%
  throughput overhead from Linux schedulers; the ones here are free.
- Tail latency beyond p99.9.
- Multi-device striping. One device, with the upstream link represented
  only as a share.
- **Real workloads.** Every request stream here is synthetic or generated
  from an algorithm's own trace. The
  [SNIA IOTTA repository](https://iotta.snia.org/),
  [Alibaba block traces](https://github.com/alibaba/block-traces) and
  [Meta CacheLib traces](https://github.com/cacheMon/cache_dataset) are
  real streams this could be driven by and is not. **The model is
  validated against real devices, not real workloads.**

## Sources

Full annotations, including which parameter each source grounds, are in
the crate-level rustdoc (`cargo doc -p veks-studies --open`).

**Measurement:** [perfscripts](https://github.com/jshook/perfscripts) ·
[Ren et al., ICPE '24](https://dl.acm.org/doi/10.1145/3629526.3645053)
([artifact](https://zenodo.org/records/10599514))

**Mechanisms:**
[MQSSD (Ransom, Lim & Mitzenmacher, 2025)](https://arxiv.org/abs/2507.06349) ·
[MQSim (Tavakkol et al., FAST '18)](https://www.usenix.org/conference/fast18/presentation/tavakkol) ·
[SimpleSSD (Jung et al.)](https://arxiv.org/pdf/1705.06419) ·
[Lebrecht, Dingle & Knottenbelt, QEST '09](http://www.doc.ic.ac.uk/~wjk/publications/lebrecht-dingle-knottenbelt-qest-2009.pdf) ·
[RAIL](https://people.ucsc.edu/~hlitz/papers/rail.pdf) ·
[Aggarwal & Vitter, CACM 31(9), 1988](https://dl.acm.org/doi/10.1145/48529.48535)

**Method:**
[Generative storage performance models](https://arxiv.org/pdf/2307.02073) ·
[Different Perspectives of Memory System Simulation (2026)](https://arxiv.org/abs/2604.16965)

## What using it looks like

The algorithm layer, checking a documented invariant:

```rust
use veks_studies::{Geometry, Map, Gsplat, Rewrite, check};

let geometry = Geometry::new(200_000, 4_096, 128 * 1024);
let map = Map::shuffled(geometry.records, 0xC0FFEE);
let (sink, trace) = Gsplat::new().run(geometry, &map, geometry.payload_bytes() / 8);

assert!(sink.matches(&map));
assert!(check::monotone_access(&trace).is_empty());
```

The storage layer, pricing that trace on a real device path:

```rust
use veks_studies::{cache::CacheConfig, io::hw::SPINNING_SATA_HW, price};

let stats = price::simulate_io(&trace, &SPINNING_SATA_HW,
                               Some(CacheConfig::new(64 << 20, 4_096)), 32);

println!("{:.2}s, {:.0}% of the device's bandwidth used, {:.0}% positioning",
         stats.elapsed_s,
         stats.bandwidth_utilization() * 100.0,
         stats.positioning_fraction() * 100.0);
```
