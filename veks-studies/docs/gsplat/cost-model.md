# gsplat — Cost Model

Order-of-growth for the naive permutation and for gsplat, the one
constant that separates them, and worked examples across storage tiers.

## Symbols

| Symbol | Meaning |
|--------|---------|
| `N` | records produced (output count) |
| `N_src` | records in the source, `N_src ≥ N` |
| `R` | bytes per record |
| `M` | memory budget |
| `S` | records per segment, `≈ M / R` |
| `P` | passes, `= max(ceil(N / S), 2)` |
| `B` | minimum addressable transfer (page, sector, block) |
| `W` | container size — the unit the tier or format fetches as a whole |
| `w` | records per container, `= W / R` |
| `BW` | sequential bandwidth |
| `IOPS` | random operations per second at the achieved concurrency |

## The naive baselines

Two ways to apply `output[i] = source[map[i]]` without a buffer. Both
are `Θ(N)` random operations; they differ in which side pays.

```
 gather — walk the output in order          scatter — walk the source in order
 for i in 0..N:                             for j in 0..N_src:
     out[i] = read(src, map[i])                 if inv[j] is set:
                                                    write(out, inv[j], src[j])

 reads  random   ← the cost                 reads  sequential
 writes sequential                          writes random   ← the cost
```

Gather is the better baseline and the one compared against below.
Scatter is worse on every tier: partial-block writes force
read-modify-write, scattered dirty pages provoke writeback storms, and
it needs the *inverse* map — either materialized or produced by a sort.

## Order of growth

| Quantity | naive gather | naive scatter | gsplat |
|---|---|---|---|
| Random operations | `Θ(N)` reads | `Θ(N)` writes | `Θ(P)` — one positioning per pass |
| Monotone / sequential ops | `Θ(N·R / B)` writes | `Θ(N_src·R / B)` reads | `Θ(N)` ascending reads + `Θ(N·R / B)` writes |
| Bytes read from the tier | `Θ(N · ⌈R/B⌉ · B)` | `Θ(N_src · R)` | `Θ(A · N · R)` — `A` below |
| Bytes written to the tier | `Θ(N · R)` | `Θ(2 · N · ⌈R/B⌉ · B)` | `Θ(N · R)` |
| Expected seek distance | `Θ(N_src · R)` — a third of the extent | same | `Θ(P · R)` forward, never backward |
| Resident memory | `Θ(R)` | `Θ(R + N_src · ordinal_bytes)` | `Θ(M)` |
| CPU | `Θ(N)` | `Θ(N_src)` | `Θ(N log(N/P))` |
| Map traffic | `Θ(N · ordinal_bytes)` | plus inversion | `Θ(N · ordinal_bytes)`, read once |
| Passes over the source | 1, scattered | 1 | `P`, ascending |

Two entries deserve emphasis, because intuition usually gets them
wrong:

- **gsplat does not reduce the number of read operations.** Both
  approaches issue `N` record reads. gsplat changes their *order*
  (monotone within a pass) and the write side (`P` contiguous bursts
  instead of `N` scattered writes).
- **gsplat trades CPU for I/O.** The `Θ(N log(N/P))` sort is work the
  naive version never does — a good trade when the tier is the
  bottleneck, a bad one when it isn't.

## Read amplification `A`

The algorithm reads each mapped record exactly once; *tiers* fetch
whole units of `W`. Per pass, a segment needs `S = N/P` of the `N_src`
source records, so a given container holding `w` records is needed
with probability

```
 p = 1 − (1 − S/N_src)^w  ≈  1 − exp(−w·S / N_src)
```

Summed over `P` passes and divided by the live data, that gives the
amplification — tier bytes read per byte of data. For a full
permutation (`N = N_src`):

```
 A(P) = P · (1 − exp(−w / P))          A ≤ min(P, w)

 general form:
 A(P) = (P · N_src / N) · (1 − exp(−w·N / (P·N_src)))
```

`A` interpolates between two regimes, crossing over at **`P = w`**:

| Regime | Condition | Behavior | `A` |
|--------|-----------|----------|-----|
| Dense | `P ≤ w` — the per-pass stride `P·R` fits inside a container | every unit is needed on every pass; each pass is a true streaming scan | `≈ P` |
| Sparse | `P > w` | units are touched once every few passes; reads become per-record fetches, still ascending | `≈ w` |

`A` for `w = 32` (4 KiB records, 128 KiB container):

| `P` | 2 | 4 | 8 | 16 | 32 | 54 | 100 |
|-----|---|---|---|----|----|----|-----|
| `A` | 2.0 | 4.0 | 7.9 | 13.8 | 20.2 | 24.1 | 27.4 |

The naive baseline's amplification is fixed and small — `⌈R/B⌉·B / R`,
around 2 for a 4 KiB record — because random access defeats prefetching
entirely, so the tier fetches only the blocks the record occupies.
**Naive moves less data only when `P > B/R`.** Its amplification is
fixed at `⌈R/B⌉·B / R`; gsplat's is `A(P)`, bounded by `min(P, w)`. Which
is smaller depends on the pass count and the record size together, and
below the crossover gsplat moves less data *as well as* moving it in a
better order.

Measured at page granularity with 512 B records in 4 KiB pages — so
`B/R = 8` — naive sits flat at 8× while gsplat runs 2.0× at two passes
and 6.4× at sixteen:

| passes | naive | gsplat |
|---|---|---|
| 2 | 8.0× | **2.0×** |
| 4 | 8.0× | **3.6×** |
| 8 | 8.0× | **5.3×** |
| 16 | 8.0× | **6.4×** |

The pass count is still the term that decides whether gsplat wins. It is
not true that it wins only on ordering.

`A(P)` is the upper bracket, realized when the tier prefetches
aggressively. If it does not, gsplat's read volume falls to the same
`N·⌈R/B⌉·B` as naive, and gsplat is then strictly better. A sequential
access hint pushes toward the upper bracket by design: **correct in the
dense regime, counterproductive in the sparse one**, where it enlarges
`W` and therefore `A`. Hosts that can should apply the hint
conditionally on `P ≤ w`.

## Time and price model

```
 T ≈ max(bytes_read / BW_read, random_ops / IOPS_eff)
     + bytes_written / BW_write
     + sort and copy time

 IOPS_eff = min(IOPS_tier, concurrency / access_latency)
 price    ≈ requests × price_per_request + bytes × price_per_byte
```

### The streaming bound

One term is missing from the expression above, and leaving it out
overstates ordered reads by roughly sixfold on seek-bound media.

A pass visits containers in **ascending** order. A reader in that
position is never obliged to seek: it can read the source straight
through and discard the containers it does not want, paying the
sequential rate for the whole extent instead of an access latency per
container. That option is always available, so it is a ceiling:

```
 T_read(pass) ≈ min( bytes_source / BW_seq ,           ← stream and discard
                     containers_touched / IOPS(W) )    ← seek to each
```

In the **dense** regime a pass touches nearly every container, so the
first term wins and each pass is simply a scan. In the **sparse** regime
the touched containers are a thin scattered subset and seeking to them
is cheaper. The crossover is the same `P = w` that governs `A`.

The bound is not available to a reader that jumps, because it cannot
know what it may skip. That asymmetry — not the block size — is the
concrete thing ordering buys.

### Measured device behavior

The figures below are transcribed from fio sweeps in the
[perfscripts](https://github.com/jshook/perfscripts) result set
(`direct=1`, `ioengine=libaio`, `iodepth=10`, `size=5G`, `time_based`,
`runtime=1m`, block sizes 512 B–16 MiB), not estimated. **Every
conclusion below inherits those run conditions**, the fixed queue depth
most of all — see [How the line moves with
concurrency](#how-the-line-moves-with-concurrency). `W` is read off the curve as the smallest block at which
random reads reach 95% of sequential throughput.

| Device | Seq read | Random 4 KiB | Penalty | `W` measured | Access latency |
|---|---|---|---|---|---|
| 7200 RPM SATA (Toshiba HDWD110) | 196 MB/s | 1.1 MB/s @ 266 IOPS | **183×** | never reached | 3.7 ms |
| SATA SSD (Samsung 850 Pro) | 542 MB/s | 302 MB/s @ 75.5k IOPS | **1.8×** | 32 KiB | 32 µs |
| Consumer NVMe (Samsung 950 Pro) | 1427 MB/s | 477 MB/s @ 122k IOPS | **2.9×** | 32 KiB | 62 µs |

Three things follow, and none of them were visible from estimated
constants:

- **`W` is 32 KiB on both flash devices**, not the 128 KiB assumed here
  previously, and it is the same on both despite a 2.6× difference in
  peak bandwidth.
- **The random penalty spans two orders of magnitude across tiers.** A
  conclusion drawn on one tier does not transfer to another.
- **The spinning disk charges for arrivals, not bytes.** Its random rate
  is flat at ~260 IOPS from 512 B to 16 KiB — the transfer is lost in the
  seek. This is why ordering wins there at *any* pass count, and why the
  "too many passes and ordering stops paying" caution below is a
  statement about flash only.

| Tier | `W` | Access latency | What dominates |
|------|-----|---------------|----------------|
| Spinning disk | ≥ 16 MiB (never reaches sequential) | 3.7 ms @ QD10 | `random_ops` — naive is hopeless |
| Network block | 128–256 KiB | 0.5–2 ms | `random_ops`, then bandwidth caps |
| Flash, deep queue | 32 KiB measured | 32–62 µs at ~10⁵ IOPS | bytes moved — the terms converge |
| Object storage | 1–8 MiB effective | 20–100 ms, **priced per request** | request count and price, not time |

### What a simulated device path shows

![Storage request path](veks-sim-request-path.svg)

The expressions above price a request stream. They cannot say what
fraction of a device's time becomes useful work, because that is a
consequence of how requests interleave — so `veks-studies/src/io/`
models the path instead of pricing it: bandwidth as a resource that is
consumed, a command queue with a finite number of slots, a controller
with a serial command rate, a head that travels and a platter that
turns, and the page cache in the request path. Nothing in it computes a
throughput. It advances a clock, and it lands within 5% (spinning disk),
9% (SATA SSD) and 17% (NVMe) of the measured random sweeps at every
block size up to 1 MiB, reproducing all three sequential figures to
within a percent.

Two of its outputs change claims made above.

**Where the time goes, at 4 KiB:**

| Device | Access order | Positioning | Bandwidth used |
|---|---|---|---|
| Spinning disk | scattered | 99% | **1%** |
| Spinning disk | ascending | 2% | **98%** |
| SATA SSD | scattered | 0% | 54% |
| SATA SSD | ascending | 0% | 55% |
| NVMe | scattered | 0% | 33% |
| NVMe | ascending | 0% | 33% |

The disk spends 99% of its busy time moving the head and waiting for the
platter, and converts nearly all of it into transfer once reads ascend.
That is this algorithm's entire thesis, stated as a measurement.

The flash rows say something the block-size framing obscures:
**at 4 KiB, ordering changes nothing measurable on flash.** Scattered and
ascending reach the same utilization to within a point, because there is
no position to pay for. Flash gains from ordering only through
*coalescing* — fewer, larger requests moving past the controller's
command-rate ceiling, which is what the 33% on NVMe is — never through
locality.

**Page size is second-order — but not for the reason first given here.**
An earlier version of this section reached that conclusion from a model
with no readahead in it, where page size was the only fetch-granularity
knob there was. That made the finding an artifact of the model.

With readahead modelled the conclusion survives and the mechanism
changes: **the kernel's readahead window, not the page size, sets the
fetch granularity for an ordered reader.** A sixteenfold page change
moves an ordered run by about 2%, because in both cases readahead is
issuing 128 KiB requests — 159 of them for 5,000 pages at 4 KiB, and 157
at 64 KiB. Ordering the same accesses is worth more than 300× on a disk.
Tune the container if you like; the ordering is the whole of it.

### Readahead is asymmetric, and that is the point

The kernel does not fetch only what was asked for. On a stream it judges
sequential it fetches ahead, doubling the window to `ra_pages` (128 KiB,
256 KiB after `POSIX_FADV_SEQUENTIAL`); on access it judges random it
fetches the requested pages and stops. Simulated, over the same device:

| Reader | Share of device traffic that is readahead | Positioning | Bandwidth used |
|---|---|---|---|
| Ascending | > 50% | 9% | 91% |
| Scattered | < 5% | 99% | 1% |

Two consequences worth separating.

**Readahead coalesces**, replacing many page faults with one window
fetch. That is what it is for, and it collapses an ordered reader's
request count by more than 4×.

**Readahead is a guess, and guesses cost bytes.** On an ascending reader
that *skips* — which is what a pass of an ordered rewrite does — it fills
the gaps the reader was deliberately stepping over, moving more than
twice the bytes. On seek-bound media that trade is roughly neutral,
because the skipped pages were nearly free to pass over anyway. Where
bandwidth is the constraint it is a straight loss. The window is
therefore worth advising *down*, not up, for a sparse-regime pass —
the opposite of the `advise_sequential()` guidance elsewhere in these
documents, which is correct only in the dense regime.

Coalescing pays where a per-request cost binds: on a modern drive with
one core issuing, readahead more than doubles the achievable application
rate, because the host was the constraint and there are now far fewer
requests to pay for.

### What else the platform costs

Three ceilings sit above the device's own, and none of them is visible in
a single-device, single-socket benchmark:

- **Memory bandwidth.** Every byte that arrives from storage is touched
  several times — DMA in, copy to a user buffer, scatter into an output
  segment — so a 7 GB/s drive can generate 25 GB/s of memory traffic.
  Modelled at three touches per byte, a host with 6 GB/s of memory
  bandwidth cuts a modern drive's usable throughput by more than half.
  **Cache hits are not free either**: a hit is a memcpy, and a workload
  served almost entirely from cache can be entirely memory-bound.
- **The upstream link.** A device's `bus_rate` is its own link, not the
  aggregate. Eight drives behind one PCIe 4.0 x16 root port get about
  3.5 GB/s apiece however fast each of them is on its own.
- **NUMA.** Issuing from the wrong socket costs a per-request latency and
  a large share of memory bandwidth. The first is **largely hidden by
  concurrency** — at 128 outstanding requests it costs a few percent —
  which is the same mechanism that erodes the case for ordering. The
  second is not hidden at all: a streaming reader on a socket with one
  channel pair loses more than a quarter of its throughput across the
  interconnect. NUMA is invisible until bandwidth is the constraint, and
  then it is most of the problem.

Every device figure quoted in this document was measured on a single
socket with a dedicated link, so all of it describes the `LOCAL`,
`DEDICATED` case and anything else is extrapolation.

### The host is a bottleneck too

[Ren et al.](https://dl.acm.org/doi/10.1145/3629526.3645053) find that
with high-performance NVMe SSDs the **CPU saturates before the device
does** at 4 KiB — 100% utilized while the drive is not — and that Linux
I/O schedulers add up to 63.4% throughput overhead on top. A storage cost
model with no host-side term is modelling the wrong constraint above
roughly half a million operations per second.

Charging ~1.7 µs of CPU per request reproduces it. On the modern drive:

| Configuration | 4 KiB IOPS | Device bandwidth used | Limited by |
|---|---|---|---|
| 2016 NVMe, 8 cores | 123,953 | 34% | device |
| Modern NVMe, 1 core | 587,743 | 34% | **host CPU** |
| Modern NVMe, 8 cores | 1,177,742 | 69% | device |

For a rewrite this matters directly: it means the throughput a plan
predicts is unreachable from one thread, and that the gain from ordering
can be masked entirely by a host that cannot issue fast enough to expose
it.

### How accurate is any of this

Scored across three devices and every block size from 512 B to 1 MiB, on
throughput *and* latency, as mean absolute percentage error against the
measured fio output. `cargo run -p veks-studies --bin veks-study`
reproduces it.

| Metric | Samples | MAPE | Worst | Bias |
|---|---|---|---|---|
| Read throughput | 36 | **3.7%** | 14.6% | −0.1% |
| Mean latency | 27 | **4.1%** | 17.1% | +1.7% |
| p50 | 27 | 6.6% | 24.2% | +5.8% |
| p95 | 27 | 8.7% | 22.3% | −5.4% |
| p99 | 27 | 11.0% | 30.3% | −5.8% |
| Sequential write throughput | 3 | **0.2%** | 0.5% | +0.2% |
| Sequential write latency | 3 | **0.7%** | 1.3% | −0.7% |

Against the bars the literature states, like for like:

| | Reported | This model |
|---|---|---|
| [MQSim](https://www.usenix.org/conference/fast18/presentation/tavakkol) (FAST '18), throughput vs 4 real SSDs | 6–18% | 3.7% MAPE / 14.6% worst |
| [SimpleSSD](https://arxiv.org/pdf/1705.06419), worst-case throughput | 28% | 14.6% |
| SimpleSSD, worst-case latency | 36% | 30.3% (p99) |
| [Generative black-box models](https://arxiv.org/pdf/2307.02073) | 4–10% IOPS, 3–16% latency | 3.7% / 4.7% |

**What that does and does not establish.** The device parameters were
fitted to the random-read throughput curve, so agreement *there* is a fit
rather than a test. Sequential throughput, the contention sweep and the
latency distribution were not fitted to and are predictions. The NAND
page-type spread and the disk's rotational-selection accuracy *were*
fitted against measured percentiles, so the distribution shape is a
calibrated output — though the read-variation draw is mean-preserving by
construction, so it cannot have flattered the means.

Latency is compared on fio's `lat` basis — submission to completion —
not `clat`. Comparing a simulator that timestamps at creation against
`clat` understates the target by a couple of percent and shows up as a
positive bias in every latency metric.

Four mechanisms were found by taking latency and the write path
seriously, and none would have surfaced from read throughput alone:

- **A die is busy for its own page read, not for the whole request.**
  Holding it for the request's duration made a 32 KiB read occupy its die
  eight times too long. It cost 2% on throughput and **51% on p99**, with
  a +47% bias — the signature of a missing term rather than noise. Fixing
  it took the SATA SSD's p99 error to 6%.
- **Reordering has to be bounded.** A device that always serves the
  cheapest request starves whatever it keeps passing over, driving a
  competing random reader to zero — which the mixed-workload measurements
  plainly do not show. Bounding deferral at 600 ms restores it, and that
  figure was arrived at by fitting the contended sweep before being found
  to match the longest completion latency fio recorded on the drive
  (607.7 ms).
- **Programming is charged by the byte, not per request.** A flat
  per-request program cost makes large writes far too cheap and
  overstated NVMe sequential write throughput by 57%.
- **A disk has no separate write path, and giving it one is actively
  harmful.** A write ceiling even slightly below the media rate
  desynchronises sequential writes from the platter — each block finishes
  a fraction after its successor's sector has passed, so the head waits
  almost a full revolution — and halves modelled sequential write
  throughput.

### What the model still does not represent

Stated so that a reader does not assume otherwise from the surrounding
detail:

- **Random writes and garbage collection.** Sequential write *is* now
  validated, to 0.2%, but the corpus has no random-write job. Write
  amplification and the volatile write buffer are modelled and behave in
  the right direction; neither is checked against a measurement.
- **Filesystem geometry.** Extent layout, fragmentation, journal traffic
  and metadata reads are absent; the address space is flat.
- **The scheduler's own cost.** Ren et al. measure up to 63.4% throughput
  overhead from Linux I/O schedulers; the schedulers modelled here are
  free.
- **Tail latency.** Everything reported is a mean or a rate. The
  read-after-write serialisation that dominates p99.99 is present as a
  mechanism but is not characterised.
- **Multi-device striping.** One device at a time, with the upstream link
  represented only as a share.

### Concurrency is a correctness concern, not a tuning knob

The same result set measures a random reader running alongside a
sequential reader and writer. Under a rate cap the three split a nearly
constant bandwidth pool — on the NVMe drive, total throughput stays
within 2% of 830 MB/s across every capped point — which is what makes it
legitimate to add read and write terms at all.

Removing the cap does not shift the split; it ends it. The random reader
collapses from 100,937 to **567** IOPS on NVMe, 48,474 to **254** on the
SATA SSD, and 208 to **25** on the spinning disk. An unthrottled
sequential writer does not slow a concurrent reader down, it removes it
from the schedule. **Transfer must rate-limit its output stream against
its input stream**, or the cost model above does not describe the run.

**Why it happens** is die-level blocking, not bandwidth sharing. A flash
write occupies its die for the whole program operation — roughly an order
of magnitude longer than a read — and a read that lands on that die waits
for it, however idle the rest of the device is. Reads also queue behind
writes at the shared channel controller. The literature puts the
resulting tail cost at [up to 20× from read-after-write
serialization](https://people.ucsc.edu/~hlitz/papers/rail.pdf), with
five-nines read latency reaching 4.5 ms under only 20% sporadic writes.

Modelling that explicitly — dies with address affinity, held for the
duration of a program — brings the simulator within range of the
measurement. Reproducing the `mixed` job faithfully (random reader at
8–16 KiB alongside a rate-capped sequential reader and writer), simulated
random-read IOPS land within about 30% of measured at every capped point,
and the uncapped collapse comes out roughly half as severe as measured
rather than two orders of magnitude too mild. An earlier version of this
model shared bandwidth fairly and produced 2× where the measurement shows
178×; that was a missing mechanism, not a tuning error.

## Worked examples

Generated by `cargo run -p veks-studies --bin veks-study`, which prices
the amplification formula against the device models in
`veks-studies/src/device.rs`. Those models reproduce the measured sweeps
to within 5% (spinning disk), 7% (SATA SSD) and 10% (NVMe) at every block
size up to 128 KiB. `W = 128 KiB`, full permutations (`N = N_src`).

### A — the collection fits the budget

Any `N·R ≤ M`. The floor of two segments applies, `A = 2`, and both
approaches finish in the time it takes to stream the data twice. Not
worth reasoning about; permute in memory instead.

### B — 100M records of 1.5 KiB (143 GiB), `M = 8 GiB`

```
 P = 18     w = 85     A = 18·(1 − e^−4.72) = 17.8   (dense)
```

| | bytes read | spinning disk | SATA SSD | NVMe |
|---|---|---|---|---|
| naive gather | 143 GiB | 4.3 days | 25 min | 15 min |
| gsplat | 2552 GiB | **4.2 h** | 86 min | 28 min |
| | | **25× faster** | 3.4× slower | 1.9× slower |

gsplat reads 18× the bytes and finishes 25× sooner on the disk, because
each pass is a scan rather than 1.2 million seeks.

It loses on flash **at this budget**. 18 passes is more than the
random-access penalty flash charges for a 1.5 KiB record, so the
re-reads cost more than the seeks they avoid. That is a statement about
`M`, not about flash — see [Where ordering starts to
pay](#where-ordering-starts-to-pay), where the same rewrite with a
larger budget wins on all three devices.

### C — 450M records of 4 KiB (1717 GiB)

```
 M = 32 GiB  → P = 54, A = 24.1        (sparse: P > w = 32)
 M = 230 GiB → P = 8,  A = 7.9         (dense)
```

| | bytes read | spinning disk | SATA SSD | NVMe |
|---|---|---|---|---|
| naive gather | 1717 GiB | 19.7 days | 2.5 h | 80 min |
| gsplat, `M = 32 GiB` | 41.4 TiB | **6.0 days** | 22.9 h | 7.4 h |
| gsplat, `M = 230 GiB` | 13.5 TiB | **23.6 h** | 8.1 h | 2.6 h |

Raising the budget drops `P` from 54 to 8 and `A` from 24.1 to 7.9 — a
3× cut in total I/O from one configuration change, no code involved. On
the disk it is worth more than that: 6.0 days to 23.6 hours, because the
extra memory also moves the run from the sparse regime into the dense
one, where the streaming bound applies and passes become scans.

## Where ordering starts to pay

Both examples above lose on flash, and it would be easy to read that as
"gsplat is a spinning-disk technique." It is not. They lose because
their memory budgets sit below the crossover, and the crossover has a
closed form.

A gather costs `N / IOPS(R)`. An ordered rewrite costs `P` scans, which
by the streaming bound is `P · N·R / BW_seq`. Setting them equal:

```
     N / IOPS(R)   =   P · N·R / BW_seq
     BW_seq / (R · IOPS(R))   =   P
     penalty(R)   =   P
```

**Ordering pays exactly when the pass count is below the random-access
penalty at the record size.** `N` cancels: the line does not depend on
how much data there is. And since `P ≈ payload / M`:

```
     M  >  payload / penalty(R)
```

`penalty(R) = BW_seq / (R · IOPS(R))` is the ratio of streaming
throughput to the throughput of fetching records where they lie.

**`penalty` is also a function of concurrency, and the table below is a
statement about one particular depth.** Random access loses to streaming
because every request pays an access latency that a stream does not, and
concurrency hides latency: with `k` requests outstanding, the latency is
amortised `k` ways. Every figure in the next table is derived from the
perfscripts corpus, which was captured at `iodepth=10`. See [How the line
moves with concurrency](#how-the-line-moves-with-concurrency) — it moves
a long way.

For a 143 GiB payload at `iodepth=10`, using the measured device models:

| `R` | \_\_\_\_\_\_ spinning disk \_\_\_\_\_\_ | | \_\_\_\_\_\_ SATA SSD \_\_\_\_\_\_ | | \_\_\_\_\_\_ NVMe \_\_\_\_\_\_ | |
|---|---|---|---|---|---|---|
| | penalty | min `M` | penalty | min `M` | penalty | min `M` |
| 128 B | 5699 | 26 MiB | 55.5 | 2.6 GiB | 110 | 1.3 GiB |
| 512 B | 1425 | 103 MiB | 13.9 | 10.3 GiB | 27.6 | 5.2 GiB |
| 1540 B | 475 | 309 MiB | 4.6 | 31.0 GiB | 9.2 | 15.6 GiB |
| 4 KiB | 179 | 818 MiB | 1.7 | 82.5 GiB | 3.6 | 40.1 GiB |
| 16 KiB | 45.5 | 3.1 GiB | 1.1 | 129 GiB | 1.6 | 90.3 GiB |
| 64 KiB | 12.1 | 11.8 GiB | 1.0 | 140 GiB | 1.1 | 132 GiB |

Example B used `M = 8 GiB` against an NVMe line of 15.6 GiB, which is
why it lost. Raise the same rewrite to `M = 32 GiB` and it flips:

| | bytes read | spinning disk | SATA SSD | NVMe |
|---|---|---|---|---|
| naive gather | 143 GiB | 4.3 days | 25 min | 15 min |
| gsplat, `M = 32 GiB` | 715 GiB | **79 min** | 27 min | **9 min** |
| | | **80× faster** | 1.1× slower | **1.7× faster** |

The correct statement is not "flash defeats ordering" but **"flash moves
the line."** Three readings of the table are worth keeping:

- **The line is a memory requirement, and it scales with payload.** The
  ratio `payload / penalty` is fixed, so a 10× larger dataset needs a 10×
  larger budget to stay on the winning side. This is the sense in which
  a working set larger than memory matters: it is not that gsplat needs
  the data to fit, but that `P` grows as the payload outgrows `M`, and `P`
  is what has to stay under the penalty.
- **Record size dominates device class.** A 128 B record on NVMe carries
  a penalty of 110; a 64 KiB record on a spinning disk carries 12. Small
  records on fast flash are a better case for ordering than large records
  on slow disk.
- **Above about 16 KiB on flash, ordering has nothing left to sell.** The
  penalty is near 1, so the required budget approaches the payload
  itself — at which point the rewrite fits in memory and the question is
  moot. That is the real boundary of the technique on flash, and it is a
  statement about record size, not about the device.

## How the line moves with concurrency

The table above holds `k = 10` fixed because its source data did. That is
not a safe thing to leave implicit. The
[MQSSD model](https://arxiv.org/abs/2507.06349) (Ransom, Lim &
Mitzenmacher, 2025), which extends the external-memory model by making
concurrency a first-class parameter, reports that on a current drive the
random-to-sequential **read** ratio falls to **1.3–1.5× at `k = 128`**,
against 38–57× for writes at `k = 1`. Concurrency, not ordering, is doing
most of the work available on modern flash.

Modelling a current drive — calibrated to ~1M 4 KiB IOPS and 7 GB/s
sequential from [Ren et al., ICPE '24](https://dl.acm.org/doi/10.1145/3629526.3645053),
and to the MQSSD ratio band — the line moves like this for a 4 KiB record
against a 143 GiB payload:

| Offered depth | 2016 NVMe penalty | min `M` | Modern NVMe penalty | min `M` |
|---|---|---|---|---|
| 1 | 35.7 | 4.0 GiB | 133.7 | 1.1 GiB |
| 4 | 8.9 | 16.0 GiB | 33.4 | 4.3 GiB |
| 10 | 3.6 | 40.1 GiB | 13.4 | 10.7 GiB |
| 32 | 3.4 | 41.5 GiB | 4.2 | 34.2 GiB |
| 64 | 3.4 | 41.5 GiB | **2.1** | 68.4 GiB |
| 128 | 3.4 | 41.5 GiB | **1.4** | 100.4 GiB |

**The conclusion this changes.** A rewrite always makes at least two
passes. Once the penalty drops below 2, no budget can win, because the
floor of the pass count is already above the number it has to beat. On a
current NVMe drive at `k ≥ 64`, ordering a 4 KiB-record rewrite **cannot
pay at any memory budget**. The 2016 drive never reaches that point
because its command rate binds first and pins its penalty at 3.4.

This does not retire the technique; it narrows where it applies:

- **Seek-bound media are unaffected.** A disk's penalty at 4 KiB is 179
  and concurrency cannot fix a single head.
- **Small records are unaffected.** At 128 B the penalty stays above 8 on
  every device at every depth modelled, because the record is far below
  what any of them serves efficiently.
- **Low-concurrency pipelines are unaffected.** A single-threaded reader
  sees a penalty of 134 on the same modern drive that offers 1.4 at
  `k = 128`. If a rewrite is not issuing deep, it is in the regime where
  ordering pays — and if it is, it may not need ordering at all.

The honest summary is that **ordering and concurrency are substitutes**.
Both exist to stop the device idling between requests. Spend whichever is
cheaper to obtain.

### D — the same rewrite on object storage

Request count, not wall-clock, is the headline. Naive gather issues `N`
ranged GETs — 450 million requests, on the order of a hundred dollars
at typical GET pricing and hours of latency-bound wall-clock even at
high concurrency. gsplat with coalesced ranges issues a few tens of
thousands of large GETs per pass, four orders of magnitude fewer, and
writes one multipart part per segment. This is the tier where the
algorithm's advantage is largest and least ambiguous.

## Reading the results

1. **`P` is the dominant lever.** Cost is linear in `P` up to the
   crossover and flat after it, so every doubling of the budget below
   `P = w` roughly halves the I/O.
2. **The win is decisive wherever random access is expensive** —
   spinning disk, network block storage, and above all per-request
   pricing.
3. **On fast flash under a deep queue, naive gather is competitive and
   can be faster**, because the comparison inverts: naive moves about
   `2×` the data, gsplat moves `A×`. gsplat still earns its place there
   for what a throughput number does not show — bounded residency, a
   cache footprint that does not evict everything else, contiguous
   output, resumability, and predictable progress.
4. **Prefetch hints are regime-dependent**, not universally good.

## Beyond one level

gsplat is a **one-level distribution**: a single partition of the
output ordinal space, so the source is revisited once per segment. The
external-memory lower bound for permutation is

```
 permute(N) = Θ(min(N, sort(N)))    sort(N) = (N/B)·log_(M/B)(N/B)
```

At example C's parameters `log_(M/B)(N/B) ≈ 1.3`, so an optimal
external permutation costs roughly 2–3 data-equivalents of I/O against
gsplat's 24. The gap closes with a **two-level** variant:

```
 level 1  stream the source once; append each record to the spill
          bucket of its destination segment     (1 read + 1 write, sequential)
 level 2  per segment: read its bucket, scatter in memory, write
          contiguously                          (1 read + 1 write, sequential)
```

Total ≈ 4 data-equivalents *independent of `P`*, paid for with one
data-sized scratch area and `P` write buffers.

### Where the two-level variant starts to pay

The hedge above — "unnecessary while they fit in a handful of passes" —
holds, and the handful is **four**.

The arithmetic is one line. The two-level form moves the payload four
times whatever `P` is; the single-level form moves it `A(P) + 1` times.
They are level at `A(P) = 3`, and while `P ≪ w` the amplification is
very close to `P` itself, so the crossover lands at three or four
segments:

| segments `P` | `A(P)` | single-level | two-level |
|---|---|---|---|
| 2 | 2.0 | 3 data-equivalents | 4 |
| 3 | 3.0 | 4 | 4 |
| 4 | 4.0 | 5 | 4 |
| 16 | 16.0 | 17 | 4 |
| 256 | 100.7 | 102 | 4 |
| 1024 | 120.3 | 121 | 4 |

`veks-studies` measures the crossover at **exactly four segments**, and
finds it in the same place on every device modelled — which is what it
should be, since both arms are sequential there and the comparison is
between two byte counts rather than between two access patterns
(`scale::crossover::staging_overtakes_the_rescan_at_about_four_segments`,
`…::the_crossover_is_the_same_on_every_device`).

Two consequences worth stating plainly:

- **Below four segments, the single-level form is genuinely better.** It
  never touches scratch. A claim that staging always wins would be
  false, and the cost model should not make it.
- **Above four segments there is no upper end to the gap.** The
  single-level cost keeps climbing with `P`; the two-level cost does
  not move at all until `P` exceeds the fan-out `f = M/W` and buys
  another stage, and `f` is a quarter of a million for a 32 GiB budget
  with 128 KiB containers. At a terabyte with 32 GiB of memory the
  measured spread is 9.8 h against 41.9 m; at 1 GiB of memory it is
  21.2 h against the same 41.9 m.

The number of stages is `ceil(log_f(P))`, so the growth in `P` is
logarithmic and the base is enormous. A terabyte and a petabyte both
need exactly one distribution stage.

Reference: A. Aggarwal and J. S. Vitter, "The Input/Output Complexity
of Sorting and Related Problems," *Commun. ACM* 31(9), 1988. The
`Θ((N/B)·log_(M/B)(N/B))` bound is what the two-level form realizes;
`log_(M/B)` is the `log_f` above.

Back to the overview: [README.md](./README.md).

## References

Every device constant, mechanism and accuracy claim in this document
traces to one of the following. Where a parameter was fitted rather than
read from a source, the text says so at the point it is used.

### Measurement

| Source | What it grounds here |
|---|---|
| [perfscripts](https://github.com/jshook/perfscripts) | Every figure for the three historical devices: random-read sweeps 512 B–16 MiB, sequential read and write, the mixed reader/writer contention sweep, and the latency distributions validated against |
| [Ren et al., ICPE '24 — *BFQ, Multiqueue-Deadline, or Kyber?*](https://dl.acm.org/doi/10.1145/3629526.3645053) ([artifact](https://zenodo.org/records/10599514)) | The modern NVMe regime (~1M 4 KiB IOPS, 7 GB/s sequential, 68 µs read latency); the finding that the CPU saturates before the device; the 63.4% scheduler-overhead figure |
| [Didona et al., SYSTOR '22 — *Understanding Modern Storage APIs*](https://atlarge-research.com/pdfs/2022-systor-apis.pdf) | Per-request host cost by API: libaio 144.9 KIOPS on one core, io_uring 171.5, io_uring polled 173.0, SPDK 305.9 — a 3.3–6.9 µs spread that is a property of the API and kernel, not the drive |
| Device teardowns (950 PRO UBX controller, 850 PRO MEX) | Die counts and controller reach: eight channels with eight-way interleaving; MLC V-NAND in both drives |
| [SNIA SSS Performance Test Specification](https://www.snia.org/tech_activities/standards/curr_standards/pts) | The device-level measurement methodology, including the preconditioning this model does not represent |

### Models and mechanisms

| Source | What it grounds here |
|---|---|
| [Ransom, Lim & Mitzenmacher, 2025 — *Multi-Queue SSD I/O Modeling*](https://arxiv.org/abs/2507.06349) | Concurrency as a first-class model parameter; the 1.3–1.5× random-to-sequential read ratio at k=128 that drives [How the line moves with concurrency](#how-the-line-moves-with-concurrency) |
| [Tavakkol et al., FAST '18 — *MQSim*](https://www.usenix.org/conference/fast18/presentation/tavakkol) | Per-page-type NAND latency parameterisation; the 6–18% accuracy bar |
| [Jung et al. — *SimpleSSD*](https://arxiv.org/pdf/1705.06419) | The same parameterisation; the 28% throughput / 36% latency worst-case bars |
| [Lebrecht, Dingle & Knottenbelt, QEST '09 — *Zoned Disk Drives with I/O Request Reordering*](http://www.doc.ic.ac.uk/~wjk/publications/lebrecht-dingle-knottenbelt-qest-2009.pdf) | Reordering on rotating media as a modelled phenomenon; the basis for separating a device's selection cost from its service cost |
| [RAIL — *Predictable, Low Tail Latency for NVMe Flash*](https://people.ucsc.edu/~hlitz/papers/rail.pdf) | Die-level read/write blocking and read-after-write serialisation, the mechanism behind the write-starvation figures |
| [Device-Level Optimization Techniques for SSDs: A Survey](https://arxiv.org/abs/2507.10573) | NAND timing by cell type — MLC reads 40–110 µs and programs 0.4–1.5 ms, TLC reads 66–170 µs — with LSB pages needing one sensing pass and MSB pages more. Grounds the read-latency spread and corroborates the program rates |
| [Park et al., *Reducing SSD Read Latency by Optimizing Read-Retry*](https://arxiv.org/pdf/2104.09611) | Read-retry at shifted reference voltages, the mechanism behind the modelled latency tail |
| [Schindler & Ganger, *Automated Disk Drive Characterization* (DIXtrac), CMU-CS-99-176](https://www.pdl.cmu.edu/PDL-FTP/DriveChar/cs-99-176.pdf) | Over 100 extracted per-drive parameters including mechanical timings — the reference standard against which the class-typical seek figures used here are an admitted shortcut |
| [Aggarwal & Vitter, CACM 31(9), 1988](https://dl.acm.org/doi/10.1145/48529.48535) | The external-memory sorting and permutation bounds this cost model extends |

### Judging a simulator

| Source | What it grounds here |
|---|---|
| [Performance Modeling of Data Storage Systems using Generative Models](https://arxiv.org/pdf/2307.02073) | The 4–10% IOPS / 3–16% latency bar for black-box storage models |
| [Different Perspectives of Memory System Simulation, 2026](https://arxiv.org/abs/2604.16965) | Why accuracy is reported end-to-end rather than from internal simulator counters: application-level performance is frequently decoupled from them |

### Workload data not used

Named so the gap stays visible. The
[SNIA IOTTA repository](https://iotta.snia.org/), the
[Alibaba block traces](https://github.com/alibaba/block-traces) and the
[Meta CacheLib traces](https://github.com/cacheMon/cache_dataset) are
real request streams this model could be driven by and currently is not.
Every workload behind the numbers above is either synthetic or generated
from an algorithm's own trace, which means the model is validated against
real *devices* but not against real *workloads*.
