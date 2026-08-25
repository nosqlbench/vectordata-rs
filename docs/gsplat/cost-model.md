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
**Naive moves less data; gsplat moves it in a better order.** That is
the whole trade, and it makes the pass count the term that decides
whether gsplat is a win.

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
(`direct=1`, `iodepth=10`, 60 s, block sizes 512 B–16 MiB), not
estimated. `W` is read off the curve as the smallest block at which
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

**Page size is second-order.** On a bytes-moved basis a larger container
is the lever, and `A(P)` treats it that way. In time it is not: once
access ascends, small contiguous reads cost a disk almost nothing extra,
so a sixteenfold page change moves an ordered run by under 15% — and a
scattered one by about as little. Ordering the same accesses is worth
more than 8×. The container size is worth tuning; the ordering is worth
having.

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
data-sized scratch area and `P` write buffers. Worth building when
rewrites routinely land in the sparse regime; unnecessary while they
fit in a handful of passes.

Reference: A. Aggarwal and J. S. Vitter, "The Input/Output Complexity
of Sorting and Related Problems," *Commun. ACM* 31(9), 1988.

Back to the overview: [README.md](./README.md).
