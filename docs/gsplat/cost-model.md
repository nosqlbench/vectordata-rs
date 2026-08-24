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

| Tier | `W` | Random access | What dominates |
|------|-----|---------------|----------------|
| Spinning disk | 128–256 KiB | 5–10 ms | `random_ops` — naive is hopeless |
| Network block | 128–256 KiB | 0.5–2 ms | `random_ops`, then bandwidth caps |
| Flash, deep queue | 4–128 KiB | ~100 µs at 10⁵–10⁶ IOPS | bytes moved — the terms converge |
| Object storage | 1–8 MiB effective | 20–100 ms, **priced per request** | request count and price, not time |

## Worked examples

`B = 4 KiB`, `W = 128 KiB`, disk 200 MB/s @ 8.5 ms, flash 3 GB/s @
600k IOPS. Full permutations (`N = N_src`).

### A — the collection fits the budget

Any `N·R ≤ M`. The floor of two segments applies, `A = 2`, and both
approaches finish in the time it takes to stream the data twice. Not
worth reasoning about; permute in memory instead.

### B — 100M records of 1.5 KiB (154 GB), `M = 8 GiB`

```
 S = 5.59M → P = 18     w = 85     A = 18·(1 − e^−4.72) = 17.8   (dense)
```

| | tier bytes read | random ops | disk | flash |
|---|---|---|---|---|
| naive gather | 563 GB | 100M | **9.8 days** | **4 min** |
| gsplat | 2.74 TB | 18 | **4 h** | **16 min** |

59× faster on disk; 4× *slower* on flash.

### C — 450M records of 4 KiB (1.84 TB)

```
 M = 32 GiB → P = 54, A = 24.1        (sparse: P > w = 32)
 M = 230 GiB → P = 8,  A = 7.9        (dense)
```

| | tier bytes read | random ops | disk | flash |
|---|---|---|---|---|
| naive gather | 3.7 TB | 450M | **44 days** | **31 min** |
| gsplat, `M = 32 GiB` | 44.4 TB | 54 | **2.6 days** | **4.1 h** |
| gsplat, `M = 230 GiB` | 14.5 TB | 8 | **20 h** | **1.5 h** |

Raising the budget drops `P` from 54 to 8 and `A` from 24.1 to 7.9 — a
3× cut in total I/O from one configuration change, no code involved.

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
