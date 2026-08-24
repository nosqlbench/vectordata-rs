# SPLAT Cost Model — naive permutation vs SPLAT

Order-of-growth for the naive scatter/gather permutation and for SPLAT,
the one constant that separates them, and worked examples at project
scale. Canonical spec:
[09-algorithms.md §9.4](../09-algorithms.md#94-splat-io-sympathetic-ordinal-rewrite).

## Symbols

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| `N` | records extracted (output count) | 10⁴ … 10⁹ |
| `N_src` | records in the source (`≥ N`) | — |
| `R` | record bytes — `4 + dim × elem_size` | 1540 B (384-d f32), 4100 B (1024-d f32), 2052 B (1024-d f16) |
| `M` | governor memory budget | 256 MiB … 100s of GiB |
| `S` | records per segment = `max(M / R, 1)` | — |
| `P` | passes = `max(ceil(N / S), 2)` | 2 … 100s |
| `B` | page / minimum transfer unit | 4 KiB |
| `W` | readahead window (device fetch granularity) | 128 KiB; 256 KiB after `FADV_SEQUENTIAL` |
| `w` | records per window = `W / R` | 31 at R=4100, W=128 KiB |
| `BW` | sequential bandwidth | 200 MB/s HDD, 3 GB/s NVMe |
| `IOPS` | random IOPS at the achieved queue depth | ~150 HDD, 10⁵–10⁶ NVMe |
| `T` | worker threads | governor-controlled |

Throughout, the reorder map assigns `out[i] = src[map[i]]`.

## The naive baselines

Two ways to apply the map without a segment buffer. Both are `Θ(N)`
random I/O operations; they differ in which side pays.

```
 gather — walk the output in order          scatter — walk the source in order
 for i in 0..N:                             for j in 0..N_src:
     out[i] = read(src, map[i])                 if inv[j] is set:
                                                    write(out, inv[j], src[j])

 reads  random   ← the cost                 reads  sequential
 writes sequential                          writes random   ← the cost
```

Gather is the better naive baseline and the one the comparisons below
use. Scatter is strictly worse on every device we target:

- Partial-block writes force read-modify-write, so each random write
  moves bytes twice.
- Random writes dirty scattered pages; writeback storms interleave with
  the read stream.
- It needs the *inverse* map, either materialized (`4·N_src` bytes of
  RAM) or produced by sorting the map (`Θ(N log N)` up front).

## Order of growth

| Quantity | naive gather | naive scatter | SPLAT |
|---|---|---|---|
| Random I/O ops | `Θ(N)` reads | `Θ(N)` writes | `Θ(P)` — one seek per pass |
| Monotone / sequential ops | `Θ(N·R / B)` writes | `Θ(N_src·R / B)` reads | `Θ(N)` ascending reads + `Θ(N·R / B)` writes |
| Bytes read from device | `Θ(N · ⌈R/B⌉ · B)` | `Θ(N_src · R)` | `Θ(A · N · R)` — `A` below |
| Bytes written to device | `Θ(N · R)` | `Θ(2 · N · ⌈R/B⌉ · B)` | `Θ(N · R)` |
| Expected seek distance | `Θ(N_src · R)` — a third of the file | same | `Θ(P · R)` forward, never backward |
| Resident memory | `Θ(R)` | `Θ(R + 4·N_src)` | `Θ(M)` = buffer + `16·S` plan |
| CPU | `Θ(N)` | `Θ(N_src)` | `Θ(N log S)` = `Θ(N log(N/P))` |
| Map I/O | `Θ(4N / B)`, read once | `Θ(4N / B)` + inversion | `Θ(4N / B)`, read once |
| Passes over the source | 1 (scattered) | 1 | `P` (ascending) |

Two entries deserve emphasis because they are where the intuition
usually goes wrong:

- **SPLAT does not reduce the number of read operations.** Both
  approaches issue `N` record reads, one per output record. SPLAT
  changes their *order* (monotone within a pass) and the write side
  (`P` contiguous bursts instead of `N` scattered writes).
- **SPLAT trades CPU for I/O.** The `Θ(N log(N/P))` sort is work the
  naive version never does. It is a good trade whenever the device is
  the bottleneck, and a bad one when it isn't (see NVMe below).

## Read amplification `A` — the constant that decides everything

Programs read each mapped record exactly once (the invariant in
[README.md](./README.md)); *devices* fetch whole windows. Per pass a
pass needs `S = N/P` of the `N_src` source records, so for a window of
`w` records the probability it holds at least one needed record is

```
 p_window = 1 − (1 − S/N_src)^w  ≈  1 − exp(−w·S / N_src)
```

Summing over `P` passes and dividing by the file size gives the
amplification factor — device bytes read per byte of live data. For a
full permutation (`N = N_src`, the shuffle split case):

```
 A(P) = P · (1 − exp(−w / P))          A ≤ min(P, w)

 general form (partial extract):
 A(P) = (P · N_src / N) · (1 − exp(−w·N / (P·N_src)))
```

`A` interpolates between two regimes, with the crossover at **`P = w`**:

| Regime | Condition | Behavior | `A` |
|--------|-----------|----------|-----|
| Dense | `P ≤ w` (stride `P·R` fits inside a window) | every window is needed on every pass; each pass is a true sequential scan | `≈ P` |
| Sparse | `P > w` | windows are touched once every few passes; reads become per-record fetches, still ascending | `≈ w` |

`A` for `R = 4100 B` (1024-d f32), `W = 128 KiB`, so `w = 31`:

| `P` | 2 | 4 | 8 | 16 | 31 | 53 | 100 | 215 |
|-----|---|---|---|----|----|----|-----|-----|
| `A` | 2.0 | 4.0 | 7.8 | 14.4 | 19.6 | 23.5 | 27.0 | 29.1 |

The naive baseline's amplification is fixed and small — `⌈R/B⌉·B / R`,
about 2.0 for `R = 4100` — because random access defeats readahead
entirely, so the kernel fetches only the pages the record occupies.
**Naive moves less data; SPLAT moves it in a better order.** That is
the whole trade, and it means the pass count `P` is not a tuning detail
— it is the term that decides whether SPLAT is a win.

`A(P)` is the upper bracket, realized when readahead engages. If it
does not, SPLAT's read volume falls to the same `N · ⌈R/B⌉ · B` as
naive and SPLAT is then strictly better. `advise_sequential()` on the
source reader (see [04-assemble.md](./04-assemble.md)) pushes toward
the upper bracket by design — correct in the dense regime, costly in
the sparse one, where it doubles `W` and therefore `A`.

## Time model

```
 T_io    ≈ max(bytes_read / BW_read, random_ops / IOPS_eff)
           + bytes_written / BW_write
 IOPS_eff = min(IOPS_device, queue_depth / access_latency)

 T_cpu   ≈ N·R / BW_mem            (assemble memcpy, both approaches)
           + N·log(N/P) / (T · c)  (SPLAT linearize only)
```

On seek-bound media the `random_ops / IOPS_eff` term dominates naive by
orders of magnitude. On NVMe with a deep queue it collapses, and the
comparison reduces to raw bytes moved — where naive is ahead.

## Worked examples

All three assume a full permutation (`N = N_src`), `B = 4 KiB`,
`W = 128 KiB`, HDD = 200 MB/s @ 8.5 ms, NVMe = 3 GB/s @ 600k IOPS.

### A — pilot passages: 28,201 × 1024-d f32

`R = 4100 B`, file 116 MB, `M = 256 MiB` (fvec floor).

```
 S = 65,472 → raw passes 1 → P = 2 (the floor)   S = 14,101
 w = 31,  A = 2·(1 − e^−15.5) = 2.00
```

| | reads | writes | random ops | NVMe |
|---|---|---|---|---|
| naive gather | 231 MB | 116 MB | 28,201 | ~0.1 s |
| SPLAT | 231 MB | 116 MB | 2 | ~0.1 s |

Below the floor of two segments the two approaches converge. Anything
that fits in the budget is not worth reasoning about.

### B — 100M × 384-d f32

`R = 1540 B`, file 154 GB, `M = 8 GiB`.

```
 S = 5,577,879 → P = 18   w = 85   A = 18·(1 − e^−4.72) = 17.8   (dense: P < w)
```

| | reads | writes | random ops | HDD | NVMe |
|---|---|---|---|---|---|
| naive gather | 564 GB | 154 GB | 100M | **9.8 days** | **4.0 min** |
| SPLAT | 2.75 TB | 154 GB | 18 | **4.0 h** | **16 min** |

SPLAT is 59× faster on the HDD and 4× *slower* on the NVMe.

### C — S2OA passage spine: 450M × 1024-d f32

`R = 4100 B`, file 1.85 TB. On this node (655 GiB RAM) the fvec budget
rule — reserve up to a quarter of RAM each for source and output page
cache, then take 10% of the remainder — yields `M ≈ 32.7 GiB`.

```
 S = 8,575,610 → P = 53   w = 31   A = 53·(1 − e^−0.585) = 23.5   (sparse: P > w)
```

| | reads | writes | random ops | HDD | NVMe |
|---|---|---|---|---|---|
| naive gather | 3.7 TB | 1.85 TB | 450M | **44 days** | **31 min** |
| SPLAT, `M = 32.7 GiB` | 43.4 TB | 1.85 TB | 53 | **2.5 days** | **4.2 h** |
| SPLAT, `M = 230 GiB` | 14.4 TB | 1.85 TB | 8 | **20 h** | **1.5 h** |

Raising the budget from the default to 230 GiB drops `P` from 53 to 8,
which drops `A` from 23.5 to 7.8 — a 3× cut in total I/O from one
`--resources mem=230G`.

## Reading the results

1. **`P` is the only lever that matters.** Cost is linear in `P` up to
   the crossover and flat after it, so every doubling of the memory
   budget below `P = w` halves the I/O. The default fvec budget is
   deliberately stingy about page-cache headroom, which is right for
   moderate extracts and leaves 3× on the table for the largest ones.
2. **SPLAT's decisive win is seek-bound media** — HDD, network block
   storage, anything where `IOPS_eff` is small. There it is one to two
   orders of magnitude, and it is the difference between hours and
   weeks.
3. **On NVMe with a deep queue, naive gather is competitive and often
   faster**, because the amplification comparison inverts: naive moves
   `≈2×` the file, SPLAT moves `A×`. SPLAT still earns its place there
   for the reasons that do not show up in a throughput number —
   bounded RSS (`pread`, not mmap faulting; see
   [04-assemble.md](./04-assemble.md)), a page-cache footprint that
   does not evict everything else on the box, contiguous output, and
   predictable progress reporting.
4. **The sparse regime wants different kernel advice.** With `P > w`,
   `advise_sequential()` doubles `W` and therefore `A` — at case C it
   costs about 1.5× in read volume. Gating the advice on `P ≤ w` (and
   using `FADV_RANDOM` above it) is a one-line change worth making.

## Beyond one level

SPLAT is a **one-level distribution**: a single partition of the output
ordinal space, so the source is revisited once per segment. The
external-memory lower bound for permutation [6] is

```
 permute(N) = Θ(min(N, sort(N)))    sort(N) = (N/B)·log_(M/B)(N/B)
```

At case C's parameters `log_(M/B)(N/B) ≈ 1.3`, so an optimal external
permutation costs about 2–3 file-equivalents of I/O, against SPLAT's
23.5. The gap closes with a **two-level** pass:

```
 level 1  stream the source once; append each record to the spill file
          of its destination segment            (1 read + 1 write, sequential)
 level 2  per segment: read its spill file, scatter in RAM, write
          contiguously                          (1 read + 1 write, sequential)
```

Total `≈ 4` file-equivalents *independent of `P`*, at the cost of one
file's worth of scratch space and `P` write buffers (`P × 8 MiB`, about
1.7 GB at `P = 215`). Worth building when a spine-scale rewrite becomes
routine; not worth it while extracts fit in a handful of passes.

## Where in code

Pass sizing: the "Determine partition count from memory budget" block in
`sorted_index_extract_{fvec,mvec,slab}`
(`veks-pipeline/src/pipeline/commands/gen_extract.rs`) —
`records_per_partition`, `num_partitions`, `partition_size` are `S`, `P`,
and the segment size above. `advise_sequential()` on the source reader
is at the head of the fvec and mvec extractors.

Back to the overview: [README.md](./README.md).
