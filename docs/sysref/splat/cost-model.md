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
| `W` | readahead window (kernel fetch granularity) | 128 KiB; 256 KiB after `FADV_SEQUENTIAL` |
| `w` | records per window = `W / R` | 31 at R=4100, W=128 KiB |
| `BW` | sequential bandwidth | 196 MB/s HDD, 1427 MB/s NVMe (measured) |
| `IOPS` | random IOPS at the achieved queue depth | 266 HDD, 122k NVMe at 4 KiB, QD10 (measured) |
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
| `A` | 2.0 | 4.0 | 7.8 | 13.7 | 19.6 | 23.5 | 26.7 | 28.9 |

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

Simulating the kernel's readahead rather than assuming a fixed fetch
size confirms the asymmetry the algorithm depends on: an ascending
reader gets readahead on more than half its device traffic, a scattered
one on under 5% of it. It also sharpens the caution above. Readahead
fills the gaps a sparse-regime pass is deliberately skipping, moving more
than twice the bytes; on seek-bound media that is roughly neutral, and
wherever bandwidth binds it is a straight loss. **In the sparse regime
the window is worth advising down, not up.**

## Time model

```
 T_io    ≈ max(bytes_read / BW_read, random_ops / IOPS_eff)
           + bytes_written / BW_write
 IOPS_eff = min(IOPS_device, queue_depth / access_latency)

 T_cpu   ≈ N·R / BW_mem            (assemble memcpy, both approaches)
           + N·log(N/P) / (T · c)  (SPLAT linearize only)
```

On seek-bound media the `random_ops / IOPS_eff` term dominates naive by
orders of magnitude. On NVMe with a deep queue it shrinks but does not
vanish, and the comparison becomes a race between the amplification
SPLAT pays and the random-access penalty naive pays.

**The streaming bound.** One term above is missing, and omitting it
overstates SPLAT by 6× on the HDD. A pass visits windows in ascending
order, so it is never obliged to seek: it can read the source straight
through and discard what it does not need, paying `BW_read` for the
whole extent rather than an access latency per window. That option is
always available, so it caps the cost:

```
 T_read(pass) ≈ min(N_src·R / BW_read, windows_touched / IOPS_eff)
```

In the dense regime the first term wins and a pass is simply a scan. A
gather has no such option, because it cannot know what it may skip.

**Where SPLAT starts to pay.** Equating the two costs gives a closed
form. A gather costs `N / IOPS(R)`; SPLAT costs `P` scans, or
`P · N·R / BW_read`. Setting them equal, `N` cancels:

```
 penalty(R) = BW_read / (R · IOPS(R))

 SPLAT wins  ⟺  P < penalty(R)  ⟺  M > N·R / penalty(R)
```

For `R = 4100 B`, `penalty` is 179 on the HDD and 3.6 on this NVMe — so
the HDD tolerates 179 passes before ordering stops paying, and the NVMe
tolerates 3. The full table is in the [gsplat cost
model](../../gsplat/cost-model.md#where-ordering-starts-to-pay).

**`penalty` depends on I/O concurrency, and both figures above are at
`iodepth=10`**, because that is what the source measurements used. Each
outstanding request hides another's access latency, so the advantage
ordering has to sell shrinks as the queue deepens. On a current NVMe
drive the 4 KiB penalty falls from 134 at one outstanding request to
**1.4 at 128** — and since SPLAT always makes at least two passes, a
penalty below 2 means **no memory budget makes it pay**. The spinning
disk is untouched by this: one head cannot be parallelised.

The practical reading for this project is that SPLAT's case rests on
seek-bound media, small records, or modest issue concurrency — and that a
deeply-pipelined reader on current flash is already buying what ordering
would have sold it. See [How the line moves with
concurrency](../../gsplat/cost-model.md#how-the-line-moves-with-concurrency).

**What the device is actually doing.** An event-driven model of the
storage path (`veks-studies/src/io/`) reproduces these sweeps to within
5% on the HDD without computing a throughput anywhere, and reports where
the time goes. Under scattered 4 KiB reads the HDD spends **99% of its
busy time positioning and 1% transferring**; reading the same bytes in
ascending order inverts that to 2% and 98%. SPLAT's whole effect on that
device is this inversion.

On flash the same comparison shows no difference at all — scattered and
ascending 4 KiB reads reach the same utilization, because there is no
position to pay for. Flash gains from SPLAT only through coalescing:
fewer, larger requests past the controller's command-rate ceiling. That
is a real gain, but it is a different mechanism from the one the HDD
numbers demonstrate, and it is much smaller.

## Worked examples

All three assume a full permutation (`N = N_src`), `B = 4 KiB`,
`W = 128 KiB`. Device figures are measured, not estimated — fio sweeps
from [perfscripts](https://github.com/jshook/perfscripts) at `direct=1`,
`iodepth=10`:

| Device | Seq read | Random 4 KiB | Access latency |
|---|---|---|---|
| HDD (Toshiba HDWD110, 7200 RPM SATA) | 196 MB/s | 266 IOPS (1.1 MB/s) | 3.7 ms |
| NVMe (Samsung 950 Pro) | 1427 MB/s | 122k IOPS (477 MB/s) | 62 µs |

Timings are generated by `cargo run -p veks-studies --bin veks-study`,
which prices the amplification formula against device models validated
against those sweeps. They include the **streaming bound**: a pass reads
containers in ascending order, so it can always stream the source and
discard what it does not want rather than seeking to each container.
That makes a dense-regime pass cost one scan, not one seek per
container — the distinction is worth 6× on the HDD.

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

| | reads | writes | HDD | NVMe |
|---|---|---|---|---|
| naive gather | 143 GiB | 143 GiB | **4.3 days** | **15 min** |
| SPLAT | 2559 GiB | 143 GiB | **4.2 h** | **28 min** |

SPLAT is 25× faster on the HDD and 1.9× *slower* on the NVMe **at this
budget**. It reads 18× the bytes and still wins on the HDD, because that
device charges for arrivals rather than bytes: at 266 IOPS, the 100M
seeks a naive gather issues cost four days no matter how small each one
is.

The NVMe result is a budget artifact, not a property of flash. Ordering
pays whenever `P < penalty(R)`, where `penalty(R)` is the ratio of
sequential throughput to random throughput at the record size — 9.2 for
a 1540-byte record on this drive. `M = 8 GiB` forces `P = 18`, which is
over the line. Raising the budget to 32 GiB drops `P` to 5 and flips the
result:

| | reads | HDD | NVMe |
|---|---|---|---|
| naive gather | 143 GiB | 4.3 days | 15 min |
| SPLAT, `M = 32 GiB` | 715 GiB | **79 min** | **9 min** |

80× on the HDD and 1.7× on the NVMe, from one configuration change.
Equivalently: ordering pays once `M > payload / penalty(R)`, which here
is 143 GiB / 9.2 ≈ 15.6 GiB. See the [gsplat cost
model](../../gsplat/cost-model.md#where-ordering-starts-to-pay) for the
derivation and the full table.

### C — S2OA passage spine: 450M × 1024-d f32

`R = 4100 B`, file 1.85 TB. On this node (655 GiB RAM) the fvec budget
rule — reserve up to a quarter of RAM each for source and output page
cache, then take 10% of the remainder — yields `M ≈ 32.7 GiB`.

```
 S = 8,575,610 → P = 53   w = 31   A = 53·(1 − e^−0.585) = 23.5   (sparse: P > w)
```

| | reads | writes | HDD | NVMe |
|---|---|---|---|---|
| naive gather | 1718 GiB | 1718 GiB | **19.7 days** | **80 min** |
| SPLAT, `M = 32.7 GiB` | 40329 GiB | 1718 GiB | **5.9 days** | **7.2 h** |
| SPLAT, `M = 230 GiB` | 13461 GiB | 1718 GiB | **23.7 h** | **2.6 h** |

Raising the budget from the default to 230 GiB drops `P` from 53 to 8,
which drops `A` from 23.5 to 7.8 — a 3× cut in total I/O from one
`--resources mem=230G`, and on the HDD a drop from 5.9 days to under a
day.

**On NVMe this case never pays, at any budget this node has.**
`penalty(4100 B)` is 3.6 at `iodepth=10`, so SPLAT would need `P ≤ 3`,
meaning `M > 1.85 TB / 3.6 ≈ 481 GiB` — more than the 655 GiB node has
left after page cache. The 4100-byte record is simply too close to an
efficient block for a fast NVMe drive: at that size it already reads at
490 MB/s against 1427 MB/s sequential.

On a *current* drive at realistic concurrency it does not merely fail to
pay at this node's budget — it cannot pay at any budget, because the
penalty falls under the two-pass floor. If the passage spine lands on
modern NVMe and the reader issues deep, SPLAT is the wrong tool for it.

Two ways out, both of which move the record size rather than the budget:
storing 1024-d as f16 halves `R` to 2052 B and roughly doubles the
penalty, and a passage spine built on smaller vectors (384-d f32 at
1540 B) moves it to 9.2. This is the practical form of the rule — when
ordering does not pay, the lever that matters is usually `R`, not `M`.

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
