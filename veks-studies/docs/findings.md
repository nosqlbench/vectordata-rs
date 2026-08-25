# What the studies found

An explainer for the results `veks-study study all` produces, and what
each of them is evidence for.

Every number here came out of the simulator or the analytic cost model in
this crate; none of it is quoted from elsewhere. The command that
reproduces each table is given beside it, and the test that keeps it
honest is named. Device parameters are grounded in the corpus listed in
[the crate README](../README.md#sources); the external-memory bound the
staged form realizes is Aggarwal & Vitter's `Θ((N/B)·log_(M/B)(N/B))`
(*Commun. ACM* 31(9), 1988).

---

## The problem, and the four strategies

A **terabyte-scale permutation**: a large body of records carrying a
monotonic ordinal, a closed transform map over those ordinals, memory
enough for only a fraction of the payload, and a storage tier on which
random access is ruinous. At 10⁸–10⁹ ordinals the binding constraints are
not the ones a small benchmark surfaces.

Four strategies are priced, not two:

| Strategy | Reads | Writes | Cost in data-equivalents |
|---|---|---|---|
| naive gather | random, one command per record | coalesced | `N` commands, `⌈R/B⌉·B` bytes each |
| naive scatter | streamed | random; a partial block costs a read-modify-write | `N` commands, bytes twice |
| ordered rescan | ascending, but the source is swept once per segment | coalesced | `A(P) + 1`, where `A(P) = P·(1−e^{−w/P})` |
| **gsplat staged** | ascending, **once** | coalesced, via a spill extent | `2·(stages + 1)`, `stages = ⌈log_f(segments)⌉` |

The distinction between the last two is what the studies keep returning
to. **Ordering the reads is not what scales — staging is.** A re-scan's
cost grows with `payload/M`; a staged rewrite's grows with
`log_f(payload/M)`, where `f = M/W` is the fan-out.

---

## 1. The corner cases

`veks-study study corners`

Fourteen configurations at the edges of the parameter space, each a
one-factor change from a fixed baseline: 1 TiB, 1 KiB records, 32 GiB of
memory, 128 KiB containers, 2016 consumer NVMe, depth 32.

### Absolute

```
corner                  gather   scatter    rescan    staged     gain  winner
baseline                  2.4h      4.8h      5.7h     41.9m     3.5x  gsplat staged
fits in memory            2.4h      4.8h     31.4m     20.9m     6.9x  gsplat staged
memory starved (1 GiB)    2.4h      4.8h     21.2h     41.9m     3.5x  gsplat staged
sub-block records (128B) 19.3h      1.6d      5.8h     41.9m    27.6x  gsplat staged
record = block (4 KiB)   37.2m     37.2m      3.7h     41.9m     0.9x  naive gather
huge records (64 KiB)    20.9m     20.9m     30.8m     41.9m     0.5x  naive gather
seek-bound device        46.8d     93.3d      2.1d      6.3h   179.4x  gsplat staged
fastest device           15.0m     29.9m      1.4h     10.5m     1.4x  gsplat staged
container = block         3.0h      5.4h      2.9h      2.4h     1.2x  gsplat staged
fan-out squeezed          2.4h      4.8h      7.5d      2.1h     1.1x  gsplat staged
depth 1                   3.4h      6.7h      6.3h     46.5m     4.4x  gsplat staged
depth 4096                2.4h      4.8h      5.7h     41.9m     3.5x  gsplat staged
worst case              372.2d    744.0d     42.3d      6.3h  1425.8x  gsplat staged
best case for naive       5.2m      5.2m      5.9m     10.5m     0.5x  naive gather
```

### Differential

`d naive` and `d staged` are multiplicative against the baseline; `/3.9`
means "3.9× cheaper".

```
corner                segments  stages   d naive  d staged     gain   d gain  naive pegs
baseline                    32       1         —         —     3.5x        —  controller
fits in memory               1       0         —      /2.0     6.9x     2.0x  controller
memory starved            1.0k       1         —         —     3.5x        —  controller
sub-block records           32       1      7.9x         —    27.6x     7.9x  controller
record = block              32       1      /3.9         —     0.9x     /3.9  controller
huge records                32       1      /6.9         —     0.5x     /6.9  bandwidth
seek-bound device           32       1    463.8x      9.0x   179.4x    51.7x  media
fastest device              32       1      /9.7      /4.0     1.4x     /2.4  controller
container = block           32       1      1.2x      3.4x     1.2x     /2.8  controller
fan-out squeezed          1.0k       5         —      3.0x     1.1x     /3.0  controller
depth 1                     32       1      1.4x      1.1x     4.4x     1.3x  controller
depth 4096                  32       1         —         —     3.5x        —  controller
worst case                1.0k       1   3684.9x      9.0x  1425.8x   410.6x  media
best case for naive          2       1     /27.8      /4.0     0.5x     /6.9  bandwidth
```

### What the differential says

**The `d staged` column is mostly empty, and that is the finding.**
Memory starved (32× less RAM), sub-block records (8× more ordinals),
record = block, huge records — every one of them leaves the staged cost
*unchanged*. Only two factors move it: a faster device (`/4.0`, because
it is bandwidth-bound and a faster link is exactly what helps), and a
collapsed fan-out.

Sorting the factors by what they do:

| Group | Factors | Why |
|---|---|---|
| Make staging matter **more** | smaller records, seek-bound device, less memory | all raise the naive cost without touching the staged one |
| Make it matter **less** | record fills a block, high device command rate, memory holds the payload | two of the three describe a job that was never in trouble |
| Move **both together** | issue depth, container size below the fan-out step | change absolute times, leave the decision alone |

**Adverse factors compound, because they act on different terms.**
`worst case` is 128 B records *and* a 1 GiB budget *and* a spinning disk:
372 days naive against 6.3 hours staged, a factor of **1426**. The naive
cost is 3685× the baseline while the staged cost is only 9× — and that 9×
is the device change alone, since neither record size nor memory reaches
the staged side at all.

**Two corners are won by a naive gather.** `record = block` and
`huge records` are bandwidth-pegged with zero read waste, and a gather
that wastes nothing has nothing left to save. `best case for naive` wins
by 2×. This is asserted, not incidental
(`manifold::tests::the_corner_set_contains_configurations_that_naive_wins`):
a corner set every configuration of which favours one strategy is an
advertisement, not a study.

---

## 2. Where the boundary is

`veks-study study frontier`

The largest record size at which the staged rewrite still beats the best
naive strategy on a 1 TiB payload:

```
device             budget   boundary R     naive    staged  naive pegs at the boundary
spinning-sata        8.0G      65536 B      1.1d      6.3h  media
spinning-sata       32.0G      65536 B      1.1d      6.3h  media
spinning-sata      128.0G      65536 B      1.1d      6.3h  media
sata-ssd             8.0G       1024 B      3.8h      2.2h  controller
sata-ssd            32.0G       1024 B      3.8h      2.2h  controller
sata-ssd           128.0G       1024 B      3.8h      2.2h  controller
nvme-consumer        8.0G       2048 B      1.2h     41.9m  controller
nvme-consumer       32.0G       2048 B      1.2h     41.9m  controller
nvme-consumer      128.0G       2048 B      1.2h     41.9m  controller
nvme-modern          8.0G       1024 B     15.0m     10.5m  controller
nvme-modern         32.0G       1024 B     15.0m     10.5m  controller
nvme-modern        128.0G       1024 B     15.0m     10.5m  controller
```

**The budget column does nothing.** A 16× change in memory does not move
the boundary on any device
(`manifold::tests::the_frontier_moves_with_the_device_and_not_with_the_budget`).
What decides whether you need a staged rewrite is the ratio of record
size to block size and the device's command rate. Memory decides how many
*stages* it takes, not whether it is worth doing.

The mechanism is visible in the record-size walk
(`veks-study study record`), where the naive gather's binding resource
changes down the page:

```
 record  ordinals      w   waste    gather    staged     gain  gather pegs
    128      8.6B   1024   32.0x     19.3h     41.9m    27.6x  controller
    256      4.3B    512   16.0x      9.6h     41.9m    13.8x  controller
    512      2.1B    256    8.0x      4.8h     41.9m     6.9x  controller
   1024      1.1B    128    4.0x      2.4h     41.9m     3.5x  controller
   4096    268.4M     32    1.0x     37.2m     41.9m     0.9x  controller
  16384     67.1M      8    1.0x     20.9m     41.9m     0.5x  bandwidth
  65536     16.8M      2    1.0x     20.9m     41.9m     0.5x  bandwidth
```

Shrinking the record does two adverse things at once: it multiplies the
ordinal count, and below the block size it makes every random read mostly
padding. The `gather pegs` column crossing from bandwidth to controller
is the same boundary the `gain` column crosses, seen from the resource
side.

---

## 3. Memory, and why more of it buys so little

`veks-study study memory`

1 TiB of 1 KiB records on a consumer NVMe:

```
   budget  segments     A(P)  fan-out    stages    rescan    staged     gain
     1.0G      1.0k    120.3     8.2k         1     21.2h     41.9m     3.5x
     4.0G       256    100.7    32.8k         1     17.8h     41.9m     3.5x
    16.0G        64     55.3   131.1k         1      9.8h     41.9m     3.5x
    64.0G        16     16.0   524.3k         1      3.0h     41.9m     3.5x
   256.0G         4      4.0     2.1M         1     52.4m     41.9m     3.5x
   512.0G         2      2.0     4.2M         1     31.4m     41.9m     3.5x
```

The re-scan column moves by a factor of **41** across this range. The
staged column does not move at all. **Staging converts memory from a
throughput parameter into a correctness-of-scale parameter**: you need
enough to hold one bucket buffer per segment, and past that more memory
buys nothing.

That is why the memory-starved corner above shows `d staged: —`. A
terabyte rewrite with 1 GiB of RAM costs the same 41.9 minutes as one
with 512 GiB.

### The crossover with the re-scan form

Measured at **exactly four segments**, and in the same place on every
device modelled — as it should be, since both arms are sequential there
and the comparison is between two byte counts rather than two access
patterns (`scale::crossover::staging_overtakes_the_rescan_at_about_four_segments`,
`…::the_crossover_is_the_same_on_every_device`).

The arithmetic is one line: the staged form moves the payload four times
whatever `P` is, the re-scan moves it `A(P) + 1` times, so they are level
at `A(P) = 3`.

| segments `P` | `A(P)` | ordered rescan | gsplat staged |
|---|---|---|---|
| 2 | 2.0 | 3 data-equivalents | 4 |
| 3 | 3.0 | 4 | 4 |
| 4 | 4.0 | 5 | 4 |
| 16 | 16.0 | 17 | 4 |
| 1024 | 120.3 | 121 | 4 |

**Below four segments the re-scan is genuinely better** — it never
touches scratch. A claim that staging always wins would be false, and the
cost model does not make it.

### The one way to make staging expensive

`veks-study study fanout` — 1 TiB of 512 B records, 4 GiB budget:

```
container      w   segments   fan-out   stages       read     staged
      16K     32        256    262.1k        1      2.00T      41.9m
      64K    128        256     65.5k        1      2.00T      41.9m
     128K    256        256     32.8k        1      2.00T      41.9m
       1M   2048        256      4.1k        1      2.00T      41.9m
      16M  32768        256       256        1      2.00T      41.9m
     256M 524288        256        16        2      3.00T       1.0h
```

Stages are `⌈log_f(segments)⌉` with `f = M/W`, so the cost of a bigger
container is a **step function, not a slope**. Everything left of the
step is free; the first configuration past it pays a full extra read and
write of the payload.

This is the only modelled factor that multiplies the staged cost
(`manifold::tests::only_a_collapsed_fan_out_makes_the_staged_form_expensive`),
and it is entirely a configuration choice. With a 32 GiB budget and
128 KiB containers, `f` is a quarter of a million: **a terabyte and a
petabyte both need exactly one distribution stage.**

---

## 4. What is saturated, and the bounds that follow

`veks-study study bounds` · `veks-study study pegged`

Service demand per record, 1 TiB of 512 B records, 32 GiB, 8 cores,
consumer NVMe:

```
naive gather     D_total 11.186 µs   D_max  8.096 µs (controller)   n* 1.4   4.83 h
naive scatter    D_total 22.045 µs   D_max 16.161 µs (controller)   n* 1.4   9.64 h
ordered rescan   D_total 10.718 µs   D_max  9.652 µs (bandwidth)    n* 1.1   5.76 h
gsplat staged    D_total  1.300 µs   D_max  1.170 µs (bandwidth)    n* 1.1   0.70 h
```

Utilizations at those rates:

| Strategy | controller | bandwidth | host cpu | media |
|---|---|---|---|---|
| naive gather | **100%** | 33% | 3% | 3% |
| naive scatter | **100%** | 31% | 3% | 3% |
| ordered rescan | 11% | **100%** | 0% | 0% |
| gsplat staged | 11% | **100%** | 0% | 0% |

**The trade, in one line: staging cuts controller utilization from 100%
to 11% and pushes bandwidth to 100%.** `D_total` falls 8.6× because the
resource it moved the load onto had headroom. It wins exactly when the
controller was the bottleneck and bandwidth could absorb the transfer —
and it loses when bandwidth was already the bottleneck, which is the same
crossover the cost model reaches from the other direction.

These are Denning & Buzen's operational bounds and need no distributional
assumption at all:

```
D_max   = max_k D_k                          the bottleneck demand
D_total = Σ_k D_k                             total demand per record

X(n)   ≤ min( n / D_total , 1 / D_max )       throughput
R(n)   ≥ max( D_total , n · D_max )           residence
U_k    = X · D_k                              utilization
n*     = D_total / D_max                      the knee
```

**More than one resource can be pegged**, and reporting only the larger
is what hides the interesting case: when two are at 1.0 there is no
headroom anywhere and only a change of strategy will help. The studies
therefore report every resource within 5% of saturation
(`queueing::tests::more_than_one_resource_can_be_saturated`).

### Issue depth is the wrong knob

`veks-study study depth` — 1 TiB of 512 B records, modern NVMe:

```
strategy              n*        n=1        n=8       n=n*     n=1024
naive gather         2.1       1.1h      29.9m      29.9m      29.9m
naive scatter        2.1       2.1h      59.8m      59.8m      59.8m
ordered rescan       1.1       1.5h       1.4h       1.4h       1.4h
gsplat staged        1.1      11.1m      10.5m      10.5m      10.5m
```

`n*` is between 1.1 and 2.1 everywhere, so anything past a handful of
outstanding requests buys nothing — the `depth 4096` corner is identical
to the baseline. The system is **capacity-limited, not
concurrency-limited**, and tuning queue depth is tuning the wrong thing.

---

## 5. What the kernel does to the rewrite

Three parts of the operating system move the answer by more than the
algorithm often does.

### Readahead is a tax on scattered access, not a neutral

`veks-study study readahead`

```
strategy             RA on     RA off      tax     bytes on    bytes off
naive gather          2.5s       1.7s    1.46x         1.1G         901M
ordered rescan        3.1s       3.7s    0.84x         711M         610M
gsplat staged         2.1s       2.3s    0.92x         494M         493M
```

A miss the kernel does not recognize as sequential starts a new region
seeded at `get_init_ra_size`: the request size is rounded to a power of
two and multiplied, so a **4 KiB fault fetches 16 KiB** against a 128 KiB
ceiling (`linux/mm/readahead.c`). Three of those four pages are
speculative and a scattered stream uses none of them.

So the same kernel feature is a **subsidy for one access pattern and a
tax on the other** — 1.46× against the gather, 0.84× and 0.92× in favour
of the ordered forms. That is a second, independent reason ordering is
worth arranging, and it is why `POSIX_FADV_RANDOM` (which sets
`ra_pages` to zero) is the documented remedy for a workload that cannot
be ordered.

### The block scheduler taxes commands you issue

`veks-study study scheduler`

At saturation — 4 KiB random reads, depth 512, 16 cores:

```
scheduler          achieved    published      error  lock busy  what bounds it
none                  1189k         786k        n/a         0%  the device
mq-deadline            567k         569k      -0.4%       100%  the scheduler's lock
kyber                 1189k         786k        n/a         0%  the device
bfq                    315k         315k      -0.2%       100%  the scheduler's lock
```

The published figures are Ren, Doekemeijer, Tehrany & Trivedi
(ICPE '24), who attribute the shortfall to lock contention rather than
policy — up to 78.0% of cycles for `bfq`. Modelling it as a *serialized*
per-dispatch cost reproduces `mq-deadline` within **0.4%** and `bfq`
within **0.2%**, with the lock at 100% busy in both cases, and the
ceiling does not move when cores are added
(`io::kernel::the_scheduler_ceiling_does_not_move_with_core_count`).
The `none`/`kyber` rows are device-bound; this crate's modern-NVMe model
is a faster drive than the paper's, so exceeding its figure there says
nothing either way.

Against the rewrites themselves, at the command rates they actually
offer (41k–214k IOPS), **nothing moves**. A scheduler can only tax
commands you issue: a staged rewrite issues few large ones and is
therefore immune to a configuration choice nobody remembers to make.

### Scattered reads cost twice — once on the device, once in the cache

`veks-study study writeback`

```
strategy         buffered    direct  throttled peak dirty  flusher wb   evict wb
naive gather         1.0s      0.9s       0.0s       736K         178       7.3k
ordered rescan       2.1s      0.6s       1.6s         4M        7.5k          0
gsplat staged        1.8s      0.3s       1.6s         4M       15.0k        141
```

All three write their *output* in order, so none of them is a scattered
writer. Yet the gather's pages go out **41× more often by eviction than
by the flusher**, and the staged rewrite's go out **106× more often by
the flusher than by eviction**.

What puts the gather's pages on the eviction path is its **reads**: they
land all over the source, each claiming a frame, and the frames they
claim are the ones holding output pages the flusher has not written yet.
A dirty page whose frame is needed must be written before the frame can
be reused — on the allocation path, in LRU order, one page at a time.

Adding memory fixes this only at roughly **half the payload**
(`manifold::tests::more_memory_helps_only_once_the_cache_approaches_the_payload`)
— and a cache that size means the rewrite fitted in memory and there was
never a problem to solve.

The pacing itself is Linux's: flusher at `dirty_background_ratio` 10%,
writer put to sleep in `balance_dirty_pages` at `dirty_ratio` 20%, pages
expire at 30 s, timer at 5 s, cubic position ratio, 200 ms maximum pause,
IO-less (the writer sleeps rather than submitting writeback itself — Wu
Fengguang, [LWN 456904](https://lwn.net/Articles/456904/)). A buffered
rewrite is not finished when its last write returns, so runs here drain
to durability before the clock stops.

---

## 6. Scale, and what actually changes with it

`veks-study study scale` — 1 KiB records, 32 GiB budget, spinning disk:

| N | naive gather | naive scatter | ordered rescan | gsplat staged |
|---|---|---|---|---|
| 1.0 M | 1.0 h | 2.1 h | 15.8 s | 10.5 s |
| 10.0 M | 10.5 h | 20.9 h | 2.6 m | 1.8 m |
| 100.0 M | 4.4 d | 8.7 d | 35.0 m | 35.0 m |
| 1.0 B | 43.6 d | 86.9 d | 1.9 d | **5.8 h** |

Both costs are linear in `N` — the naive one at a positioning time per
record, the staged one at a streaming time per byte — so **the ratio
barely moves**. What moves is the absolute time: the same advantage that
is academic at a million ordinals is the difference between a shift and a
quarter at a billion
(`manifold::tests::the_advantage_is_flat_in_the_ordinal_count_but_the_absolute_time_is_not`).

The staged column steps exactly once, where the payload outgrows memory
and buys a second pass — `⌈log_f(segments)⌉` showing through
(`…::the_staged_cost_steps_with_the_stage_count`).

---

## Reproducing all of it

```
veks-study study all          # every table above
veks-study study <name>       # one of: scale memory record strategies pegged
                              #         bounds depth fanout readahead scheduler
                              #         writeback corners frontier
veks-study validate           # the accuracy scorecard the device models rest on
cargo test -p veks-studies --release
```

The analytic pricing used above a traceable scale is cross-checked
against the discrete-event simulator wherever both can run
(`scale::tests::the_staged_arm_agrees_with_its_implementation_at_traceable_scale`),
and the staged implementation is held to its stated invariants —
single read, single write, monotone access, bounded memory, balanced
spill — at every budget
(`scale::tests::the_staged_implementation_keeps_its_invariants`).

Related reading: [the cost model](gsplat/cost-model.md) for the
derivations, [the gsplat overview](gsplat/README.md) for the algorithm,
and [the crate README](../README.md) for what the device models are
validated against and where they are known to diverge.
