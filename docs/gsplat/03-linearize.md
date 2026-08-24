# gsplat Step 3 — Linearize

Sort the read plan by source ordinal. This is the step that buys the
I/O win: after linearizing, the pass's reads ascend monotonically
through the source, so the storage tier's prefetching works with the
access pattern instead of against it, and no fetched unit is faulted
twice within a pass.

It also completes the inversion [P](./02-plan.md) began. Plan reversed
each destination-ordered entry into `(source, local)`; sorting on the
first element leaves the pass holding this segment's slice of the
*inverse* permutation, in source order — the orientation assemble needs
to walk the source once, forwards.

```
 plan (output order)                 plan (source order)
 (902117, 0)                         (3,      1)
 (3,      1)        ── sort by ──►   (71442,  2)
 (71442,  2)        source ordinal   (902117, 0)
     │                                   │
     ▼                                   ▼
 reads seek all over the store       reads sweep left → right
 ╲__╱▔╲___╱╲_                        ────────────────►
```

The permutation itself does not disappear — it moves into the second
tuple element, where it becomes the memory-side scatter position for
[assemble](./04-assemble.md).

## Cost and parallelism

The sort is `Θ(S log S)` per pass and `Θ(N log(N/P))` overall: the only
super-linear term in the algorithm, and work the naive permutation
never does. It is a good trade whenever storage is the bottleneck and a
bad one when it isn't — a consideration that matters on fast flash
under a deep queue, where the I/O advantage narrows
([cost-model.md](./cost-model.md)).

At hundreds of millions of entries per pass a single-threaded
comparison sort becomes the bottleneck. Because the keys are ordinals
in a known range, a distribution sort parallelizes cleanly and avoids
a merge step entirely:

```
 read plan ──► chunk across workers ──► k range buckets per worker
                                            │  bucket b = src / (N_src / k)
                                            ▼
               merge worker-local buckets (reserve, then extend)
                                            │
                                            ▼
               sort within each bucket, in parallel
                                            │
                                            ▼
               concatenate buckets in order ──► sorted plan
```

Bucketing by source *range* is what makes the concatenation globally
sorted, so no merge is needed. A few hundred buckets is a reasonable
default: enough to spread the work, few enough that per-bucket
overheads stay negligible. Skewed maps produce uneven buckets; if the
map may be skewed, size buckets from a sample of the plan rather than
assuming uniformity.

## The already-sorted fast path

Some maps arrive ascending — a selection list in source order (dedup
survivors, a filter's output, a range slice) is a permutation only in
the trivial sense. Check for sortedness first; it is one linear scan
and it unlocks a much larger saving:

- **Sorted** — skip the sort *and* the scatter. Reads and writes are
  both sequential, so records can stream chunk-wise from source to sink
  without transposing through a segment buffer at all. The rewrite
  degenerates to a streaming filter, which is the cheapest shape
  available.
- **Unsorted** — sort and proceed.

Detecting this is worth it precisely because the fast path is not a
small constant-factor win: it removes a `Θ(N log(N/P))` sort, a full
buffer's worth of memory traffic, and the buffer itself.

## Variable-length records

Sort by source ordinal exactly as above — sequential reads are the
point, and they depend only on source order. Because such records have
no fixed stride, there is no transpose position to scatter into, so
assemble re-sorts by output position before writing; see
[04-assemble.md](./04-assemble.md). That second sort is memory-only and
does not disturb the storage-facing invariants.

Next: [04-assemble.md](./04-assemble.md) — sequential gather, in-memory
scatter.
