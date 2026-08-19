# SPLAT Step 3 — Linearize

Sort the read plan by source index. This is the step that buys the
I/O win: after linearizing, the pass's reads ascend monotonically
through the source file, so kernel readahead streams pages in ahead
of the readers and no page is faulted twice.

```
 plan (output order)                 plan (source order)
 (902117, 0)                         (3,      1)
 (3,      1)        ── sort by ──►   (71442,  2)
 (71442,  2)        source index     (902117, 0)
     │                                   │
     ▼                                   ▼
 reads seek all over the file        reads sweep left → right
 ╲__╱▔╲___╱╲_                        ────────────────►
```

The permutation itself doesn't disappear — it moves into the second
tuple element, where it becomes RAM-side scatter positions for the
assemble step.

## The mvec sort: parallel bucket + sort + flatten

For hundreds of millions of entries per pass, a single-threaded
comparison sort is the bottleneck, so the mvec variant distributes
the work:

```
 read plan ──► par_chunks(64Ki) ──► 256 range buckets per thread
                                        │  bucket b = src / (count/256)
                                        ▼
               merge thread-local buckets (reserve, then extend)
                                        │
                                        ▼
               par: sort_unstable_by_key(src) within each bucket
                                        │
                                        ▼
               par: flatten via prefix-sum offsets ──► sorted plan
```

Bucketing by source range means the concatenation of sorted buckets
is globally sorted — no merge step. All phases run on the
governor-controlled rayon pool.

## The fvec fast path: detect already-sorted plans

Dedup-ordinal extractions produce maps that are already ascending. The
fvec variant checks `windows(2)` for sortedness first:

- **Sorted** — skip the sort *and* the scatter: reads and writes are
  both sequential, so records stream chunk-wise straight to the
  output without transposing through a segment buffer.
- **Shuffled** — `sort_unstable_by_key(src)` and proceed as usual.

## Slab

The slab variant sorts the plan by source ordinal the same way
(sequential slab reads), but variable-length records force assemble
to re-sort by *output* position later — see
[04-assemble.md](./04-assemble.md).

## Where in code

"Step 2: Sort by source position" in `sorted_index_extract_mvec`, the
`is_sorted` check in `sorted_index_extract_fvec`
(`veks-pipeline/src/pipeline/commands/gen_extract.rs`).

Next: [04-assemble.md](./04-assemble.md) — sequential gather, in-RAM
scatter.
