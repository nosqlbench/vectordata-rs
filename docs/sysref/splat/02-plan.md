# SPLAT Step 2 — Plan

For the active segment, collect every reorder-map entry whose output
position falls inside the segment, producing the pass's **read plan**:
a list of `(source_idx, local_out_pos)` pairs.

## The window shortcut

In the general pattern, planning is a filter over the whole map. Here
the map's entry order *is* the output order — `out[i] = src[map[i]]`
— so the entries for output range `[part_start, part_end)` are
exactly the contiguous map window at those positions. The scan jumps
straight to the window instead of filtering all N entries:

```
 reorder map (ivec)
 ┌──────────────┬=====================┬──────────────────┐
 │              │ window for segment k│                  │
 └──────────────┴=====================┴──────────────────┘
        [range_start + part_start, range_start + part_end)

   map[i] = 902117  ─►  (source 902117, local_out 0)
   map[i+1] = 3     ─►  (source 3,      local_out 1)
   map[i+2] = 71442 ─►  (source 71442,  local_out 2)
                        └── read plan, in output order ──┘
```

`range_start` is the extraction's window into the map itself — e.g.
splitting a shuffle into `[0,K)` queries and `[K,N)` base vectors
means two extractions planning over disjoint map windows.

Per pass this reads `segment_size` 4-byte integers — the cheap side
of the I/O ledger. Over all passes the map is read once in total,
since the windows partition it.

## Validation

Every source index is bounds-checked against the source record count
as it enters the plan; an out-of-range entry fails the step
immediately, before any output is written for that pass.

## Memory

`local_out_pos` is segment-relative (`0..part_len`), so plan entries
stay small and the same plan vector is reused across passes
(`Vec::with_capacity` once, `clear()` per pass). Plan cost is
`16 bytes × segment_size`, alongside — not inside — the segment
buffer's budget.

## Where in code

"Step 1: Scan ivec for this partition's entries" in
`sorted_index_extract_{fvec,mvec,slab}`
(`veks-pipeline/src/pipeline/commands/gen_extract.rs`).

Next: [03-linearize.md](./03-linearize.md) — sorting the plan into
source order.
