# SPLAT Step 2 — Plan

For the active segment, collect **only** the reorder-map entries whose
*destination* falls inside that segment, reversing each one into a
`(source_idx, local_out_pos)` pair. That pair list is the pass's **read
plan**: everything this pass will read, and where each record lands in
the segment buffer.

## Segments partition the output, not the input

The single most important thing to hold onto: a segment is a
contiguous range of the **output** ordinal space. A pass owns
`output[part_start .. part_end)` and gathers the records that belong
there from *arbitrary* positions all over the source.

```
 source ordinal space                 output ordinal space
 ┌──────────────────────────────┐     ┌──────────┬══════════┬──────────┐
 │  reads land wherever the map │     │ segment 0│ segment 1│ segment 2│
 │  points — all over the file  │ ──► │          │ ← pass 1 │          │
 └──────────────────────────────┘     └──────────┴══════════┴──────────┘
   scattered, then linearized            one contiguous range per pass
```

Partitioning the *output* is what makes the write side contiguous and
the read side merely unordered — and unordered reads can be sorted
(step [L](./03-linearize.md)), while scattered writes cannot be
un-scattered. The alternative choice is the naive scatter:

| Partitioned on | Reads | Writes |
|----------------|-------|--------|
| **Output** (SPLAT) | scattered across the source, then sorted into one ascending sweep per pass | one contiguous range per pass |
| Input (naive scatter) | contiguous | scattered across the whole output, with read-modify-write on partial blocks |

SPLAT partitions the side it can make contiguous, and buys back the
other side with a RAM buffer. The code calls segments *partitions*
(`num_partitions`, `partition_size`, `part_start`); the terms are
interchangeable, and both always refer to output ranges.

## Two orientations, and the one SPLAT requires

A reorder relation can be stored either way round, and the difference
decides whether this step is a seek or a scan:

| Orientation | Layout | Reading it in order yields |
|-------------|--------|----------------------------|
| **Destination-ordered** (required) | position = output ordinal, value = source ordinal | `(to = i, from = map[i])` |
| Source-ordered | position = source ordinal, value = output ordinal | `(from = j, to = map[j])` |

SPLAT requires the destination-ordered form — it is the map format
contract, the same statement as `out[i] = src[map[i]]`. Everything
downstream in this step follows from it.

## Reversing the map

Read in output order the map yields `(to = i, from = map[i])` — a
relation keyed by where records are going.

Assemble needs the opposite orientation, keyed by where records come
from, so the plan flips every entry it keeps:

```
 stored map entry (destination-keyed)      plan entry (source-keyed)
   i = 17,  map[17] = 902117        ──►     (902117,  17 − part_start)
                                             └ from ┘  └─── to ────┘
                                                       segment-local
```

Plan and linearize together are exactly *invert the permutation,
restricted to this segment*: the plan reverses each pair, the sort
orders them by source. No global inverse map is ever materialized —
inversion happens per pass, inside a `Vec` bounded by `segment_size`
entries, which is why an out-of-core permutation needs no second index
file.

## The window shortcut — and what licenses it

In the general pattern, selecting "the entries destined for segment k"
is a filter over the whole map — `Θ(N)` per pass, `Θ(N·P)` overall.
The shortcut that removes it is **entirely a consequence of the
destination-ordered contract**: if the map's position axis *is* the
output ordinal space, then the entries for output range
`[part_start, part_end)` are exactly the contiguous map window at those
positions, and the scan can jump straight to it:

```
 reorder map (ivec), in output order
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

## If the map is source-ordered

Two consequences, one of them silent:

- **The shortcut is gone.** Entries destined for a given output segment
  are scattered through the map, so planning reverts to a full filter
  per pass — `Θ(N·P)` map reads instead of `Θ(N)` total. At 450M
  entries over 53 passes that is 95 GB of map scanning.
- **Nothing detects it.** Plan bounds-checks values against the source
  record count; it does not check orientation, and cannot — both
  orientations of a permutation are valid permutations. Hand SPLAT a
  source-ordered permutation and it runs clean and emits the *inverse*
  of the intended rewrite: wrong data, no error. (A source-ordered
  *selection* list, being shorter than the source, usually trips the
  bounds or length checks instead and fails loudly.)

The fix belongs upstream of the extract, and it is cheap:

```
 for j in 0..N:  inv[map[j]] = j        # RAM-side scatter, 4-byte entries
```

Map entries are 4 bytes against records of `R` bytes, so inverting is
RAM-resident even at spine scale — 450M ordinals is a 1.8 GB map
standing in front of a 1.8 TB dataset. That thousand-to-one asymmetry
is exactly why SPLAT insists on one orientation rather than handling
both: **inverting the map is cheap; inverting the data is the entire
problem this algorithm exists to solve.**

## Validation

Every source index is bounds-checked against the source record count
as it enters the plan; an out-of-range entry fails the step
immediately, before any output is written for that pass. Orientation,
as above, is a contract rather than a check.

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
