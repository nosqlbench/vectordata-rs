# gsplat Step 2 — Plan

For the active segment, collect **only** the map entries whose
*destination* falls inside that segment, reversing each into a
`(source_ordinal, segment_local_ordinal)` pair. That pair list is the
pass's **read plan**: everything this pass will read, and where each
record lands in the segment buffer.

## Segments partition the output, not the input

The single most important thing to hold onto: a segment is a
contiguous range of the **output** ordinal space. A pass owns
`output[segment_start .. segment_end)` and gathers the records that
belong there from *arbitrary* positions in the source.

```
 source ordinal space                 output ordinal space
 ┌──────────────────────────────┐     ┌──────────┬══════════┬──────────┐
 │  reads land wherever the map │     │ segment 0│ segment 1│ segment 2│
 │  points — all over the store │ ──► │          │ ← pass 1 │          │
 └──────────────────────────────┘     └──────────┴══════════┴──────────┘
   scattered, then linearized            one contiguous range per pass
```

Partitioning the *output* is what makes the write side contiguous and
leaves the read side merely unordered — and unordered reads can be
sorted ([L](./03-linearize.md)), while scattered writes cannot be
un-scattered. The alternative choice is the naive scatter:

| Partitioned on | Reads | Writes |
|----------------|-------|--------|
| **Output** (gsplat) | scattered across the source, then sorted into one ascending sweep per pass | one contiguous range per pass |
| Input | contiguous | scattered across the whole output, with read-modify-write on partial blocks |

gsplat partitions the side it can make contiguous and buys back the
other side with a memory buffer.

## Two orientations, and the one gsplat requires

A reorder relation can be stored either way round, and the difference
decides whether this step is a seek or a scan:

| Orientation | Layout | Reading it in order yields |
|-------------|--------|----------------------------|
| **Destination-ordered** (required) | position = output ordinal, value = source ordinal | `(to = i, from = map[i])` |
| Source-ordered | position = source ordinal, value = output ordinal | `(from = j, to = map[j])` |

gsplat requires the destination-ordered form; it is the same statement
as `output[i] = source[map[i]]`.

## Reversing the map

Read in output order, the map yields `(to = i, from = map[i])` — a
relation keyed by where records are going. Assemble needs the opposite
orientation, keyed by where they come from, so the plan flips every
entry it keeps:

```
 map entry (destination-keyed)          plan entry (source-keyed)
   i = 17,  map[17] = 902117    ──►      (902117,  17 − segment_start)
                                          └ from ┘  └───── to ──────┘
                                                     segment-local
```

Plan and linearize together are exactly *invert the permutation,
restricted to this segment*: the plan reverses each pair, the sort
orders them by source. No global inverse is ever materialized —
inversion happens per pass, inside a structure bounded by
`segment_size` entries, which is why an out-of-core permutation needs
no second index.

## The window shortcut — and what licenses it

In the general pattern, selecting "the entries destined for segment k"
is a filter over the whole map: `Θ(N)` per pass, `Θ(N·P)` overall. The
shortcut that removes it is **entirely a consequence of the
destination-ordered contract**. If the map's position axis *is* the
output ordinal space, the entries for output range `[segment_start,
segment_end)` are exactly the contiguous window at those positions, and
the read jumps straight to it:

```
 reorder map, in output order
 ┌──────────────┬=====================┬──────────────────┐
 │              │ window for segment k│                  │
 └──────────────┴=====================┴──────────────────┘
        [range_start + segment_start, range_start + segment_end)

   map[i]   = 902117  ─►  (source 902117, local 0)
   map[i+1] = 3       ─►  (source 3,      local 1)
   map[i+2] = 71442   ─►  (source 71442,  local 2)
                          └── read plan, in output order ──┘
```

`range_start` is the rewrite's own window into the map, for the case
where one map drives several disjoint outputs — splitting a shuffle
into a `[0,K)` slice and a `[K,N)` slice is two rewrites planning over
disjoint windows of one map.

Per pass this reads `segment_size` ordinals — the cheap side of the I/O
ledger, typically 4 or 8 bytes against records of `R` bytes. Over all
passes the map is read once in total, since the windows partition it.

## Computable maps

If the map is a function rather than a stored array — a seeded
permutation, a modular stride, a rank over a sort key — the plan step
evaluates it over the window instead of reading it. The contract is
unchanged (the function must be defined *from* output ordinal *to*
source ordinal), the window shortcut is automatic, and the map's I/O
cost disappears entirely.

## Orientation is a contract, not a check

Two consequences, one of them silent:

- **The shortcut is gone** if the map is source-ordered: entries for a
  given output segment are scattered through it, so planning reverts to
  a full filter per pass, `Θ(N·P)`.
- **Nothing detects it.** Plans bounds-check values against the source
  record count; orientation cannot be checked, because both
  orientations of a permutation are valid permutations. A
  source-ordered permutation runs clean and emits the *inverse* of the
  intended rewrite: wrong data, no error. (A source-ordered *selection*
  list, being shorter than the source, usually trips a bounds or length
  check instead and fails loudly.)

The fix belongs upstream of the rewrite, and it is cheap:

```
 for j in 0..N:  inv[map[j]] = j        # in-memory scatter over ordinals
```

Map entries are a handful of bytes against records of `R` bytes, so
inverting is memory-resident at scales where the data is nowhere near
it — a collection of 450 million records is a map of under two
gigabytes standing in front of terabytes of data. That asymmetry is why
gsplat insists on one orientation rather than handling both:
**inverting the map is cheap; inverting the data is the entire problem
this algorithm exists to solve.**

## Validation and memory

Bounds-check every source ordinal against `record_count()` as it enters
the plan, so an out-of-range entry fails before any output is written
for that pass.

Keep `segment_local_ordinal` segment-relative, not global: entries stay
small, and the same plan structure can be reused across passes
(allocate once, clear per pass). Plan cost is `sizeof(entry) ×
segment_size`, alongside — not inside — the segment buffer's budget.

Next: [03-linearize.md](./03-linearize.md) — sorting the plan into
source order.
