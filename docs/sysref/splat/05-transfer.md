# SPLAT Step 5 — Transfer

Write the assembled segment buffer to its slot in the output file as
one contiguous range, durably, then loop back to plan the next
segment.

```
 segment buffer (RAM)                output file (preallocated)
 ┌──────────────────────┐            ┌─────────┬─────────┬─────────┐
 │ assembled segment k  │ ─────────► │ seg 0   │▓seg k▓▓▓│  ...    │
 └──────────────────────┘  seek to   └─────────┴─────────┴─────────┘
                           part_start × record_bytes,
                           write in 8 MiB chunks, sync_data
                                        │
              ┌─────────────────────────┘
              ▼
        governor checkpoint ──► next pass (P–L–A–T)
        after the last pass: sync_all
```

One seek per pass, then pure streaming writes — the disk-facing
mirror image of the assemble step's RAM-side scatter.

## Durability and progress

- Writes go out in 8 MiB `write_all` chunks so the progress bar moves
  at disk speed rather than jumping at the end.
- `sync_data` per pass bounds the dirty-page debt: without it, tens
  of GiB of writeback can land in the *next* pass's read phase and
  masquerade as slow reads. The progress bar reserves its final 5%
  for the sync so the flush is visible, not a silent stall.
- A final `sync_all` after the last pass covers file metadata.
- `ctx.governor.checkpoint()` closes each pass, letting the resource
  governor observe utilization between passes
  ([08-architecture.md §8.3](../08-architecture.md)).

## fvec: compaction accounting

When assemble skipped near-zero vectors, the segment is shorter than
planned. The fvec variant tracks `total_written` across passes and
seeks to `total_written × record_bytes` instead of the nominal
segment offset, so segments pack tightly; after the last pass the
file is truncated to the true length with `set_len`, and the skipped
source ordinals are written alongside as an ivec. Downstream step
variables (`extract_output_count` and friends) report the compacted
count.

## Slab: sequential writer with per-segment resume

Variable-length records go through `SlabWriter::add_record` in output
order (already sorted by assemble) — the slab format is append-only,
so "seek to segment offset" is replaced by strict segment ordering.

Each completed segment is also persisted as a cache file
(`slab-extract.part_NNNN.cache`) under the step cache, keyed by a
`meta.json` recording the exact parameters (source, map, range,
segment count, page size). A re-run with matching parameters replays
finished segments from cache and recomputes only the remainder; any
parameter mismatch invalidates the whole cache set. This makes the
most expensive slab rewrites resumable at segment granularity.

## Where in code

"Step 4: Write partition" in `sorted_index_extract_mvec`, the
`total_written` bookkeeping and final truncate in
`sorted_index_extract_fvec`, and the cache replay/persist blocks in
`sorted_index_extract_slab`
(`veks-pipeline/src/pipeline/commands/gen_extract.rs`).

Back to the overview: [README.md](./README.md).
