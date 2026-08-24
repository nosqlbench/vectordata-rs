# gsplat Step 5 — Transfer

Write the assembled segment buffer to its slot in the output as one
contiguous range, make it durable, then loop back to plan the next
segment.

```
 segment buffer (memory)             output (preallocated)
 ┌──────────────────────┐            ┌─────────┬─────────┬─────────┐
 │ assembled segment k  │ ─────────► │ seg 0   │▓seg k▓▓▓│  ...    │
 └──────────────────────┘  position  └─────────┴─────────┴─────────┘
                           segment_start × record_bytes,
                           written in chunks, then a durability barrier
                                        │
              ┌─────────────────────────┘
              ▼
        next pass (P–L–A–T) ──► after the last: finalize
```

One positioning per pass, then pure streaming — the storage-facing
mirror image of assemble's memory-side scatter.

## Chunking and durability

- **Write in chunks** of a few megabytes rather than one enormous
  call, so progress reporting moves at storage speed instead of
  jumping at the end, and so a partial failure is localized.
- **Barrier per pass.** Without one, the dirty data from this pass
  lands in the *next* pass's read phase and masquerades as slow reads.
  A per-pass barrier bounds the debt and makes pass timings mean
  something. Reserve the tail of the pass's progress budget for it —
  the flush is real work, and hiding it produces a silent stall right
  where users are watching.
- **Finalize once at the end** for whatever the tier keeps outside the
  data path: file metadata, a manifest, a multipart completion.
- **Yield between passes.** Pass boundaries are the natural place to
  re-negotiate the memory budget, re-check admission control, honor
  cancellation, or emit a checkpoint.

## Compaction accounting

When [assemble](./04-assemble.md) dropped records, the segment is
shorter than planned and the nominal segment position is wrong. Track a
running output total across passes and position each write at
`total_written × record_bytes` instead, so segments pack tightly; after
the last pass, truncate the output to the true length. Report the
compacted count as the output's record count — anything downstream that
assumed the planned count will be wrong by exactly the number dropped.

## Append-only sinks

A sink that cannot position — object storage parts, log-structured
stores, pipes, or any variable-length record format — still satisfies
the contract as long as passes execute in output order and each segment
is written completely before the next begins. "Seek to the segment
position" becomes "append this segment now", which is why
variable-length rewrites require strict pass ordering while
fixed-stride ones do not.

## Checkpoint and resume

Passes are natural resume points: each one is an independent,
deterministic unit of work whose output occupies a known range. Persist
each completed segment (or a marker for it) keyed by a **fingerprint of
everything that determines the output**:

```
 fingerprint = hash(source identity, map identity, output range,
                    segment count, per-record hook configuration)
```

A re-run with a matching fingerprint replays completed segments and
recomputes only the remainder; any mismatch invalidates the whole set
rather than part of it. A cache keyed too loosely is worse than no
cache — it produces output that is silently a mixture of two
configurations.

This is what makes an expensive rewrite interruptible at segment
granularity, which matters most where the compute is rented and
preemptible.

Back to the overview: [README.md](./README.md).
