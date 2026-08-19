# SPLAT Step 4 — Assemble

Walk the linearized plan, reading source records in ascending file
order and scattering each into the segment buffer at its final output
position (`local_out_pos × record_bytes`). This is the transpose at
the heart of the rewrite — and the scatter-write the name winks at.

```
 source file (read ascending)          segment buffer (RAM)
 ┌────┬────┬────┬────┬────┬────┐      ┌──────┬──────┬──────┐
 │ .. │ 3  │ .. │71442 .. 902117      │out 0 │out 1 │out 2 │
 └────┴─┬──┴────┴──┬─┴────┴─┬──┘      └──────┴──────┴──────┘
        │          │        │             ▲      ▲      ▲
        └──────────┼────────┼─────────────┘      │      │
                   └────────┼────────────────────┘      │
                            └───────────────────────────┘
        sequential gather            random scatter — in RAM,
                                     where seeks are free
```

## Lock-free parallelism

Each output position occurs exactly once in the plan, so concurrent
records land in disjoint buffer regions. The rayon workers share the
buffer through `SharedBuf` (unsafe disjoint-slice access) with no
locking; a relaxed atomic counter drives the progress bar. Reads are
issued in plan order across the pool, so the aggregate access pattern
stays ascending even though individual threads interleave.

`advise_sequential()` on the source reader tells the kernel what the
sorted plan guarantees.

## Per-record work

Everything that must touch each record rides along in this step,
while the bytes are already in cache:

| Variant | Per-record work |
|---------|-----------------|
| mvec | Prepend 4-byte dim header; copy f16 payload; optional in-buffer L2 normalize after the scatter completes |
| fvec | `pread` the record (see below); near-zero norm check against `zero_threshold`; optional L2 normalize (skipped when a 100k-vector sample shows the source already normalized, per the [§9.1 threshold](../09-algorithms.md)); source/output norm statistics accumulation |
| slab | Copy the variable-length record; collect `(local_out_pos, bytes)` pairs |

**Why fvec uses `pread` instead of mmap slices:** faulting shuffled
records through a mapping pulls every touched page into RSS — 
multi-100-GiB apparent footprints on multi-TB sources. `pread` moves
the same bytes through the same page cache without mapping them into
the process, so RSS stays bounded. A single `File` is `Sync` for
`read_at`, so the workers share one descriptor.

**fvec near-zero skip:** records failing the norm threshold are
omitted and their source ordinals recorded; each worker writes into
its own scratch buffer, and buffers are compacted in output order so
the gaps close. The output therefore may be shorter than the plan —
transfer reconciles this ([05-transfer.md](./05-transfer.md)).

**Slab exception:** with no fixed stride there is no transpose
position to scatter into, so assemble collects pairs and sorts them
by `local_out_pos` before writing. The re-sort is RAM-only and
preserves the disk-facing invariants.

## Buffer hygiene

The segment buffer is re-zeroed between passes in 256 MiB chunks with
a progress bar — `vec![0u8; N]` at tens of GiB blocks silently for
tens of seconds in kernel page-zeroing, which reads as a hang.

## Where in code

"Step 3: Read source data in parallel" in `sorted_index_extract_mvec`,
the `transpose_fn`/chunked extract closures in
`sorted_index_extract_fvec`, and the pair-collection loop in
`sorted_index_extract_slab`
(`veks-pipeline/src/pipeline/commands/gen_extract.rs`).

Next: [05-transfer.md](./05-transfer.md) — flushing the segment.
