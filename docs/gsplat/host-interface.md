# gsplat — Host Interface

What a runtime must supply to run gsplat, stated as primitives rather
than APIs. Six are required; three are optional and buy throughput,
observability, or resumability.

## Required primitives

```
 map_window(a, b)        -> sequence of source ordinals for output positions [a, b)
 record_count()          -> N_src
 read_record(ordinal)    -> bytes                     cost bounded by its container
 write_range(pos, bytes) -> ()                        contiguous, at a computed position
 memory_budget()         -> bytes                     what one segment may occupy
 spawn(tasks)            -> ()                        run tasks, join, propagate failure
```

| Primitive | Used by | Requirement | If the host cannot |
|-----------|---------|-------------|--------------------|
| `map_window` | **P** | Sequential read of a contiguous ordinal range; the map must be destination-ordered | Filter the whole map per pass: `Θ(N·P)` instead of `Θ(N)` |
| `record_count` | **P** | Bounds validation | Skip validation and risk silent corruption |
| `read_record` | **A** | Locating a record at a cost bounded by its container, not the store; concurrent reads from several workers | Fall back to one worker, or pre-build an ordinal index |
| `write_range` | **T** | Either positional writes, or strictly ordered appends (see below) | No sink, no algorithm |
| `memory_budget` | **S** | A byte figure; a constant is fine | Pick a conservative constant; the pass count follows from it |
| `spawn` | **A**, **L** | Concurrent execution over disjoint work items | Run serially — correctness is unaffected, throughput is not |

### Ordered-append sinks

`write_range` at a computed position is the natural fit for
fixed-stride records. A sink that only appends — object storage
multipart uploads, log-structured stores, a pipe — satisfies the
contract too, provided passes run in output order and each pass writes
its segment completely before the next begins. That is the only place
where pass *ordering* (as opposed to pass *independence*) matters, and
it is what a variable-length record store requires anyway.

### Concurrency requirement

Assemble writes into the segment buffer from many workers at once. The
plan guarantees each output position appears exactly once, so the
target regions are disjoint by construction and no locking is needed —
but the host language must let you express *"disjoint mutable slices of
one buffer"*. In Rust that is a split-at-mut or a checked unsafe
wrapper; in C, raw pointers; in Java, distinct `ByteBuffer` slices or
an `Unsafe`-backed region; in Go, subslices; in Python, a shared
`memoryview` or per-worker arrays merged afterward. Where the language
cannot express it, give each worker its own scratch buffer and
concatenate in output order — the disk-facing invariants survive, at
the cost of one extra copy.

## Optional primitives

```
 hint_sequential(range)  -> ()      tell the tier to read ahead
 durability_barrier()    -> ()      flush this pass before the next
 observe(event)          -> ()      progress, counters, timings
 checkpoint_store        -> ()      persist and replay completed segments
```

| Primitive | Why | Cost of omitting |
|-----------|-----|------------------|
| `hint_sequential` | Turns a pass's ascending reads into streamed ones | Reads still ascend; prefetch is left to the tier's own heuristics. **Not always a win** — see the amplification regimes in [cost-model.md](./cost-model.md) |
| `durability_barrier` | Bounds dirty-page debt per pass; without it, one pass's writeback lands inside the next pass's read phase and reads as slow reads | Throughput noise, and a weaker crash story |
| `observe` | Long rewrites are opaque; passes give natural checkpoints to report | No visibility |
| `checkpoint_store` | **Out of scope for now.** Persist each finished segment keyed by a fingerprint of the parameters; a re-run replays completed segments and recomputes only the remainder | A failure at pass `k` costs all `k` passes |

A checkpoint fingerprint must cover everything that changes the output:
source identity, map identity, output range, segment count, and any
per-record hook configuration. A mismatch must invalidate the whole
set, not part of it — a cache keyed too loosely is worse than none.

## Per-record hooks

Anything that must touch every record should ride along inside
assemble, while the bytes are already in cache, rather than in a
separate pass:

```
 transform(bytes)  -> bytes         reshape, re-encode, normalize
 predicate(bytes)  -> bool          drop records that fail a test
```

A `predicate` that rejects records makes the output shorter than the
plan. The host must then either compact (pack surviving records
tightly, track a running output count, and truncate at the end — which
turns positional writes into "append at the running total") or emit
placeholders. Compaction is usually what callers want; it means the
output ordinal space is no longer the map's ordinal space, so record
the dropped ordinals if downstream consumers need the correspondence.

Sampling can elide a hook entirely: if a check over a bounded sample
proves the transform is a no-op for this input, skip it for all records
and say so in the log.

## Bindings

| Primitive | POSIX / Linux | JVM | Object storage (S3-like) | In-memory / embedded |
|-----------|---------------|-----|--------------------------|----------------------|
| `map_window` | `pread` on the map file, or `mmap` a window | `FileChannel.read(buf, pos)` | one ranged GET per window | slice of an array |
| `read_record` | `pread(fd, buf, R, ordinal×R)` | `FileChannel.read` at position | ranged GET, coalesced | index into a buffer |
| `write_range` | `pwrite` / `write` after `lseek`; preallocate with `ftruncate` | `FileChannel.write` at position | multipart upload, one part per segment | copy into a buffer |
| `memory_budget` | cgroup limit, or a fraction of RAM | `-Xmx` share, off-heap budget | container limit | caller-supplied |
| `spawn` | threads | executor / virtual threads | async request pool | threads or serial |
| `hint_sequential` | `posix_fadvise(SEQUENTIAL)`, `madvise` | (no direct equivalent) | request coalescing | n/a |
| `durability_barrier` | `fdatasync` per pass, `fsync` at the end | `FileChannel.force(false)` | part upload completes | n/a |

### Notes per tier

**POSIX.** Prefer positional reads over memory-mapped slices for the
source. Faulting shuffled records through a mapping pulls every touched
page into the process's resident set, which on a multi-terabyte source
means a resident set measured in hundreds of gigabytes; positional
reads move the same bytes through the same page cache without mapping
them. Preallocate the output before the first pass so passes never
extend the file.

**JVM.** Note the 2 GiB `ByteBuffer` ceiling: a segment buffer above
that must be a list of chunks, which the scatter step must address
accordingly. Direct buffers avoid a copy but are billed outside the
heap — count them against the same budget.

**Object storage.** This is where gsplat pays best and behaves most
differently. Requests are priced individually and have a large
effective container size (`W` of a megabyte or more), so `N` random
GETs is both slow *and* a line item. Two adaptations follow: coalesce
adjacent plan entries into single ranged GETs after linearizing (the
sort makes them adjacent), and size segments so that each pass's reads
coalesce into a small number of large ranges. The write side maps
cleanly onto multipart upload with one part per segment.

**In-memory / embedded.** If the whole collection fits, gsplat reduces
to a permute-in-place and there is no reason to run it. It stays useful
when "memory" is a scarce accelerator memory and "storage" is host RAM:
the same segment/pass structure bounds device-side residency.

## Validation surface

A conforming host should fail loudly on:

- a map value outside `[0, N_src)` — check as entries enter the plan,
  before any output is written for that pass;
- a segment buffer allocation that exceeds the budget;
- a short read or short write at any position;
- a checkpoint fingerprint mismatch (invalidate, do not partially
  reuse).

It cannot check map *orientation* — both orientations of a permutation
are valid permutations — which is why orientation is stated as a
precondition and worth asserting upstream where the map is produced.
