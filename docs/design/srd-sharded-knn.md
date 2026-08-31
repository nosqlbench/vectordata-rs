# SRD — KNN over a sharded base

**Status:** proposed
**Scope:** `compute_knn`'s executors and partition loop, the KNN segment
cache, `dataset_lookup`.

**Depends on** [srd-multifile-facet-shards.md](srd-multifile-facet-shards.md).
That document specifies how a facet spread over several files is
declared, realized and read. This one is about the single consumer that
still cannot take one: the neighbour computation.

## 1. Problem

`compute knn` reads a base facet as one memory-mapped file. Its
executors take a `&Path`, open an `XvecReader<T>` from it, and hand that
reader through about ten signatures down to the distance kernels, which
call `get_slice(j)` on the hot loop and expect a borrow into a mapping.

A sharded base has no single path. Today the pipeline's facet lookup
refuses one by name and points at `veks datasets derive` with no
`--shard-stride`, which writes the series back as a single file
(SH-38). That is a real route and it is tested, but it costs a full copy
of the base — for the datasets that motivated sharding, a copy is the
thing sharding existed to avoid.

**K-1.** `compute knn` must accept a base facet declared as a series,
and produce results identical to the same vectors in one file.

## 2. Why the obvious approaches do not work

Stated so the design below is read against them.

**K-2.** *Concatenate at read time.* Rejected: it is the copy the
workaround already performs, spelled differently.

**K-3.** *Make the reader span files.* `XvecReader` holds one
`Arc<Storage>` and computes `index * entry_size` into its mapping.
Teaching it a shard map puts a lookup on the hot path of every
single-file read in the project, which is the opposite of SH-73's
requirement that the unsharded case add no indirection. A separate
sharded reader type does not help either: the ten signatures name
`XvecReader<T>` concretely, and widening them to a trait object costs
the borrow — `VectorReader::get_slice` returns `Option<&[T]>`, and the
kernels want `&[T]`.

**K-4.** *Run the kernel per shard and merge.* This is the design. The
partition machinery **already** segments the base into ordinal ranges,
runs the kernel per range, and merges — because that is how it bounds
page-cache pressure on a base too large to map at once. A series is a
segmentation whose boundaries happen to be file boundaries.

## 3. Design

**K-5.** The base is presented to `execute_with_partitions` as a
**provider** rather than a reader: given a partition's ordinal range, it
answers the reader that serves it and the offset of that reader's
ordinal zero within the facet. A single file is the provider that
answers the same reader for every range with offset zero, which is the
case that must compile to what it does today.

**K-6.** Partitions are **clipped to shard boundaries**. A partition
that straddled two files would need two readers, so the partition
planner takes the shard seams as mandatory split points and its own
sizing as advisory within them. A shard larger than the memory-derived
partition size is still divided; a shard smaller than it is not merged
with its neighbour.

**K-7.** Neighbour indices are **global facet ordinals**. The kernel
reports indices in the reader's own space, so the provider's offset is
added as results come back, before merge and before anything is
written. This is the step with no safe default: an unoffset index is a
valid-looking ordinal pointing at the wrong vector, and nothing
downstream can detect it.

## 4. The segment cache

**K-8.** Cached segments are keyed by `(start, end)` in the base's
ordinal space, alongside engine, cache version, k and metric. Shard-
aligned partitions produce ranges that are a **subset** of the ranges an
unsharded run produces, so a cache written by one is readable by the
other only where the boundaries coincide. That is acceptable — a miss
costs recomputation, not a wrong answer — provided a *hit* is always
correct.

**K-9.** The per-dataset component of the key is
`{base_stem}.{query_stem}.{base_size}_{query_size}`. A series has
neither a single stem nor a single size, so one has to be defined for
it, and the choice is not free:

- **The first shard's stem and the series' total size.** Cheap, and it
  reads naturally. But two series sharing a basename pattern and a total
  differ only in content, which this key cannot see.
- **A fold over every shard's stem and size.** Distinguishes those, and
  is still free — the sizes are already stat'd during realization.
- **The realized shard list itself, hashed.** Same cost, and it also
  distinguishes two series whose shards are the same sizes in a
  different order.

The existing key already accepts a collision between same-path,
same-size, different-content inputs, and documents it as the user's to
avoid by regenerating through the pipeline or clearing `.cache`. A
sharded key should be **no weaker than that**, which the second and
third options both satisfy and the first does not.

**K-10.** A flat copy and the series it was derived from are the same
vectors under different names, and under any of the above they miss each
other. That is the safe direction and probably the right one: the copy
exists because a run needed a single file, and a cache hit that crossed
between them would rest on the two being byte-equivalent, which nothing
checks.

**K-11.** A segment is reused only when its range lies entirely within
the current run's base. A partial overlap is a miss, not a trim:
trimming a k-nearest result to a sub-range does not give the k-nearest
of that sub-range.

## 5. What must not change

**K-12.** The unsharded path's inner loop. Every change above is at the
partition boundary, which runs once per partition; nothing may be added
to the per-vector path. A provider that resolves one range per partition
and hands back the same `&XvecReader<T>` the code has today satisfies
this by construction.

**K-13.** Results. The acceptance case is numerical: the same vectors,
sharded and flat, produce byte-identical neighbour and distance files
for the same k and metric.

## 6. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | sharded base, one shard | identical to the flat file |
| 2 | sharded base, several equal shards | identical to the flat file |
| 3 | last shard short | identical; no partition past the end |
| 4 | partition smaller than a shard | shard divided, results identical |
| 5 | partition larger than a shard | not merged across the seam |
| 6 | neighbour indices | global facet ordinals, never shard-local |
| 7 | a query whose true neighbours span a seam | found, and ranked correctly |
| 8 | cache from a flat run | not reused by the sharded run (K-10) |
| 9 | cache from a different series, same shard sizes | not reused (K-9) |
| 10 | cache from the same series, same boundaries | reused |
| 11 | unsharded run | unchanged, and no slower |

**K-14.** Case 7 is the one that fails silently if K-7 is wrong, and
case 11 is the one that fails silently if K-12 is. Both need to be
measured rather than reasoned about — the first because a wrong index
is a plausible one, the second because a regression in the inner loop
shows up as "the machine was busy".

## 7. Open

**The sharded cache key (K-9).** Between the fold and the hashed shard
list, the difference is whether two series with identically-sized shards
in a different order must be told apart. Worth settling against a real
catalog before implementing; the fold is less to write and the hash is
harder to get wrong later.

**Whether `--partition-size` keeps its meaning.** Under K-6 it becomes
advisory within a shard. Either the flag's documentation changes or the
planner reports the size it actually used, and it is probably both.
