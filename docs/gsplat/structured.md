# sgsplat — Structured sources and sinks

gsplat assumes a flat ordinal space on both sides: record `n` lives at
`base + n × R`, so ordinal order *is* address order and the segment
buffer is a plain array. sgsplat drops that assumption. Source and sink
are hierarchies — groups, row groups, chunks, pages, members, nested
records — whose leaves hold the payload, and the rewrite must read from
one clustering and write into a different one.

This document states the reduction that makes it tractable, the four
places the reduction leaks, and the step-by-step deltas that close
them. It layers on [gsplat](./README.md); everything unstated here is
unchanged.

## The reduction

Two moves turn the structured problem back into the flat one.

**1. A pre-order DFS fixes each ordinal space.** Number the leaves in
pre-order visit order; that numbering *is* the ordinal space, on both
sides. Nothing else about the hierarchy enters the algorithm. DFS is
**required**, not merely conventional — see [DFS normal
form](#dfs-normal-form) — and BFS is out of scope.

**2. A skeleton carries the map.** Build the output structure holding
*logical pointers* — source ordinals — in the leaf slots where payload
will eventually go. Read the skeleton's leaves in output-traversal
order and you have exactly

```
 map[i] = source ordinal of the record belonging at output position i
```

which is gsplat's destination-ordered map, by construction. The rewrite
is then a **value substitution**: traverse the skeleton, replace each
ordinal with the bytes of the record it names, emit structure around
them.

This is the right backbone. The structure is confined to the read and
write boundaries; the middle stays exactly gsplat, and the invariants —
single read, single write, monotone access, bounded memory,
determinism — survive intact.

## DFS normal form

The traversal requirement is not about traversal for its own sake. It
buys back exactly one property, and that property is what makes the
whole reduction pay:

> **ordinal order == address order**

Stated as a condition on the *format* rather than on the walk, a store
is in **DFS normal form** when:

1. leaves are numbered in pre-order,
2. a subtree's bytes are contiguous, and
3. siblings are serialized in their logical order.

Under (1)–(3), `i < j ⟹ address(i) < address(j)`. Note what is *not*
claimed: the address function is **monotone, not affine**. Compressed or
variable-length containers make byte offsets nonlinear in the ordinal,
and that is fine — monotonicity is the entire requirement, because
Linearize only needs a correct sort order, never an arithmetic address.

### Two tiers, and why columnar formats still qualify

| Tier | Condition | Formats | Instances |
|------|-----------|---------|-----------|
| **Whole-file** | the entire leaf sequence is monotone in address | record-oriented containers: archives, record streams, contiguous arrays, directory trees | one |
| **Per-path** | each leaf path's value sequence is monotone in address | columnar and chunked: Parquet, ORC, chunk-object stores | one per leaf path |

Columnar formats violate the whole-file version — the file is
row-group-major and then column-major, so a whole-file DFS interleaves
columns that are physically far apart — but they satisfy the per-path
version exactly: within one column, values run in record order,
contiguous inside each row group and ascending across them.

**A columnar file is a transposed DFS, and the per-leaf-path
decomposition undoes the transposition.** This is the useful surprise
of requiring DFS: the fix for [leak 4](#4-a-record-is-not-one-contiguous-range)
and the fix for [leak 1](#1-ordinal-order-versus-address-order) are the
same fix. Decompose per leaf path and each instance is in whole-file
normal form.

### What the requirement buys

1. **Linearize sorts by ordinal again** — exact gsplat, with no address
   machinery in the inner loop.
2. **Plans compress to ranges.** A run of consecutive source ordinals is
   a contiguous byte range (clipped at container boundaries), so the
   sorted plan run-length-encodes into ranged reads by construction.
   This is the difference between `N` requests and a few thousand on
   object storage.
3. **A container index is sufficient addressing.** Cumulative record
   counts per container plus container start offsets — which container
   formats already carry in their footers — replace any per-record
   address table. Within a container, addressing is either affine or a
   sequential decode you are performing anyway.
4. **Pass cost becomes computable before running.** Which containers a
   pass touches is an interval intersection between the segment's sorted
   ordinal set and the container index, so the amplification `A(P)` is
   *exact* for a given map rather than estimated from the uniformity
   assumption in [cost-model.md](./cost-model.md). `P` can then be chosen
   by evaluating the real map instead of a model of it.
5. **Resume is derivable.** A segment's input container set is a
   deterministic function of the map and the index, so a restarted run
   knows what to fetch without replaying prior passes.

Point 4 deserves emphasis: it converts the pass-count decision from a
tuning exercise into a calculation, and it costs one pass over the map
plus the footer — no payload reads at all.

**What it does not buy.** Normal form is about *ordering* only. It says
nothing about divisibility, so container-atomic fetch and decode
amplification stand unchanged; nothing about output sizing, so
ordered-append and container-aligned segments stand; and nothing about
record shape, so repeated fields still produce variable-length runs.
Leaks 2 through 4 below are unaffected — only leak 1 dissolves.

## Where the reduction leaks

### 1. Ordinal order versus address order

**Closed by the normal form**, and worth recording as the reason the
normal form is mandatory rather than advisory.

gsplat's Linearize sorts by *source ordinal* on the strength of
`address = base + ordinal × R`. In a hierarchy that identity is gone in
general: a BFS numbering over a depth-first-laid-out file, a
column-major store traversed row-major, or any whole-file walk of a
columnar layout all produce plans that *look* linearized while the
device still seeks. The two concepts flat storage let us conflate are
genuinely distinct:

| Concept | Defined by | Used for |
|---------|-----------|----------|
| **ordinal** | the traversal | identity, the map, output placement |
| **address** | the store's physical layout | ordering the reads |

[DFS normal form](#dfs-normal-form) collapses them back together —
monotonically, not affinely — so Linearize can sort by ordinal and be
sorting by address. That is the whole reason to require it, and why BFS
is excluded: BFS has no such correspondence with any common
serialization, so admitting it would mean carrying an address key
through every step to serve a traversal nobody needs here.

**The escape hatch.** A store that is not in normal form on either tier
— a hash-partitioned layout, an LSM store where record order and
storage order are unrelated, a chunk store with no canonical key order —
must supply `address_of(ordinal)` explicitly, and Linearize sorts by
that. In practice the cheapest route is one preliminary sequential
traversal that emits an `ordinal → address` index, which is small
(ordinal-width per record), reusable across rewrites, and turns the
store into normal form for every subsequent operation. Prefer building
that index once over paying the address indirection forever.

### 2. The fetch unit becomes a container, and containers are atomic

In flat gsplat, `W` is a device readahead window and reading "too much"
wastes bandwidth. In a structured format the unit is imposed by the
format: you cannot read one record out of a compressed row group, chunk,
or page without materializing the whole thing. Three consequences:

- **`w` (records per fetch unit) becomes large** — often 10⁴–10⁶ rather
  than tens. Since the amplification regimes cross over at `P = w`
  ([cost-model.md](./cost-model.md)), structured rewrites sit almost
  always in the **dense** regime, where `A ≈ P`. The model simplifies:
  every pass effectively streams the source.
- **Amplification now multiplies CPU, not just bytes.** Touching a
  container in `A` passes means decoding it `A` times. For compressed
  columnar data the decode is frequently the dominant cost, so the pass
  count is worth even more than the byte model suggests.
- **The natural unit of work becomes the container, not the record.**
  Assemble should iterate containers in address order, decode each
  once, and drain every plan entry that falls inside it before moving
  on. That is a small restructuring of the assemble loop and it is what
  makes the dense regime affordable.

### 3. The output cannot be positioned or preallocated

Output containers are serialized with compression and encoding, so
their sizes are unknown until written, and their metadata — footers,
offset indexes, dictionaries, statistics, checksums — is computed from
the data. Preallocating an output extent and seeking to
`segment_start × R` is meaningless.

sgsplat therefore requires the **ordered-append sink** that gsplat
already permits, and adds a constraint on segmentation:

> **Segment boundaries must align to output container boundaries.**

Otherwise a container straddles two passes and cannot be finalized when
its first pass ends. Practically: round the segment size down to a whole
number of output containers, or accept carrying one partial container
in memory across a pass boundary. Round *down*, so the buffer stays
inside the budget.

A **finalize** step joins the pipeline after the last pass, emitting
whatever the format keeps outside the payload — footer, index, manifest,
checksum tree. It is the structured analogue of gsplat's final
durability barrier and it is the only step that needs global knowledge.

### 4. A record is not one contiguous range

For nested or columnar layouts, one logical record's bytes are scattered
across `K` leaf paths — a value in each column chunk, plus repetition
and definition levels for nested fields. There is no single byte range
to gather, so "substitute the ordinal with the record" has no literal
meaning.

The clean generalization: **sgsplat is `K` gsplat instances sharing one
map.** Each leaf path is its own flat ordinal space with its own
addresses and its own record width; all of them are driven by the same
destination-ordered map; a structural merge assembles the per-column
outputs into containers.

That decomposition is a *win*, not a tax:

- per-column records are narrower, so a segment holds more of them and
  `P` drops;
- each column's reads are contiguous in its own strip, which is the
  best locality available anywhere in the problem;
- columns are independent, so they parallelize without coordination.

Two constraints come with it: the memory budget divides across columns
processed concurrently, and every column must use **identical segment
boundaries** so their outputs line up into the same containers.

**Repeated fields break the 1:1 ordinal correspondence.** Where a leaf
path sits under a repeated field, one logical record contributes a
*run* of values to that column, not a single value — and the run length
varies per record. Two consequences for the per-column instances:

- The column's ordinal space is not the record ordinal space. Each
  instance needs a `record → (value offset, run length)` index, and it
  permutes **runs**, which makes its records variable-length even when
  the underlying values are fixed-width. That is gsplat's
  variable-length path ([04](./04-assemble.md), [05](./05-transfer.md)),
  reached here for a structural reason rather than a payload one.
- The structure-encoding streams that accompany the values — Dremel's
  repetition and definition levels, or whatever the format uses to
  reconstruct nesting — are value-aligned and must be permuted with
  them, as part of the same run. Treat `(values, levels)` as one
  composite record per run; splitting them into separate instances
  would triple the pass count for no locality gain.

## Step deltas

| Step | gsplat | sgsplat |
|------|--------|---------|
| **S** Segment | `S = M / R`, floor of 2 | plus: snap `S` down to whole output containers; divide the budget across concurrently-processed leaf paths |
| **P** Plan | read a contiguous map window | flatten the skeleton to a dense map first (one traversal); then identical, with entries still `(source ordinal, local)` |
| **L** Linearize | sort by source ordinal | unchanged under normal form; then run-length the sorted ordinals into ranges and group them by container, so each container is fetched and decoded exactly once per pass |
| **A** Assemble | read record, scatter to `local × R` | iterate containers in address order, decode once, drain all entries inside; scatter per leaf path |
| **T** Transfer | positional contiguous write | ordered append of whole containers, with per-container metadata; global finalize after the last pass |

## Fast paths

Structure creates two shortcuts that flat gsplat has no analogue for,
and both are worth detecting explicitly:

- **Container-granular permutations.** If the map permutes whole
  containers — the destination of every record in a container is the
  same container, in the same internal order — then no payload moves at
  all. Copy containers verbatim, or in formats that separate index from
  payload, rewrite only the index. This collapses a terabyte rewrite
  into a metadata edit, and it is exactly the class that re-blocking
  tools address.
- **Clustering-preserving permutations.** If the map is monotone within
  each container (records keep their relative order and only containers
  interleave), the read plan is already ascending inside each container
  and Linearize degenerates to a merge of `P` sorted runs.

The flat analogue — an already-ascending map streaming through without
a transpose — still applies per leaf path.

## Cost model deltas

The amplification formula is unchanged; the parameters move:

```
 w = records per container       (format-imposed, typically 10⁴–10⁶)
 A(P) = P · (1 − exp(−w / P)) ≈ P   because P ≪ w in practice

 tier bytes read  ≈ P × (source payload bytes)
 decode CPU       ≈ P × (source decode cost)      ← new dominant term
 write bytes      ≈ output payload bytes (once)
```

So for structured formats the entire cost story reduces to: **`P`
multiplies everything on the read side, including decompression.** The
two-level extension from [cost-model.md](./cost-model.md) maps onto
structure directly and is more attractive here than in the flat case:

```
 level 1  stream source containers once, decode once; append each
          record to the spill bucket of its destination container
 level 2  per output container: read its bucket, order in memory,
          encode and emit
```

Cost becomes ~2 decodes and ~2 encodes total, independent of `P`. Where
the flat version trades scratch space for I/O, the structured version
trades scratch space for **codec work**, which is usually the more
expensive resource.

## Host interface additions

Beyond gsplat's six primitives
([host-interface.md](./host-interface.md)):

```
 traverse_leaves(structure)   -> ordinals in pre-order       (skeleton flattening)
 container_index()            -> per container: first ordinal, record count, start offset
 read_container(id)           -> decoded records
 begin_container() / end_container(metadata)                 (ordered-append sink)
 finalize(structural metadata)

 address_of(ordinal)          -> monotone key   (escape hatch only; stores not in normal form)
```

`container_index` is the load-bearing addition, and it is deliberately
coarse: per *container*, not per record. Every container format already
carries it in a footer or superblock, it is small enough to hold
resident for any realistic file, and under [normal
form](#dfs-normal-form) it is sufficient — ordinal ranges map to
container ranges by binary search, and within a container the records
are located either arithmetically or by the sequential decode the
container demands anyway.

A store that cannot supply it, or that is not in normal form, falls
back to `address_of` and pays an indirection on every plan entry. The
better move in that case is to build the `ordinal → address` index once
and put the store into normal form permanently.

## Related work, assessed

**Aggarwal & Vitter (1988)** and **Vitter's survey (2001)** are the load
bearing theory: permuting has the same I/O complexity as sorting in the
parallel-disk model, and the two paradigms that reach it are
distribution and merging. gsplat is a one-level distribution; sgsplat's
two-level variant is the honest instantiation of the bound. This is the
literature that actually constrains what we can hope for.

**Out-of-core matrix transposition** (Krishnamoorthy et al. and the
surrounding IEEE literature) is the closest real ancestor. Transposition
*is* a structured permutation, and that work already reasons about the
two things that matter here — multi-level tiling chosen from device
characteristics, and the tradeoff between index computation and I/O.
Where it is narrower than sgsplat: the permutation is known in closed
form, so plans are computed rather than materialized, and there is no
codec in the loop.

**Dremel** (Melnik et al., 2010) supplies the decomposition in leak 4:
nested records shred into per-leaf-path strips that can be processed
independently. That is precisely why "one gsplat per column, one shared
map" is the right shape rather than a hack.

**Cache-oblivious algorithms** (Frigo et al., 1999) are the principled
alternative to tuning `P` against `W`. Worth knowing about, but the
structured case makes `w` an explicit, format-declared quantity rather
than an unknown device property — so tuning is easy and obliviousness
buys little.

**Space-filling-curve clustering** — Z-order, Hilbert, and the
liquid-clustering line in current lakehouse systems — answers a
*different* question: what the target order should be. It produces
maps; sgsplat consumes them. The two compose cleanly, and it is worth
noting how those systems apply their maps today: a distributed
sort-and-rewrite, which is the merge paradigm at cluster scale, paying
the full sorting bound and requiring the cluster. sgsplat is the
bounded-memory, single-node counterpart to that rewrite, which is the
regime those systems do not serve.

**Structured permutations** with closed forms — bit-reversal, butterfly,
and the transposition family — need no materialized map at all: the
destination is computed per ordinal. That is gsplat's computable-map
case ([02-plan.md](./02-plan.md#computable-maps)) and it removes the
skeleton entirely when it applies.

**Tree layout in memory hierarchies** (van Emde Boas layouts, blocked
and succinct tree encodings) is adjacent but inverted: it chooses where
nodes live so that *traversal* is I/O-efficient, taking the data as
given. sgsplat takes the layout as given and moves the data. The two
meet only in that both care about which traversal defines locality.

**On the gap.** The components above are each well covered — external
permutation theory, out-of-core transposition, columnar shredding,
clustering-order selection. What I could not find is the *composition*:
an arbitrary, data-dependent permutation applied across hierarchical
container formats with codecs in the loop, under a fixed memory budget,
on one machine. Systems that need it today reach for a distributed
shuffle; the theory that covers it stops at the flat record model. That
gap is the reason this document is worth writing down rather than
citing.

**Rechunker and Cubed** (chunked-array re-blocking) come up in searches
for this problem and are worth being precise about, because the
resemblance is superficial. They solve *re-blocking*: every element
keeps its logical index and only chunk boundaries move. There is no
reorder map, no scatter, and therefore no linearize step — both sides
are monotone in the same index space throughout. Their central trick,
choosing intermediate chunks as the elementwise minimum of read and
write chunks, works because chunk grids intersect into a grid; an
arbitrary permutation has no grid to intersect and the construction
degenerates to individual records. Their headline bound (fewer than
`N + M` tasks) targets scheduler overhead in a particular execution
substrate, not device-level amplification. The one genuine point of
contact is the container-granular fast path above — re-blocking is
roughly what sgsplat degenerates to when the map does not actually
permute anything. Nothing in that line of work addresses the problem
sgsplat exists for.

## Open questions

1. **Skeleton materialization.** Building the skeleton is assumed here.
   Where does it come from, and is it always cheaper than computing the
   map directly? For sorts and clustering rewrites the map is a
   by-product of the sort; for structure-defined rewrites the skeleton
   is the natural artifact.
2. **Budget split across columns.** Per-column instances want the whole
   budget; running them concurrently divides it and raises `P` for each.
   Sequential-by-column with the full budget is likely better whenever
   the decode cost dominates — worth measuring rather than assuming.
3. **Partial containers at segment boundaries.** Rounding down wastes
   budget; carrying a tail complicates the write path. Which is better
   probably depends on the ratio of container size to budget.
4. **Two-level bucket count.** Level 1 needs one open spill bucket per
   output container, which at fine container granularity is a lot of
   open handles and write buffers. A hierarchical bucketing pass may be
   needed at the extremes.
5. **How much does requiring normal form exclude?** The tier-two
   condition is weak enough that the formats worth targeting appear to
   satisfy it, but that is an assertion from a handful of examples, not
   a survey. The cost of being wrong is bounded — a non-conforming store
   pays one indexing pass — but the claim deserves checking against real
   candidates before the requirement hardens into an interface.
