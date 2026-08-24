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

**1. A traversal fixes each ordinal space.** Choose a deterministic
traversal of the native structure — DFS/pre-order for most container
formats — and number the leaves in visit order. That numbering *is* the
ordinal space, on both sides. Nothing else about the hierarchy needs to
enter the algorithm.

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

## Where the reduction leaks

### 1. Ordinal order is not address order

gsplat's Linearize sorts the plan by *source ordinal* on the strength
of `address = base + ordinal × R`. In a hierarchy that identity is
gone. A BFS numbering over a depth-first-laid-out file, a column-major
store traversed row-major, a container whose byte offset depends on the
compressed size of its predecessors — in all of these, ascending
ordinals do not mean ascending bytes, and sorting by ordinal produces
a plan that *looks* linearized while the device still seeks.

The fix is to separate the two concepts that flat storage let us
conflate:

| Concept | Defined by | Used for |
|---------|-----------|----------|
| **ordinal** | the chosen traversal | identity, the map, output placement |
| **address** | the source's own index or footer | ordering the reads |

**Linearize sorts by address, not ordinal.** The address need not be a
byte offset; any key monotone in physical layout works — `(container
position, offset within container)` is the usual one, and it comes from
the format's existing index. Flat gsplat is the special case where the
two keys are the same function of the ordinal.

A corollary worth stating for implementers: prefer the traversal whose
order matches physical layout (usually pre-order DFS). If the
application needs BFS semantics for identity, keep BFS for the ordinal
space and still sort by address for I/O — the two roles are
independent, which is the whole point of splitting them.

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
| **P** Plan | read a contiguous map window | flatten the skeleton to a dense map first (one traversal); then identical. Plan entries carry `(address, ordinal, local)` |
| **L** Linearize | sort by source ordinal | **sort by source address**; group by container so each is decoded once |
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
 traverse_leaves(structure)   -> ordinals in traversal order   (skeleton flattening)
 address_of(ordinal)          -> monotone physical key         (from the format index)
 container_of(address)        -> container id + offset
 read_container(id)           -> decoded records
 begin_container() / end_container(metadata)                   (ordered-append sink)
 finalize(structural metadata)
```

`address_of` is the load-bearing addition: everything else is
bookkeeping around it. A format that cannot answer it — no index, no
footer, no way to locate a leaf without scanning — forces either a
preliminary indexing pass (one sequential traversal producing
`ordinal → address`, which is small and reusable) or abandonment of the
monotone-read guarantee.

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
