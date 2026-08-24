# Annex — Multiple ordinal spaces and interleaved output

Status: **exploratory annex.** Nothing here is part of the sgsplat core,
and nothing in [structured.md](./structured.md) or the
[gsplat](./README.md) set depends on it. This records a model, the step
deltas it would imply, and the questions that should be answered before
any of it is adopted.

## The scenario

The input holds more than one **type**, physically distinct, each in its
own ordinal space:

- each space numbers its own records internally, and — as everywhere in
  this family — that numbering is **strictly monotonic** in address
  ([premise 3](./README.md#contract-and-preconditions)); the spaces are
  *independent*, never unordered,
- there is **no visible mapping between spaces** — space A's ordinal 7
  has no relationship to space B's ordinal 7,
- each space carries its own transform, so each has its **own map**,
- records of every space are dispersed through both the input and the
  output.

Independence is a statement about what holds *between* spaces. Within a
space nothing is relaxed at all.

What is known is the **dispersal in the output**: for each slot of the
prototypical output container, which space supplies it. Two spaces is
the simplest instance; nothing below is limited to two.

## The model: spaces and a schedule

```
 spaces        S_1 … S_K      each with its own store, ordinals, and map m_k
 schedule      τ(i) -> k      which space supplies output slot i
 maps          m_k[j]         source ordinal in S_k for that space's j-th output record
```

The schedule therefore does exactly one thing:

> **τ interleaves the spaces; it never reorders within one.** Each
> space's output records are consumed in its own ordinal order.

This follows from the family premise rather than adding to it. Each
space is strictly monotonic internally, so a schedule can only decide
*where* a space's next record goes, never *which* one comes next.

### Why segmentation survives

Let `cum_k(i)` be the number of slots before `i` that belong to space
`k`. Because `cum_k` is monotone and each space's records are consumed
in order, a contiguous range of output slots induces a **contiguous
range in every space's ordinal space**:

```
 output slots [s, e)   →   space k needs ordinals [cum_k(s), cum_k(e))
```

So segmentation is unchanged, and — the part that matters — each space
still gets a **contiguous map window**, which is what the plan-window
shortcut in [02-plan.md](./02-plan.md) depends on. No space needs its
map filtered.

## Step deltas this would imply

| Step | sgsplat today | With K spaces |
|------|---------------|---------------|
| **S** Segment | contiguous range of output ordinals | unchanged — but the segment is a range of *slots*, and the per-space windows follow from `cum_k` at the two boundaries |
| **P** Plan | scan the map window, reverse each entry | scan the slot window, **bucket by space** while scanning; entries become `(space, source ordinal, segment-local slot)` |
| **L** Linearize | sort by source ordinal | sort **per space** — ordinals from different spaces are incomparable and must never share a sort |
| **A** Assemble | read source ascending, contribute | run per space against its own store; merge across spaces first if they share one (below) |
| **T** Transfer | builder emits | unchanged |

The schedule is consumed **sequentially** by Plan, in slot order, so the
normal path needs no random access into it: as the scan walks slots it
can carry each space's running index forward. Random access (`rank` to
find a segment's per-space window without replaying, `select` to map a
space ordinal back to its slot) is needed only for **resume** and for
computing boundaries out of order.

### The builder gets simpler, not harder

Since τ determines the space of every slot, the builder's `contribute`
is addressed by **output slot alone** — the space is a property of the
slot, not a parameter. That is the same addressing the columnar case
wants, which is the first hint that these are one mechanism.

## When spaces share a source store

Two sub-cases behave very differently, and the annex should not assume
either.

**Separate stores per space.** The spaces are genuinely independent: K
read streams against K stores, schedulable concurrently with no
coordination. Clean, and the parallelism is free.

**One interleaved source store.** The spaces share containers, and two
consequences follow:

1. Their linearized plans should be **merged before reading**, so one
   container touch serves every space with records inside it. Processing
   spaces one after another would touch shared containers repeatedly.
2. Merging requires a common comparison domain, and per-space ordinals
   are incomparable by construction. That domain is the **address** —
   practically `(container id, offset)` from the container index.

This is a genuine caveat to the "address = ordinal" substitution in
[structured.md](./structured.md#scope): the identity holds *within* a
space, and **address is the join key across spaces.** The container
index already supplies it, so nothing new is needed — but the concept
that DFS normal form retires comes back at exactly this seam.

## Small-space pinning

A shared segmentation serves spaces of very different size badly: `P` is
set by the builder's capacity for the whole segment, so a small space is
traversed `P` times for no reason.

The fix is the small-side broadcast from join planning. A space whose
entire footprint fits in a slice of the budget is **loaded once and
pinned**, and its amplification drops from `A(P)` to 1. For a small
dictionary space interleaved with a large payload space, this is most of
the available win.

## Cost model

The cost becomes a sum over spaces rather than a single term, with each
space carrying its own container size `w_k`:

```
 total ≈ Σ_k A_k(P) × cost_k        A_k(P) = P · (1 − exp(−w_k / P))
 pinned space:  A_k = 1
 shared source: touches counted per container, not per space
```

`P` is shared, so it is a compromise: a space with small containers or
expensive decode pays for its neighbours. The alternative — per-space
`P_k` with several segments held open at once — trades memory for
per-space optimality and is a real design fork, not an obvious
improvement.

## Schedule representation

The schedule's cost depends entirely on its shape, and the shapes have
names: this is the out-of-core generalization of the AoS/SoA/AoSoA
layout family.

| τ shape | Layout analogue | Representation |
|---------|-----------------|----------------|
| one long run per space | SoA | closed form; the spaces do not really interleave |
| periodic, period K | AoS | closed form, nothing stored |
| periodic in blocks | AoSoA | closed form |
| arbitrary or data-dependent | — | materialized tag stream |

Only the last case costs anything. Materialized, it is a **rank/select**
problem: an uncompressed bitvector with rank/select support runs about
126% of the raw bits, while compressed representations (RRR,
Elias-Fano) land near 30% and compress run-structured schedules much
further. At 10⁹ slots that is a few hundred megabytes uncompressed and
far less if the schedule is clustered — resident either way, but not
free.

A design-time schedule is almost certainly periodic and therefore free.
A data-dependent one is where this gets expensive, and it is worth
knowing which you have before building anything.

## Generalizing to trees: free versus induced ordering

Pushing this toward tree structures forces a distinction the two-space
case does not expose. A node type's ordering is either:

- **free** — it has its own map, chosen independently, or
- **induced** — its ordinals are determined by its parent's map plus run
  lengths, as with repeated fields.

So a schema is a **forest of ordering roots**: every root is an
independent space with a free map, and every non-root is induced by run
expansion from its parent. That single model subsumes what we already
have:

| Case | Roots | Induced levels |
|------|-------|----------------|
| flat gsplat | one | none |
| columnar / nested leaf paths | one | one per leaf path |
| this annex's scenario | K | none |
| parent-child nesting | one | one per level |

If this direction is taken, the forest is probably the right spine, and
the current per-leaf-path and multi-space adaptations become two
readings of it rather than two mechanisms.

## Consistency checks

One new global check, cheap and worth enforcing: **the number of slots
of space `k` must equal `|m_k|`.** A mismatch means the skeleton and the
maps disagree, which otherwise mis-places everything downstream of the
divergence with no error. Alongside it, the existing checks become
per-space: bounds validation against each space's own record count, and
DFS normal form as a **per-space property** — one space may conform
while another needs an index pass.

## Related work

**Nested and interleaved storage in production is always the *induced*
form.** Spanner's `INTERLEAVE IN PARENT` physically colocates child rows
with their parent up to seven levels deep, but the interleaving is
derived from key prefixes — the child's key contains the parent's — so
there is no independent schedule. Oracle clusters and Parquet's repeated
fields are the same shape.

**AoS/SoA/AoSoA** is the flat, in-memory ancestor of the schedule: fixed
interleavings of several typed streams, chosen for locality. The
literature there is about choosing among a handful of regular layouts,
not about applying an arbitrary schedule out of core.

**Succinct rank/select** supplies the tool for a materialized schedule,
with well-characterised space and time costs.

**The gap:** I could not find prior art for the *free* case — several
independently-ordered spaces interleaved by an explicit schedule. Either
it is genuinely unusual, or motivating cases tend to turn out induced
once the relationship between the spaces is made explicit. That question
is first on the list below.

## Open questions

1. **Is the motivating case free or induced?** If the spaces turn out to
   be related after all, the forest model handles it and most of this
   annex collapses into the existing run-expansion machinery. This
   decides whether the free case needs to exist at all.
2. **Is the schedule a design artifact or data-dependent?** Periodic
   schedules are free; materialized ones cost a resident structure and
   put rank/select on the resume path.
3. **Shared `P` or per-space `P_k`?** The latter needs several segments
   open at once. Which wins depends on how unequal the spaces are.
4. **Budget allocation.** Builder slots, K plans, pinned spaces, and any
   decoded-container caches now claim one budget. Four claimants is
   probably past the point where ad-hoc splits are defensible.
5. **Composition corner:** a data-dependent schedule *and* variable-length
   records means two indirections to locate anything. Composable, but
   this is where an implementation would get fiddly.
