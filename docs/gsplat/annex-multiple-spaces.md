# Annex — Multiple ordinal spaces and interleaved output

Status: **active annex.** The multi-root case it describes is germane to
current work — the spaces are genuinely free, not induced by a
relationship that has yet to be made explicit. Nothing in
[structured.md](./structured.md) or the [gsplat](./README.md) set
depends on this document, and the material stays here rather than being
folded into the core, but it is a live design line rather than a
speculative one.

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
 spaces        S_1 … S_K      each with its own ordinals and map m_k, interleaved in one store
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

Let `rank_k(i)` be the number of slots before `i` that belong to space
`k`. Because `rank_k` is monotone and each space's records are consumed
in order, a contiguous range of output slots induces a **contiguous
range in every space's ordinal space**:

```
 output slots [s, e)   →   space k needs ordinals [rank_k(s), rank_k(e))
```

So segmentation is unchanged, and — the part that matters — each space
still gets a **contiguous map window**, which is what the plan-window
shortcut in [02-plan.md](./02-plan.md) depends on. No space needs its
map filtered.

## Step deltas this would imply

| Step | sgsplat today | With K spaces |
|------|---------------|---------------|
| **S** Segment | contiguous range of output ordinals | unchanged — but the segment is a range of *slots*, and the per-space windows follow from `rank_k` at the two boundaries |
| **P** Plan | scan the map window, reverse each entry | scan the slot window, **bucket by space** while scanning; entries become `(space, source ordinal, segment-local slot)` |
| **L** Linearize | sort by source ordinal | sort by **address** — equivalently by M-ordinal — in one sort across all spaces; per-space ordinals are incomparable and must never share a sort key |
| **A** Assemble | read source ascending, contribute | one ascending sweep; each container touch drains every space's entries that fall inside it |
| **T** Transfer | builder emits | unchanged |

The schedule is consumed **sequentially** by Plan, in slot order: as the
scan walks slots it carries each space's running index — that is,
`rank_k` — forward, computing it incrementally rather than looking it
up. With
resume out of scope, nothing else needs random access into it, so **a
sequential read of the tag stream is the whole requirement** — no
rank/select structure, no residency claim beyond a streaming buffer.
Random access would only be wanted for computing a segment's windows
without replaying the prefix, which matters if resume ever comes back
into scope.

### The builder gets simpler, not harder

Since τ determines the space of every slot, the builder's `contribute`
is addressed by **output slot alone** — the space is a property of the
slot, not a parameter. That is the same addressing the columnar case
wants, which is the first hint that these are one mechanism.

## The spaces are always interleaved

The serialized form is **always** an interleaving of the spaces, by some
pattern that may vary along the stream. Concatenated layouts — all of
one space, then all of the next — are not a case this annex supports or
optimizes for, and neither are separate per-space stores. There is one
address space, and every space's records are dispersed through it.

Two things follow immediately, and they simplify rather than complicate:

1. **Container touches are shared.** A container generally holds records
   of several spaces, so one touch serves all of them. Reading is never
   organized per space.
2. **Per-space ordinals are incomparable, but addresses are not.**
   Anything that must order work across spaces orders it by address —
   practically `(container id, offset)` from the container index.

This is a genuine caveat to the "address = ordinal" substitution in
[structured.md](./structured.md#scope): the identity holds *within* a
space, and **address is the join key across spaces.** The container
index already supplies it, so nothing new is needed — but the concept
that DFS normal form retires comes back at exactly this seam.

## The global-space lens

The cleanest way to hold all of this is to stop thinking in K spaces at
all. Define **M**, a single global ordinal space over every source
record, **numbered in physical address order**:

```
 |M| = Σ_k |S_k|  = N          every record lands at exactly one slot
 gmap[slot] = M-ordinal        one map, one space
```

Under this lens the multi-space problem is not a generalization — it is
the base problem wearing a costume. `gmap` is a permutation between two
spaces of equal size, and Plan, Linearize, Assemble and Transfer run
unmodified. Two properties make it more than relabelling:

- **Sorting by M *is* the cross-space merge.** Because M is numbered by
  address, a single sorted plan is already the correct read order. The
  "merge the per-space plans" step never exists.
- **The output schedule is redundant.** With `σ(m)` the space of source
  M-ordinal `m`, the output schedule is just `τ(slot) = σ(gmap[slot])`.
  Only one of the two need be materialized; the other is derived.

What the lens costs, and how to pay it:

- **Tags cannot be inferred from value ranges.** Because M interleaves,
  a space is not a contiguous run of M-ordinals, so nothing about the
  numbering identifies a record's space. Carry the tag explicitly — one
  bit per entry for two spaces.
- **Validation weakens if you do not.** With per-space maps, an ordinal
  past a space's end is locally, instantly wrong; with a bare global M
  every value in `0..N-1` is structurally valid, so a cross-space error
  reads the wrong type's record silently. The explicit tag restores the
  per-space bounds check.
- **Per-space accounting must be asked for.** Cost attribution and
  pinning both need to know which records belong to which space; the tag
  supplies it, the numbering does not.

Tagged-M and spaces-plus-schedule then differ only in whether the plan
is one tagged vector or K vectors — an implementation choice, not a
different model. Both are lenses on one structure: use M when reasoning
about I/O order and correctness, use spaces when reasoning about cost.

### Building M

`gmap` is derived from the per-space maps and the source tag stream,
which requires `S_k → M` for each space. Since M is address-ordered and
the spaces interleave, those translations are a **rank/select over the
source tag stream**, not arithmetic. Two ways to pay:

- once, as a preprocessing pass that rewrites the per-space maps into a
  single `gmap` — one pass over ordinal-width data, the same class of
  cost as flattening the skeleton; or
- per entry during Plan, translating as the window is scanned.

Either is acceptable; neither touches the inner loop of Linearize or
Assemble.

### A stronger consistency check

Where both `σ` (source tags) and `τ` (output schedule) are available,
they are redundant, and their agreement is checkable per record:

```
 for every slot:  σ(gmap[slot]) == τ(slot)
```

This is strictly stronger than comparing slot counts against map
lengths: it catches individual misplacement, not just aggregate
disagreement.

## Cost model

Interleaving makes this simpler than it first appears. Container touches
are counted over the **union** of containers a pass needs, not per
space, and because every container generally holds records of several
spaces, that union is very close to what a single-space rewrite of the
same total record count would touch:

```
 touches ≈ A(P) × (number of source containers)     one global term
 A(P)    = P · (1 − exp(−w / P))                    w = records per container, all spaces
```

**Multi-space costs essentially nothing extra in I/O.** A per-space sum
would double-count: two spaces sharing a container do not touch it
twice. Where the spaces genuinely differ is in **per-record work** —
decode, extract, contribute — which scales with each space's record
count and its own record shape, not with container touches.

`P` remains shared, and now unavoidably so: one address space means one
traversal schedule. The per-space `P_k` alternative that a
separate-store layout would allow does not arise here.

## Pinning a small space

Pinning changes character under interleaving, and the earlier
join-planning intuition needs adjusting rather than importing.

Because containers are shared, you cannot read "just the small space"
cheaply — its containers are touched anyway on behalf of the others, so
**pinning saves no container touches at all**. What it saves is repeated
per-record work: extract the small space's records once while its
containers are resident, hold them, and let later passes skip
re-extracting and re-decoding them.

That makes it worth doing when a space is small *and* its per-record
work is significant — a dictionary space that must be decoded to be
useful, say. It is worth nothing when the per-record work is a memcpy.

## Schedule representation

The assumed case is an **arbitrary pattern that varies along the
stream**, so the tag stream is materialized by default. The regular
shapes are worth naming only as lucky special cases, and as the
in-memory ancestors of the idea — this is the out-of-core generalization
of the AoS/SoA/AoSoA layout family:

| Pattern | Layout analogue | Representation |
|---------|-----------------|----------------|
| one long run per space | SoA | excluded — that is concatenation, not interleaving |
| periodic, period K | AoS | closed form, nothing stored |
| periodic in blocks | AoSoA | closed form |
| **arbitrary or variable** | — | **materialized tag stream — assume this** |

Periodic patterns cost nothing and need no structure, but nothing should
be built on the expectation of finding one. Materialized, it is a
**rank/select**
problem: an uncompressed bitvector with rank/select support runs about
126% of the raw bits, while compressed representations (RRR,
Elias-Fano) land near 30% and compress run-structured schedules much
further. At 10⁹ slots that is a few hundred megabytes uncompressed and
far less if the schedule is clustered — resident either way, but not
free.

A design-time schedule is almost certainly periodic and therefore free.
A data-dependent one is where this gets expensive, and it is worth
knowing which you have before building anything.

## The ordering forest

The question a schema has to answer, type by type, is not "what is this
type's order" but **"who decides it."** There are only two answers:

- **free** — the type has its own map, chosen independently. It is an
  **ordering root**.
- **induced** — its order is fixed by its parent's, as runs attached to
  each parent record. It has no map of its own; run expansion derives
  one.

Annotate every type in the schema with one of the two. The induced edges
then form a **forest** whose roots are exactly the free types, and each
root together with everything induced beneath it is an **ordering
domain**: a region of the schema governed by a single map.

```
 schema                        ordering annotation
 ─────────────────────         ─────────────────────────────────
 A ──┬── B                     A  free      ┐
     └── C ── D                B  induced   │ domain 1: one map at A
                               C  induced   │
 E ──┬── F                     D  induced   ┘
     └── G                     E  free      ┐ domain 2: one map at E
                               F  induced   │
                               G  induced   ┘
```

Inside a domain, everything follows from the root's map by run
expansion — the machinery already in
[structured.md](./structured.md#3-a-record-is-not-one-contiguous-range).
Between domains, nothing follows from anything: their relative
arrangement in the output is precisely what the schedule specifies.

### What the model tells you

**How many maps a rewrite needs: one per ordering root, not one per
type.** A forty-type schema with one root needs one map; the other
thirty-nine orders are derived. This is the answer to "what must the
caller actually supply."

**Why a schedule exists at all.** With a single root, the output
arrangement is fully determined — the root map fixes root order, each
record's induced children follow it, and the schema fixes the nesting.
Nothing is left to say. With two roots, nothing determines whether
root A's fifth record precedes root B's third, so that has to be stated
externally. **The schedule is exactly the freedom that multiple roots
introduce**, which is why it appears in this annex and nowhere else in
the set.

**Where the cost concentrates.** A root's map is an arbitrary
permutation, and arbitrary permutations are what the whole SPLAT
apparatus exists to survive. Induced levels are run copies: sequential,
no scatter, cheap. So the expensive part of a rewrite scales with the
number and size of **roots**, not with schema size. One root and forty
induced types is a far cheaper problem than forty roots.

**Degrees of freedom, exactly.** The choices available in a rewrite are
one permutation per root, times the schedule. Induced levels contribute
none. That is a precise way to check whether a requested rewrite is even
expressible before attempting it.

### What it subsumes

| Case | Roots | Induced | Schedule needed |
|------|-------|---------|-----------------|
| flat gsplat | 1 | none | no |
| columnar / nested leaf paths | 1 | one per leaf path | no |
| parent-child nesting (Spanner-style) | 1 | one per level | no |
| **this annex** | **K** | none or any | **yes** |

Production interleaving is overwhelmingly the single-root shape, which
is consistent with finding no prior art for the multi-root case: with
one root the schedule is implied by the structure, so nobody needs to
name it.

### Where the model breaks

The annotation is well-formed only if each type has **at most one
ordering parent** — that is what makes the induced edges a forest rather
than a general graph. A type that must be clustered by two different
parents at once (an edge or join type wanted in both endpoints' order)
has two candidate ordering parents and **cannot be expressed**.

That is a real limit, not a gap in the write-up. The standard escape is
the one relational systems already take: duplicate the type, once per
clustering, which in these terms means one domain per copy. The
alternative is to pick one parent and let the other relationship take
whatever order falls out.

### Segmentation across the forest

Segments remain ranges of output slots and cut across domains freely. A
slot range induces a contiguous ordinal range in each root (the
`rank_k` argument above) and a contiguous set of runs in each induced
level beneath it, so every domain gets a contiguous window from the same
segment boundaries. Nothing about the forest disturbs the pass
structure.

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

1. **Settled: the motivating case is free.** The spaces carry
   independent maps, so this is a genuine multi-root forest and the
   schedule is a first-class artifact rather than an implied one. What
   remains open is narrower: how many roots real schemas turn out to
   have, and whether any of them also want the induced machinery
   beneath them.
2. **What does the source tag stream cost in practice?** An arbitrary,
   variable pattern is the assumed case, so the tags are materialized by
   default — but with resume out of scope they are only ever scanned
   sequentially, which makes them a streaming cost rather than a
   resident one. The question is whether any real pattern is regular
   enough to skip materializing them at all.
3. **Shared `P` or per-space `P_k`?** The latter needs several segments
   open at once. Which wins depends on how unequal the spaces are.
4. **Budget allocation.** Builder slots, K plans, pinned spaces, and any
   decoded-container caches now claim one budget. Four claimants is
   probably past the point where ad-hoc splits are defensible.
5. **Composition corner:** a data-dependent schedule *and* variable-length
   records means two indirections to locate anything. Composable, but
   this is where an implementation would get fiddly.
