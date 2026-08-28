# SRD — Multi-file facet shards

**Status:** proposed
**Scope:** `vectordata` access layer, `veks` pipeline, `dataset.yaml`,
catalogs, `push`, `precache`, conformance.

## 1. Problem

A facet is currently one file. That breaks against hard limits the data
does not care about: object-store per-object size caps, filesystem file
size caps, per-file download timeouts, and the practical ceiling on
re-transferring a single corrupt multi-terabyte object.

The data itself is a flat ordinal space. Splitting it across files must
not change that: a caller asks for ordinal `o`, or the window
`[a..b)`, and the layer resolves which file and which offset inside it.

## 2. Terminology

**Shard** — one contiguous run of ordinals in a facet's series, drawn
from one file. In the uniform form that is always the whole file; in the
explicit form it may be a window into one (SH-61), so a shard and a file
are not the same thing and the SRD keeps the words apart. Chosen over the alternatives
because each is already taken in this codebase and would collide at the
worst moment: *chunk* is the merkle/`ChunkStore` transfer unit (~1090
uses), *segment* belongs to the KNN segment cache, *part* is too common
a word to grep for. `shard` currently appears twice.

**Series** — the ordered set of shards that together are one facet.

**Stride** — the number of ordinals in every shard but the last.

**Global ordinal** — the facet-wide record index a caller uses.

**Local ordinal** — the record index within one shard.

**File ordinal** — the record index within the file a shard is drawn
from. Equal to the local ordinal for a whole-file shard; offset by the
window's lower bound otherwise (SH-64).

**SH-81.** Shards and files are counted separately throughout this
document, and the split is normative. **Ordinal concerns are per shard**
— mapping, `count()`, `get()`, window decomposition, plan
qualification. **Storage concerns are per file** — one `Storage`, one
cache file, one `.mref`, one `IDXFOR__`, one descriptor, one fetch. Two
entries drawn from one file (SH-66) are two shards and one file, and
every storage-side rule below must say *file* or it will double-count
them.

## 3. Naming and layout

A series comes in one of two **layout forms** (§4). SH-1 through SH-5
govern the *uniform* form, whose filenames this project generates. The
*explicit* form carries arbitrary filenames it did not choose and is
bound by SH-50 instead.

**SH-1.** In the uniform form a shard's filename is
`<basename>__<NNNN>.<ext>`, where `<NNNN>` is the shard index as
**exactly four** zero-padded decimal digits. `<basename>` and `<ext>` are the same for every shard in the
series and follow the existing facet spec (`StandardFacet::basenames`,
`FacetFormat::extensions`).

**SH-2.** The width is **always four digits** — not a minimum, not a
per-series choice. One width everywhere means lexicographic order equals
numeric order (so a directory listing or object-store prefix listing is
already sorted), and it means no reader, validator, or glob ever has to
discover which width a series chose. Four digits caps a series at 10,000
shards; at a 1 GiB shard that is 10 TiB, and a series that outgrows it
wants a larger shard, not more digits.

**SH-3.** Indices start at `0000` and are contiguous. A gap is an error,
never a silently-shortened facet.

**SH-4.** A facet with exactly **one shard collapses**: it is written in
the existing single-file form — a plain `source:` string naming the
file, with no `NNNN` field, no `shard_stride`/`shard_count`, and no
array. That is the canonical spelling and the only one a producer may
emit. A sharded declaration therefore always describes **two or more**
shards, and `shard_count >= 2`.

**SH-70.** The collapse is what preserves **backwards** compatibility.
Every dataset in circulation today is one-shard, so every one of them is
already in canonical form and stays valid, unedited, indefinitely.
Sharding adds a spelling for data that needs one; it does not re-spell
data that does not — and by SH-95 there is no existing data that needs
one, so the set of declarations this feature rewrites is empty.

**SH-71.** It is also what makes **forwards** compatibility a question
about *facets* rather than about *descriptions*. A reader either
supports sharded facets or it does not — it never has to support a
sharded description of an unsharded facet. A producer that wrote
`[base_vectors.fvec=1000000]` for a single file would break every older
reader and buy nothing; this rule removes that failure entirely rather
than documenting it.

**SH-72.** Producers **must** collapse. Validators **must** report a
non-canonical single-shard declaration (`NonCanonicalSingleShard`).
Readers **accept** one and resolve it as the single-file facet it
describes, in memory and without rewriting anything. The asymmetry is
deliberate: reject where it can still be fixed — at creation, and at
check — and tolerate where refusing would help nobody, at read.

**SH-5.** An empty facet is a plain empty file in the canonical
single-file form. There is no empty *series*: an empty facet has one
shard, and one shard collapses. Discovery never has to distinguish "no
shards" from "cannot list", because a facet always names at least one
file directly.

**SH-6.** `StandardFacet::classify` must strip a trailing
`__<4 digits>`
from the basename before matching, so `base_vectors__007.fvec`
classifies as `BaseVectors`. The existing `IDXFOR__` prefix-stripping
and the shard-suffix stripping compose:
`IDXFOR__base_vectors__007.ivvec.i32` classifies as `BaseVectors`.

## 4. How the series is declared

This is the load-bearing decision. Locally you can list a directory;
remotely you cannot. Sequential probing (`__0000`, `__0001`, … until a
404) costs one round trip per shard and cannot distinguish "end of
series" from "transient 404".

**SH-7.** The series is **declared**, not probed, in one of two forms.
`dataset.yaml` — and the catalog entry that stands in for it — is
authoritative; the files are validated against it, never the reverse.
It is the canonical map of content and windows for every open, local or
remote, and there is no path around it (SH-78, SH-79).

### 3a. What the collapse does and does not change

**SH-73.** The collapse is a rule about **declarations**, not about
machinery. Internally a single-file facet is a one-shard uniform series
resolved by the same `Shards` and the same `OrdinalMap::Uniform` as any
other — there is no second, unsharded read path to keep in step, and no
class of bug that can afflict one and not the other.

What must not follow is a *cost*. The one-shard case is the overwhelming
majority of every access this library serves, and it must add **no
indirection per access** relative to direct addressing. An
implementation may specialize it to guarantee that. This is SH-54's
principle pointed at the common case rather than the exotic one.

**SH-74.** A sharded declaration must **fail loudly** in a reader that
predates sharding — never read as something else. Both spellings already
have this property: an array `source` is a type error where a string was
expected, and `base_vectors__NNNN.fvec` names a file that does not
exist. Neither can be mistaken for valid data. Any future addition to
the sharded declaration must preserve it; a field that an older reader
would silently ignore while still resolving the facet is the one shape
this design must not grow.

A sharded dataset also declares `format_version: 2`
([srd-dataset-format-version.md](srd-dataset-format-version.md), V-7),
which turns that failure from a symptom into a diagnosis for every
reader built from that point on. It does **not** help the readers that
already exist — they do not know to look for the field — so SH-74's
loud-failure requirement stands on its own and is not weakened by
having a version.

### 4a. Uniform form

**SH-49.** `source` is a string carrying the `NNNN` field, with
`shard_stride` and `shard_count` giving the layout. Every shard but the
last holds exactly `shard_stride` ordinals. This is the form the
pipeline generates, and the form that must stay cheapest.

### 4b. Explicit form

**SH-50.** `source` is an **array** of shard entries in ordinal order,
and the filenames are arbitrary. This is the form for data produced
elsewhere and handed to `veks` as-is: it does not follow the `__NNNN`
convention, and requiring it to be renamed before it can be read would
make the convention a tax on importing rather than a convenience for
generating. Shard lengths in this form may be **non-uniform**.

Its motivation is **foreign** data, not legacy data. Nothing this
project has already written needs it (SH-95); what needs it is a corpus
someone else chunked, by their rules, that should be readable without
being rewritten first.

**SH-51.** An entry is a **source string in the grammar this project
already parses** — `parse_source_string` / `DSSource` — not a new type.
That grammar already carries a path, an optional namespace, and an
optional ordinal window with SI suffixes. A shard entry is exactly that,
which means the explicit form introduces no parser, no schema, and no
second spelling of "a file and a range within it".

**SH-61.** An entry therefore has four spellings, in increasing
specificity:

| Spelling | Cardinality | Use |
|---|---|---|
| `a.u8` | discovered by opening the file | local, whole file |
| `a.u8=4194304` | declared | remote-safe, whole file |
| `a.u8[0..1M]` | implied by the interval | a slice |
| `a.u8[0..1M]=1M` | implied *and* declared | a slice, cross-checked |

The `=<count>` suffix is taken from the end, accepts the same SI
suffixes intervals do, and is bound by two parse restrictions.

**A source containing `?` is never split on `=`.** The `=` is the
key/value separator inside a URL query string, so
`https://h/f.fvec?token=12345` would otherwise read as a path with a
count of 12345 — a wrong answer that looks like a right one. The
restriction costs nothing, because such a source declares its
cardinality by window instead:
`https://h/f.fvec?token=12345[0..1M]`. That spelling is available to
every source, so no entry is left unable to state its length.

**An `=` whose tail does not parse as a count stays in the path.**
`weird=name.u8` is a filename, not a malformed count, and a grammar
that raised on it would make an unrelated character a parse error.
Recognition is therefore positive — a count is taken only when one is
unambiguously present — rather than a rule the path has to escape.

**SH-62.** The `=<count>` suffix is an **edifying count**: redundant by
construction, and that is the point. It documents an entry's
cardinality where a reader would otherwise have to do arithmetic on
seven-digit bounds, and it catches a typo in either bound.

What it is checked against depends on whether the entry carries an
interval, and the two claims differ in strength:

| Entry | `=<count>` asserts | Checked against |
|---|---|---|
| `a.u8=N` | the file holds exactly `N` records | the file's actual cardinality |
| `a.u8[x..y]=N` | the slice is `N` records | `y - x` |

The first is the stronger claim — it pins the whole file, so a file that
later grows or shrinks is caught. The second only pins the slice; that
the slice *fits* is SH-42's separate window check. A disagreement in
either case is `SliceCountMismatch`, never a silent preference for one
number over the other. This is SH-8's principle at
entry scale — declared, redundant, checked — and together the two give a
mistyped bound two independent chances to be caught: once against its
own entry's count, once against the series total.

A query-string source (SH-61) reaches the same guarantee through the
windowed row rather than the bare one: `f.fvec?t=1[0..1M]` is checked
against its interval exactly as `a.u8[0..1M]` is. What it cannot have is
the *whole-file* claim of `a.u8=N`, since that spelling is the one the
restriction withholds. A series that needs a file's total pinned and
whose sources carry query strings states the window explicitly and gets
the weaker-but-sufficient check.

**SH-63.** A window on an entry makes that entry's cardinality
self-declaring, so **a windowed or `=`-counted entry is legal remotely**
with no file access. In an explicit series a bare filename is a
local-only convenience, because resolving it opens every shard before a
single record is read — the exact expense declaration exists to avoid.

The restriction is on the **series**, not on remoteness. A plain
single-file facet — `base_vectors: s3://…/base.fvecs`, which is how
almost every remote facet in existence is written — stays legal bare.
Its reader must open that file to read anything, so learning its count
is the same open rather than an extra one, and there is no per-shard
multiplication to avoid. Requiring a declared count there would break
every remote facet ever written in exchange for nothing.

**SH-52.** Ordinals are assigned by **concatenation in declared order**:
entry order is ordinal order, and the global space is the entries'
lengths laid end to end. An entry's window bounds ordinals **within its
own file**, never within the facet — the two coordinate systems are
distinct and are never mixed (SH-64).

**SH-65.** An entry carries at most one interval. A multi-interval
window on one entry is rejected — not because it means anything
ambiguous, but because it is exactly equivalent to listing the file once
per interval, and one spelling is better than two.

**SH-66.** The same file may appear in more than one entry, at disjoint
or even overlapping windows. This is what "piecing together an ordinal
view" means, and forbidding it would buy no safety: the global space is
dense and gapless whatever the entries read. One file appearing twice
opens one `Storage` — the registry is keyed on the path, and the window
is applied above it.

**SH-56.** Every entry has length `>= 1`, whether implied or declared,
with **no exception**. A zero-length entry is rejected: it contributes
no ordinals and puts two shards at the same prefix-sum boundary. An
empty facet is not a series at all — it has one shard, and one shard
collapses to a plain file (SH-4, SH-5).

### 4c. Record count

**SH-8.** The declaration carries the **total record count**, and it
must match what the shards actually contain. Declaring it is what lets a
consumer know a facet's length without opening anything, which is what
makes remote planning cheap; checking it is what keeps the declaration
honest. A mismatch is `RecordCountMismatch` — never a silent preference
for one number over the other.

**SH-9.** The derived count is
`(shard_count - 1) * shard_stride + count(last shard)` in the uniform
form, and the sum of per-shard counts in the explicit form.
`count(last shard)` comes from the last shard alone by the format's
existing rule (file size / stride for uniform xvec and scalar; index
length for vvec). This derivation is a **check on the declaration**, not
the source of truth for it.

**SH-53.** The match is verified eagerly by `veks check` and at
creation. At open time it is verified **opportunistically** — when a
shard is opened for other reasons — and never by fetching something
solely in order to check. A declaration is trusted for planning and
validated by use.

**SH-10.** A sidecar manifest is **not** required. Rejected because it
adds a third place the same facts live (declaration, files, manifest)
and two of them can disagree. A facet series that must be usable
without its `dataset.yaml` is out of scope; that is what a catalog
entry is for.

## 4d. Where the shape is resolved

**SH-85.** Declaration shape is resolved **in the dataset serde layer**,
at deserialization. Loading a `dataset.yaml` yields one realized model —
`Shards { map, entries }` (SH-69) — whatever form the declaration took:
uniform or explicit, pinned or bare, sharded or collapsed. There is
precedent and it is the same shape of problem: `DatasetConfig`'s
hand-written `Deserialize` already runs `apply_default_inheritance`, so
no downstream consumer has to know that a sized profile inherited its
`base_vectors` from `default`. Shard shape resolves in the same place
for the same reason.

**SH-86.** Downstream stages **never branch on declaration form**.
Planning, prefetch, the readers, `push`, and the CLI all consume the
realized model. No stage re-parses `source`, and no stage asks whether a
facet was written as a string or an array — by the time any of them see
it, that question has no answer left to give. A second parse anywhere is
a second opinion about what the data is.

**SH-87.** Bare-name resolution happens **once, at load, in that
layer** — never per plan and never per access. It is local-only
(SH-63); a bare name in a remote series fails at deserialization, before
any planning stage has been entered, so the error names the declaration
rather than surfacing later as an unexplained fetch.

**SH-90.** Both dataset deserializers must realize the shape
**identically**. `DatasetConfig` (`model.rs`) and `DSProfileGroup` (the
catalog path) already have to agree about profile inheritance, and the
comment in `model.rs` marking that duplication is a standing warning:
two loaders that drift produce a dataset that reads one way through
`TestDataGroup::load` and another through a catalog. Shard realization
must be shared code, not mirrored code.

## 5. Ordinal algebra

**SH-11.** When the shard lengths are **uniform**, the lookup is
**O(1)** — the standard division and remainder against the shard size,
and nothing else:

```
shard(o) = o / stride
local(o) = o % stride
```

Only when the lengths are genuinely uneven does the lookup become a
binary search over prefix sums:

```
shard(o) = upper_bound(starts, o) - 1
local(o) = o - starts[shard(o)]
```

**SH-68.** Uniformity is a property of the **lengths**, not of the
declaration form. An explicit series whose entries are all the same
length — the last permitted to be shorter — is uniform, and takes the
O(1) map. The declaration form chooses the *spelling*; the lengths
choose the *mapping*. Anything else would penalize a perfectly regular
series for having been written as a list, which is precisely the
combination an importer produces.

**SH-69.** Ordinal mapping is therefore separated from source
resolution. The map answers "which shard, and how far into it"; the
entry table answers "which file, and how far into that". Both are
indexed by shard number, so either map arm serves arbitrary per-shard
sources:

```rust
enum OrdinalMap {                                    // shard(o), local(o)
    Uniform  { stride: u64, count: u32, total: u64 },  // O(1)
    Explicit { starts: Vec<u64>, total: u64 },         // O(log n)
}

struct Entry { source: DSSource, file_base: u64, len: u64 }
struct Shards { map: OrdinalMap, entries: Vec<Entry> }
```

`starts` has length `count + 1`; `file_base` is the entry window's lower
bound, and is `0` for a whole-file shard.

**SH-54.** The uniform case must not pay for the uneven one. It
allocates no prefix-sum table and performs no search — two integer
operations, exactly as if the uneven case did not exist. The arms are
siblings of one dispatch resolved once at load, never a general path
with a fast case tested for on every lookup.

**SH-55.** The uneven case precomputes prefix sums once at load:
`starts[0] == 0`, `starts[n] == total`, strictly increasing. Lookup is a
binary search over `starts`. Building it costs the declared or implied
lengths and nothing else — no shard is opened to construct the map,
provided every entry is windowed or `=`-counted (SH-63). A uniform
series never builds it at all.

**SH-64.** Resolving an ordinal is therefore **three coordinate levels**,
and they are never conflated:

```
global ordinal  o
   ↓  shard(o), local(o)                     — SH-11
local ordinal   l   within shard s
   ↓  + entries[s].file_base                 — the entry's window lower bound
file ordinal    f   within the file s is drawn from
   ↓  format's record→byte rule              — §6
byte offset     within that file
```

For the uniform form and for whole-file explicit shards `file_base` is
zero and the middle step vanishes. It is one addition, not a second
mapping.

**SH-12.** In the uniform form every shard except the last holds exactly
`stride` ordinals and the last holds `1..=stride`; a short interior
shard is `ShortInteriorShard`, not a cue to switch to a scan. In the
explicit form lengths are arbitrary but **declared**, and a shard whose
actual length differs from its declared `records` is
`ShardRecordCountMismatch`. Non-uniformity is legal exactly where it is
declared, and nowhere else.

**SH-13.** A record never spans a shard boundary. For variable-length
formats this constrains the writer, not the reader.

**SH-67.** A facet may carry a profile-level `window:` *and* have
entries that carry windows of their own. They compose in one direction
only, and the order is fixed: **entry windows build the facet's ordinal
space; the facet window then selects within that space.** A profile
window is never interpreted against a file. Stating the order matters
because both are spelled with the same grammar, and a reader who assumes
the other order gets a plausible, wrong answer.

**SH-14.** A window `[a..b)` decomposes into per-shard sub-windows.
Writing `base(s)` for the first global ordinal of shard `s`
(`s * stride` uniform, `starts[s]` explicit) and `len(s)` for its
length, the decomposition is one expression for both forms:

```
for s in shard(a)..=shard(b-1):
    lo = max(a, base(s))     - base(s)
    hi = min(b, base(s)+len(s)) - base(s)
```

Each sub-window then maps to bytes by the format's existing rule
against that shard.

## 6. Per-format rules

All three rules apply to a shard's **file ordinals** (SH-64), which for
a whole-file shard are its local ordinals and for a sliced one are
offset by the window's lower bound.

**SH-15.** *Uniform xvec* — every file carries its own 4-byte dim
header. All files in a series must agree on `dim` and element width. A
disagreement is `SeriesFormatDisagreement`, detected on open, never
tolerated.

**SH-16.** *Scalar packed* — no header; a file ordinal maps to
`file_ordinal * elem_size`.

**SH-17.** *Variable-length (vvec)* — each **file** has one
`IDXFOR__<filename>.<i32|i64>` sidecar whose offsets are **local to that
file** (SH-82), which for the uniform form means
`IDXFOR__<basename>__<NNNN>.<ext>.<i32|i64>`. This is a net improvement
over the single-file case: a window touching one shard loads one file's
index, not the whole facet's — and two shards slicing one file load that
index once.

**SH-18.** *Slab* — a slab **shards like any other format**, and needs
no change to slabtastic to do it.

An earlier draft of this SRD excluded slabs on the grounds that "a slab
is a container with its own page structure, and splitting one is a
slab-format problem." That reasoning was wrong twice over. A slab is
*already an ordinal-addressed container*: each page footer carries a
start ordinal, and a trailing pages-page indexes
`(start_ordinal, file_offset)` for `O(log n)` lookup. And the explicit
form does not split anything — it composes whole files (SH-50). A series
of slabs composes exactly as a series of xvec files does, with SH-64's
file-ordinal level handing the slab reader an ordinal it already knows
how to resolve.

What stays out of scope is **splitting an existing slab file** into
several. That is a slab-format operation — rewriting pages and the
index — and this SRD cuts nothing.

**SH-96.** Shards of a slab series carry **relative** ordinals: each
shard is an ordinary slab based at zero, and the global base comes from
the shard map's `file_base` (SH-69).

The format would permit the alternative. `SlabReader::get_ref` resolves
`ordinal - page.start_ordinal` and bounds-checks within the page, so a
slab based at 4,194,304 reads correctly today; only a fresh
`SlabWriter` assumes zero, and the append path already continues from
`max_ordinal`. Absolute ordinals are rejected anyway, because they would
put ordinal identity in **two** places — the map and the page footers —
and a shard whose footers disagreed with the declaration would have two
defensible answers. SH-10 refused a second source of truth for exactly
this reason, and SH-78 makes the map the only one.

**SH-97.** A sharded slab's namespace selector is spelled as it always
was: `metadata_content__0000.slab:mnodes`. The namespace follows the
path, the shard index sits inside the filename, and neither parse
reaches into the other.

**SH-98.** An embedded `layout` namespace does **not** travel into a
sharded content slab. Per
[metadata-facets-and-layout-namespace.md](metadata-facets-and-layout-namespace.md),
the standalone `metadata_layout.slab` is authoritative and the embedded
copy is a byte-identical convenience. A sharded content facet omits it
rather than duplicating a schema across every shard or placing it
arbitrarily in shard `0000` — either of which would invent a rule about
where a schema lives that the unsharded case never needed.

**SH-95.** Splitting is not merely out of scope — **it is a case that
cannot arise.** A file that exists was creatable; a file that was
creatable was under whatever limit applied; so no existing file needs
splitting. A dataset that would have required sharding could not have
been produced in the first place, which is precisely the gap this design
closes.

Sharding is therefore **generative, not retroactive**. It enables
datasets that could not previously be made; it does not re-lay-out
datasets that already exist, and there is no corpus in the problematic
state waiting to be migrated. Earlier drafts asked which metadata
encodings were in use, as though existing data had to be surveyed before
this design could be trusted. It does not: whatever exists is, by
construction, small enough not to need this.

The one shape that looks like a counterexample is a file that meets a
*newer, tighter* limit than the one it was written under — a filesystem
artifact later published to an object store with a smaller per-object
cap. Even there the answer is not splitting: it is deriving a sharded
series from it, which SH-38 already covers as a copy. In-place division
is never the mechanism.

**SH-19.** *Parquet* — excluded, consistent with
[prefetch-windows.md](prefetch-windows.md).

## 7. Sidecars

**SH-20.** Sidecars are **per file**, never per series and never per
shard: one `.mref` and, for vvec, one `IDXFOR__` beside each file. Each
file is independently verifiable and independently re-fetchable, which
is most of the point of sharding. Two shards sliced from one file
(SH-66) share that file's sidecars — there is one index over the file's
records, and both shards read it at their own offsets.

**SH-21.** A missing sidecar degrades every shard drawn from that file,
per the rules already in [prefetch-windows.md](prefetch-windows.md) — a
planning path must not walk-download a file to price it.

**SH-82.** A vvec `IDXFOR__` covers its **file's** records, in file
ordinals. A sliced shard reads the sub-range of that index its window
names; it does not want, and must not be given, a re-based index of its
own. This is what makes slicing free: no sidecar is rebuilt to support
it.

## 8. Access layer

**SH-22.** A sharded facet presents the **same reader surface** as a
single-file one: `VectorReader<T>`, `VvecReader<T>`, `TypedReader<T>`.
Callers that never ask about layout never learn it exists.

**SH-23.** `count()` is the series total; `dim()` is the shared dim.

**SH-24.** `get(o)` resolves and reads from the owning shard.

**SH-25.** `get_slice(o)` returns `Some` only when the record lies
wholly inside one resident, mmap-promoted **file**. A record never
spans a shard boundary (SH-13) and a shard never spans a file, so the
two conditions coincide — the requirement names the file because that is
what the memory belongs to. It **must not** synthesize a contiguous
slice across files — there is no such memory.
This is the one place the abstraction is permitted to leak, and it
leaks in the safe direction (an honest `None`).

**SH-26.** Shard storages open lazily. Reading ordinal 0 must not open
shard 400.

**SH-59.** Open **files** are capped, with LRU eviction, and the cap is
**derived from the process file-descriptor limit** rather than
hardcoded. It counts files, not shards (SH-81): two shards sliced from
one file hold one descriptor between them, and capping shards would
under-use the budget on exactly the layout slicing exists to serve. A fixed constant is wrong in both directions — it strands
descriptors on a generous host and thrashes on a constrained one, and it
is invisible to the operator who raised `ulimit -n` precisely to make
this work. The derivation leaves headroom for everything else the
process holds open (transport sockets, other facets, sidecars); the
fraction is an implementation constant, the limit itself is not.

**SH-99.** A whole-facet accessor that **cannot fail** must not answer
`0` for a facet it cannot size. A declared shard that will not open is a
broken facet, and `0` is indistinguishable from an empty one — the
silent shape this design exists to forbid.

The fallible form is the real accessor: it propagates the reason and
names the file. The infallible one delegates to it, logs the failure,
and returns `0` — which is the only answer its signature allows, and is
why every caller that can report a failure uses the other. This applies
wherever a series is summarized behind a signature with no error
channel.

**SH-27.** `is_complete()` means **every byte this facet can address is
resident** — not every byte of every file it draws from. `precache()`
drives exactly that set. A file referenced by two shards is fetched once
and reported once.

**SH-92.** The distinction only bites for sliced shards, and there it
decides whether the feature is usable. A facet slicing a tenth of a
large file would, under whole-file completeness, report incomplete
forever unless the other nine tenths were downloaded — bytes it can
never address, fetched to satisfy a predicate about them. Addressable
completeness is the only reading under which `precache()` of a sliced
facet costs what the facet is worth.

For whole-file shards the two readings coincide, so the common case is
unchanged. Where they diverge, the addressable range is what
`is_complete()`, `precache()`, and `PrefetchPlan::is_resident()` all
mean — one definition across the three, because a caller that precaches
until complete and then plans must not be told there is still work.

## 9. Prefetch and windowing

**SH-28.** `PrefetchPlan` becomes shard-aware. Byte ranges must be
qualified by which shard they are in — a bare `(start, end)` pair is
meaningless across a series. The reported totals (`bytes_to_fetch`,
`facet_bytes`, `overfetch_bytes`, `is_resident`) are aggregates over the
**files** the window touches, so a file two shards reach into is counted
once and not twice.

**SH-29.** `prerequisite_bytes` sums only the indices of the **files**
the window actually touches, counting a file once however many shards
reach into it.

**SH-30.** `degrades_to_full_download` aggregates disjunctively: if any
touched shard cannot be windowed, the plan degrades — but to the
**files** those shards are drawn from, not to the whole series. The
distinction is the point: degrading is a statement about bytes that must
move, and bytes live in files (SH-81). Consent is still gated by
`WholeFacetFallback`, and the refusal must name the bytes at stake,
which are now a subset — and, for a sliced shard, may exceed what the
shard itself reads, because a whole file is the smallest thing that can
be fetched (SH-84).

**SH-31.** Fetches across shards may proceed concurrently. Sharding is
the first structure in this codebase that makes a single facet's window
naturally parallel.

**SH-60.** That concurrency is governed by the **existing transport
saturation controls**, not a new knob. It is parallelism across
**files** — a file referenced by two shards is fetched once. Shard
parallelism and the
per-transfer parallelism already in the transport layer contend for one
connection pool and one link; two independent limits would multiply
rather than compose, and the operator would have to reason about their
product to predict load. One budget, spent across whatever work is
outstanding.

## 10. Storage, cache, transport

**SH-32.** Each **file** is one `Storage` instance and one cache file,
under the existing layout `<cache_root>/<dataset>/<relpath>`. This is
not a layout change — the files a series names are ordinary files. Two
shards sliced from one file share that instance: the registry keys on
the canonical path, and the view layer strips the window before opening,
so one file resolves to one `Storage` without anything new being
built.

**SH-33.** The cache-relpath collision guard applies: two *facets*
resolving to one cache path is rejected before any byte lands. Two
entries of one facet naming one file is not that case — see SH-80.

**SH-34.** Range support, no-range fallback, and merkle promotion are
per **file** and may differ between the files of one series. A series
assembled from files with different publishing histories — the explicit
form's reason to exist — will routinely be mixed.

**SH-93.** A facet nevertheless reports **one** access mode, and it is
the **weakest** among its files. A facet's mode is a promise about every
read it will serve, and a caller that plans against "supports range" and
then reaches a no-range file has been misled by an average. Reporting
the weakest link makes the promise true for every shard, at the cost of
understating the good ones — which is the safe direction, because
understating costs a caller some efficiency while overstating costs it
correctness.

Per-file detail remains available to `describe` and the explore TUI
(SH-46), where it is information rather than a promise.

## 11. Creation

**SH-35.** The pipeline writes the **uniform form** only, rolling over
at exactly `shard_stride` records. The stride is an input, never
inferred from memory pressure or output size. The explicit form
describes data this project did not write (SH-50); nothing generates
it.

**SH-83.** If the output would be a single shard, the producer emits the
**single-file form** instead — no `NNNN`, no stride, no count (SH-4,
SH-72). A pipeline run that happens to fit in one shard must not leave
behind a declaration that older readers cannot open.

**SH-36.** Output is deterministic: the same input records and the same
stride produce byte-identical shards.

**SH-37.** Each file is written atomically (temp + rename), and the
`dataset.yaml` declaration is written **after** every file and sidecar
is durable. A reader must never see a declaration promising shards whose
files are not there.

**SH-38.** Derivation from a sharded source to a sharded output may
re-stride. Re-striding is a copy, not a rename.

## 12. Publication

**SH-39.** `push` publishes every **file** the series names and every
sidecar, and lists them all in `SHA256SUMS`. A file referenced by two
shards is published once.

**SH-89.** A published declaration is **serialized from the realized
model** (SH-85), so it carries stated cardinalities by construction —
not because a pinning step ran over it. `push` already turns a local
declaration into a published one; emitting what the loader realized is
that same transformation, not a new one. There is no separate pinning
pass anywhere in the system, and the user's own file is never rewritten.

**SH-100.** Publication is **filesystem-driven**, so shards need no
special handling to be published: they are ordinary files and are picked
up, checksummed, and listed by the same walk that finds every other
file. What does need handling is the *inverse* — a partial write must
not ship. Shard temps therefore carry a suffix the publish walk already
excludes, so a temp surviving a killed run cannot be published as if it
were a shard.

**SH-84.** A sliced series publishes its files **whole**. A window
selects which ordinals a facet exposes, not which bytes exist — the file
is the unit of publication, verification, and transfer, and cutting it
down to a window would invalidate its `.mref`, its `IDXFOR__`, and any
other facet slicing the same file. Publishing more bytes than a facet
reads is the honest cost of reading a file in place instead of copying
it, and a producer that wants only the window should derive a new file
(SH-38) rather than truncate a shared one.

**SH-40.** A partially-published series is invalid. Publication must
either complete or leave the prior series intact — the declaration is
published last, mirroring SH-37.

**SH-41.** For the **uniform** form, catalog entries describe the series
— basename pattern, stride, count — and never enumerate shard URLs,
which are derivable and would be a second source of truth. For the
**explicit** form the entry list *is* the description: there is no
pattern to derive from, so the catalog carries the entries verbatim,
windows and counts included. In both cases the catalog says exactly what
`dataset.yaml` says, because it stands in for it (SH-78).

## 13. Validation

**SH-42.** `veks check` is local (SH-76) and verifies, without reading
record payloads:

*Uniform form:*

- every index in `0000..shard_count` is present
- no index beyond `shard_count` is present (a stale extra shard is an
  error, not ignored)
- every index is exactly four digits
- every interior shard has exactly `shard_stride` records
- the last shard has `1..=shard_stride`
- the basename classifies to a standard facet after suffix-stripping

*Explicit form:*

- every listed file is present
- every entry's window lies within its file's actual cardinality
- every `=<count>` matches what it asserts (SH-62): the interval's
  length where there is one, the file's cardinality where there is not
- no entry resolves to zero length (SH-56)
- no entry carries more than one interval (SH-65)
- every entry is windowed or `=`-counted when any shard is remote
  (SH-63)
- prefix sums are strictly increasing and end at `record_count`

*Both forms:*

- a facet with one shard is in canonical single-file form (SH-4);
  a sharded declaration names at least two shards
- the declared `record_count` matches the derived total (SH-8)
- dim and element type agree across the files the series names
- required sidecars exist per file (SH-20)

**SH-88.** `veks check` **verifies; it does not repair.** It compares
the declaration against the files and reports mismatches — it never
rewrites a declaration, and never emits a corrected one to be pasted
back. Rewriting a user's config is not a validator's job, and a
validator that fixed what it found would make its own clean run
meaningless.

**SH-43.** Violations are reported per shard with the index named — and
per file where the fault is the file's (a missing sidecar, an absent
path) rather than the shard's. All violations are reported, not just the
first.

## 14. CLI and YAML surface

**SH-44.** Every knob is reachable from both surfaces with congruent
names:

| `dataset.yaml` | CLI |
|---|---|
| `shard_stride: 1000000` | `--shard-stride 1M` |
| `shard_count: 12` | *(derived on write; validated on read)* |

**SH-45.** Facet-selecting flags (`--facets`) name the facet, never a
shard. There is no CLI surface for "precache shard 7" — that is what a
window is for.

**SH-46.** `describe`, `list`, and the explore TUI present a sharded
facet as one facet. Shard count is always an attribute; **stride is
reported only when the lengths are uniform** (SH-68) — a non-uniform
series has no stride, and printing one would be a number that does not
exist. The drill-down names per-shard ordinal ranges and per-file
residency, which are different columns for the same reason SH-81 keeps
them apart.

## 15. Declaration shape

**Uniform** — what the pipeline generates:

```yaml
profiles:
  default:
    base_vectors:
      source: base_vectors__NNNN.fvec  # NNNN marks the shard field
      shard_stride: 1000000
      shard_count: 12
      record_count: 11412003
```

**Explicit, whole files** — data produced elsewhere, read where it
lies. The `=` counts make each entry self-declaring, so this is legal
remotely with no file access:

```yaml
    metadata_content:
      source:
        - corpus-part-a.u8=4194304
        - corpus-part-b.u8=4194304
        - corpus-tail.u8=918211
      record_count: 9306819
```

**Explicit, sliced** — an ordinal view pieced together from parts of
existing files, without copying a byte. `corpus-b.u8` appears twice, at
disjoint windows (SH-66):

```yaml
    metadata_content:
      source:
        - corpus-a.u8[0..1M]=1M
        - corpus-b.u8[500K..1500K]=1M
        - corpus-b.u8[3M..3250K]=250K
      record_count: 2250000
```

**Explicit, local shorthand** — bare names, cardinality discovered by
opening each file. SH-63 permits this only when nothing in the series is
remote:

```yaml
    metadata_content:
      source: [ corpus-part-a.u8, corpus-part-b.u8, corpus-tail.u8 ]
      record_count: 9306819
```

**SH-47.** The literal token `NNNN` in a string `source` marks the shard
field. It is the only accepted spelling — there is no width to choose
(SH-2), so there is no `NNN` or `NNNNN` form. Its presence is what makes
a string-form facet sharded; `shard_stride`/`shard_count` without it is
an error, and `NNNN` without them is an error. One spelling, no
inference.

**SH-57.** The two forms are mutually exclusive. An array `source`
carrying `shard_stride` or `shard_count` is `MixedShardDeclaration` —
the array already says everything those fields would.

**SH-58.** `record_count` is required in both forms (SH-8).

## 16. Error taxonomy

**SH-94.** The taxonomy below is **normative and shared**, not a
suggestion per implementation. It extends the existing error types
rather than introducing a parallel one: declaration faults — anything a
`dataset.yaml` states wrongly — surface at load through the dataset-load
error, and resolution faults surface through `IoError`, which is where
every other open-time failure already arrives. Every variant carries the
shard index or the file path it concerns, because a message that says
which is the difference between a fixable report and a puzzle.

This matters beyond one implementation: the Java port mirrors this
taxonomy, and two error sets that disagree make the same broken dataset
diagnosable in one runtime and inscrutable in the other.


| Error | Raised when |
|---|---|
| `MissingShard(i)` | declared index absent |
| `UnexpectedShard(i)` | index present beyond `shard_count` |
| `ShortInteriorShard{i, expected, actual}` | interior shard ≠ stride |
| `OverlongLastShard{actual, stride}` | last shard > stride |
| `SeriesFormatDisagreement{i, field}` | dim/elem type differs |
| `ShardIndexWidth{i}` | index is not exactly four digits |
| `ShardDeclarationIncomplete` | `NNNN` without stride/count, or vice versa |
| `RecordCountMismatch{declared, derived}` | declared total ≠ what the shards hold |
| `SliceCountMismatch{i, declared, implied}` | `=<count>` ≠ the interval's length |
| `WindowExceedsFile{i, window, cardinality}` | entry window runs past its file's end |
| `EmptyShardEntry{i}` | entry resolves to zero length |
| `MultiIntervalShardEntry{i}` | entry carries more than one interval |
| `MixedShardDeclaration` | array `source` alongside `shard_stride`/`shard_count` |
| `NonCanonicalSingleShard` | a sharded declaration describing one shard |
| `UnboundedRemoteShardEntry{i}` | bare filename in a remote series |
| `ShardCacheCollision{a, b, relpath}` | two files of one series share a cache path |

## 17. Invariants

1. `stride > 0` (uniform form).
2. Uniform form: interior shards are exactly `stride` records and the
   last is `1..=stride`. Explicit form: each shard's length is its
   window's length, and that window lies within its file.
3. Indices are contiguous from `0000` (uniform form); entry order is
   ordinal order (explicit form).
4. All shards share format, dim, and element type.
5. No record spans a shard boundary.
6. The declaration is authoritative; files are validated against it.
7. Global ordinal space is dense and gapless.
8. The declared `record_count` equals the sum of shard lengths.
9. Uniform lengths map in O(1) — division and remainder — allocating
   nothing and searching nothing, however the series was spelled.
10. Entry windows are in **file** ordinals; a profile window is in
    **facet** ordinals. The two are never interpreted against each
    other (SH-67).
11. Every `=<count>` equals the length it annotates. It is never the
    source of a length, only a check on one.
12. A sharded declaration describes two or more shards. One shard is
    always spelled as a single file, so every dataset written before
    sharding existed is already canonical.
13. The map is the only source of shard identity and window. Nothing
    is discovered by listing, probing, or filename inference.
14. A raw file open yields a file. A facet — with an ordinal space, a
    window, and possibly a series — comes only from the map.

## 18. Non-goals

**SH-91.** A dataset is an **immutable artifact**. Nothing in this
design mutates a facet in place: shards are not appended to, not
truncated, not re-strided without a copy, and not replaced beneath a
reader. Deriving a differently-shaped series is a new series (SH-38),
and publishing is write-once (SH-40). This was assumed throughout rather
than stated, and stating it forecloses a class of question — cache
coherence under mutation, readers observing a series mid-rewrite,
`record_count` drifting under a live handle — that the design otherwise
appears to leave open.


- Variable stride within the *uniform* form. Non-uniform lengths are
  the explicit form's job (SH-50), and mixing the two spellings is
  `MixedShardDeclaration`.
- Resharding in place.
- Splitting any existing file, slab or otherwise — a case that cannot
  arise (SH-95). Composing a series of files, slabs included, is
  supported.
- Retroactively re-laying-out datasets that already exist.
- Sharded parquet.
- Records spanning shards.
- Discovery by probing.

## 19. Interaction with parameterized profiles

A facet belonging to a parameterized profile may itself be sharded, and
generators build filenames from the profile name — giving names like
`postfiltered_neighbor_indices__sel001__0000.ivecs`.

**SH-101.** A token placed before the shard field must never be
**all-digits**. In the uniform form the shard field is always last
before the extension and always four digits (SH-1, SH-2), so
`…__0010__0000.ivecs` has two readings and neither is decidable. This
binds any generator that interpolates a profile name into a filename —
see
[srd-profile-parameterization.md](srd-profile-parameterization.md), P-9.

The constraint is specific to the uniform form, because that is the only
one whose filenames are *parsed* for a shard field. Explicit-form
filenames are read from the declaration and never parsed, so they carry
no naming constraint at all — including none on digits.

## 20. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | window inside one shard | one shard's ranges; others untouched |
| 2 | window spanning two shards | correct sub-windows, no gap or overlap at the seam |
| 3 | window spanning all shards | equals whole-facet byte total |
| 4 | window ending exactly on a boundary | no empty trailing sub-window |
| 5 | `get(o)` across every boundary | matches unsharded reference |
| 6 | `count()` | `(n-1)*stride + last` |
| 7 | `get_slice` across a boundary | `None`, never a synthesized slice |
| 8 | vvec series | per-shard `IDXFOR__` used; only touched shards' indices load |
| 9 | missing interior shard | `MissingShard`, not a short facet |
| 10 | short interior shard | `ShortInteriorShard` |
| 11 | dim disagreement | `SeriesFormatDisagreement` on open |
| 12 | remote series, one shard resident | plan reports only the rest |
| 13 | sharded == unsharded | same records, same order, byte-identical reads |
| 14 | 1000+ shards | fd cap respected; LRU eviction |
| 15 | empty facet | single empty `__0000`, `count()==0` |
| 16 | conformance | every violation reported, per shard |
| 17 | explicit series, non-uniform lengths | window spanning uneven seams resolves exactly |
| 18 | uniform vs explicit over identical records | indistinguishable through every reader |
| 19 | declared `record_count` wrong | `RecordCountMismatch`, no silent preference |
| 20 | explicit entry length ≠ declared | `ShardRecordCountMismatch` |
| 21 | explicit + remote, counts omitted | `UndeclaredRemoteShardCounts` before any fetch |
| 22 | array `source` + `shard_stride` | `MixedShardDeclaration` |
| 23 | uniform mapping | O(1): no allocation, no search — asserted on `OrdinalMap` directly |
| 32 | explicit series, all entries equal length | collapses to the O(1) map, not binary search |
| 33 | every pre-sharding fixture in the suite | reads unchanged, byte-identical, no declaration edited |
| 34 | producer given one shard's worth of data | emits the single-file form, not a one-element series |
| 35 | one-element sharded declaration | `NonCanonicalSingleShard` from check; readers still resolve it |
| 36 | single-file hot path | no added indirection per access vs. pre-sharding |
| 37 | consumer built on the unsharded plan shape | fails to build or parse; never reads a range as facet-wide |
| 38 | fd cap below transport concurrency | reported at startup, not silently throttled |
| 39 | facet open with the map absent | fails; no directory listing or URL probe substitutes for it |
| 40 | `IndexedVvecReader::open` on one shard file | opens that file only — no series, no window, no facet ordinals |
| 41 | two entries naming one file | accepted; the relpath guard does not fire |
| 42 | bare-name and pinned declarations of one dataset | realize to identical models |
| 43 | `DatasetConfig` vs `DSProfileGroup` load | same dataset realizes identically through both |
| 44 | bare name in a remote series | fails at deserialization, before any planning |
| 45 | `veks check` on a valid dataset | reports, rewrites nothing; file mtime unchanged |
| 46 | `push` of a bare-name local series | published declaration states its counts |
| 47 | sliced facet, window resident | `is_complete()` true without the rest of the file |
| 48 | `precache()` on a sliced facet | fetches the addressable range, not the whole file |
| 49 | series mixing range and no-range files | facet reports the weakest mode |
| 50 | slab series | composes and reads like any other series; no slabtastic change |
| 51 | sharded slab, namespace selector | `x__0000.slab:mnodes` resolves both parses |
| 52 | slab shard ordinals | relative per shard; base comes from the map, not the footers |
| 53 | every dataset predating this feature | unchanged on disk and in declaration; no migration step exists |
| 54 | series wider than the descriptor budget | reads; open files never exceed the cap |
| 55 | sliced facet, window resident | complete; precache asks for the window, not the file |
| 56 | shard temp from a killed run | excluded from publication |
| 57 | derive with a stride | shards, sidecars, and a declaration that reads back |
| 58 | derive that fits one shard | single-file form; no shard fields declared |
| 24 | explicit prefix sums | strictly increasing, end at `record_count` |
| 25 | sliced entries | windowed view equals the same records copied into whole files |
| 26 | one file, two disjoint windows | both resolve; one `Storage` opened |
| 27 | `=<count>` disagrees with its interval | `SliceCountMismatch` |
| 27a | `=<count>` disagrees with its file's cardinality | `SliceCountMismatch` |
| 27b | source with a query string | `=` not split; window still declares the count |
| 27c | `=` whose tail is not a count | stays in the path; no parse error |
| 28 | entry window past end of file | `WindowExceedsFile` |
| 29 | bare filename in a remote series | `UnboundedRemoteShardEntry` before any fetch |
| 29a | bare single-file remote facet | legal; opens once as it always has |
| 30 | profile window over a sliced series | selects in facet ordinals, not file ordinals |
| 31 | multi-interval entry | `MultiIntervalShardEntry` |

**SH-48.** Tests 13 and 18 are the anchors: one fixture built as a
single file, as a uniform series, and as an explicit non-uniform series
must be indistinguishable through every reader and every window. Whatever
the layout says, the ordinal space is the same space.

## 21. Settled questions

Four things that could have cost real work. Each is now a requirement
rather than an open question.

**SH-75.** **The `PrefetchPlan` shape change must break loudly.**
SH-28 makes byte ranges shard-qualified, and that type shipped in 1.9.0
and is mirrored by the Java port. The change is accepted and the port is
trued up immediately after this lands. What the change must *not* be is
quiet: a consumer built against the unsharded shape has to fail to
compile, or fail to parse, and never read a shard-qualified range as a
facet-wide one. A bare `(start, end)` in a sharded world is not a
degraded answer, it is a wrong one, so there is no compatibility shim
worth having here.

**SH-76.** **`veks check` is a local operation.** It may open every shard
and every sidecar, because there is no network to economize against. The
cost argument behind SH-7 is about *remote resolution* — round trips to
discover what a declaration could have stated — and it does not transfer
to validation. The two must not be conflated: a validator that refused
to open files in the name of a rule about round trips would be enforcing
the letter of SH-7 against its purpose. A published dataset is validated
against a local copy, before push or after fetch.

**SH-77.** **The open-file cap is a backstop, never a throttle.** The open-file cap
derived in SH-59 is sized **above** the transport's maximum fetch
concurrency (SH-60), which is the reasonable assumption on any host this
runs on. If a host's descriptor limit is low enough that the cap would
bind on concurrency, that is a misconfiguration and must be **reported
at startup** — never absorbed by silently reducing parallelism, which
would surface as unexplained slowness with no line of output pointing at
`ulimit`.

**SH-78.** **`dataset.yaml` is the canonical map of content and windows.**
It — or the catalog entry standing in for it — is fetched, cached, and
used as the authority for which files exist, which windows they carry,
and what ordinal space they compose. Every facet open, local or remote,
resolves through it. Nothing discovers a shard by listing a directory,
probing a URL, or inferring structure from a filename; the `__NNNN`
convention is a *naming* rule for files the map already named, never a
substitute for the map.

**SH-79.** **The programmatic APIs offer no bypass.** Everything reached
through `TestDataGroup` / `TestDataView` goes via the map. The
direct-file constructors — `XvecReader::open`, `IndexedVvecReader::open`,
`TypedReader::open` — open **a file, not a facet**: they have no ordinal
space beyond that file, no window, and no shard series, and they must
not become a way to assemble one. Drawing the line at *file versus
facet* keeps a raw open honest instead of turning it into a second,
weaker resolver that would then have to be kept in step with the first.

**SH-80.** Because every entry resolves through one authority, the
cache-relpath collision guard (SH-33) sees the whole picture. Two
entries naming one file is a **declared relationship**, not a collision
between independent facets, and the guard must tell them apart — the
collision it exists to catch is two *facets* claiming one cache path,
which the map can still detect.

## 22. Decisions

**Settled.** Shard indices are exactly four digits (SH-2). The
open-shard cap derives from the process fd limit (SH-59) and is sized
above fetch concurrency (SH-77). Shard fetch concurrency reuses the
existing transport saturation controls (SH-60). The plan-shape change
breaks loudly and the Java port trues up after (SH-75). `veks check` is
local (SH-76). The map is canonical and unbypassable (SH-78, SH-79).
Each is a requirement above rather than a question here.

Nothing remains open, and nothing awaits a survey of existing data.
Sharding is generative (SH-95): it widens what can be produced, and
every dataset already in circulation is one-shard, canonical, and
untouched by any of this (SH-4, SH-70).

### Settled — nothing pins, because nothing rewrites

The question was *when* to rewrite a bare-name declaration into a
counted one. It dissolves: **nothing rewrites at all.**

The serde layer resolves the shape at load (SH-85), so every stage above
it already sees stated cardinalities whether or not the file on disk
spelled them out. There is no unresolved shape in the system to
normalize later, and therefore no moment at which a rewrite would be the
fix.

That leaves each participant with one job:

| | Does |
|---|---|
| **serde layer** | resolves the declaration into one realized model, once, at load |
| **planning / prefetch / readers** | consume that model; never re-parse `source` (SH-86) |
| **`veks check`** | verifies the declaration against the files; repairs nothing (SH-88) |
| **`push`** | serializes the realized model, so the published declaration states its counts by construction (SH-89) |
| **producers** | write counts they already know |
| **the user's file** | is never rewritten by any of them |

The bare-name form stays exactly what SH-63 says it is — a local
convenience that costs an open per file, paid once at load rather than
per plan. Its only real cost was that it might silently survive into a
published artifact, and SH-89 closes that: publication serializes the
realized model, so the published form is stated regardless of how the
local one was written.

Explicitly rejected: caching derived counts in the config directory.
That is a third place the same facts live, which SH-10 already turned
down for the same reason.
