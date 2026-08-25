# Prefetch: arbitrary windows

**Status:** proposal, not implemented. Nothing in this document is built.

## What is true today

The byte-range plumbing already exists and is already exercised:

```rust
FacetStorage::prebuffer_range_with_progress(byte_start, byte_end, cb)
Storage::ensure_range(first_chunk, last_chunk)
```

Chunked storage (`MerkleHashed`, `MerkleChunked`) fetches per chunk, and
the chunk bitmap already records what is resident. Reads outside a
fetched range already work — they fault the covering chunk in on demand.
So this is not new transport work, and it is not a new sparse-access
capability.

**What does not exist is a way for a caller to name the window.** The
only caller of the range path is `view::facet_window_byte_range`, which
returns a range only when *both* of these hold:

1. the facet's **configuration** declares a window — either the
   `base.fvec[0..1M)` suffix on `raw_source`, or a sibling `window:`
   key; and
2. the format is uniform-stride xvec, so record→byte is
   `4 + dim × elem_size`.

Anything else falls back to downloading the whole file. A caller holding
a `TestDataView` cannot say "I am about to read records 5M..6M — fetch
those" unless a profile was defined ahead of time that says so.

That is the gap. The window is a property of the dataset when it should
be able to be an argument from the reader.

## What this proposes

A **prefetch** surface that takes a caller-supplied window, in the
coordinate system the caller is already thinking in, for any facet and
any format.

### The window is the type that already exists

`DSWindow` is already a list of half-open record intervals with a
settled syntax:

```
[0..1M)              one interval
[0..1K, 5K..6K]      several
[..1M)               open start
```

Prefetch takes a `DSWindow`. No new spelling, no second grammar.

> **A claim I made here was wrong and is retracted.** I wrote that
> `facet_window_byte_range` ignoring intervals past the first was a bug.
> It is not. `open_uniform` — the *reader* — reads `window.0[0]` too,
> and `WindowedVectorReader` holds `start + len`, so it cannot express a
> disjoint window at all. Precaching only the first interval therefore
> fetches exactly what the reader will expose. `parse_window_first`
> documents the restriction deliberately and tells authors to split
> disjoint windows into separate facet configs. Nothing to fix.
>
> Taking a `DSWindow` for *prefetch* is still the right call, because
> prefetch has no `WindowedVectorReader` in the way — it resolves
> intervals to chunk ranges and unions them. But that makes prefetch the
> **first** consumer able to honor a multi-interval window, which is a
> new capability rather than the repair of an old defect, and it means
> `window: "[0..1K, 5K..6K]"` would mean one thing to a reader and
> another to a prefetch. See open question 5.

### Core API

On `TestDataView`, addressing a facet by name:

```rust
/// What a prefetch would fetch, without fetching it.
fn prefetch_plan(&self, facet: &str, window: &DSWindow) -> Result<PrefetchPlan>;

/// Fetch the window and return when it is resident.
fn prefetch(&self, facet: &str, window: &DSWindow) -> Result<PrefetchReport>;

/// Same, with chunk-level progress.
fn prefetch_with_progress(
    &self,
    facet: &str,
    window: &DSWindow,
    cb: &mut dyn FnMut(&PrebufferProgress),
) -> Result<PrefetchReport>;
```

And on the readers, for the common case of a reader you already hold:

```rust
impl XvecReader        { fn prefetch(&self, window: &DSWindow) -> Result<PrefetchReport>; }
impl IndexedVvecReader { fn prefetch(&self, window: &DSWindow) -> Result<PrefetchReport>; }
impl TypedReader       { fn prefetch(&self, window: &DSWindow) -> Result<PrefetchReport>; }
```

The reader form is the one most callers want: you are iterating a range,
you know the range, you say so.

### Plan before fetch

`PrefetchPlan` answers "what will this actually cost" before anything
moves, which matters because the unit of fetch is a chunk, not a byte.

**The accessors this needs now exist** (`FacetStorage::range_fill` →
`RangeFill`), so the plan is assembly rather than new capability:

```rust
pub struct RangeFill {          // implemented
    pub first_chunk: u32,
    pub last_chunk: u32,
    pub chunk_size: u64,
    pub chunks: u32,
    pub chunks_resident: u32,   // range-aware, not whole-file
    pub aligned_start: u64,
    pub aligned_end: u64,
}
// with chunks_to_fetch(), bytes_to_fetch(),
// overfetch_bytes(requested_start, requested_end), is_resident()
```

The range-aware residency count is the part that did not exist before:
whole-file `valid_count` cannot answer "is *this window* already warm",
because a file 90% resident says nothing about whether the 10% you are
about to read is the missing part.

`PrefetchPlan` is then one `RangeFill` per interval, plus the format's
prerequisite bytes and the degrade flag:

```rust
pub struct PrefetchPlan {
    pub requested: DSWindow,
    pub byte_ranges: Vec<Range<u64>>,
    pub fills: Vec<RangeFill>,
    pub prerequisite_bytes: u64,      // vvec index; 0 for xvec
    pub degrades_to_full_download: bool,
}
```

### Byte coordinates as the escape hatch

```rust
fn prefetch_bytes(&self, facet: &str, ranges: &[Range<u64>]) -> Result<PrefetchReport>;
```

Records are the default because that is what users and the existing
window syntax speak. Bytes are there for callers who have already done
their own offset arithmetic, and for formats where record coordinates
are unavailable.

## Records to bytes, per format

Decided. The three formats get three different answers, and two of them
are deliberately narrow.

### xvec — exact, no prerequisite

`4 + dim × elem_size`, with `dim` read from the 4-byte header. That read
pulls one chunk on remote storage, which is the same first-chunk fetch
any reader does on first access, so it is not a new cost.

**Status: implemented.** `facet_window_byte_range` already does this; it
needs splitting so the record→byte half can take a caller-supplied
window instead of a config-derived one.

### vvec — exact, one whole-index prerequisite

Variable-length records have no computable stride, so the sibling offset
index is the only way to map an ordinal to a byte. Three rules:

1. **The index is required in `dataset.yaml`.** A vvec facet that does
   not name its index is a configuration error, caught at load, not a
   facet that silently degrades to full download.
2. **The reader presumes it is there.** No probing, no fallback path,
   no "does this dataset have an index" branch threaded through the
   readers.
3. **The index is fetched whole.** No differential fetch of the index
   itself. It is flat and small relative to the data, and making the
   *prerequisite* incremental buys little while doubling the number of
   partial-fetch state machines to reason about.

**And the vvec data file should be merkle-published.** Merkle mode
carries integrity *and* download-state management — one mechanism, two
benefits, no reason to separate them or to treat either as the "real"
motivation. A facet published with `.mref` gets a chunk bitmap, and a
chunk bitmap is what windowed fetch needs to know what is already
resident. A vvec facet without one simply has no partial state to track,
so it fetches whole.

### parquet — deferred

No ordinal-windowed prefetch for now. Parquet's row-group and page
structure means a record range snaps outward by an amount the caller
cannot predict, and the footer must be read before even that is known.

What *is* allowed is merkle-based dynamic access for the purpose of
**mapping** ordinals — reading the footer and row-group metadata through
the ordinary chunked path. Only once those offsets are known does a
differential fetch of the resulting byte ranges become possible, and
that is a later step. Until then, `prefetch` on a parquet facet reports
`degrades_to_full_download`.

| Format | Record → byte | Prerequisite | Requires `.mref` | Status |
|---|---|---|---|---|
| xvec | `4 + dim × elem_size` | 4-byte header | for windowing | ready |
| vvec | sibling offset index | whole index file | **yes** | needs index plumbing |
| parquet | row groups via footer | footer | for mapping only | deferred |

## Access modes that cannot window

`AccessMode` already distinguishes these. Prefetch behaves as follows:

| Mode | Behaviour |
|---|---|
| `Local` | no-op; report zero bytes, zero chunks |
| `MerkleHashed` | fetch and verify the covering chunks |
| `MerkleChunked` | fetch the covering chunks |
| full-download | `degrades_to_full_download: true` in the plan |

The last row is the one to decide (see open questions): a window against
a source that cannot range-request is not an error exactly, but silently
downloading 1.3 TiB when the caller asked for 150 MiB is the failure
mode this whole proposal exists to avoid.

## CLI

Mirroring the library, on the existing `precache` command:

```
veks datasets precache -d glove-100 \
    --facet base \
    --window '[0..1M)'

veks datasets precache -d glove-100 \
    --facet base --window '[0..1K, 5K..6K]' \
    --plan
```

- `--facet <name>` — repeatable; default is every facet, as today.
- `--window <spec>` — the existing `DSWindow` syntax. Applies to the
  named facets. Absent means the whole facet, as today.
- `--bytes` — interpret `--window` as byte offsets rather than records.
- `--plan` — print the plan and exit without fetching.

`--plan` output is the `PrefetchPlan` rendered as a table: per facet, the
requested window, the resolved byte range, chunks total and resident,
bytes to fetch, overfetch, and any prerequisite.

## Naming

The codebase currently has three words for overlapping things:

- `precache()` — drive a facet to fully resident
- `prebuffer_*()` — the download driver underneath
- `prefetch` — not used in `vectordata` today

Adding a fourth meaning would be worse than picking. **Proposed:**
`prefetch` becomes the public verb for "make this window resident",
`precache` stays as the whole-facet special case (`prefetch` with an
empty window), and `prebuffer_*` stops being public — it is the
implementation, and `FacetStorage::prebuffer_range_with_progress` is
already a leak of it into the public API.

That would mean deprecating two public methods on `FacetStorage`. Worth
it, but it is an API break and should be a deliberate decision rather
than a side effect.

## Feasibility, reassessed

With the parse fix and the range accessors landed, here is what is left,
sized honestly.

### Ready — xvec windowed prefetch

| Piece | State |
|---|---|
| chunk-range fetch | `prebuffer_range_with_progress`, `ensure_range` — exists |
| range-aware residency | `RangeFill` — **landed** |
| chunk arithmetic | `chunk_span`, unit-tested — **landed** |
| window parsing, degenerate rejection | **landed** |
| malformed vs absent | **landed** |
| record→byte for xvec | exists, welded to config resolution |

**Done.** `record_range_to_bytes` is extracted and both callers use it;
`TestDataView` has `prefetch_plan`, `prefetch` and
`prefetch_with_progress` taking a `DSWindow` in record coordinates.
Seven integration tests cover the ad-hoc path, multi-interval windows,
empty-window-means-everything, clamping past the end, the degrade
report, and parity between an ad-hoc window and a declared one.

Multi-interval windows resolve every interval here, which no other
consumer does — the reader's single contiguous window is a
`WindowedVectorReader` limit, not a fetch limit, and prefetch has no
such structure in the way.

### Moderate — vvec

The index requirement makes this tractable, but it is not free:

- **Config**: a `index:` key on the facet, required for vvec, validated
  at load. New surface in `dataset.yaml` and in the conformance check.
- **Fetch**: pull the whole index before mapping. Needs somewhere to
  live — it is a second file per facet, and the cache layout has to
  hold it.
- **Map**: read offsets, resolve the window, hand byte ranges to the
  same chunk-range fetch xvec uses.
- **Guard**: refuse to window a vvec facet with no `.mref`, since there
  is no chunk bitmap to track partial state with.

The mapping itself is trivial once the index is resident. The work is
plumbing the index as a first-class per-facet artifact.

### Deferred — parquet

Out of scope for ordinal windows. Merkle-based dynamic access for
footer/row-group *mapping* is permitted; differential fetch by ordinal
is a later step gated on that.

### The honest risks

- **The `dataset.yaml` index key is new required surface.** Making it
  required means existing vvec facets that lack it become configuration
  errors. That is the right call for a format that cannot be windowed
  without one, but it is a breaking change to any dataset already
  published without it, and it needs a migration story rather than a
  hard failure on next load.
- **A vvec facet without `.mref` fetches whole.** Not a failure, and
  not a separate mode — just the absence of the chunk bitmap that
  partial fetch tracks state in. Worth reporting in the plan
  (`degrades_to_full_download`) so it is visible rather than inferred.
- **`degrades_to_full_download` still has no agreed behaviour.** See the
  remaining open question.

### Suggested order

1. Extract `record_range_to_bytes`; add `prefetch`/`prefetch_plan` for
   xvec only. End-to-end vertical slice, small.
2. CLI `--facet` / `--window` / `--plan` against that slice.
3. vvec index as required config + conformance check + whole-index
   fetch.
4. vvec windowed prefetch on top.
5. Parquet mapping, then parquet differential fetch, as separate work.

Step 1 is worth doing on its own regardless: it exercises the whole path
and makes the vvec cost concrete before committing to it.

## Open questions — the remaining ones

**Resolved since the first draft:** the per-format policy (vvec requires
its index and `.mref`, index fetched whole, parquet deferred) and the
plan accessors, both above. The comma hazard is fixed and committed.


**1. Resolved — profile windows are conveniences, not fences.**

A profile's `window:` qualifies a range someone wants repeatedly, with
the finesse of a name and a stable definition. It is not the set of
ranges anyone is permitted to ask for. The API takes a caller-supplied
window directly, and the two paths resolve through the same mapping.

This dissolves the congruence question rather than answering it: the
YAML key remains the knob it always was, and an ad-hoc window is an
argument. Nothing is mirrored because nothing new is configured.

**2. Blocking, background, or both?**

Everything above is blocking: `prefetch` returns when the window is
resident. That is really "precache a window", and "prefetch" usually
implies fire-and-forget. A background form —

```rust
fn prefetch_async(&self, facet: &str, window: &DSWindow) -> PrefetchHandle;
```

— is more useful for the actual use case (warm ahead of a scan while the
scan runs) but adds a lifetime, a cancellation story, and a failure
channel that nobody is waiting on. I would ship blocking first and add
the background form only if a real caller needs it. Tell me if you want
both from the start.

**3. What should a window against a non-rangeable source do?**

Options: (a) fetch the whole file, reporting it in the plan;
(b) return an error and make the caller ask for the full download
explicitly; (c) fetch nothing and let reads fault it in later. I lean
(b) — the surprise is the thing worth preventing — but it makes
prefetch fallible in a way that a "just warm it if you can" caller will
find annoying.

**4. Should scattered windows be coalesced?**

Given `[0..1K, 1K..2K]`, obviously merge. Given
`[0..1K, 8M..8M+1K]` against 8 MiB chunks, the two chunks are adjacent
and merging them changes nothing. Given a thousand scattered
single-record windows, coalescing decides whether this is one request or
a thousand. Proposed: merge intervals whose *chunk* ranges touch or
overlap, report the resulting request count in the plan, and leave the
policy at that.

**5. Should prefetch honor multi-interval windows when no reader can?**

Prefetch can trivially union disjoint chunk ranges. Every existing
consumer takes only the first interval, on purpose. So accepting a
multi-interval `DSWindow` at the prefetch API makes the same string mean
two different things depending on which surface reads it.

Options: (a) prefetch honors all intervals and we accept the divergence
as prefetch simply being more capable; (b) prefetch takes a single
interval, matching every other consumer, and disjoint prefetch is spelled
as repeated calls; (c) prefetch honors all intervals *and* the reader
grows a multi-window form so the two agree. I lean (b) for the first cut
— matching the existing convention costs nothing and repeated calls are
not onerous — but (a) is defensible if disjoint prefetch is a real use
case you have.

**6. Fixed — retained for the record.**

Verified, not inferred:

```
parse_window("0,1000")           → [0..0), [0..1000)
parse_source_string("f.fvec[0,1000)") → window [0..0), [0..1000)
parse_source_string("f.fvec[0..1000)") → window [0..1000)
```

A comma where `..` belongs is not rejected. It parses as two intervals,
the first of them degenerate, because a bare number is shorthand for
`0..N`. Downstream the two paths then **disagree**: the reader builds
`WindowedVectorReader::new(_, 0, 0)` and reports `count() == 0`, while
`facet_window_byte_range` sees `win_end <= win_start`, returns `None`,
and falls back to downloading the **entire** file.

So `base.fvec[0,1000)` — which is exactly how someone used to Python
slices or interval notation would write it — yields an empty dataset and
a full-size download, silently. `dataset/expansion.rs` already carries a
comment warning about this and works around it by emitting `..` in the
generated path; the input handling itself was never fixed.

Cheapest honest fix: reject a window whose first interval is empty, at
parse time, with a message naming `..` as the separator. That is a
behaviour change for anything currently relying on a degenerate window
being tolerated, which is why I am asking rather than doing it.
