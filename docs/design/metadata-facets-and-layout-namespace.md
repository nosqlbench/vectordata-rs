# Metadata facets, the facet↔file model, and the layout namespace

## Why this exists
A latent inconsistency in the metadata facets surfaced via
`e2e_partition_profiles_full_pipeline` (see
`docs/analysis/e2e-partition-profiles-failure.md`). Fixing it forced a
clarification of the facet model and a concrete plan for storing the
metadata **layout (schema)** as a namespace inside the metadata slabs.

## The corrected model (design rulings)

1. **A facet is a logical identity, not a single file.** A facet resolves
   through its **declared view** to a physical locator, which may be a
   standalone file *or* a **namespace within a shared slab**. It may own
   **more than one file** (e.g. a data file plus its `IDXFOR__…` sidecar)
   and be reachable under more than one basename (canonical + legacy).

2. **View locator syntax is `path#namespace`.** A view value of
   `metadata_content.slab#layout` addresses the `layout` namespace of that
   slab. No `#` means the whole file.

3. **`metadata_results`** is the **R facet** — the per-query
   predicate-match index. Its file is named canonically
   `metadata_results.{ivvecs,slab,…}`; the legacy name `metadata_indices.*`
   is also resolved for extant datasets. `metadata_indices` is a recognized
   alias of `metadata_results`.

4. **`metadata_layout`** is the **metadata field-schema** facet — *optional*
   and exposed through the `vectordata` API. It is **not** a standalone
   artifact and is **not** the R index. The earlier
   `metadata_layout => metadata_indices` basename mapping was a bug and is
   removed.

5. **The layout is stored as a `layout` namespace inside the metadata
   slabs — in both the content slab and the results slab.** We do **not**
   force the metadata facets into one file; each slab keeps its own
   content *and* carries a copy of the schema.

6. **Compatibility = byte-for-byte schema match.** Because both
   `metadata_content.slab#layout` and `metadata_results.slab#layout` carry
   the schema, a content slab and a results slab are compatible iff their
   `layout` namespaces are byte-identical. This is the design driver.

7. **The schema is opaque `anode` bytes.** The namespace stores the schema
   as bytes; each caller decides whether to decode it with an `anode`
   implementation. The facet exposes raw bytes.

The slab format already supports this: `slabtastic` is multi-namespace
(footer `namespace_index`, `NamespacesPage`; default namespace = index 1 =
`""`, so existing single-namespace files are unaffected). Writer:
`start_namespace("layout")` / `append_namespace`; reader:
`open_namespace(path, Some("layout"))`.

## Stage 1 — naming correctness + namespace-aware resolution (DONE)

Implemented and verified (`cargo test --workspace --no-fail-fast`: 2445
passed, 0 failed):

- `veks-pipeline/dataset_lookup.rs`: `metadata_results` owns its file under
  candidate basenames `["metadata_results","metadata_indices"]` with the
  index extensions (`ivvecs`/…/`slab`) — the missing `ivvecs` here was the
  original reason R was unreachable. `metadata_layout`'s bogus
  `metadata_indices` basename removed. `facet_present` iterates basename
  candidates and strips `#namespace`. `lookup_facet` is now **alias-aware**
  (a view keyed `metadata_indices` satisfies a `metadata_results` lookup)
  and parses/preserves `path#namespace`. Anchors for
  `PostfilteredKnnConsolidated`, `PredicateResults`, `PredicatesConsolidated`
  and `verify_predicates` repointed from `metadata_layout` → `metadata_results`.
- `vectordata/dataset/facet.rs`: `metadata_indices` registered as an alias
  of `MetadataResults`.

No files were renamed and no schema is written yet — the facet now simply
*owns both names*, so extant datasets keep working while the canonical name
is established.

## How the design fork was found and resolved

**The `vectordata` layout API** (`vectordata/src/dataset/layout.rs`):
- `LAYOUT_NAMESPACE` (`"layout"`), `read_layout_bytes(locator)` (reads the
  opaque schema bytes from a `path` / `path#namespace` slab locator; a
  missing namespace is `Ok(None)`, not an error), and
  `layouts_compatible(a, b)` (the byte-for-byte content↔results / dataset↔
  dataset compatibility test — the design driver, ruling 6). 6 unit tests.
  This realizes rulings 6 & 7: schema exposed as raw bytes; compatibility =
  byte match.
- The producer (`gen_metadata`) writes the `layout` namespace into the
  content slab on the `generate metadata` slab path.

**The fork surfaced while wiring the producer/view side.** The
`layout` namespace, as written today, only survives to the
`metadata_content.*` path in the **non-self-search synthesize-slab** case.
It is **absent** in the other two common paths:

1. **Self-search (`extract-metadata`).** `gen_extract` rebuilds the content
   slab with plain `SlabWriter::new` / `SlabReader::open` (default namespace
   only) — sibling namespaces are not carried across. This is the same
   limitation `derive::materialize_slab` documents for windowed slicing:
   **windowed/rebuilt slab derives drop sibling namespaces.**
2. **Imported / converted metadata.** `convert-metadata` (and identity
   symlink) producers do not write a `layout` namespace at all.

So emitting an unconditional `metadata_content.slab#layout` view would
**dangle** in exactly the dominant paths. **Resolution (chosen): option B —
the standalone `metadata_layout.slab` is the authoritative home.** The
content-slab `layout` namespace remains a byte-identical convenience copy
where it naturally survives; the standalone file is slicing-proof and
unaffected by content rebuilds.

## Stage 2 — layout namespace + compatibility (DONE, standalone design)

Implemented and verified (`cargo test --workspace --no-fail-fast`: **2462
passed, 0 failed**):

1. **The authoritative schema is a standalone `metadata_layout.slab`.** Its
   *default* namespace holds the single schema record (the whole file *is*
   the layout). `gen_metadata` gained a `layout-output` option (role
   `Output`, declared in `project_artifacts`) that writes it from the same
   `field_N` schema backing every content format — so the layout is emitted
   even when content is a flat scalar/`ivec`. The in-content `layout`
   namespace is kept as a byte-identical convenience copy on the slab path.
2. **The `metadata_layout` facet view** is declared by `import` as
   `…/metadata_layout.slab` (bare locator → default namespace) — but only
   when the `generate metadata` step actually produces it (gated on
   `metadata_all.step_id() == "generate-metadata"`), so it never dangles for
   imported/converted metadata. Conformant under the facet spec
   (`MetadataLayout` accepts `Slab`).
3. **Reading** is `vectordata::dataset::layout::read_layout_bytes(locator)`
   — honours the locator's namespace (default for the standalone bare path,
   `#layout` for the embedded copy). Schema exposed as raw bytes (ruling 7).
4. **Compatibility** is `layouts_compatible(a, b)` — byte-for-byte (ruling
   6). The producer guarantees the standalone and the embedded copy are
   byte-identical by construction (same `metadata_layout_bytes`), so a
   content slab and the standalone layout always compare equal.

**Contradiction fixed in passing.** Wiring this surfaced that
`--synthesize-metadata` did *not* imply the `M` facet: `resolve_facets` only
inferred `MPRF` when metadata or `G` was present (despite its own comment
"metadata can be synthesized"), and the `--provided-facets` validator
demanded an `M` *input* even under synthesis. Both now honour
`synthesize_metadata`, so `--synthesize-metadata` alone produces the
metadata chain.

**Canonical file rename (DONE).** R files are now emitted under the
canonical `metadata_results.*` name, and the dataset.yaml view key is
`metadata_results`. Legacy reading is fully retained — the facet spec lists
`metadata_indices` as a legacy basename, the resolver/verifiers probe
canonical-first then legacy, the `model.rs` config field accepts
`metadata_indices`/`predicate_results` as serde aliases, and the public
`TestDataView` method was renamed `metadata_indices()` → `metadata_results()`
(a deliberate public-API change; callers adjust). The `synthetic-1k/1m`
fixtures and `typed_access.rs` still pass unchanged on the legacy name,
proving backcompat. The verifier probes are now driven by
`StandardFacet::MetadataResults.basenames()` (single source of truth) rather
than hardcoded names. Surfaced & fixed in passing: `--synthesize-metadata`
now implies the `M` facet (see above).

**Embedded-copy survival across slab derives (DONE for `materialize_slab`).**
Windowed `derive::materialize_slab` now carries sibling namespaces forward
verbatim (windowing applies only to the default content namespace; metadata
namespaces like `layout` are copied whole). This also resolved an internal
contradiction — the function's own doc already *claimed* namespace
preservation while a test documented the opposite; code and doc now agree.

**Remaining (optional follow-on):**
- **`extract-metadata` (`gen_extract`)** uses a bespoke partition/reorder
  slab writer and does not carry the embedded `layout` namespace across.
  Harmless under the standalone design (the standalone `metadata_layout.slab`
  is authoritative and is written directly by `generate metadata`, not via
  extract; the `metadata_layout` facet view points at the standalone, never
  the embedded copy). Carrying it through the partition writer would make the
  content slab's convenience copy survive self-search too, but adds no
  correctness.

## Stage 3 — standardized facet↔resource spec + conformance enforcement (DONE)

The facet↔file model above is now **standardized in the `vectordata`
crate as the single authority**, and enforced. This makes the design
rulings mechanically checkable rather than convention:

> "We still require, as a matter of design, an explicit set of possible
> mappings for each facet type… standardized in vectordata so that we can
> always tell what facet a file or resource goes with, or how to look for,
> given a facet, what files or resources it may contain."

**The spec — `vectordata/src/dataset/facet.rs`:**
- `FacetFormat` enumerates the coarse on-disk shapes (`FloatXvec`,
  `IntegerXvec`, `IntegerVarXvec`, `ScalarPacked`, `Slab`) and owns the
  extension↔format mapping (`extensions`, `from_extension`).
- `StandardFacet` gains the authoritative spec methods: `formats()` (which
  shapes a facet may take), `basenames()` (canonical + legacy filenames it
  may own), `namespaces()` (e.g. `metadata_layout` → `["layout",""]`),
  `accepts_format`/`accepts_extension`, and `classify(name)` (given any
  resource path, return the `(facet, format)` it belongs to — strips dir,
  `#namespace`, and `IDXFOR__` sidecar prefix).

  This directly answers the design questions: *"Can I store metadata in an
  integer xvec file?"* → `MetadataContent.accepts_format(IntegerXvec)`;
  *"Does this file belong to `metadata_content`?"* → `classify(path)`;
  *"What resources may facet R contain?"* →
  `MetadataResults.basenames() × formats().extensions() × namespaces()`.

**The resolver consumes the spec — `veks-pipeline/dataset_lookup.rs`:**
The duplicated, divergent `canonical_basenames_for` /
`canonical_extensions_for` tables (the root cause of the original
three-name R-facet drift) are **deleted**; `facet_present` now derives the
filesystem probe entirely from `StandardFacet::basenames()` ×
`formats().extensions()`. There is one source of truth.

**Enforcement is a check-time gate —
`vectordata/src/dataset/conformance.rs` + `veks/src/check`:**
- `validate_conformance(&DatasetConfig) -> Result<(), Vec<FacetViolation>>`
  verifies every profile view whose key resolves to a standard facet
  declares a resource whose format the facet permits (custom keys and
  templated/synthetic locators are out of scope and skipped).
- Wired into `veks check` as the **`facet-conformance`** category. **Load
  stays lenient** (a mid-pipeline dataset may declare facets not yet
  produced); the strict gate runs when the dataset is meant to be complete.
- Guarded by `import_generated_dataset_conforms_to_facet_spec` — the
  generator↔spec agreement test proving `import` emits conformant YAML.

Full workspace green after Stage 3: `cargo test --workspace
--no-fail-fast` = **2453 passed, 0 failed**.

## Stage 4 — the reader (DONE)

Stages 1–3 settled where a metadata facet's bytes live, which namespace
is authoritative, and how conformance is enforced. What none of them
provided was a way to **read** one through this crate. The vector
readers in `vectordata::io` are built on fixed-width elements; a slab
record is neither fixed nor an element run, so every route refused it —
`facet()` answered *"cannot infer element size from extension '.slab'"*
for a facet Stage 3's own spec calls conformant.

The codecs were not the gap. Both stages of the record pipeline were
already present and public:

```text
slab ──[container]──▶ &[u8] ──[stage 1]──▶ ANode ──[stage 2]──▶ text ──[serde]──▶ T
        Stage 4               formats::anode      formats::anode_vernacular
```

**The container is `vectordata::records`.** `RecordFacet` resolves a
facet ordinal to the container holding it and that container's local
ordinal, then asks slabtastic. For a single file there is one container;
for a series there is one per shard, in ordinal order.

**Currying, not a second implementation.** Applying a codec to a facet
produces a typed reader:

```rust
let facet = view.open_facet_records("metadata_content")?;

facet.decode(Anode)                    // get(o) -> ANode
facet.decode(Text(Vernacular::Cql))    // get(o) -> String
facet.decode(Serde::<Row>::new())      // get(o) -> Row
```

The untyped level is `Records<Anode>` — the codec that stops after stage
1 — rather than a path of its own. A codec is a **value**, not only a
type, so one named in a setting reaches the same `decode` as one written
in a type signature; `codec_by_name("cql")` resolves through the same
`Vernacular::parse` every other by-name surface uses. `Serde<T>` routes a
record through the JSON vernacular into any `Deserialize` target, so a
caller names its own struct and this crate knows nothing about it.
`record_bytes(ordinal) -> &[u8]` sits beneath all of it for anything that
wants to decode some other way.

### Rulings this stage adds

8. **The dialect comes from the record, never from the facet.**
   `MNode::to_bytes` writes `DIALECT_MNODE` and `PNode::to_bytes_named`
   writes `DIALECT_PNODE` as the leading byte, and those buffers are what
   the producers hand to `add_record`. The same container holds MNodes in
   content position and PNodes in predicate position; deriving the
   dialect from the facet table would put record identity in two places
   when the bytes already carry it. A facet holding a mix reads without
   the caller declaring which is which.

9. **A namespace is a facet of its own.** `facet.namespace("schema")`
   returns a `RecordFacet` over the same containers under a different
   name, so the schema sidecar, the `layout` convenience copy and the
   `survey` report are one operation against three names rather than
   three special cases.

10. **An absent namespace reports empty, not broken.** Ruling 2 makes the
    embedded `layout` copy optional and several producers never write
    one. The container opens the file first — so a real I/O or format
    failure stays a failure — and lets only the namespace probe answer
    "absent", which is the separation `dataset::layout` already made and
    for the same reason.

11. **Access is incremental, by the same means as every other format.**
    Reads go through `Storage`, not a memory map of the reader's own. A
    slab ends with a pages-page indexing every data page by start
    ordinal, so opening a facet costs its tail and reading a record
    costs that record's page — each fetched and merkle-verified as a
    byte range by the same chunked source the vector readers use.

    An earlier cut of this reader took the other road: it called
    `SlabReader::open(path)`, which memory-maps, and then required the
    facet to be fully resident so the map could not fall on a sparse
    cache file. That made a slab a bulk-download-only format and broke
    the incremental-access property every other type has — for a
    reason that was really only "the mmap constructor was one line".
    The sparse-map hazard is an argument against mapping a partial
    file, not for downloading a whole one. Nothing in slabtastic
    required it: `Footer::read_from`, `PagesPage::deserialize`,
    `NamespacesPage::deserialize` and `Page::get_record_from_buf` are
    all public and take byte slices.

12. **Shape is a property of the data, and is exposed as one.** Some
    facets hold runs of a fixed-width element and some hold opaque
    records of their own length. `FacetShape` says which, derived from
    the format by the spec that already owns formats, and
    `view.facet_shape(name)` answers it before anything is opened — so
    a consumer handling both branches on it rather than trying a reader
    and interpreting the failure.

    Everything that does not depend on what a record *is* stays common
    to both: ordinal count, windows, shards, residency, precache cost,
    merkle verification. Only the record differs, and only the record's
    reader differs with it.

13. **Wrong-door errors name the right door.** Opening a record facet
    as vectors raised *"cannot infer element size from extension
    '.slab'"* — true, a description of the symptom, and no help. It now
    raises `Error::WrongFacetShape`, carrying the facet, the shape it
    holds, the shape attempted, and the reader that opens it — a value
    to branch on rather than prose to parse. The mirror holds: an
    element facet brought to `open_facet_records` is refused by name
    instead of failing as a slab footer parse.

### What this unblocks

A **sharded** metadata facet now reads through the same surface, which
makes `srd-multifile-facet-shards.md`'s slab requirements testable in
this crate rather than only in slabtastic:

- **SH-96** — shards carry relative ordinals. Each shard is an ordinary
  slab based at zero and the global base lives only in the shard map;
  the reader proves it by answering facet ordinal 20 with the record that
  shard 2 calls its own ordinal 0.
- **SH-98** — an embedded `layout` namespace does not travel into a
  sharded content facet. Asked for one, the facet reports nothing, from
  every shard alike.
