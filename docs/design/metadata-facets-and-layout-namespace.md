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
