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

## Stage 2 — the layout namespace + compatibility (TODO)

1. **Write** the schema into a `layout` namespace of the metadata content
   slab and the metadata results slab (`slabtastic` `start_namespace` on
   fresh writes, `append_namespace` when augmenting an existing slab). The
   bytes are the `anode`-encoded schema, written **identically** to both.
2. **Declare** the optional `metadata_layout` facet view as
   `…/metadata_content.slab#layout` (and/or the results slab). `vectordata`
   exposes it as raw bytes (caller decodes via `anode`).
3. **Read** via the namespace-aware locator (the `#namespace` plumbing from
   Stage 1) → `SlabReader::open_namespace(path, Some("layout"))`.
4. **Compatibility check**: a content↔results compatibility test compares
   the two `layout` namespaces byte-for-byte.
5. **Canonical file rename** (optional follow-on): emit new R files as
   `metadata_results.*` instead of `metadata_indices.*`. Not required for
   correctness — the facet already owns both names — but completes the
   canonicalization. Carries the only remaining backcompat surface, so do
   it deliberately with the suite as guard.

Stage 2 touches the metadata production path (the slab writers) and the
`anode` schema bytes; it is feature work on top of the Stage-1 cleanup.
