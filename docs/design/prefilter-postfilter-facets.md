# Pre-Filter and Post-Filter Filtered-KNN Facets

Status: **proposed** — design memo, awaiting sign-off before implementation.
Audience: anyone touching the filtered-KNN pipeline, dataset schema, or
downstream analyze/verify tooling.

## 1. Why this memo exists

The `F` facet — "filtered KNN" — has been carrying two distinct semantics
inside a single name. Today's `compute filtered-knn` produces one of them
(the top-K nearest neighbors *among predicate-passing base vectors*), but
the artifact name is silent about which semantic is on disk. Downstream
consumers — `verify`, `analyze explain-filtered-knn`, the swimlane docs,
the dataset wizard, every place that references the `F` letter — read
whichever shape happens to be there and treat it as authoritative
ground truth for "filtered KNN".

Two problems follow:

1. **There is no separate artifact for the other semantic.** When we want
   to evaluate an ANN engine that implements *post-filter* search (search
   the unfiltered index, then drop predicate failures), the right
   ground-truth target is **not** what's currently on disk. The current
   shape always has K passing neighbors when the predicate is satisfiable
   for K; the post-filter ground truth can be sparse for restrictive
   predicates and is the realistic comparison target for those engines.
2. **The single `F` facet code is ambiguous.** Anything that reads a
   profile keyed by `filtered_neighbor_indices` and reasons about
   "filtered KNN" cannot tell which mode produced it. Documentation,
   diagrams, and verify tooling silently conflate the two.

This memo establishes both modes as **first-class, distinctly named
facets**, with the canonical terminology drawn from the ACORN paper
[Patel et al., 2024], with file layout, producer/consumer contract,
and a phased implementation plan covering every touch point.

## 2. ACORN-grounded terminology

ACORN §3.2 defines the two predicate-agnostic baselines for hybrid
(vector + structured predicate) search. We adopt these definitions
verbatim:

> **Pre-filtering** linearly scans X_p (the subset of base vectors that
> pass the query predicate), computing distances over each point that
> passes the search predicate. This yields a hybrid search complexity of
> O(|X_p|) = O(sn + K). Pre-filtering always achieves perfect recall.
>
> **Post-filtering**, by contrast, performs ANN-search over X to find
> the closest query vector to x_q, then expands the search scope to find
> K vectors that pass the query predicate, p.

For *ground-truth* (brute-force, exact) computation, we further
specialize:

- **Pre-filter ground truth** = top-K of X_p by distance. Equivalent to
  ACORN's G_K — the canonical filtered ground truth. `|result| = K`
  whenever `|X_p| ≥ K`. Always perfect recall.
- **Post-filter ground truth** (no-scope-expansion variant) = the
  unfiltered top-K of X, intersected with X_p. Formally: `top_K(X) ∩ X_p`.
  `|result| ∈ [0, K]`; sparse when the predicate has low correlation
  with the query or low selectivity.

We will store both. They have orthogonal use cases:

- Pre-filter ground truth is the verification target for **any** filtered
  ANN engine that aspires to perfect recall — pre-filter scan engines,
  ACORN-style predicate-subgraph engines, oracle-partition HNSW.
- Post-filter ground truth is the verification target for **post-filter
  ANN engines without rescue scope** (a common naive implementation:
  query the unfiltered index, drop predicate failures, return survivors).
  Comparing such an engine against pre-filter ground truth conflates
  algorithm error with semantic mismatch.

## 3. Facet codes — full table

The repo uses single-character facet codes (concatenated into strings
like `"BQG"` for partition sub-facet scoping in `expansion.rs`). The
revised table:

| Code | Name                                | Description                                                  |
|------|-------------------------------------|--------------------------------------------------------------|
| B    | base                                | Base vectors                                                 |
| Q    | query                               | Query vectors                                                |
| M    | metadata                            | Per-base-vector metadata content                             |
| P    | predicates                          | Per-query predicate                                          |
| R    | predicate-results                   | Per-query base-vector ordinals matching the predicate        |
| G    | knn-ground-truth                    | Unfiltered top-K neighbors of each query                     |
| D    | knn-distances                       | Distances paired with G                                      |
| F    | **prefiltered-knn-ground-truth**    | **Pre-filter top-K: KNN over R, considering full R cardinality** — ACORN G_K, the legacy `filtered-knn` shape |
| **E**| postfiltered-knn-ground-truth       | **Post-filter top-K: `G ∩ R`** (intersection of unfiltered top-K with R) — new sparse artifact |

E and F are paired with indices+distances files (like G/D).

### 3.1 On-disk facet keys

| Facet | Canonical indices key                 | Canonical distances key                 |
|-------|---------------------------------------|-----------------------------------------|
| F     | `prefiltered_neighbor_indices`        | `prefiltered_neighbor_distances`        |
| E     | `postfiltered_neighbor_indices`       | `postfiltered_neighbor_distances`       |

Both pairs are **new** canonical names. The repo previously stored
filtered-KNN output under `filtered_neighbor_indices` /
`filtered_neighbor_distances`; those keys remain recognised as
**backwards-compat aliases for F** (post-filter):

| Facet | Aliases (read-only)                                                                 |
|-------|-------------------------------------------------------------------------------------|
| E     | `prefiltered_gt`, `prefiltered_ground_truth`, `prefilter_indices`, `prefilter_distances` |
| F     | `filtered_neighbor_indices`, `filtered_neighbor_distances` (legacy),  `filtered_gt`, `filtered_ground_truth`, `postfilter_indices`, `postfilter_distances` |

Resolution rules:

- **Producers** always write to the canonical `prefiltered_*` /
  `postfiltered_*` keys. New datasets get the new names.
- **Consumers** resolve E/F by checking the canonical key first, then
  each alias in declaration order. The legacy `filtered_*` alias is
  the last fallback for F, so an existing dataset.yaml referencing
  `filtered_neighbor_indices` keeps working without edits.
- The wizard / file-pattern inference recognises `filtered_*.ivec` files
  during dataset discovery and emits a deprecation note suggesting the
  user re-run with the new canonical names.

**Legacy alias resolution: `filtered_*` → F.** Already-published
datasets carry `filtered_neighbor_indices` / `filtered_neighbor_distances`
as their canonical filtered-KNN artifact. By code inspection of the
legacy `compute filtered-knn` producer (now renamed to
`compute prefiltered-knn`), those files contain pre-filter shape —
ACORN's `G_K`. The legacy keys therefore resolve to **F** (pre-filter):
the on-disk content matches its new typing without any regeneration.

`E` (post-filter, `G ∩ R`) is the new facet. It is *cheap* to derive
from already-computed G and R, so producing it for an existing dataset
is a one-pass derivation that needs no base/query rereads — see §5.2.

Loaders that find a `filtered_*` key emit a one-line note advising
migration to the canonical `prefiltered_*` name (see §5.4 migration
helper). They do **not** refuse to read the file; the rename is opt-in.

### 3.2 Profile YAML example

```yaml
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    query_vectors: profiles/default/query_vectors.fvec
    predicates: profiles/default/predicates.slab
    metadata_content: profiles/default/metadata_content.u8
    metadata_indices: profiles/default/metadata_indices.ivvec     # R
    neighbor_indices: profiles/default/neighbor_indices.ivec      # G
    neighbor_distances: profiles/default/neighbor_distances.fvec  # D
    prefiltered_neighbor_indices:   profiles/default/prefiltered_neighbor_indices.ivec    # F (ACORN G_K)
    prefiltered_neighbor_distances: profiles/default/prefiltered_neighbor_distances.fvec
    postfiltered_neighbor_indices:  profiles/default/postfiltered_neighbor_indices.ivec   # E (G ∩ R)
    postfiltered_neighbor_distances: profiles/default/postfiltered_neighbor_distances.fvec
```

## 4. Relationship to the current implementation

Today's `compute filtered-knn`, by code inspection of
`veks-pipeline/src/pipeline/commands/compute_prefiltered_knn.rs` (in
particular `find_top_k_filtered_*` and the per-partition compute
loops), iterates the predicate-matching ordinals for each query,
computes distances, and keeps a top-K heap — i.e., the algorithm
matches the **pre-filter (F)** definition. The artifact it emits is
stored under the legacy `filtered_*` keys, which now resolve through
the facet-alias table to the F slot — the on-disk content and the
typed slot agree without any regeneration.

We resolve this by:

- **Renaming** the existing producer to `compute prefiltered-knn`,
  which produces **F** (pre-filter). The algorithm is unchanged.
- **Adding** `compute postfiltered-knn`, which produces **E**
  (post-filter, `G ∩ R`) as a new first-class artifact, derived cheaply
  from already-computed G and R.
- **Aliasing** the legacy `filtered_*` keys → F (pre-filter) per §3.1.
  Existing on-disk content matches the F slot's typing, so no
  regeneration is required.
- **Keeping** `compute filtered-knn` registered as an alias to
  `compute prefiltered-knn`, so legacy pipeline.yaml files keep running
  unchanged.
- **Providing** a one-shot helper (`veks dataset migrate-facet-names`)
  that rewrites the dataset.yaml keys from `filtered_*` →
  `prefiltered_*` and renames the on-disk files to match. Opt-in.

To gain the E artifact for an existing dataset, run
`compute postfiltered-knn` against the dataset's G + R — cheap,
single-pass, and writes the canonical `postfiltered_*` files.

## 5. Producers

### 5.1 `compute prefiltered-knn` — produces F (pre-filter, ACORN G_K)

Implementation: lift-and-rename the current `compute filtered-knn` body
into `compute_prefiltered_knn.rs`, retaining its partitioning, cache
key, and SIMD distance code. Cache version tag becomes `pfknn-v1` to
prevent collision with any leftover `fknn-v1` files from the old name.
Output keys: `prefiltered_neighbor_indices`,
`prefiltered_neighbor_distances`. Facet code: `F`. The legacy command
name `compute filtered-knn` is retained as a registry alias to the same
factory.

### 5.2 `compute postfiltered-knn` — produces E (post-filter, G ∩ R)

Output keys: `postfiltered_neighbor_indices`,
`postfiltered_neighbor_distances`. Facet code: `E`.

Implementation: **new**, derives E from already-computed G and R. The
math is `E = G ∩ R`:

```
for each query q:
    G_q ← unfiltered top-K neighbors of q   # from neighbor_indices
    R_q ← base ordinals matching predicate  # from metadata_indices
    E_q ← [o for o in G_q if o ∈ R_q]       # preserve G's distance ordering
    pad E_q to length K with sentinel (-1 / +inf)
    distances for survivors copied from D
```

Properties:

- **No base/query reread.** The producer only opens G, D, R (and the
  metadata-indices reader for membership testing).
- **Deterministic ordering.** Survivors keep their G rank.
- **Cheap.** O(K) per query for the intersection test; orders of
  magnitude faster than `compute prefiltered-knn`.
- **Sparse output is normal.** E rows that don't fill K positions are
  padded with `-1` indices and `+inf` distances, matching the sentinel
  convention used by the prefilter producer.

Distances for E are taken from D (the unfiltered distances file)
directly — no recomputation — and use the same FAISS publication sign
convention as D. There is no per-partition cache (the work is too cheap
to be worth partitioning); the output is written in a single sequential
pass.

### 5.3 Verifiers

`verify filtered-knn-consolidated` splits into two:

- `verify prefiltered-knn-consolidated` — recomputes F from B/Q/R/k and
  checks against stored F. Same shape as today's verify-filtered-knn,
  pointed at the new producer's output.
- `verify postfiltered-knn-consolidated` — recomputes E from stored
  G + R and checks against stored E. Trivially cheap; tests should
  catch any sentinel-padding regression.

Both map to their respective facet codes (`F` / `E`) in `command_facet`.

## 6. Schema and API impact

### 6.1 `vectordata::dataset::facet::StandardFacet`

Rename the existing `FilteredNeighborIndices` / `…Distances` variants to
`PrefilteredNeighborIndices` / `…Distances`, with canonical keys
`prefiltered_neighbor_indices` / `prefiltered_neighbor_distances` and
facet code `F`. Add new `PostfilteredNeighborIndices` /
`PostfilteredNeighborDistances` variants with canonical keys
`postfiltered_*` and facet code `E`. Aliases per §3.1 — including the
legacy `filtered_neighbor_indices` / `filtered_neighbor_distances` keys,
which resolve to **F** (pre-filter; the on-disk shape produced by the
legacy `compute filtered-knn` is pre-filter, so this alias points at the
correct typing without any regeneration). Roundtrip and alias-resolution
tests added, including a regression test that the legacy alias resolves
to F.

### 6.2 `veks-core::formats::facet::Facet`

Parallel enum gets the same rename + new variants and the same facet-
code mapping (`F` for prefiltered, `E` for postfiltered).
`preferred_format()`: indices → `ivec`, distances → xvec-by-element-size.

### 6.3 `vectordata::model::ProfileConfig` and `dataset::profile::DSProfile`

Rename fields `filtered_neighbor_indices` / `…_distances` to
`prefiltered_neighbor_indices` / `…_distances` (F facet). Add new
`postfiltered_neighbor_indices` / `…_distances` fields (E facet). The
new postfiltered fields carry `#[serde(alias = "filtered_neighbor_*")]`
attributes so legacy YAML keys load straight into the appropriate
slot. Profile inheritance honours all four like the G/D fields.

### 6.4 `vectordata::view::TestDataView`

Rename `filtered_neighbor_indices()` / `…_distances()` trait methods to
`prefiltered_neighbor_indices()` / `…_distances()` (F facet). Add new
`postfiltered_neighbor_indices()` / `…_distances()` methods (E facet).
Update the facet→path resolution table to recognise both the canonical
`prefiltered_*` / `postfiltered_*` keys and the legacy `filtered_*`
fallback (legacy → F).

### 6.5 `command_facet()` in `vectordata/src/dataset/expansion.rs`

```rust
"compute prefiltered-knn"
    | "compute filtered-knn"                              // legacy alias
    | "verify prefiltered-knn-consolidated" => Some('F'),
"compute postfiltered-knn"
    | "verify postfiltered-knn-consolidated" => Some('E'),
```

The legacy `compute filtered-knn` command name remains registered as an
alias for `compute prefiltered-knn`. The legacy
`verify filtered-knn-consolidated` keeps mapping to `F`; Phase 3 may
split it into prefilter / postfilter verifiers.

## 7. Consumer impact map

This change reaches every place that touches an F-shaped artifact. The
table below covers the 39 files identified in the survey, grouped by the
update each requires.

### 7.1 Must distinguish E vs F

| File                                                              | Update                                          |
|-------------------------------------------------------------------|-------------------------------------------------|
| `veks-pipeline/src/pipeline/commands/inspect_filtered_knn.rs`     | Single bimodal command. Auto-detects which of F and E the active profile carries; reports per-facet histogram, exemplars, and rank-shift stats for each present facet. Optional `--mode F|E|both` flag forces selection. When both are present the output juxtaposes them so the user can read the post-filter sparsity against the pre-filter full-K result. The current intersection-with-G analysis is literally the E-definition computation (`G ∩ R`), so for any profile that does not yet carry an E facet on disk the analyze command can preview E from G+R directly — no need to run `compute postfiltered-knn` first. |
| `veks-pipeline/src/pipeline/commands/verify_consolidated.rs`      | Split into prefilter and postfilter verifiers. |
| `veks-pipeline/src/pipeline/commands/describe_dataset.rs`         | Enumerate both E and F when present.           |
| `veks-pipeline/src/pipeline/commands/inspect_partition.rs`        | Oracle-partition profiles need to declare which of F/E they carry. Partition profiles are derived from default; if default has both, partitions can carry both, but post-filter inside a partition is degenerate (every base vector passes by construction). For partitions, **F only** by default; E at partition scope is the same data as F. |
| `veks-pipeline/src/pipeline/dataset_lookup.rs`                    | `CommandType` enum gets `PrefilteredKnnConsolidated` and `PostfilteredKnnConsolidated`. |
| `veks/src/prepare/wizard.rs`                                      | New role variants `PrefilteredNeighborIndices`/`Distances`. Wizard prompts add detection patterns for `prefiltered_*` files. |
| `veks/src/prepare/import.rs`                                      | Pipeline scaffolding emits both producers (and both verifiers). |
| `veks/src/prepare/infer_manifest.rs`                              | File-pattern inference recognises `prefiltered_neighbor_*` and updated `filtered_neighbor_*` semantics. |
| `veks/src/prepare/stratify.rs`                                    | Partition-profile construction honours the new pairing (see partition note above). |
| `veks/src/explore/dataset_picker.rs`                              | TUI lists both facets when present. |
| `tools/src/bin/gen_swimlane.rs`                                   | Re-label F (pre-filter) to make the semantic explicit and add an E (post-filter, G ∩ R) column. Today's F tooltip already describes pre-filtering — keep it on F and add a new tooltip for E that describes G ∩ R. |

### 7.2 Schema only (variants/keys)

| File                                                              |
|-------------------------------------------------------------------|
| `vectordata/src/dataset/facet.rs`                                 |
| `veks-core/src/formats/facet.rs`                                  |
| `vectordata/src/model.rs`                                         |
| `vectordata/src/dataset/profile.rs`                               |
| `vectordata/src/view.rs`                                          |
| `vectordata/src/knn_entries.rs`                                   |
| `vectordata/src/dataset/expansion.rs` (`command_facet`)           |

### 7.3 Docs

| File                                                              | Update                                      |
|-------------------------------------------------------------------|---------------------------------------------|
| `docs/sysref/01-data-model.md`                                    | Replace single-F section with E+F.          |
| `docs/sysref/02-api.md`                                           | New TestDataView accessors.                 |
| `docs/sysref/05-commands.md`                                      | New command rows (prefiltered/postfiltered).|
| `docs/sysref/07-import.md`                                        | Pipeline DAG split.                         |
| `docs/sysref/08-architecture.md`                                  | Architecture diagram split.                 |
| `docs/sysref/commands/compute-filtered-knn.md`                    | Replace with `compute-prefiltered-knn.md` and `compute-postfiltered-knn.md`. |
| `docs/sysref/commands/analyze-explain-filtered-knn.md`            | Update to describe both modes.              |
| `docs/sysref/commands/README.md`                                  | Command index.                              |
| `docs/sysref/commands/prepare-partitions.md`                      | Partition rule for E/F.                     |
| `docs/tutorials/build-predicated-dataset.md`                      | Tutorial covers both producers.             |
| `docs/tutorials/dataset-recipes.md`                               | Recipe examples updated.                    |
| `docs/tutorials/startup-and-publish.md`                           | Workflow updated.                           |
| `docs/tutorials/verify-with-knn-utils.md`                         | Cross-check both modes.                     |
| `README.md`                                                       | One-line feature list update.               |

### 7.4 Tests & fixtures

| File                                                              | Update                                      |
|-------------------------------------------------------------------|---------------------------------------------|
| `veks/tests/fixtures/synthetic-1k/dataset.yaml`                   | Add E facet entries.                        |
| `veks/tests/e2e_pipeline.rs`                                      | Include both `compute prefiltered-knn` and `compute postfiltered-knn`; assert |F| ≤ K and |E| = K when matching ≥ K. |
| `veks/tests/check_and_import.rs`                                  | Detection covers E.                         |
| New: integration test asserting `F == G ∩ R` byte-for-byte.       |                                             |
| New: integration test asserting `E == top_K(B|R) by distance` against a reference scan. |                  |

## 8. Implementation phases

Sequenced so each phase compiles and tests pass before the next starts.

1. **Schema & facet registry** (task #2) — split the legacy filtered
   variants of `StandardFacet` and `Facet` into prefiltered (F) and
   postfiltered (E). Rename `ProfileConfig` / `DSProfile` /
   `TestDataView` filtered-* members to prefiltered-* (F) and add
   postfiltered-* (E). Add facet-code mapping (F=prefilter, E=postfilter)
   and `command_facet` entries. Roundtrip and alias tests.
2. **Producers** (task #3) — rename `compute filtered-knn` →
   `compute prefiltered-knn` (F). Implement new `compute postfiltered-knn`
   (E = G ∩ R). Keep `compute filtered-knn` registered as a legacy alias
   for the prefilter factory. Update cache version tags. Update artifact
   manifests.
3. **Verifiers and consumers** (task #4) — split verifier into prefilter
   and postfilter variants. Update `analyze explain-filtered-knn`,
   `describe-dataset`, `inspect_partition`, `dataset_lookup`,
   `prepare/wizard`, `prepare/import`, `prepare/infer_manifest`,
   `prepare/stratify`, `explore/dataset_picker`.
4. **Docs, tests, swimlane** (task #5) — update all docs listed in
   §7.3, swimlane generator, test fixtures, add the two new integration
   tests (`E == G ∩ R`, `F == reference scan`).
5. **Migration helper** (task #6) — opt-in
   `veks dataset migrate-facet-names` that rewrites legacy
   `filtered_*` keys to `prefiltered_*` and renames the matching files
   on disk. Independent of the consumer phase; documented as the
   recommended one-shot for existing dataset.yaml files.

After Phase 4, the codebase has no remaining ambiguous F references.
Phase 5 lands separately and is run by users on a per-dataset basis.

## 9. Open questions

- **Partition profiles**: do oracle partitions carry both F and E, or
  F only? Proposed: F only, because within a partition every base
  vector matches the partition's predicate, so post-filter at partition
  scope is degenerate (equals pre-filter). Confirm before Phase 3.
- **Distance sign convention for E**: E's distances come from D
  directly (already in FAISS publication convention). No conversion
  needed. Verify this matches what consumers expect — call out in the
  test.
- **`knn_entries.yaml` (legacy jvector format)**: has no slot for E or
  F. Proposed: no change; the loader stays as-is, and richer datasets
  use `dataset.yaml`.

## 10. Decisions resolved

For traceability — these were live decisions during memo drafting,
resolved by the user:

- **Facet codes:** **F = pre-filter** (ACORN G_K, the legacy
  filtered-knn shape — keeps the F letter so published datasets retain
  their meaning), **E = post-filter** (G ∩ R, the new sparse artifact).
- **Canonical key names:** `prefiltered_neighbor_indices` /
  `prefiltered_neighbor_distances` (F),
  `postfiltered_neighbor_indices` / `postfiltered_neighbor_distances`
  (E). Both are new canonical names.
- **Legacy alias direction:** `filtered_neighbor_indices` /
  `filtered_neighbor_distances` → F (the on-disk content from the
  legacy `compute filtered-knn` is pre-filter shape, so the alias
  points at the matching typed slot; no regeneration required).
- **Legacy command name:** `compute filtered-knn` stays registered as
  an alias for `compute prefiltered-knn` so existing pipeline.yaml
  files keep working.
- **`analyze explain-filtered-knn` shape:** single bimodal command;
  auto-detects which of F/E are available; optional `--mode` to
  force; when both present, juxtaposes them.

## 11. References

- Patel, Kraft, Guestrin, Zaharia. *ACORN: Performant and
  Predicate-Agnostic Search Over Vector Embeddings and Structured Data.*
  Stanford / UC Berkeley, 2024. Local copy at
  `local/ACORN-formatted.md`. Relevant sections: §3.1 (problem
  definition), §3.2 (pre/post-filter baselines), §4 (oracle partition).
- This repo: `docs/design/knn-parity-memo.md` (precedent for memo
  format and "what parity means" framing).
