# analyze explain-filtered-knn

Bimodal trace of the filtered-KNN pipeline for one or more queries.
Reads whichever of the two filtered-KNN ground-truth facets the profile
carries: **F** (pre-filter, `prefiltered_neighbor_*`) and/or **E**
(post-filter, `postfiltered_neighbor_*`). When the dataset uses the
legacy `filtered_neighbor_*` key, it resolves to F (the on-disk shape
of files produced by the legacy `compute filtered-knn` is pre-filter).

For each query ordinal: predicate → selectivity → unfiltered GT →
filtered GT → intersection analysis.

## Usage

```bash
# Single-ordinal verbose trace
veks pipeline analyze explain-filtered-knn --ordinal <n> [--profile default] [--limit 20]

# Range or sample — aggregate-distribution mode
veks pipeline analyze explain-filtered-knn --ordinals 0..1000
veks pipeline analyze explain-filtered-knn --sample 500 [--seed 42]
```

Auto-resolves all file paths from `dataset.yaml`, including both pre-
and post-filter facets when present.

## Histogram resolution

The intersection-size histogram in aggregate mode auto-sizes its bin
width to ~50 bins:

- `bin_width = 1` when the observed max intersection is `< 50`
- `bin_width = floor(max/50)` otherwise

Override with `--histogram-bin-width N` (must be ≥ 1).

## Exemplar ordinals

After the histogram, three exemplar query ordinals are surfaced — one
with the lowest intersection, one at the median, one with the highest.
Drop any of those into `--ordinal N` to re-run the verbose narrative on
a representative query at each end of the distribution.

## Facet selection

Path resolution priority for the F (pre-filter) facet:

1. Explicit `--prefiltered-indices` / `--prefiltered-distances` flags.
2. Legacy `--filtered-indices` / `--filtered-distances` flags (aliases).
3. Profile `prefiltered_neighbor_*` key.
4. Profile legacy `filtered_neighbor_*` key.

Path resolution for the E (post-filter) facet:

1. Explicit `--postfiltered-indices` / `--postfiltered-distances` flags.
2. Profile `postfiltered_neighbor_*` key.

## Example

```bash
veks pipeline analyze explain-filtered-knn --ordinal 0 --limit 5
```

```
═══ Query ordinal 0 ═══════════════════════════════════════

┌─ Stage 1: Predicate ─────────────────────────────────────
│  field_0 == 9
│
├─ Stage 2: Selectivity ───────────────────────────────────
│  81 of 1000 base vectors pass filter
│  selectivity: 0.081000 (8.10%)
│  1/selectivity: 12.3× reduction
│
├─ Stage 3: Unfiltered Ground Truth (G) ──────────────────
│  k=100 neighbors for query 0
│    [0] ordinal      288  ·
│    [1] ordinal      434  ·
│    ...
│
├─ Stage 4: Filtered KNN Results (F, pre-filter) ─────────
│    [0] ordinal      ...  dist=...  meta=9
│    ...
│
└─ Stage 5: Intersection Analysis ────────────────────────
   unfiltered GT:     100 neighbors
   prefiltered (F):   81 neighbors
   intersection:      ... neighbors
```
