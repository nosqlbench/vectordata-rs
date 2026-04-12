# analyze explain-filtered-knn

Full query trace through every pipeline stage: predicate → selectivity →
unfiltered GT → filtered GT → intersection analysis.

## Usage

```bash
veks pipeline analyze explain-filtered-knn --ordinal <n> [--profile default] [--limit 20]
```

Auto-resolves all file paths from `dataset.yaml`.

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
├─ Stage 4: Filtered KNN Results (F) ─────────────────────
│    [0] ordinal      ...  dist=...  meta=9
│    ...
│
└─ Stage 5: Intersection Analysis ────────────────────────
   unfiltered GT:     100 neighbors
   filtered GT:       81 neighbors
   intersection:      ... neighbors
```
