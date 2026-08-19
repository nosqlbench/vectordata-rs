# Plan: A PVS-Capable Dataset from a Reference Vector-Search Benchmark Corpus

Status: PROPOSED — pending four decisions (§7)
Date: 2026-08-14

This plan describes how to construct a predicated-vector-search (PVS) capable
dataset — full facet set `BQGDMPRF` (+`E`, optionally +`O`) — from the corpus
used by a published commercial vector-search benchmark (hereafter *the
reference benchmark*), at full corpus cardinality on every facet, using
`veks prepare bootstrap` and the veks pipeline.

---

## 1. Source analysis: what the reference benchmark actually built

### 1.1 Corpus

- **Source**: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
  (McAuley Lab) — item *metadata* records (not the reviews), 33 product
  categories, ~48.2M unique items.
- **Cleaning** (per the benchmark's published test harness,
  `save_voyage_embeddings.py`): per-category load of `raw_meta_<Category>`,
  filter to non-null description, dedupe on (title, description), drop rows
  missing title/description.
- **Corpora**: **5.5M items ("medium")** and **15.3M items ("large")**.

### 1.2 Embeddings

- `"Item: Title: {title} Description: {description}"` → **voyage-3-large,
  2048-d** (dotProduct), with Matryoshka slices to 1024/512/256-d (cosine)
  tested via database-side views.
- Generated via the Voyage API in batches of 200 and inserted directly into
  the benchmark database as binary vectors. **The embeddings were never
  published.**

### 1.3 Metadata / filter fields

- `category` — 33 string values, one per source file.
- `price` — regex-parsed to numeric (`re.sub(r"[^\d.]", "", price_str)`);
  **unparseable prices replaced with random values**; stored as integer cents.

### 1.4 Queries and ground truth

- Only **50** hand-written e-commerce query strings (craft-supply themed),
  embedded at 2048-d.
- Ground truth computed in-database via the engine's exact-nearest-neighbor
  (ENN) query mode, not stored as an artifact; recall = |ANN ∩ ENN| / k.

### 1.5 Filtered-search test

- Single hardcoded predicate: `category == "Pet_Supplies" AND price <= 1000`
  on the 15.3M corpus — ~500k matches, **~3% selectivity**.
- Finding: at that selectivity, binary-quantized queries were ~4× as expensive
  to reach 90–95% recall vs unfiltered.

### 1.6 Gaps that make it thin as a PVS dataset

- 50 queries, one predicate, no published ground-truth artifacts, no published
  embeddings, random-filled missing prices (non-deterministic filter
  semantics). Our build keeps the corpus and its full cardinality while fixing
  all of these.

---

## 2. Target dataset shape

- **Name**: `amazon-reviews-2023-pvs`
- **Metric**: `DOT_PRODUCT` (matches voyage-3-large publication convention)
- **Facets**: `BQGDMPRF` (+`E` always emitted with F; `O` optional, §4.6)

| Facet | Reference benchmark | Our target | Cardinality |
|---|---|---|---|
| B base vectors | 15.3M × 2048-d f32 | same corpus, full cardinality | 15.3M |
| Q queries | 50 hand-written strings | 10k held-out perturbed items (self-search) | 10k |
| G/D ground truth | ENN at query time, not stored | exact KNN, k=100, stored | 10k × 100 |
| M metadata | category + price per item | category, price_cents, avg_rating, rating_number, store | 15.3M rows |
| P predicates | 1 fixed filter | survey-calibrated compound (`category EQ ∧ price LE`, plus richer shapes), selectivity ~0.1%–3% | 10k |
| R predicate results | none | full match-ordinal lists per predicate | 10k × ~15k–450k |
| F prefiltered GT | none | exact top-k over each predicate's passing set | 10k × 100 |
| E postfiltered GT | none | G ∩ R, rank-preserving, sentinel-padded | 10k × 100 |
| O oracle partitions | none | per-category base+GT partitions (33 labels) | optional |
| strata | separate 5.5M corpus | windowed sized profiles 1m/2m/4m/8m (+ literal ~5.5m) | prefix windows |

Because bootstrap shuffles before the query/base split, every sized-profile
prefix window is an unbiased sample of the full category mixture — the ~5.5m
stratum is a statistical stand-in for the reference benchmark's medium corpus
(their exact item set was never published either).

---

## 3. The one genuine decision: where the vectors come from

The embeddings were never published, so there are three options:

- **Option A — replicate with the Voyage API** (`voyage-3-large`, 2048-d).
  Faithful to the benchmark. ~15.3M items × ~150 avg tokens ≈ 2–3B tokens →
  roughly **$350–600** of API spend plus rate-limit-bound wall time. Choose
  this if the goal is comparing numbers against the reference benchmark's
  published results.
- **Option B — local open-weight embedder at full cardinality
  (RECOMMENDED).** Real text in, real vectors out — e.g. Qwen3-Embedding-4B
  (MRL, can emit 2048-d) or a 1024-d model. Preserves what makes PVS
  realistically hard: the *correlation between embedding clusters and the
  category/price metadata*. One GPU, hours-to-a-day of embedding time, zero
  API cost.
- **Option C — synthesized vectors** (`veks prepare synthesize` /
  `generate from-model`). **Recommended against** for this dataset: synthetic
  vectors destroy the vector↔metadata correlation, which is the entire point
  of a PVS benchmark.

Everything downstream of this choice is identical.

---

## 4. Phased execution

### Phase 0 — small veks prep changes

All congruent with the CLI↔yaml mirror rule (every knob reachable from both
surfaces):

1. **Surface `--sized-profiles` on `prepare bootstrap`** — exists in
   `ImportArgs` (`veks/src/prepare/import.rs`) but is wizard-only today.
   Declaring strata at bootstrap time gets deferred sized expansion in one
   `veks run`, and the KNN segment cache shares work across profiles.
   *Fallback*: bootstrap → run → `veks prepare stratify --spec` → run again
   (provenance keeps the second pass incremental).
2. **Verify/plumb `--predicate-strategy compound` and `--selectivity`**
   through to the survey-based `generate predicates` step. With real metadata,
   bootstrap emits the survey-calibrated generator; we need compound
   `category EQ ∧ price LE` shapes with a selectivity target.
   *Fallback*: hand-tune the `generate-predicates` step options in
   `dataset.yaml` after bootstrap (a supported surface — the fixtures do it).
3. **Only if O with a custom scope: fix the `oracle_scope` round-trip bug** —
   bootstrap writes it as a bare `attributes:` key but the runner reads
   `attributes.tags["oracle_scope"]`, so custom scopes are silently dropped
   (`import.rs` yaml emission vs `veks-pipeline/src/pipeline/mod.rs` reader).
   *Workaround*: hand-write it under `attributes.tags`.
4. **Confirm multi-shard parquet ingestion** for base vectors, or consolidate
   shards to one file in Phase 2's output step.

### Phase 1 — corpus acquisition and normalization

- Fetch the 33 `raw_meta_<Category>` configs from HF
  `McAuley-Lab/Amazon-Reviews-2023` via `veks pipeline download huggingface`
  (glob + revision + resume). HF's auto-converted parquet branch keeps us on
  parquet, which veks-core reads natively.
- One preprocessing pass (the only genuinely external stage alongside
  embedding — treat as upstream provenance like any foreign dataset):
  - Apply the reference benchmark's cleaning: non-null description, dedupe on
    (title, description) — doing it *pre*-embedding saves embedding cost;
    veks's vector-level dedup still runs as a second net.
  - Parse `price` with the same regex. **Deliberate deviation**: do not
    random-fill unparseable prices — encode missing as an explicit sentinel
    (`price_cents = -1`) so predicates stay deterministic; the survey simply
    sees the sentinel mass as part of the real distribution.
  - Assign each item an ordinal.
- Emit two ordinal-aligned artifacts:
  - **text shards** (input to embedding)
  - **`metadata_all.parquet`** with columns `category` (int id 0–32; name
    table in the layout), `price_cents` (i64), `average_rating`,
    `rating_number`, `store`. The parquet→MNode reader
    (`veks-core/src/formats/reader/parquet_mnode.rs`) ingests this directly
    into the M facet.

### Phase 2 — embedding (per the §3 decision)

Embed `"Item: Title: {title} Description: {description}"` in ordinal order;
write vectors as `.npy`/parquet shards, consolidated to one base artifact.

**Invariant**: row `i` of vectors ↔ row `i` of metadata. The pipeline
preserves it from there (`extract-metadata` reorders M through the same
shuffle as B).

### Phase 3 — pilot at small scale

Run the *entire* facet graph on a slice before spending the big compute:

```bash
veks prepare bootstrap \
  --name arv-pvs-pilot --output work/pilot \
  --base-vectors work/vectors/base_all.parquet \
  --metadata work/meta/metadata_all.parquet \
  --self-search --query-count 1000 \
  --metric DOT_PRODUCT --neighbors 100 --seed 42 \
  --predicate-count 1000 --selectivity 0.01 \
  --base-fraction 1% \
  --required-facets "BQGDMPRF" --force
veks run work/pilot --output batch
veks check work/pilot --check-integrity
```

This validates: parquet fast-path import (`require_fast`), metadata slab +
layout, survey output (`metadata_survey.json` — inspect price/category
distributions here), predicate selectivity calibration, the
`zero-match-threshold` gate (aborts if >50% of predicates match nothing — the
canary for generator/schema mismatch), F/E computation, and all consolidated
verifiers including the SQLite predicate re-evaluation. Numerical verification
via the built-in verify steps — no bash demos.

### Phase 4 — full build

```bash
veks prepare bootstrap \
  --name amazon-reviews-2023-pvs --output <dataset-dir> \
  --base-vectors work/vectors/base_all.parquet \
  --metadata work/meta/metadata_all.parquet \
  --self-search --query-count 10000 \
  --metric DOT_PRODUCT --neighbors 100 --seed 42 \
  --predicate-count 10000 --selectivity 0.01 \
  --sized-profiles "mul:1m..8m/2,5500k" \
  --required-facets "BQGDMPRF" --force
veks run <dataset-dir> --governor maximize --output batch
```

(`--sized-profiles` is the Phase-0 flag; use the stratify fallback otherwise.)

Post-bootstrap, before `run`, review the generated `dataset.yaml` — it is the
tuning surface:

- `generate-predicates`: strategy `compound`, `selectivity` /
  `selectivity-max` band
- `evaluate-predicates`: `segment_size` (default 1M rows; auto-shrinks under
  memory pressure)
- `maxk: 100`

Predicate evaluation and KNN both checkpoint per-segment in `.cache/`, so the
long stages are resumable; the resource governor handles memory/thread pacing.

### Phase 5 — verification, docs, publish

`veks run` already chains the verifiers (`verify-knn`,
`verify-predicates-sqlite`, `verify-prefiltered/postfiltered-knn`) and the
finalizers (dataset.json, catalog, `docs/dataset.md` + exemplars, merkle
`.mref`). Then:

```bash
veks check <dataset-dir> --check-integrity
veks prepare cache-compress   # optional
vectordata push               # when ready to publish
```

### Phase 6 (optional) — oracle partitions and MRL variants

- Re-run with `--required-facets "+O"` to add per-category partitions
  (33 labels, well under the 100-partition default) — models "one index per
  category" with per-label G/D. Cost: roughly doubles base-vector disk
  (~+125GB) since partitions materialize their own ordinal spaces.
- For 1024/512/256-d Matryoshka variants: slice dimensions during Phase 2
  preprocessing into sibling vector artifacts and bootstrap them as sibling
  datasets — dimension slicing is not a veks transform today.

---

## 5. Resource budget (15.3M target)

| Resource | Estimate |
|---|---|
| Base vectors (f32, 2048-d) | ~125 GB |
| `.cache` working copies during prepare/sort | transiently ~2× base |
| M/P/layout slabs | a few GB |
| R facet at ~1% mean selectivity | ~6 GB (at the reference 3%: ~18 GB — `selectivity-max` is the lever) |
| G/D/F/E | megabytes |
| **Total disk to plan for** | **~400–500 GB** (+125 GB if O enabled) |
| Exact KNN (10k × 15.3M × 2048-d ≈ 6×10¹⁴ FLOPs) | tens of minutes to a few hours on a many-core box (SimSIMD/BLAS); strata nearly free via segment reuse |
| Prefiltered KNN | ~100× cheaper than full KNN |
| Predicate evaluation (15.3M × 10k) | segmented, resumable |
| Embedding (Option B) | hours-to-a-day on one GPU; the wall-clock long pole |
| Embedding (Option A) | ~$350–600 API spend, rate-limit bound |

---

## 6. Cardinality-fidelity summary

"Utilizes the whole dataset, at least in cardinality of the key data for the
facets" is satisfied as follows: B and M carry the full 15.3M-item corpus
(real vectors from real text, real category/price per item); Q/P/G/D/R/F/E are
populated at benchmark-grade cardinality (10k queries, 10k predicates, k=100)
rather than the reference benchmark's 50-query/1-predicate shortcut; strata
provide the medium (~5.5M) and smaller scale points as windows of the same
corpus rather than separate builds.

---

## 7. Open decisions

1. **Vector source** — Option A (Voyage replication, ~$350–600) vs
   **Option B (local embedder, RECOMMENDED)** vs Option C (synthetic,
   recommended against).
2. **Corpus ceiling** — the 15.3M "large" replica (**recommended**; the whole
   dataset as the reference benchmark used it), or the full cleaned corpus
   (~30M+ items; everything scales linearly ×2).
3. **Price handling** — sentinel encoding (**recommended**) vs bug-for-bug
   random-fill parity with the reference benchmark.
4. **O facet now or later** — disk doubles; also gates the Phase-0
   `oracle_scope` fix.

---

## 8. Corpus alternatives (annex)

The corpus-selection criteria, texture-aspect taxonomy, and the survey of
alternative corpora with richer PVS textures live in the companion annex:
[`pvs-corpus-annex.md`](pvs-corpus-annex.md). Its recommendations feed
decision 2 in §7 (corpus ceiling / successor corpus) and may add a
multi-valued-field support check to Phase 0.
