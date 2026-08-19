# Hand-off: S2OA Passage Pilot — Minimal veks Pipeline (1000 documents)

Status: HAND-OFF PACKAGE — start-of-session context for planning elaboration
Date: 2026-08-19
Companions: [`pvs-s2orc-openalex-union.md`](pvs-s2orc-openalex-union.md) (the
full S2OA design), [`s2oa-passage-pipeline.drawio.png`](s2oa-passage-pipeline.drawio.png)
(the six-phase data-flow view this pilot instantiates a slice of)

## 1. Objective

Construct the **minimal veks pipeline** that exercises every stage of the
S2OA passage pipeline **for the passage elements only** — acquire → derive
passages → embed → veks dataset build — capped at the **first 1000 full-text
documents** as a pilot. The next session elaborates this into a concrete
plan, **including specifying and adding the veks pipeline commands that are
missing** (§5 is the gap list).

Scope guards:

- **Passage elements only**: no OpenAlex join, no metadata star, no M/P/R
  facets yet. Target facet set for the pilot dataset: **BQGD**
  (self-search queries by default; TLDR-query option noted in §7).
- **1000 documents, not 1000 passages**: passages derive from documents at
  ~60–120/doc, so expect ~60k–120k passages. The cap is parent-level,
  consistent with the parent-granularity principle (union doc §6.1).
- Pilot-scale everything: single machine, no fleet, no strata beyond
  default; the point is exercising the *stages and seams*, not scale.

## 2. Stage map (pilot slice of the pipeline diagram)

| Pipeline phase | Pilot form | Exists in veks? |
|---|---|---|
| 1 Acquire | fetch a pinned S2ORC release shard subset via S2AG Datasets API (free key; signed bulk-file URLs; gzipped JSONL) | partially — `download bulk` (template/cos modes); check signed-URL + auth handling |
| 2 Join spine | **skipped** (metadata-only concern) | n/a |
| 3 Derive passages | chunker over s2orc body text + annotation spans → `passages.parquet` (corpusid, section, ordinal, text), first-1000-docs cap | **missing — new command** |
| 4 Embed | Qwen3-Embedding-0.6B @ 1024-d over ~100k passages (minutes on one GPU); ordinal-aligned vector artifact | **missing — command or external contract** (§5.2) |
| 5 Metadata star | **skipped** | n/a |
| 6 veks build | `veks prepare bootstrap` (B in, self-search Q) → `veks run` → `veks check` | exists, verified this session |

## 3. Source specifics (established this session)

- **Passages come exclusively from S2ORC full text**: the `s2orc` dataset of
  the [S2AG Datasets API](https://api.semanticscholar.org/api-docs/datasets)
  (~15.7M open-access papers, 2026 release). Each record: `corpusid` + body
  text + annotation spans (sections, paragraphs, captions) with character
  offsets. OpenAlex has **no body text** — irrelevant to this pilot.
- License: ODC-BY (attribution). Published pilot facets carry no prose —
  vectors + passage *coordinates* only (union doc §1.2 posture).
- Chunker policy (union doc §3.3): deterministic, versioned, section-aware,
  ~150–300 tokens; passage id = (corpusid, section, ordinal).
- "First 1000 documents" needs a deterministic rule — proposal: the 1000
  lowest `corpusid`s within the lexically-first file of a pinned release,
  with release id + file + rule recorded in provenance. **Decide in new
  session** (§7).

## 4. What veks already has (session exploration findings)

- Command registry: `veks-pipeline/src/pipeline/commands/mod.rs` (~120-291).
- Vector import sources (`veks-core/src/formats/`): npy, parquet, hdf5,
  slab, xvec — **no JSONL/text sources**; parquet→xvec has a fast path
  (`require_fast`), parquet→MNode exists for later metadata work.
- `download huggingface` (HF tree API) and `download bulk`
  (`commands/fetch_bulkdl/`, template + S3/COS modes, resume via status
  file) — the S2AG Datasets API returns *signed, expiring* URLs; verify
  `mode: template` can consume them or note the small extension needed.
- `veks prepare bootstrap`: full flow mapped this session (emit_steps →
  dataset.yaml; `veks run` executes). Key flags for the pilot:
  `--self-search --query-count N --metric ... --neighbors 100
  --required-facets BQGD --seed 42`. Canonical e2e pattern:
  `veks/tests/e2e_http_sized_profiles.rs` (`default_args()` at ~:114) and
  the tutorial script `docs/tutorials/vecd-end-to-end/02-generate-dataset.sh`.
- Known quirks that may bite: `--sized-profiles` is wizard-only on the
  bootstrap CLI (not needed at pilot scale); `oracle_scope` round-trip bug
  (O facet only — out of scope here).

## 5. Gap list — commands to add (the core of the next session's plan)

### 5.1 Passage derivation (required)

No veks command today parses record-oriented text corpora or chunks text.
Needed: a command (working name `transform chunk-text` or a new
`text`/`prepare passages` verb — naming to be settled against the
CLI↔yaml congruence rule) that:

- reads s2orc-format JSONL(.gz) shards; selects documents by the
  deterministic first-N rule (`--doc-limit 1000`);
- applies chunker vX (versioned — **chunker id + params must be a
  provenance axis**, since passage identity depends on it);
- emits ordinal'd `passages.parquet` (+ a parent manifest), ordering
  passages in **parent blocks** so future prefix windows respect parents;
- is a first-class pipeline command: `describe_options`, provenance,
  resume, `dataset.log` — not a side script (repo rule: no bash demos;
  integration tests with numerical verification).

### 5.2 Embedding (decision required)

Two structurally honest options — decide, don't hedge, in the new session:

- (a) **In-veks embed command** (e.g. `generate embed` wrapping a local
  inference backend): keeps the whole pilot inside `veks run` provenance;
  adds a heavy dependency (model runtime) behind a feature gate.
- (b) **External embed stage with a hard artifact contract**: veks treats
  `vectors/base_all.npy` as a foreign input (exactly how the main plan
  treats voyage/Qwen output), with the ordinal-alignment invariant asserted
  at import. Less new code; upstream stage invisible to provenance unless
  §5.1's pattern is extended.

Pilot-scale reality: ~100k passages × ~250 tokens is minutes on one GPU or
even feasible on CPU for a first smoke run — option (b) is fastest to first
light; option (a) is the architecturally complete end state. Greenfield
persona says: pick the end state deliberately, possibly staging (b)→(a).

### 5.3 Acquisition polish (verify, maybe extend)

Confirm `download bulk` can drive S2AG Datasets API bulk files (signed URL
expiry, resume semantics); if not, extend it rather than scripting around it.

### 5.4 Explicitly deferred (do not build for the pilot)

Polars `tabular` command family (join/flatten/profile — metadata star only);
parent-sampled strata beyond the emission-order trick; set-valued M/P (T6)
support; TLDR query path (optional add-on, §7).

## 6. Pilot definition of done

1. One command sequence (documented in the plan, runnable end-to-end):
   fetch → chunk (1000 docs) → embed → `veks prepare bootstrap` →
   `veks run` → `veks check --check-integrity` all green.
2. Resulting dataset: BQGD facets, ~60k–120k base vectors @1024-d f32,
   self-search queries, k=100, verified by the built-in verify steps.
3. Integration test(s) with a small synthetic s2orc-format fixture
   (shared data-generation layer; tmp under `target/`; no process-env
   mutation; numerical verification of chunk counts + ordinal alignment).
4. Gate measurements recorded: passages/doc fan-out distribution (compare
   against the ≈60–120 desk estimate), chunker determinism (re-run →
   byte-identical passages).

## 7. Open decisions for the new session

- Chunk command name/verb + option surface (CLI ↔ dataset.yaml mirrored).
- Embed option (a) vs (b), and the model/runtime if (a).
- Deterministic first-1000 rule (proposal in §3).
- Metric: Qwen3 embeddings are normalized → cosine ≡ dot; pick
  `DOT_PRODUCT` vs `COSINE`+mode flags deliberately and record why.
- Ordinal scheme details: global passage ordinal assignment point
  (chunker output vs import), parent-block ordering guarantees.
- Whether the pilot also emits TLDR/title query vectors (real asymmetric
  queries) alongside self-search, or defers them.

## 8. Wider session context (for orientation, not pilot scope)

- Corpus strategy and scoring: [`pvs-corpus-annex.md`](pvs-corpus-annex.md)
  (FOC scoring §4, S2ORC profile §5.10, cardinality ceiling §6.4).
- Full S2OA design: [`pvs-s2orc-openalex-union.md`](pvs-s2orc-openalex-union.md)
  (entity graph §2 + `s2oa-entity-graph.drawio.png`, star §3.2,
  constellations §5, measurement gates §7).
- Facet anatomy teaching set: [`dataset-anatomy.md`](../dataset-anatomy.md) +
  `facets-BQGD` / `facets-BQGDMPR` / `facets-BQGDMPREF` diagrams.
- Reference-benchmark plan (Amazon corpus) with phases + sizing model:
  [`amazon-reviews-2023-pvs-plan.md`](amazon-reviews-2023-pvs-plan.md).
- Cloud sizing decided this session (not needed at pilot scale): spot
  g6e.xlarge fleet for embed; i4i-class for CPU phases; integral-cost
  framing; ~1,000 L40S-GPU-hours for the full 1.5B-passage embed.
- drawio-in-docker recipe (for diagram updates):
  `rlespinasse/drawio-desktop-headless` needs `--shm-size=1g` and
  `-e DRAWIO_DESKTOP_COMMAND_TIMEOUT=240s`; cell id `flat` is reserved
  (breaks export); preview without `-e`, final with `-e` + repair_png.
