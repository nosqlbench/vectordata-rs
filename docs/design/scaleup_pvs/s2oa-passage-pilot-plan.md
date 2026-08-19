# S2OA Passage Pilot — Concrete Plan (1000 documents, BQGD)

Status: PLAN — elaborates [`s2oa-passage-pilot-handoff.md`](s2oa-passage-pilot-handoff.md)
Date: 2026-08-19
Companions: [`pvs-s2orc-openalex-union.md`](pvs-s2orc-openalex-union.md) (full
S2OA design), [`s2oa-passage-pipeline.drawio.png`](s2oa-passage-pipeline.drawio.png)

This plan settles the hand-off's §7 decisions, specifies the three new
pipeline commands, and defines the runnable end-to-end sequence and its
tests. Codebase facts referenced here were verified this session
(command contract: `veks-pipeline/src/pipeline/command.rs:62`; registry:
`commands/mod.rs:120`; CLI↔YAML mirror: `cli.rs:114-165` — option names
are simultaneously the `--flags` and the `dataset.yaml` step keys, so the
congruence rule is satisfied by construction).

---

## 1. Decisions (hand-off §7, settled)

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Chunk command name | **`generate passages`** | SRD §1.5: group names are verbs, subcommands are nouns. `transform chunk-text` puts a verb in the noun slot; `transform` today means artifact-format transforms (convert/extract/ordinals), while `generate` already covers derived-artifact creation (shuffle, metadata, predicates). |
| D2 | Acquire command | **`download s2ag`** (new, source-specific) | S2AG's flow (API call with `x-api-key` → JSON of signed, expiring URLs) doesn't fit `download bulk`'s template mode: no header injection, per-file signed URLs inexpressible, HEAD re-verify 403s on signed GETs, basename collisions. Precedent: `download huggingface` is likewise source-specific with env-token auth (`HF_TOKEN`). Extending `bulk` with manifest+headers would still leave the API-negotiation step homeless. |
| D3 | Embed | **Option (a): in-veks `generate embed` command** (candle backend, feature-gated `embed`/`embed-cuda`), with **`verify alignment`** still gating the ordinal contract | REVISED 2026-08-19 during implementation: originally decided as option (b) (external stage), but the no-Python rule made the external reference script inadmissible, which collapsed the decision to its recorded end state. The whole flow now lives inside `veks run` provenance (model id + pinned revision + every byte-affecting knob are step options). The backend is a bespoke cache-free Qwen3 forward (the stock candle-transformers module's KV cache can't be reset across independent sequences) with right-padded causal batching. |
| D4 | First-1000 rule | Within the **lexically-first file** (by URL basename) of a **pinned release**, the **1000 lowest `corpusid`s among records with non-empty body text** | Adopts the hand-off §3 proposal; "non-empty body text" is required so the cap counts chunkable parents. Release id, file name, and rule parameters are all step options → provenance axes. |
| D5 | Metric | **`Cosine`** | Qwen3-Embedding vectors are trained for cosine similarity; they ship unit-normalized so cosine ≡ dot, and bootstrap's self-search extract normalizes unconditionally (no-op on unit vectors), keeping the invariant harmless. Declaring `Cosine` states the semantic; `DOT_PRODUCT` would state an implementation coincidence. |
| D6 | Ordinal scheme | **Global ordinal = row index of `passages.parquet`** (parent-block order, assigned at chunker output). **Passage identity = (corpusid, section, ordinal-in-section)** per union doc §3.3 — both are columns/derivable | Assigning at chunker output (not import) makes the parquet the single source of truth the embed contract and `verify alignment` both reference. Parent-block row order is what makes future prefix windows parent-respecting (union doc §6.1) for free. |
| D7 | Chunk budgets | **Word-count budgets, no tokenizer dependency** (chunker `para-v1`, §3.2) | The policy needs deterministic + versioned more than it needs exact BPE counts. A `tokenizers` dep would drag a model-specific `tokenizer.json` into a data-prep command. Words ≈ tokens/1.3; defaults chosen to land in the 150–300-token policy band. If a real tokenizer is ever wanted, it becomes a new chunker id — passage identity already keys on chunker id + params. |
| D8 | Query vectors | **Self-search only** (held-out, `--query-count 1000`) | TLDR/title-as-query deferred with the metadata star; the seam it exercises (a second embedded artifact as Q) is identical to B's, already covered. |

## 2. `download s2ag`

New file `veks-pipeline/src/pipeline/commands/fetch_s2ag.rs`, registered in
the `download` block of `register_all`. `CAT_DOWNLOAD` / `LVL_PRIMARY`.

Options (names = YAML keys = `--flags`):

| option | type | required | default | role | notes |
|---|---|---|---|---|---|
| `release` | string | yes | — | Config | explicit release id (e.g. `2026-08-12`). `latest` is rejected: the resolved value wouldn't match the recorded option, breaking provenance. |
| `dataset-name` | string | no | `s2orc` | Config | S2AG dataset name |
| `files` | string | no | `first:1` | Config | deterministic file selection over URL basenames sorted lexically: `first:N` or a glob (`all` = `*`) |
| `output` | Path | yes | — | Output | shard dir; `.down-rs-status.json`-style status file alongside |
| `tries` | int | no | `3` | Config | per-file attempts |
| `concurrency` | int | no | `4` | Config | parallel downloads |
| `api-key-file` | Path | no | — | Config | key by file *reference* (single-line or YAML `S2-API-KEY` entry); overrides `S2_API_KEY`; only the path can enter provenance |
| `api-base` | string | no | Datasets API URL | Config | override point for tests/mirrors |

(`dataset-name`, not `dataset`: the binary's spec parse-consistency guard
requires one parse-definition per option name, and `--dataset`/`-d` is
already taken by the hand-written dataset commands.)

- **API key**: env `S2_API_KEY` (mirrors `HF_TOKEN`), sent as `x-api-key` on
  the file-list request only. Deliberately **not** an option: options land in
  provenance sidecars and `dataset.log`; secrets must not.
- Flow: `GET /datasets/v1/release/{release}/dataset/{dataset}` → sort file
  URLs by basename → select per `files` → GET each signed URL directly (no
  HEAD probe — signed URLs sign the method). Signed query string is used for
  the fetch, stripped for the local filename. Files kept as `.jsonl.gz`
  exactly as served (wire bytes; no decompression — the chunker streams gz).
- Resume: `.s2ag-status.json` lists completed basenames; signed URLs are
  re-negotiated from the API on every run, so expiry only matters within a
  single run (and is retried via `tries`).
- `check_artifact`: expected-count comparison like bulk's template mode,
  using the status file (the URL list needs the API, so offline freshness
  falls back to `PartialResumable`).

## 3. `generate passages`

New file `veks-pipeline/src/pipeline/commands/gen_passages.rs`, registered
as `generate passages`. `CAT_GENERATE` / `LVL_PRIMARY`. Parquet writing goes
through a new small table codec in **veks-core** (which already carries
`arrow`/`parquet` 54 unconditionally): `veks-core/src/formats/passage_table.rs`
(schema authority, staged atomic writers, read-back, footer row-count probe).
veks-pipeline stays free of direct arrow/parquet deps, consistent with the
existing pattern (parquet is reached only through veks-core). This is the
workspace's first production-code parquet writer; it is a table writer and
deliberately does **not** enter `VecFormat::is_writable()`/`open_sink()`,
which are vector-sink APIs.

### 3.1 Options

| option | type | required | default | role | provenance axis? |
|---|---|---|---|---|---|
| `source` | Path | yes | — | Input | file or dir of s2orc `.jsonl` / `.jsonl.gz` shards (dir → lexical filename order) |
| `output` | Path | yes | — | Output | `passages.parquet` |
| `parents` | Path | no | `<output dir>/parents.parquet` | Output | parent manifest |
| `doc-limit` | int | no | — (all) | Config | yes — first-N rule (D4): N lowest corpusids with non-empty body |
| `doc-order` | enum `corpusid`\|`shuffle`\|`source` | no | `corpusid` | Config | yes — parent-block ordering; `shuffle` is the union-doc §6.1 parent-granularity shuffle; `source` streams in shard order with constant memory (the at-scale path; the other two buffer selected docs' passages) |
| `seed` | int | no | `0` | Config | yes — used by `doc-order: shuffle` |
| `chunker` | string | no | `para-v1` | Config | yes — chunker id; unknown id is a hard error |
| `min-words` | int | no | `40` | Config | yes |
| `target-words` | int | no | `170` | Config | yes (≈220 tokens) |
| `max-words` | int | no | `230` | Config | yes (≈300 tokens) |

Every knob that changes output bytes is an option, so the OPTIONS provenance
axis covers chunker identity completely (union doc §6 item 4).

### 3.2 Chunker `para-v1` (deterministic, section-aware)

Input: s2orc record = `corpusid` + `content.text` + `content.annotations`
(`sectionheader`, `paragraph` — JSON-encoded span arrays with char offsets).

1. Parse section-header spans; each paragraph span is labeled with the
   nearest preceding header's text (whitespace-collapsed, trimmed);
   paragraphs before any header get section `""`.
2. Within a section, greedily pack consecutive paragraphs into a chunk while
   `words ≤ max-words`; close the chunk when adding the next paragraph would
   exceed it. A single paragraph longer than `max-words` is split at word
   boundaries into `target-words` windows (no overlap). A trailing chunk
   under `min-words` merges into its predecessor within the section (or
   stands alone if it is the only chunk).
3. Records with empty/missing body text or no paragraph spans yield zero
   passages and are counted + logged (they are also excluded from
   `doc-limit` selection, per D4).

Two-pass streaming over gz (bounded memory): pass 1 scans `corpusid` +
body-presence to fix the selected parent set; pass 2 parses and chunks only
selected records. Progress via `ctx.ui.bar` per pass (record counts).

### 3.3 Output contract

`passages.parquet` (row order = parent blocks per `doc-order`; sections in
document order; chunks in order; **global ordinal = row index**):

| column | type | notes |
|---|---|---|
| `corpusid` | int64 | parent id |
| `section` | utf8 | label per §3.2 |
| `ordinal` | int32 | within (corpusid, section) — the identity triple of union doc §3.3 |
| `char_start` / `char_end` | int64 | source-text span (first packed paragraph's start … last's end) — the publishable *coordinates* (no-prose posture) |
| `text` | utf8 | passage prose — upstream artifact only, never a published facet |

`parents.parquet`: `corpusid: int64`, `passage_count: int32`,
`row_start: int64` (global ordinal of the parent's first passage).

Variables saved via `variables::set_and_save`: `passage_count`,
`parent_count` (interpolatable downstream; also the fan-out gate numbers).
The result message reports the passages/doc distribution (min/p50/mean/p90/max)
against the ≈60–120 desk estimate (DoD gate 4).

## 4. Embed contract + `verify alignment`

**Contract** (option (b)): the embed stage reads `passages.parquet` column
`text` in row order and writes `vectors/base_all.npy` — dtype `<f4`, shape
`[passage_count, 1024]`, C-order, row i = embedding of parquet row i,
unit-normalized (Qwen3-Embedding-0.6B, query/document prompt = document).
Reference implementation is a ~30-line Python script recorded in §7; it is
not a repo artifact (external stage by decision D3).

**`verify alignment`** — new file
`veks-pipeline/src/pipeline/commands/verify_alignment.rs`, registered as
`verify alignment`. `CAT_VERIFY` / `LVL_PRIMARY`. Asserts the ordinal-
alignment invariant *before* bootstrap consumes the vectors:

| option | type | required | role | notes |
|---|---|---|---|---|
| `source` | Path | yes | Input | vectors artifact (npy file/dir, xvec, parquet — counted via veks-core readers) |
| `reference` | Path | yes | Input | `passages.parquet` (row count via parquet metadata) |
| `dim` | int | no | Config | when set, also assert vector dimensionality |

Row-count inequality (or dim mismatch) → `Status::Error` with both counts in
the message. This closes the verified gap that no import-time cardinality
assertion exists between a foreign vectors file and a parallel artifact
(the only existing check is `query_count < base_n` at bootstrap).

## 5. End-to-end pilot sequence (DoD 1)

```bash
export S2_API_KEY=...   # free key, api.semanticscholar.org
cd upstream-s2oa/       # workspace for the upstream stages

veks pipeline download s2ag --release 2026-08-12 --dataset-name s2orc \
    --files first:1 --output s2orc
veks pipeline generate passages --source s2orc \
    --output passages/passages.parquet --doc-limit 1000
# external embed stage (§4 contract) → vectors/base_all.npy
veks pipeline verify alignment --source vectors/base_all.npy \
    --reference passages/passages.parquet --dim 1024

veks prepare bootstrap --name s2oa-pilot-1k --output ../datasets/s2oa-pilot-1k \
    --base-vectors vectors/base_all.npy --self-search --query-count 1000 \
    --metric Cosine --neighbors 100 --seed 42 --required-facets BQGD
veks run ../datasets/s2oa-pilot-1k/dataset.yaml --output batch
veks check ../datasets/s2oa-pilot-1k --check-integrity
```

Every `veks pipeline …` invocation is also a `dataset.yaml` step verbatim
(`--emit-yaml` prints it), so the upstream stages can be captured as a
checked-in upstream pipeline once the pilot is proven. No bootstrap changes
are needed: expected passage volume (~60k–120k × 1024-d f32 ≈ 250–500 MB)
is within the existing npy → fvecs convert path.

## 6. Tests (DoD 3)

Shared synthetic s2orc fixture generator (known doc/section/paragraph
structure, exact expected chunk counts), written once in the veks-pipeline
test tree; all temp under `target/` (already enforced via `.cargo/config.toml`
TMPDIR); no process-env mutation — `S2_API_KEY` handling is tested through a
pure resolver fn taking the env value as a parameter.

1. **Chunker unit tests** (`gen_passages.rs`): section labeling, budget
   packing/splitting/merging edges, zero-passage records, doc-limit
   selection rule, `doc-order` shuffle determinism per seed.
2. **`generate passages` integration** (`veks-pipeline/tests/`): registry-
   driven execute over `.jsonl` and `.jsonl.gz` fixtures → read back parquet
   (arrow/parquet as dev-deps, matching veks's precedent) → numerical
   verification: passage/parent counts, parent-block contiguity, global-
   ordinal = row-index, (corpusid, section, ordinal) uniqueness,
   `char_start/char_end` slice equals `text` modulo the packing joins,
   **re-run → byte-identical output** (DoD gate 4 determinism).
3. **`download s2ag` integration**: local test server (hand-rolled or axum
   dev-dep) serving the release-listing JSON (requiring `x-api-key`) +
   signed-query-string shard URLs → assert selection rule, filename
   stripping, resume-skip on second run, failure without key.
4. **`verify alignment`**: aligned/misaligned npy×parquet pairs; dim check.
5. **e2e slice** (`veks/tests/`): fixture passages + synthetic aligned
   vectors → `verify alignment` → bootstrap (BQGD, Cosine, self-search) →
   `veks run` → integrity checks green — the DoD-2 shape at toy scale.

## 7. Embed stage (superseded)

An external Python reference script originally stood here (option (b)).
Withdrawn 2026-08-19 with the D3 revision: the embed stage is the in-veks
`generate embed` command (candle backend; see
`docs/sysref/commands/generate-embed.md`), and no Python is used anywhere
in this pipeline. The §4 artifact contract is unchanged and still enforced
by `verify alignment`.

## 8. Follow-ons recorded (not pilot scope)

- GPU-scale embed run: build with `embed-cuda`, `device: cuda`,
  `dtype: bf16`, large `batch-size` on the planned A100/H100-class host.
- TLDR/title-as-query Q facet (D8), metadata star / M-P-R facets, polars
  `tabular` family, set-valued M/P — per hand-off §5.4.
- `docs/sysref/05-commands.md:124-146` documents a `download bulk` YAML
  config (`datasets:`/`savedir:`) the command never reads
  (`fetch_bulkdl/config.rs:20-46` is dead code) — clean up when next
  touching that command family.
