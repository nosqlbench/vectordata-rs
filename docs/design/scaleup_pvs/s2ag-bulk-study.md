# S2AG Bulk-Acquisition Study — Measurements and Projections

Status: MEASURED (proxies) — re-measure the exact signed-URL path once
`S2_API_KEY` arrives (one `veks pipeline download s2ag` run)
Date: 2026-08-19
Host: the dev box (AWS, /mnt/datamir, 2.3TB free at time of study)

## What was measured

The Datasets API file-list endpoint is 401 without a key, so the exact
signed-URL path could not be timed yet. Two labeled proxies were used:

1. **Content statistics** — two chunks of `allenai/peS2o` v2 (the AI2-cleaned
   S2ORC derivative; same corpus lineage, ODC-BY, keyless via HF):
   `train-00000` (1.57GB) and `train-00010` (7.09GB), fully stream-parsed
   (3.88M records total; 5% full-JSON sample for word counts).
2. **Transport throughput** — HF CDN single/dual-stream on those chunks, and
   a raw public-S3 single-stream pull of a 1.3GB OpenAlex parquet file
   (proxy for S2's signed S3 URLs; same transport class as ai2-s2ag).

Artifacts kept under `target/tmp/s2ag-study/` (purged by `cargo clean`).

## Measurements

| Quantity | Value |
|---|---|
| HF CDN, single stream | 16.5–19.2 MB/s |
| HF CDN, two parallel streams | 15.6 + 18.4 = ~34 MB/s aggregate (scales) |
| Raw S3, single stream | **39.4 MB/s** |
| peS2o chunk composition | chunk 0: 3,058,025 records, all `s2ag` abstracts; chunk 10: 825,161 records, all `s2orc` full text (chunks are source-segregated) |
| Abstract record | ~1.48 KB raw JSON, ~194 words |
| **Full-text record (cleaned)** | **~28.1 KB raw JSON, ~8.59 KB gz, ~4,372 words** |
| gz ratio (full text) | ~3.3× |

## Projections

### Volume — real `s2orc_v2` (16M records, release 2026-08-11)

peS2o is cleaned text only; real s2orc records add external ids and span
annotations (est. +30–60% gz bytes/doc over 8.6 KB):

- **~180–220 GB gz central estimate** (bounds ~150–350 GB) for the full
  `s2orc_v2` dataset; ~30–40 shard files of ~5–10 GB each.

### Time — via the Datasets API signed URLs (39 MB/s/stream measured proxy)

| Scope | Single stream | concurrency=4 (default) |
|---|---|---|
| Pilot: 1 shard (~5–10 GB) | **2–4 min** | n/a (one file) |
| Full s2orc_v2 (~200 GB) | ~85–90 min | **~20–25 min** |
| Pessimistic 350 GB | ~2.5 h | ~40 min |

API-call budget: 3 calls total (releases, datasets, file list) — the 1 RPS
key limit is irrelevant to bulk acquisition; the constraint is pure S3
bandwidth, which scales with `concurrency`.

Keyless fallback (peS2o via HF, ~100 GB gz for the s2orc portion): ~15–45
min at 4–8 streams.

### Fan-out revision (gate measurement, provisional)

Desk estimate said 60–120 passages/doc. Measured words/doc (~4,372 ≈ ~5,700
tokens) at the para-v1 budget (~170–230 words/passage) gives
**~20–26 passages/doc** — the desk figure looks ~3–4× high, subject to
peS2o's cleaning having dropped some body content. Consequences if it holds
on real s2orc_v2:

- Pilot (1,000 docs): **~20k–26k passages** (not 60k–120k) — still ample
  for BQGD at q=1000/k=100.
- Full spine (16M docs): **~350–420M passages**, not 1–2B — a ~4× reduction
  in embed cost (~250–300 L40S-GPU-hours vs ~1,000) and in B-facet size
  (~1.4–1.7 TB f32 @1024-d).

Verify against real s2orc_v2 annotations during the pilot run (the fan-out
distribution is already logged by `generate passages`).

## Conclusions

1. Bulk acquisition via the Datasets API is **not** rate-limited in any way
   that matters: 3 API calls + S3-speed downloads; full corpus ≈ tens of
   minutes at default concurrency.
2. The only operational gate is key issuance latency (human review, days) —
   submit early; peS2o remains the zero-key fallback.
3. Desk passage fan-out is likely overestimated ~3–4×; treat 60–120/doc as
   an upper bound and re-baseline after the pilot's measured distribution.
4. `s2orc_v2` (16M records, improved annotations) supersedes `s2orc` (10M)
   as the pilot's `--dataset-name` choice; check its annotation schema
   against the `para-v1` parser before the run.
