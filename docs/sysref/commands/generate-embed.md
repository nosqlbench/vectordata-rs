# generate embed

Embed a parquet text column into an npy vector artifact with an in-process
candle backend (Qwen3-Embedding family: last-token pooling,
unit-normalized). Row i of the output embeds source row i — the ordinal
contract asserted downstream by
[`verify alignment`](./verify-alignment.md). Model weights are fetched once
into the shared HuggingFace cache and reused.

Feature-gated: build with `--features embed` (CPU), or `--features
embed-cuda` for CUDA hosts (A100/H100-class), where `device: cuda` with
`dtype: bf16` and a large `batch-size` is the intended full-scale
configuration. `--features embed-cuda-flash` additionally compiles the
fused flash-attention (FA2) attention core plus three custom fused
kernels (residual-add+rmsnorm, silu·mul, per-head qk-norm+rope; PTX via
build.rs, needs nvcc), used automatically on CUDA with bf16. The fused
path never materializes the (b, h, l, l) score tensor, runs the custom
kernels at 92–94% of DRAM bandwidth, and leaves the GEMMs (~2/3 of GPU
time) at cuBLAS-peak 83–84% SM throughput (measured on Blackwell sm120).

## Usage (pipeline step)

```yaml
- id: generate-embed
  run: generate embed
  source: upstream/passages/passages.parquet
  output: upstream/vectors/base_all.npy
  model: Qwen/Qwen3-Embedding-0.6B
  revision: <pinned commit sha>
  batch-size: 32
  device: auto
  dtype: auto
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--source` | yes | Parquet file whose text column is embedded row-by-row |
| `--output` | yes | Output path; the format follows the extension — `.fvecs` (or any xvec) writes natively, anything else writes a C-order f32 `.npy`. Row i embeds source row i either way |
| `--column` | no | Source column holding the text (default: `text`) |
| `--range` | no | Half-open row window of the source, e.g. `[0,50M)` (default: all rows). Row i of the output embeds source row `start + i` |
| `--model` | no | HuggingFace model id (default: `Qwen/Qwen3-Embedding-0.6B`) |
| `--revision` | no | Model revision (default: `main`; pin a commit sha for strict reproducibility) |
| `--batch-size` | no | Sequences per forward pass (default: 128 — the GPU-tuned value; wall time is flat from 128 to 512 on the fused path) |
| `--max-length` | no | Token cap per row, including the terminal EOS pooling token (default: 1024) |
| `--device` | no | `auto` (default) claims **every** visible CUDA device, one worker each, else CPU; `cpu`; or `cuda[:N]`. A comma-separated list (`cuda:0,cuda:1`) shards batches round-robin across one worker per entry — repeating a device runs overlapping workers on one GPU (needs the `embed-cuda` build feature) |
| `--dtype` | no | `auto` (f32 on cpu, bf16 on cuda), `f32`, `bf16`, `f16` |

## Notes

- **Prefer `.fvecs` when the output feeds a dataset build.** `prepare
  bootstrap` emits a conversion step only when its base vectors are not
  already a native xvec format, so a `.fvecs` artifact collapses that step
  to an identity symlink — removing both the conversion pass and a second
  full copy of the vectors (410 GB at 100M x 1024-d). `.npy` remains the
  default for anything else and is byte-for-byte the same embedding.
- **Embedding a corpus larger than memory**: the source text for a window
  is held in RAM while that window runs (roughly 1 KB per passage), so a
  billion-row corpus is embedded in passes with `--range`. Only the
  parquet row groups a window touches are decoded, and consecutive
  windows tile the file exactly, so the `.fvecs` outputs concatenate into
  one artifact — each record carries its own dimension prefix, so `cat`
  is a valid join. Each pass is its own step with its own freshness, so
  an interrupted multi-day embed resumes at a pass boundary.
  Splitting changes how rows group into batches, so passes are
  numerically equivalent to a single run within the usual bf16
  reduction-reorder band (measured cosine ≥ 0.9998), not bit-identical
  to it. Passes are individually reproducible.
- Rows stream to the output as they complete rather than accumulating in
  memory: buffering 100M x 1024-d would need ~410 GB of RAM. Workers
  finish batches out of order, so completed rows are held only until the
  next contiguous run can be written, and batches are planned within a
  tokenize chunk so that window stays small. The tokenizer feeds a bounded
  queue for the same reason.
- The backend is a bespoke cache-free Qwen3 forward: right-padded batching
  under a causal mask (pads can never influence real tokens), so batches
  mix lengths safely; batches are planned longest-first to minimize
  padding waste, and results scatter back into source row order.
- Unfused-path limit: keep `batch-size × 16 heads × max-length²` below
  2^32 — at or past it a candle kernel's u32 indexing overflows and the
  run dies with `CUDA_ERROR_ILLEGAL_ADDRESS` (e.g. batch 256 at
  max-length 1024 crashes; batch 128 is the measured edge). The
  flash-attention path has no such ceiling.
- Model id, revision, and every byte-affecting knob are step options and
  therefore provenance axes. Sets the `embed_dim` pipeline variable.
  `device` and thread counts are not: sharding batches across GPUs
  reorders nothing within a row.
- A CPU run over more than 10k rows logs a warning with the rate
  arithmetic. CPU throughput is single-digit rows/s against ~2,500/s
  measured on a two-GPU host, so a large CPU embed is nearly always a
  stale recipe or a build missing `embed-cuda-flash` rather than an
  intent.
- The full-model smoke test is `#[ignore]`d (downloads ~1.2GB); run it
  with `cargo test -p veks-pipeline --features embed real_model_embeds --
  --ignored` on a networked host.
