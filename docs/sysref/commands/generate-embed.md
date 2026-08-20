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
| `--output` | yes | Output `.npy` path (C-order f32, row i embeds source row i) |
| `--column` | no | Source column holding the text (default: `text`) |
| `--model` | no | HuggingFace model id (default: `Qwen/Qwen3-Embedding-0.6B`) |
| `--revision` | no | Model revision (default: `main`; pin a commit sha for strict reproducibility) |
| `--batch-size` | no | Sequences per forward pass (default: 16; raise substantially on GPU) |
| `--max-length` | no | Token cap per row, including the terminal EOS pooling token (default: 1024) |
| `--device` | no | `auto` (default), `cpu`, or `cuda[:N]`; a comma-separated list (`cuda:0,cuda:1`) shards batches round-robin across one worker per entry — repeating a device runs overlapping workers on one GPU (needs the `embed-cuda` build feature) |
| `--dtype` | no | `auto` (f32 on cpu, bf16 on cuda), `f32`, `bf16`, `f16` |

## Notes

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
- The full-model smoke test is `#[ignore]`d (downloads ~1.2GB); run it
  with `cargo test -p veks-pipeline --features embed real_model_embeds --
  --ignored` on a networked host.
