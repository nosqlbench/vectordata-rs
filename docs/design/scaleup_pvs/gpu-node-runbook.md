# GPU Node Runbook — scaleup-pvs-gpu (g7e.12xlarge)

Status: RUNBOOK for the agent/operator on the GPU node
Date: 2026-08-19
Node: i-0fe055611095d03ee, us-east-1d, 2× NVIDIA RTX PRO 6000 Blackwell
(96 GB each), 48 vCPU, 512 GB RAM, 2 TiB gp3 root (8k IOPS / 500 MB/s),
3.8 TB instance-store NVMe (raw; wiped on stop — scratch only).
AMI: stock Ubuntu 24.04 (no NVIDIA driver preinstalled).
~$8.29/hr on-demand — **stop the instance when idle**.

Context docs (same directory): `s2oa-passage-pilot-plan.md` (decisions
D1–D8), `s2ag-bulk-study.md` (measured rates/sizes), and the working
`dataset.yaml` pattern described in §3 below.

## 1. System setup

```bash
# NVIDIA driver (Blackwell needs the R570/580+ line) + CUDA toolkit for
# candle's nvcc build. Use NVIDIA's CUDA repo for Ubuntu 24.04 (noble):
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-drivers cuda-toolkit
sudo reboot   # then verify: nvidia-smi shows 2x RTX PRO 6000, driver 570/580+

# Build deps
sudo apt-get install -y build-essential pkg-config libssl-dev libopenblas-dev git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH=$HOME/.cargo/bin:$PATH
# nvcc must be on PATH for the embed-cuda build:
export PATH=/usr/local/cuda/bin:$PATH
```

Instance-store scratch (optional, recommended for HF cache):

```bash
lsblk   # identify the raw instance-store NVMe device(s)
sudo mkfs.ext4 /dev/nvme1n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch
sudo chown ubuntu /scratch
export HF_HOME=/scratch/hf   # model cache is re-fetchable; fine to lose on stop
```

## 2. Build veks

```bash
git clone <repo> vectordata-rs && cd vectordata-rs   # see "source transfer" note below
cargo install --path veks --features embed-cuda
veks help embed   # must resolve; confirms the command registry has generate embed
```

Verify the CUDA path end-to-end with the real-model smoke test:

```bash
cargo test -p veks-pipeline --features embed-cuda real_model_embeds -- --ignored
```

**Source transfer**: the passage-pilot work (download s2ag / generate
passages / generate embed / verify alignment, the umbrella crate, the
logging fix) may not be on the remote yet — check `git log` for the
passage-pilot commits before assuming a clone is current; otherwise rsync
the working tree from the datamir node or wait for the push.

## 3. Workspace

```bash
sudo mkdir -p /work/scaleup_pvs && sudo chown ubuntu /work/scaleup_pvs
cd /work/scaleup_pvs
# From the origin node (or operator): dataset.yaml and keys.yaml
#   keys.yaml: S2-API-KEY entry, chmod 600 — NEVER in git; carried by operator.
# Re-download the corpus in-region (~20 min at concurrency 8; resumable):
veks run dataset.yaml   # step 1 fetches 316 shards (~301 GB) to sources/s2orc_v2
```

The chunk step reproduces byte-identically (deterministic chunker), so
regenerating `upstream/passages/` here is equivalent to copying it.

## 4. GPU embed settings (edit dataset.yaml deliberately)

- `device: cuda` (or `cuda:0`/`cuda:1` for range-sharded parallel runs)
- `dtype: bf16` — **pin explicitly**; a pinned dtype is a recorded identity
  axis, and bf16 output is not byte-identical to the CPU/f32 pilot run.
- `batch-size: 256` as a starting point; tune upward while watching
  `nvidia-smi` utilization (96 GB leaves enormous headroom at 0.6B).

Two-GPU pattern: two `veks run` invocations over disjoint parent/passage
ranges, one per `device: cuda:N`. (At full scale the work should be split
into ~1–2M-passage steps anyway for spot/interruption resilience.)

## 5. Day-one gate measurements (before committing the full run)

1. Re-embed the 28,201-passage pilot set at 0.6B/bf16 — measure real
   passages/s; compare desk estimate (~1.5–3k/s per GPU).
2. Model gate: rate + quality across 0.6B / 4B / 8B on the pilot set.
   **Blocker for 4B/8B**: `generate embed` currently loads single-file
   `model.safetensors` only; the larger models ship sharded weights —
   the `model.safetensors.index.json` extension must land first.
3. Record the chosen model × output-dims as a plan decision (storage and
   KNN cost scale with dims: 1024-d → ~1.8 TB B facet at 450M; 4096-d →
   ~7.4 TB).

## 6. Cost hygiene

- `sudo shutdown` or stop via console when idle; the 2 TiB root persists
  (~$190/mo) and the instance-store scratch is lost (by design).
- Everything on the node is re-derivable: corpus re-downloads with the
  key, chunking is deterministic, embeddings re-run from provenance.
