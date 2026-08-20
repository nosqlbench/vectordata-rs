// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cache-free, batch-capable Qwen3 embedding forward on candle.
//!
//! `candle_transformers::models::qwen3` targets autoregressive decoding: its
//! KV cache appends unconditionally and its reset is private, so a model
//! instance cannot be reused across independent sequences — exactly the
//! wrong shape for embedding. This module mirrors that implementation's
//! architecture (same tensor paths and weight names, so checkpoints load
//! identically) minus the cache, plus right-padded batching:
//!
//! With right padding and causal attention, no real token can attend a pad
//! token (pads only ever appear at later positions), so the standard causal
//! mask is sufficient for correct last-real-token hidden states; pad-row
//! outputs are simply never read. Pooling is last-token (Qwen3-Embedding's
//! documented scheme), followed by L2 normalization.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, RmsNorm, VarBuilder};
use std::collections::HashMap;
use std::sync::Mutex;

/// The subset of Qwen3 `config.json` this forward needs. Fields that some
/// exports omit get serde defaults.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub eos_token_id: Option<u32>,
}

impl Config {
    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

fn linear_maybe_bias(inp: usize, out: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        candle_nn::linear(inp, out, vb)
    } else {
        candle_nn::linear_no_bias(inp, out, vb)
    }
}

/// Precomputed RoPE tables (identical construction to the stock module).
struct Rotary {
    sin: Tensor,
    cos: Tensor,
}

impl Rotary {
    fn new(cfg: &Config, dtype: DType, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Rotate q/k in (batch, seq, heads, head_dim) layout. The projections
    /// produce this layout contiguously, so `rope_thd` needs no copies —
    /// the profiled `ucopy_bf16` cost of rotating in (b, h, t, d) came
    /// entirely from the `.contiguous()` after `transpose(1, 2)`.
    fn apply_thd(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, seq_len, _, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q = candle_nn::rotary_emb::rope_thd(q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_thd(k, &cos, &sin)?;
        Ok((q, k))
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let bias = cfg.attention_bias;
        // Fold the 1/sqrt(head_dim) attention scale into the q_norm weight:
        // RMSNorm ends in an elementwise multiply by the weight, so scaling
        // the weight scales q for free — the profiled alternative was an
        // `affine_bf16` pass over the full (b, h, l, l) score tensor.
        let scale = 1.0 / (head_dim as f64).sqrt();
        let q_norm_w = (vb.pp("q_norm").get(head_dim, "weight")? * scale)?;
        Ok(Self {
            q_proj: linear_maybe_bias(cfg.hidden_size, num_heads * head_dim, bias, vb.pp("q_proj"))?,
            k_proj: linear_maybe_bias(cfg.hidden_size, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?,
            v_proj: linear_maybe_bias(cfg.hidden_size, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?,
            o_proj: linear_maybe_bias(num_heads * head_dim, cfg.hidden_size, bias, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(q_norm_w, cfg.rms_norm_eps),
            k_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: head_dim * num_heads,
        })
    }

    /// Whether the fused flash-attention core applies to this tensor: the
    /// FA2 kernels exist only in the `embed-cuda-flash` build and only for
    /// f16/bf16 on CUDA. Everything else takes the unfused path.
    fn use_flash(x: &Tensor) -> bool {
        cfg!(feature = "embed-cuda-flash")
            && x.device().is_cuda()
            && matches!(x.dtype(), DType::BF16 | DType::F16)
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kv_h, d) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let g = self.num_kv_groups;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Stay in the contiguous (b, l, heads, d) projection layout for the
        // per-head RMSNorm (Qwen3's signature detail; q_norm carries the
        // folded attention scale) and RoPE — flatten and rope_thd are then
        // copy-free.
        let q = self.q_norm.forward(&q.reshape((b * l * h, d))?)?.reshape((b, l, h, d))?;
        let k = self.k_norm.forward(&k.reshape((b * l * kv_h, d))?)?.reshape((b, l, kv_h, d))?;
        let (q, k) = rotary.apply_thd(&q, &k)?;
        let v = v.reshape((b, l, kv_h, d))?;

        // Fused path: FA2 consumes the (b, l, heads, d) layout directly
        // (GQA handled in-kernel), never materializes the (b, h, l, l)
        // scores, and needs no mask, no transposes, and no scale (folded
        // into q_norm, so softmax_scale = 1).
        #[cfg(feature = "embed-cuda-flash")]
        if Self::use_flash(x) {
            let ctx = candle_flash_attn::flash_attn(&q, &k, &v, 1.0, true)?;
            return ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj);
        }

        // Unfused path: move to (b, heads, l, d) — the one copy each — and
        // run GQA without repeat_kv by folding the query-head groups into
        // the row dimension so the batched matmul broadcasts each KV head
        // over its g query heads. Head g*j+i of the (b, h, ...) view is row
        // block i of KV-batch j, which is exactly repeat_kv's interleaving,
        // so the reshape round-trips losslessly.
        let mask = mask.expect("unfused attention path requires a causal mask");
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let qg = q.reshape((b, kv_h, g * l, d))?;
        let scores = qg.matmul(&k.transpose(2, 3)?)?; // (b, kv_h, g*l, l), pre-scaled
        let scores = scores.reshape((b, h, l, l))?.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.reshape((b, kv_h, g * l, l))?.matmul(&v)?;

        ctx.reshape((b, h, l, d))?
            .transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            ln1: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, rotary, mask)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }
}

/// The embedding model: Qwen3 trunk, no cache, last-token pooling.
pub struct EmbeddingModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: Rotary,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    /// Causal masks by sequence length. Batches are planned length-sorted,
    /// so lengths repeat heavily; without this the mask was rebuilt on the
    /// host and re-uploaded every batch.
    mask_cache: Mutex<HashMap<usize, Tensor>>,
}

impl EmbeddingModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // Causal-LM checkpoints prefix the trunk with `model.`; embedding
        // checkpoints (sentence-transformers layout) store the bare trunk.
        let root = if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb.clone()
        };
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, root.pp("embed_tokens"))?;
        let vb_l = root.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, root.pp("norm"))?,
            rotary: Rotary::new(cfg, vb.dtype(), vb.device())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            mask_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Additive causal mask (0 on allowed, -inf on future positions), in
    /// the model dtype, shaped for broadcast over (b, heads, l, l) scores.
    /// Cached per sequence length.
    fn causal_mask(&self, l: usize) -> Result<Tensor> {
        if let Some(m) = self.mask_cache.lock().unwrap().get(&l) {
            return Ok(m.clone());
        }
        let mask: Vec<f32> = (0..l)
            .flat_map(|i| (0..l).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, l, l), &self.device)?.to_dtype(self.dtype)?;
        self.mask_cache.lock().unwrap().insert(l, mask.clone());
        Ok(mask)
    }

    /// Embed one right-padded batch. `rows` are token-id sequences (each
    /// already ending in EOS); returns unit-normalized f32 embeddings in
    /// row order, shape [rows.len(), hidden].
    pub fn embed_batch(&self, rows: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let b = rows.len();
        let max_len = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut ids = Vec::with_capacity(b * max_len);
        for row in rows {
            ids.extend_from_slice(row);
            ids.extend(std::iter::repeat_n(0u32, max_len - row.len()));
        }
        let input = Tensor::from_vec(ids, (b, max_len), &self.device)?;

        let h0 = self.embed_tokens.forward(&input)?;
        let mask = if Attention::use_flash(&h0) {
            None // FA2 applies causality in-kernel; no materialized mask.
        } else {
            Some(self.causal_mask(max_len)?)
        };
        let mut h = h0;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rotary, mask.as_ref())?;
        }
        let h = self.norm.forward(&h)?;

        // Last-token pooling at each row's final real position, then L2
        // normalization — all batched on-device. The per-row formulation
        // profiled at 2 blocking device→host round-trips per passage
        // (92.8% of CUDA API time); this shape syncs once per batch.
        let positions: Vec<u32> = rows
            .iter()
            .enumerate()
            .map(|(i, row)| (i * max_len + row.len() - 1) as u32)
            .collect();
        let idx = Tensor::from_vec(positions, (b,), &self.device)?;
        let pooled = h
            .reshape((b * max_len, self.hidden_size))?
            .index_select(&idx, 0)?
            .to_dtype(DType::F32)?;
        let norms = pooled.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(1e-30)?;
        let out = pooled.broadcast_div(&norms)?.to_vec2::<f32>()?;
        debug_assert!(out.iter().all(|v| v.len() == self.hidden_size));
        Ok(out)
    }
}

