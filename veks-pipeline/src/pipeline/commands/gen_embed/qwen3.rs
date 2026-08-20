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
//!
//! Projections are fused at load time (q|k|v and gate|up become single
//! GEMMs). On CUDA with bf16 under `embed-cuda-flash`, the forward runs a
//! fused fast path: flash-attention plus the custom kernels in `kernels.cu`
//! (add+rmsnorm, silu·mul, per-head qk-norm+rope) — profiling showed the
//! composed elementwise ops 2-4x off the DRAM roofline. Everywhere else the
//! composed candle ops below remain the reference implementation.

use candle_core::{DType, Device, Module, Result, Tensor, D};
#[cfg(feature = "embed-cuda-flash")]
use candle_core::IndexOp;
use candle_nn::{Activation, Linear, RmsNorm, VarBuilder};
use std::collections::HashMap;
use std::sync::Mutex;

#[cfg(feature = "embed-cuda-flash")]
use super::kernels;

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

/// Precomputed RoPE tables (identical construction to the stock module).
struct Rotary {
    sin: Tensor,
    cos: Tensor,
    /// Per-position `[cos (d/2) | sin (d/2)]` rows for the fused kernel.
    cos_sin: Tensor,
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
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let cos_sin = Tensor::cat(&[&cos, &sin], 1)?.contiguous()?;
        Ok(Self { sin, cos, cos_sin })
    }

    /// Rotate q/k in (batch, seq, heads, head_dim) layout. The projections
    /// produce this layout contiguously, so `rope_thd` needs no copies.
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
    /// gate_proj and up_proj concatenated row-wise into one GEMM.
    gate_up: Linear,
    down_proj: Linear,
    act_fn: Activation,
    intermediate: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate_w = vb.pp("gate_proj").get((i, h), "weight")?;
        let up_w = vb.pp("up_proj").get((i, h), "weight")?;
        let gate_up = Linear::new(Tensor::cat(&[&gate_w, &up_w], 0)?.contiguous()?, None);
        Ok(Self {
            gate_up,
            down_proj: candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
            intermediate: i,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gu = x.apply(&self.gate_up)?;
        let g = gu.narrow(D::Minus1, 0, self.intermediate)?;
        let u = gu.narrow(D::Minus1, self.intermediate, self.intermediate)?;
        (g.apply(&self.act_fn)? * u)?.apply(&self.down_proj)
    }

    #[cfg(feature = "embed-cuda-flash")]
    fn forward_fused(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.gate_up)?
            .apply_op1_no_bwd(&kernels::SiluMul)?
            .apply(&self.down_proj)
    }
}

struct Attention {
    /// q_proj, k_proj, v_proj concatenated row-wise into one GEMM.
    qkv: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    /// `[q_norm_w (scaled) | k_norm_w]` for the fused kernel.
    qk_w: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rms_eps: f32,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let h = cfg.hidden_size;
        let q_w = vb.pp("q_proj").get((num_heads * head_dim, h), "weight")?;
        let k_w = vb.pp("k_proj").get((num_kv_heads * head_dim, h), "weight")?;
        let v_w = vb.pp("v_proj").get((num_kv_heads * head_dim, h), "weight")?;
        let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?.contiguous()?;
        let qkv_b = if cfg.attention_bias {
            let q_b = vb.pp("q_proj").get(num_heads * head_dim, "bias")?;
            let k_b = vb.pp("k_proj").get(num_kv_heads * head_dim, "bias")?;
            let v_b = vb.pp("v_proj").get(num_kv_heads * head_dim, "bias")?;
            Some(Tensor::cat(&[&q_b, &k_b, &v_b], 0)?.contiguous()?)
        } else {
            None
        };
        let o_proj = if cfg.attention_bias {
            candle_nn::linear(num_heads * head_dim, h, vb.pp("o_proj"))?
        } else {
            candle_nn::linear_no_bias(num_heads * head_dim, h, vb.pp("o_proj"))?
        };
        // Fold the 1/sqrt(head_dim) attention scale into the q_norm weight:
        // RMSNorm ends in an elementwise multiply by the weight, so scaling
        // the weight scales q for free — the profiled alternative was an
        // `affine_bf16` pass over the full (b, h, l, l) score tensor.
        let scale = 1.0 / (head_dim as f64).sqrt();
        let q_norm_w = (vb.pp("q_norm").get(head_dim, "weight")? * scale)?;
        let k_norm_w = vb.pp("k_norm").get(head_dim, "weight")?;
        let qk_w = Tensor::cat(&[&q_norm_w, &k_norm_w], 0)?.contiguous()?;
        Ok(Self {
            qkv: Linear::new(qkv_w, qkv_b),
            o_proj,
            q_norm: RmsNorm::new(q_norm_w, cfg.rms_norm_eps),
            k_norm: RmsNorm::new(k_norm_w, cfg.rms_norm_eps),
            qk_w,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: head_dim * num_heads,
            rms_eps: cfg.rms_norm_eps as f32,
        })
    }

    /// Whether the flash-attention core applies to this tensor: FA2 kernels
    /// exist only in the `embed-cuda-flash` build and only for f16/bf16 on
    /// CUDA. Everything else takes the composed path.
    fn use_flash(x: &Tensor) -> bool {
        cfg!(feature = "embed-cuda-flash")
            && x.device().is_cuda()
            && matches!(x.dtype(), DType::BF16 | DType::F16)
    }

    /// Fully fused attention block: one QKV GEMM, one qk-norm+rope kernel,
    /// flash-attention over zero-copy head views, one output GEMM.
    #[cfg(feature = "embed-cuda-flash")]
    fn forward_fused(&self, x: &Tensor, rotary: &Rotary, seq_len: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kv_h, d) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let qkv = x.apply(&self.qkv)?;
        let cs = rotary.cos_sin.narrow(0, 0, seq_len)?;
        let op = kernels::QkNormRope {
            eps: self.rms_eps,
            n_q: h as u32,
            n_kv: kv_h as u32,
            seq_len: seq_len as u32,
        };
        let qkvr = qkv
            .apply_op3_no_bwd(&self.qk_w, &cs, &op)?
            .reshape((b, l, h + 2 * kv_h, d))?;
        let q = qkvr.narrow(2, 0, h)?;
        let k = qkvr.narrow(2, h, kv_h)?;
        let v = qkvr.narrow(2, h + kv_h, kv_h)?;
        // Scale is folded into q_norm's weight, so softmax_scale = 1.
        let ctx = candle_flash_attn::flash_attn(&q, &k, &v, 1.0, true)?;
        ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kv_h, d) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let g = self.num_kv_groups;

        let qkv = x.apply(&self.qkv)?;
        let q = qkv.narrow(D::Minus1, 0, h * d)?.contiguous()?;
        let k = qkv.narrow(D::Minus1, h * d, kv_h * d)?.contiguous()?;
        let v = qkv.narrow(D::Minus1, (h + kv_h) * d, kv_h * d)?.contiguous()?;

        // Stay in the contiguous (b, l, heads, d) projection layout for the
        // per-head RMSNorm (Qwen3's signature detail; q_norm carries the
        // folded attention scale) and RoPE — flatten and rope_thd are then
        // copy-free.
        let q = self.q_norm.forward(&q.reshape((b * l * h, d))?)?.reshape((b, l, h, d))?;
        let k = self.k_norm.forward(&k.reshape((b * l * kv_h, d))?)?.reshape((b, l, kv_h, d))?;
        let (q, k) = rotary.apply_thd(&q, &k)?;
        let v = v.reshape((b, l, kv_h, d))?;

        // Flash path without the custom kernels (e.g. f16): FA2 consumes the
        // (b, l, heads, d) layout directly, GQA handled in-kernel, no mask.
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
    ln1_w: Tensor,
    ln2_w: Tensor,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln1_w = vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?;
        let ln2_w = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?;
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            ln1: RmsNorm::new(ln1_w.clone(), cfg.rms_norm_eps),
            ln2: RmsNorm::new(ln2_w.clone(), cfg.rms_norm_eps),
            ln1_w,
            ln2_w,
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
    norm_w: Tensor,
    rotary: Rotary,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    rms_eps: f64,
    fused_ok: bool,
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
        let norm_w = root.pp("norm").get(cfg.hidden_size, "weight")?;
        // The fused kernels assume head_dim 128 (the whole Qwen3-Embedding
        // family), bias-free projections, and even hidden/intermediate
        // sizes for bf162 vectorization.
        let fused_ok = cfg.head_dim() == 128
            && !cfg.attention_bias
            && cfg.hidden_size % 2 == 0
            && cfg.intermediate_size % 2 == 0;
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(norm_w.clone(), cfg.rms_norm_eps),
            norm_w,
            rotary: Rotary::new(cfg, vb.dtype(), vb.device())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            rms_eps: cfg.rms_norm_eps,
            fused_ok,
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

    /// Fused trunk: custom add+rmsnorm carries (residual, normed) across
    /// blocks so no separate residual adds or norms ever touch DRAM alone.
    #[cfg(feature = "embed-cuda-flash")]
    fn forward_fused(&self, h0: Tensor, max_len: usize) -> Result<Tensor> {
        let eps = self.rms_eps as f32;
        let n = self.layers.len();
        let mut res = h0.clone();
        let mut normed =
            h0.apply_op2_no_bwd(&self.layers[0].ln1_w, &kernels::RmsNormOnly { eps })?;
        for (i, layer) in self.layers.iter().enumerate() {
            let a = layer.self_attn.forward_fused(&normed, &self.rotary, max_len)?;
            let both =
                res.apply_op3_no_bwd(&a, &layer.ln2_w, &kernels::AddRmsNorm { eps })?;
            res = both.i(0)?;
            normed = both.i(1)?;
            let m = layer.mlp.forward_fused(&normed)?;
            let next_w = if i + 1 < n { &self.layers[i + 1].ln1_w } else { &self.norm_w };
            let both = res.apply_op3_no_bwd(&m, next_w, &kernels::AddRmsNorm { eps })?;
            res = both.i(0)?;
            normed = both.i(1)?;
        }
        Ok(normed)
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
        #[cfg(feature = "embed-cuda-flash")]
        let h = if self.fused_ok && self.dtype == DType::BF16 && Attention::use_flash(&h0) {
            self.forward_fused(h0, max_len)?
        } else {
            self.forward_composed(h0, max_len)?
        };
        #[cfg(not(feature = "embed-cuda-flash"))]
        let h = self.forward_composed(h0, max_len)?;

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

    /// The composed-op trunk (reference path: CPU, f32, or non-flash builds).
    fn forward_composed(&self, h0: Tensor, max_len: usize) -> Result<Tensor> {
        let mask = if Attention::use_flash(&h0) {
            None // FA2 applies causality in-kernel; no materialized mask.
        } else {
            Some(self.causal_mask(max_len)?)
        };
        let mut h = h0;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rotary, mask.as_ref())?;
        }
        self.norm.forward(&h)
    }
}
