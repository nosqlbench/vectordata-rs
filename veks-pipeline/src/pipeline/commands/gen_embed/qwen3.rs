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

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, Linear, RmsNorm, VarBuilder};
use candle_transformers::utils::repeat_kv;

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

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
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
        Ok(Self {
            q_proj: linear_maybe_bias(cfg.hidden_size, num_heads * head_dim, bias, vb.pp("q_proj"))?,
            k_proj: linear_maybe_bias(cfg.hidden_size, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?,
            v_proj: linear_maybe_bias(cfg.hidden_size, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?,
            o_proj: linear_maybe_bias(num_heads * head_dim, cfg.hidden_size, bias, vb.pp("o_proj"))?,
            q_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: head_dim * num_heads,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, l, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Per-head RMSNorm on q/k (the Qwen3 signature detail).
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = rotary.apply(&q, &k)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        ctx.transpose(1, 2)?
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

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
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
        })
    }

    /// Additive causal mask (0 on allowed, -inf on future positions), in
    /// the model dtype, shaped for broadcast over (b, heads, l, l) scores.
    fn causal_mask(&self, l: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..l)
            .flat_map(|i| (0..l).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::from_vec(mask, (1, 1, l, l), &self.device)?.to_dtype(self.dtype)
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

        let mask = self.causal_mask(max_len)?;
        let mut h = self.embed_tokens.forward(&input)?;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rotary, &mask)?;
        }
        let h = self.norm.forward(&h)?;

        // Last-token pooling at each row's final real position, then L2
        // normalization.
        let mut out = Vec::with_capacity(b);
        for (i, row) in rows.iter().enumerate() {
            let v = h.i((i, row.len() - 1))?.to_dtype(DType::F32)?;
            let norm = v.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            let v = if norm > 0.0 { (v / norm as f64)? } else { v };
            out.push(v.to_vec1::<f32>()?);
        }
        debug_assert!(out.iter().all(|v| v.len() == self.hidden_size));
        Ok(out)
    }
}

