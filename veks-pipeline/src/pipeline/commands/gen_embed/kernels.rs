// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Launchers for the fused bf16 CUDA kernels in `kernels.cu` (compiled to
//! PTX by build.rs under `embed-cuda-flash`). CUDA + bf16 + contiguous
//! inputs only — callers gate on those and keep the composed candle ops as
//! the reference path everywhere else, so `cpu_fwd` is unreachable.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use half::bf16;
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Result, Shape};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gen_embed_kernels.ptx"));
const MODULE: &str = "gen_embed_fused";
const BLOCK: u32 = 256;

fn contiguous_bf16<'a>(
    op: &'static str,
    s: &'a CudaStorage,
    l: &Layout,
) -> Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, bf16>> {
    let (start, _) = l.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(format!("{op}: input must be contiguous"))
    })?;
    Ok(s.as_cuda_slice::<bf16>()?.slice(start..))
}

/// out[0] = x + a (next residual); out[1] = rmsnorm(x + a) * w.
pub(super) struct AddRmsNorm {
    pub eps: f32,
}

impl CustomOp3 for AddRmsNorm {
    fn name(&self) -> &'static str {
        "fused-add-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused-add-rmsnorm is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x_s: &CudaStorage,
        x_l: &Layout,
        a_s: &CudaStorage,
        a_l: &Layout,
        w_s: &CudaStorage,
        w_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = x_s.device().clone();
        let dims = x_l.shape().dims();
        let d = *dims.last().unwrap();
        let rows = x_l.shape().elem_count() / d;
        let x = contiguous_bf16(self.name(), x_s, x_l)?;
        let a = contiguous_bf16(self.name(), a_s, a_l)?;
        let w = contiguous_bf16(self.name(), w_s, w_l)?;

        let elems = x_l.shape().elem_count();
        let out = unsafe { dev.alloc::<bf16>(2 * elems)? };
        let out_sum = out.slice(0..elems);
        let out_norm = out.slice(elems..);

        let func = dev.get_or_load_custom_func("add_rmsnorm_bf16", MODULE, PTX)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_half = (d / 2) as u32;
        let mut b = func.builder();
        b.arg(&out_sum);
        b.arg(&out_norm);
        b.arg(&x);
        b.arg(&a);
        b.arg(&w);
        b.arg(&d_half);
        b.arg(&self.eps);
        unsafe { b.launch(cfg) }.w()?;

        let mut shape = vec![2usize];
        shape.extend_from_slice(dims);
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev,
            },
            Shape::from_dims(&shape),
        ))
    }
}

/// rmsnorm(x) * w — the layer-0 entry norm.
pub(super) struct RmsNormOnly {
    pub eps: f32,
}

impl CustomOp2 for RmsNormOnly {
    fn name(&self) -> &'static str {
        "fused-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused-rmsnorm is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x_s: &CudaStorage,
        x_l: &Layout,
        w_s: &CudaStorage,
        w_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = x_s.device().clone();
        let d = *x_l.shape().dims().last().unwrap();
        let rows = x_l.shape().elem_count() / d;
        let x = contiguous_bf16(self.name(), x_s, x_l)?;
        let w = contiguous_bf16(self.name(), w_s, w_l)?;

        let out = unsafe { dev.alloc::<bf16>(x_l.shape().elem_count())? };
        let func = dev.get_or_load_custom_func("rmsnorm_only_bf16", MODULE, PTX)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_half = (d / 2) as u32;
        let mut b = func.builder();
        b.arg(&out);
        b.arg(&x);
        b.arg(&w);
        b.arg(&d_half);
        b.arg(&self.eps);
        unsafe { b.launch(cfg) }.w()?;

        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev,
            },
            x_l.shape().clone(),
        ))
    }
}

/// silu(gate) * up over a fused (…, 2i) [gate | up] projection.
pub(super) struct SiluMul;

impl CustomOp1 for SiluMul {
    fn name(&self) -> &'static str {
        "fused-silu-mul"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused-silu-mul is CUDA-only")
    }

    fn cuda_fwd(&self, gu_s: &CudaStorage, gu_l: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = gu_s.device().clone();
        let dims = gu_l.shape().dims();
        let two_i = *dims.last().unwrap();
        let i = two_i / 2;
        let rows = (gu_l.shape().elem_count() / two_i) as u64;
        let gu = contiguous_bf16(self.name(), gu_s, gu_l)?;

        let out = unsafe { dev.alloc::<bf16>(rows as usize * i)? };
        let func = dev.get_or_load_custom_func("silu_mul_bf16", MODULE, PTX)?;
        let i_half = (i / 2) as u32;
        let total = rows * i_half as u64;
        let grid = u32::try_from(total.div_ceil(BLOCK as u64)).unwrap_or(u32::MAX).min(65535 * 32);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&out);
        b.arg(&gu);
        b.arg(&rows);
        b.arg(&i_half);
        unsafe { b.launch(cfg) }.w()?;

        let mut shape = dims.to_vec();
        *shape.last_mut().unwrap() = i;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev,
            },
            Shape::from_dims(&shape),
        ))
    }
}

/// Per-head RMSNorm (+folded q scale) and RoPE over the fused (b, l, H·128)
/// QKV projection; v heads pass through. Output shape mirrors the input.
pub(super) struct QkNormRope {
    pub eps: f32,
    pub n_q: u32,
    pub n_kv: u32,
    pub seq_len: u32,
}

impl CustomOp3 for QkNormRope {
    fn name(&self) -> &'static str {
        "fused-qknorm-rope"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused-qknorm-rope is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        qkv_s: &CudaStorage,
        qkv_l: &Layout,
        w_s: &CudaStorage,
        w_l: &Layout,
        cs_s: &CudaStorage,
        cs_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = qkv_s.device().clone();
        let heads = (self.n_q + 2 * self.n_kv) as u64;
        let n_tokens = (qkv_l.shape().elem_count() as u64) / (heads * 128);
        let qkv = contiguous_bf16(self.name(), qkv_s, qkv_l)?;
        let w = contiguous_bf16(self.name(), w_s, w_l)?;
        let cs = contiguous_bf16(self.name(), cs_s, cs_l)?;

        let out = unsafe { dev.alloc::<bf16>(qkv_l.shape().elem_count())? };
        let func = dev.get_or_load_custom_func("qknorm_rope_bf16", MODULE, PTX)?;
        let warps = n_tokens * heads;
        let grid = u32::try_from(warps.div_ceil((BLOCK / 32) as u64)).map_err(|_| {
            candle_core::Error::Msg("qknorm-rope: grid too large".to_string())
        })?;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&out);
        b.arg(&qkv);
        b.arg(&w);
        b.arg(&cs);
        b.arg(&n_tokens);
        b.arg(&self.seq_len);
        b.arg(&self.n_q);
        b.arg(&self.n_kv);
        b.arg(&self.eps);
        unsafe { b.launch(cfg) }.w()?;

        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev,
            },
            qkv_l.shape().clone(),
        ))
    }
}
