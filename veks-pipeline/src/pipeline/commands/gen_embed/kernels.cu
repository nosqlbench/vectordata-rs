// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0
//
// Fused bf16 kernels for the Qwen3 embedding forward. Each replaces a
// chain of candle elementwise ops that ncu measured far off the DRAM
// roofline (latency-bound reductions, 32-thread-block norms, separate
// silu/mul/add passes). f32 accumulation throughout; outputs bf16.

#include <cuda_bf16.h>

extern "C" {

// ── residual add + RMSNorm, one block per row ────────────────────────────
// out_sum[row]  = x[row] + a[row]                (the next residual)
// out_norm[row] = rmsnorm(x[row] + a[row]) * w   (the next block input)
// d must be even (bf162 loads); blockDim.x = 256.
__global__ void add_rmsnorm_bf16(
    __nv_bfloat16* __restrict__ out_sum,
    __nv_bfloat16* __restrict__ out_norm,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w,
    unsigned int d_half,
    float eps)
{
    const unsigned long long row = blockIdx.x;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x) + row * d_half;
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a) + row * d_half;
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
    __nv_bfloat162* s2 = reinterpret_cast<__nv_bfloat162*>(out_sum) + row * d_half;
    __nv_bfloat162* n2 = reinterpret_cast<__nv_bfloat162*>(out_norm) + row * d_half;

    float acc = 0.f;
    for (unsigned int p = threadIdx.x; p < d_half; p += blockDim.x) {
        float2 xf = __bfloat1622float2(x2[p]);
        float2 af = __bfloat1622float2(a2[p]);
        float2 s = make_float2(xf.x + af.x, xf.y + af.y);
        s2[p] = __float22bfloat162_rn(s);
        acc += s.x * s.x + s.y * s.y;
    }
    __shared__ float warp_sums[8];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31u) == 0) warp_sums[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 8) {
        float v = warp_sums[threadIdx.x];
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor_sync(0xffu, v, off);
        if (threadIdx.x == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (2.f * (float)d_half) + eps);
    for (unsigned int p = threadIdx.x; p < d_half; p += blockDim.x) {
        float2 s = __bfloat1622float2(s2[p]);   // L2-hot re-read of the sum
        float2 wf = __bfloat1622float2(w2[p]);
        n2[p] = __float22bfloat162_rn(make_float2(s.x * rms * wf.x, s.y * rms * wf.y));
    }
}

// ── RMSNorm only (layer-0 entry), one block per row ──────────────────────
__global__ void rmsnorm_only_bf16(
    __nv_bfloat16* __restrict__ out_norm,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    unsigned int d_half,
    float eps)
{
    const unsigned long long row = blockIdx.x;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x) + row * d_half;
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
    __nv_bfloat162* n2 = reinterpret_cast<__nv_bfloat162*>(out_norm) + row * d_half;

    float acc = 0.f;
    for (unsigned int p = threadIdx.x; p < d_half; p += blockDim.x) {
        float2 xf = __bfloat1622float2(x2[p]);
        acc += xf.x * xf.x + xf.y * xf.y;
    }
    __shared__ float warp_sums[8];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31u) == 0) warp_sums[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 8) {
        float v = warp_sums[threadIdx.x];
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor_sync(0xffu, v, off);
        if (threadIdx.x == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (2.f * (float)d_half) + eps);
    for (unsigned int p = threadIdx.x; p < d_half; p += blockDim.x) {
        float2 xf = __bfloat1622float2(x2[p]);
        float2 wf = __bfloat1622float2(w2[p]);
        n2[p] = __float22bfloat162_rn(make_float2(xf.x * rms * wf.x, xf.y * rms * wf.y));
    }
}

// ── silu(gate) * up over a fused [gate | up] row layout ──────────────────
// gu rows hold gate (i elems) then up (i elems); out rows hold i elems.
__global__ void silu_mul_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gu,
    unsigned long long rows,
    unsigned int i_half)
{
    const unsigned long long total = rows * (unsigned long long)i_half;
    const __nv_bfloat162* gu2 = reinterpret_cast<const __nv_bfloat162*>(gu);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    for (unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * (unsigned long long)blockDim.x) {
        const unsigned long long row = idx / i_half;
        const unsigned int j = (unsigned int)(idx % i_half);
        const unsigned long long base = row * (unsigned long long)(2 * i_half);
        float2 g = __bfloat1622float2(gu2[base + j]);
        float2 u = __bfloat1622float2(gu2[base + i_half + j]);
        float s1 = g.x / (1.f + __expf(-g.x));
        float s2 = g.y / (1.f + __expf(-g.y));
        out2[idx] = __float22bfloat162_rn(make_float2(s1 * u.x, s2 * u.y));
    }
}

// ── per-head RMSNorm + RoPE over a fused [q | k | v] projection row ──────
// qkv rows: nq q-heads, nkv k-heads, nkv v-heads, each head_dim=128.
// One warp per (token, head): q/k heads are RMSNorm'd (q weight carries the
// folded attention scale) and rotated; v heads are copied through. wcat is
// [q_norm_w (128) | k_norm_w (128)]; cs is per-position [cos (64) | sin (64)]
// pairs stored as bf16, laid out row-major over max_len positions.
__global__ void qknorm_rope_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ wcat,
    const __nv_bfloat16* __restrict__ cs,
    unsigned long long n_tokens,
    unsigned int seq_len,
    unsigned int nq,
    unsigned int nkv,
    float eps)
{
    const unsigned int heads = nq + 2 * nkv;
    const unsigned long long warp =
        (blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x) >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned long long token = warp / heads;
    if (token >= n_tokens) return;
    const unsigned int head = (unsigned int)(warp % heads);
    const unsigned long long base = (token * heads + head) * 128ull;
    const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(qkv + base);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + base);

    if (head >= nq + nkv) { // v head: copy through
        out2[lane] = in2[lane];
        out2[32 + lane] = in2[32 + lane];
        return;
    }

    // Load pair-positions 2*lane, 2*lane+1 from each half (x1: 0..64, x2: 64..128).
    float2 x1 = __bfloat1622float2(in2[lane]);
    float2 x2 = __bfloat1622float2(in2[32 + lane]);

    float acc = x1.x * x1.x + x1.y * x1.y + x2.x * x2.x + x2.y * x2.y;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    const float rms = rsqrtf(acc / 128.f + eps);

    const __nv_bfloat162* w2 =
        reinterpret_cast<const __nv_bfloat162*>(wcat + (head < nq ? 0 : 128));
    float2 w1 = __bfloat1622float2(w2[lane]);
    float2 wv2 = __bfloat1622float2(w2[32 + lane]);
    x1 = make_float2(x1.x * rms * w1.x, x1.y * rms * w1.y);
    x2 = make_float2(x2.x * rms * wv2.x, x2.y * rms * wv2.y);

    const unsigned long long pos = token % seq_len;
    const __nv_bfloat162* cs2 = reinterpret_cast<const __nv_bfloat162*>(cs + pos * 128ull);
    float2 c = __bfloat1622float2(cs2[lane]);
    float2 s = __bfloat1622float2(cs2[32 + lane]);

    out2[lane] = __float22bfloat162_rn(
        make_float2(x1.x * c.x - x2.x * s.x, x1.y * c.y - x2.y * s.y));
    out2[32 + lane] = __float22bfloat162_rn(
        make_float2(x1.x * s.x + x2.x * c.x, x1.y * s.y + x2.y * c.y));
}

} // extern "C"
