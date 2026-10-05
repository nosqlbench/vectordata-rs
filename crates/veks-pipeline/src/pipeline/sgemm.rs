// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Single-precision matrix multiply for the sgemm-based KNN scans.
//!
//! The one shape the scans need is `C = alpha · A · Bᵀ` with every matrix
//! row-major: `A` is a block of queries (`m × k`), `B` a block of base
//! vectors (`n × k`), and `C` the `m × n` score matrix. [`sgemm_nt`]
//! computes it on either backend:
//!
//! - [`MatmulBackend::Gemm`] (default): the pure-Rust `gemm` crate (the
//!   engine under faer), with runtime ISA dispatch and rayon threading.
//!   Available on every target.
//! - [`MatmulBackend::System`]: `cblas_sgemm` from the system BLAS
//!   (OpenBLAS, MKL, Accelerate), with the `blas-system` feature on unix.
//!   This is the routine numpy and FAISS call, kept for kernel-level
//!   parity with them.
//!
//! The two backends sum in different orders, so their scores differ in
//! the last bits; the commands key their caches by backend and the f64
//! rerank makes their final outputs agree.

use crate::pipeline::command::{OptionDesc, OptionRole, Options};

/// Which implementation computes the score matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulBackend {
    /// The pure-Rust `gemm` crate.
    Gemm,
    /// The system BLAS `cblas_sgemm` (`blas-system` feature, unix).
    System,
}

impl MatmulBackend {
    /// The option name, shared by every command that takes it.
    pub const OPTION: &'static str = "backend";

    /// Whether this build can call the system BLAS.
    pub const SYSTEM_AVAILABLE: bool = cfg!(all(feature = "blas-system", unix));

    /// The backend named by the `backend` option, [`Gemm`] when unset.
    ///
    /// [`Gemm`]: MatmulBackend::Gemm
    pub fn from_options(options: &Options) -> Result<Self, String> {
        Self::parse(options.get(Self::OPTION).unwrap_or("gemm"))
    }

    /// Parse a backend name. `system` errors in a build without the
    /// system BLAS.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "gemm" => Ok(MatmulBackend::Gemm),
            "system" if Self::SYSTEM_AVAILABLE => Ok(MatmulBackend::System),
            "system" => Err(
                "backend 'system' is not available: this binary was built without the `blas-system` feature (or for a non-unix target)".into(),
            ),
            other => Err(format!("unknown backend '{other}': expected gemm or system")),
        }
    }

    /// The backend's name as written in logs and engine identifiers.
    pub fn name(self) -> &'static str {
        match self {
            MatmulBackend::Gemm => "gemm",
            MatmulBackend::System => "system",
        }
    }

    /// The `backend` option, as each command that takes it declares it.
    pub fn option_desc() -> OptionDesc {
        OptionDesc {
            name: Self::OPTION.to_string(),
            type_name: "enum".to_string(),
            required: false,
            default: Some("gemm".to_string()),
            description: "Matrix-multiply backend: gemm (pure Rust, default) or system (system BLAS cblas_sgemm; needs the `blas-system` build feature)".to_string(),
            extended_description: Some(
                "gemm: the pure-Rust `gemm` crate, available on every target, threaded with rayon.\n\
                 system: the system BLAS (OpenBLAS / MKL / Accelerate) `cblas_sgemm` — the routine \
                 numpy and FAISS call — for kernel-level parity with them. Unix only, and only in \
                 builds with the `blas-system` feature. The two backends' results are cached separately."
                    .to_string(),
            ),
            role: OptionRole::Config,
        }
    }
}

/// `C = alpha · A · Bᵀ`, row-major, overwriting `C`.
///
/// `a` is `m × k` with row stride `lda`, `b` is `n × k` with row stride
/// `ldb`, `c` is `m × n` with row stride `ldc`. `threads` bounds the
/// `gemm` backend's rayon parallelism (0 = rayon's pool size); the
/// system backend uses the BLAS library's own threading.
///
/// Panics if a slice is too short for its shape.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_nt(
    backend: MatmulBackend,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    threads: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(lda >= k && ldb >= k && ldc >= n, "sgemm_nt: a stride is shorter than its row");
    assert!(a.len() >= (m - 1) * lda + k, "sgemm_nt: A is too short for {m}×{k}");
    assert!(b.len() >= (n - 1) * ldb + k, "sgemm_nt: B is too short for {n}×{k}");
    assert!(c.len() >= (m - 1) * ldc + n, "sgemm_nt: C is too short for {m}×{n}");
    match backend {
        MatmulBackend::Gemm => gemm_nt(m, n, k, alpha, a, lda, b, ldb, c, ldc, threads),
        MatmulBackend::System => system_nt(m, n, k, alpha, a, lda, b, ldb, c, ldc),
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_nt(
    m: usize, n: usize, k: usize, alpha: f32,
    a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize,
    threads: usize,
) {
    let parallelism = if threads == 1 { gemm::Parallelism::None } else { gemm::Parallelism::Rayon(threads) };
    // `gemm` computes dst := alpha·dst + beta·lhs·rhs with explicit
    // column (cs) and row (rs) strides. With read_dst = false the old
    // contents of C are ignored, so its `beta` is our `alpha`.
    //
    //   lhs = A    (m × k): row stride lda, column stride 1
    //   rhs = Bᵀ   (k × n): element (p, j) is b[j·ldb + p], so moving
    //                        down a row (p) is stride 1 and across a
    //                        column (j) is stride ldb
    //   dst = C    (m × n): row stride ldc, column stride 1
    //
    // SAFETY: the asserts in `sgemm_nt` bound every element the strides
    // can reach inside its slice; `c` is exclusively borrowed.
    unsafe {
        gemm::gemm(
            m, n, k,
            c.as_mut_ptr(), 1, ldc as isize,
            false,
            a.as_ptr(), 1, lda as isize,
            b.as_ptr(), ldb as isize, 1,
            0.0f32, alpha,
            false, false, false,
            parallelism,
        );
    }
}

#[cfg(all(feature = "blas-system", unix))]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(all(feature = "blas-system", unix))]
#[allow(clippy::too_many_arguments)]
fn system_nt(
    m: usize, n: usize, k: usize, alpha: f32,
    a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize,
) {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;
    let dim = |v: usize| i32::try_from(v).expect("sgemm dimension exceeds the CBLAS int range");
    // SAFETY: the asserts in `sgemm_nt` bound every element CBLAS reads
    // or writes inside its slice.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR, NO_TRANS, TRANS,
            dim(m), dim(n), dim(k),
            alpha,
            a.as_ptr(), dim(lda),
            b.as_ptr(), dim(ldb),
            0.0,
            c.as_mut_ptr(), dim(ldc),
        );
    }
}

#[cfg(not(all(feature = "blas-system", unix)))]
#[allow(clippy::too_many_arguments)]
fn system_nt(
    _: usize, _: usize, _: usize, _: f32,
    _: &[f32], _: usize, _: &[f32], _: usize, _: &mut [f32], _: usize,
) {
    unreachable!("MatmulBackend::parse never yields System without the system BLAS");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[i * k + p] as f64 * b[j * k + p] as f64;
                }
                c[i * n + j] = (alpha as f64 * s) as f32;
            }
        }
        c
    }

    fn data(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    fn backends() -> Vec<MatmulBackend> {
        let mut v = vec![MatmulBackend::Gemm];
        if MatmulBackend::SYSTEM_AVAILABLE {
            v.push(MatmulBackend::System);
        }
        v
    }

    #[test]
    fn matches_naive_product_for_awkward_shapes() {
        for backend in backends() {
            for &(m, n, k) in &[(1, 1, 1), (3, 5, 7), (17, 33, 384), (64, 129, 100), (2048, 40, 16)] {
                let a = data(m * k, 1);
                let b = data(n * k, 2);
                for alpha in [1.0f32, -2.0] {
                    let mut c = vec![f32::NAN; m * n];
                    sgemm_nt(backend, m, n, k, alpha, &a, k, &b, k, &mut c, n, 0);
                    let want = naive(m, n, k, alpha, &a, &b);
                    for (i, (g, w)) in c.iter().zip(&want).enumerate() {
                        let tol = 1e-4 * (k as f32).sqrt() * alpha.abs();
                        assert!((g - w).abs() <= tol, "{backend:?} {m}×{n}×{k} α={alpha} [{i}]: {g} vs {w}");
                    }
                }
            }
        }
    }

    #[test]
    fn strided_operands_and_output() {
        // Row strides wider than the row: the padding must be neither
        // read into the product nor written.
        let (m, n, k) = (4, 6, 5);
        let (lda, ldb, ldc) = (8, 7, 9);
        let mut a = vec![99.0f32; m * lda];
        let mut b = vec![99.0f32; n * ldb];
        let dense_a = data(m * k, 3);
        let dense_b = data(n * k, 4);
        for i in 0..m {
            a[i * lda..i * lda + k].copy_from_slice(&dense_a[i * k..(i + 1) * k]);
        }
        for j in 0..n {
            b[j * ldb..j * ldb + k].copy_from_slice(&dense_b[j * k..(j + 1) * k]);
        }
        let want = naive(m, n, k, 1.0, &dense_a, &dense_b);
        for backend in backends() {
            let mut c = vec![-7.0f32; m * ldc];
            sgemm_nt(backend, m, n, k, 1.0, &a, lda, &b, ldb, &mut c, ldc, 1);
            for i in 0..m {
                for j in 0..n {
                    assert!((c[i * ldc + j] - want[i * n + j]).abs() < 1e-4, "{backend:?} ({i},{j})");
                }
                assert!(c[i * ldc + n..(i + 1) * ldc].iter().all(|&v| v == -7.0), "{backend:?} wrote padding");
            }
        }
    }

    #[test]
    fn backend_option_parses() {
        assert_eq!(MatmulBackend::from_options(&Options::new()), Ok(MatmulBackend::Gemm));
        assert_eq!(MatmulBackend::parse("GEMM"), Ok(MatmulBackend::Gemm));
        assert!(MatmulBackend::parse("mkl").unwrap_err().contains("unknown backend"));
        let system = MatmulBackend::parse("system");
        if MatmulBackend::SYSTEM_AVAILABLE {
            assert_eq!(system, Ok(MatmulBackend::System));
        } else {
            assert!(system.unwrap_err().contains("blas-system"));
        }
    }
}
