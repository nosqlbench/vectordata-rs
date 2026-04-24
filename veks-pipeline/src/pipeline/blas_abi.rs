// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! BLAS ABI / threading workarounds for the faiss-sys static-MKL bug.
//!
//! ## The bug
//!
//! When this crate is compiled with the `faiss` feature, `faiss-sys`
//! (v0.7.0, `static` feature) pulls in a statically-linked MKL. That
//! MKL has an ABI mismatch — `FINTEGER=long` vs MKL LP64's 32-bit
//! integer argument convention — plus a second, subtler issue in the
//! tiled distance-computation paths. Either one can silently corrupt
//! `cblas_sgemm` output.
//!
//! The corruption is **process-wide**, not FAISS-internal. Once this
//! binary links the static MKL, every `cblas_sgemm` symbol resolves
//! to the poisoned copy — FAISS's own search path, our
//! `compute knn-blas`, `verify knn-consolidated`'s sgemm scan, and
//! anywhere else anyone calls sgemm. Multi-threaded MKL amplifies
//! the bug: empirically, sgemm goes wrong starting at `dim ≥ 384`
//! for small batch sizes when MKL threads > 1.
//!
//! The only complete mitigation we have is **force single-threaded
//! BLAS at the top of every `execute()` that might touch an sgemm
//! path**. That's what this module provides. See
//! [`docs/design/faiss-blas-abi-bug.md`](../../../docs/design/faiss-blas-abi-bug.md)
//! for the full background, alternatives considered, and timeline.
//!
//! ## Calling convention
//!
//! Every command whose `execute()` might eventually call `sgemm`
//! (directly or via a helper) calls [`set_single_threaded_if_faiss`]
//! as its first line. The function is a no-op when `faiss` feature
//! is off — if faiss isn't compiled in, the MKL static library is
//! not linked, and whatever BLAS the binary resolved to (likely
//! OpenBLAS or system MKL dynamically) is trusted to be correct
//! under its default threading.
//!
//! ## Why env vars
//!
//! MKL, OpenBLAS, BLIS, and Apple Accelerate all honor per-process
//! thread-count env vars, and they honor them lazily (read on each
//! call, not just at library load). Setting them before any BLAS
//! work in the process is sufficient. `unsafe` because
//! [`std::env::set_var`] can race with `getenv` from other threads;
//! we call this at the top of `execute()` before any worker threads
//! are spawned, so the race window is empty.

/// When the `faiss` feature is compiled in, force every supported
/// BLAS library to single-threaded mode by setting the relevant
/// environment variables. No-op otherwise.
///
/// Safe to call multiple times from the same process — it's
/// idempotent, and the env-var setter is cheap.
pub fn set_single_threaded_if_faiss() {
    #[cfg(feature = "faiss")]
    {
        for var in &[
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ] {
            // Safety: `set_var` races with concurrent `getenv`. Every
            // caller invokes this at the top of `execute()` before
            // spawning any BLAS work, so the race window is empty.
            // On the first call in a process, nothing else has read
            // the BLAS thread count yet; subsequent calls are idempotent.
            unsafe { std::env::set_var(var, "1"); }
        }
    }
}
