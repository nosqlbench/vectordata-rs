// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

fn main() {
    // Link system BLAS for knnutils personality commands.
    // On Ubuntu: `apt install libopenblas-dev` (or `libmkl-dev` for MKL).
    // The system's libblas.so resolves to whichever BLAS is configured
    // via update-alternatives.
    #[cfg(feature = "knnutils")]
    println!("cargo:rustc-link-lib=blas");
}
