// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vecd` binary entry point — a thin shim over [`vecd::shell::bin_main`],
//! where the whole CLI shell lives so wrapper binaries (the workspace-root
//! `vectordata` multiplexer) can embed the identical personality.

fn main() {
    vecd::shell::bin_main(std::env::args().skip(1).collect());
}
