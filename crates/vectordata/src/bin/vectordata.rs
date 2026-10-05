// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata` — minimal admin CLI for the vectordata cache.
//!
//! A thin shim over [`vectordata::shell::bin_main`], where the whole CLI
//! shell lives. This standalone binary exists so downstream consumers of
//! the `vectordata` library can inspect and curate their local cache
//! (`vectordata cache list`, `vectordata explore`, `vectordata config
//! get`) without building or installing the larger veks toolkit; the
//! workspace-root umbrella binary embeds the identical shell as its
//! default personality.

fn main() {
    vectordata::shell::bin_main(std::env::args().skip(1).collect());
}
