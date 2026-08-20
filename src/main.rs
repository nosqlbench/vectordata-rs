// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata` — umbrella binary for the vectordata-rs workspace, so the
//! whole toolkit installs with one `cargo install --path .` at the
//! project root.
//!
//! The **default personality is the `vectordata` CLI itself** (cache
//! admin, datasets browsing, config, the `explore` TUI) — embedded from
//! `vectordata::shell`, so this binary is a strict superset of the
//! crate's standalone `vectordata` bin: every existing invocation
//! (`vectordata explore`, `vectordata cache list`, …) behaves
//! identically. On top of that it multiplexes the workspace's other CLI
//! personalities, each embedded from its crate's `shell` module:
//!
//! - `veks` — vector dataset toolkit
//! - `vecd` — dataset gateway daemon / admin tool
//! - `slab` — slab file toolkit (alias: `slabtastic`)
//!
//! Dispatch: a first argument naming a personality (`vectordata veks
//! run …`), or argv0 — a symlink or hardlink named after a personality
//! behaves exactly like that binary (`ln -s vectordata veks`), which
//! also keeps shell-completion registration working per personality.
//! Anything else routes to the default vectordata personality.

use std::path::Path;

/// Run `name`'s personality with `args` (everything after the personality
/// word). Returns false when `name` is not an alternate personality.
fn run_personality(name: &str, args: Vec<String>) -> bool {
    match name {
        "veks" => veks::shell::bin_main(args),
        "vecd" => vecd::shell::bin_main(args),
        "slab" | "slabtastic" => slabtastic::shell::bin_main(args),
        _ => return false,
    }
    true
}

fn main() {
    let mut argv = std::env::args();
    let argv0 = argv.next().unwrap_or_default();
    let args: Vec<String> = argv.collect();

    // argv0 dispatch: honor personality symlinks/hardlinks.
    let base = Path::new(&argv0)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if base != "vectordata" && run_personality(base, args.clone()) {
        return;
    }

    // First-argument dispatch: `vectordata veks …` etc.
    if let Some(first) = args.first()
        && run_personality(first, args[1..].to_vec())
    {
        return;
    }

    // Default personality: the vectordata CLI itself — its own help,
    // version, completions, and error handling apply unchanged.
    vectordata::shell::bin_main(args);
}
