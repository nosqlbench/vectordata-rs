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

#[cfg(test)]
mod packaging_tests {
    /// Read a manifest from the workspace root.
    fn manifest(rel: &str) -> String {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        std::fs::read_to_string(root.join(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"))
    }

    /// Does `toml` declare a `[[bin]]` with this name?
    fn declares_bin(toml: &str, name: &str) -> bool {
        toml.split("[[bin]]")
            .skip(1)
            .any(|section| {
                let head = section.split("[[").next().unwrap_or(section);
                let head = head.split("\n[").next().unwrap_or(head);
                head.lines().any(|l| l.trim() == format!("name = \"{name}\""))
            })
    }

    /// **Installing the umbrella installs `veks` too.**
    ///
    /// Without this the root install leaves whatever `veks` was already
    /// on PATH, which is how a ten-hour build ran on a binary two weeks
    /// older than the fix it was meant to have.
    #[test]
    fn the_root_package_installs_veks() {
        let root = manifest("Cargo.toml");
        assert!(declares_bin(&root, "vectordata"), "the umbrella's own name");
        assert!(
            declares_bin(&root, "veks"),
            "the root package must install `veks`, or `cargo install --path .` \
             silently leaves a stale one in place"
        );
    }

    /// **`cargo install veks` from crates.io must keep working.**
    ///
    /// The umbrella shares the binary name with the published `veks`
    /// crate, which is only safe because the two can never meet on
    /// crates.io: this package is unpublishable, and the veks crate
    /// keeps its own bin. Removing either half breaks one of the two
    /// install paths, and only locally — a registry install would fail
    /// for everyone else first.
    #[test]
    fn the_published_veks_crate_still_provides_its_own_binary() {
        let veks = manifest("veks/Cargo.toml");
        assert!(
            declares_bin(&veks, "veks"),
            "the veks crate must keep its own bin for `cargo install veks`"
        );
        assert!(
            !veks.lines().any(|l| l.trim() == "publish = false"),
            "the veks crate has to stay publishable"
        );

        let root = manifest("Cargo.toml");
        assert!(
            root.lines().any(|l| l.trim() == "publish = false"),
            "the umbrella must stay unpublishable, or it would collide with \
             the veks crate on crates.io rather than only in a local install"
        );
    }

    /// Both names are the same program, not two builds of it — argv0
    /// dispatch is what makes the second name work, so they must share
    /// one source path.
    #[test]
    fn both_root_binaries_are_the_same_source() {
        let root = manifest("Cargo.toml");
        let paths: Vec<&str> = root
            .split("[[bin]]")
            .skip(1)
            .filter_map(|s| {
                let head = s.split("\n[").next().unwrap_or(s);
                head.lines().find(|l| l.trim().starts_with("path = "))
            })
            .collect();
        assert_eq!(paths.len(), 2, "expected exactly two root bins: {paths:?}");
        assert!(
            paths.iter().all(|p| p.trim() == "path = \"src/main.rs\""),
            "both root bins must be the same multiplexer: {paths:?}"
        );
    }
}
