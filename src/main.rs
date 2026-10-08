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
        let veks = manifest("crates/veks/Cargo.toml");
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

    /// Dependencies that compile or link native code for SIMD or linear
    /// algebra: C/C++ kernel crates and BLAS bindings.
    const NATIVE_KERNEL_DEPS: &[&str] = &["simsimd", "faiss", "faiss-sys", "openblas-src", "blas-src", "cblas-sys", "intel-mkl-src"];

    /// The `[dependencies]` line for `name` in a manifest, if any.
    fn dependency_line<'a>(toml: &'a str, name: &str) -> Option<&'a str> {
        let deps = toml.split("\n[dependencies]").nth(1)?;
        let deps = deps.split("\n[").next().unwrap_or(deps);
        deps.lines().find(|l| l.trim_start().starts_with(&format!("{name} ")) || l.trim_start().starts_with(&format!("{name}=")))
    }

    /// The value list of feature `name` in a manifest's `[features]`.
    fn feature_list(toml: &str, name: &str) -> Vec<String> {
        let Some(features) = toml.split("\n[features]").nth(1) else { return vec![] };
        let features = features.split("\n[").next().unwrap_or(features);
        let Some(start) = features.find(&format!("\n{name} = [")) else { return vec![] };
        let rest = &features[start..];
        let list = &rest[rest.find('[').unwrap() + 1..rest.find(']').unwrap()];
        list.split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// **Every manifest's `readme` names a file that exists.**
    ///
    /// `cargo publish` refuses a package whose readme path is dangling,
    /// and a relative path breaks silently when a crate moves: the
    /// crates/ restructuring left two of them pointing at a nonexistent
    /// `crates/README.md` until a release tripped over it.
    #[test]
    fn every_readme_path_resolves() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let crates = std::fs::read_dir(root.join("crates")).expect("crates/");
        for dir in crates.flatten().map(|e| e.path()).filter(|p| p.join("Cargo.toml").is_file()) {
            let toml = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
            let package = toml.split("\n[").next().unwrap_or(&toml);
            let Some(line) = package.lines().find(|l| l.trim_start().starts_with("readme")) else {
                continue;
            };
            let value = line.split('=').nth(1).unwrap().trim().trim_matches('"');
            if value == "false" {
                continue;
            }
            assert!(
                dir.join(value).is_file(),
                "{}: readme = \"{value}\" does not exist",
                dir.join("Cargo.toml").display()
            );
        }
    }

    /// **A published crate's README links absolutely.**
    ///
    /// crates.io and docs.rs render the README outside the repository,
    /// where a relative link (`../../docs/…`) is dead. Unpublished crates
    /// (`publish = false`) are exempt.
    #[test]
    fn published_readmes_have_no_relative_links() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let crates = std::fs::read_dir(root.join("crates")).expect("crates/");
        for dir in crates.flatten().map(|e| e.path()).filter(|p| p.join("Cargo.toml").is_file()) {
            let toml = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
            if toml.lines().any(|l| l.trim() == "publish = false") {
                continue;
            }
            let Ok(readme) = std::fs::read_to_string(dir.join("README.md")) else { continue };
            for (n, line) in readme.lines().enumerate() {
                for target in line.split("](").skip(1).filter_map(|rest| rest.split(')').next()) {
                    let absolute = target.starts_with('#')
                        || target.contains("://")
                        || target.starts_with("mailto:");
                    assert!(
                        absolute,
                        "{}:{}: relative link `{target}` is dead on crates.io",
                        dir.join("README.md").display(),
                        n + 1
                    );
                }
            }
        }
    }

    /// **`vectordata` ships its agent guide** (SRD DX-22, DX-29).
    ///
    /// `AGENTS.md` is read from the registry copy of the crate, so it has
    /// to be in the published package, not only in the repository; the
    /// examples it and the crate docs point at have to be there too.
    /// That its links resolve is checked by rustdoc, which renders it as
    /// `vectordata::_agents`. Any other crate that grows an `AGENTS.md`
    /// must ship it the same way.
    #[test]
    fn agent_guides_are_in_the_published_packages() {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let crates = std::fs::read_dir(root.join("crates")).expect("crates/");
        let mut checked = Vec::new();
        for dir in crates.flatten().map(|e| e.path()).filter(|p| p.join("AGENTS.md").is_file()) {
            let name = dir.file_name().unwrap().to_string_lossy().to_string();
            let out = std::process::Command::new(&cargo)
                .current_dir(root)
                .args(["package", "--list", "--allow-dirty", "--offline", "-p", &name])
                .output()
                .expect("run cargo package --list");
            assert!(out.status.success(), "{name}: {}", String::from_utf8_lossy(&out.stderr));
            let files = String::from_utf8_lossy(&out.stdout);
            let listed: Vec<&str> = files.lines().collect();
            assert!(listed.contains(&"AGENTS.md"), "{name} does not package its AGENTS.md");
            if dir.join("examples").is_dir() {
                assert!(
                    listed.iter().any(|f| f.starts_with("examples/")),
                    "{name} does not package its examples"
                );
            }
            checked.push(name);
        }
        assert!(checked.contains(&"vectordata".to_string()), "vectordata must have an AGENTS.md");
    }

    /// **The default build compiles no native SIMD or BLAS code (SRD SK-1).**
    ///
    /// Every such dependency is optional in every manifest that names
    /// it, and no default feature — directly or through `knnutils`,
    /// which `veks` enables by default — turns one on. Native backends
    /// are opt-in features (`simsimd`, `blas-system`, `faiss`).
    #[test]
    fn default_features_pull_no_native_kernels() {
        for rel in [
            "Cargo.toml",
            "crates/vectordata/Cargo.toml",
            "crates/veks-simd/Cargo.toml",
            "crates/veks-core/Cargo.toml",
            "crates/veks-pipeline/Cargo.toml",
            "crates/veks/Cargo.toml",
        ] {
            let toml = manifest(rel);
            for dep in NATIVE_KERNEL_DEPS {
                if let Some(line) = dependency_line(&toml, dep) {
                    assert!(line.contains("optional = true"), "{rel}: `{dep}` must be optional: {line}");
                }
            }
            let mut enabled = feature_list(&toml, "default");
            enabled.extend(feature_list(&toml, "knnutils"));
            for f in &enabled {
                for native in ["simsimd", "faiss", "blas-system"] {
                    assert!(
                        !f.contains(native),
                        "{rel}: a default-path feature enables `{f}`; `{native}` must stay opt-in"
                    );
                }
            }
        }
        // The system BLAS link is tied to `blas-system`, not `knnutils`.
        let build_rs = manifest("crates/veks-pipeline/build.rs");
        let link = build_rs.find("rustc-link-lib=blas").expect("the blas link directive");
        let gate = build_rs[..link].rfind("#[cfg(feature = ").expect("a feature gate on the link");
        assert!(build_rs[gate..link].starts_with("#[cfg(feature = \"blas-system\")]"));
    }

    /// **`vectordata` with `cli` resolves no native kernel crate (SRD
    /// acceptance case 9).** Checked against the resolved graph, so a
    /// transitive path the manifest test cannot see is still caught.
    #[test]
    fn vectordata_cli_graph_has_no_native_kernels() {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let out = std::process::Command::new(cargo)
            .current_dir(root)
            .args([
                "tree", "--offline", "--locked", "-p", "vectordata", "--features", "cli",
                "-e", "normal,build", "--prefix", "none", "--format", "{p}",
            ])
            .output()
            .expect("run cargo tree");
        assert!(out.status.success(), "cargo tree failed: {}", String::from_utf8_lossy(&out.stderr));
        let tree = String::from_utf8_lossy(&out.stdout);
        let names: Vec<&str> = tree.lines().filter_map(|l| l.split_whitespace().next()).collect();
        assert!(names.contains(&"veks-simd"), "explore must use the native kernels");
        for dep in NATIVE_KERNEL_DEPS {
            assert!(!names.contains(dep), "vectordata --features cli resolves `{dep}`");
        }
    }
}
