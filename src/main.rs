// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata` — umbrella multiplexer binary for the vectordata-rs
//! workspace, so the whole toolkit installs with one
//! `cargo install --path .` at the project root.
//!
//! Personalities (each embedded from its crate's `shell` module, so
//! behavior is byte-identical to the standalone binaries):
//!
//! - `veks` — vector dataset toolkit
//! - `vecd` — dataset gateway daemon / admin tool
//! - `slab` — slab file toolkit (alias: `slabtastic`)
//!
//! Dispatch is by first argument (`vectordata veks run …`) or by argv0:
//! a symlink or hardlink named after a personality behaves exactly like
//! that binary (`ln -s vectordata veks && ./veks --help`), which also
//! keeps shell-completion registration working per personality.

use std::path::Path;

/// Run `name`'s personality with `args` (everything after the personality
/// word). Returns false when `name` is not a personality.
fn run_personality(name: &str, args: Vec<String>) -> bool {
    match name {
        "veks" => veks::shell::bin_main(args),
        "vecd" => vecd::shell::bin_main(args),
        "slab" | "slabtastic" => slabtastic::shell::bin_main(args),
        _ => return false,
    }
    true
}

fn print_help() {
    println!(
        "vectordata — umbrella binary for the vectordata-rs toolkit\n\
         \n\
         Usage:\n\
         \x20 vectordata <personality> [args...]\n\
         \n\
         Personalities:\n\
         \x20 veks   Vector dataset toolkit (prepare, run, check, publish, ...)\n\
         \x20 vecd   Dataset gateway daemon / admin tool\n\
         \x20 slab   Slab file toolkit (alias: slabtastic)\n\
         \n\
         A symlink or hardlink named after a personality dispatches to it\n\
         directly, exactly like the standalone binary:\n\
         \x20 ln -s vectordata veks && ./veks --help\n\
         \n\
         Use `vectordata <personality> --help` for that tool's own help."
    );
}

fn print_version() {
    println!("vectordata-rs {}", env!("CARGO_PKG_VERSION"));
    println!("  veks {}", veks::shell::version_line());
    println!("  vecd {}", vecd::shell::version_line());
    println!("  slab {}", slabtastic::shell::version_line());
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

    match args.first().map(String::as_str) {
        None => print_help(),
        Some("--help") | Some("-h") | Some("help") => print_help(),
        Some("--version") | Some("-V") => print_version(),
        Some(name) => {
            if !run_personality(name, args[1..].to_vec()) {
                eprintln!(
                    "vectordata: unknown personality '{}' — expected veks, vecd, or slab",
                    name
                );
                std::process::exit(2);
            }
        }
    }
}
