// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Basic example: build a command tree and generate completions.
//!
//! This demonstrates how to set up veks-completion for a CLI tool with
//! subcommands, options, flags, and dynamic value providers.
//!
//! To test interactively, add to your `.bashrc`:
//! ```bash
//! eval "$(./target/debug/examples/basic completions)"
//! ```

use veks_completion::{CommandTree, Node, complete, handle_complete_env, print_bash_script};

/// Example value provider: suggests dataset names for `--dataset`.
fn complete_datasets(partial: &str, _context: &[&str]) -> Vec<String> {
    let datasets = ["sift128", "glove100", "cohere768", "ada1536"];
    datasets.iter()
        .filter(|d| partial.is_empty() || d.starts_with(partial))
        .map(|d| d.to_string())
        .collect()
}

/// Build the command tree for our example CLI.
fn build_tree() -> CommandTree {
    CommandTree::new("example")
        // Simple leaf command with options and flags
        .command("run", Node::leaf_with_flags(
            &["--input", "--output", "--threads"],
            &["--verbose", "--dry-run"],
        ))
        // Nested group → subcommand structure
        .group("compute", Node::group(vec![
            ("knn", Node::leaf_with_flags(
                &["--base", "--query", "--metric", "--dataset"],
                &["--exact"],
            ).with_value_provider("--dataset", veks_completion::fn_provider(complete_datasets))),
            ("stats", Node::leaf(&["--input", "--output"])),
        ]))
        // Hidden command (completable if typed, but not listed)
        .hidden_command("_internal", Node::leaf(&["--debug"]))
}

fn main() {
    let tree = build_tree();

    // Check for completion callback (bash sets _EXAMPLE_COMPLETE=bash)
    if handle_complete_env("example", &tree) {
        return;
    }

    // Handle "completions" subcommand to print the bash script
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("completions") {
        print_bash_script("example");
        return;
    }

    // Demonstrate programmatic completion
    println!("=== Completion Demo ===\n");

    let scenarios: Vec<(&str, Vec<&str>)> = vec![
        ("Root commands",          vec!["example", ""]),
        ("Compute subcommands",    vec!["example", "compute", ""]),
        ("KNN options",            vec!["example", "compute", "knn", ""]),
        ("--dataset values",       vec!["example", "compute", "knn", "--dataset", ""]),
        ("--dataset prefix 'co'",  vec!["example", "compute", "knn", "--dataset", "co"]),
        ("After consuming --base", vec!["example", "compute", "knn", "--base", "x.fvec", ""]),
    ];

    for (label, words) in scenarios {
        let candidates = complete(&tree, &words);
        println!("{label}:");
        println!("  input: {:?}", &words[1..]);
        println!("  candidates: {candidates:?}\n");
    }
}
