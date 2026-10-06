# veks-completion-derive

`#[derive(VeksCli)]` for
[`veks-completion`](https://crates.io/crates/veks-completion), the
in-tree replacement for clap used by `veks`, `vectordata`, `vecd` and
`slabtastic`.

From one annotated declaration the derive produces the command's
`CommandSpec`, which drives parsing, `--help` and shell completion
alike, and the typed extraction of the parsed arguments.

```rust,ignore
use veks_completion_derive::VeksCli;

#[derive(VeksCli)]
struct Fetch {
    /// Dataset name to fetch.
    name: String,
    /// Profiles to fetch (repeatable).
    #[arg(long)]
    profile: Vec<String>,
    /// Report what would be fetched without fetching.
    #[arg(long)]
    dry_run: bool,
}
```

- A field with `#[arg(long)]` or `#[arg(short)]` is an option, a `bool`
  is a flag, and a bare field is a positional.
- `Vec<T>` is repeatable, `Option<T>` optional, and a bare `T` required
  unless it has `#[arg(default = "…")]`.
- `#[command(flatten)]` pulls in another `VeksCli` struct's options, and
  `#[command(subcommand)]` on an enum-typed field wires up subcommands.
- On an enum, each variant becomes a subcommand.

Depend on both crates: this one for the derive, and `veks-completion`
for the `VeksCli` trait, the `CommandSpec` model and the parser and
completion runtime the generated code calls into. The full attribute
reference is in the crate documentation.

License: Apache-2.0
