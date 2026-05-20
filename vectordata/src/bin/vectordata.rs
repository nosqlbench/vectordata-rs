// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata` — minimal admin CLI for the vectordata cache.
//!
//! This binary is intentionally small. It exists so that downstream
//! consumers of the `vectordata` library can inspect and curate their
//! local cache (`vectordata cache list`, `vectordata cache prune-legacy`,
//! `vectordata config show`) without building or installing the
//! larger `veks` toolkit. The `veks` CLI delegates the same
//! operations into the same library entry points so there is exactly
//! one implementation of each command.
//!
//! Built by default. Library-only consumers who want to avoid pulling
//! `clap` into their build can disable the `cli` feature explicitly
//! (`vectordata = { ..., default-features = false }`).

use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{CompleteEnv, Shell};

use vectordata::cache_admin::{
    CacheEntry, CacheListing, PruneFilter, is_legacy_layout_name,
    list_entries, prune_by_filter, prune_legacy_layout,
};

#[derive(Parser)]
#[command(name = "vectordata", version, about = "vectordata cache + config admin")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Inspect, list, or curate the on-disk cache.
    Cache {
        #[command(subcommand)]
        command: CacheCmd,
    },
    /// Inspect or modify vectordata's settings.yaml.
    Config {
        #[command(subcommand)]
        command: ConfigCmd,
    },
    /// Operations on catalog-published datasets. With no
    /// subcommand, launches an interactive TUI browser of every
    /// dataset reachable through the configured catalogs.
    Datasets {
        #[command(subcommand)]
        command: Option<DatasetsCmd>,
        /// (TUI-mode only) Configuration directory containing catalogs.yaml.
        #[arg(long, default_value = "~/.config/vectordata", global = true)]
        configdir: String,
        /// (TUI-mode only) Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long, global = true)]
        catalog: Vec<String>,
        /// (TUI-mode only) Catalog URLs or paths to use *instead* of configured catalogs.
        #[arg(long = "at", global = true)]
        at: Vec<String>,
    },
    /// Print or activate tab-completion for the current shell.
    ///
    /// With no arguments, auto-detects your shell from `$SHELL` and
    /// emits a sourceable wrapper that activates completions for the
    /// current session:
    ///
    ///   eval "$(vectordata completions)"
    ///
    /// With `--shell <name>`, emits the raw completion script for that
    /// shell (bash / zsh / fish / elvish / powershell).
    #[command(long_about = "Print or activate tab-completion for the current shell.\n\
        \n\
        To activate for the current session:\n    \
        eval \"$(vectordata completions)\"\n\
        \n\
        To persist (bash):\n    \
        echo 'eval \"$(vectordata completions)\"' >> ~/.bashrc\n\
        \n\
        To persist (zsh):\n    \
        echo 'eval \"$(vectordata completions)\"' >> ~/.zshrc\n\
        \n\
        To emit a raw script for a specific shell:\n    \
        vectordata completions --shell bash > /etc/bash_completion.d/vectordata\n")]
    Completions {
        /// Emit the raw completion script for this shell instead of
        /// the auto-detected wrapper.
        #[arg(long, value_enum)]
        shell: Option<Shell>,
    },
}

#[derive(Subcommand)]
enum CacheCmd {
    /// Show every cached entry, grouped by category, with origin and size.
    List {
        /// Override the configured cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Print the full origin URL for every blob (default: summarise by host).
        #[arg(short = 'v', long)]
        verbose: bool,
    },
    /// Remove pre-1.1 `<host>` / `<host>:<port>` cache directories.
    PruneLegacy {
        /// Print what would be removed without deleting.
        #[arg(long)]
        dry_run: bool,
        /// Override the configured cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Remove content-addressed entries by origin dataset/profile.
    ///
    /// Walks every blob/http entry, reads its `origin.json` sidecar,
    /// extracts the dataset name and (when present) profile name
    /// from the URL path, and removes the entry when both filters
    /// (if provided) match. Globs use `*` (any run) and `?` (one
    /// char). At least one of `--dataset` / `--profile` is required
    /// — an empty filter would nuke the whole cache, which is what
    /// you would do by deleting `<cache_root>/{blobs,http}/` by
    /// hand if that's actually what you want.
    Prune {
        /// Glob matched against the dataset name segment of each
        /// entry's origin URL.
        #[arg(long)]
        dataset: Option<String>,
        /// Glob matched against the profile name segment.
        #[arg(long)]
        profile: Option<String>,
        /// Print what would be removed without deleting.
        #[arg(long)]
        dry_run: bool,
        /// Override the configured cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum DatasetsCmd {
    /// List datasets from configured or specified catalogs.
    ///
    /// Reads catalogs in this order: `--at` URLs/paths (when given)
    /// override everything; otherwise `--configdir`'s `catalogs.yaml`
    /// is loaded and any extra `--catalog` locations appended. Each
    /// resolved location may contain a `catalog.{json,yaml}` or a
    /// legacy `knn_entries.yaml`; both are honored.
    #[command(alias = "ls")]
    List(vectordata::datasets::list::ListArgs),

    /// Verify remote dataset access. Walks every facet declared by
    /// the dataset's profile, reads the first record via HTTP
    /// range, and reports success/failure per facet.
    Probe {
        #[command(flatten)]
        args: vectordata::datasets::probe::ProbeArgs,
        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value = "~/.config/vectordata")]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs
        #[arg(long)]
        catalog: Vec<String>,
    },

    /// Generate a curl download script for a dataset's published files.
    Curlify(vectordata::datasets::curlify::CurlifyArgs),

    /// Materialize a profile of an existing dataset into a new,
    /// self-standing dataset directory.
    ///
    /// Where `precache` brings a profile's bytes into the cache
    /// (still resolved through the parent dataset.yaml), `derive`
    /// copies them out into a fresh directory with its own
    /// dataset.yaml — including flattening any windowed views into
    /// their own files, so the result has no ties back to the
    /// donor dataset.
    Derive {
        /// Source dataset: catalog name, local directory containing
        /// `dataset.yaml`, path to a `dataset.yaml` file, or HTTPS URL.
        /// Local directories take a fast path that bypasses the
        /// runtime cache + access layer entirely.
        #[arg(long)]
        dataset: String,
        /// Profile to derive. Required.
        #[arg(long)]
        profile: String,
        /// Output directory for the new dataset.
        #[arg(long, short = 'o')]
        output: PathBuf,
        /// Override the derived dataset's name (default:
        /// `<source>-<profile>`).
        #[arg(long)]
        name: Option<String>,
        /// Overwrite the output directory if it already exists.
        #[arg(long)]
        force: bool,
        /// Configuration directory containing `catalogs.yaml`
        /// (only used when `--dataset` is a catalog name).
        #[arg(long, default_value = "~/.config/vectordata")]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long)]
        catalog: Vec<String>,
        /// Catalog URLs or paths to use *instead* of configured catalogs.
        #[arg(long = "at")]
        at: Vec<String>,
    },
    /// Download and cache every facet of a dataset profile into the
    /// configured cache directory. Renders a live per-facet +
    /// aggregate progress meter on stderr.
    Precache {
        /// `name[:profile]`, a path to a `dataset.yaml` or its
        /// containing directory, or an `http(s)://…` URL.
        spec: String,
        /// Configuration directory containing `catalogs.yaml`.
        #[arg(long, default_value = "~/.config/vectordata")]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long)]
        catalog: Vec<String>,
        /// Catalog URLs or paths to use *instead* of configured catalogs.
        #[arg(long = "at")]
        at: Vec<String>,
        /// Override cache directory location (informational; the
        /// active cache root still comes from `settings.yaml`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Show settings.yaml path, cache_dir, used space, and protect flag.
    Show,
    /// Print the configured cache_dir on a single line (scriptable).
    /// Exits 1 with an actionable error on stderr if not configured.
    GetCache,
    /// Set the cache directory in settings.yaml. Creates the directory
    /// if it doesn't exist. Refuses to overwrite a protected
    /// settings.yaml unless `--force`.
    ///
    /// The special value `largest-writable-mount` (alias `auto`)
    /// auto-resolves to the largest writable mount with `vectordata-cache`
    /// as a subdir, falling back to `$HOME/.cache/vectordata` when the
    /// largest mount is the same filesystem as `$HOME`.
    SetCache {
        /// New cache directory path, or `largest-writable-mount` / `auto`
        /// to auto-resolve.
        path: PathBuf,
        /// Overwrite existing protected settings.
        #[arg(long)]
        force: bool,
    },
    /// List writable mount points with available + total disk space,
    /// to help pick a `cache_dir`.
    ListMounts {
        /// Include mounts with less than 100 MiB free.
        #[arg(long)]
        all: bool,
    },
    /// Add a catalog source (URL or path) to catalogs.yaml.
    #[command(alias = "add")]
    AddCatalog {
        /// Catalog URL or path (e.g. https://host/path/ or /local/path).
        source: String,
    },
    /// Remove a catalog source from catalogs.yaml (by URL/path or index).
    #[command(alias = "rm")]
    RemoveCatalog {
        /// Catalog URL or path to remove.
        source: Option<String>,
        /// Remove by 1-based index (see `list-catalogs`).
        #[arg(long)]
        at: Option<usize>,
    },
    /// List configured catalog sources from catalogs.yaml.
    ListCatalogs,
}

fn main() {
    // Dynamic-completion entry: when invoked with `COMPLETE=<shell>`,
    // emit candidates and exit. The shell snippet from
    // `vectordata completions` is a one-liner that re-invokes the
    // binary with this env var set, so completion logic lives here
    // (in clap-derived metadata) rather than in a frozen script.
    CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();
    match cli.command {
        Cmd::Cache { command } => match command {
            CacheCmd::List { cache_dir, verbose } => cmd_cache_list(cache_dir, verbose),
            CacheCmd::PruneLegacy { dry_run, cache_dir } => {
                cmd_cache_prune_legacy(cache_dir, dry_run)
            }
            CacheCmd::Prune { dataset, profile, dry_run, cache_dir } => {
                cmd_cache_prune(cache_dir, dataset, profile, dry_run)
            }
        },
        Cmd::Datasets { command, configdir, catalog, at } => {
            // No subcommand → TUI browser. The `Option<DatasetsCmd>`
            // shape makes `vectordata datasets` valid on its own.
            let Some(command) = command else {
                std::process::exit(vectordata::datasets::browser::run(&configdir, &catalog, &at));
            };
            let code = match command {
                DatasetsCmd::List(args) => {
                    vectordata::datasets::list::run_args(args);
                    0
                }
                DatasetsCmd::Probe { args, configdir, catalog } => {
                    vectordata::datasets::probe::run_args(args, &configdir, &catalog, &[]);
                    0
                }
                DatasetsCmd::Curlify(args) => {
                    vectordata::datasets::curlify::run_args(args);
                    0
                }
                DatasetsCmd::Derive {
                    dataset, profile, output, name, force, configdir, catalog, at,
                } => vectordata::datasets::derive::run(
                    &dataset, &profile, &output, &configdir, &catalog, &at,
                    name.as_deref(), force),
                DatasetsCmd::Precache { spec, configdir, catalog, at, cache_dir } => {
                    vectordata::datasets::precache::run(
                        &spec, &configdir, &catalog, &at, cache_dir.as_deref())
                }
            };
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Config { command } => {
            let code = match command {
                ConfigCmd::Show => vectordata::config::show(),
                ConfigCmd::GetCache => vectordata::config::get_cache(),
                ConfigCmd::SetCache { path, force } => {
                    vectordata::config::set_cache(&path, force)
                }
                ConfigCmd::ListMounts { all } => vectordata::config::list_mounts(all),
                ConfigCmd::AddCatalog { source } => vectordata::config::add_catalog(&source),
                ConfigCmd::RemoveCatalog { source, at } => match (source, at) {
                    (Some(s), _) => vectordata::config::remove_catalog(
                        vectordata::config::RemoveCatalogSpec::Source(&s)),
                    (None, Some(n)) => vectordata::config::remove_catalog(
                        vectordata::config::RemoveCatalogSpec::Index(n)),
                    (None, None) => {
                        eprintln!("Specify a catalog URL/path or --at <index>.");
                        eprintln!("Use `vectordata config list-catalogs` to see indices.");
                        1
                    }
                },
                ConfigCmd::ListCatalogs => vectordata::config::list_catalogs(),
            };
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Completions { shell } => cmd_completions(shell),
    }
}

/// Emit a *dynamic* completion wrapper. Every completion candidate is
/// resolved live by re-invoking this binary with `COMPLETE=<shell>`
/// set — the shell snippet is one line that hands control back to
/// `CompleteEnv` in `main()`. No frozen completion script ever lives
/// on disk; subcommands added later automatically show up next time
/// the user hits Tab.
///
/// With no `--shell`, detects from `$SHELL` and emits the snippet for
/// that shell.
fn cmd_completions(shell: Option<Shell>) {
    let argv0 = std::env::args_os().next()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "vectordata".to_string());

    let shell = match shell.or_else(detect_shell) {
        Some(s) => s,
        None => {
            eprintln!("Could not auto-detect your shell from $SHELL.");
            eprintln!("Pass --shell explicitly: bash | zsh | fish | elvish | powershell");
            std::process::exit(1);
        }
    };

    let name = shell_name(shell);
    println!("# vectordata tab-completion for {name} (dynamic — defers to the binary)");
    println!("# To activate now:  eval \"$(vectordata completions)\"");
    println!("# To persist:       add the activation line to your shell rc file");
    match shell {
        Shell::Fish       => println!("COMPLETE=fish \"{argv0}\" | source"),
        Shell::Elvish     => println!("eval (COMPLETE=elvish \"{argv0}\" | slurp)"),
        Shell::PowerShell => println!(
            r#"(& {{ $env:COMPLETE="powershell"; "{argv0}" }}) | Invoke-Expression"#),
        _ /* Bash / Zsh */ => println!("source <(COMPLETE={name} \"{argv0}\")"),
    }
}

/// Detect the user's shell from `$SHELL`. Returns `None` if the env
/// var is missing or the basename isn't recognised — the caller
/// surfaces a helpful error in that case.
fn detect_shell() -> Option<Shell> {
    let raw = std::env::var_os("SHELL")?;
    let path = std::path::PathBuf::from(raw);
    let name = path.file_name()?.to_str()?;
    match name {
        "bash"       => Some(Shell::Bash),
        "zsh"        => Some(Shell::Zsh),
        "fish"       => Some(Shell::Fish),
        "elvish"     => Some(Shell::Elvish),
        "pwsh" | "powershell" => Some(Shell::PowerShell),
        _ => None,
    }
}

fn shell_name(s: Shell) -> &'static str {
    match s {
        Shell::Bash       => "bash",
        Shell::Zsh        => "zsh",
        Shell::Fish       => "fish",
        Shell::Elvish     => "elvish",
        Shell::PowerShell => "powershell",
        _                 => "bash",
    }
}

/// Resolve the cache root: caller override > settings.yaml. Exits
/// with a printable error if neither is set so users see actionable
/// guidance (e.g. the `veks datasets config set-cache` hint).
fn resolve_cache_dir(override_: Option<PathBuf>) -> PathBuf {
    if let Some(p) = override_ { return p; }
    match vectordata::settings::cache_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

fn cmd_cache_list(cache_dir: Option<PathBuf>, verbose: bool) {
    let root = resolve_cache_dir(cache_dir);
    let listing = match list_entries(&root) {
        Ok(l) => l,
        Err(e) => { eprintln!("error scanning {}: {}", root.display(), e); std::process::exit(1); }
    };
    print_listing(&root, &listing, verbose);
}

fn cmd_cache_prune(
    cache_dir: Option<PathBuf>,
    dataset: Option<String>,
    profile: Option<String>,
    dry_run: bool,
) {
    let filter = PruneFilter { dataset, profile };
    if filter.is_empty() {
        eprintln!("Refusing to prune with no filter — pass at least one of \
            --dataset or --profile.");
        eprintln!("(To wipe the entire content-addressed cache, remove \
            `<cache_root>/{{blobs,http}}/` by hand.)");
        std::process::exit(2);
    }
    let root = resolve_cache_dir(cache_dir);
    let report = match prune_by_filter(&root, &filter, dry_run) {
        Ok(r) => r,
        Err(e) => { eprintln!("error pruning {}: {}", root.display(), e); std::process::exit(1); }
    };

    if report.matched.is_empty() {
        println!("No cache entries match the filter.");
        return;
    }

    println!("Cache directory: {}", root.display());
    let verb = if dry_run { "Would remove" } else { "Removed" };
    println!("{verb} {} entry/entries, freeing {}:",
        report.matched.len(),
        fmt_size(report.matched.iter().map(|e| e.size_bytes).sum::<u64>()));
    for entry in &report.matched {
        let rel = entry.path.strip_prefix(&root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| entry.path.display().to_string());
        let origin = entry.origin_url.as_deref().unwrap_or("<unknown origin>");
        println!("  {:>12}  {origin}", fmt_size(entry.size_bytes));
        println!("              path: {rel}");
    }
    if dry_run {
        println!("\nRun without --dry-run to delete.");
    }
}

fn cmd_cache_prune_legacy(cache_dir: Option<PathBuf>, dry_run: bool) {
    let root = resolve_cache_dir(cache_dir);
    if dry_run {
        let mut would_remove = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&root) {
            for e in entries.flatten() {
                if let Some(name) = e.file_name().to_str()
                    && is_legacy_layout_name(name)
                    && e.path().is_dir()
                {
                    would_remove.push(name.to_string());
                }
            }
        }
        if would_remove.is_empty() {
            println!("No legacy cache directories found in {}.", root.display());
        } else {
            println!("Would remove {} legacy dir(s) from {}:",
                would_remove.len(), root.display());
            for n in &would_remove { println!("  {}", n); }
            println!("\nRun without --dry-run to delete.");
        }
        return;
    }
    match prune_legacy_layout(&root) {
        Ok(removed) if removed.is_empty() => {
            println!("No legacy cache directories found in {}.", root.display());
        }
        Ok(removed) => {
            println!("Removed {} legacy cache dir(s) from {}:",
                removed.len(), root.display());
            for n in &removed { println!("  {}", n); }
        }
        Err(e) => { eprintln!("error pruning {}: {}", root.display(), e); std::process::exit(1); }
    }
}

// ─── Formatting ──────────────────────────────────────────────────────

fn print_listing(root: &std::path::Path, l: &CacheListing, verbose: bool) {
    println!("Cache directory: {}", root.display());
    println!("Total on disk:   {}", fmt_size(l.total_bytes()));
    println!();

    if !l.catalog_datasets.is_empty() {
        println!("Catalog datasets (top-level with dataset.yaml):");
        print_simple_rows(&l.catalog_datasets, root);
        println!();
    }

    if !l.blobs.is_empty() {
        println!("Content-addressed blobs (blobs/<hex>/...):");
        if verbose {
            print_blob_rows_verbose(&l.blobs, root);
        } else {
            print_blob_rows_by_host(&l.blobs);
        }
        println!();
    }

    if !l.http.is_empty() {
        println!("Direct-HTTP cache (http/<hex>/...) — no merkle verification:");
        if verbose {
            print_blob_rows_verbose(&l.http, root);
        } else {
            print_blob_rows_by_host(&l.http);
        }
        println!();
    }

    if !l.legacy.is_empty() {
        println!("Legacy <host>[:<port>] dirs (pre-1.1 layout):");
        print_simple_rows(&l.legacy, root);
        println!("  → remove with: vectordata cache prune-legacy");
        println!();
    }

    if !l.other.is_empty() {
        println!("Other top-level entries (unrecognised):");
        print_simple_rows(&l.other, root);
        println!();
    }

    if l.catalog_datasets.is_empty() && l.blobs.is_empty()
        && l.http.is_empty() && l.legacy.is_empty() && l.other.is_empty()
    {
        println!("(empty)");
    }
}

/// Two-column path/size rows for the catalog/legacy/other categories.
fn print_simple_rows(rows: &[CacheEntry], root: &std::path::Path) {
    let name_w = rows.iter()
        .map(|e| rel_name(&e.path, root).len())
        .max().unwrap_or(20).max(20);
    for e in rows {
        let name = rel_name(&e.path, root);
        println!("  {:<name_w$}  {:>12}  ({} file{})",
            name, fmt_size(e.size_bytes),
            e.file_count, if e.file_count == 1 { "" } else { "s" });
    }
}

/// Grouped-by-host summary for content-addressed blobs (the default
/// listing mode — keeps `cache list` readable when there are
/// thousands of blobs).
fn print_blob_rows_by_host(rows: &[CacheEntry]) {
    use std::collections::BTreeMap;
    let mut by_host: BTreeMap<String, (u64, usize)> = BTreeMap::new();
    for e in rows {
        let host = e.origin_host.clone()
            .unwrap_or_else(|| "<unknown origin>".to_string());
        let slot = by_host.entry(host).or_insert((0u64, 0usize));
        slot.0 += e.size_bytes;
        slot.1 += 1;
    }
    let mut pairs: Vec<_> = by_host.into_iter().collect();
    pairs.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    let host_w = pairs.iter().map(|(h, _)| h.len()).max().unwrap_or(20).max(20);
    for (host, (size, count)) in pairs {
        let note = if host.starts_with("127.0.0.1") || host == "localhost" {
            "  ← looks like test debris"
        } else { "" };
        println!("  {:<host_w$}  {:>12}  ({} blob{}){}",
            host, fmt_size(size), count,
            if count == 1 { "" } else { "s" }, note);
    }
}

/// Verbose mode: one row per blob, full origin URL.
fn print_blob_rows_verbose(rows: &[CacheEntry], root: &std::path::Path) {
    for e in rows {
        let rel = rel_name(&e.path, root);
        let origin = e.origin_url.as_deref().unwrap_or("<unknown origin>");
        println!("  {:>12}  {}", fmt_size(e.size_bytes), origin);
        println!("              path: {}", rel);
    }
}

fn rel_name(path: &std::path::Path, root: &std::path::Path) -> String {
    path.strip_prefix(root)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.display().to_string())
}

fn fmt_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    const TIB: u64 = 1024 * GIB;
    if bytes >= TIB { format!("{:.1} TiB", bytes as f64 / TIB as f64) }
    else if bytes >= GIB { format!("{:.1} GiB", bytes as f64 / GIB as f64) }
    else if bytes >= MIB { format!("{:.1} MiB", bytes as f64 / MIB as f64) }
    else if bytes >= KIB { format!("{:.1} KiB", bytes as f64 / KIB as f64) }
    else { format!("{} B", bytes) }
}
