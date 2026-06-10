// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata` — minimal admin CLI for the vectordata cache.
//!
//! This binary is intentionally small. It exists so that downstream
//! consumers of the `vectordata` library can inspect and curate their
//! local cache (`vectordata cache list`, `vectordata cache prune-legacy`,
//! `vectordata config get`) without building or installing the
//! larger `veks` toolkit. The `veks` CLI delegates the same
//! operations into the same library entry points so there is exactly
//! one implementation of each command.
//!
//! Built by default. Library-only consumers who want to avoid pulling
//! `clap` into their build can disable the `cli` feature explicitly
//! (`vectordata = { ..., default-features = false }`).

use std::path::PathBuf;

use veks_completion::cli as vcli;
use veks_completion::VeksCli;

use vectordata::cache_admin::{
    CacheEntry, CacheListing, PruneFilter, is_legacy_layout_dir,
    list_entries, prune_by_filter, prune_legacy_layout,
};

/// Compose a richer --version string from the build.rs-emitted
/// environment variables: package version + git describe + build
/// profile + build date. Surfaces enough triage info that a user-
/// reported bug includes "which exact binary" without further
/// archaeology.
const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("VECTORDATA_GIT_DESCRIBE"),
    ", ",
    env!("VECTORDATA_BUILD_PROFILE"),
    " build, ",
    env!("VECTORDATA_BUILD_DATE"),
    ")",
);

#[derive(veks_completion_derive::VeksCli)]
#[command(
    name = "vectordata",
    version = LONG_VERSION,
    about = "Inspect catalog-published vector datasets, manage the local cache, and \
             launch the interactive explorer.",
    long_about = "vectordata — the user-facing entry point for working with \
                  published vector-search benchmark datasets.\n\n\
                  Common starting points:\n  \
                  • `vectordata datasets`          — TUI browser of every reachable dataset\n  \
                  • `vectordata datasets list`     — text catalog listing\n  \
                  • `vectordata explore`           — interactive value/distance explorer\n  \
                  • `vectordata config get`        — review the active configuration\n  \
                  • `vectordata cache list`        — see what's on disk\n\n\
                  First-time users typically start by configuring a catalog source \
                  (`vectordata config catalog add <url-or-path>`) and a cache directory \
                  (`vectordata config set cache <dir>`)."
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(veks_completion_derive::VeksCli)]
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
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir(), global = true)]
        configdir: String,
        /// (TUI-mode only) Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long, global = true)]
        catalog: Vec<String>,
        /// (TUI-mode only) Catalog URLs or paths to use *instead* of configured catalogs.
        #[arg(long = "at", global = true)]
        at: Vec<String>,
    },
    /// Unified vector space explorer — norms, distances, eigenvalues,
    /// and PCA projections rendered in a single ratatui TUI. Run without
    /// flags to pop the catalog picker; pass `--source` or `--dataset`
    /// to launch directly against a known view.
    #[cfg(feature = "explore")]
    Explore(vectordata::explore::ExploreArgs),
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
        #[arg(long, value_parser = ["bash", "zsh", "fish", "elvish", "powershell"])]
        shell: Option<String>,
    },

    /// Establish a stored credential for a `vecd` endpoint.
    Login {
        /// Endpoint URL (e.g. https://vecd-host/). Omit with --list.
        url: Option<String>,
        /// Username for a password grant.
        #[arg(long)]
        user: Option<String>,
        /// Store this pre-issued token directly (no password exchange).
        #[arg(long)]
        token: Option<String>,
        /// Password (else $VECTORDATA_PASSWORD, else an interactive prompt).
        #[arg(long)]
        password: Option<String>,
        /// Requested token lifetime (e.g. 90d), subject to the server max.
        #[arg(long)]
        expires: Option<String>,
        /// List endpoints with stored credentials and exit.
        #[arg(long)]
        list: bool,
    },
    /// Forget the stored credential for an endpoint.
    Logout {
        /// Endpoint URL (defaults to the one you're logged in to).
        url: Option<String>,
    },
    /// Show your stored identity + access at an endpoint.
    Whoami {
        /// Endpoint URL (defaults to the one you're logged in to).
        url: Option<String>,
    },
    /// Probe what your access lets you see and do at a datasource URL.
    Ping {
        /// Endpoint URL (defaults to the one you're logged in to).
        url: Option<String>,
    },
    /// Mint or revoke delegated API tokens at an endpoint.
    Token {
        #[command(subcommand)]
        command: TokenCmd,
    },
    /// Mirror a readable `vecd` store off-system (resumable, content-addressed).
    Backup {
        url: String,
        #[arg(long)]
        to: String,
        /// Skip versions already completely mirrored.
        #[arg(long)]
        incremental: bool,
    },
    /// Push a mirror's latest state back into a `vecd` endpoint.
    Restore {
        src: String,
        #[arg(long)]
        to: String,
    },
}

#[derive(veks_completion_derive::VeksCli)]
enum TokenCmd {
    /// Mint a delegated key (≤ your access) to hand to someone.
    Issue {
        /// Endpoint URL (defaults to the one you're logged in to).
        url: Option<String>,
        #[arg(long)]
        description: String,
        /// A (class, scope) subset, e.g. "read datasets/glove, publish datasets/scratch".
        #[arg(long)]
        profile: Option<String>,
        #[arg(long)]
        expires: Option<String>,
    },
    /// Revoke a key you issued.
    Revoke {
        /// ID of the token to revoke.
        id: i64,
        /// Endpoint URL (defaults to the one you're logged in to).
        url: Option<String>,
    },
}

#[derive(veks_completion_derive::VeksCli)]
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
    /// Remove pre-cutover legacy directories: `blobs/`, `http/`, and
    /// the older `<host>[:<port>]/` shape. Natural-layout dataset
    /// directories are not touched — those have a dataset name and
    /// are removed via `cache prune --dataset` or the picker's
    /// per-dataset Purge action.
    PruneLegacy {
        /// Print what would be removed without deleting.
        #[arg(long)]
        dry_run: bool,
        /// Override the configured cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Remove natural-layout dataset directories by name.
    ///
    /// Walks every natural-layout dataset directory (one with
    /// `origin.json` at its root) and removes those whose directory
    /// name matches `--dataset`. Globs use `*` (any run) and `?`
    /// (one char). `--dataset` is required — an empty filter would
    /// wipe every cached dataset, which is what you would do by
    /// deleting `<cache_root>` by hand if that's actually what you
    /// want.
    Prune {
        /// Glob matched against each dataset directory's name.
        #[arg(long)]
        dataset: Option<String>,
        /// Print what would be removed without deleting.
        #[arg(long)]
        dry_run: bool,
        /// Override the configured cache directory.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

#[derive(veks_completion_derive::VeksCli)]
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
    #[command(alias = "probe")]
    Ping {
        #[command(flatten)]
        args: vectordata::datasets::ping::PingArgs,
        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir())]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs
        #[arg(long)]
        catalog: Vec<String>,
    },

    /// Print the full catalog descriptor for a dataset + profile.
    ///
    /// Same data the picker's `Describe` overlay shows — dataset
    /// path + type, catalog attributes, profile metadata
    /// (`maxk`, `base_count`, `partition`), every facet with its
    /// source / namespace / window — but as plain text suitable for
    /// `less`, `grep`, redirects, and diff-against-expected scripts.
    #[command(alias = "desc")]
    Describe {
        #[command(flatten)]
        args: vectordata::datasets::describe::DescribeArgs,
        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir())]
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
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir())]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long)]
        catalog: Vec<String>,
        /// Catalog URLs or paths to use *instead* of configured catalogs.
        #[arg(long = "at")]
        at: Vec<String>,
    },
    /// Push an already known-good dataset (or, with `--raw`, an ad-hoc
    /// directory) up to the remote its `.publish_url` names.
    ///
    /// Unlike `veks publish` (which builds + checks + syncs a whole
    /// publish root), `push` only moves bytes for one already-good
    /// source: it confirms the dataset, generates per-directory
    /// `SHA256SUMS`, refuses to clobber remote data without an audit
    /// message, and brackets the upload with `begin`/`complete` events
    /// in a single-provenance `pushlog.jsonl`. The transport is chosen
    /// from the URL scheme (`s3://`, `https://`, `file://`).
    Push(vectordata::push::PushArgs),
    /// Download and cache every facet of a dataset profile into the
    /// configured cache directory. Renders a live per-facet +
    /// aggregate progress meter on stderr.
    Precache {
        /// `name[:profile]`, a path to a `dataset.yaml` or its
        /// containing directory, or an `http(s)://…` URL.
        spec: String,
        /// Configuration directory containing `catalogs.yaml`.
        #[arg(long, default_value_t = vectordata::catalog::sources::config_dir())]
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

#[derive(veks_completion_derive::VeksCli)]
enum ConfigCmd {
    /// Read config — all settings, or one value.
    ///
    ///   vectordata config get          # settings.yaml path, cache_dir, etc.
    ///   vectordata config get cache    # the cache_dir on one line (scriptable)
    Get {
        /// A single setting key (currently `cache`); omit to show everything.
        key: Option<String>,
    },
    /// Write a config value, e.g. `vectordata config set cache <dir>`.
    ///
    /// For `cache`, the special value `auto` (alias `largest-writable-mount`)
    /// auto-resolves to the largest writable mount with `vectordata-cache` as a
    /// subdir, falling back to `$HOME/.cache/vectordata`.
    Set {
        /// Setting key (currently `cache`).
        key: String,
        /// New value (for `cache`: a path, or `auto`).
        value: String,
        /// Overwrite a protected settings.yaml.
        #[arg(long)]
        force: bool,
    },
    /// Manage catalog sources (the list in catalogs.yaml).
    Catalog {
        #[command(subcommand)]
        command: CatalogCmd,
    },
    /// List writable mount points with available + total disk space, to help
    /// pick a cache dir for `config set cache`.
    Mounts {
        /// Include mounts with less than 100 MiB free.
        #[arg(long)]
        all: bool,
    },
}

#[derive(veks_completion_derive::VeksCli)]
enum CatalogCmd {
    /// Add a catalog source (URL or path).
    Add {
        /// Catalog URL or path (e.g. https://host/path/ or /local/path).
        source: String,
    },
    /// Remove a catalog source (by URL/path or `--at <index>`).
    #[command(alias = "rm")]
    Remove {
        /// Catalog URL or path to remove.
        source: Option<String>,
        /// Remove by 1-based index (see `config catalog list`).
        #[arg(long)]
        at: Option<usize>,
    },
    /// List configured catalog sources.
    List,
}

/// Tab-completion for `--to` (the push destination): the namespace names on the
/// endpoint you're logged in to, plus `root` (the `/` namespace). A bare name is
/// expanded to a full URL at push time. Best-effort and short-timeout: no
/// suggestions when you're not logged in or the server is unreachable.
fn complete_push_to(partial: &str, _ctx: &[&str]) -> Vec<String> {
    let Ok(endpoint) = vectordata::credentials::resolve_endpoint(None) else {
        return Vec::new();
    };
    let base = endpoint.trim_end_matches('/').to_string();
    let token = vectordata::credentials::stored_token(&base);
    let mut out = vectordata::endpoint::candidate_namespaces(&base, token.as_deref());
    out.push("root".to_string());
    out.into_iter()
        .filter(|s| partial.is_empty() || s.starts_with(partial))
        .collect()
}

fn main() {
    let spec = Cli::veks_command_spec("vectordata");

    // Dynamic-completion entry: when invoked with `COMPLETE=<shell>` (or
    // `_VECTORDATA_COMPLETE=…`), emit candidates and exit. The snippet from
    // `vectordata completions` re-invokes the binary with that env var set, so
    // completion logic lives in the spec rather than a frozen script.
    let mut resolvers: std::collections::BTreeMap<String, veks_completion::ValueProvider> =
        std::collections::BTreeMap::new();
    // `--to` (push destination): suggest namespace URLs on the logged-in endpoint.
    resolvers.insert("--to".to_string(), veks_completion::fn_provider(complete_push_to));
    let tree = vcli::build_completion_tree(&spec, &resolvers);
    if veks_completion::handle_complete_env("vectordata", &tree) {
        return;
    }
    if veks_completion::handle_diagnostic_args("vectordata", &tree) {
        return;
    }
    veks_completion::hint_completions_unregistered("vectordata");

    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--version" || a == "-V") {
        println!("vectordata {LONG_VERSION}");
        return;
    }
    if argv.first().is_none() || argv.iter().any(|a| a == "--help" || a == "-h") {
        // Help for the deepest subcommand named on the line (group → leaf).
        print!("{}", vcli::render_help_for(&spec, &argv));
        return;
    }

    let parsed = vcli::parse(&spec, &argv).unwrap_or_else(|e| {
        eprintln!("vectordata: {e}");
        std::process::exit(2);
    });
    let cli = <Cli as VeksCli>::veks_from_parsed(&parsed).unwrap_or_else(|e| {
        eprintln!("vectordata: {e}");
        std::process::exit(2);
    });
    match cli.command {
        Cmd::Cache { command } => match command {
            CacheCmd::List { cache_dir, verbose } => cmd_cache_list(cache_dir, verbose),
            CacheCmd::PruneLegacy { dry_run, cache_dir } => {
                cmd_cache_prune_legacy(cache_dir, dry_run)
            }
            CacheCmd::Prune { dataset, dry_run, cache_dir } => {
                cmd_cache_prune(cache_dir, dataset, dry_run)
            }
        },
        Cmd::Datasets { command, configdir, catalog, at } => {
            // No subcommand → TUI browser. The `Option<DatasetsCmd>`
            // shape makes `vectordata datasets` valid on its own.
            let Some(command) = command else {
                std::process::exit(vectordata::datasets::browser::run(&configdir, &catalog, &at));
            };
            let code = match command {
                DatasetsCmd::List(args) =>
                    vectordata::datasets::list::run_args(args),
                DatasetsCmd::Ping { args, configdir, catalog } =>
                    vectordata::datasets::ping::run_args(args, &configdir, &catalog, &[]),
                DatasetsCmd::Describe { args, configdir, catalog } =>
                    vectordata::datasets::describe::run_args(args, &configdir, &catalog, &[]),
                DatasetsCmd::Curlify(args) =>
                    vectordata::datasets::curlify::run_args(args),
                DatasetsCmd::Push(args) =>
                    vectordata::push::run(args),
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
                ConfigCmd::Get { key } => match key.as_deref() {
                    None => vectordata::config::show(),
                    Some("cache") => vectordata::config::get_cache(),
                    Some(other) => {
                        eprintln!("unknown config key '{other}' (try `config get cache`, or `config get` for all)");
                        2
                    }
                },
                ConfigCmd::Set { key, value, force } => match key.as_str() {
                    "cache" => vectordata::config::set_cache(std::path::Path::new(&value), force),
                    other => {
                        eprintln!("unknown config key '{other}' (settable keys: cache)");
                        2
                    }
                },
                ConfigCmd::Mounts { all } => vectordata::config::list_mounts(all),
                ConfigCmd::Catalog { command } => match command {
                    CatalogCmd::Add { source } => vectordata::config::add_catalog(&source),
                    CatalogCmd::Remove { source, at } => match (source, at) {
                        (Some(s), _) => vectordata::config::remove_catalog(
                            vectordata::config::RemoveCatalogSpec::Source(&s)),
                        (None, Some(n)) => vectordata::config::remove_catalog(
                            vectordata::config::RemoveCatalogSpec::Index(n)),
                        (None, None) => {
                            eprintln!("Specify a catalog URL/path or --at <index>.");
                            eprintln!("Use `vectordata config catalog list` to see indices.");
                            1
                        }
                    },
                    CatalogCmd::List => vectordata::config::list_catalogs(),
                },
            };
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Completions { shell } => cmd_completions(shell),
        Cmd::Login { url, user, token, password, expires, list } => {
            let code = if list {
                vectordata::client_cli::list_logins()
            } else if let Some(url) = url {
                let code = vectordata::client_cli::login(
                    &url, user.as_deref(), token.as_deref(), password.as_deref(), expires.as_deref(),
                );
                // After a successful login, offer (interactively) to register the
                // endpoint's namespaces as catalogs so `datasets list` reflects them.
                // This is the interactive entry point — kept out of the pure handler.
                if code == 0 {
                    vectordata::client_cli::offer_endpoint_catalogs(&url);
                }
                code
            } else {
                eprintln!("login needs a <url> (or --list)");
                2
            };
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Logout { url } => {
            let code = vectordata::client_cli::logout(&endpoint_or_exit(url));
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Whoami { url } => {
            let code = vectordata::client_cli::ping(&endpoint_or_exit(url), false);
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Ping { url } => {
            let code = vectordata::client_cli::ping(&endpoint_or_exit(url), true);
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Token { command } => {
            let code = match command {
                TokenCmd::Issue { url, description, profile, expires } => {
                    vectordata::client_cli::token_issue(&endpoint_or_exit(url), &description, profile.as_deref(), expires.as_deref())
                }
                TokenCmd::Revoke { url, id } => vectordata::client_cli::token_revoke(&endpoint_or_exit(url), id),
            };
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Backup { url, to, incremental } => {
            let code = vectordata::client_cli::backup(&url, &to, incremental);
            if code != 0 { std::process::exit(code); }
        }
        Cmd::Restore { src, to } => {
            let code = vectordata::client_cli::restore(&src, &to);
            if code != 0 { std::process::exit(code); }
        }
        #[cfg(feature = "explore")]
        Cmd::Explore(args) => {
            let code = vectordata::explore::run(args);
            if code != 0 { std::process::exit(code); }
        }
    }
}

/// Emit a *dynamic* completion wrapper. Every completion candidate is
/// resolved live by re-invoking this binary with `COMPLETE=<shell>`
/// set — the shell snippet is one line that hands control back to
/// `CompleteEnv` in `main()`. No frozen completion script ever lives
/// on disk; subcommands added later automatically show up next time
/// the user hits Tab.
/// Resolve an endpoint-URL argument, defaulting to the logged-in endpoint when
/// omitted. Exits 2 with guidance when not logged in or several are stored.
fn endpoint_or_exit(url: Option<String>) -> String {
    match vectordata::credentials::resolve_endpoint(url.as_deref()) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}

///
/// With no `--shell`, auto-detects from `$SHELL` and emits the indirect
/// `source <(…)` wrapper.
fn cmd_completions(shell: Option<String>) {
    // Delegate to the completion framework, which owns the registration
    // protocol (the emitted bash shim calls back via `_VECTORDATA_COMPLETE`,
    // never a bare `COMPLETE=bash vectordata`).
    match shell {
        Some(s) => match veks_completion::Shell::from_name(&s) {
            Some(sh) => veks_completion::print_completions("vectordata", sh),
            None => {
                eprintln!("unknown shell '{s}' (expected bash | zsh | fish | elvish | powershell)");
                std::process::exit(1);
            }
        },
        None => veks_completion::print_indirect_wrapper("vectordata"),
    }
}

/// Resolve the cache root: caller override > settings.yaml. Exits
/// with a printable error if neither is set so users see actionable
/// guidance (e.g. the `veks datasets config set cache` hint).
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
    dry_run: bool,
) {
    let filter = PruneFilter { dataset };
    if filter.is_empty() {
        eprintln!("error: refusing to prune with no filter — pass --dataset \
            with a name or glob pattern.");
        eprintln!("(To wipe every cached dataset, remove `<cache_root>` by \
            hand. To clean up pre-cutover detritus, run \
            `vectordata cache prune-legacy`.)");
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
                    && is_legacy_layout_dir(name)
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

    if !l.datasets.is_empty() {
        println!("Datasets:");
        if verbose {
            print_dataset_rows_verbose(&l.datasets, root);
        } else {
            print_dataset_rows(&l.datasets, root);
        }
        println!();
    }

    if !l.legacy.is_empty() {
        println!("Legacy detritus (pre-cutover layout):");
        print_simple_rows(&l.legacy, root);
        println!("  → remove with: vectordata cache prune-legacy");
        println!();
    }

    if !l.other.is_empty() {
        println!("Other top-level entries (unrecognised):");
        print_simple_rows(&l.other, root);
        println!();
    }

    if l.datasets.is_empty() && l.legacy.is_empty() && l.other.is_empty() {
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

/// Default dataset listing: one row per dataset, name + size +
/// file count + origin host.
fn print_dataset_rows(rows: &[CacheEntry], root: &std::path::Path) {
    let name_w = rows.iter()
        .map(|e| rel_name(&e.path, root).len())
        .max().unwrap_or(20).max(20);
    for e in rows {
        let name = rel_name(&e.path, root);
        let host = e.origin_host.as_deref().unwrap_or("");
        let host_note = if host.is_empty() {
            String::new()
        } else {
            format!("  from {host}")
        };
        println!("  {:<name_w$}  {:>12}  ({} file{}){}",
            name, fmt_size(e.size_bytes),
            e.file_count, if e.file_count == 1 { "" } else { "s" },
            host_note);
    }
}

/// Verbose dataset listing: full origin URL per entry, on its own row.
fn print_dataset_rows_verbose(rows: &[CacheEntry], root: &std::path::Path) {
    for e in rows {
        let rel = rel_name(&e.path, root);
        let origin = e.origin_url.as_deref().unwrap_or("<unknown origin>");
        println!("  {:>12}  {} ({} file{})",
            fmt_size(e.size_bytes), rel,
            e.file_count, if e.file_count == 1 { "" } else { "s" });
        println!("              origin: {}", origin);
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
