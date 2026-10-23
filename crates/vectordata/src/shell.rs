// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI support. CLI shell for the `vectordata` command — cache admin, datasets
//! browsing, config, and the `explore` TUI.
//!
//! Lives in the library (gated on the `cli` feature) so both binaries
//! that answer to the name `vectordata` share one implementation:
//! the crate's own minimal bin (`src/bin/vectordata.rs`, a thin shim —
//! installable standalone without the larger veks toolkit) and the
//! workspace-root umbrella multiplexer, which embeds this shell as its
//! default personality so `vectordata explore` et al. behave
//! identically through either install.
//!
//! A program can run the same command line as its own:
//!
//! - [`bin_main_as`] and [`bin_main_with_catalogs`] are the binary's
//!   entry point under another name, with the program's catalogs; they
//!   end the process when the command fails.
//! - [`run`] and [`run_with_catalogs`] return the command's exit status
//!   instead (0, 1 for a failed command, 2 for a usage error), so a
//!   program wrapping the argv carries on after the command.
//! - A [`Host`] carries what the program adds: its catalogs, its own
//!   output after `datasets describe` ([`Host::after_describe`]) and
//!   `datasets list` ([`Host::after_list`]), its own precache specs
//!   ([`Host::before_precache`]), and, when its whole command line is
//!   this one, shell completion under its own name
//!   ([`Host::owns_completion`]).
//! - A host that embeds this CLI under a subcommand of its own completes
//!   it with [`completion_node`] or [`complete`].

use std::path::PathBuf;

use veks_completion::VeksCli;
use veks_completion::cli as vcli;

use crate::cache_admin::{
    CacheEntry, CacheListing, PruneFilter, is_legacy_layout_dir, list_entries, prune_by_filter,
    prune_legacy_layout,
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
             launch the interactive explorer."
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
        command: Option<CacheCmd>,
    },
    /// Inspect or modify vectordata's settings.yaml.
    Config {
        #[command(subcommand)]
        command: ConfigCmd,
    },
    /// Operations on catalog-published datasets, the interactive
    /// explorer among them. Run bare to see the available subcommands.
    Datasets {
        #[command(subcommand)]
        command: Option<DatasetsCmd>,
        // The three globals below are declared here so the flags
        // are accepted between `datasets` and its subcommand; the
        // values the dispatch arms read come from each subcommand's
        // own fields.
        #[allow(dead_code)]
        /// Configuration directory containing catalogs.yaml.
        #[arg(long, default_value_t = crate::catalog::sources::config_dir(), global = true)]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long, global = true)]
        #[allow(dead_code)]
        catalog: Vec<String>,
        /// Catalogs to use *instead* of the configured ones, in order: each a URL, a path, a configured catalog's number, or the name of a configured or program catalog (repeatable).
        #[arg(long = "at", global = true)]
        #[allow(dead_code)]
        at: Vec<String>,
    },
    /// Short for `datasets explore`: the interactive explorer.
    #[cfg(feature = "explore")]
    Explore(ExploreCommand),
    // Its `--help` is `completions_long_about`, which names the
    // program run.
    /// Print or activate tab-completion for the current shell.
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

/// `datasets explore` and its short form `explore`: the explorer's
/// options and the catalogs it reads.
#[cfg(feature = "explore")]
#[derive(veks_completion_derive::VeksCli)]
struct ExploreCommand {
    #[command(flatten)]
    args: crate::explore::ExploreArgs,
    /// Configuration directory containing `catalogs.yaml`.
    #[arg(long, default_value_t = crate::catalog::sources::config_dir())]
    configdir: String,
    /// Additional catalog directories, file paths, or HTTP URLs.
    #[arg(long)]
    catalog: Vec<String>,
    /// Catalogs to use *instead* of the configured ones, in order: each a URL, a path, a configured catalog's number, or the name of a configured or program catalog (repeatable).
    #[arg(long = "at")]
    at: Vec<String>,
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
    List(crate::datasets::list::ListArgs),

    /// Verify remote dataset access. Walks every facet declared by
    /// the dataset's profile, reads the first record via HTTP
    /// range, and reports success/failure per facet.
    #[command(alias = "probe")]
    Ping {
        #[command(flatten)]
        args: crate::datasets::ping::PingArgs,
        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value_t = crate::catalog::sources::config_dir())]
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
        args: crate::datasets::describe::DescribeArgs,
        /// Configuration directory containing catalogs.yaml
        #[arg(long, default_value_t = crate::catalog::sources::config_dir())]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs
        #[arg(long)]
        catalog: Vec<String>,
    },

    /// Write the dataset.yaml a directory of loose files implies.
    ///
    /// Reads the files the way an open of a directory with no
    /// descriptor does — canonical facet names, then one role token per
    /// name (`base`, `query`, `gt`, …), then shapes — verifies they
    /// belong together, and writes the result as `<dir>/dataset.yaml`,
    /// so later opens are declared rather than inferred. Prints what
    /// it decided and why.
    Init(crate::datasets::init::InitArgs),

    /// Generate a curl download script for a dataset's published files.
    Curlify(crate::datasets::curlify::CurlifyArgs),

    /// Unified vector space explorer — norms, distances, eigenvalues,
    /// and PCA projections rendered in a single ratatui TUI. Run without
    /// flags to pop the catalog picker; pass `--source` or `--dataset`
    /// to launch directly against a known view. `explore` at the top
    /// level is its short form.
    #[cfg(feature = "explore")]
    Explore(ExploreCommand),

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
        #[arg(long, short = 'd')]
        dataset: String,
        /// Profile selector; outranks a selector carried by `--dataset`
        /// (`ds:size=10m`). Must name exactly one profile; none means
        /// `default`.
        #[arg(long)]
        profile: Option<String>,
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
        /// Ordinals per shard: split each fixed-stride facet across
        /// files of this many records, the last shorter.
        ///
        /// Mirrors `shard_stride:` in `dataset.yaml`, which the derived
        /// dataset declares. Output that fits in one shard is written
        /// as a single file, so a run that happened to fit stays
        /// readable by anything that predates multi-file facets.
        /// Accepts SI suffixes: `--shard-stride 1M`.
        #[arg(long, value_name = "RECORDS")]
        shard_stride: Option<String>,
        /// Maximum size of one shard file; the stride is whatever
        /// number of records fits under it.
        ///
        /// The stride chosen is the largest power of ten that fits,
        /// so `shard(o) = o / stride` stays arithmetic a person can do
        /// in their head. The cap is a ceiling, not a target: a shard
        /// will usually be well under it.
        ///
        /// Accepts size suffixes: `--max-shard-bytes 1TB`. Mutually
        /// exclusive with `--shard-stride`.
        #[arg(long, value_name = "BYTES", conflicts_with = "shard_stride")]
        max_shard_bytes: Option<String>,
        /// Configuration directory containing `catalogs.yaml`
        /// (only used when `--dataset` is a catalog name).
        #[arg(long, default_value_t = crate::catalog::sources::config_dir())]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long)]
        catalog: Vec<String>,
        /// Catalogs to use *instead* of the configured ones, in order: each a URL, a path, a configured catalog's number, or the name of a configured or program catalog (repeatable).
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
    Push(crate::push::PushArgs),
    /// Download and cache every facet of the profiles one or more specs
    /// select into the configured cache directory, as one run. Renders a
    /// live per-facet + aggregate progress meter on stderr.
    Precache {
        /// One or more `<head>[:<selector>]` specs (PS-1): the head is a
        /// catalog name, a path to a `dataset.yaml` or its containing
        /// directory, or an `http(s)://…` URL; the selector names the
        /// profiles to fetch — a name, `profile=*` for every profile, or an
        /// expression such as `size=10m,predicates=uniform*`. A spec with
        /// no selector and no `--profile` is refused naming the spellings.
        /// Several specs are fetched as one plan, with one progress
        /// display and one report.
        specs: Vec<String>,
        /// Configuration directory containing `catalogs.yaml`.
        #[arg(long, default_value_t = crate::catalog::sources::config_dir())]
        configdir: String,
        /// Additional catalog directories, file paths, or HTTP URLs.
        #[arg(long)]
        catalog: Vec<String>,
        /// Catalogs to use *instead* of the configured ones, in order: each a URL, a path, a configured catalog's number, or the name of a configured or program catalog (repeatable).
        #[arg(long = "at")]
        at: Vec<String>,
        /// Fetch into this cache directory instead of the configured
        /// one; `settings.yaml` is left unchanged.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Fetch only these facets. Repeatable. Default: every facet
        /// the profile declares.
        #[arg(long)]
        facet: Vec<String>,
        /// Fetch only these records, e.g. `0..1M` or `[0..1K, 5K..6K]`.
        /// Default: the whole facet. Needs a single profile.
        #[arg(long)]
        window: Option<String>,
        /// Print what would be fetched and stop.
        #[arg(long)]
        plan: bool,
        /// Profile to use, when the spec does not name one. Required
        /// for `--facet`/`--window`/`--plan` on a multi-profile dataset.
        #[arg(long)]
        profile: Option<String>,
        /// Accept fetching a whole facet when `--window` cannot be
        /// resolved for its format (parquet; vvec with no index).
        #[arg(long)]
        allow_whole_facet: bool,
    },
}

#[derive(veks_completion_derive::VeksCli)]
enum ConfigCmd {
    /// Read config — all settings, or one value.
    ///
    ///   vectordata config get          # settings.yaml path, cache_dir, etc.
    ///   vectordata config get cache    # the cache_dir on one line (scriptable)
    Get {
        /// A single setting key (`cache`, `palette`, `curve`, `update_check`,
        /// `offline`); omit to show everything.
        key: Option<String>,
    },
    /// Write a config value, e.g. `vectordata config set cache <dir>`.
    ///
    /// For `cache`, the special value `auto` (alias `largest-writable-mount`)
    /// auto-resolves to the largest writable mount with `vectordata-cache` as a
    /// subdir, falling back to `$HOME/.cache/vectordata`. `offline on` stops
    /// every request to dataset servers: cached data, definitions and
    /// catalogs are used as kept (VECTORDATA_OFFLINE overrides it).
    Set {
        /// Setting key: `cache`, `palette`, `curve`, `update_check` or `offline`.
        key: String,
        /// New value (for `cache`: a path, or `auto`; for `update_check` and
        /// `offline`: `on` or `off`).
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
    /// State or correct a dataset's attributes on this host
    /// (overlays.yaml): what its catalog does not say or says wrong,
    /// read by every command and every program through vectordata.
    Overlay {
        #[command(subcommand)]
        command: OverlayCmd,
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
        /// Symbolic name for the catalog (map key in catalogs.yaml).
        /// Derived from the location when omitted.
        #[arg(long)]
        name: Option<String>,
        /// Register without the parse+ping verification gate — so you can
        /// name an endpoint before any dataset is published there (then
        /// address it by name via `login`, `push --to`, etc.).
        #[arg(long)]
        no_verify: bool,
        /// Accept this endpoint's self-signed TLS cert WITHOUT verification
        /// (records its origin under `trust_self_signed` in settings.yaml).
        /// Insecure — local dev only.
        #[arg(long)]
        trust_self_signed: bool,
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
    /// List configured catalog sources, numbered as `--at <n>` takes
    /// them; a disabled one is marked `(disabled)`.
    List,
    /// Stop assembling a catalog with the others (the `disabled_catalogs`
    /// setting): no command reads it unless named with `--at <name>` or
    /// `--catalog <name>`. Its number in `config catalog list` stays.
    Disable {
        /// The catalog's name (see `config catalog list`).
        name: String,
    },
    /// Assemble a disabled catalog with the others again.
    Enable {
        /// The catalog's name (see `config catalog list`).
        name: String,
    },
}

#[derive(veks_completion_derive::VeksCli)]
enum OverlayCmd {
    /// Set an attribute: `vectordata config overlay set ada002-100k distance_function DOT_PRODUCT`.
    ///
    /// The key is a dataset attribute (`distance_function`,
    /// `is_normalized`, `model`, `base_count`, `maxk`, …) or a
    /// consumer's own key with a `.` (`fvec.ncentroids`). The value is
    /// read as YAML: `true`, `300` and `COSINE` are a boolean, a number
    /// and a string.
    Set {
        /// The dataset, by name.
        dataset: String,
        /// The attribute.
        key: String,
        /// Its value.
        value: String,
        /// Only for the dataset of this name in this catalog.
        #[arg(long)]
        catalog: Option<String>,
    },
    /// Remove an attribute the overlay sets.
    #[command(alias = "rm")]
    Unset {
        /// The dataset, by name.
        dataset: String,
        /// The attribute.
        key: String,
        /// The catalog block it was set in.
        #[arg(long)]
        catalog: Option<String>,
    },
    /// Show what the overlay states: for one dataset, or for all.
    Show {
        /// The dataset, by name; all when omitted.
        dataset: Option<String>,
    },
}

/// Tab-completion for `--to` (the push destination): the namespace names on the
/// endpoint you're logged in to, plus `root` (the `/` namespace). A bare name is
/// expanded to a full URL at push time. Best-effort and short-timeout: no
/// suggestions when you're not logged in or the server is unreachable.
fn complete_push_to(partial: &str, _ctx: &[&str]) -> Vec<String> {
    let Ok(endpoint) = crate::credentials::resolve_endpoint(None) else {
        return Vec::new();
    };
    let base = endpoint.trim_end_matches('/').to_string();
    let token = crate::credentials::stored_token(&base);
    let mut out = crate::endpoint::candidate_namespaces(&base, token.as_deref());
    out.push("root".to_string());
    out.into_iter()
        .filter(|s| partial.is_empty() || s.starts_with(partial))
        .collect()
}

/// Tab-completion for `config set <key> <value>`: keys at the first
/// slot; key-specific values at the second. For `cache`, values are
/// live system facts — `auto`, the XDG default under `$HOME`, and a
/// `vectordata-cache` directory on each rw-mounted filesystem — so
/// the completion shows exactly the places `auto` could pick.
fn complete_config_set(partial: &str, ctx: &[&str]) -> Vec<String> {
    let positionals: Vec<&str> = ctx
        .iter()
        .copied()
        .filter(|w| !w.starts_with('-'))
        .collect();
    let mut out: Vec<String> = match positionals.first() {
        None => vec![
            "cache".to_string(),
            "palette".to_string(),
            "curve".to_string(),
            "update_check".to_string(),
            "offline".to_string(),
        ],
        Some(&"palette") => crate::config::ui_palette_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
        Some(&"curve") => crate::config::ui_curve_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
        Some(&"update_check" | &"offline") => vec!["on".to_string(), "off".to_string()],
        Some(&"cache") => {
            let mut vals = vec!["auto".to_string()];
            if let Some(home) = std::env::var_os("HOME") {
                vals.push(
                    std::path::PathBuf::from(home)
                        .join(".cache/vectordata")
                        .display()
                        .to_string(),
                );
            }
            for m in crate::mounts::enumerate() {
                if m.writable && m.path != "/" {
                    vals.push(format!("{}/vectordata-cache", m.path.trim_end_matches('/')));
                }
            }
            vals
        }
        Some(_) => Vec::new(),
    };
    out.retain(|s| partial.is_empty() || s.starts_with(partial));
    out.dedup();
    out
}

/// Tab-completion for `config get <key>` — the same key set as
/// `config set`.
fn complete_config_get(partial: &str, _ctx: &[&str]) -> Vec<String> {
    ["cache", "palette", "curve", "update_check", "offline"]
        .iter()
        .filter(|k| partial.is_empty() || k.starts_with(partial))
        .map(|k| k.to_string())
        .collect()
}

/// Tab-completion for `config catalog disable <name>`: the configured
/// catalogs not disabled.
fn complete_catalog_disable(partial: &str, _ctx: &[&str]) -> Vec<String> {
    catalog_toggle_candidates(&configured_catalog_names(), &crate::settings::disabled_catalogs(), true, partial)
}

/// Tab-completion for `config catalog enable <name>`: the catalogs
/// disabled.
fn complete_catalog_enable(partial: &str, _ctx: &[&str]) -> Vec<String> {
    catalog_toggle_candidates(&configured_catalog_names(), &crate::settings::disabled_catalogs(), false, partial)
}

/// The names `catalogs.yaml` gives, in order.
fn configured_catalog_names() -> Vec<String> {
    crate::catalog::sources::named_catalog_entries(&crate::catalog::sources::config_dir())
        .into_iter()
        .map(|e| e.name)
        .collect()
}

/// Pure core of the `disable`/`enable` completions: for `disable`, the
/// `configured` names `disabled` does not list; for `enable`, those it
/// lists. Each starting with `partial`.
fn catalog_toggle_candidates(configured: &[String], disabled: &[String], disable: bool, partial: &str) -> Vec<String> {
    let pool: Vec<String> = if disable {
        configured.iter().filter(|n| !disabled.contains(n)).cloned().collect()
    } else {
        disabled.to_vec()
    };
    pool.into_iter().filter(|n| n.starts_with(partial)).collect()
}

/// Every dynamic value resolver this binary registers with the
/// completion tree: the shared dataset-domain map (`--dataset`,
/// `--profile`, `--at`, and the `datasets ping`/`describe`/`precache`
/// positionals — see [`crate::datasets::dyncomp`]) plus the
/// vectordata-specific entries.
fn completion_resolvers() -> std::collections::BTreeMap<String, veks_completion::ValueProvider> {
    let mut resolvers = crate::datasets::dyncomp::datasets_resolvers();
    // `--to` (push destination): suggest namespace URLs on the logged-in endpoint.
    resolvers.insert(
        "--to".to_string(),
        veks_completion::fn_provider(complete_push_to),
    );
    // Positional completion, keyed by subcommand path.
    resolvers.insert(
        "config set".to_string(),
        veks_completion::fn_provider(complete_config_set),
    );
    resolvers.insert(
        "config get".to_string(),
        veks_completion::fn_provider(complete_config_get),
    );
    resolvers.insert(
        "config catalog disable".to_string(),
        veks_completion::fn_provider(complete_catalog_disable),
    );
    resolvers.insert(
        "config catalog enable".to_string(),
        veks_completion::fn_provider(complete_catalog_enable),
    );
    resolvers
}

/// Binary entry point over explicit args (everything after the program
/// name), for the `vectordata` binary itself. Wrapper binaries that
/// embed this CLI under another name call [`bin_main_as`]; a program
/// that must keep running after the command uses [`run`].
///
/// Exits the process with the command's status when it is not 0, and
/// returns when it is.
pub fn bin_main(argv: Vec<String>) {
    bin_main_as("vectordata", argv)
}

/// [`bin_main`] for a binary that embeds this CLI: `invoked` is what the
/// user typed to reach it — `"vectordata"`, or e.g. `"fvec bench
/// vectordata"` — and is what help, usage lines, `--version` and parse
/// errors name, so a user is told about the command they ran rather than
/// a binary they may not have installed.
///
/// Shell completion stays bound to the real `vectordata` executable
/// (`COMPLETE=<shell>` / `_VECTORDATA_COMPLETE`, and the scripts
/// `vectordata completions` prints): completing an embedded CLI means
/// answering to the embedding executable's own variable, which is that
/// executable's to set up. A host whose whole command line is this one
/// says so with [`Host::owns_completion`] and calls [`run`].
pub fn bin_main_as(invoked: &str, argv: Vec<String>) {
    bin_main_with_catalogs(invoked, argv, crate::catalog::CatalogSources::new())
}

/// [`bin_main_as`] for a program with catalogs of its own: every
/// command that reads catalogs (`datasets list`, `describe`, `ping`,
/// `precache`, `derive`, and `explore`) sees `program`'s catalogs after
/// the configured ones and before any `--catalog` location, so a
/// dataset the program registered (with
/// [`CatalogBuilder`](crate::catalog::CatalogBuilder), say) is listed
/// and opens by name. `--at` replaces all of them for the run, as it
/// replaces the configured ones; numbered shortcuts (`--at 2`) still
/// count the configured catalogs. The assembly is
/// [`datasets::build_sources`](crate::datasets::build_sources).
///
/// `program` is the program's own catalogs only. The configured ones
/// are added for every command; a set that already holds them (one
/// built with `configure_default()`) lists each twice, and every name
/// in one becomes ambiguous.
///
/// Exits the process with the command's status when it is not 0, and
/// returns when it is; [`run_with_catalogs`] returns the status always.
///
/// ```no_run
/// use vectordata::catalog::{CatalogBuilder, CatalogSources};
/// # let entry: vectordata::dataset::CatalogEntry = unimplemented!();
/// let mine = CatalogSources::new().with(CatalogBuilder::named("bench").entry(entry).build());
/// let argv: Vec<String> = std::env::args().skip(2).collect();
/// vectordata::shell::bin_main_with_catalogs("fvec bench vectordata", argv, mine);
/// ```
pub fn bin_main_with_catalogs(invoked: &str, argv: Vec<String>, program: crate::catalog::CatalogSources) {
    let code = run_with_catalogs(invoked, argv, program);
    if code != 0 {
        std::process::exit(code);
    }
}

/// [`bin_main_with_catalogs`], returning the command's exit status
/// instead of ending the process: 0 for success, 1 when the command
/// fails, 2 for a usage error (a malformed command line, a refused
/// request). Standard output and standard error are flushed before it
/// returns. It is [`run`] with a [`Host`] holding only `program`.
///
/// For a program that wraps the command line and carries on after it —
/// to print its own output, or to run the next command:
///
/// ```
/// use vectordata::catalog::CatalogSources;
/// let home = tempfile::tempdir().unwrap();
/// let configdir = home.path().to_string_lossy().into_owned();
/// let words = |w: &[&str]| w.iter().map(|s| s.to_string()).collect::<Vec<_>>();
///
/// let ok = vectordata::shell::run_with_catalogs("bench", words(&["--version"]), CatalogSources::new());
/// assert_eq!(ok, 0);
/// let usage = vectordata::shell::run_with_catalogs("bench", words(&["datasets", "no-such-command"]), CatalogSources::new());
/// assert_eq!(usage, 2);
/// let failed = vectordata::shell::run_with_catalogs(
///     "bench",
///     words(&["datasets", "describe", "nothing-here", "--configdir", &configdir]),
///     CatalogSources::new(),
/// );
/// assert_eq!(failed, 1);
/// ```
pub fn run_with_catalogs(invoked: &str, argv: Vec<String>, program: crate::catalog::CatalogSources) -> i32 {
    run(invoked, argv, &Host::new().catalogs(program))
}

/// What [`Host::after_describe`] is called with.
type DescribeHook = dyn Fn(&crate::catalog::Catalog, &crate::dataset::CatalogEntry, &str);
/// What [`Host::after_list`] is called with.
type ListHook = dyn Fn(&crate::catalog::Catalog, &[&crate::dataset::CatalogEntry]);
/// What [`Host::before_precache`] is called with.
type PrecacheHook = dyn Fn(&str) -> Result<Vec<String>, String>;

/// A program that runs this command line as its own: its catalogs, the
/// output it adds after some commands, and whether shell completion is
/// its. Built fluently and passed to [`run`]; [`Host::new`] adds
/// nothing, so `run` with it is the `vectordata` binary.
///
/// Each hook is called for its own command only, on the thread [`run`]
/// is called on, so a hook may hold `Rc`/`RefCell` state; none is
/// `Send` or `Sync`.
///
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use vectordata::shell::{Host, run};
///
/// let listed = Rc::new(RefCell::new(Vec::new()));
/// let seen = listed.clone();
/// let host = Host::new()
///     .after_list(move |_catalog, entries| {
///         seen.borrow_mut().extend(entries.iter().map(|e| e.name().to_string()));
///     })
///     .before_precache(|spec| match spec.strip_prefix("group:") {
///         Some("pair") => Ok(vec!["a:default".into(), "b:default".into()]),
///         Some(other) => Err(format!("no group '{other}'")),
///         None => Ok(vec![spec.to_string()]),
///     });
///
/// // A group the host does not know is the command's error.
/// let argv = ["datasets", "precache", "group:none"].map(String::from).to_vec();
/// assert_eq!(run("bench", argv, &host), 2);
/// assert!(listed.borrow().is_empty(), "after_list is for `datasets list` alone");
/// ```
pub struct Host {
    catalogs: crate::catalog::CatalogSources,
    after_describe: Option<Box<DescribeHook>>,
    after_list: Option<Box<ListHook>>,
    before_precache: Option<Box<PrecacheHook>>,
    owns_completion: bool,
}

impl Default for Host {
    fn default() -> Self {
        Host::new()
    }
}

impl std::fmt::Debug for Host {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Host")
            .field("catalogs", &self.catalogs)
            .field("after_describe", &self.after_describe.is_some())
            .field("after_list", &self.after_list.is_some())
            .field("before_precache", &self.before_precache.is_some())
            .field("owns_completion", &self.owns_completion)
            .finish()
    }
}

impl Host {
    /// A host that adds nothing: no catalogs of its own, no hooks, and
    /// completion left to the `vectordata` executable.
    pub fn new() -> Self {
        Host {
            catalogs: crate::catalog::CatalogSources::new(),
            after_describe: None,
            after_list: None,
            before_precache: None,
            owns_completion: false,
        }
    }

    /// The program's own catalogs, read by every command that reads
    /// catalogs after the configured ones and before `--catalog`'s, as
    /// [`bin_main_with_catalogs`] describes. Replaces any set before.
    pub fn catalogs(mut self, program: crate::catalog::CatalogSources) -> Self {
        self.catalogs = program;
        self
    }

    /// Called after `datasets describe` has printed its description,
    /// only when it succeeded, with `(catalog, entry, profile)`:
    ///
    /// - `catalog`: the catalog the dataset was looked up in; empty when
    ///   the spec was a path or URL, which opens directly.
    /// - `entry`: the dataset described, as the catalog lists it or as
    ///   its `dataset.yaml` reads.
    /// - `profile`: the one profile the spec's selector resolved to
    ///   (`default` when it named none).
    ///
    /// Not called for a describe that fails, nor for any other command.
    /// Replaces any hook set before.
    pub fn after_describe(
        mut self,
        hook: impl Fn(&crate::catalog::Catalog, &crate::dataset::CatalogEntry, &str) + 'static,
    ) -> Self {
        self.after_describe = Some(Box::new(hook));
        self
    }

    /// Called after `datasets list` has printed its listing, only when
    /// it succeeded, with the catalog it read and exactly the entries it
    /// listed, in listing order: those the filters kept, less any none
    /// of whose profiles `--matching-profile` admits. An empty listing
    /// that succeeds (no datasets in the configured catalogs) is called
    /// with none.
    ///
    /// Not called for `--select`, whose output is one `dataset:profile`
    /// line for a script rather than a listing, nor for a list that
    /// fails, nor for any other command. Every `--output-format` calls
    /// it; a hook that prints after a `json` listing is the host's
    /// choice, and the host has the argv to tell. Replaces any hook set
    /// before.
    pub fn after_list(
        mut self,
        hook: impl Fn(&crate::catalog::Catalog, &[&crate::dataset::CatalogEntry]) + 'static,
    ) -> Self {
        self.after_list = Some(Box::new(hook));
        self
    }

    /// Expands each spec `datasets precache` is given, in order, before
    /// any is resolved: a host's own `group:<name>`, say, into the specs
    /// it stands for. Every spec is passed, the host's or not; return it
    /// unchanged (`Ok(vec![spec.to_string()])`) to leave it be. The
    /// expansions are concatenated, in order, and run as one precache
    /// ([`run_specs`](crate::datasets::precache::run_specs)): one plan,
    /// one progress display, one report.
    ///
    /// An `Err` is printed as the command's error (`error: <message>`)
    /// and the command exits 2 with nothing fetched, as does an
    /// expansion to no specs at all. Not called for any other command.
    /// Replaces any hook set before.
    pub fn before_precache(mut self, hook: impl Fn(&str) -> Result<Vec<String>, String> + 'static) -> Self {
        self.before_precache = Some(Box::new(hook));
        self
    }

    /// This host's command line is this one, under the host's own
    /// executable name: `invoked` (as passed to [`run`]) is what the
    /// shell completes.
    ///
    /// `<invoked> completions` then prints scripts bound to `invoked`,
    /// and the completion protocol the scripts call back with
    /// (`_<INVOKED>_COMPLETE`, the name upper-cased with `-` as `_`) is
    /// answered for it, from this CLI's own tree. The scripts call the
    /// host back as `<host> "$COMP_LINE" "$COMP_POINT"`; with the
    /// variable set, the host passes those two arguments to [`run`] as
    /// its `argv`, and `run` prints the candidates (no arguments prints
    /// the registration script). `invoked` must be the
    /// executable's name, one word (`fvec-bench`); a command path
    /// (`fvec bench vectordata`) is no name a shell completes, and
    /// `completions` under one fails saying so.
    ///
    /// Without it, `completions` under any name but `vectordata` says the
    /// host must graft [`completion_node`] (or answer with [`complete`])
    /// itself, and the protocol stays the `vectordata` executable's.
    pub fn owns_completion(mut self) -> Self {
        self.owns_completion = true;
        self
    }

    /// The name shell completion is bound to for a run invoked as
    /// `invoked`: the host's own when it owns completion, else the
    /// `vectordata` executable's. `None` for an owning host invoked by
    /// something no shell completes (a command path).
    fn completion_name<'a>(&self, invoked: &'a str) -> Option<&'a str> {
        if !self.owns_completion {
            return Some("vectordata");
        }
        (!invoked.is_empty() && !invoked.contains(char::is_whitespace)).then_some(invoked)
    }
}

/// The executable a reminder to enable tab-completion names, for a run
/// invoked as `invoked`: its first word, the program the user typed —
/// `vectordata`, or the host that embeds this CLI (`fvec` for `fvec
/// bench vectordata`), whose shell completion it is. `None` when
/// `invoked` is empty.
/// The command line as `invoked` reaches it, with the long help of the
/// root and of `completions` naming `invoked`, so a host embedding it
/// shows its own commands.
fn cli_spec(invoked: &str) -> veks_completion::CommandSpec {
    let mut spec = Cli::veks_command_spec(invoked).long_about(top_long_about(invoked));
    if let Some(c) = spec.subcommands.iter_mut().find(|c| c.name == "completions") {
        c.long_about = Some(completions_long_about(invoked));
    }
    spec
}

/// What `<invoked> --help` opens with: the starting points, as typed.
fn top_long_about(invoked: &str) -> String {
    let starts = [
        ("datasets explore", "interactive dataset picker + visualizer (`explore` for short)"),
        ("datasets list", "text catalog listing"),
        ("config get", "review the active configuration"),
        ("cache list", "see what's on disk"),
    ];
    let width = starts.iter().map(|(c, _)| invoked.len() + 1 + c.len()).max().unwrap_or(0);
    let mut s = String::from(
        "The user-facing entry point for working with published vector-search benchmark datasets.\n\n\
         Common starting points:\n",
    );
    for (cmd, what) in starts {
        let typed = format!("{invoked} {cmd}");
        s.push_str(&format!("  • `{typed}`{:pad$}  — {what}\n", "", pad = width - typed.len()));
    }
    s.push_str(&format!(
        "\nFirst-time users typically start by configuring a catalog source \
         (`{invoked} config catalog add <url-or-path>`) and a cache directory \
         (`{invoked} config set cache <dir>`)."
    ));
    s
}

/// What `<invoked> completions --help` shows: how to activate and keep
/// tab-completion for the program run.
fn completions_long_about(invoked: &str) -> String {
    let program = completion_hint_name(invoked).unwrap_or(invoked);
    format!(
        "Print or activate tab-completion for the current shell.\n\n\
         With no arguments, detects your shell from `$SHELL` and prints a wrapper to \
         source. With `--shell <name>`, prints that shell's raw completion script \
         (bash / zsh / fish / elvish / powershell).\n\n\
         To activate for the current session:\n    \
         eval \"$({invoked} completions)\"\n\n\
         To persist (bash):\n    \
         echo 'eval \"$({invoked} completions)\"' >> ~/.bashrc\n\n\
         To persist (zsh):\n    \
         echo 'eval \"$({invoked} completions)\"' >> ~/.zshrc\n\n\
         To emit a raw script for a specific shell:\n    \
         {invoked} completions --shell bash > /etc/bash_completion.d/{program}\n"
    )
}

fn completion_hint_name(invoked: &str) -> Option<&str> {
    invoked.split_whitespace().next()
}

/// Run the command line `argv` (everything after the program name) for
/// `host`, and return its exit status instead of ending the process: 0
/// for success, 1 when the command fails, 2 for a usage error (a
/// malformed command line, a refused request). Standard output and
/// standard error are flushed before it returns.
///
/// It parses and dispatches exactly as [`bin_main_with_catalogs`] does,
/// with `host`'s catalogs; `invoked` is what help, usage lines,
/// `--version` and errors name. The hooks `host` carries run after (or,
/// for precache, before) their own command; see [`Host`].
///
/// `explore`, the interactive TUI, returns its status like every other
/// command, but can still end the process itself on a terminal it
/// cannot drive or a dataset it cannot open.
///
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use vectordata::catalog::{CatalogBuilder, CatalogSources};
/// use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
/// use vectordata::shell::{Host, run};
///
/// // A dataset the host registered: one tiny local base file.
/// let dir = tempfile::tempdir().unwrap();
/// let mut fvec = Vec::new();
/// for i in 0..3 {
///     fvec.extend(2i32.to_le_bytes());
///     fvec.extend((i as f32).to_le_bytes());
///     fvec.extend((i as f32 + 0.5).to_le_bytes());
/// }
/// std::fs::write(dir.path().join("b.fvec"), fvec).unwrap();
/// let entry = CatalogEntry::builder("bench-ds")
///     .home(&vectordata::Location::from_path(dir.path()))
///     .dataset(
///         DatasetConfig::builder("bench-ds")
///             .profile("default", |p| p.facet(StandardFacet::BaseVectors, "b.fvec"))
///             .build()
///             .unwrap(),
///     )
///     .build()
///     .unwrap();
/// let catalogs = CatalogSources::new().with(CatalogBuilder::named("bench").entry(entry).build());
///
/// let described = Rc::new(RefCell::new(None));
/// let seen = described.clone();
/// let host = Host::new()
///     .catalogs(catalogs)
///     .after_describe(move |_catalog, entry, profile| {
///         *seen.borrow_mut() = Some(format!("{}:{profile}", entry.name()));
///     });
///
/// // An empty config directory: the host's catalog is the only one.
/// let config = tempfile::tempdir().unwrap();
/// let argv = ["datasets", "describe", "bench-ds", "--configdir"]
///     .into_iter()
///     .map(String::from)
///     .chain([config.path().to_string_lossy().into_owned()])
///     .collect();
/// assert_eq!(run("bench", argv, &host), 0);
/// assert_eq!(described.borrow().as_deref(), Some("bench-ds:default"));
/// ```
pub fn run(invoked: &str, argv: Vec<String>, host: &Host) -> i32 {
    let code = dispatch(invoked, argv, host);
    flush_std();
    code
}

/// Flush standard output and standard error, as ending the process
/// would.
fn flush_std() {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
}

/// The body of [`run`].
fn dispatch(invoked: &str, argv: Vec<String>, host: &Host) -> i32 {
    // Detached update-probe child (see update_check module docs):
    // marked by an internal env var, does one fetch + state write,
    // and exits before any CLI machinery runs.
    if crate::update_check::run_probe_child_if_marked() {
        return 0;
    }

    // Two specs, one tree: the completion machinery keys on the name
    // the shell completes — the `vectordata` executable's, or an owning
    // host's; everything a user reads names `invoked`.
    let spec = cli_spec(invoked);
    let completing = host.completion_name(invoked);

    // Dynamic-completion entry: when invoked with `_<NAME>_COMPLETE=bash`
    // (`_VECTORDATA_COMPLETE`, or an owning host's), emit candidates and
    // exit. The snippet from `<name> completions` re-invokes the binary with
    // that env var set, so completion logic lives in the spec rather than a
    // frozen script. An owning host's request is the argv it handed over
    // (the line and the cursor); the `vectordata` executable's is read from
    // the process's arguments, as it always was.
    if let Some(name) = completing {
        let exe_spec = cli_spec(name);
        let tree = vcli::build_completion_tree(&exe_spec, &completion_resolvers());
        let answered = if host.owns_completion {
            veks_completion::handle_complete_args(name, &tree, &argv)
        } else {
            veks_completion::handle_complete_env(name, &tree)
        };
        if answered {
            return 0;
        }
        if veks_completion::handle_diagnostic_args(name, &tree) {
            return 0;
        }
    }
    if let Some(name) = completion_hint_name(invoked) {
        veks_completion::hint_completions_unregistered(name);
    }

    if argv.iter().any(|a| a == "--version" || a == "-V") {
        println!("{invoked} {LONG_VERSION}");
        return 0;
    }
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        // Help for the deepest subcommand named on the line (group → leaf).
        print!("{}", vcli::render_help_for(&spec, &argv));
        return 0;
    }

    let parsed = match vcli::parse(&spec, &argv) {
        Ok(p) => p,
        Err(e) => return parse_failure(invoked, &spec, &argv, &e),
    };
    let cli = match <Cli as VeksCli>::veks_from_parsed(&parsed) {
        Ok(c) => c,
        Err(e) => return parse_failure(invoked, &spec, &argv, &e),
    };
    // After parsing only — completion, --help, --version, and
    // malformed invocations never reach the update check. Prints a
    // cached notice instantly (if any) and refreshes in a detached
    // background thread at most once a day; `config set
    // update_check off`, VECTORDATA_NO_UPDATE_CHECK, CI, or a
    // non-terminal stderr each disable it entirely.
    crate::update_check::startup(env!("CARGO_PKG_VERSION"));
    let program = &host.catalogs;
    match cli.command {
        Cmd::Cache { command } => match command {
            // No subcommand → where the cache is and how much it
            // holds, then the subcommands.
            None => {
                print!("{}", cache_summary());
                println!();
                print!("{}", vcli::render_help_for(&spec, &["cache".to_string()]));
                0
            }
            Some(CacheCmd::List { cache_dir, verbose }) => cmd_cache_list(cache_dir, verbose),
            Some(CacheCmd::PruneLegacy { dry_run, cache_dir }) => {
                cmd_cache_prune_legacy(cache_dir, dry_run)
            }
            Some(CacheCmd::Prune {
                dataset,
                dry_run,
                cache_dir,
            }) => cmd_cache_prune(cache_dir, dataset, dry_run),
        },
        Cmd::Datasets {
            command,
            configdir: _,
            catalog: _,
            at: _,
        } => {
            // No subcommand → describe the subcommands, nothing
            // else; the interactive surface is `datasets explore`.
            // Launching a TUI from a bare noun command was a surprise,
            // not a feature.
            let Some(command) = command else {
                print!(
                    "{}",
                    vcli::render_help_for(&spec, &["datasets".to_string()])
                );
                return 0;
            };
            match command {
                DatasetsCmd::List(args) => crate::datasets::list::run_args_listing(args, program, &mut |catalog, entries| {
                    if let Some(hook) = &host.after_list {
                        flush_std();
                        hook(catalog, entries);
                    }
                }),
                DatasetsCmd::Ping {
                    args,
                    configdir,
                    catalog,
                } => crate::datasets::ping::run_args(args, program, &configdir, &catalog),
                DatasetsCmd::Describe {
                    args,
                    configdir,
                    catalog,
                } => crate::datasets::describe::run_args_described(
                    args,
                    program,
                    &configdir,
                    &catalog,
                    &mut |catalog, entry, profile| {
                        if let Some(hook) = &host.after_describe {
                            flush_std();
                            hook(catalog, entry, profile);
                        }
                    },
                ),
                DatasetsCmd::Init(args) => crate::datasets::init::run(&args),
                DatasetsCmd::Curlify(args) => crate::datasets::curlify::run_args(args),
                #[cfg(feature = "explore")]
                DatasetsCmd::Explore(explore) => cmd_explore(invoked, explore, program),
                DatasetsCmd::Push(args) => crate::push::run(args),
                DatasetsCmd::Derive {
                    dataset,
                    profile,
                    output,
                    name,
                    force,
                    shard_stride,
                    max_shard_bytes,
                    configdir,
                    catalog,
                    at,
                } => {
                    let sharding = match crate::dataset::Sharding::from_flags(
                        shard_stride.as_deref(),
                        max_shard_bytes.as_deref(),
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("error: {e}");
                            return 1;
                        }
                    };
                    crate::datasets::derive::run(
                        &dataset,
                        profile.as_deref(),
                        &output,
                        program,
                        &configdir,
                        &catalog,
                        &at,
                        name.as_deref(),
                        force,
                        sharding,
                    )
                }
                DatasetsCmd::Precache {
                    specs,
                    configdir,
                    catalog,
                    at,
                    cache_dir,
                    facet,
                    window,
                    plan,
                    profile,
                    allow_whole_facet,
                } => {
                    let specs = match expand_precache_specs(&specs, host) {
                        Ok(specs) => specs,
                        Err(e) => {
                            eprintln!("error: {e}");
                            return 2;
                        }
                    };
                    crate::datasets::precache::run_specs(
                        &specs,
                        crate::datasets::precache::PrecacheRequest {
                            // run_specs reads the specs, not this.
                            dataset_spec: String::new(),
                            configdir,
                            extra_catalogs: catalog,
                            at,
                            cache_dir,
                            profile,
                            facets: facet,
                            window,
                            plan_only: plan,
                            allow_whole_facet,
                            catalog: None,
                            program_catalogs: program.clone(),
                        },
                    )
                }
            }
        }
        Cmd::Config { command } => match command {
            ConfigCmd::Get { key } => match key.as_deref() {
                None => crate::config::show(),
                Some("cache") => crate::config::get_cache(),
                Some(k @ ("palette" | "curve")) => crate::config::get_ui_setting(k),
                Some("update_check") => crate::config::get_update_check(),
                Some("offline") => crate::config::get_offline(),
                Some(other) => {
                    eprintln!(
                        "unknown config key '{other}' (keys: cache, palette, curve, update_check, offline; or `config get` for all)"
                    );
                    2
                }
            },
            ConfigCmd::Set { key, value, force } => match key.as_str() {
                "cache" => crate::config::set_cache(std::path::Path::new(&value), force),
                k @ ("palette" | "curve") => crate::config::set_ui_setting(k, &value),
                "update_check" => crate::config::set_update_check(&value),
                "offline" => crate::config::set_offline(&value),
                other => {
                    eprintln!(
                        "unknown config key '{other}' (settable keys: cache, palette, curve, update_check, offline)"
                    );
                    2
                }
            },
            ConfigCmd::Mounts { all } => crate::config::list_mounts(all),
            ConfigCmd::Catalog { command } => match command {
                CatalogCmd::Add {
                    source,
                    name,
                    no_verify,
                    trust_self_signed,
                } => crate::config::add_catalog_ex(
                    &source,
                    name.as_deref(),
                    no_verify,
                    trust_self_signed,
                ),
                CatalogCmd::Remove { source, at } => match (source, at) {
                    (Some(s), _) => crate::config::remove_catalog(
                        crate::config::RemoveCatalogSpec::Source(&s),
                    ),
                    (None, Some(n)) => crate::config::remove_catalog(
                        crate::config::RemoveCatalogSpec::Index(n),
                    ),
                    (None, None) => {
                        eprintln!("Specify a catalog URL/path or --at <index>.");
                        eprintln!("Use `vectordata config catalog list` to see indices.");
                        1
                    }
                },
                CatalogCmd::List => crate::config::list_catalogs(),
                CatalogCmd::Disable { name } => crate::config::set_catalog_disabled(&name, true, program),
                CatalogCmd::Enable { name } => crate::config::set_catalog_disabled(&name, false, program),
            },
            ConfigCmd::Overlay { command } => {
                let dir = crate::catalog::sources::config_dir();
                match command {
                    OverlayCmd::Set { dataset, key, value, catalog } => {
                        crate::config::overlay_set(std::path::Path::new(&dir), catalog.as_deref(), &dataset, &key, &value)
                    }
                    OverlayCmd::Unset { dataset, key, catalog } => {
                        crate::config::overlay_unset(std::path::Path::new(&dir), catalog.as_deref(), &dataset, &key)
                    }
                    OverlayCmd::Show { dataset } => crate::config::overlay_show(std::path::Path::new(&dir), dataset.as_deref()),
                }
            }
        },
        Cmd::Completions { shell } => match completing {
            _ if !host.owns_completion && invoked != "vectordata" => embedded_completions_help(invoked),
            Some(name) => cmd_completions(name, shell),
            None => {
                eprintln!(
                    "{invoked}: completion is bound to an executable's name, and '{invoked}' is a command path; \
                     invoke the host by its own name, or graft vectordata::shell::completion_node() under it."
                );
                2
            }
        },
        Cmd::Login {
            url,
            user,
            token,
            password,
            expires,
            list,
        } => {
            if list {
                crate::client_cli::list_logins()
            } else if let Some(url) = url {
                // Accept a URL or a configured catalog name/index.
                let url = match endpoint(Some(url)) {
                    Ok(u) => u,
                    Err(code) => return code,
                };
                let code = crate::client_cli::login(
                    &url,
                    user.as_deref(),
                    token.as_deref(),
                    password.as_deref(),
                    expires.as_deref(),
                );
                // After a successful login, offer (interactively) to register the
                // endpoint's namespaces as catalogs so `datasets list` reflects them.
                // This is the interactive entry point — kept out of the pure handler.
                if code == 0 {
                    crate::client_cli::offer_endpoint_catalogs(&url);
                }
                code
            } else {
                eprintln!("login needs a <url> (or --list)");
                2
            }
        }
        Cmd::Logout { url } => match endpoint(url) {
            Ok(url) => crate::client_cli::logout(&url),
            Err(code) => code,
        },
        Cmd::Whoami { url } => match endpoint(url) {
            Ok(url) => crate::client_cli::ping(&url, false),
            Err(code) => code,
        },
        Cmd::Ping { url } => match endpoint(url) {
            Ok(url) => crate::client_cli::ping(&url, true),
            Err(code) => code,
        },
        Cmd::Token { command } => match command {
            TokenCmd::Issue {
                url,
                description,
                profile,
                expires,
            } => match endpoint(url) {
                Ok(url) => crate::client_cli::token_issue(
                    &url,
                    &description,
                    profile.as_deref(),
                    expires.as_deref(),
                ),
                Err(code) => code,
            },
            TokenCmd::Revoke { url, id } => match endpoint(url) {
                Ok(url) => crate::client_cli::token_revoke(&url, id),
                Err(code) => code,
            },
        },
        Cmd::Backup {
            url,
            to,
            incremental,
        } => {
            // `url` (the source endpoint) accepts a URL or catalog name/index.
            match endpoint(Some(url)) {
                Ok(url) => crate::client_cli::backup(&url, &to, incremental),
                Err(code) => code,
            }
        }
        Cmd::Restore { src, to } => {
            // `to` (the target endpoint) accepts a URL or catalog name/index.
            match endpoint(Some(to)) {
                Ok(to) => crate::client_cli::restore(&src, &to),
                Err(code) => code,
            }
        }
        #[cfg(feature = "explore")]
        Cmd::Explore(explore) => cmd_explore(invoked, explore, program),
    }
}

/// `datasets explore`, and `explore` for short: the explorer over the
/// run's catalogs — the program's after the configured ones, as in
/// every other command, or `--at`'s instead.
#[cfg(feature = "explore")]
fn cmd_explore(invoked: &str, explore: ExploreCommand, program: &crate::catalog::CatalogSources) -> i32 {
    match crate::datasets::sources_for_run(program, &explore.configdir, &explore.catalog, &explore.at) {
        Ok(run) => crate::explore::run_with_sources_as(invoked, explore.args, run.sources().clone()),
        Err(code) => code,
    }
}

/// The exit status for a command line `spec` refused with `e`, after
/// saying why. A group named alone (`cache`, `config catalog`) asks
/// what it holds: its help, listing its subcommands, is printed and the
/// status is 0. A group given options but no subcommand is a usage
/// error that lists them too; any other refusal is its message alone.
fn parse_failure(invoked: &str, spec: &veks_completion::CommandSpec, argv: &[String], e: &vcli::ParseError) -> i32 {
    if !matches!(e, vcli::ParseError::MissingSubcommand { .. }) {
        eprintln!("{invoked}: {e}");
        return 2;
    }
    let help = vcli::render_help_for(spec, argv);
    if argv.iter().all(|w| !w.starts_with('-')) {
        print!("{help}");
        return 0;
    }
    eprintln!("{invoked}: {e}");
    eprint!("\n{help}");
    2
}

/// The specs `datasets precache` runs: each of `specs` through the
/// host's [`before_precache`](Host::before_precache), in order, or as
/// given when it has none. `Err` is the command's error: the hook's, or
/// a spec it expanded to nothing.
fn expand_precache_specs(specs: &[String], host: &Host) -> Result<Vec<String>, String> {
    let Some(expand) = &host.before_precache else {
        return Ok(specs.to_vec());
    };
    let mut out = Vec::new();
    for spec in specs {
        let expanded = expand(spec)?;
        if expanded.is_empty() {
            return Err(format!("'{spec}' names no dataset to precache"));
        }
        out.extend(expanded);
    }
    Ok(out)
}

/// Resolve an endpoint-URL argument, defaulting to the logged-in endpoint when
/// omitted. `Err` is exit status 2, with guidance printed, when not logged in
/// or several are stored.
fn endpoint(url: Option<String>) -> Result<String, i32> {
    crate::credentials::resolve_endpoint(url.as_deref()).map_err(|e| {
        eprintln!("{e}");
        2
    })
}

/// Emit a *dynamic* completion wrapper bound to `name` — `vectordata`,
/// or an owning host's executable. Every completion candidate is
/// resolved live by re-invoking this binary with `COMPLETE=<shell>`
/// set — the shell snippet is one line that hands control back to
/// `CompleteEnv` in `main()`. No frozen completion script ever lives
/// on disk; subcommands added later automatically show up next time
/// the user hits Tab.
///
/// With no `--shell`, auto-detects from `$SHELL` and emits the indirect
/// `source <(…)` wrapper. Returns the exit status: 1 for a shell it
/// does not know.
fn cmd_completions(name: &str, shell: Option<String>) -> i32 {
    // Delegate to the completion framework, which owns the registration
    // protocol (the emitted bash shim calls back via `_<NAME>_COMPLETE`,
    // never a bare `COMPLETE=bash <name>`).
    match shell {
        Some(s) => match veks_completion::Shell::from_name(&s) {
            Some(sh) => {
                veks_completion::print_completions(name, sh);
                0
            }
            None => {
                eprintln!("unknown shell '{s}' (expected bash | zsh | fish | elvish | powershell)");
                1
            }
        },
        None => {
            veks_completion::print_indirect_wrapper(name);
            0
        }
    }
}

/// The CLI's completion tree, as a node: for a program that embeds the
/// CLI under a subcommand of its own (`fvec bench vectordata …`) and
/// answers its own completion protocol, to graft under that
/// subcommand — `host.with_child("vectordata", completion_node())` —
/// so completing the embedded commands needs nothing from the
/// `vectordata` executable.
///
/// ```
/// let node = vectordata::shell::completion_node();
/// assert!(node.child_names().contains(&"datasets"));
/// ```
pub fn completion_node() -> veks_completion::Node {
    vcli::build_completion_tree(&Cli::veks_command_spec("vectordata"), &completion_resolvers()).root
}

/// Completion candidates for `words`, the words after the embedding
/// command (`["datasets", "desc"]` for `fvec bench vectordata datasets
/// desc<TAB>`): for a host that is not built on veks-completion and
/// answers its shell's completion protocol itself.
///
/// ```
/// assert!(vectordata::shell::complete(&["datasets", "desc"]).iter().any(|c| c == "describe"));
/// ```
pub fn complete(words: &[&str]) -> Vec<String> {
    let tree = vcli::build_completion_tree(&Cli::veks_command_spec("vectordata"), &completion_resolvers());
    // The completer's first word is the command's own name.
    let line: Vec<&str> = std::iter::once("vectordata").chain(words.iter().copied()).collect();
    veks_completion::complete(&tree, &line)
}

/// `completions` under an embedding program's name: a script bound to
/// the `vectordata` executable would complete the wrong command, so
/// this says how the host wires completion in, and fails.
fn embedded_completions_help(invoked: &str) -> i32 {
    eprintln!("{invoked}: completion for an embedded vectordata CLI is the host program's to answer:");
    eprintln!("  graft vectordata::shell::completion_node() under its own `vectordata` subcommand, or");
    eprintln!("  answer with vectordata::shell::complete(words), the words after `{invoked}`.");
    1
}

/// Resolve the cache root: caller override > settings.yaml. `Err` is
/// exit status 1, with a printable error, if neither is set, so users
/// see actionable guidance (e.g. the `veks datasets config set cache`
/// hint).
fn resolve_cache_dir(override_: Option<PathBuf>) -> Result<PathBuf, i32> {
    if let Some(p) = override_ {
        return Ok(p);
    }
    crate::settings::cache_dir().map_err(|e| {
        eprintln!("{e}");
        1
    })
}

fn cmd_cache_list(cache_dir: Option<PathBuf>, verbose: bool) -> i32 {
    let root = match resolve_cache_dir(cache_dir) {
        Ok(root) => root,
        Err(code) => return code,
    };
    let listing = match list_entries(&root) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error scanning {}: {}", root.display(), e);
            return 1;
        }
    };
    print_listing(&root, &listing, verbose);
    0
}

fn cmd_cache_prune(cache_dir: Option<PathBuf>, dataset: Option<String>, dry_run: bool) -> i32 {
    let filter = PruneFilter { dataset };
    if filter.is_empty() {
        eprintln!(
            "error: refusing to prune with no filter — pass --dataset \
            with a name or glob pattern."
        );
        eprintln!(
            "(To wipe every cached dataset, remove `<cache_root>` by \
            hand. To clean up pre-cutover detritus, run \
            `vectordata cache prune-legacy`.)"
        );
        return 2;
    }
    let root = match resolve_cache_dir(cache_dir) {
        Ok(root) => root,
        Err(code) => return code,
    };
    let report = match prune_by_filter(&root, &filter, dry_run) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error pruning {}: {}", root.display(), e);
            return 1;
        }
    };

    for (entry, why) in &report.failed {
        eprintln!("warning: failed to remove {}: {why}", entry.path.display());
    }
    if report.matched.is_empty() {
        println!("No cache entries match the filter.");
        return 0;
    }

    println!("Cache directory: {}", root.display());
    let verb = if dry_run { "Would remove" } else { "Removed" };
    println!(
        "{verb} {} entry/entries, freeing {}:",
        report.matched.len(),
        fmt_size(report.matched.iter().map(|e| e.size_bytes).sum::<u64>())
    );
    for entry in &report.matched {
        let rel = entry
            .path
            .strip_prefix(&root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| entry.path.display().to_string());
        let origin = entry.origin_url.as_deref().unwrap_or("<unknown origin>");
        println!("  {:>12}  {origin}", fmt_size(entry.size_bytes));
        println!("              path: {rel}");
    }
    if dry_run {
        println!("\nRun without --dry-run to delete.");
    }
    0
}

fn cmd_cache_prune_legacy(cache_dir: Option<PathBuf>, dry_run: bool) -> i32 {
    let root = match resolve_cache_dir(cache_dir) {
        Ok(root) => root,
        Err(code) => return code,
    };
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
            println!(
                "Would remove {} legacy dir(s) from {}:",
                would_remove.len(),
                root.display()
            );
            for n in &would_remove {
                println!("  {}", n);
            }
            println!("\nRun without --dry-run to delete.");
        }
        return 0;
    }
    match prune_legacy_layout(&root) {
        Ok(removed) if removed.is_empty() => {
            println!("No legacy cache directories found in {}.", root.display());
        }
        Ok(removed) => {
            println!(
                "Removed {} legacy cache dir(s) from {}:",
                removed.len(),
                root.display()
            );
            for n in &removed {
                println!("  {}", n);
            }
        }
        Err(e) => {
            eprintln!("error pruning {}: {}", root.display(), e);
            return 1;
        }
    }
    0
}

// ─── Formatting ──────────────────────────────────────────────────────

/// The two lines that open `cache list` and a bare `cache`: where the
/// cache is and how much of the disk it takes.
fn cache_header(root: &std::path::Path, total: &str) -> String {
    format!("Cache directory: {}\nTotal on disk:   {total}\n", root.display())
}

/// What a bare `cache` opens with: [`cache_header`] for the configured
/// cache, or why there is none.
fn cache_summary() -> String {
    match crate::settings::cache_dir() {
        Ok(root) => {
            let total = match list_entries(&root) {
                Ok(l) => fmt_size(l.total_bytes()),
                Err(e) => format!("unknown ({e})"),
            };
            cache_header(&root, &total)
        }
        Err(e) => format!("Cache directory: none ({e})\n"),
    }
}

fn print_listing(root: &std::path::Path, l: &CacheListing, verbose: bool) {
    print!("{}", cache_header(root, &fmt_size(l.total_bytes())));
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

    if !l.url_derived.is_empty() {
        println!("URL-derived caches (from direct URL opens, no catalog identity):");
        print_simple_rows(&l.url_derived, root);
        println!();
    }

    if !l.legacy.is_empty() {
        println!("Legacy detritus (pre-cutover layout):");
        print_simple_rows(&l.legacy, root);
        println!("  → remove with: vectordata cache prune-legacy");
        println!();
    }

    if !l.catalogs.is_empty() {
        println!("Kept catalog copies (read when a catalog server is unreachable, or offline):");
        print_simple_rows(&l.catalogs, root);
        println!();
    }

    if !l.other.is_empty() {
        println!("Other top-level entries (unrecognised):");
        print_simple_rows(&l.other, root);
        println!();
    }

    if l.datasets.is_empty()
        && l.url_derived.is_empty()
        && l.catalogs.is_empty()
        && l.legacy.is_empty()
        && l.other.is_empty()
    {
        println!("(empty)");
    }
}

/// Two-column path/size rows for the catalog/legacy/other categories.
fn print_simple_rows(rows: &[CacheEntry], root: &std::path::Path) {
    let name_w = rows
        .iter()
        .map(|e| rel_name(&e.path, root).len())
        .max()
        .unwrap_or(20)
        .max(20);
    for e in rows {
        let name = rel_name(&e.path, root);
        println!(
            "  {:<name_w$}  {:>12}  ({} file{})",
            name,
            fmt_size(e.size_bytes),
            e.file_count,
            if e.file_count == 1 { "" } else { "s" }
        );
    }
}

/// Default dataset listing: one row per dataset, name + size +
/// file count + origin host.
fn print_dataset_rows(rows: &[CacheEntry], root: &std::path::Path) {
    let name_w = rows
        .iter()
        .map(|e| rel_name(&e.path, root).len())
        .max()
        .unwrap_or(20)
        .max(20);
    for e in rows {
        let name = rel_name(&e.path, root);
        let host = e.origin_host.as_deref().unwrap_or("");
        let host_note = if host.is_empty() {
            String::new()
        } else {
            format!("  from {host}")
        };
        println!(
            "  {:<name_w$}  {:>12}  ({} file{}){}",
            name,
            fmt_size(e.size_bytes),
            e.file_count,
            if e.file_count == 1 { "" } else { "s" },
            host_note
        );
    }
}

/// Verbose dataset listing: full origin URL per entry, on its own row.
fn print_dataset_rows_verbose(rows: &[CacheEntry], root: &std::path::Path) {
    for e in rows {
        let rel = rel_name(&e.path, root);
        let origin = e.origin_url.as_deref().unwrap_or("<unknown origin>");
        println!(
            "  {:>12}  {} ({} file{})",
            fmt_size(e.size_bytes),
            rel,
            e.file_count,
            if e.file_count == 1 { "" } else { "s" }
        );
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
    if bytes >= TIB {
        format!("{:.1} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The completion reminder names the command the user ran**: an
    /// embedded CLI's is its host's, not `vectordata`'s.
    #[test]
    fn the_completion_hint_names_the_program_run() {
        assert_eq!(completion_hint_name("vectordata"), Some("vectordata"));
        assert_eq!(completion_hint_name("fvec-bench"), Some("fvec-bench"));
        assert_eq!(completion_hint_name("fvec bench vectordata"), Some("fvec"));
        assert_eq!(completion_hint_name(""), None);
    }

    /// Every command path the binary accepts, to depth two (`datasets
    /// precache`, `cache prune`, `token issue`) — the granularity the
    /// CLI ↔ library map in `AGENTS.md` is kept at.
    fn command_paths() -> Vec<String> {
        let root = <Cli as VeksCli>::veks_command_spec("vectordata");
        let mut out = Vec::new();
        for top in &root.subcommands {
            if top.subcommands.is_empty() {
                out.push(top.name.clone());
            }
            for sub in &top.subcommands {
                out.push(format!("{} {}", top.name, sub.name));
            }
        }
        out
    }

    /// Every command path the binary accepts, at any depth (`config
    /// catalog disable` as well as `config catalog`): a row may name a
    /// command below the table's granularity, but only one that exists.
    fn all_command_paths() -> Vec<String> {
        fn walk(prefix: &str, spec: &veks_completion::CommandSpec, out: &mut Vec<String>) {
            for sub in &spec.subcommands {
                let path = if prefix.is_empty() { sub.name.clone() } else { format!("{prefix} {}", sub.name) };
                out.push(path.clone());
                walk(&path, sub, out);
            }
        }
        let mut out = Vec::new();
        walk("", &<Cli as VeksCli>::veks_command_spec("vectordata"), &mut out);
        out
    }

    /// **An embedding binary's invocation is what help names.** The
    /// spec `bin_main_as` builds carries the path the user typed, so the
    /// usage line at every depth reads as a command they can rerun.
    #[test]
    fn help_names_the_embedding_invocation() {
        let spec = <Cli as VeksCli>::veks_command_spec("fvec bench vectordata");
        let top = vcli::render_help_for(&spec, &["--help"]);
        assert!(top.contains("Usage: fvec bench vectordata <COMMAND>"), "{top}");
        let leaf = vcli::render_help_for(&spec, &["datasets", "list", "--help"]);
        assert!(leaf.contains("Usage: fvec bench vectordata datasets list"), "{leaf}");
        let err = vcli::parse(&spec, &["no-such-command".to_string()]).unwrap_err();
        assert!(!err.to_string().contains("vectordata vectordata"), "{err}");
    }

    /// **Every command names the library call it runs** (SRD DX-28).
    ///
    /// The CLI ↔ library table in `AGENTS.md` is how a program finds
    /// the call behind a command it has watched work, which is the
    /// miss the SRD opens with. A command added without a row fails
    /// here, and a row naming a command that no longer exists does too.
    /// Rows are required to depth two and allowed deeper.
    /// (The links in each row are checked by rustdoc: the file renders
    /// as `crate::_agents`.)
    #[test]
    fn every_command_has_a_row_in_the_agents_cli_map() {
        let agents = include_str!("../AGENTS.md");
        let map = &agents[agents.find("## CLI ↔ library").expect("AGENTS.md has the CLI map")..];
        let rows: Vec<&str> = map
            .lines()
            .filter_map(|l| l.strip_prefix("| `vectordata "))
            .filter_map(|l| l.split('`').next())
            .collect();
        let commands = command_paths();
        let missing: Vec<&String> = commands.iter().filter(|c| !rows.contains(&c.as_str())).collect();
        assert!(missing.is_empty(), "commands with no CLI ↔ library row in AGENTS.md: {missing:?}");
        let all = all_command_paths();
        let stale: Vec<&&str> = rows.iter().filter(|r| !all.iter().any(|c| c == **r)).collect();
        assert!(stale.is_empty(), "AGENTS.md rows for commands that do not exist: {stale:?}");
    }

    /// The registration contract for this binary's dynamic
    /// completion: the shared dataset-domain resolvers plus the
    /// vectordata-specific ones, no more and no less. The
    /// key → tree attachment mechanics are veks-completion's own
    /// tested behavior; what this binary owns is WHICH resolvers it
    /// registers.
    #[test]
    fn completion_resolvers_cover_datasets_and_local_surfaces() {
        let keys: Vec<String> = completion_resolvers().into_keys().collect();
        assert_eq!(
            keys,
            vec![
                "--at".to_string(),
                "--dataset".to_string(),
                "--matching-name".to_string(),
                "--matching-profile".to_string(),
                "--profile".to_string(),
                "--select".to_string(),
                "--to".to_string(),
                "--with-count".to_string(),
                "--with-data".to_string(),
                "--with-dim".to_string(),
                "--with-facet".to_string(),
                "--with-max-count".to_string(),
                "--with-max-data".to_string(),
                "--with-max-dim".to_string(),
                "--with-metric".to_string(),
                "--with-min-count".to_string(),
                "--with-min-data".to_string(),
                "--with-min-dim".to_string(),
                "--with-vtype".to_string(),
                "config catalog disable".to_string(),
                "config catalog enable".to_string(),
                "config get".to_string(),
                "config set".to_string(),
                "datasets describe".to_string(),
                "datasets ping".to_string(),
                "datasets precache".to_string(),
            ]
        );
    }

    /// **`config catalog disable`/`enable` are in the completion tree,
    /// and complete the names each can take**: disable the catalogs not
    /// disabled, enable the disabled ones.
    #[test]
    fn catalog_disable_and_enable_complete() {
        let verbs = complete(&["config", "catalog", ""]);
        for verb in ["add", "remove", "list", "disable", "enable"] {
            assert!(verbs.iter().any(|v| v.trim_end() == verb), "{verb}: {verbs:?}");
        }
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let configured = names(&["c1", "c2", "c3"]);
        let disabled = names(&["c2"]);
        assert_eq!(catalog_toggle_candidates(&configured, &disabled, true, ""), ["c1", "c3"]);
        assert_eq!(catalog_toggle_candidates(&configured, &disabled, true, "c3"), ["c3"]);
        assert_eq!(catalog_toggle_candidates(&configured, &disabled, false, ""), ["c2"]);
        assert!(catalog_toggle_candidates(&configured, &disabled, false, "c1").is_empty());
    }
}
