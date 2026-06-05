// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `vecd` command-line surface (clap derive) and its dispatch. Mirrors
//! the `vectordata` binary's idiom: a thin binary over library entry
//! points, with `clap_complete` dynamic completions (no frozen script).
//!
//! The admin verbs operate directly on the SQLite DB
//! (superuser-by-filesystem-access); `serve` runs the gateway.

use std::net::SocketAddr;
use std::path::PathBuf;

use crate::model::{Level, Listable, VecdError};
use crate::{admin, backup, config, db::Db, server};

#[derive(veks_completion_derive::VeksCli)]
#[command(
    name = "vecd",
    about = "vectordata endpoint daemon — an AAA gateway in front of object storage",
    version
)]
pub struct Cli {
    /// Data directory (DB, pidfile, local-backend objects). Defaults to
    /// the config dir's `data/`.
    #[arg(long, global = true)]
    pub data_dir: Option<PathBuf>,

    /// Config directory to use (overrides `$VECD_CONFIG`; warns if both are
    /// set). For a one-shot from a json/yaml file, see `--config-import`.
    #[arg(long, global = true, value_name = "DIR")]
    pub conf: Option<PathBuf>,

    /// When a config exists in BOTH the current dir and your home config,
    /// use the current-dir one (`./vecd.conf`).
    #[arg(long, global = true, conflicts_with = "config_is_home")]
    pub config_is_local: bool,

    /// When a config exists in BOTH locations, use the home one
    /// (`~/.config/vecd/vecd.conf`).
    #[arg(long, global = true)]
    pub config_is_home: bool,

    #[command(subcommand)]
    pub command: Cmd,
}

#[derive(veks_completion_derive::VeksCli)]
pub enum Cmd {
    /// Create the DB + schema and mint the first superuser token.
    Init {
        /// Name for the bootstrap superuser.
        #[arg(long, default_value = "root")]
        superuser: String,
        /// Print only the token plaintext (for `$(…)` capture in scripts).
        #[arg(long)]
        quiet: bool,
        /// Single-shot standup: seed the config from this json/yaml/dir before
        /// creating the DB (allowed with no pre-existing config).
        #[arg(long, value_name = "PATH")]
        config_import: Option<PathBuf>,
    },
    /// Inspect and edit operator configuration (`vecd.conf`).
    Config {
        #[command(subcommand)]
        command: ConfigCmd,
    },
    /// Authenticate to a vecd endpoint and store its token (in the config
    /// dir's `credentials.json`); it's then used automatically for that
    /// endpoint (e.g. `vecd whoami`).
    Login {
        /// Endpoint URL (its origin identifies the credential). Omit with `--list`.
        url: Option<String>,
        /// Store this token directly instead of logging in with a password.
        #[arg(long)]
        token: Option<String>,
        /// User to authenticate as (with `--password`).
        #[arg(long)]
        user: Option<String>,
        /// Password for `--user` (a password grant via `/auth/token`); else
        /// `$VECD_PASSWORD`.
        #[arg(long)]
        password: Option<String>,
        /// Requested token lifetime, e.g. `30d`.
        #[arg(long)]
        expires: Option<String>,
        /// List endpoints with stored credentials and exit.
        #[arg(long)]
        list: bool,
    },
    /// Forget the stored token for an endpoint.
    Logout {
        /// Endpoint URL (defaults to the one you're logged in to).
        url: Option<String>,
    },
    /// Show your effective access at an endpoint (uses the stored token).
    Whoami {
        /// Endpoint URL (defaults to the one you're logged in to).
        url: Option<String>,
    },
    /// Run the gateway in the foreground (the workhorse `start` execs).
    Serve {
        #[command(flatten)]
        serve: ServeArgs,
    },
    /// Start the gateway as a background daemon (self-daemonizing).
    Start {
        #[command(flatten)]
        serve: ServeArgs,
    },
    /// Stop the background daemon (SIGTERM, graceful drain).
    Stop,
    /// Report whether the daemon is running and where it is bound.
    Status,
    /// Restart the background daemon.
    Restart {
        #[command(flatten)]
        serve: ServeArgs,
    },
    /// Manage users.
    Users {
        #[command(subcommand)]
        command: UsersCmd,
    },
    /// Manage API tokens.
    Tokens {
        #[command(subcommand)]
        command: TokensCmd,
    },
    /// Manage roles.
    Roles {
        #[command(subcommand)]
        command: RolesCmd,
    },
    /// Manage named storage backend configs.
    Backends {
        #[command(subcommand)]
        command: BackendsCmd,
    },
    /// Manage namespaces.
    Ns {
        #[command(subcommand)]
        command: NsCmd,
    },
    /// Bind a role to a principal on a namespace subtree.
    Bind {
        #[arg(long)]
        to: String,
        #[arg(long)]
        role: String,
        #[arg(long)]
        ns: String,
    },
    /// Remove a binding (omit `--role` to remove all for the principal).
    Unbind {
        #[arg(long)]
        to: String,
        #[arg(long)]
        role: Option<String>,
        #[arg(long)]
        ns: String,
    },
    /// Convenience grant: map `--read/--write/--delete/--admin` to a role.
    Grant {
        #[arg(long)]
        to: String,
        #[arg(long)]
        ns: String,
        #[arg(long)]
        read: bool,
        #[arg(long)]
        write: bool,
        #[arg(long)]
        delete: bool,
        #[arg(long)]
        admin: bool,
    },
    /// Manage system privileges (e.g. IGNORE-QUOTAS).
    Priv {
        #[command(subcommand)]
        command: PrivCmd,
    },
    /// Manage named, parameterized privilege profiles.
    Profiles {
        #[command(subcommand)]
        command: ProfilesCmd,
    },
    /// Control-plane DB backup.
    Backup {
        #[command(subcommand)]
        command: BackupCmd,
    },
    /// Install a snapshot as the active DB (stop the daemon first).
    Restore {
        snapshot: String,
    },
    /// Inspect and act on the lifecycle cleanup queue (stasis versions).
    Cleanup {
        #[command(subcommand)]
        command: CleanupCmd,
    },
    /// List live objects under a namespace (key + size), with a total.
    Objects {
        /// Namespace path, e.g. `datasets`.
        ns: String,
    },
    /// List session-published versions of a namespace (newest first). Plain
    /// `push` writes are lone objects (see `vecd objects`); versions appear
    /// here only for transactional, session-based publication.
    Versions {
        /// Namespace path, e.g. `datasets/mydata`.
        ns: String,
    },
    /// List role bindings (who has what access where).
    Bindings {
        /// Only bindings whose namespace path starts with this prefix.
        #[arg(long)]
        ns: Option<String>,
    },
    /// Read the access log (newest last), with optional filters.
    Log {
        #[arg(long, default_value = "50")]
        tail: usize,
        /// Only rows for this principal.
        #[arg(long)]
        principal: Option<String>,
        /// Only rows with this HTTP status code (e.g. 403).
        #[arg(long)]
        status: Option<i64>,
        /// Only rows for this action (HTTP method, e.g. GET, PUT).
        #[arg(long)]
        action: Option<String>,
    },
    /// Emit a sourceable shell snippet that activates dynamic completions:
    ///   eval "$(vecd completions)"
    Completions {
        #[arg(long, value_parser = ["bash", "zsh", "fish", "elvish", "powershell"])]
        shell: Option<String>,
    },
}

/// Flags shared by `serve`, `start`, and `restart`.
#[derive(Clone, veks_completion_derive::VeksCli)]
pub struct ServeArgs {
    /// Bind address (use `:0` for an ephemeral port). Falls back to the
    /// `bind` key in `vecd.conf`, then to the local-only default
    /// `127.0.0.1:8443`. Bind a non-loopback address to expose vecd on the
    /// network (configure TLS too, or vecd warns about cleartext).
    #[arg(long)]
    pub bind: Option<String>,
    /// TLS certificate (PEM). With `--tls-key`, serves HTTPS.
    #[arg(long)]
    pub tls_cert: Option<PathBuf>,
    /// TLS private key (PEM).
    #[arg(long)]
    pub tls_key: Option<PathBuf>,
    /// Back the control-plane DB up here (file path or `s3://…`).
    #[arg(long)]
    pub db_backup: Option<String>,
    /// Backup interval, e.g. `1h`.
    #[arg(long, default_value = "1h")]
    pub backup_interval: String,
    /// Keep the newest N snapshots.
    #[arg(long, default_value = "24")]
    pub backup_retain: usize,
    /// Per-connection (IP:port) **download** cap, e.g. `8MiB`. `0`/unset =
    /// unlimited. Falls back to `ratelimit_connection_download` in
    /// `vecd.conf`. With more connections, aggregate scales past this.
    #[arg(long, value_name = "RATE")]
    pub ratelimit_connection_download: Option<String>,
    /// Per-connection (IP:port) **upload** cap. Falls back to
    /// `ratelimit_connection_upload` in `vecd.conf`.
    #[arg(long, value_name = "RATE")]
    pub ratelimit_connection_upload: Option<String>,
    /// Per-client (IP) **download** cap — shared across all of one host's
    /// connections, so concurrency can't exceed it. Falls back to
    /// `ratelimit_client_download` in `vecd.conf`.
    #[arg(long, value_name = "RATE")]
    pub ratelimit_client_download: Option<String>,
    /// Per-client (IP) **upload** cap. Falls back to
    /// `ratelimit_client_upload` in `vecd.conf`.
    #[arg(long, value_name = "RATE")]
    pub ratelimit_client_upload: Option<String>,
}

#[derive(veks_completion_derive::VeksCli)]
pub enum ConfigCmd {
    /// Write sensible defaults (local-only bind + data dir) and confirm.
    Auto {
        /// Don't prompt — write the defaults straight away (for scripts).
        #[arg(long)]
        yes: bool,
        /// Overwrite an existing config (required once one is established).
        #[arg(long)]
        force: bool,
    },
    /// Read config — the whole config, one value, or to a file/format.
    ///
    ///   vecd config get                 # whole config (native) → stdout
    ///   vecd config get bind            # one value → stdout
    ///   vecd config get --format json   # whole config as JSON → stdout
    ///   vecd config get --out cfg.yaml  # whole config → a .json/.yaml file or dir
    Get {
        /// A single config key; omit to read the whole config.
        param: Option<String>,
        /// Write the whole config to this path (`.json`/`.yaml` file, a
        /// directory's `vecd.conf`, or `-` for stdout) instead of stdout.
        #[arg(long, value_name = "PATH")]
        out: Option<PathBuf>,
        /// Stdout format for the whole config: `native` (default), `json`, or
        /// `yaml`.
        #[arg(long, value_name = "FMT")]
        format: Option<String>,
    },
    /// Write config — set one value, or replace the whole config from a source.
    ///
    ///   vecd config set bind 0.0.0.0:8443    # one value
    ///   vecd config set --from cfg.json      # replace from json/yaml/conf/dir
    ///   vecd config get | vecd config set --from -   # round-trip via stdin
    ///
    /// Lock against edits with `vecd config set lock_config on`.
    Set {
        /// Config key (with `<value>`); omit when using `--from`.
        param: Option<String>,
        /// Value for `<key>`.
        value: Option<String>,
        /// Replace the whole config from this source: a `.json`/`.yaml`/`.conf`
        /// file, a directory's `vecd.conf`, or `-` for native text on stdin.
        #[arg(long, value_name = "PATH")]
        from: Option<PathBuf>,
        /// Required to change an already-set value, or to replace an existing
        /// config.
        #[arg(long)]
        force: bool,
    },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum UsersCmd {
    Add {
        name: String,
        #[arg(long, default_value = "user")]
        level: String,
        /// Set a password (for `vectordata login`); prompts are Phase 2.
        #[arg(long)]
        password: Option<String>,
        /// Auto-provision a home namespace on this backend config.
        #[arg(long)]
        home_backend: Option<String>,
    },
    List,
    Level { name: String, level: String },
    Passwd { name: String, password: String },
    Disable { name: String },
    Enable { name: String },
    Remove { name: String },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum TokensCmd {
    Create {
        #[arg(long)]
        user: String,
        #[arg(long)]
        description: String,
        #[arg(long)]
        expires: Option<String>,
        /// Ad-hoc profile, e.g. "read datasets/glove, publish datasets/scratch".
        #[arg(long)]
        profile: Option<String>,
        /// Expand a named profile instead.
        #[arg(long)]
        from: Option<String>,
        /// Fill a placeholder of `--from` (repeatable): `pos=val`.
        #[arg(long = "set", value_name = "POS=VAL")]
        sets: Vec<String>,
        /// Print only the token plaintext (for `$(…)` capture in scripts).
        #[arg(long)]
        quiet: bool,
        /// Print the token record as JSON ({token,user,id,expires_at,…}) —
        /// readable back via `--token <file>` and `login --token <file>`.
        #[arg(long, conflicts_with = "quiet")]
        json: bool,
    },
    List {
        #[arg(long)]
        user: Option<String>,
    },
    Revoke { id: i64 },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum RolesCmd {
    List,
    Add {
        name: String,
        #[arg(long)]
        actions: String,
    },
    Remove { name: String },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum BackendsCmd {
    Add {
        name: String,
        #[arg(long)]
        kind: String,
        #[arg(long)]
        endpoint: String,
        #[arg(long)]
        endpoint_url: Option<String>,
        #[arg(long)]
        region: Option<String>,
        #[arg(long)]
        aws_profile: Option<String>,
        #[arg(long)]
        active: bool,
    },
    List,
    Set {
        name: String,
        #[arg(long)]
        active: bool,
        #[arg(long)]
        no_active: bool,
    },
    Remove { name: String },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum NsCmd {
    Add {
        path: String,
        #[arg(long)]
        owner: String,
        #[arg(long)]
        backend_config: Option<String>,
        #[arg(long)]
        active: bool,
        #[arg(long, default_value = "grantees")]
        listable: String,
        #[arg(long)]
        ttl: Option<String>,
        #[arg(long)]
        quota: Option<String>,
    },
    List,
    Set {
        path: String,
        #[arg(long)]
        owner: Option<String>,
        #[arg(long)]
        backend_config: Option<String>,
        #[arg(long)]
        active: bool,
        #[arg(long)]
        no_active: bool,
        #[arg(long)]
        listable: Option<String>,
        #[arg(long)]
        ttl: Option<String>,
        #[arg(long)]
        quota: Option<String>,
    },
    Remove { path: String },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum PrivCmd {
    Grant {
        privilege: String,
        #[arg(long)]
        to: String,
    },
    Revoke {
        privilege: String,
        #[arg(long)]
        to: String,
    },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum ProfilesCmd {
    Add {
        name: String,
        #[arg(long)]
        owner: String,
        #[arg(long)]
        spec: String,
    },
    List,
    Remove { name: String },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum CleanupCmd {
    /// List versions in stasis (expired, awaiting extend or purge).
    List,
    /// Restore a stasis version (`<ns>@<selector>`), optionally re-lifecycled.
    Extend {
        target: String,
        #[arg(long)]
        duration: Option<String>,
    },
    /// Physically delete a stasis version (`<ns>@<selector>`) — removes bytes.
    Purge { target: String },
}

#[derive(veks_completion_derive::VeksCli)]
pub enum BackupCmd {
    Now {
        /// Destination (file path or `s3://…`); defaults to config `db_backup`.
        dest: Option<String>,
    },
    List {
        dest: Option<String>,
    },
}

/// Resolve the data dir: `--data-dir`, else config `data_dir`, else the
/// default under the config dir.
fn resolve_data_dir(flag: &Option<PathBuf>, cfg: &config::Config, config_dir: &std::path::Path) -> PathBuf {
    if let Some(d) = flag {
        return d.clone();
    }
    if let Some(d) = cfg.get("data_dir") {
        return PathBuf::from(d);
    }
    config_dir.join("data")
}

/// Every command except `config` and `completions` needs a configured base
/// path, so they require a `vecd.conf` to exist. `init --config-import` is the
/// single-shot exception — it bootstraps the config itself.
fn requires_config(cmd: &Cmd) -> bool {
    !matches!(
        cmd,
        Cmd::Config { .. }
            | Cmd::Completions { .. }
            | Cmd::Login { .. }
            | Cmd::Logout { .. }
            | Cmd::Whoami { .. }
            | Cmd::Init { config_import: Some(_), .. }
    )
}

/// Resolve the bind address: `--bind`, else config `bind`, else the default
/// **loopback** `127.0.0.1:8443`. Mirrors [`resolve_data_dir`]'s
/// flag-over-config precedence. The default is loopback so a bare
/// `vecd start` is safe out of the box — reachable only from the local host;
/// exposing it to the network is an explicit choice (set `bind` to a
/// non-loopback address), and pairing that with TLS is checked by
/// [`exposes_plaintext`].
fn resolve_bind(flag: &Option<String>, cfg: &config::Config) -> String {
    flag.clone()
        .or_else(|| cfg.get("bind").map(|s| s.to_string()))
        .unwrap_or_else(|| "127.0.0.1:8443".to_string())
}

/// Whether binding `addr` without TLS would serve **plaintext beyond the
/// local host** — i.e. a non-loopback address with no TLS configured. The
/// operator is warned (not refused) in that case: terminating TLS at a
/// reverse proxy in front of a plaintext vecd is a legitimate setup, so this
/// can't be a hard error — but the bare "exposed plaintext on a public
/// interface" mistake deserves a loud heads-up. Pure function so the policy
/// is unit-tested without binding a socket.
fn exposes_plaintext(addr: &SocketAddr, tls_configured: bool) -> bool {
    !tls_configured && !addr.ip().is_loopback()
}

/// Resolve the four bandwidth caps with flag-over-config precedence (each
/// axis independent). A `--ratelimit-*` flag wins; else the matching
/// `vecd.conf` key; else `0` (unlimited). CLI flags and config keys are
/// congruent mirrors — every cap is reachable from both surfaces.
fn resolve_rate_limits(
    a: &ServeArgs,
    cfg: &config::Config,
) -> Result<crate::ratelimit::RateLimits, VecdError> {
    let axis = |flag: &Option<String>, key: &str| -> Result<u64, VecdError> {
        match flag {
            Some(v) => config::parse_byte_rate(v)
                .map_err(|m| VecdError::usage(format!("--{}: {m}", key.replace('_', "-")))),
            None => cfg.byte_rate(key),
        }
    };
    Ok(crate::ratelimit::RateLimits {
        connection_download: axis(&a.ratelimit_connection_download, "ratelimit_connection_download")?,
        connection_upload: axis(&a.ratelimit_connection_upload, "ratelimit_connection_upload")?,
        client_download: axis(&a.ratelimit_client_download, "ratelimit_client_download")?,
        client_upload: axis(&a.ratelimit_client_upload, "ratelimit_client_upload")?,
    })
}

/// Open the DB, surfacing the SQLCipher key from config into the env (used
/// only when built with the `sqlcipher` feature).
fn open_db(data_dir: &std::path::Path, cfg: &config::Config) -> Result<Db, VecdError> {
    if let Some(key) = cfg.db_key() {
        // SAFETY: single-threaded CLI startup, before any DB open.
        unsafe { std::env::set_var("VECD_DB_KEY", key) };
    }
    Db::open(&config::db_path(data_dir))
}

/// Entry point. Returns a process exit code.
pub fn run(cli: Cli) -> i32 {
    let prefer = if cli.config_is_local {
        Some(config::Prefer::Local)
    } else if cli.config_is_home {
        Some(config::Prefer::Home)
    } else {
        None
    };
    let resolved = match config::resolve(cli.conf.as_deref(), prefer) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("vecd: {e}");
            return e.exit_code();
        }
    };

    // Single-shot standup: seed the config from --config-import before anything
    // reads it, so the rest of startup (data-dir resolution, the DB) sees it.
    // This is for a *fresh* standup — it won't clobber an existing config.
    if let Cmd::Init { config_import: Some(src), .. } = &cli.command {
        if resolved.exists {
            eprintln!(
                "vecd: a config already exists at {} ({}); `init --config-import` is for fresh \
                 standup. Use `vecd config set --from {} --force` then `vecd init`.",
                resolved.dir.display(),
                resolved.source.label(),
                src.display()
            );
            return VecdError::usage("config already exists").exit_code();
        }
        match config::import_from(src).and_then(|c| c.write_to(&resolved.dir).map(|_| ())) {
            Ok(()) => eprintln!(
                "vecd: seeded config from {} → {}",
                src.display(),
                resolved.conf_path().display()
            ),
            Err(e) => {
                eprintln!("vecd: {e}");
                return e.exit_code();
            }
        }
    }

    // Gate: refuse to operate against an unconfigured base path.
    if requires_config(&cli.command) && !resolved.exists {
        eprintln!(
            "vecd: not configured — no vecd.conf at {} (resolved via {}).\n\
             Run `vecd config auto` to create one, or `vecd init --config-import <json|yaml|dir>` \
             for a one-shot standup. (`vecd config …` and `vecd completions` don't need a config.)",
            resolved.dir.display(),
            resolved.source.label()
        );
        return VecdError::usage("not configured").exit_code();
    }

    let cfg = match config::Config::load(&resolved.dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("vecd: {e}");
            return e.exit_code();
        }
    };
    let data_dir = resolve_data_dir(&cli.data_dir, &cfg, &resolved.dir);

    let result = dispatch(cli.command, &resolved, &data_dir, &cfg);
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("vecd: {e}");
            e.exit_code()
        }
    }
}

fn dispatch(
    cmd: Cmd,
    resolved: &config::Resolved,
    data_dir: &std::path::Path,
    cfg: &config::Config,
) -> Result<(), VecdError> {
    match cmd {
        // config_import was already applied in `run`; just build the DB here.
        Cmd::Init { superuser, quiet, config_import: _ } => {
            cmd_init(resolved, data_dir, cfg, &superuser, quiet)
        }
        Cmd::Config { command } => cmd_config(resolved, command),
        Cmd::Login { url, token, user, password, expires, list } => {
            cmd_login(resolved, url, token, user, password, expires.as_deref(), list)
        }
        Cmd::Logout { url } => cmd_logout(resolved, resolve_endpoint(resolved, url)?.as_str()),
        Cmd::Whoami { url } => cmd_client_whoami(resolved, resolve_endpoint(resolved, url)?.as_str()),
        Cmd::Serve { serve } => cmd_serve(data_dir, cfg, &serve),
        Cmd::Start { serve } => {
            let dd = data_dir.to_path_buf();
            let cfg = cfg.clone();
            crate::daemon::start(data_dir, move || match cmd_serve(&dd, &cfg, &serve) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("vecd: {e}");
                    e.exit_code()
                }
            })
        }
        Cmd::Stop => crate::daemon::stop(data_dir),
        Cmd::Status => crate::daemon::status(data_dir),
        Cmd::Restart { serve } => {
            // Best-effort stop (ignore "not running"), then start.
            let _ = crate::daemon::stop(data_dir);
            let dd = data_dir.to_path_buf();
            let cfg = cfg.clone();
            crate::daemon::start(data_dir, move || match cmd_serve(&dd, &cfg, &serve) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("vecd: {e}");
                    e.exit_code()
                }
            })
        }
        Cmd::Users { command } => cmd_users(data_dir, cfg, command),
        Cmd::Tokens { command } => cmd_tokens(data_dir, cfg, command),
        Cmd::Roles { command } => cmd_roles(data_dir, cfg, command),
        Cmd::Backends { command } => cmd_backends(data_dir, cfg, command),
        Cmd::Ns { command } => cmd_ns(data_dir, cfg, command),
        Cmd::Bind { to, role, ns } => {
            let mut db = open_db(data_dir, cfg)?;
            admin::bind(&mut db, &to, &role, &ns)?;
            println!("bound {role} → {to} on {ns}");
            Ok(())
        }
        Cmd::Unbind { to, role, ns } => {
            let mut db = open_db(data_dir, cfg)?;
            admin::unbind(&mut db, &to, role.as_deref(), &ns)?;
            println!("unbound {to} on {ns}");
            Ok(())
        }
        Cmd::Grant { to, ns, read, write, delete, admin: adm } => {
            cmd_grant(data_dir, cfg, &to, &ns, read, write, delete, adm)
        }
        Cmd::Priv { command } => cmd_priv(data_dir, cfg, command),
        Cmd::Profiles { command } => cmd_profiles(data_dir, cfg, command),
        Cmd::Backup { command } => cmd_backup(data_dir, cfg, command),
        Cmd::Restore { snapshot } => {
            backup::restore(&snapshot, &config::db_path(data_dir))?;
            println!("restored {snapshot} → {}", config::db_path(data_dir).display());
            Ok(())
        }
        Cmd::Cleanup { command } => cmd_cleanup(data_dir, cfg, command),
        Cmd::Objects { ns } => cmd_objects(data_dir, cfg, &ns),
        Cmd::Versions { ns } => cmd_versions(data_dir, cfg, &ns),
        Cmd::Bindings { ns } => cmd_bindings(data_dir, cfg, ns.as_deref()),
        Cmd::Log { tail, principal, status, action } => {
            cmd_log(data_dir, cfg, tail, principal.as_deref(), status, action.as_deref())
        }
        Cmd::Completions { shell } => {
            cmd_completions(shell);
            Ok(())
        }
    }
}

fn cmd_init(
    resolved: &config::Resolved,
    data_dir: &std::path::Path,
    cfg: &config::Config,
    superuser: &str,
    quiet: bool,
) -> Result<(), VecdError> {
    // The config already exists by here (the gate requires it, and
    // `--config-import` seeded it in `run`) — `config auto`/`import` own config
    // creation now, so init only builds the DB.
    let db_path = config::db_path(data_dir);
    if db_path.exists() {
        return Err(VecdError::usage(format!(
            "a vecd database already exists at {} — refusing to re-init",
            db_path.display()
        )));
    }
    let mut db = Db::init(&db_path)?;
    admin::add_user(&mut db, superuser, Level::Superuser, None, None)?;
    let tok = admin::create_token(&mut db, superuser, "vecd init superuser key", None, None)?;

    // Auto-login: store the superuser token in the credential store, keyed by
    // the local endpoint the configured bind implies — so the admin can use
    // the client commands (`vecd whoami`, pushes, …) the moment `vecd start`
    // is up, with no separate `vecd login`. The token is irrecoverable, so we
    // point the user at the file that now holds it.
    let endpoint = local_endpoint_url(cfg);
    let origin = vectordata::credentials::origin_of_str(&endpoint).unwrap_or_else(|| endpoint.clone());
    let mut store = crate::credentials::Store::load(&resolved.dir);
    store.set(
        origin.clone(),
        crate::credentials::Entry {
            token: tok.plaintext.clone(),
            user: Some(superuser.to_string()),
            expires: Some(tok.expires_at.to_string()),
        },
    );
    store.save(&resolved.dir)?;
    let cred_path = resolved.dir.join("credentials.json");

    if quiet {
        println!("{}", tok.plaintext);
        eprintln!("superuser token stored in {}", cred_path.display());
        return Ok(());
    }
    println!("initialized vecd at {}", data_dir.display());
    println!("superuser:   {superuser}");
    println!("token:       {}", tok.plaintext);
    println!("expires:     {}", admin::fmt_epoch(tok.expires_at));
    println!("logged in:   {origin}  (used automatically by vecd/vectordata)");
    println!("credentials: {}", cred_path.display());
    println!("\nThat credentials file now holds the superuser token — the");
    println!("irrecoverable admin key. Keep it safe (it is created mode 0600).");
    Ok(())
}

/// The local client URL a freshly-`init`ed admin should use, derived from the
/// configured bind: scheme from any TLS config, host from `bind` (wildcard
/// addresses map to loopback so the URL is usable), and the configured port.
fn local_endpoint_url(cfg: &config::Config) -> String {
    let scheme = if cfg.get("tls_cert").is_some() && cfg.get("tls_key").is_some() {
        "https"
    } else {
        "http"
    };
    let bind = resolve_bind(&None, cfg);
    match bind.parse::<SocketAddr>() {
        Ok(addr) => {
            let host = if addr.ip().is_unspecified() {
                if addr.is_ipv6() { "[::1]".to_string() } else { "127.0.0.1".to_string() }
            } else if addr.is_ipv6() {
                format!("[{}]", addr.ip())
            } else {
                addr.ip().to_string()
            };
            format!("{scheme}://{host}:{}/", addr.port())
        }
        Err(_) => format!("{scheme}://{bind}/"),
    }
}

/// `vecd config …` — inspect and edit `vecd.conf` at the resolved location.
fn cmd_config(resolved: &config::Resolved, cmd: ConfigCmd) -> Result<(), VecdError> {
    match cmd {
        ConfigCmd::Auto { yes, force } => cmd_config_auto(resolved, yes, force),
        ConfigCmd::Get { param, out, format } => cmd_config_get(resolved, param, out, format),
        ConfigCmd::Set { param, value, from, force } => {
            cmd_config_set(resolved, param, value, from, force)
        }
    }
}

/// `vecd config get` — read the whole config or a single value, to stdout or a
/// file/dir in the chosen format. Stdout output is clean (no decoration) so it
/// round-trips: `vecd config get | vecd config set --from -`.
fn cmd_config_get(
    resolved: &config::Resolved,
    param: Option<String>,
    out: Option<PathBuf>,
    format: Option<String>,
) -> Result<(), VecdError> {
    let cfg = config::Config::load(&resolved.dir)?;

    // Single key → just its value.
    if let Some(key) = param {
        if out.is_some() || format.is_some() {
            return Err(VecdError::usage(
                "--out/--format apply to the whole config, not a single key".to_string(),
            ));
        }
        let key = key.replace('-', "_");
        return match cfg.get(&key) {
            Some(v) => {
                println!("{v}");
                Ok(())
            }
            None => Err(VecdError::usage(format!("config '{key}' is not set"))),
        };
    }

    // Whole config → a file/dir (by extension), or stdout in `format`.
    if let Some(out) = out {
        if format.is_some() {
            return Err(VecdError::usage(
                "pass --out (file/dir) or --format (stdout), not both".to_string(),
            ));
        }
        config::export_to(&cfg, &out)?;
        if out.as_os_str() != "-" {
            eprintln!("wrote {}", out.display());
        }
        return Ok(());
    }
    let text = match format.as_deref() {
        None | Some("native") => cfg.to_text(),
        Some("json") => cfg.to_json(),
        Some("yaml") | Some("yml") => cfg.to_yaml(),
        Some(other) => {
            return Err(VecdError::usage(format!(
                "unknown --format '{other}' (expected native|json|yaml)"
            )));
        }
    };
    print!("{text}");
    Ok(())
}

/// `vecd config set` — set one value, or replace the whole config from a
/// source. Honors the lock and the `--force` change guard.
fn cmd_config_set(
    resolved: &config::Resolved,
    param: Option<String>,
    value: Option<String>,
    from: Option<PathBuf>,
    force: bool,
) -> Result<(), VecdError> {
    match (param, value, from) {
        // Set one key.
        (Some(param), Some(value), None) => {
            let key = param.replace('-', "_"); // accept lock-config / ratelimit-… too
            let mut cfg = config::Config::load(&resolved.dir)?;
            // The one mutation allowed on a locked config is unlocking it.
            let unlocking = key == "lock_config" && config::as_bool(&value) == Some(false);
            guard_locked(&cfg, unlocking)?;
            // Changing an already-set value (to something different) needs --force.
            if let Some(cur) = cfg.get(&key)
                && cur != value
                && !force
            {
                return Err(VecdError::usage(format!(
                    "config '{key}' is already set to '{cur}' — pass --force to change it to '{value}'."
                )));
            }
            cfg.set(&key, &value)?;
            let path = cfg.write_to(&resolved.dir)?;
            println!("set {key} = {value}  → {}", path.display());
            Ok(())
        }
        // Replace the whole config from a source.
        (None, None, Some(from)) => {
            let existing = config::Config::load(&resolved.dir)?;
            guard_replace(&existing, resolved.exists, force)?;
            let cfg = if from.as_os_str() == "-" {
                let cfg = config::Config::parse(&read_file_or_stdin(None)?)?;
                cfg.validate_all()?;
                cfg
            } else {
                config::import_from(&from)?
            };
            let path = cfg.write_to(&resolved.dir)?;
            println!("replaced config from {} → {}", from.display(), path.display());
            Ok(())
        }
        _ => Err(VecdError::usage(
            "usage: `vecd config set <key> <value>` to set one value, or \
             `vecd config set --from <json|yaml|conf|dir|->` to replace the whole config"
                .to_string(),
        )),
    }
}

/// Refuse a config change when `lock_config` is on — unless the change is the
/// one act of unlocking it.
fn guard_locked(existing: &config::Config, unlocking: bool) -> Result<(), VecdError> {
    if existing.is_locked() && !unlocking {
        return Err(VecdError::usage(
            "config is locked (lock_config=on) — run `vecd config set lock_config off --force` \
             to allow changes."
                .to_string(),
        ));
    }
    Ok(())
}

/// Guard a whole-config replacement (`auto`/`load`/`import`): refuse on a
/// locked config, and require `--force` to overwrite an existing one.
fn guard_replace(existing: &config::Config, exists: bool, force: bool) -> Result<(), VecdError> {
    guard_locked(existing, false)?;
    if exists && !force {
        return Err(VecdError::usage(
            "a config already exists here — pass --force to replace it (`vecd config get` to view it)."
                .to_string(),
        ));
    }
    Ok(())
}

/// `vecd config auto` — propose safe defaults, confirm (unless `--yes`), write.
fn cmd_config_auto(resolved: &config::Resolved, yes: bool, force: bool) -> Result<(), VecdError> {
    let existing = config::Config::load(&resolved.dir)?;
    guard_replace(&existing, resolved.exists, force)?;
    let cfg = config::auto_defaults(&resolved.dir);
    let path = resolved.conf_path();
    println!("Proposed config for {} ({}):\n", path.display(), resolved.source.label());
    print!("{}", cfg.to_text());
    println!();
    if resolved.exists {
        println!("(a config already exists here and will be overwritten)");
    }
    if !yes && !confirm(&format!("Write this to {}?", path.display()))? {
        println!("aborted — no changes written.");
        return Ok(());
    }
    let written = cfg.write_to(&resolved.dir)?;
    println!("wrote {written}", written = written.display());
    println!("next: `vecd init` to create the DB and mint a superuser token.");
    Ok(())
}

/// Prompt for a yes/no on stdin; defaults to no.
fn confirm(prompt: &str) -> Result<bool, VecdError> {
    use std::io::Write;
    print!("{prompt} [y/N] ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).map_err(VecdError::Io)?;
    Ok(matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes"))
}

/// Read native config text from a file, or stdin when the path is absent or `-`.
fn read_file_or_stdin(file: Option<&std::path::Path>) -> Result<String, VecdError> {
    match file {
        Some(p) if p.as_os_str() != "-" => std::fs::read_to_string(p)
            .map_err(|e| VecdError::usage(format!("reading {}: {e}", p.display()))),
        _ => {
            use std::io::Read;
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).map_err(VecdError::Io)?;
            Ok(s)
        }
    }
}

/// `vecd login <url>` — authenticate to a vecd endpoint and store its token in
/// the config dir's `credentials.json`, keyed by origin. Either `--token`
/// (store a token directly) or `--user` + `--password` (a password grant via
/// the endpoint's `/auth/token`).
#[allow(clippy::too_many_arguments)]
fn cmd_login(
    resolved: &config::Resolved,
    url: Option<String>,
    token: Option<String>,
    user: Option<String>,
    password: Option<String>,
    expires: Option<&str>,
    list: bool,
) -> Result<(), VecdError> {
    // `--list`: print every endpoint with a stored credential and exit.
    if list {
        let store = crate::credentials::Store::load(&resolved.dir);
        let entries = store.list();
        if entries.is_empty() {
            println!("no stored credentials ({})", resolved.dir.join("credentials.json").display());
        } else {
            for (origin, e) in entries {
                match &e.user {
                    Some(u) => println!("{origin}  (user: {u})"),
                    None => println!("{origin}"),
                }
            }
        }
        return Ok(());
    }

    let url = url.ok_or_else(|| VecdError::usage("login needs a <url> (or --list)"))?;
    let origin = vectordata::credentials::origin_of_str(&url)
        .ok_or_else(|| VecdError::usage(format!("not a valid endpoint URL: {url}")))?;
    let (token, user, expires_at) = match token {
        Some(t) => {
            // --token is a literal token or a file (JSON record / credential
            // store / bare token); a store is keyed by origin, so pass the url.
            let r = vectordata::credentials::resolve_token_arg(&t, Some(url.as_str()))
                .map_err(VecdError::op)?;
            // The token's user: from the JSON record, else --user, else ask the
            // endpoint, so the stored credential is labeled with its user.
            let user = r.user.or(user).or_else(|| whoami_user(&url, &r.token));
            (r.token, user, r.expires)
        }
        None => {
            let user = user.ok_or_else(|| {
                VecdError::usage("login needs --token, or --user with --password")
            })?;
            let password = password
                .or_else(|| std::env::var("VECD_PASSWORD").ok())
                .ok_or_else(|| {
                    VecdError::usage("login with --user needs --password or $VECD_PASSWORD")
                })?;
            let resp =
                vectordata::endpoint::login_password(&url, &user, &password, Some("vecd login"), expires)
                    .map_err(VecdError::op)?;
            let exp = resp.expires_at.map(|e| e.to_string());
            (resp.token, resp.user.or(Some(user)), exp)
        }
    };
    let user_note = user.as_deref().map(|u| format!(" as {u}")).unwrap_or_default();
    let mut store = crate::credentials::Store::load(&resolved.dir);
    store.set(origin.clone(), crate::credentials::Entry { token, user, expires: expires_at });
    store.save(&resolved.dir)?;
    println!(
        "logged in to {origin}{user_note} — token stored at {}",
        resolved.dir.join("credentials.json").display()
    );
    Ok(())
}

/// Best-effort: ask the endpoint which user a token authenticates as, so a
/// stored credential can be labeled. `None` on any error or anonymous access.
fn whoami_user(url: &str, token: &str) -> Option<String> {
    let v = vectordata::endpoint::whoami(url, Some(token)).ok()?;
    if v.get("authenticated").and_then(|a| a.as_bool()) == Some(true) {
        v.get("identity").and_then(|i| i.as_str()).map(String::from)
    } else {
        None
    }
}

/// Resolve the endpoint a command acts on: the given `url`, else the sole
/// stored credential's endpoint — so commands "just work" while logged in.
/// Errors (asking for an explicit `<url>`) when none or several are stored.
fn resolve_endpoint(resolved: &config::Resolved, url: Option<String>) -> Result<String, VecdError> {
    if let Some(u) = url {
        return Ok(u);
    }
    let store = crate::credentials::Store::load(&resolved.dir);
    let entries = store.list();
    if entries.is_empty() {
        return Err(VecdError::usage(
            "not logged in to any endpoint — give a <url>, or run `vecd login <url>` first".to_string(),
        ));
    }
    if entries.len() > 1 {
        let list = entries.iter().map(|(o, _)| o.as_str()).collect::<Vec<_>>().join(", ");
        return Err(VecdError::usage(format!(
            "logged in to several endpoints — specify one as <url>: {list}"
        )));
    }
    Ok(entries[0].0.clone())
}

/// `vecd logout [url]` — forget the stored token for an endpoint's origin.
fn cmd_logout(resolved: &config::Resolved, url: &str) -> Result<(), VecdError> {
    let origin = vectordata::credentials::origin_of_str(url)
        .ok_or_else(|| VecdError::usage(format!("not a valid endpoint URL: {url}")))?;
    let mut store = crate::credentials::Store::load(&resolved.dir);
    if store.remove(&origin) {
        store.save(&resolved.dir)?;
        println!("logged out of {origin}");
    } else {
        println!("no stored credential for {origin}");
    }
    Ok(())
}

/// `vecd whoami <url>` — show effective access at an endpoint, using the
/// stored token automatically.
fn cmd_client_whoami(resolved: &config::Resolved, url: &str) -> Result<(), VecdError> {
    let origin = vectordata::credentials::origin_of_str(url)
        .ok_or_else(|| VecdError::usage(format!("not a valid endpoint URL: {url}")))?;
    let token = crate::credentials::stored_token(&resolved.dir, url);
    let view = vectordata::endpoint::whoami(url, token.as_deref()).map_err(|e| {
        if e == "not-a-vecd" {
            VecdError::op(format!("{origin} is not a vecd endpoint"))
        } else if token.is_none() {
            VecdError::op(format!("{e} (no stored credential — try `vecd login {url}`)"))
        } else {
            VecdError::op(e)
        }
    })?;
    println!("endpoint:  {origin}");
    let identity = view.get("identity").and_then(|v| v.as_str()).unwrap_or("(anonymous)");
    let level = view.get("level").and_then(|v| v.as_str()).unwrap_or("-");
    let how = if token.is_some() { "stored token" } else { "anonymous (no stored token)" };
    println!("identity:  {identity:<20} level: {level}   [{how}]");
    if let Some(nss) = view.get("namespaces").and_then(|v| v.as_array())
        && !nss.is_empty()
    {
        println!("namespaces:");
        for ns in nss {
            let path = ns.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            let actions: Vec<&str> = ns
                .get("actions")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
                .unwrap_or_default();
            println!("  {path:<28} {}", actions.join(","));
        }
    }
    if let Some(hidden) = view.get("hidden").and_then(|v| v.as_u64())
        && hidden > 0
    {
        println!("hidden from you: {hidden} namespace(s)");
    }
    Ok(())
}

fn cmd_serve(data_dir: &std::path::Path, cfg: &config::Config, a: &ServeArgs) -> Result<(), VecdError> {
    let bind = resolve_bind(&a.bind, cfg);
    let addr: SocketAddr = bind
        .parse()
        .map_err(|e| VecdError::usage(format!("bad bind address '{bind}': {e}")))?;
    // TLS paths mirror the other settings: a flag wins, else the vecd.conf
    // key (`tls_cert` / `tls_key`) — so a persistent secure deploy configures
    // TLS once in the file instead of on every `restart`.
    let tls_cert = a.tls_cert.clone().or_else(|| cfg.get("tls_cert").map(PathBuf::from));
    let tls_key = a.tls_key.clone().or_else(|| cfg.get("tls_key").map(PathBuf::from));
    let tls = match (tls_cert, tls_key) {
        (Some(cert), Some(key)) => Some(server::TlsConfig { cert, key }),
        (None, None) => None,
        _ => return Err(VecdError::usage(
            "TLS needs both cert and key — give --tls-cert and --tls-key together, \
             or set both tls_cert and tls_key in vecd.conf",
        )),
    };
    if exposes_plaintext(&addr, tls.is_some()) {
        let banner = format!(
            "WARNING: vecd is binding {addr} (a non-loopback address) WITHOUT TLS.\n\
             Traffic — including bearer tokens — will cross the network in cleartext.\n\
             Use --tls-cert/--tls-key (or set tls_cert/tls_key in vecd.conf) for direct\n\
             exposure, or terminate TLS at a reverse proxy in front of vecd. Bind to\n\
             127.0.0.1 to keep vecd local-only."
        );
        log::warn!("{banner}");
        eprintln!("\n\x1b[1;33m{banner}\x1b[0m\n");
    }
    let db = open_db(data_dir, cfg)?;
    let limits = resolve_rate_limits(a, cfg)?;
    if !limits.is_unlimited() {
        log::info!(
            "vecd: rate limits (bytes/sec, 0=off) — per-connection down={} up={}, per-client down={} up={}",
            limits.connection_download, limits.connection_upload,
            limits.client_download, limits.client_upload,
        );
    }
    let state = server::AppState::new(db)?.with_rate_limits(limits);

    let backup_dest = a.db_backup.clone().or_else(|| cfg.db_backup().map(|s| s.to_string()));
    let interval_secs = admin::parse_duration(&a.backup_interval)?;
    let backup_retain = a.backup_retain;
    let addr_file = data_dir.join("vecd.addr");

    let rt = tokio::runtime::Runtime::new().map_err(VecdError::Io)?;
    rt.block_on(async move {
        if let Some(dest) = backup_dest {
            spawn_db_backup(state.clone(), dest, interval_secs as u64, backup_retain);
        }
        match tls {
            // TLS path: axum-server owns the bind; no ephemeral-port file.
            Some(tls) => server::serve(state, addr, Some(tls))
                .await
                .map_err(|e| VecdError::op(e.to_string())),
            // Plain HTTP: bind here so the actual address (incl. an
            // ephemeral `:0` port) is published to `<data_dir>/vecd.addr`,
            // and shut down gracefully on SIGINT/SIGTERM.
            None => {
                let listener = tokio::net::TcpListener::bind(addr).await.map_err(VecdError::Io)?;
                if let Ok(actual) = listener.local_addr() {
                    let _ = std::fs::write(&addr_file, actual.to_string());
                    log::info!("vecd: serving HTTP on {actual}");
                }
                let r = server::serve_listener(state, listener, shutdown_signal())
                    .await
                    .map_err(|e| VecdError::op(e.to_string()));
                let _ = std::fs::remove_file(&addr_file);
                r
            }
        }
    })
}

/// Resolve on the first of SIGINT (Ctrl-C) or SIGTERM (`vecd stop`).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Periodic control-plane DB backup task.
fn spawn_db_backup(state: server::AppState, dest: String, interval_secs: u64, retain: usize) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(interval_secs.max(60)));
        loop {
            tick.tick().await;
            let dest = dest.clone();
            let db = state.db.clone();
            let r = tokio::task::spawn_blocking(move || {
                let db = db.lock().unwrap();
                backup::backup_now(&db, &dest, Some(retain))
            })
            .await;
            match r {
                Ok(Ok(uri)) => log::info!("vecd: backed up control-plane DB → {uri}"),
                Ok(Err(e)) => log::warn!("vecd: DB backup failed: {e}"),
                Err(e) => log::warn!("vecd: DB backup task panicked: {e}"),
            }
        }
    });
}

fn cmd_users(data_dir: &std::path::Path, cfg: &config::Config, c: UsersCmd) -> Result<(), VecdError> {
    let mut db = open_db(data_dir, cfg)?;
    match c {
        UsersCmd::Add { name, level, password, home_backend } => {
            admin::add_user(&mut db, &name, level.parse()?, password.as_deref(), home_backend.as_deref())?;
            println!("added user {name}");
        }
        UsersCmd::List => {
            for (name, level, disabled) in admin::list_users(&db)? {
                let flag = if disabled { " (disabled)" } else { "" };
                println!("{name:<24} {level}{flag}");
            }
        }
        UsersCmd::Level { name, level } => {
            admin::set_level(&mut db, &name, level.parse()?)?;
            println!("{name} → {level}");
        }
        UsersCmd::Passwd { name, password } => {
            admin::set_password(&mut db, &name, &password)?;
            println!("password set for {name}");
        }
        UsersCmd::Disable { name } => {
            admin::set_disabled(&mut db, &name, true)?;
            println!("disabled {name}");
        }
        UsersCmd::Enable { name } => {
            admin::set_disabled(&mut db, &name, false)?;
            println!("enabled {name}");
        }
        UsersCmd::Remove { name } => {
            admin::remove_user(&mut db, &name)?;
            println!("removed {name}");
        }
    }
    Ok(())
}

fn cmd_tokens(data_dir: &std::path::Path, cfg: &config::Config, c: TokensCmd) -> Result<(), VecdError> {
    let mut db = open_db(data_dir, cfg)?;
    match c {
        TokensCmd::Create { user, description, expires, profile, from, sets, quiet, json } => {
            let profile_json = match (profile, from) {
                (Some(_), Some(_)) => {
                    return Err(VecdError::usage("give either --profile or --from, not both"))
                }
                (Some(spec), None) => Some(admin::parse_profile_spec(&spec)?),
                (None, Some(name)) => {
                    let kv = parse_sets(&sets)?;
                    Some(admin::expand_profile(&db, &name, &kv)?)
                }
                (None, None) => None,
            };
            let tok = admin::create_token(&mut db, &user, &description, expires.as_deref(), profile_json)?;
            if json {
                // A machine-readable token record — feed it to `--token <file>`
                // / `login --token <file>` to use the token as its user.
                let record = serde_json::json!({
                    "token": tok.plaintext,
                    "user": tok.user,
                    "id": tok.id,
                    "expires_at": tok.expires_at,
                    "description": description,
                });
                println!("{}", serde_json::to_string_pretty(&record).map_err(|e| VecdError::op(e.to_string()))?);
            } else if quiet {
                println!("{}", tok.plaintext);
            } else {
                println!("token:   {}", tok.plaintext);
                println!("user:    {}", tok.user);
                println!("id:      {}", tok.id);
                println!("expires: {}", admin::fmt_epoch(tok.expires_at));
                println!("\nStore this token now — it is not recoverable.");
            }
        }
        TokensCmd::List { user } => {
            for (id, u, desc, exp) in admin::list_tokens(&db, user.as_deref())? {
                println!("{id:<6} {u:<16} expires {}  {desc}", admin::fmt_epoch(exp));
            }
        }
        TokensCmd::Revoke { id } => {
            admin::revoke_token(&mut db, id)?;
            println!("revoked token {id}");
        }
    }
    Ok(())
}

fn cmd_roles(data_dir: &std::path::Path, cfg: &config::Config, c: RolesCmd) -> Result<(), VecdError> {
    let mut db = open_db(data_dir, cfg)?;
    match c {
        RolesCmd::List => {
            for (name, actions, builtin) in admin::list_roles(&db)? {
                let tag = if builtin { " (built-in)" } else { "" };
                println!("{name:<16} {actions}{tag}");
            }
        }
        RolesCmd::Add { name, actions } => {
            admin::add_role(&mut db, &name, &actions)?;
            println!("added role {name}");
        }
        RolesCmd::Remove { name } => {
            admin::remove_role(&mut db, &name)?;
            println!("removed role {name}");
        }
    }
    Ok(())
}

fn cmd_backends(data_dir: &std::path::Path, cfg: &config::Config, c: BackendsCmd) -> Result<(), VecdError> {
    let mut db = open_db(data_dir, cfg)?;
    match c {
        BackendsCmd::Add { name, kind, endpoint, endpoint_url, region, aws_profile, active } => {
            admin::add_backend(
                &mut db, &name, &kind, &endpoint, endpoint_url.as_deref(), region.as_deref(),
                aws_profile.as_deref(), active,
            )?;
            println!("added backend {name}");
        }
        BackendsCmd::List => {
            for (name, kind, endpoint, active) in admin::list_backends(&db)? {
                let tag = if active { "active" } else { "standby" };
                println!("{name:<16} {kind:<6} {endpoint:<40} {tag}");
            }
        }
        BackendsCmd::Set { name, active, no_active } => {
            let want = resolve_toggle(active, no_active, "backend")?;
            admin::set_backend_active(&mut db, &name, want)?;
            println!("backend {name} → {}", if want { "active" } else { "standby" });
        }
        BackendsCmd::Remove { name } => {
            admin::remove_backend(&mut db, &name)?;
            println!("removed backend {name}");
        }
    }
    Ok(())
}

fn cmd_ns(data_dir: &std::path::Path, cfg: &config::Config, c: NsCmd) -> Result<(), VecdError> {
    let mut db = open_db(data_dir, cfg)?;
    match c {
        NsCmd::Add { path, owner, backend_config, active, listable, ttl, quota } => {
            let l: Listable = listable.parse()?;
            admin::add_namespace(
                &mut db, &path, &owner, backend_config.as_deref(), active, l, ttl.as_deref(),
                quota.as_deref(),
            )?;
            println!("added namespace {path}");
        }
        NsCmd::List => {
            for (path, owner, backend, active, listable) in admin::list_namespaces(&db)? {
                let p = if path.is_empty() { "(root)".to_string() } else { path };
                let backend = backend.unwrap_or_else(|| "-".to_string());
                let tag = if active { "active" } else { "config-only" };
                println!("{p:<28} owner={owner:<14} backend={backend:<12} {tag:<12} listable={listable}");
            }
        }
        NsCmd::Set { path, owner, backend_config, active, no_active, listable, ttl, quota } => {
            let active_opt = toggle_opt(active, no_active, "namespace")?;
            let l = listable.map(|s| s.parse()).transpose()?;
            admin::set_namespace(
                &mut db,
                &path,
                owner.as_deref(),
                backend_config.as_ref().map(|s| Some(s.as_str())),
                active_opt,
                l,
                ttl.as_ref().map(|s| Some(s.as_str())),
                quota.as_deref(),
            )?;
            println!("updated namespace {path}");
        }
        NsCmd::Remove { path } => {
            admin::remove_namespace(&mut db, &path)?;
            println!("removed namespace {path}");
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_grant(
    data_dir: &std::path::Path,
    cfg: &config::Config,
    to: &str,
    ns: &str,
    read: bool,
    write: bool,
    delete: bool,
    adm: bool,
) -> Result<(), VecdError> {
    use crate::model::{Action, ActionSet, Class};
    let mut set = ActionSet::EMPTY;
    if read {
        set = set.union(ActionSet::of(&[Action::Read]));
    }
    if write {
        set = set.union(ActionSet::of(&[Action::Read, Action::Write]));
    }
    if delete {
        set = set.union(ActionSet::of(&[Action::Read, Action::Write, Action::Delete]));
    }
    if adm {
        set = ActionSet::ALL;
    }
    // Map the flag set to the smallest covering class shorthand.
    let class = [Class::Read, Class::Publish, Class::Maintain, Class::Curate]
        .into_iter()
        .find(|c| c.actions() == set)
        .ok_or_else(|| {
            VecdError::usage(
                "no action flag given, or a non-nesting combination; use \
                 `vecd bind --role <role>` for an ad-hoc role",
            )
        })?;
    let mut db = open_db(data_dir, cfg)?;
    admin::bind(&mut db, to, class.name(), ns)?;
    println!("granted {} → {to} on {ns}", class.name());
    Ok(())
}

fn cmd_priv(data_dir: &std::path::Path, cfg: &config::Config, c: PrivCmd) -> Result<(), VecdError> {
    let mut db = open_db(data_dir, cfg)?;
    match c {
        PrivCmd::Grant { privilege, to } => {
            admin::grant_privilege(&mut db, &to, &privilege)?;
            println!("granted {privilege} → {to}");
        }
        PrivCmd::Revoke { privilege, to } => {
            admin::revoke_privilege(&mut db, &to, &privilege)?;
            println!("revoked {privilege} from {to}");
        }
    }
    Ok(())
}

fn cmd_profiles(data_dir: &std::path::Path, cfg: &config::Config, c: ProfilesCmd) -> Result<(), VecdError> {
    let mut db = open_db(data_dir, cfg)?;
    match c {
        ProfilesCmd::Add { name, owner, spec } => {
            admin::add_profile(&mut db, &name, &owner, &spec)?;
            println!("added profile {name}");
        }
        ProfilesCmd::List => {
            for (name, owner, spec) in admin::list_profiles(&db)? {
                println!("{name:<20} owner={owner:<14} {spec}");
            }
        }
        ProfilesCmd::Remove { name } => {
            admin::remove_profile(&mut db, &name)?;
            println!("removed profile {name}");
        }
    }
    Ok(())
}

fn cmd_backup(data_dir: &std::path::Path, cfg: &config::Config, c: BackupCmd) -> Result<(), VecdError> {
    let resolve_dest = |dest: Option<String>| -> Result<String, VecdError> {
        dest.or_else(|| cfg.db_backup().map(|s| s.to_string())).ok_or_else(|| {
            VecdError::usage("no backup destination (pass one, or set db_backup in vecd.conf)")
        })
    };
    match c {
        BackupCmd::Now { dest } => {
            let dest = resolve_dest(dest)?;
            let db = open_db(data_dir, cfg)?;
            let uri = backup::backup_now(&db, &dest, None)?;
            println!("snapshot → {uri}");
        }
        BackupCmd::List { dest } => {
            let dest = resolve_dest(dest)?;
            for name in backup::list_backups(&dest)? {
                println!("{name}");
            }
        }
    }
    Ok(())
}

fn cmd_cleanup(data_dir: &std::path::Path, cfg: &config::Config, c: CleanupCmd) -> Result<(), VecdError> {
    let mut db = open_db(data_dir, cfg)?;
    match c {
        CleanupCmd::List => {
            let items = crate::lifetime::pending(&db)?;
            if items.is_empty() {
                println!("cleanup queue is empty");
            }
            for it in items {
                let when = it.stasis_at.map(admin::fmt_epoch).unwrap_or_else(|| "-".into());
                println!("{}@{:<10} stasis since {when}  ({})", it.namespace_path, it.tag, &it.manifest_hash[..16.min(it.manifest_hash.len())]);
            }
        }
        CleanupCmd::Extend { target, duration } => {
            let (ns, sel) = split_target(&target)?;
            let dur = duration.as_deref().map(admin::parse_duration).transpose()?;
            crate::lifetime::extend(&mut db, &ns, &sel, dur)?;
            println!("restored {ns}@{sel}");
        }
        CleanupCmd::Purge { target } => {
            let (ns, sel) = split_target(&target)?;
            let backend = open_backend_for_ns(&db, &ns)?;
            let reclaimed = crate::lifetime::purge(&mut db, backend.as_ref(), &ns, &sel)?;
            println!("purged {ns}@{sel} ({reclaimed} blob(s) reclaimed)");
        }
    }
    Ok(())
}

/// Parse a `<ns>@<selector>` cleanup target.
fn split_target(target: &str) -> Result<(String, String), VecdError> {
    target
        .rsplit_once('@')
        .map(|(ns, sel)| (crate::authz::normalize(ns), sel.to_string()))
        .filter(|(ns, sel)| !ns.is_empty() && !sel.is_empty())
        .ok_or_else(|| VecdError::usage(format!("target must be '<namespace>@<version>', got '{target}'")))
}

/// Open the storage backend serving a namespace (for `cleanup purge`).
fn open_backend_for_ns(
    db: &Db,
    ns: &str,
) -> Result<std::sync::Arc<dyn crate::backend::Backend>, VecdError> {
    let snap = crate::authz::Snapshot::build(&db.load_control_plane()?);
    let (storage_ns, _) = crate::namespace::resolve_for_list(&snap, ns)?;
    let n = snap
        .namespace(&storage_ns)
        .ok_or_else(|| VecdError::usage(format!("no such namespace '{ns}'")))?;
    let bc = n
        .backend_config
        .as_ref()
        .ok_or_else(|| VecdError::usage(format!("namespace '{ns}' has no backend")))?;
    let row = snap
        .backend(bc)
        .ok_or_else(|| VecdError::usage(format!("backend config '{bc}' missing")))?;
    crate::backend::open(row)
}

/// `vecd objects <ns>` — the live objects under a namespace subtree (what a
/// plain `push` writes), with sizes and a total. This is the "what's actually
/// published here" view, distinct from session `versions`.
fn cmd_objects(data_dir: &std::path::Path, cfg: &config::Config, ns: &str) -> Result<(), VecdError> {
    let db = open_db(data_dir, cfg)?;
    let mut stmt = db.conn().prepare(
        "SELECT key, size, created FROM objects \
         WHERE namespace_path=?1 OR namespace_path LIKE ?1||'/%' ORDER BY key",
    )?;
    let rows = stmt
        .query_map([ns], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<i64>>(2)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    if rows.is_empty() {
        println!("no objects under '{ns}'");
        return Ok(());
    }
    println!("{:>12}  {:<20}  key", "bytes", "created");
    let mut total: i64 = 0;
    for (key, size, created) in &rows {
        total += *size;
        println!(
            "{:>12}  {:<20}  {}",
            size,
            created.map(admin::fmt_epoch).unwrap_or_else(|| "-".into()),
            key,
        );
    }
    println!("{} object(s), {total} bytes total", rows.len());
    Ok(())
}

/// `vecd versions <ns>` — the session-published versions of a namespace,
/// newest first.
fn cmd_versions(data_dir: &std::path::Path, cfg: &config::Config, ns: &str) -> Result<(), VecdError> {
    let db = open_db(data_dir, cfg)?;
    let rows = crate::session::list_versions(&db, ns)?;
    if rows.is_empty() {
        println!("no session versions for '{ns}' (plain `push` writes show under `vecd objects {ns}`)");
        return Ok(());
    }
    println!("{:>4}  {:<10} {:<10} {:<20} manifest", "seq", "state", "tag", "committed");
    for v in rows {
        let manifest = &v.manifest_hash[..v.manifest_hash.len().min(16)];
        println!(
            "{:>4}  {:<10} {:<10} {:<20} {}",
            v.seq,
            v.state,
            if v.tag.is_empty() { "-".to_string() } else { v.tag },
            v.committed_at.map(admin::fmt_epoch).unwrap_or_else(|| "-".into()),
            manifest,
        );
    }
    Ok(())
}

/// `vecd bindings [--ns <prefix>]` — every role binding, optionally filtered to
/// a namespace prefix. Answers "who can do what, where" from the admin CLI
/// instead of needing the introspection endpoint or raw DB access.
fn cmd_bindings(
    data_dir: &std::path::Path,
    cfg: &config::Config,
    ns_prefix: Option<&str>,
) -> Result<(), VecdError> {
    let db = open_db(data_dir, cfg)?;
    let rows: Vec<_> = admin::list_bindings(&db)?
        .into_iter()
        .filter(|(_, _, nsp)| ns_prefix.is_none_or(|p| nsp.starts_with(p)))
        .collect();
    if rows.is_empty() {
        match ns_prefix {
            Some(p) => println!("no bindings under '{p}'"),
            None => println!("no bindings"),
        }
        return Ok(());
    }
    println!("{:<20} {:<12} namespace", "principal", "role");
    for (principal, role, nsp) in rows {
        println!("{principal:<20} {role:<12} {nsp}");
    }
    Ok(())
}

fn cmd_log(
    data_dir: &std::path::Path,
    cfg: &config::Config,
    tail: usize,
    principal: Option<&str>,
    status: Option<i64>,
    action: Option<&str>,
) -> Result<(), VecdError> {
    let db = open_db(data_dir, cfg)?;
    // Build the optional filter clause, keeping each predicate's positional
    // placeholder (`?N`) in lockstep with its pushed param.
    let mut clauses: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if let Some(p) = principal {
        params.push(Box::new(p.to_string()));
        clauses.push(format!("principal=?{}", params.len()));
    }
    if let Some(s) = status {
        params.push(Box::new(s));
        clauses.push(format!("status=?{}", params.len()));
    }
    if let Some(a) = action {
        params.push(Box::new(a.to_string()));
        clauses.push(format!("action=?{}", params.len()));
    }
    let where_sql = if clauses.is_empty() { String::new() } else { format!("WHERE {}", clauses.join(" AND ")) };
    params.push(Box::new(tail as i64));
    let limit_idx = params.len();
    let sql = format!(
        "SELECT ts,principal,action,key,status,bytes,remote_addr FROM access_log \
         {where_sql} ORDER BY id DESC LIMIT ?{limit_idx}"
    );

    let mut stmt = db.conn().prepare(&sql)?;
    let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();
    let rows = stmt
        .query_map(param_refs.as_slice(), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<i64>>(4)?,
                r.get::<_, Option<i64>>(5)?,
                r.get::<_, Option<String>>(6)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    for (ts, principal, action, key, status, bytes, addr) in rows.into_iter().rev() {
        println!(
            "{}  {:<14} {:<6} {:>3} {:>10}  {}  {}",
            admin::fmt_epoch(ts),
            principal.unwrap_or_else(|| "-".into()),
            action.unwrap_or_else(|| "-".into()),
            status.unwrap_or(0),
            bytes.unwrap_or(0),
            addr.unwrap_or_else(|| "-".into()),
            key.unwrap_or_default(),
        );
    }
    Ok(())
}

/// Emit a *dynamic* completion activation snippet (mirrors `vectordata`): one
/// sourceable line that hands control back to the `CompleteEnv` wired in
/// `main()` by re-invoking the binary with `COMPLETE=<shell>`. No frozen
/// script — subcommands added later show up automatically. With no `--shell`,
/// detect from `$SHELL`.
fn cmd_completions(shell: Option<String>) {
    let argv0 = std::env::args_os()
        .next()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "vecd".to_string());
    let name = match shell.or_else(detect_shell_from_env) {
        Some(s) => s,
        None => {
            eprintln!("Could not auto-detect your shell from $SHELL.");
            eprintln!("Pass --shell explicitly: bash | zsh | fish | elvish | powershell");
            std::process::exit(1);
        }
    };
    println!("# vecd tab-completion for {name} (dynamic — defers to the binary)");
    println!("# To activate now:  eval \"$(vecd completions)\"");
    println!("# To persist:       add the activation line to your shell rc file");
    match name.as_str() {
        "fish" => println!("COMPLETE=fish \"{argv0}\" | source"),
        "elvish" => println!("eval (COMPLETE=elvish \"{argv0}\" | slurp)"),
        "powershell" => {
            println!(r#"(& {{ $env:COMPLETE="powershell"; "{argv0}" }}) | Invoke-Expression"#)
        }
        _ /* Bash / Zsh */ => println!("source <(COMPLETE={name} \"{argv0}\")"),
    }
}

/// Best-effort shell name from `$SHELL` (the basename), or `None`.
fn detect_shell_from_env() -> Option<String> {
    let sh = std::env::var("SHELL").ok()?;
    let base = std::path::Path::new(&sh).file_name()?.to_string_lossy().to_string();
    match base.as_str() {
        "bash" | "zsh" | "fish" | "elvish" | "powershell" | "pwsh" => Some(
            if base == "pwsh" { "powershell".to_string() } else { base },
        ),
        _ => None,
    }
}

fn parse_sets(sets: &[String]) -> Result<Vec<(String, String)>, VecdError> {
    sets.iter()
        .map(|s| {
            s.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .ok_or_else(|| VecdError::usage(format!("--set expects pos=val, got '{s}'")))
        })
        .collect()
}

fn resolve_toggle(active: bool, no_active: bool, what: &str) -> Result<bool, VecdError> {
    match (active, no_active) {
        (true, false) => Ok(true),
        (false, true) => Ok(false),
        _ => Err(VecdError::usage(format!("give exactly one of --active / --no-active for {what}"))),
    }
}

fn toggle_opt(active: bool, no_active: bool, what: &str) -> Result<Option<bool>, VecdError> {
    match (active, no_active) {
        (false, false) => Ok(None),
        (true, false) => Ok(Some(true)),
        (false, true) => Ok(Some(false)),
        _ => Err(VecdError::usage(format!("--active and --no-active conflict for {what}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_bind_precedence() {
        let empty = config::Config::default();
        let with_bind = config::Config::parse("bind = 10.0.0.1:9000").unwrap();

        // --bind flag wins over everything.
        assert_eq!(
            resolve_bind(&Some("127.0.0.1:1".to_string()), &with_bind),
            "127.0.0.1:1"
        );
        // No flag → fall back to vecd.conf's `bind`.
        assert_eq!(resolve_bind(&None, &with_bind), "10.0.0.1:9000");
        // Neither set → the built-in loopback default (safe out of the box).
        assert_eq!(resolve_bind(&None, &empty), "127.0.0.1:8443");
    }

    #[test]
    fn local_endpoint_url_from_bind() {
        let url = |conf: &str| local_endpoint_url(&config::Config::parse(conf).unwrap());
        // Loopback default (no config) stays loopback.
        assert_eq!(local_endpoint_url(&config::Config::default()), "http://127.0.0.1:8443/");
        // A fixed bind is used as-is.
        assert_eq!(url("bind = 127.0.0.1:18443"), "http://127.0.0.1:18443/");
        // Wildcard binds map to loopback for a usable client URL.
        assert_eq!(url("bind = 0.0.0.0:8443"), "http://127.0.0.1:8443/");
        // TLS config → https scheme.
        assert_eq!(
            url("bind = 10.0.0.5:9000\ntls_cert = /c.pem\ntls_key = /k.pem"),
            "https://10.0.0.5:9000/"
        );
    }

    #[test]
    fn exposes_plaintext_policy() {
        let loopback: SocketAddr = "127.0.0.1:8443".parse().unwrap();
        let loopback6: SocketAddr = "[::1]:8443".parse().unwrap();
        let any: SocketAddr = "0.0.0.0:8443".parse().unwrap();
        let public: SocketAddr = "10.0.0.5:8443".parse().unwrap();

        // Loopback is always fine, TLS or not.
        assert!(!exposes_plaintext(&loopback, false));
        assert!(!exposes_plaintext(&loopback6, false));
        // Non-loopback without TLS is the cleartext-exposure case → warn.
        assert!(exposes_plaintext(&any, false));
        assert!(exposes_plaintext(&public, false));
        // Non-loopback WITH TLS is fine (and so is loopback with TLS).
        assert!(!exposes_plaintext(&any, true));
        assert!(!exposes_plaintext(&public, true));
    }
}
