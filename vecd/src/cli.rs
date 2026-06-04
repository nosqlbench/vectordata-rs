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

use clap::{Args, Parser, Subcommand};
use clap_complete::Shell;

use crate::model::{Level, Listable, VecdError};
use crate::{admin, backup, config, db::Db, server};

#[derive(Parser)]
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

    #[command(subcommand)]
    pub command: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Create the DB + schema and mint the first superuser token.
    Init {
        /// Name for the bootstrap superuser.
        #[arg(long, default_value = "root")]
        superuser: String,
        /// Print only the token plaintext (for `$(…)` capture in scripts).
        #[arg(long)]
        quiet: bool,
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
    /// Read the access log.
    Log {
        #[arg(long, default_value_t = 50)]
        tail: usize,
    },
    /// Emit a sourceable shell snippet that activates dynamic completions:
    ///   eval "$(vecd completions)"
    Completions {
        #[arg(long)]
        shell: Option<Shell>,
    },
}

/// Flags shared by `serve`, `start`, and `restart`.
#[derive(Args, Clone)]
pub struct ServeArgs {
    /// Bind address (use `:0` for an ephemeral port). Falls back to the
    /// `bind` key in `vecd.conf`, then to `0.0.0.0:8443`.
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
    #[arg(long, default_value_t = 24)]
    pub backup_retain: usize,
}

#[derive(Subcommand)]
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

#[derive(Subcommand)]
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
    },
    List {
        #[arg(long)]
        user: Option<String>,
    },
    Revoke { id: i64 },
}

#[derive(Subcommand)]
pub enum RolesCmd {
    List,
    Add {
        name: String,
        #[arg(long)]
        actions: String,
    },
    Remove { name: String },
}

#[derive(Subcommand)]
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

#[derive(Subcommand)]
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

#[derive(Subcommand)]
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

#[derive(Subcommand)]
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

#[derive(Subcommand)]
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

#[derive(Subcommand)]
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
fn resolve_data_dir(flag: &Option<PathBuf>, cfg: &config::Config) -> PathBuf {
    if let Some(d) = flag {
        return d.clone();
    }
    if let Some(d) = cfg.get("data_dir") {
        return PathBuf::from(d);
    }
    config::default_data_dir()
}

/// Resolve the bind address: `--bind`, else config `bind`, else the default
/// `0.0.0.0:8443`. Mirrors [`resolve_data_dir`]'s flag-over-config precedence.
fn resolve_bind(flag: &Option<String>, cfg: &config::Config) -> String {
    flag.clone()
        .or_else(|| cfg.get("bind").map(|s| s.to_string()))
        .unwrap_or_else(|| "0.0.0.0:8443".to_string())
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
    let cfg = match config::Config::load(&config::config_dir()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("vecd: {e}");
            return e.exit_code();
        }
    };
    let data_dir = resolve_data_dir(&cli.data_dir, &cfg);

    let result = dispatch(cli.command, &data_dir, &cfg);
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("vecd: {e}");
            e.exit_code()
        }
    }
}

fn dispatch(cmd: Cmd, data_dir: &std::path::Path, cfg: &config::Config) -> Result<(), VecdError> {
    match cmd {
        Cmd::Init { superuser, quiet } => cmd_init(data_dir, &superuser, quiet),
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
        Cmd::Log { tail } => cmd_log(data_dir, cfg, tail),
        Cmd::Completions { shell } => {
            cmd_completions(shell);
            Ok(())
        }
    }
}

fn cmd_init(data_dir: &std::path::Path, superuser: &str, quiet: bool) -> Result<(), VecdError> {
    config::write_starter(&config::config_dir())?;
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
    if quiet {
        println!("{}", tok.plaintext);
        return Ok(());
    }
    println!("initialized vecd at {}", data_dir.display());
    println!("superuser:  {superuser}");
    println!("token:      {}", tok.plaintext);
    println!("expires:    {}", admin::fmt_epoch(tok.expires_at));
    println!("\nStore this token now — it is not recoverable.");
    Ok(())
}

fn cmd_serve(data_dir: &std::path::Path, cfg: &config::Config, a: &ServeArgs) -> Result<(), VecdError> {
    let bind = resolve_bind(&a.bind, cfg);
    let addr: SocketAddr = bind
        .parse()
        .map_err(|e| VecdError::usage(format!("bad bind address '{bind}': {e}")))?;
    let tls = match (a.tls_cert.clone(), a.tls_key.clone()) {
        (Some(cert), Some(key)) => Some(server::TlsConfig { cert, key }),
        (None, None) => None,
        _ => return Err(VecdError::usage("--tls-cert and --tls-key must be given together")),
    };
    let db = open_db(data_dir, cfg)?;
    let state = server::AppState::new(db)?;

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
        TokensCmd::Create { user, description, expires, profile, from, sets, quiet } => {
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
            if quiet {
                println!("{}", tok.plaintext);
            } else {
                println!("token:   {}", tok.plaintext);
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

fn cmd_log(data_dir: &std::path::Path, cfg: &config::Config, tail: usize) -> Result<(), VecdError> {
    let db = open_db(data_dir, cfg)?;
    let mut stmt = db.conn().prepare(
        "SELECT ts,principal,action,key,status,bytes,remote_addr FROM access_log ORDER BY id DESC LIMIT ?1",
    )?;
    let rows = stmt
        .query_map([tail as i64], |r| {
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

/// Print the completions activation snippet (mirrors `vectordata`).
fn cmd_completions(shell: Option<Shell>) {
    let shell = shell
        .or_else(Shell::from_env)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "bash".to_string());
    println!("COMPLETE={shell} vecd");
    eprintln!("# To activate now:  eval \"$(vecd completions)\"");
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
        // Neither set → the built-in default.
        assert_eq!(resolve_bind(&None, &empty), "0.0.0.0:8443");
    }
}
