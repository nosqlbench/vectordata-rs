// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! systemd integration: generate and install a unit that runs the
//! gateway under systemd (`vecd serve`), plus a best-effort interlock
//! against running the ad-hoc daemon (`vecd daemon start`) and the systemd
//! service at the same time.
//!
//! vecd has two ways to run as a background service:
//!
//!   1. **systemd** — a unit whose `ExecStart` runs the foreground
//!      `vecd serve`; systemd owns the process lifecycle. This module
//!      writes that unit.
//!   2. **ad-hoc** — `vecd daemon start`/`stop`/`status`, which self-daemonize
//!      via `fork`/`setsid` (see [`crate::daemon`]).
//!
//! Use one OR the other. Running both points two daemons at the same
//! bind address and control DB. The interlock here ([`install`]'s
//! ad-hoc check and [`refuse_if_systemd_active`] on the `start` path)
//! catches the cases it can detect and refuses; it's best-effort
//! (a missing `systemctl`, a custom unit name, or a stale pidfile can
//! defeat it), never a hard guarantee.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::{Config, DaemonMode, Resolved};
use crate::model::VecdError;

/// Default install location for the system-scope unit.
pub const DEFAULT_UNIT_PATH: &str = "/etc/systemd/system/vecd.service";

/// The unit name `systemctl` operates on (the `DEFAULT_UNIT_PATH`
/// basename). The `start`-path interlock checks this name; a unit
/// installed elsewhere under another name won't be detected (the
/// interlock is best-effort).
pub const UNIT_NAME: &str = "vecd";

/// Render the systemd unit text for a vecd gateway. `exec` is the
/// absolute path to the vecd binary; `conf_dir` is the config
/// directory the daemon should read (passed as `--conf`). Pure — no
/// I/O — so it backs both `--print` and the install path, and is
/// directly testable.
pub fn unit_text(exec: &str, conf_dir: &Path) -> String {
    format!(
        "[Unit]\n\
         Description=vectordata endpoint daemon (vecd)\n\
         Documentation=https://github.com/nosqlbench/vectordata-rs\n\
         After=network-online.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         # Foreground gateway under systemd's control — NOT `vecd daemon start`\n\
         # (that self-daemonizes and would fight systemd for the process).\n\
         Type=simple\n\
         ExecStart={exec} --conf {conf} serve\n\
         Restart=on-failure\n\
         RestartSec=2\n\
         \n\
         [Install]\n\
         WantedBy=multi-user.target\n",
        exec = exec,
        conf = conf_dir.display(),
    )
}

/// True when the `vecd` systemd unit is currently active. Best-effort:
/// shells out to `systemctl is-active --quiet`; a missing `systemctl`,
/// a non-systemd host, or any error reads as "not active / can't tell"
/// so the caller never blocks on an inability to check.
pub fn systemd_unit_active() -> bool {
    Command::new("systemctl")
        .args(["is-active", "--quiet", UNIT_NAME])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Refuse a command (the ad-hoc `start`/`restart` path under
/// `daemon_mode=adhoc`) when the systemd `vecd` service is
/// nonetheless active — e.g. a unit was enabled out-of-band without
/// `service install` flipping the mode. Best-effort — see
/// [`systemd_unit_active`]; a no-op when systemd isn't managing vecd.
pub fn refuse_if_systemd_active() -> Result<(), VecdError> {
    if systemd_unit_active() {
        return Err(VecdError::usage(
            "the vecd systemd service is active but daemon_mode=adhoc — refusing to also \
             start an ad-hoc daemon (they would collide on the bind address and control DB). \
             Run `vecd daemon install` to adopt systemd mode, or `systemctl disable --now vecd` \
             to stay ad-hoc.",
        ));
    }
    Ok(())
}

/// Route an ad-hoc lifecycle verb (`start`/`stop`/`status`/`restart`)
/// to systemd, used when `daemon_mode=systemd`. Prints a one-line
/// stderr notice so the operator sees the indirection, then runs
/// `systemctl <verb> [--no-pager] vecd`. `status` is informational —
/// its "inactive" non-zero exit is shown via the inherited output, not
/// raised as an error.
pub fn delegate(verb: &str) -> Result<(), VecdError> {
    eprintln!(
        "vecd: daemon_mode=systemd — using systemd; routing `vecd {verb}` to \
         `systemctl {verb} {UNIT_NAME}`"
    );
    let mut args: Vec<&str> = vec![verb];
    if verb == "status" {
        args.push("--no-pager");
    }
    args.push(UNIT_NAME);
    let status = Command::new("systemctl").args(&args).status().map_err(|e| {
        VecdError::op(format!("running `systemctl {}`: {e} — is systemd present?", args.join(" ")))
    })?;
    if verb == "status" {
        return Ok(()); // the status output is the answer; inactive isn't an error
    }
    if !status.success() {
        return Err(VecdError::op(format!(
            "`systemctl {}` failed (exit {})",
            args.join(" "),
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

/// Install (or, with `print`, just emit) the systemd unit.
///
/// `--print` writes the unit to stdout and changes nothing — no root,
/// no interlock. Otherwise the unit is written to [`DEFAULT_UNIT_PATH`]
/// (needs root), `systemctl daemon-reload` is run, and the unit is
/// `enable`d — `--now` to start it immediately unless `no_start`.
///
/// Before any system change, refuses if an ad-hoc daemon is already
/// running (it would conflict with the service this is about to start).
pub fn install(
    resolved: &Resolved,
    data_dir: &Path,
    exec: Option<PathBuf>,
    print: bool,
    no_start: bool,
) -> Result<(), VecdError> {
    let exec = match exec {
        Some(p) => p,
        None => std::env::current_exe()
            .map_err(|e| VecdError::op(format!("cannot determine the vecd binary path: {e} \
                (pass `--exec <path>`)")))?,
    };
    // systemd runs the unit with an unspecified cwd, so both the binary
    // and the `--conf` dir must be absolute. current_exe() is already
    // absolute; canonicalize the conf dir (it exists — the config was
    // loaded), falling back to the given path if that somehow fails.
    let conf_dir = std::fs::canonicalize(&resolved.dir).unwrap_or_else(|_| resolved.dir.clone());
    let unit = unit_text(&exec.to_string_lossy(), &conf_dir);

    if print {
        // Emit the unit and change nothing — no root, no interlock, no
        // config write.
        print!("{unit}");
        return Ok(());
    }

    // Install records `daemon_mode=systemd`, so it needs a writable,
    // unlocked config. Check up front so we don't half-install (unit
    // written, mode not flipped) when the config is locked.
    let mut cfg = Config::load(&resolved.dir)?;
    if cfg.is_locked() {
        return Err(VecdError::usage(
            "config is locked (lock_config on); run `vecd config set lock_config off --force` \
             before installing the systemd service",
        ));
    }

    // Interlock: enabling + starting the unit while an ad-hoc daemon is
    // up would run two gateways against the same bind + DB.
    if let Some(pid) = crate::daemon::running_pid(data_dir) {
        return Err(VecdError::usage(format!(
            "an ad-hoc vecd daemon is running (pid {pid}); stop it with `vecd daemon stop` before \
             installing the systemd service — the two would conflict on the bind address \
             and control DB",
        )));
    }

    let unit_path = PathBuf::from(DEFAULT_UNIT_PATH);
    std::fs::write(&unit_path, &unit).map_err(|e| {
        VecdError::op(format!(
            "writing {}: {e}\n  installing a system unit needs root — re-run with `sudo`, or \
             write it yourself: `vecd daemon install --print | sudo tee {}`",
            unit_path.display(),
            unit_path.display()
        ))
    })?;
    println!("wrote {}", unit_path.display());

    systemctl(&["daemon-reload"])?;
    if no_start {
        systemctl(&["enable", UNIT_NAME])?;
        println!("enabled {UNIT_NAME} (not started; run `systemctl start {UNIT_NAME}` when ready)");
    } else {
        systemctl(&["enable", "--now", UNIT_NAME])?;
        println!("enabled and started {UNIT_NAME} — check `systemctl status {UNIT_NAME}`");
    }

    // Flip the mode so `start`/`stop`/`status`/`restart` now delegate to
    // systemd from here on.
    cfg.set("daemon_mode", DaemonMode::Systemd.as_str())?;
    cfg.write_to(&resolved.dir)?;
    println!("set daemon_mode = systemd in {}", resolved.conf_path().display());
    Ok(())
}

/// Reverse [`install`]: disable + stop the unit, delete the unit file,
/// reload systemd, and set `daemon_mode=adhoc` so the lifecycle verbs
/// self-daemonize again. The caller gates this on the mode actually
/// being systemd. Disable/stop is best-effort (a warning if it fails);
/// removing the unit file and reverting the mode are required.
pub fn uninstall(resolved: &Resolved) -> Result<(), VecdError> {
    let mut cfg = Config::load(&resolved.dir)?;
    if cfg.is_locked() {
        return Err(VecdError::usage(
            "config is locked (lock_config on); run `vecd config set lock_config off --force` \
             before uninstalling the systemd service",
        ));
    }

    // Best-effort disable + stop; if the unit is already gone this may
    // fail harmlessly, so warn rather than abort.
    match Command::new("systemctl").args(["disable", "--now", UNIT_NAME]).status() {
        Ok(s) if s.success() => {}
        Ok(_) => eprintln!("vecd: warning: `systemctl disable --now {UNIT_NAME}` reported a problem"),
        Err(e) => eprintln!("vecd: warning: could not run systemctl ({e}); skipping disable"),
    }

    let unit_path = PathBuf::from(DEFAULT_UNIT_PATH);
    if unit_path.exists() {
        std::fs::remove_file(&unit_path).map_err(|e| {
            VecdError::op(format!(
                "removing {}: {e} — removing a system unit needs root, re-run with `sudo`",
                unit_path.display()
            ))
        })?;
        println!("removed {}", unit_path.display());
        systemctl(&["daemon-reload"])?;
    } else {
        eprintln!("vecd: note: no unit at {} (already removed)", unit_path.display());
    }

    cfg.set("daemon_mode", DaemonMode::Adhoc.as_str())?;
    cfg.write_to(&resolved.dir)?;
    println!(
        "set daemon_mode = adhoc — `vecd daemon start`/`stop`/`status`/`restart` now self-daemonize"
    );
    Ok(())
}

/// Run `systemctl <args>`, mapping a missing binary or a non-zero exit
/// to a [`VecdError`]. Inherits stdio so systemctl's own diagnostics
/// reach the operator.
fn systemctl(args: &[&str]) -> Result<(), VecdError> {
    let status = Command::new("systemctl").args(args).status().map_err(|e| {
        VecdError::op(format!("running `systemctl {}`: {e} — is systemd present?", args.join(" ")))
    })?;
    if !status.success() {
        return Err(VecdError::op(format!(
            "`systemctl {}` failed (exit {})",
            args.join(" "),
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_text_runs_serve_with_conf_under_systemd() {
        let unit = unit_text("/usr/local/bin/vecd", Path::new("/etc/vecd"));
        // ExecStart runs the FOREGROUND serve (not the self-daemonizing
        // `start`), with the config dir wired in — the whole point.
        let execstart = unit
            .lines()
            .find(|l| l.starts_with("ExecStart="))
            .expect("unit has an ExecStart line");
        assert_eq!(execstart, "ExecStart=/usr/local/bin/vecd --conf /etc/vecd serve");
        assert!(execstart.ends_with(" serve"), "must run `serve`: {execstart}");
        assert!(!execstart.contains(" start"), "must not run `start`: {execstart}");
        assert!(unit.contains("Type=simple"));
        assert!(unit.contains("WantedBy=multi-user.target"));
        assert!(unit.contains("Restart=on-failure"));
    }
}
