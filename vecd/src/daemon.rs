// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Daemon lifecycle — `start`/`stop`/`status`/`restart` (decision #2:
//! self-daemonizing). `serve` remains the foreground workhorse (what
//! `start` runs after detaching, and what `systemd`/containers exec
//! directly).
//!
//! `start` double-forks + `setsid`s so the daemon is reparented and
//! session-detached, redirects stdio to `<data_dir>/vecd.log`, and writes
//! `<data_dir>/vecd.pid`. `stop` sends `SIGTERM` (graceful drain). The
//! fork happens **before** any tokio runtime exists — the runtime is built
//! only inside the foreground closure the grandchild runs. Unix only;
//! elsewhere, run `vecd serve` under a supervisor.

use std::path::{Path, PathBuf};

use crate::model::VecdError;

/// Path to the pidfile under a data dir.
pub fn pidfile_path(data_dir: &Path) -> PathBuf {
    data_dir.join("vecd.pid")
}

fn addr_path(data_dir: &Path) -> PathBuf {
    data_dir.join("vecd.addr")
}

/// The live daemon's pid, if one is recorded and the process is alive.
pub fn running_pid(data_dir: &Path) -> Option<i32> {
    let text = std::fs::read_to_string(pidfile_path(data_dir)).ok()?;
    let pid: i32 = text.trim().parse().ok()?;
    process_alive(pid).then_some(pid)
}

#[cfg(unix)]
fn process_alive(pid: i32) -> bool {
    // signal 0 probes existence without delivering a signal.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[cfg(not(unix))]
fn process_alive(_pid: i32) -> bool {
    false
}

/// Detach into the background and run `run_foreground` (the blocking
/// `serve`) in the daemonized grandchild. Returns in the **parent** once
/// the daemon has recorded its pidfile.
#[cfg(unix)]
pub fn start(data_dir: &Path, run_foreground: impl FnOnce() -> i32) -> Result<(), VecdError> {
    if let Some(pid) = running_pid(data_dir) {
        return Err(VecdError::usage(format!("vecd is already running (pid {pid})")));
    }
    std::fs::create_dir_all(data_dir)?;
    let pidfile = pidfile_path(data_dir);
    let logfile = data_dir.join("vecd.log");

    // First fork: parent waits for the pidfile, then returns success.
    match unsafe { libc::fork() } {
        -1 => return Err(VecdError::op("fork failed")),
        0 => {} // child continues below
        _parent => {
            // Wait briefly for the daemon to come up (publish its pidfile).
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while std::time::Instant::now() < deadline {
                if running_pid(data_dir).is_some() {
                    println!("vecd started (pid {})", running_pid(data_dir).unwrap());
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            return Err(VecdError::op("vecd did not come up within 10s; check vecd.log"));
        }
    }

    // Child: new session, then second fork so we can't reacquire a TTY.
    unsafe { libc::setsid() };
    match unsafe { libc::fork() } {
        -1 => unsafe { libc::_exit(1) },
        0 => {} // grandchild: the daemon
        _ => unsafe { libc::_exit(0) }, // first child exits immediately
    }

    // Grandchild: detach fully and become the daemon.
    redirect_stdio(&logfile);
    let _ = std::fs::write(&pidfile, std::process::id().to_string());

    let code = run_foreground();

    let _ = std::fs::remove_file(&pidfile);
    let _ = std::fs::remove_file(addr_path(data_dir));
    unsafe { libc::_exit(code) };
}

#[cfg(not(unix))]
pub fn start(_data_dir: &Path, _run_foreground: impl FnOnce() -> i32) -> Result<(), VecdError> {
    Err(VecdError::usage(
        "`vecd start` (self-daemonize) is unix-only; run `vecd serve` under a supervisor",
    ))
}

/// Send `SIGTERM` and wait for the daemon to drain and exit.
#[cfg(unix)]
pub fn stop(data_dir: &Path) -> Result<(), VecdError> {
    let Some(pid) = running_pid(data_dir) else {
        return Err(VecdError::usage("vecd is not running"));
    };
    if unsafe { libc::kill(pid, libc::SIGTERM) } != 0 {
        return Err(VecdError::op(format!("failed to signal pid {pid}")));
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while std::time::Instant::now() < deadline {
        if !process_alive(pid) {
            let _ = std::fs::remove_file(pidfile_path(data_dir));
            println!("vecd stopped (pid {pid})");
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Err(VecdError::op(format!("vecd (pid {pid}) did not exit within 15s")))
}

#[cfg(not(unix))]
pub fn stop(_data_dir: &Path) -> Result<(), VecdError> {
    Err(VecdError::usage("`vecd stop` is unix-only"))
}

/// Report whether the daemon is running, and where it is bound.
pub fn status(data_dir: &Path) -> Result<(), VecdError> {
    match running_pid(data_dir) {
        Some(pid) => {
            let addr = std::fs::read_to_string(addr_path(data_dir)).ok();
            match addr {
                Some(a) if !a.trim().is_empty() => {
                    println!("vecd is running (pid {pid}) on {}", a.trim())
                }
                _ => println!("vecd is running (pid {pid})"),
            }
        }
        None => println!("vecd is not running"),
    }
    Ok(())
}

/// Redirect stdin from `/dev/null` and stdout/stderr to the log file.
#[cfg(unix)]
fn redirect_stdio(logfile: &Path) {
    use std::os::unix::io::AsRawFd;
    if let Ok(devnull) = std::fs::OpenOptions::new().read(true).open("/dev/null") {
        unsafe { libc::dup2(devnull.as_raw_fd(), 0) };
    }
    if let Ok(log) = std::fs::OpenOptions::new().create(true).append(true).open(logfile) {
        let fd = log.as_raw_fd();
        unsafe {
            libc::dup2(fd, 1);
            libc::dup2(fd, 2);
        }
        // Keep `log` from closing the fd we just dup'd over 1/2.
        std::mem::forget(log);
    }
}
