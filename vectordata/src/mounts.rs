// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Mount-point enumeration with available/total space + writability,
//! shared by [`crate::settings`] (auto-bootstrap of `cache_dir`) and
//! [`crate::config::list_mounts`] (the user-facing CLI).
//!
//! Linux: reads `/proc/mounts`, filters virtual filesystems, calls
//! `statvfs` per mount point. Non-Linux Unix: falls back to a small
//! fixed set (`/`, `/home`, `/tmp`). Non-Unix targets return an
//! empty list — auto-resolution can still work as long as `$HOME`
//! is readable; it just won't have anywhere to compare against.

use std::path::{Path, PathBuf};

/// One row in the live mount table.
#[derive(Debug)]
pub struct MountInfo {
    /// The mount point (e.g. `/`, `/mnt/datamir`).
    pub path: String,
    /// Bytes available to the current user.
    pub available: u64,
    /// Total bytes on the filesystem.
    pub total: u64,
    /// Whether the current user can write to this mount.
    pub writable: bool,
}

/// Enumerate writable mount points, sorted by available space
/// descending. The first entry, when non-empty, is the natural
/// candidate for a "where should the cache live" auto-pick.
pub fn enumerate() -> Vec<MountInfo> {
    let mut mounts = enumerate_raw();
    mounts.sort_by(|a, b| b.available.cmp(&a.available));
    mounts
}

#[cfg(target_os = "linux")]
fn enumerate_raw() -> Vec<MountInfo> {
    let mut mounts = Vec::new();
    let Ok(content) = std::fs::read_to_string("/proc/mounts") else { return mounts; };
    let mut seen = std::collections::HashSet::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 { continue; }
        let mount = parts[1];
        let fs_type = parts[2];
        if matches!(fs_type,
            "proc" | "sysfs" | "devtmpfs" | "devpts" | "cgroup" | "cgroup2"
            | "securityfs" | "debugfs" | "tracefs" | "hugetlbfs" | "mqueue"
            | "pstore" | "bpf" | "configfs" | "fusectl" | "autofs"
            | "rpc_pipefs" | "nfsd" | "binfmt_misc")
        { continue; }
        if !seen.insert(mount.to_string()) { continue; }
        let path = PathBuf::from(mount);
        if let Some((available, total)) = statvfs_bytes(&path) {
            mounts.push(MountInfo {
                path: mount.to_string(),
                available, total,
                writable: is_writable(&path),
            });
        }
    }
    mounts
}

#[cfg(not(target_os = "linux"))]
fn enumerate_raw() -> Vec<MountInfo> {
    let mut mounts = Vec::new();
    for p in ["/", "/home", "/tmp"] {
        let path = PathBuf::from(p);
        if !path.exists() { continue; }
        if let Some((available, total)) = statvfs_bytes(&path) {
            mounts.push(MountInfo {
                path: p.to_string(), available, total, writable: true,
            });
        }
    }
    mounts
}

/// Available / total bytes for the filesystem hosting `path`.
/// Returns `None` on syscall failure or unsupported platform.
#[cfg(unix)]
pub fn statvfs_bytes(path: &Path) -> Option<(u64, u64)> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    unsafe {
        let mut s: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut s) != 0 { return None; }
        let avail = s.f_bavail as u64 * s.f_frsize as u64;
        let total = s.f_blocks as u64 * s.f_frsize as u64;
        Some((avail, total))
    }
}

#[cfg(not(unix))]
pub fn statvfs_bytes(_path: &Path) -> Option<(u64, u64)> { None }

/// Best-effort "can the current user write here" check. Uses POSIX
/// mode bits + uid/gid; returns the coarse readonly flag on non-Unix.
#[cfg(unix)]
pub fn is_writable(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(meta) = path.metadata() else { return false; };
    let mode = meta.mode();
    let uid = unsafe { libc::getuid() };
    if uid == 0 { return true; }
    if meta.uid() == uid { return mode & 0o200 != 0; }
    let gid = unsafe { libc::getegid() };
    if meta.gid() == gid { return mode & 0o020 != 0; }
    let mut groups = vec![0u32; 64];
    let ret = unsafe {
        libc::getgroups(groups.len() as libc::c_int,
            groups.as_mut_ptr() as *mut libc::gid_t)
    };
    if ret > 0 {
        groups.truncate(ret as usize);
        if groups.contains(&meta.gid()) { return mode & 0o020 != 0; }
    }
    mode & 0o002 != 0
}

#[cfg(not(unix))]
pub fn is_writable(path: &Path) -> bool {
    path.metadata().map(|m| !m.permissions().readonly()).unwrap_or(false)
}

/// The device id of `path`'s filesystem. Used to test whether two
/// paths live on the same mount (e.g. `$HOME` vs the largest mount
/// in [`enumerate`]).
#[cfg(unix)]
pub fn device_id(path: &Path) -> Result<u64, String> {
    use std::os::unix::fs::MetadataExt;
    path.metadata()
        .map(|m| m.dev())
        .map_err(|e| format!("{e}"))
}

#[cfg(not(unix))]
pub fn device_id(_path: &Path) -> Result<u64, String> {
    Err("device-id comparison is unsupported on this platform".to_string())
}
