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
        if !is_persistent_storage_fs(fs_type) { continue; }
        if is_system_mount_path(mount) { continue; }
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

/// Reject filesystem types that aren't persistent disk-backed
/// storage. The list is *exclusionary* — anything that:
///
/// - Is RAM-backed (`tmpfs`, `ramfs`, `hugetlbfs`) — volatile, lost
///   on reboot, and consumes process-visible memory the user
///   didn't budget for.
/// - Is a kernel virtual filesystem (`proc`, `sysfs`, `cgroup*`,
///   `debugfs`, `tracefs`, `securityfs`, `bpf`, `pstore`,
///   `efivarfs`, `mqueue`, `devpts`, `devtmpfs`, `binfmt_misc`,
///   `autofs`, `configfs`, `fusectl`, `nsfs`, `pipefs`, `nfsd`,
///   `rpc_pipefs`).
/// - Is a read-only image (`squashfs`, `iso9660`, `cramfs`).
/// - Is an overlay / union mount (`overlay`, `overlayfs`,
///   `aufs`) — the actual backing may be tmpfs and the
///   write-through layer rules are caller-specific.
///
/// Intentionally a denylist rather than an allowlist so unknown
/// disk-backed types (bcachefs, sshfs-via-network-but-persistent,
/// etc.) are still considered. The user can always override by
/// passing an explicit `cache_dir` path.
pub(crate) fn is_persistent_storage_fs(fs_type: &str) -> bool {
    !matches!(
        fs_type,
        // RAM-backed.
        "tmpfs" | "ramfs" | "hugetlbfs"
        // Kernel virtual.
        | "proc" | "sysfs" | "devtmpfs" | "devpts" | "cgroup" | "cgroup2"
        | "securityfs" | "debugfs" | "tracefs" | "mqueue" | "pstore"
        | "bpf" | "configfs" | "fusectl" | "autofs" | "binfmt_misc"
        | "efivarfs" | "nsfs" | "pipefs" | "rpc_pipefs" | "nfsd"
        // Read-only images.
        | "squashfs" | "iso9660" | "cramfs"
        // Overlay / union (rules depend on backing layers; defer to
        // the caller-supplied explicit path).
        | "overlay" | "overlayfs" | "aufs",
    )
}

/// Reject mount paths that belong to the system rather than user
/// data, even when the filesystem type would otherwise pass.
/// These are the canonical Linux distro locations for boot
/// firmware, snap packages, and runtime state.
pub(crate) fn is_system_mount_path(mount: &str) -> bool {
    matches!(mount, "/boot" | "/boot/efi")
        || mount.starts_with("/boot/")
        || mount.starts_with("/snap")
        || mount.starts_with("/var/snap")
        || mount.starts_with("/var/lib/snapd")
        || mount.starts_with("/proc")
        || mount.starts_with("/sys")
        || mount == "/dev"
        || mount.starts_with("/dev/")
        || mount == "/run"
        || mount.starts_with("/run/")
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

#[cfg(test)]
mod tests {
    use super::*;

    /// RAM-backed and kernel-virtual filesystems are always
    /// rejected, regardless of size. This is the core fix for the
    /// auto-cache picking `tmpfs on /run` / `/dev/shm`.
    #[test]
    fn ram_and_virtual_fs_types_rejected() {
        for fs in [
            "tmpfs", "ramfs", "hugetlbfs",
            "proc", "sysfs", "devtmpfs", "devpts",
            "cgroup", "cgroup2", "securityfs", "debugfs", "tracefs",
            "mqueue", "pstore", "bpf", "configfs", "fusectl", "autofs",
            "binfmt_misc", "efivarfs", "nsfs",
        ] {
            assert!(!is_persistent_storage_fs(fs), "{fs} should be rejected");
        }
    }

    /// Read-only image filesystems used for snap / installer
    /// content are not viable cache homes.
    #[test]
    fn read_only_image_fs_types_rejected() {
        for fs in ["squashfs", "iso9660", "cramfs"] {
            assert!(!is_persistent_storage_fs(fs), "{fs} should be rejected");
        }
    }

    /// Overlay/union mounts are rejected — the backing layer may be
    /// tmpfs, and the caller is the only one who can reason about
    /// the write-through semantics.
    #[test]
    fn overlay_fs_types_rejected() {
        for fs in ["overlay", "overlayfs", "aufs"] {
            assert!(!is_persistent_storage_fs(fs), "{fs} should be rejected");
        }
    }

    /// Common disk-backed filesystems pass through. Allow-by-default
    /// is intentional so unusual but persistent filesystems
    /// (bcachefs, zfs subvolumes, network-attached but durable
    /// stores) are still considered.
    #[test]
    fn common_disk_fs_types_accepted() {
        for fs in [
            "ext4", "ext3", "ext2", "xfs", "btrfs", "zfs", "f2fs",
            "jfs", "reiserfs", "ntfs", "ntfs3", "exfat", "vfat",
            "bcachefs", "nfs", "nfs4", "cifs", "smb3",
        ] {
            assert!(is_persistent_storage_fs(fs), "{fs} should be accepted");
        }
    }

    /// System mount points are rejected even when their filesystem
    /// type would otherwise pass (e.g. `vfat on /boot/efi`).
    #[test]
    fn system_mount_paths_rejected() {
        for p in [
            "/boot", "/boot/efi", "/boot/grub",
            "/snap", "/snap/core22/2339", "/snap/snapd/26865",
            "/var/snap/lxd/common",
            "/var/lib/snapd/snaps/snapd_26865.snap",
            "/proc", "/proc/sys",
            "/sys", "/sys/fs/cgroup",
            "/dev", "/dev/shm", "/dev/pts",
            "/run", "/run/lock", "/run/user/1000",
        ] {
            assert!(is_system_mount_path(p), "{p} should be system-mount-rejected");
        }
    }

    /// User data paths and the root filesystem pass through. The
    /// root mount is the most common cache home on single-disk
    /// systems and must never be filtered out.
    #[test]
    fn user_data_paths_accepted() {
        for p in [
            "/", "/home", "/home/alice",
            "/mnt", "/mnt/datamir",
            "/media/usb",
            "/data", "/var", "/var/cache",
            "/opt",
        ] {
            assert!(!is_system_mount_path(p), "{p} should be accepted");
        }
    }

    /// Regression: with the mount-table sample the user provided,
    /// only the root ext4 mount should survive both filters. This
    /// is the bug that originally surfaced.
    #[test]
    fn user_supplied_mount_table_picks_only_root() {
        // (fs_type, mount) pairs lifted directly from the user's
        // `mount` output. The expectation: only `/` survives.
        let table = vec![
            ("ext4", "/"),
            ("devtmpfs", "/dev"),
            ("proc", "/proc"),
            ("sysfs", "/sys"),
            ("securityfs", "/sys/kernel/security"),
            ("tmpfs", "/dev/shm"),
            ("devpts", "/dev/pts"),
            ("tmpfs", "/run"),
            ("tmpfs", "/run/lock"),
            ("cgroup2", "/sys/fs/cgroup"),
            ("pstore", "/sys/fs/pstore"),
            ("efivarfs", "/sys/firmware/efi/efivars"),
            ("bpf", "/sys/fs/bpf"),
            ("autofs", "/proc/sys/fs/binfmt_misc"),
            ("hugetlbfs", "/dev/hugepages"),
            ("mqueue", "/dev/mqueue"),
            ("debugfs", "/sys/kernel/debug"),
            ("tracefs", "/sys/kernel/tracing"),
            ("fusectl", "/sys/fs/fuse/connections"),
            ("configfs", "/sys/kernel/config"),
            ("squashfs", "/snap/amazon-ssm-agent/12322"),
            ("squashfs", "/snap/core22/2339"),
            ("squashfs", "/snap/amazon-ssm-agent/13009"),
            ("squashfs", "/snap/core22/2411"),
            ("squashfs", "/snap/snapd/26382"),
            ("ext4", "/boot"),
            ("vfat", "/boot/efi"),
            ("binfmt_misc", "/proc/sys/fs/binfmt_misc"),
            ("squashfs", "/snap/snapd/26865"),
            ("tmpfs", "/run/user/1000"),
        ];
        let survivors: Vec<&str> = table
            .iter()
            .filter(|(fs, mount)| is_persistent_storage_fs(fs) && !is_system_mount_path(mount))
            .map(|(_, m)| *m)
            .collect();
        assert_eq!(
            survivors,
            vec!["/"],
            "only `/` should survive the filter on this mount table",
        );
    }
}
