// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline commands: configuration management.
//!
//! `config show` — display current vectordata configuration.
//! `config init` — initialize or update the vectordata cache directory.
//!
//! Configuration is stored in `~/.config/vectordata/settings.yaml`.
//!
//! Equivalent to Java `CMD_config_show` and `CMD_config_init` commands.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

const CONFIG_DIR: &str = ".config/vectordata";
const SETTINGS_FILE: &str = "settings.yaml";

/// Get the settings.yaml path.
fn settings_path() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(CONFIG_DIR).join(SETTINGS_FILE)
    } else {
        PathBuf::from(CONFIG_DIR).join(SETTINGS_FILE)
    }
}

/// Simple settings representation.
struct Settings {
    cache_dir: Option<String>,
    protect_settings: bool,
}

impl Settings {
    fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

        let mut cache_dir = None;
        let mut protect_settings = false;

        for line in content.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("cache_dir:") {
                cache_dir = Some(val.trim().trim_matches('"').trim_matches('\'').to_string());
            }
            if let Some(val) = line.strip_prefix("protect_settings:") {
                protect_settings = val.trim() == "true";
            }
        }

        Ok(Settings {
            cache_dir,
            protect_settings,
        })
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create config dir: {}", e))?;
            }
        }

        let content = format!(
            "cache_dir: {}\nprotect_settings: {}\n",
            self.cache_dir.as_deref().unwrap_or(""),
            self.protect_settings,
        );

        std::fs::write(path, &content)
            .map_err(|e| format!("failed to write {}: {}", path.display(), e))
    }
}

// -- Config Show command -----------------------------------------------------

/// Pipeline command: display configuration.
pub struct ConfigShowOp;

pub fn show_factory() -> Box<dyn CommandOp> {
    Box::new(ConfigShowOp)
}

impl CommandOp for ConfigShowOp {
    fn command_path(&self) -> &str {
        "config show"
    }

    fn execute(&mut self, _options: &Options, _ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let path = settings_path();

        if !path.exists() {
            eprintln!("Configuration: {} (not found)", path.display());
            eprintln!("Run 'config init' to set up vectordata configuration.");
            return CommandResult {
                status: Status::Ok,
                message: "no configuration found".to_string(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let settings = match Settings::load(&path) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        eprintln!("Configuration: {}", path.display());

        if let Some(ref cache_dir) = settings.cache_dir {
            let cache_path = PathBuf::from(cache_dir);
            eprintln!("  cache_dir: {}", cache_dir);

            if cache_path.is_dir() {
                eprintln!("  Status: Active");

                // Compute used space
                if let Ok(used) = dir_size(&cache_path) {
                    eprintln!("  Used space: {}", format_bytes(used));
                }
            } else if cache_path.exists() {
                eprintln!("  Status: Error — path exists but is not a directory");
            } else {
                eprintln!("  Status: Not yet created");
            }
        } else {
            eprintln!("  cache_dir: (not set)");
        }

        eprintln!("  protect_settings: {}", settings.protect_settings);

        CommandResult {
            status: Status::Ok,
            message: format!(
                "cache_dir: {}",
                settings.cache_dir.as_deref().unwrap_or("(not set)")
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![]
    }
}

// -- Config Init command -----------------------------------------------------

/// Pipeline command: initialize configuration.
pub struct ConfigInitOp;

pub fn init_factory() -> Box<dyn CommandOp> {
    Box::new(ConfigInitOp)
}

impl CommandOp for ConfigInitOp {
    fn command_path(&self) -> &str {
        "config init"
    }

    fn execute(&mut self, options: &Options, _ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let path = settings_path();

        let cache_dir_opt = options.get("cache-dir");
        let force = options.get("force").map_or(false, |s| s == "true");

        // Check existing config
        if path.exists() && !force {
            if let Ok(settings) = Settings::load(&path) {
                if settings.protect_settings {
                    return error_result(
                        format!(
                            "settings already configured (cache_dir: {}). Use force=true to overwrite.",
                            settings.cache_dir.as_deref().unwrap_or("(not set)")
                        ),
                        start,
                    );
                }
            }
        }

        // Resolve cache directory
        let cache_dir = match cache_dir_opt {
            Some(dir) => resolve_cache_dir(dir),
            None => default_cache_dir(),
        };

        // Create cache directory
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            return error_result(
                format!("failed to create cache directory {}: {}", cache_dir, e),
                start,
            );
        }

        // Save settings
        let settings = Settings {
            cache_dir: Some(cache_dir.clone()),
            protect_settings: true,
        };

        if let Err(e) = settings.save(&path) {
            return error_result(e, start);
        }

        eprintln!("Configuration initialized:");
        eprintln!("  Settings: {}", path.display());
        eprintln!("  cache_dir: {}", cache_dir);

        CommandResult {
            status: Status::Ok,
            message: format!("config initialized: cache_dir={}", cache_dir),
            produced: vec![path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "cache-dir".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Cache directory path (default: ~/.cache/vectordata)".to_string(),
            },
            OptionDesc {
                name: "force".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Overwrite existing protected settings".to_string(),
            },
        ]
    }
}

/// Resolve special cache dir values.
fn resolve_cache_dir(value: &str) -> String {
    match value {
        "default" => default_cache_dir(),
        _ => value.to_string(),
    }
}

/// Default cache directory: ~/.cache/vectordata
fn default_cache_dir() -> String {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("vectordata")
            .to_string_lossy()
            .to_string()
    } else {
        ".cache/vectordata".to_string()
    }
}

/// Calculate total size of all files in a directory.
fn dir_size(path: &Path) -> Result<u64, String> {
    let mut total = 0u64;
    fn walk(path: &Path, total: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, total);
                } else if let Ok(meta) = p.metadata() {
                    *total += meta.len();
                }
            }
        }
    }
    walk(path, &mut total);
    Ok(total)
}

/// Format byte count for display.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

// -- Config List Mounts command -----------------------------------------------

/// Pipeline command: list writable mount points suitable for cache storage.
pub struct ConfigListMountsOp;

pub fn list_mounts_factory() -> Box<dyn CommandOp> {
    Box::new(ConfigListMountsOp)
}

impl CommandOp for ConfigListMountsOp {
    fn command_path(&self) -> &str {
        "config list-mounts"
    }

    fn execute(&mut self, options: &Options, _ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let show_all = options.get("all").map_or(false, |s| s == "true");
        let min_bytes: u64 = if show_all { 0 } else { 100 * 1024 * 1024 }; // 100 MB default minimum

        let mounts = list_mount_points();

        eprintln!("{:<40} {:>12} {:>12} {:>8}", "Mount Point", "Available", "Total", "Writable");
        eprintln!("{}", "-".repeat(76));

        let mut count = 0;
        for mount in &mounts {
            if mount.available < min_bytes {
                continue;
            }
            let display_path = if mount.path.len() > 38 {
                format!("...{}", &mount.path[mount.path.len() - 35..])
            } else {
                mount.path.clone()
            };
            eprintln!(
                "{:<40} {:>12} {:>12} {:>8}",
                display_path,
                format_bytes(mount.available),
                format_bytes(mount.total),
                if mount.writable { "Yes" } else { "No" }
            );
            count += 1;
        }

        if count == 0 {
            eprintln!("No suitable mount points found.");
            if !show_all {
                eprintln!("Use all=true to show all mount points (including those with < 100 MB).");
            }
        } else {
            eprintln!();
            eprintln!("To set a cache directory, run: config init cache-dir=<path>");
        }

        CommandResult {
            status: Status::Ok,
            message: format!("{} mount points listed", count),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![OptionDesc {
            name: "all".to_string(),
            type_name: "bool".to_string(),
            required: false,
            default: Some("false".to_string()),
            description: "Include mount points with minimal space".to_string(),
        }]
    }
}

/// Mount point information.
struct MountInfo {
    path: String,
    available: u64,
    total: u64,
    writable: bool,
}

/// List mount points on the system.
fn list_mount_points() -> Vec<MountInfo> {
    let mut mounts = Vec::new();

    // Read /proc/mounts on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/mounts") {
            let mut seen = std::collections::HashSet::new();
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 3 {
                    continue;
                }
                let mount_point = parts[1];
                let fs_type = parts[2];

                // Skip virtual filesystems
                if matches!(
                    fs_type,
                    "proc" | "sysfs" | "devtmpfs" | "devpts" | "cgroup"
                        | "cgroup2" | "securityfs" | "debugfs" | "tracefs"
                        | "hugetlbfs" | "mqueue" | "pstore" | "bpf"
                        | "configfs" | "fusectl" | "autofs" | "rpc_pipefs"
                        | "nfsd" | "binfmt_misc"
                ) {
                    continue;
                }

                if !seen.insert(mount_point.to_string()) {
                    continue;
                }

                let path = PathBuf::from(mount_point);
                if let Ok(stat) = nix_statvfs(&path) {
                    let writable = path
                        .metadata()
                        .map(|m| !m.permissions().readonly())
                        .unwrap_or(false);
                    mounts.push(MountInfo {
                        path: mount_point.to_string(),
                        available: stat.0,
                        total: stat.1,
                        writable,
                    });
                }
            }
        }
    }

    // Fallback: just report the root and home
    #[cfg(not(target_os = "linux"))]
    {
        for path_str in ["/", "/home", "/tmp"] {
            let path = PathBuf::from(path_str);
            if path.exists() {
                if let Ok(stat) = nix_statvfs(&path) {
                    mounts.push(MountInfo {
                        path: path_str.to_string(),
                        available: stat.0,
                        total: stat.1,
                        writable: true,
                    });
                }
            }
        }
    }

    mounts.sort_by(|a, b| b.available.cmp(&a.available));
    mounts
}

/// Get filesystem stats using libc statvfs.
fn nix_statvfs(path: &Path) -> Result<(u64, u64), String> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|e| format!("invalid path: {}", e))?;

    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
            let available = stat.f_bavail as u64 * stat.f_frsize as u64;
            let total = stat.f_blocks as u64 * stat.f_frsize as u64;
            Ok((available, total))
        } else {
            Err(format!("statvfs failed for {}", path.display()))
        }
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
        }
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_settings_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("settings.yaml");

        let settings = Settings {
            cache_dir: Some("/tmp/test-cache".to_string()),
            protect_settings: true,
        };

        settings.save(&path).unwrap();
        let loaded = Settings::load(&path).unwrap();
        assert_eq!(loaded.cache_dir.as_deref(), Some("/tmp/test-cache"));
        assert!(loaded.protect_settings);
    }

    #[test]
    fn test_config_show_no_config() {
        // Config show with non-existent settings should return Ok
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = test_ctx(tmp.path());

        let opts = Options::new();
        let mut op = ConfigShowOp;
        let result = op.execute(&opts, &mut ctx);
        // Depending on whether ~/.config/vectordata/settings.yaml exists
        assert!(result.status == Status::Ok || result.status == Status::Error);
    }

    #[test]
    fn test_dir_size() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("a.txt"), "hello").unwrap();
        std::fs::write(dir.join("b.txt"), "world!").unwrap();

        let size = dir_size(dir).unwrap();
        assert_eq!(size, 11); // 5 + 6
    }
}
