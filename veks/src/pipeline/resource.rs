// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Resource management and governance for pipeline commands.
//!
//! Implements the resource governor system described in SRD §06. Provides:
//!
//! - [`ResourceValue`]: Parsed resource values (absolute, percentage, ranges)
//! - [`ResourceBudget`]: Per-resource floor/ceiling pairs from `--resources`
//! - [`ResourceGovernor`]: Runtime governor with monitoring and adaptive control
//! - [`GovernorStrategy`]: Pluggable strategy trait for resource adjustment
//! - [`ResourceDesc`]: Command-level resource declarations

use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::Serialize;

// ---------------------------------------------------------------------------
// Resource types (centralized definitions)
// ---------------------------------------------------------------------------

/// Known resource types with centralized semantics.
///
/// Each variant defines the canonical name, value kind, default value, and
/// display formatting for a resource type. Using this enum instead of raw
/// strings prevents typos and ensures consistent behavior across the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// Memory budget (RSS ceiling). Supports sizes with units and percentages.
    Mem,
    /// CPU thread pool size. Integer count.
    Threads,
    /// Maximum concurrent segments in flight. Integer count.
    Segments,
    /// Records per processing segment. Integer count.
    SegmentSize,
    /// Concurrent I/O operations. Integer count.
    IoThreads,
    /// CPU core limit. Integer count.
    Cpu,
    /// Maximum disk space for `.cache/`. Supports sizes with units and percentages.
    Cache,
    /// Read-ahead buffer / prefetch window. Supports sizes with units and percentages.
    Readahead,
}

/// Whether a resource type accepts memory-like values (sizes, percentages)
/// or count-like values (plain integers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// Sizes with units (GiB, MB, etc.) and percentages of system RAM.
    Memory,
    /// Plain integer counts.
    Count,
}

impl ResourceType {
    /// Canonical string name used in `--resources` and `ResourceDesc`.
    pub fn name(&self) -> &'static str {
        match self {
            ResourceType::Mem => "mem",
            ResourceType::Threads => "threads",
            ResourceType::Segments => "segments",
            ResourceType::SegmentSize => "segmentsize",
            ResourceType::IoThreads => "iothreads",
            ResourceType::Cpu => "cpu",
            ResourceType::Cache => "cache",
            ResourceType::Readahead => "readahead",
        }
    }

    /// What kind of values this resource accepts.
    pub fn value_kind(&self) -> ValueKind {
        match self {
            ResourceType::Mem | ResourceType::Cache | ResourceType::Readahead => ValueKind::Memory,
            ResourceType::Threads | ResourceType::Segments | ResourceType::SegmentSize
            | ResourceType::IoThreads | ResourceType::Cpu => ValueKind::Count,
        }
    }

    /// Human-readable description of this resource type.
    pub fn description(&self) -> &'static str {
        match self {
            ResourceType::Mem => "Memory budget (RSS ceiling)",
            ResourceType::Threads => "CPU thread pool size",
            ResourceType::Segments => "Maximum concurrent segments in flight",
            ResourceType::SegmentSize => "Records per processing segment",
            ResourceType::IoThreads => "Concurrent I/O operations",
            ResourceType::Cpu => "CPU core limit",
            ResourceType::Cache => "Maximum disk space for .cache/",
            ResourceType::Readahead => "Read-ahead buffer / prefetch window",
        }
    }

    /// Whether this resource supports percentage values (% of system RAM).
    pub fn supports_percentage(&self) -> bool {
        self.value_kind() == ValueKind::Memory
    }

    /// Look up a resource type by its canonical name.
    pub fn from_name(name: &str) -> Option<ResourceType> {
        match name {
            "mem" => Some(ResourceType::Mem),
            "threads" => Some(ResourceType::Threads),
            "segments" => Some(ResourceType::Segments),
            "segmentsize" => Some(ResourceType::SegmentSize),
            "iothreads" => Some(ResourceType::IoThreads),
            "cpu" => Some(ResourceType::Cpu),
            "cache" => Some(ResourceType::Cache),
            "readahead" => Some(ResourceType::Readahead),
            _ => None,
        }
    }

    /// All known resource types.
    pub fn all() -> &'static [ResourceType] {
        &[
            ResourceType::Mem,
            ResourceType::Threads,
            ResourceType::Segments,
            ResourceType::SegmentSize,
            ResourceType::IoThreads,
            ResourceType::Cpu,
            ResourceType::Cache,
            ResourceType::Readahead,
        ]
    }
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Resource description (per-command declarations)
// ---------------------------------------------------------------------------

/// Describes a resource type consumed by a command.
///
/// Returned by `CommandOp::describe_resources()` to declare which resources
/// the command uses and whether it can adjust them dynamically.
#[derive(Debug, Clone)]
pub struct ResourceDesc {
    /// Resource name (e.g., "mem", "threads", "segmentsize").
    /// Should match a [`ResourceType`] canonical name.
    pub name: String,
    /// Human-readable description of how this command uses the resource.
    pub description: String,
    /// Whether the command can dynamically adjust this resource mid-execution.
    pub adjustable: bool,
}

impl ResourceDesc {
    /// Create a `ResourceDesc` from a `ResourceType` with a custom description.
    pub fn new(resource_type: ResourceType, description: impl Into<String>, adjustable: bool) -> Self {
        ResourceDesc {
            name: resource_type.name().to_string(),
            description: description.into(),
            adjustable,
        }
    }
}

// ---------------------------------------------------------------------------
// Resource values and parsing
// ---------------------------------------------------------------------------

/// A parsed resource value — either a single point or a range.
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceValue {
    /// Fixed value (single number, e.g., `8` or `32GiB`).
    Fixed(u64),
    /// Range with floor and ceiling (e.g., `25%-50%` or `4-8`).
    Range { floor: u64, ceiling: u64 },
}

impl ResourceValue {
    /// The effective floor (minimum). For Fixed, this equals the value.
    pub fn floor(&self) -> u64 {
        match self {
            ResourceValue::Fixed(v) => *v,
            ResourceValue::Range { floor, .. } => *floor,
        }
    }

    /// The effective ceiling (maximum). For Fixed, this equals the value.
    pub fn ceiling(&self) -> u64 {
        match self {
            ResourceValue::Fixed(v) => *v,
            ResourceValue::Range { ceiling, .. } => *ceiling,
        }
    }

    /// The midpoint of the range. For Fixed, this equals the value.
    pub fn midpoint(&self) -> u64 {
        match self {
            ResourceValue::Fixed(v) => *v,
            ResourceValue::Range { floor, ceiling } => floor + (ceiling - floor) / 2,
        }
    }

    /// Whether this is a range (governor can adjust).
    pub fn is_range(&self) -> bool {
        matches!(self, ResourceValue::Range { .. })
    }
}

/// Parse a resource value string into a `ResourceValue`.
///
/// Supports:
/// - Integers: `8`, `1000000`
/// - Sizes with units: `32GB`, `32GiB`, `1024MB`, `64MiB`
/// - Percentages: `50%` (resolved against `system_total`)
/// - Ranges: `25%-50%`, `4-8`, `16GiB-48GiB`
pub fn parse_resource_value(input: &str, system_total: Option<u64>) -> Result<ResourceValue, String> {
    let input = input.trim();

    // Check for range (contains `-` that's not part of a suffix)
    if let Some((left, right)) = split_range(input) {
        let floor = parse_single_value(left, system_total)?;
        let ceiling = parse_single_value(right, system_total)?;
        if floor > ceiling {
            return Err(format!(
                "range floor ({}) exceeds ceiling ({})",
                floor, ceiling
            ));
        }
        Ok(ResourceValue::Range { floor, ceiling })
    } else {
        let val = parse_single_value(input, system_total)?;
        Ok(ResourceValue::Fixed(val))
    }
}

/// Split a range string like "25%-50%" or "4-8" into left and right parts.
/// Returns None if there's no range separator.
fn split_range(input: &str) -> Option<(&str, &str)> {
    // We need to find a '-' that separates two values.
    // Heuristic: find '-' that's preceded by a digit or '%' and followed by a digit.
    let bytes = input.as_bytes();
    for i in 1..bytes.len() {
        if bytes[i] == b'-' {
            let before = bytes[i - 1];
            let after = if i + 1 < bytes.len() {
                bytes[i + 1]
            } else {
                continue;
            };
            if (before.is_ascii_digit() || before == b'%' || before == b'B' || before == b'b')
                && after.is_ascii_digit()
            {
                return Some((&input[..i], &input[i + 1..]));
            }
        }
    }
    None
}

/// Parse a single (non-range) value string.
fn parse_single_value(input: &str, system_total: Option<u64>) -> Result<u64, String> {
    let input = input.trim();

    // Percentage
    if let Some(pct_str) = input.strip_suffix('%') {
        let pct: f64 = pct_str
            .parse()
            .map_err(|_| format!("invalid percentage: '{}'", input))?;
        if !(0.0..=100.0).contains(&pct) {
            return Err(format!("percentage out of range: {}%", pct));
        }
        let total = system_total.ok_or_else(|| {
            "percentage values require system total (e.g., for 'mem', total system RAM)".to_string()
        })?;
        Ok((total as f64 * pct / 100.0) as u64)
    } else {
        // Try size suffixes (IEC first, then SI, to avoid "GB" matching before "GiB")
        parse_size_with_units(input)
    }
}

/// Parse a value with optional size units.
fn parse_size_with_units(input: &str) -> Result<u64, String> {
    let suffixes: &[(&str, u64)] = &[
        ("TiB", 1_099_511_627_776),
        ("GiB", 1_073_741_824),
        ("MiB", 1_048_576),
        ("KiB", 1_024),
        ("TB", 1_000_000_000_000),
        ("GB", 1_000_000_000),
        ("MB", 1_000_000),
        ("KB", 1_000),
        ("B", 1),
    ];

    for (suffix, multiplier) in suffixes {
        if let Some(num_str) = input.strip_suffix(suffix) {
            let num: f64 = num_str
                .trim()
                .parse()
                .map_err(|_| format!("invalid number in '{}' (before '{}')", input, suffix))?;
            return Ok((num * *multiplier as f64) as u64);
        }
    }

    // Plain integer
    input
        .parse::<u64>()
        .map_err(|_| format!("invalid resource value: '{}'", input))
}

// ---------------------------------------------------------------------------
// Resource budget (parsed from --resources)
// ---------------------------------------------------------------------------

/// Parsed resource budget — the user's configured ranges for each resource.
#[derive(Debug, Clone, Default)]
pub struct ResourceBudget {
    /// Per-resource values (floor/ceiling pairs).
    pub resources: HashMap<String, ResourceValue>,
}

impl ResourceBudget {
    /// Create an empty budget.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a `--resources` string like `"mem:25%-50%,threads:4-8,segmentsize:500000"`.
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut resources = HashMap::new();
        let system_ram = get_system_ram();

        for pair in input.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let (name, value_str) = pair
                .split_once(':')
                .ok_or_else(|| format!("invalid resource spec '{}' (expected name:value)", pair))?;
            let name = name.trim();
            let value_str = value_str.trim();

            // Use ResourceType to determine if % is supported
            let resource_type = ResourceType::from_name(name);
            let system_total = match resource_type {
                Some(rt) if rt.supports_percentage() => system_ram,
                _ => None,
            };

            if resource_type.is_none() {
                log::warn!(
                    "Unknown resource type '{}'. Known types: {}",
                    name,
                    ResourceType::all().iter().map(|t| t.name()).collect::<Vec<_>>().join(", "),
                );
            }

            let value = parse_resource_value(value_str, system_total)?;
            resources.insert(name.to_string(), value);
        }

        Ok(ResourceBudget { resources })
    }

    /// Get a resource value, if configured.
    pub fn get(&self, name: &str) -> Option<&ResourceValue> {
        self.resources.get(name)
    }
}

/// Get total system RAM in bytes (Linux: from /proc/meminfo).
fn get_system_ram() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let rest = rest.trim();
                    if let Some(kb_str) = rest.strip_suffix("kB").or_else(|| rest.strip_suffix("KB")) {
                        if let Ok(kb) = kb_str.trim().parse::<u64>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// System snapshot (telemetry data)
// ---------------------------------------------------------------------------

/// Point-in-time snapshot of system resource usage.
#[derive(Debug, Clone, Serialize)]
pub struct SystemSnapshot {
    /// Resident set size in bytes.
    pub rss_bytes: u64,
    /// RSS as percentage of system RAM.
    pub rss_pct: f64,
    /// CPU user time percentage (0-100 per core × num cores).
    pub cpu_user_pct: f64,
    /// CPU system time percentage.
    pub cpu_system_pct: f64,
    /// Read throughput in bytes/sec.
    pub io_read_bps: u64,
    /// Write throughput in bytes/sec.
    pub io_write_bps: u64,
    /// Major page faults since last snapshot.
    pub major_faults: u64,
    /// Minor page faults since last snapshot.
    pub minor_faults: u64,
    /// Current active thread count.
    pub active_threads: usize,
    /// Raw cumulative CPU user time in clock ticks (for delta computation).
    #[serde(skip)]
    pub cpu_user_ticks: u64,
    /// Raw cumulative CPU system time in clock ticks (for delta computation).
    #[serde(skip)]
    pub cpu_system_ticks: u64,
    /// Clock ticks per second (for converting ticks to seconds).
    #[serde(skip)]
    pub ticks_per_sec: f64,
}

impl SystemSnapshot {
    /// Sample current system state (Linux: from /proc/self/stat and /proc/self/io).
    pub fn sample() -> Self {
        let (rss_bytes, major_faults, minor_faults, utime, stime) = read_proc_self_stat_full();
        let system_ram = get_system_ram().unwrap_or(1);
        let rss_pct = (rss_bytes as f64 / system_ram as f64) * 100.0;
        let (io_read, io_write) = read_proc_self_io();
        let active_threads = read_proc_self_threads();

        // Convert utime/stime (in clock ticks) to percentage (rough estimate).
        // This is cumulative, not per-interval, so it's a running average.
        let ticks_per_sec = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get() as f64)
            .unwrap_or(1.0);
        let uptime_secs = read_proc_uptime();
        let (cpu_user_pct, cpu_system_pct) = if uptime_secs > 0.0 && ticks_per_sec > 0.0 {
            let user_secs = utime as f64 / ticks_per_sec;
            let sys_secs = stime as f64 / ticks_per_sec;
            let user_pct = (user_secs / uptime_secs / num_cpus) * 100.0;
            let sys_pct = (sys_secs / uptime_secs / num_cpus) * 100.0;
            (user_pct.min(100.0), sys_pct.min(100.0))
        } else {
            (0.0, 0.0)
        };

        SystemSnapshot {
            rss_bytes,
            rss_pct,
            cpu_user_pct,
            cpu_system_pct,
            io_read_bps: io_read,
            io_write_bps: io_write,
            major_faults,
            minor_faults,
            active_threads,
            cpu_user_ticks: utime,
            cpu_system_ticks: stime,
            ticks_per_sec,
        }
    }
}

/// Read RSS, page fault counts, and CPU times from /proc/self/stat.
///
/// Returns (rss_bytes, major_faults, minor_faults, utime_ticks, stime_ticks).
fn read_proc_self_stat_full() -> (u64, u64, u64, u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/stat") {
            // Fields are space-separated; RSS is field 24 (0-indexed: 23) in pages.
            // minflt is field 10 (idx 9), majflt is field 12 (idx 11).
            // utime is field 14 (idx 13), stime is field 15 (idx 14).
            let fields: Vec<&str> = content.split_whitespace().collect();
            if fields.len() > 23 {
                let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
                let rss_pages: u64 = fields[23].parse().unwrap_or(0);
                let minflt: u64 = fields[9].parse().unwrap_or(0);
                let majflt: u64 = fields[11].parse().unwrap_or(0);
                let utime: u64 = fields[13].parse().unwrap_or(0);
                let stime: u64 = fields[14].parse().unwrap_or(0);
                return (rss_pages * page_size, majflt, minflt, utime, stime);
            }
        }
        (0, 0, 0, 0, 0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        (0, 0, 0, 0, 0)
    }
}

/// Read cumulative I/O bytes from /proc/self/io.
///
/// Returns (read_bytes, write_bytes). These are cumulative totals, not rates.
fn read_proc_self_io() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/io") {
            let mut read_bytes = 0u64;
            let mut write_bytes = 0u64;
            for line in content.lines() {
                if let Some(val) = line.strip_prefix("read_bytes: ") {
                    read_bytes = val.trim().parse().unwrap_or(0);
                } else if let Some(val) = line.strip_prefix("write_bytes: ") {
                    write_bytes = val.trim().parse().unwrap_or(0);
                }
            }
            return (read_bytes, write_bytes);
        }
        (0, 0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        (0, 0)
    }
}

/// Read active thread count from /proc/self/status.
fn read_proc_self_threads() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                if let Some(val) = line.strip_prefix("Threads:") {
                    return val.trim().parse().unwrap_or(1);
                }
            }
        }
        1
    }
    #[cfg(not(target_os = "linux"))]
    {
        1
    }
}

/// Read system uptime in seconds from /proc/uptime.
fn read_proc_uptime() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/uptime") {
            if let Some(first) = content.split_whitespace().next() {
                return first.parse().unwrap_or(0.0);
            }
        }
        0.0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Governor strategy
// ---------------------------------------------------------------------------

/// Adjustments produced by a governor strategy evaluation.
#[derive(Debug, Clone, Default)]
pub struct ResourceAdjustments {
    /// Resource name → new effective value.
    pub updates: HashMap<String, u64>,
    /// Whether commands should throttle (reduce consumption).
    pub throttle: bool,
    /// Whether this is an emergency (commands must flush immediately).
    pub emergency: bool,
}

/// Pluggable strategy for resource adjustment decisions.
pub trait GovernorStrategy: Send + Sync {
    /// Evaluate current system state and return adjustments.
    fn evaluate(
        &mut self,
        snapshot: &SystemSnapshot,
        budget: &ResourceBudget,
        current: &HashMap<String, u64>,
    ) -> ResourceAdjustments;

    /// Strategy name for logging.
    fn name(&self) -> &str;
}

/// Default strategy that maximizes system utilization without saturating.
pub struct MaximizeUtilizationStrategy {
    /// Number of consecutive stable evaluations.
    stable_count: u32,
    /// Damping factor (increases when adjustments worsen metrics).
    damping: f64,
    /// Last RSS percentage for trend detection.
    last_rss_pct: f64,
}

impl Default for MaximizeUtilizationStrategy {
    fn default() -> Self {
        Self {
            stable_count: 0,
            damping: 1.0,
            last_rss_pct: 0.0,
        }
    }
}

// Default utilization targets (from SRD §06)
const MEM_TARGET_LOW: f64 = 70.0;
const MEM_TARGET_HIGH: f64 = 85.0;
const MEM_THROTTLE: f64 = 90.0;
const MEM_EMERGENCY: f64 = 95.0;
const MAX_ADJUSTMENT_PCT: f64 = 0.25;
const MAX_DAMPING: f64 = 4.0;
const MIN_DAMPING: f64 = 1.0;
const DAMPING_INCREASE: f64 = 1.5;
const DAMPING_DECREASE: f64 = 0.75;
const STABLE_WINDOW: u32 = 3;

impl GovernorStrategy for MaximizeUtilizationStrategy {
    fn name(&self) -> &str {
        "maximize"
    }

    fn evaluate(
        &mut self,
        snapshot: &SystemSnapshot,
        budget: &ResourceBudget,
        current: &HashMap<String, u64>,
    ) -> ResourceAdjustments {
        let mut adjustments = ResourceAdjustments::default();
        let rss_pct = snapshot.rss_pct;

        // Determine operating band
        if rss_pct >= MEM_EMERGENCY {
            // EMERGENCY: flush everything
            adjustments.throttle = true;
            adjustments.emergency = true;
            self.stable_count = 0;
            self.damping = (self.damping * DAMPING_INCREASE).min(MAX_DAMPING);

            // Reduce all adjustable resources to floor
            for (name, value) in &budget.resources {
                if value.is_range() {
                    adjustments.updates.insert(name.clone(), value.floor());
                }
            }
        } else if rss_pct >= MEM_THROTTLE {
            // THROTTLE: signal commands to reduce consumption
            adjustments.throttle = true;
            self.stable_count = 0;
            self.damping = (self.damping * DAMPING_INCREASE).min(MAX_DAMPING);

            // Scale down toward floor
            apply_scale_down(budget, current, &mut adjustments.updates, self.damping);
        } else if rss_pct >= MEM_TARGET_HIGH {
            // CAUTION: begin scaling down
            self.stable_count = 0;

            // Gentle scale-down
            apply_scale_down(budget, current, &mut adjustments.updates, self.damping);
        } else if rss_pct >= MEM_TARGET_LOW {
            // NOMINAL: no adjustments, this is the target
            self.stable_count = self.stable_count.saturating_add(1);
            self.damping = (self.damping * DAMPING_DECREASE).max(MIN_DAMPING);
        } else {
            // UNDERUSED: scale up if stable
            self.stable_count = self.stable_count.saturating_add(1);
            self.damping = (self.damping * DAMPING_DECREASE).max(MIN_DAMPING);

            if self.stable_count >= STABLE_WINDOW {
                apply_scale_up(budget, current, &mut adjustments.updates, self.damping);
            }
        }

        // Track RSS trend for damping feedback
        if rss_pct > self.last_rss_pct + 5.0 && !adjustments.updates.is_empty() {
            // RSS got worse after our last adjustment — increase damping
            self.damping = (self.damping * DAMPING_INCREASE).min(MAX_DAMPING);
        }
        self.last_rss_pct = rss_pct;

        adjustments
    }
}

/// Scale adjustable resources down toward their floor.
fn apply_scale_down(
    budget: &ResourceBudget,
    current: &HashMap<String, u64>,
    updates: &mut HashMap<String, u64>,
    damping: f64,
) {
    for (name, value) in &budget.resources {
        if !value.is_range() {
            continue;
        }
        if let Some(&cur) = current.get(name) {
            let floor = value.floor();
            if cur > floor {
                let step = ((cur as f64 * MAX_ADJUSTMENT_PCT) / damping) as u64;
                let new_val = cur.saturating_sub(step.max(1)).max(floor);
                if new_val != cur {
                    updates.insert(name.clone(), new_val);
                }
            }
        }
    }
}

/// Scale adjustable resources up toward their ceiling.
fn apply_scale_up(
    budget: &ResourceBudget,
    current: &HashMap<String, u64>,
    updates: &mut HashMap<String, u64>,
    damping: f64,
) {
    for (name, value) in &budget.resources {
        if !value.is_range() {
            continue;
        }
        if let Some(&cur) = current.get(name) {
            let ceiling = value.ceiling();
            if cur < ceiling {
                let step = ((cur as f64 * MAX_ADJUSTMENT_PCT) / damping) as u64;
                let new_val = (cur + step.max(1)).min(ceiling);
                if new_val != cur {
                    updates.insert(name.clone(), new_val);
                }
            }
        }
    }
}

/// Conservative strategy: starts at floor, only increases after sustained stability.
pub struct ConservativeStrategy {
    stable_count: u32,
}

impl Default for ConservativeStrategy {
    fn default() -> Self {
        Self { stable_count: 0 }
    }
}

impl GovernorStrategy for ConservativeStrategy {
    fn name(&self) -> &str {
        "conservative"
    }

    fn evaluate(
        &mut self,
        snapshot: &SystemSnapshot,
        budget: &ResourceBudget,
        current: &HashMap<String, u64>,
    ) -> ResourceAdjustments {
        let mut adjustments = ResourceAdjustments::default();
        let rss_pct = snapshot.rss_pct;

        if rss_pct >= MEM_EMERGENCY {
            adjustments.throttle = true;
            adjustments.emergency = true;
            self.stable_count = 0;
            for (name, value) in &budget.resources {
                if value.is_range() {
                    adjustments.updates.insert(name.clone(), value.floor());
                }
            }
        } else if rss_pct >= MEM_THROTTLE {
            adjustments.throttle = true;
            self.stable_count = 0;
            apply_scale_down(budget, current, &mut adjustments.updates, 1.0);
        } else if rss_pct < MEM_TARGET_LOW {
            self.stable_count = self.stable_count.saturating_add(1);
            // Conservative: require 2× the stable window before scaling up
            if self.stable_count >= STABLE_WINDOW * 2 {
                apply_scale_up(budget, current, &mut adjustments.updates, 2.0);
            }
        } else {
            self.stable_count = self.stable_count.saturating_add(1);
        }

        adjustments
    }
}

/// Fixed strategy: uses midpoint values, never adjusts. Useful for benchmarking.
pub struct FixedStrategy;

impl GovernorStrategy for FixedStrategy {
    fn name(&self) -> &str {
        "fixed"
    }

    fn evaluate(
        &mut self,
        _snapshot: &SystemSnapshot,
        _budget: &ResourceBudget,
        _current: &HashMap<String, u64>,
    ) -> ResourceAdjustments {
        ResourceAdjustments::default()
    }
}

// ---------------------------------------------------------------------------
// Governor log
// ---------------------------------------------------------------------------

/// JSON log entry types for the governor utilization log.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum GovernorLogEntry {
    Observation {
        ts: String,
        step_id: String,
        rss_bytes: u64,
        rss_pct: f64,
        cpu_user_pct: f64,
        cpu_system_pct: f64,
        major_faults: u64,
        minor_faults: u64,
    },
    Decision {
        ts: String,
        step_id: String,
        resource: String,
        old_value: u64,
        new_value: u64,
        reason: String,
    },
    Throttle {
        ts: String,
        step_id: String,
        reason: String,
        emergency: bool,
    },
    Request {
        ts: String,
        step_id: String,
        resource: String,
        requested: u64,
        granted: u64,
    },
    Ignored {
        ts: String,
        step_id: String,
        resource: String,
        reason: String,
    },
}

/// Writer for the governor utilization log.
struct GovernorLog {
    file: Option<std::fs::File>,
}

impl GovernorLog {
    /// Create a new log file at the given path (overwrites existing).
    fn new(path: &Path) -> Self {
        let file = std::fs::File::create(path)
            .map_err(|e| log::warn!("Cannot create governor log: {}", e))
            .ok();
        GovernorLog { file }
    }

    /// Create a no-op log (when no workspace is available).
    fn noop() -> Self {
        GovernorLog { file: None }
    }

    fn write_entry(&mut self, entry: &GovernorLogEntry) {
        if let Some(ref mut file) = self.file {
            if let Ok(json) = serde_json::to_string(entry) {
                let _ = writeln!(file, "{}", json);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceGovernor
// ---------------------------------------------------------------------------

/// Runtime resource governor.
///
/// The governor is the sole entity permitted to adjust effective resource
/// values at runtime (REQ-RM-12). Commands hold read-only views of the
/// effective values and interact via `current()`, `checkpoint()`,
/// `should_throttle()`, and `request()`.
pub struct ResourceGovernor {
    /// User-configured budget (floor/ceiling per resource).
    budget: ResourceBudget,
    /// Current effective values (written only by the governor).
    effective: Arc<RwLock<HashMap<String, u64>>>,
    /// Throttle flag (set by governor, read by commands).
    throttle: Arc<AtomicBool>,
    /// Emergency flag.
    emergency: Arc<AtomicBool>,
    /// Governor strategy.
    strategy: Mutex<Box<dyn GovernorStrategy>>,
    /// Governor log.
    log: Mutex<GovernorLog>,
    /// Current step ID for logging.
    step_id: RwLock<String>,
    /// Last evaluation time.
    last_eval: Mutex<Instant>,
    /// Evaluation interval.
    eval_interval: Duration,
    /// Stop flag for the background thread.
    stop_flag: Arc<AtomicBool>,
    /// Background evaluation thread handle.
    bg_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl ResourceGovernor {
    /// Create a new governor from a parsed budget.
    ///
    /// Merges system defaults for any resource types not explicitly specified:
    /// - `mem`: 80% of system RAM
    /// - `threads`: available CPU parallelism
    /// - `segmentsize`: 1,000,000 records
    pub fn new(mut budget: ResourceBudget, workspace: Option<&Path>) -> Self {
        // Backfill defaults for unspecified resources
        if !budget.resources.contains_key("mem") {
            if let Some(ram) = get_system_ram() {
                budget.resources.insert(
                    "mem".to_string(),
                    ResourceValue::Fixed((ram as f64 * 0.8) as u64),
                );
            }
        }
        if !budget.resources.contains_key("threads") {
            let cpus = std::thread::available_parallelism()
                .map(|n| n.get() as u64)
                .unwrap_or(4);
            budget.resources.insert("threads".to_string(), ResourceValue::Fixed(cpus));
        }
        if !budget.resources.contains_key("segmentsize") {
            budget.resources.insert("segmentsize".to_string(), ResourceValue::Fixed(1_000_000));
        }

        // Initialize effective values at midpoints
        let mut effective = HashMap::new();
        for (name, value) in &budget.resources {
            effective.insert(name.clone(), value.midpoint());
        }

        let log = if let Some(ws) = workspace {
            GovernorLog::new(&ws.join(".governor.log"))
        } else {
            GovernorLog::noop()
        };

        let has_mem_budget = budget.get("mem").is_some();
        let governor = ResourceGovernor {
            budget,
            effective: Arc::new(RwLock::new(effective)),
            throttle: Arc::new(AtomicBool::new(false)),
            emergency: Arc::new(AtomicBool::new(false)),
            strategy: Mutex::new(Box::new(MaximizeUtilizationStrategy::default())),
            log: Mutex::new(log),
            step_id: RwLock::new(String::new()),
            last_eval: Mutex::new(Instant::now()),
            eval_interval: Duration::from_millis(500),
            stop_flag: Arc::new(AtomicBool::new(false)),
            bg_thread: Mutex::new(None),
        };

        // Start background evaluation thread if a memory budget is configured.
        // The thread periodically samples RSS and adjusts effective values,
        // ensuring the governor reacts between command checkpoint() calls.
        if has_mem_budget {
            governor.start_background_thread();
        }

        governor
    }

    /// Start the background governor evaluation thread (REQ-RM-12).
    ///
    /// The thread runs at twice the eval_interval, sampling system state
    /// and publishing throttle/emergency signals. It stops when the
    /// governor is dropped or `stop_background_thread()` is called.
    fn start_background_thread(&self) {
        let effective = Arc::clone(&self.effective);
        let throttle = Arc::clone(&self.throttle);
        let emergency = Arc::clone(&self.emergency);
        let stop = Arc::clone(&self.stop_flag);
        let interval = self.eval_interval;

        // Read budget limits we need for the background thread
        let mem_ceiling = self.budget.get("mem").map(|v| v.ceiling()).unwrap_or(0);
        let mem_floor = self.budget.get("mem").map(|v| v.floor()).unwrap_or(0);

        let handle = std::thread::Builder::new()
            .name("governor-bg".into())
            .spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(interval);
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }

                    let snapshot = SystemSnapshot::sample();

                    // Quick RSS-based throttle/emergency check
                    if mem_ceiling > 0 {
                        let rss_pct_of_ceiling =
                            (snapshot.rss_bytes as f64 / mem_ceiling as f64) * 100.0;

                        if rss_pct_of_ceiling >= MEM_EMERGENCY {
                            emergency.store(true, Ordering::Relaxed);
                            throttle.store(true, Ordering::Relaxed);
                        } else if rss_pct_of_ceiling >= MEM_THROTTLE {
                            emergency.store(false, Ordering::Relaxed);
                            throttle.store(true, Ordering::Relaxed);
                        } else if rss_pct_of_ceiling < MEM_TARGET_HIGH {
                            // Only clear if we're well below the threshold
                            throttle.store(false, Ordering::Relaxed);
                            emergency.store(false, Ordering::Relaxed);
                        }

                        // Adjust effective mem value based on actual RSS
                        if let Ok(mut eff) = effective.write() {
                            if rss_pct_of_ceiling >= MEM_TARGET_HIGH {
                                // Scale down effective mem toward floor
                                let current = eff.get("mem").copied().unwrap_or(mem_ceiling);
                                let reduced = current
                                    .saturating_sub(
                                        (mem_ceiling - mem_floor) / 10
                                    )
                                    .max(mem_floor);
                                eff.insert("mem".to_string(), reduced);
                            } else if rss_pct_of_ceiling < MEM_TARGET_LOW {
                                // Scale up toward ceiling
                                let current = eff.get("mem").copied().unwrap_or(mem_floor);
                                let increased = (current + (mem_ceiling - mem_floor) / 10)
                                    .min(mem_ceiling);
                                eff.insert("mem".to_string(), increased);
                            }
                        }
                    }
                }
            })
            .ok();

        *self.bg_thread.lock().unwrap() = handle;
    }

    /// Create a no-op governor with default values (for when --resources is not specified).
    ///
    /// Unlike `new()`, this does NOT start a background evaluation thread
    /// since no explicit resource limits were requested.
    pub fn default_governor() -> Self {
        let mut budget = ResourceBudget::new();

        // Default mem: 80% of system RAM
        if let Some(ram) = get_system_ram() {
            budget
                .resources
                .insert("mem".to_string(), ResourceValue::Fixed((ram as f64 * 0.8) as u64));
        }

        // Default threads: num_cpus
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u64)
            .unwrap_or(4);
        budget
            .resources
            .insert("threads".to_string(), ResourceValue::Fixed(cpus));

        // Default segmentsize: 1M
        budget
            .resources
            .insert("segmentsize".to_string(), ResourceValue::Fixed(1_000_000));

        let mut effective = HashMap::new();
        for (name, value) in &budget.resources {
            effective.insert(name.clone(), value.midpoint());
        }

        ResourceGovernor {
            budget,
            effective: Arc::new(RwLock::new(effective)),
            throttle: Arc::new(AtomicBool::new(false)),
            emergency: Arc::new(AtomicBool::new(false)),
            strategy: Mutex::new(Box::new(MaximizeUtilizationStrategy::default())),
            log: Mutex::new(GovernorLog::noop()),
            step_id: RwLock::new(String::new()),
            last_eval: Mutex::new(Instant::now()),
            eval_interval: Duration::from_millis(500),
            stop_flag: Arc::new(AtomicBool::new(false)),
            bg_thread: Mutex::new(None),
        }
    }

    /// Set the governor strategy.
    pub fn set_strategy(&self, strategy: Box<dyn GovernorStrategy>) {
        *self.strategy.lock().unwrap() = strategy;
    }

    /// Return the names of all resources in the configured budget.
    pub fn budget_resource_names(&self) -> Vec<String> {
        self.budget.resources.keys().cloned().collect()
    }

    /// Set the current step ID (called by the runner before each step).
    pub fn set_step_id(&self, step_id: &str) {
        *self.step_id.write().unwrap() = step_id.to_string();
    }

    /// Get the current effective value for a resource.
    ///
    /// Returns the effective value if the resource is configured, or None.
    /// Commands MUST use this instead of hardcoded defaults.
    pub fn current(&self, name: &str) -> Option<u64> {
        self.effective.read().unwrap().get(name).copied()
    }

    /// Get the current effective value, or a default if not configured.
    pub fn current_or(&self, name: &str, default: u64) -> u64 {
        self.current(name).unwrap_or(default)
    }

    /// Request a specific value for a resource.
    ///
    /// The governor evaluates the request against the current budget and
    /// system state, then returns the granted value (which may differ).
    pub fn request(&self, name: &str, requested: u64) -> u64 {
        let granted = if let Some(value) = self.budget.get(name) {
            // Clamp to configured range
            requested.max(value.floor()).min(value.ceiling())
        } else {
            requested
        };

        let step_id = self.step_id.read().unwrap().clone();
        if let Ok(mut log) = self.log.lock() {
            log.write_entry(&GovernorLogEntry::Request {
                ts: chrono::Utc::now().to_rfc3339(),
                step_id,
                resource: name.to_string(),
                requested,
                granted,
            });
        }

        granted
    }

    /// Log that a resource limit was ignored because the command does not
    /// declare the resource type.
    ///
    /// Called by the runner when a budget resource is not in the command's
    /// `describe_resources()`. The limit is silently ignored at runtime,
    /// but recorded as a debug entry in the governor log.
    pub fn log_ignored(&self, command_path: &str, resource: &str) {
        let step_id = self.step_id.read().unwrap().clone();
        if let Ok(mut log) = self.log.lock() {
            log.write_entry(&GovernorLogEntry::Ignored {
                ts: chrono::Utc::now().to_rfc3339(),
                step_id,
                resource: resource.to_string(),
                reason: format!(
                    "command '{}' does not declare resource '{}'; limit ignored",
                    command_path, resource,
                ),
            });
        }
    }

    /// Checkpoint: commands call this at segment/partition boundaries.
    ///
    /// Triggers a governor evaluation if enough time has passed since
    /// the last one. Returns the current throttle state.
    pub fn checkpoint(&self) -> bool {
        let should_eval = {
            let last = self.last_eval.lock().unwrap();
            last.elapsed() >= self.eval_interval
        };

        if should_eval {
            self.evaluate();
        }

        self.throttle.load(Ordering::Relaxed)
    }

    /// Whether commands should throttle (reduce resource consumption).
    pub fn should_throttle(&self) -> bool {
        self.throttle.load(Ordering::Relaxed)
    }

    /// Whether this is an emergency (commands must flush immediately).
    pub fn is_emergency(&self) -> bool {
        self.emergency.load(Ordering::Relaxed)
    }

    /// Get the configured memory ceiling in bytes, if a `mem` budget is set.
    pub fn mem_ceiling(&self) -> Option<u64> {
        self.budget.get("mem").map(|v| v.ceiling())
    }

    /// Run a governor evaluation cycle.
    ///
    /// Samples system state, runs the strategy, applies adjustments,
    /// and logs decisions.
    fn evaluate(&self) {
        let snapshot = SystemSnapshot::sample();
        let step_id = self.step_id.read().unwrap().clone();

        // Log observation
        if let Ok(mut log) = self.log.lock() {
            log.write_entry(&GovernorLogEntry::Observation {
                ts: chrono::Utc::now().to_rfc3339(),
                step_id: step_id.clone(),
                rss_bytes: snapshot.rss_bytes,
                rss_pct: snapshot.rss_pct,
                cpu_user_pct: snapshot.cpu_user_pct,
                cpu_system_pct: snapshot.cpu_system_pct,
                major_faults: snapshot.major_faults,
                minor_faults: snapshot.minor_faults,
            });
        }

        // Run strategy
        let current = self.effective.read().unwrap().clone();
        let adjustments = {
            let mut strategy = self.strategy.lock().unwrap();
            strategy.evaluate(&snapshot, &self.budget, &current)
        };

        // Apply adjustments
        if !adjustments.updates.is_empty() || adjustments.throttle || adjustments.emergency {
            let mut effective = self.effective.write().unwrap();
            let mut log = self.log.lock().unwrap();

            for (name, new_value) in &adjustments.updates {
                let old_value = effective.get(name).copied().unwrap_or(0);
                if old_value != *new_value {
                    effective.insert(name.clone(), *new_value);

                    let reason = if snapshot.rss_pct >= MEM_EMERGENCY {
                        format!("RSS at {:.1}% of ceiling; emergency flush", snapshot.rss_pct)
                    } else if snapshot.rss_pct >= MEM_THROTTLE {
                        format!("RSS at {:.1}% of ceiling; throttling", snapshot.rss_pct)
                    } else if snapshot.rss_pct >= MEM_TARGET_HIGH {
                        format!("RSS at {:.1}% of ceiling; reducing", snapshot.rss_pct)
                    } else {
                        format!("RSS at {:.1}%; scaling up", snapshot.rss_pct)
                    };

                    log.write_entry(&GovernorLogEntry::Decision {
                        ts: chrono::Utc::now().to_rfc3339(),
                        step_id: step_id.clone(),
                        resource: name.clone(),
                        old_value,
                        new_value: *new_value,
                        reason,
                    });
                }
            }

            self.throttle.store(adjustments.throttle, Ordering::Relaxed);
            self.emergency.store(adjustments.emergency, Ordering::Relaxed);

            if adjustments.throttle {
                log.write_entry(&GovernorLogEntry::Throttle {
                    ts: chrono::Utc::now().to_rfc3339(),
                    step_id: step_id.clone(),
                    reason: if adjustments.emergency {
                        format!(
                            "RSS at {:.1}% — emergency flush requested",
                            snapshot.rss_pct
                        )
                    } else {
                        format!(
                            "RSS at {:.1}% — throttle requested",
                            snapshot.rss_pct
                        )
                    },
                    emergency: adjustments.emergency,
                });
            }
        } else {
            // No throttle, clear flags
            self.throttle.store(false, Ordering::Relaxed);
            self.emergency.store(false, Ordering::Relaxed);
        }

        *self.last_eval.lock().unwrap() = Instant::now();
    }

    /// Whether this governor has a user-specified budget (not just defaults).
    pub fn has_explicit_budget(&self) -> bool {
        !self.budget.resources.is_empty()
    }

    /// Create a thread-safe status source for the resource status bar.
    pub fn status_source(&self) -> ResourceStatusSource {
        ResourceStatusSource {
            budget: self.budget.resources.clone(),
            effective: Arc::clone(&self.effective),
            throttle: Arc::clone(&self.throttle),
            emergency: Arc::clone(&self.emergency),
            prev_cpu: Arc::new(Mutex::new(None)),
        }
    }

    /// Render a one-line status string showing current resource utilization.
    ///
    /// Delegates to [`ResourceStatusSource::status_line`] for consistent
    /// formatting.
    pub fn status_line(&self) -> String {
        self.status_source().status_line()
    }

    /// Print a summary of the current effective resource values to stdout.
    pub fn print_summary(&self) {
        let effective = self.effective.read().unwrap();
        if effective.is_empty() {
            return;
        }
        let strategy_name = self.strategy.lock().unwrap().name().to_string();
        println!("Resource governor: strategy={}", strategy_name);
        let mut entries: Vec<_> = effective.iter().collect();
        entries.sort_by_key(|(k, _)| (*k).clone());
        for (name, value) in entries {
            let budget_info = if let Some(bv) = self.budget.get(name) {
                if bv.is_range() {
                    format!(" (range: {}-{})", bv.floor(), bv.ceiling())
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            println!("  {}: {}{}", name, format_value(name, *value), budget_info);
        }
        println!();
    }
}

/// Lightweight, thread-safe handle for reading resource status.
///
/// Created via [`ResourceGovernor::status_source`]. The source shares the
/// governor's internal state through `Arc` references, so reads are always
/// up-to-date without requiring `&ResourceGovernor` to be `Send`.
#[derive(Clone)]
pub struct ResourceStatusSource {
    budget: HashMap<String, ResourceValue>,
    effective: Arc<RwLock<HashMap<String, u64>>>,
    throttle: Arc<AtomicBool>,
    emergency: Arc<AtomicBool>,
    /// Previous CPU tick count and sample time for instantaneous CPU % computation.
    prev_cpu: Arc<Mutex<Option<(u64, Instant)>>>,
}

impl ResourceStatusSource {
    /// Render a one-line status string showing current resource utilization.
    pub fn status_line(&self) -> String {
        let snapshot = SystemSnapshot::sample();
        let effective = self.effective.read().unwrap();
        let mut parts: Vec<String> = Vec::new();

        // Always show RSS
        if let Some(mem_budget) = self.budget.get("mem") {
            let ceiling = mem_budget.ceiling();
            parts.push(format!("rss: {}/{}", format_value("mem", snapshot.rss_bytes), format_value("mem", ceiling)));
        } else {
            parts.push(format!("rss: {}", format_value("mem", snapshot.rss_bytes)));
        }

        // CPU usage — instantaneous % from tick deltas between samples
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let total_ticks = snapshot.cpu_user_ticks + snapshot.cpu_system_ticks;
        let cpu_pct = {
            let mut prev = self.prev_cpu.lock().unwrap();
            let pct = if let Some((prev_ticks, prev_time)) = *prev {
                let dt = prev_time.elapsed().as_secs_f64();
                if dt > 0.0 && snapshot.ticks_per_sec > 0.0 {
                    let delta_ticks = total_ticks.saturating_sub(prev_ticks) as f64;
                    let delta_secs = delta_ticks / snapshot.ticks_per_sec;
                    (delta_secs / dt) * 100.0 // percentage across all cores
                } else {
                    0.0
                }
            } else {
                0.0
            };
            *prev = Some((total_ticks, Instant::now()));
            pct
        };
        parts.push(format!("cpu: {:.0}%/{}", cpu_pct, num_cpus * 100));

        // Show thread budget (effective/ceiling), not process thread count
        if let Some(thread_budget) = self.budget.get("threads") {
            let eff = effective.get("threads").copied().unwrap_or(0);
            parts.push(format!("threads: {}/{}", eff, thread_budget.ceiling()));
        } else {
            parts.push(format!("threads: {}", snapshot.active_threads));
        }

        // I/O throughput (cumulative bytes — UI computes rates from deltas)
        parts.push(format!("io_r: {}", format_value("mem", snapshot.io_read_bps)));
        parts.push(format!("io_w: {}", format_value("mem", snapshot.io_write_bps)));

        // Page faults (cumulative counts)
        parts.push(format!("pgfault: {}/{}", snapshot.major_faults, snapshot.minor_faults));

        // Show remaining budgeted resources (skip mem/threads already shown)
        let mut names: Vec<&String> = self.budget.keys()
            .filter(|n| *n != "mem" && *n != "threads")
            .collect();
        names.sort();

        for name in &names {
            let eff = effective.get(*name).copied().unwrap_or(0);
            let ceiling = self.budget.get(*name).map(|v| v.ceiling()).unwrap_or(eff);
            parts.push(format!("{}: {}/{}", name, format_value(name, eff), format_value(name, ceiling)));
        }

        // Throttle/emergency indicator
        if self.emergency.load(Ordering::Relaxed) {
            parts.push("⚠ EMERGENCY".to_string());
        } else if self.throttle.load(Ordering::Relaxed) {
            parts.push("⏳ throttle".to_string());
        }

        parts.join(" | ")
    }

    /// Whether the governor is in emergency state.
    pub fn is_emergency(&self) -> bool {
        self.emergency.load(Ordering::Relaxed)
    }

    /// Return how many percentage points RSS exceeds the memory ceiling.
    ///
    /// Returns 0.0 if RSS is at or below the ceiling, or if no memory
    /// budget is configured.
    pub fn rss_overage_pct(&self) -> f64 {
        if let Some(mem_budget) = self.budget.get("mem") {
            let ceiling = mem_budget.ceiling();
            if ceiling > 0 {
                let snapshot = SystemSnapshot::sample();
                let pct = (snapshot.rss_bytes as f64 / ceiling as f64) * 100.0;
                (pct - 100.0).max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

impl Drop for ResourceGovernor {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.bg_thread.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

/// Format a resource value with appropriate units for display.
fn format_value(name: &str, value: u64) -> String {
    let is_memory = ResourceType::from_name(name)
        .map(|rt| rt.value_kind() == ValueKind::Memory)
        .unwrap_or(false);
    if is_memory {
        if value >= 1_073_741_824 {
            format!("{:.1} GiB", value as f64 / 1_073_741_824.0)
        } else if value >= 1_048_576 {
            format!("{:.1} MiB", value as f64 / 1_048_576.0)
        } else if value >= 1024 {
            format!("{:.1} KiB", value as f64 / 1024.0)
        } else {
            format!("{} B", value)
        }
    } else {
        value.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plain_integers() {
        assert_eq!(
            parse_resource_value("8", None).unwrap(),
            ResourceValue::Fixed(8)
        );
        assert_eq!(
            parse_resource_value("1000000", None).unwrap(),
            ResourceValue::Fixed(1_000_000)
        );
    }

    #[test]
    fn test_parse_size_units() {
        assert_eq!(
            parse_resource_value("32GiB", None).unwrap(),
            ResourceValue::Fixed(32 * 1_073_741_824)
        );
        assert_eq!(
            parse_resource_value("32GB", None).unwrap(),
            ResourceValue::Fixed(32_000_000_000)
        );
        assert_eq!(
            parse_resource_value("1024MB", None).unwrap(),
            ResourceValue::Fixed(1_024_000_000)
        );
        assert_eq!(
            parse_resource_value("64MiB", None).unwrap(),
            ResourceValue::Fixed(64 * 1_048_576)
        );
    }

    #[test]
    fn test_parse_percentages() {
        let total = Some(100_000_000_000u64); // 100 GB
        assert_eq!(
            parse_resource_value("50%", total).unwrap(),
            ResourceValue::Fixed(50_000_000_000)
        );
        assert_eq!(
            parse_resource_value("80%", total).unwrap(),
            ResourceValue::Fixed(80_000_000_000)
        );
    }

    #[test]
    fn test_parse_ranges() {
        assert_eq!(
            parse_resource_value("4-8", None).unwrap(),
            ResourceValue::Range {
                floor: 4,
                ceiling: 8
            }
        );
        assert_eq!(
            parse_resource_value("16GiB-48GiB", None).unwrap(),
            ResourceValue::Range {
                floor: 16 * 1_073_741_824,
                ceiling: 48 * 1_073_741_824,
            }
        );
        let total = Some(100_000_000_000u64);
        assert_eq!(
            parse_resource_value("25%-50%", total).unwrap(),
            ResourceValue::Range {
                floor: 25_000_000_000,
                ceiling: 50_000_000_000,
            }
        );
    }

    #[test]
    fn test_parse_range_floor_exceeds_ceiling() {
        assert!(parse_resource_value("10-5", None).is_err());
    }

    #[test]
    fn test_parse_percentage_without_total() {
        assert!(parse_resource_value("50%", None).is_err());
    }

    #[test]
    fn test_resource_budget_parse() {
        let budget = ResourceBudget::parse("threads:4-8,segmentsize:500000").unwrap();
        assert_eq!(
            budget.get("threads"),
            Some(&ResourceValue::Range {
                floor: 4,
                ceiling: 8
            })
        );
        assert_eq!(
            budget.get("segmentsize"),
            Some(&ResourceValue::Fixed(500_000))
        );
    }

    #[test]
    fn test_resource_value_midpoint() {
        assert_eq!(ResourceValue::Fixed(10).midpoint(), 10);
        assert_eq!(
            ResourceValue::Range {
                floor: 4,
                ceiling: 8
            }
            .midpoint(),
            6
        );
    }

    #[test]
    fn test_governor_current_and_request() {
        let budget = ResourceBudget::parse("threads:4-8,segmentsize:500000").unwrap();
        let gov = ResourceGovernor::new(budget, None);

        // threads midpoint is 6
        assert_eq!(gov.current("threads"), Some(6));
        // segmentsize fixed at 500000
        assert_eq!(gov.current("segmentsize"), Some(500_000));
        // unknown resource
        assert_eq!(gov.current("unknown"), None);
        assert_eq!(gov.current_or("unknown", 42), 42);
    }

    #[test]
    fn test_governor_request_clamped() {
        let budget = ResourceBudget::parse("threads:4-8").unwrap();
        let gov = ResourceGovernor::new(budget, None);

        // Request within range
        assert_eq!(gov.request("threads", 6), 6);
        // Request above ceiling
        assert_eq!(gov.request("threads", 100), 8);
        // Request below floor
        assert_eq!(gov.request("threads", 1), 4);
    }

    #[test]
    fn test_maximize_strategy_nominal() {
        let mut strategy = MaximizeUtilizationStrategy::default();
        let budget = ResourceBudget::parse("threads:4-8").unwrap();
        let mut current = HashMap::new();
        current.insert("threads".to_string(), 6u64);

        // RSS at 75% — nominal, no adjustments
        let snapshot = SystemSnapshot {
            rss_bytes: 0,
            rss_pct: 75.0,
            cpu_user_pct: 50.0,
            cpu_system_pct: 5.0,
            io_read_bps: 0,
            io_write_bps: 0,
            major_faults: 0,
            minor_faults: 0,
            active_threads: 6,
            cpu_user_ticks: 0,
            cpu_system_ticks: 0,
            ticks_per_sec: 100.0,
        };

        let adj = strategy.evaluate(&snapshot, &budget, &current);
        assert!(!adj.throttle);
        assert!(!adj.emergency);
        assert!(adj.updates.is_empty());
    }

    #[test]
    fn test_maximize_strategy_emergency() {
        let mut strategy = MaximizeUtilizationStrategy::default();
        let budget = ResourceBudget::parse("threads:4-8").unwrap();
        let mut current = HashMap::new();
        current.insert("threads".to_string(), 6u64);

        let snapshot = SystemSnapshot {
            rss_bytes: 0,
            rss_pct: 96.0,
            cpu_user_pct: 80.0,
            cpu_system_pct: 5.0,
            io_read_bps: 0,
            io_write_bps: 0,
            major_faults: 0,
            minor_faults: 0,
            active_threads: 6,
            cpu_user_ticks: 0,
            cpu_system_ticks: 0,
            ticks_per_sec: 100.0,
        };

        let adj = strategy.evaluate(&snapshot, &budget, &current);
        assert!(adj.throttle);
        assert!(adj.emergency);
        // Should reduce to floor
        assert_eq!(adj.updates.get("threads"), Some(&4));
    }

    #[test]
    fn test_fixed_strategy_never_adjusts() {
        let mut strategy = FixedStrategy;
        let budget = ResourceBudget::parse("threads:4-8").unwrap();
        let current = HashMap::new();

        let snapshot = SystemSnapshot {
            rss_bytes: 0,
            rss_pct: 99.0,
            cpu_user_pct: 100.0,
            cpu_system_pct: 0.0,
            io_read_bps: 0,
            io_write_bps: 0,
            major_faults: 0,
            minor_faults: 0,
            active_threads: 0,
            cpu_user_ticks: 0,
            cpu_system_ticks: 0,
            ticks_per_sec: 100.0,
        };

        let adj = strategy.evaluate(&snapshot, &budget, &current);
        assert!(!adj.throttle);
        assert!(adj.updates.is_empty());
    }

    #[test]
    fn test_default_governor() {
        let gov = ResourceGovernor::default_governor();
        // Should have defaults for mem, threads, segmentsize
        assert!(gov.current("threads").is_some());
        assert!(gov.current("segmentsize").is_some());
    }

    #[test]
    fn test_resource_type_from_name() {
        assert_eq!(ResourceType::from_name("mem"), Some(ResourceType::Mem));
        assert_eq!(ResourceType::from_name("threads"), Some(ResourceType::Threads));
        assert_eq!(ResourceType::from_name("segmentsize"), Some(ResourceType::SegmentSize));
        assert_eq!(ResourceType::from_name("iothreads"), Some(ResourceType::IoThreads));
        assert_eq!(ResourceType::from_name("cache"), Some(ResourceType::Cache));
        assert_eq!(ResourceType::from_name("readahead"), Some(ResourceType::Readahead));
        assert_eq!(ResourceType::from_name("segments"), Some(ResourceType::Segments));
        assert_eq!(ResourceType::from_name("unknown"), None);
    }

    #[test]
    fn test_resource_type_value_kind() {
        assert_eq!(ResourceType::Mem.value_kind(), ValueKind::Memory);
        assert_eq!(ResourceType::Cache.value_kind(), ValueKind::Memory);
        assert_eq!(ResourceType::Readahead.value_kind(), ValueKind::Memory);
        assert_eq!(ResourceType::Threads.value_kind(), ValueKind::Count);
        assert_eq!(ResourceType::Segments.value_kind(), ValueKind::Count);
        assert_eq!(ResourceType::SegmentSize.value_kind(), ValueKind::Count);
        assert_eq!(ResourceType::IoThreads.value_kind(), ValueKind::Count);
    }

    #[test]
    fn test_resource_type_supports_percentage() {
        assert!(ResourceType::Mem.supports_percentage());
        assert!(ResourceType::Cache.supports_percentage());
        assert!(ResourceType::Readahead.supports_percentage());
        assert!(!ResourceType::Threads.supports_percentage());
        assert!(!ResourceType::SegmentSize.supports_percentage());
    }

    #[test]
    fn test_resource_type_roundtrip() {
        for rt in ResourceType::all() {
            assert_eq!(
                ResourceType::from_name(rt.name()),
                Some(*rt),
                "ResourceType::from_name failed for {}",
                rt.name(),
            );
        }
    }

    #[test]
    fn test_resource_desc_new() {
        let desc = ResourceDesc::new(ResourceType::Mem, "Test memory usage", true);
        assert_eq!(desc.name, "mem");
        assert_eq!(desc.description, "Test memory usage");
        assert!(desc.adjustable);
    }
}
