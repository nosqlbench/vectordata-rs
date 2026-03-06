<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 08 — Progress Display

## 8.1 Overview

All progress output across the veks CLI MUST use a consistent, non-scrolling
progress display. Progress updates appear on a fixed status line at the
bottom of the terminal, overwriting in place rather than emitting new lines.
Informational log messages (step transitions, warnings, errors) still scroll
above the progress line, but ongoing within-step progress (bars, spinners,
counters, ETA) is always rendered in-place.

This ensures a clean, predictable terminal experience regardless of how many
progress updates a long-running operation produces.

## 8.2 Requirements

### REQ-PD-01: Fixed progress line

All within-step progress indicators (progress bars, spinners, counters,
ETA displays) MUST render on a single fixed line at the bottom of the
terminal. Updates overwrite the same line using carriage-return (`\r`)
or ANSI cursor control — they MUST NOT emit newlines that scroll the
terminal.

When a progress indicator completes or is replaced, it clears the line
and any subsequent log message scrolls normally above.

### REQ-PD-02: Unified progress API

All commands MUST use a shared progress facility rather than constructing
`indicatif::ProgressBar` instances ad-hoc. The facility provides:

```rust
/// Terminal-aware progress display.
///
/// Manages a single fixed status line at the bottom of the terminal.
/// Log messages print above the progress line; the progress line
/// itself overwrites in place.
pub struct ProgressDisplay { /* ... */ }

impl ProgressDisplay {
    /// Create a determinate progress bar (known total).
    pub fn bar(&self, total: u64, message: &str) -> ProgressHandle;

    /// Create an indeterminate spinner.
    pub fn spinner(&self, message: &str) -> ProgressHandle;

    /// Print a log line above the progress bar without disrupting it.
    pub fn log(&self, msg: &str);

    /// Clear the progress line entirely.
    pub fn clear(&self);
}
```

The `ProgressHandle` wraps the underlying progress bar and ensures it
renders in the fixed status line position.

### REQ-PD-03: indicatif integration

The implementation SHOULD use `indicatif` (already a dependency) with a
draw target configured for in-place rendering:

1. A single `MultiProgress` instance owned by `ProgressDisplay`,
   configured with `TermLike` or `DrawTarget` that holds the cursor
   on the last line.
2. `ProgressBar::set_draw_target()` pointed at the shared multi-progress
   so all bars share the same line region.
3. `indicatif::ProgressBar::println()` used for log messages that must
   appear above the progress line.

When `stderr` is not a TTY (e.g., piped to a file), progress bars MUST
degrade gracefully: either emit periodic one-line status updates at
coarse intervals, or suppress progress entirely and only emit milestone
log lines.

### REQ-PD-04: Runner-level coordination

The pipeline runner (`runner.rs`) manages the `ProgressDisplay` lifetime:

1. Creates a `ProgressDisplay` before step execution begins.
2. Passes it into `StreamContext` so commands can access it.
3. Between steps, clears the progress line and prints the step
   transition message (SKIP, running, OK, ERROR) as a normal scrolling
   log line.
4. After all steps, clears the progress display.

This means `StreamContext` gains a field:

```rust
pub struct StreamContext {
    // ... existing fields ...
    /// Shared progress display for the fixed status line.
    pub display: ProgressDisplay,
}
```

### REQ-PD-05: Command-level usage

Individual commands obtain progress handles from `ctx.display` instead
of constructing their own `ProgressBar`:

```rust
// Before (scrolling):
let pb = ProgressBar::new(total);
pb.set_style(ProgressStyle::default_bar().template("..."));
pb.inc(1);
pb.finish_and_clear();

// After (fixed line):
let pb = ctx.display.bar(total, "computing KNN");
pb.inc(1);
pb.finish();      // clears the line
```

For commands that emit diagnostic log lines during processing, they
MUST use `ctx.display.log()` (or `pb.println()`) so the message
appears above the progress bar:

```rust
// Prints above the progress line, doesn't disrupt it
ctx.display.log(&format!("  partition {}/{} cached", i, n));
```

### REQ-PD-06: Multi-phase progress

Some commands have multiple sequential phases (e.g., `compute knn` has
planning, prefetching, computing, merging). Each phase creates a new
progress handle that replaces the previous one on the same status line.
The transition between phases prints a completion summary as a log line
above:

```
  partition 1/4: 25000 queries (3.2s)        ← scrolled log line
  partition 2/4: 25000 queries (3.1s)        ← scrolled log line
  [███████████████░░░░░░░░░░░░░░] 62500/100000 queries (eta 6s)  ← fixed line
```

### REQ-PD-07: No scrolling counters or percentage updates

The following patterns are prohibited:

```
  Processing: 10% ...
  Processing: 20% ...
  Processing: 30% ...
```

Any counter, percentage, or ETA that updates more than once MUST use
in-place overwriting on the fixed progress line.

One-shot informational messages (e.g., "opening base vectors: foo.fvec")
are allowed as normal scrolling log lines since they appear only once.

### REQ-PD-08: Direct CLI mode

When a command is invoked directly via `veks pipeline <group> <command>`,
the same `ProgressDisplay` is used. `run_direct()` in `cli.rs` creates
a `ProgressDisplay` and wires it into the `StreamContext`.

## 8.3 Current State

Progress display is currently ad-hoc:

- `indicatif::ProgressBar` is constructed directly by individual commands
  (`compute_knn.rs`, `compute_filtered_knn.rs`, `convert.rs`, `import.rs`)
- Some commands use `MultiProgress` locally (`compute_knn.rs`, `bulkdl`)
- The runner uses `eprintln!` for all step-level output (~15 calls in
  `runner.rs`: step transitions, skip reasons, error reporting, timing)
- There is no shared progress facility or fixed-line coordination
- Some commands emit scrolling counter updates via `eprintln!`:
  - `slab.rs:1786`: `"slab index: {}/{} pages scanned"` (called per page)
  - `gen_predicate_keys.rs:580`: `"slab index: {}/{} pages scanned"` (same pattern)
- 37 command files use `eprintln!` in some capacity; most are one-shot
  informational messages (allowed under REQ-PD-07), but the two scrolling
  counter patterns above violate it
- `pipeline/cli.rs` uses `eprintln!` for error/usage messages and
  post-execution status reporting

## 8.4 Affected Files

### Files with `indicatif` progress bars (must migrate to `ProgressDisplay`)

| File | Current | Required |
|------|---------|----------|
| `pipeline/commands/compute_knn.rs` | `MultiProgress` + multiple `ProgressBar` (partition validation bar, prefetch I/O bar, query progress bar, cache flush spinner, wait spinner) | Use `ctx.display.bar()` / `ctx.display.spinner()` / `ctx.display.log()` |
| `pipeline/commands/compute_filtered_knn.rs` | Ad-hoc `ProgressBar` for f32 and f16 query loops | Use `ctx.display.bar()` |
| `pipeline/commands/convert.rs` | Conditional `ProgressBar` (bar if count known, spinner if not) + `eprintln!` for governor throttle | Use `ctx.display.bar()` or `ctx.display.spinner()` |
| `pipeline/commands/import.rs` | Conditional `ProgressBar` with custom RPS metric | Use `ctx.display.bar()` or `ctx.display.spinner()` |
| `convert/mod.rs` | Standalone `ProgressBar` (non-pipeline CLI path) | Use standalone `ProgressDisplay` or pass display as parameter |
| `import/mod.rs` | Standalone `ProgressBar` (non-pipeline CLI path) | Use standalone `ProgressDisplay` or pass display as parameter |
| `bulkdl/mod.rs` | `MultiProgress` coordinating per-dataset `ProgressBar` instances | Use shared `ProgressDisplay` |

### Files with scrolling counter violations (REQ-PD-07)

| File | Current | Required |
|------|---------|----------|
| `pipeline/commands/slab.rs` | `eprintln!("slab index: {}/{} pages scanned")` in a loop (line 1786) | Use `ctx.display.bar()` for page scanning progress |
| `pipeline/commands/gen_predicate_keys.rs` | `eprintln!("slab index: {}/{} pages scanned")` in a loop (line 580) | Use `ctx.display.bar()` for page scanning progress |

### Infrastructure files (must add `ProgressDisplay` support)

| File | Current | Required |
|------|---------|----------|
| `pipeline/runner.rs` | `eprintln!` for step status (~15 call sites) | Create `ProgressDisplay`, use `display.log()` for step messages |
| `pipeline/command.rs` | `StreamContext` has no display field | Add `display: ProgressDisplay` field |
| `pipeline/cli.rs` | No progress display in `run_direct()` | Create `ProgressDisplay` and wire into `StreamContext` |
| `pipeline/progress.rs` | Persistent progress log (step records, not UI) | No change needed — this is orthogonal to UI progress |

## 8.5 Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-PD-01 | Done | `ProgressDisplay` wraps `MultiProgress` for fixed-line bars |
| REQ-PD-02 | Done | `display.rs` provides unified `bar()`, `spinner()`, `log()` API |
| REQ-PD-03 | Done | All bars created through `ProgressDisplay` (indicatif `MultiProgress`) |
| REQ-PD-04 | Done | Runner uses `ctx.display.log()` for all step output |
| REQ-PD-05 | Done | `import.rs`, `convert.rs`, `compute_knn.rs`, `compute_filtered_knn.rs` migrated |
| REQ-PD-06 | Done | Multi-phase commands (compute_knn) use shared `ProgressDisplay` |
| REQ-PD-07 | Done | Scrolling counters in `slab.rs` and `gen_predicate_keys.rs` replaced with progress bars |
| REQ-PD-08 | Done | `pipeline/cli.rs` `run_direct()` creates `ProgressDisplay` for `StreamContext` |
