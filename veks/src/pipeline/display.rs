// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Terminal-aware progress display (SRD §08).
//!
//! Manages a single fixed status line at the bottom of the terminal.
//! Log messages print above the progress line; the progress line itself
//! overwrites in place.  When stderr is not a TTY, progress bars are
//! hidden and only milestone log lines are emitted.

use std::io::IsTerminal;

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};

/// Terminal-aware progress display.
///
/// All within-step progress indicators render on a single fixed region
/// at the bottom of the terminal, managed by an inner `MultiProgress`.
/// Commands obtain handles via [`bar`] and [`spinner`]; log messages
/// emitted through [`log`] scroll above the progress line.
#[derive(Clone)]
pub struct ProgressDisplay {
    mp: MultiProgress,
    is_tty: bool,
}

impl ProgressDisplay {
    /// Create a new display.
    ///
    /// If stderr is not a TTY the draw target is set to hidden so
    /// progress bars produce no output; only [`log`] messages are
    /// emitted (via `eprintln!`).
    pub fn new() -> Self {
        let is_tty = std::io::stderr().is_terminal();
        let mp = MultiProgress::new();
        if !is_tty {
            mp.set_draw_target(ProgressDrawTarget::hidden());
        }
        ProgressDisplay { mp, is_tty }
    }

    /// Create a determinate progress bar with a known total.
    ///
    /// The returned `ProgressBar` is added to the shared multi-progress
    /// region so it renders on the fixed status line.
    pub fn bar(&self, total: u64, message: &str) -> ProgressBar {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg} — {rps} — ETA {eta}")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .with_key("rps", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    let rate = state.per_sec();
                    if rate >= 1_000_000.0 {
                        let _ = write!(w, "{:.1}M/s", rate / 1_000_000.0);
                    } else if rate >= 1_000.0 {
                        let _ = write!(w, "{:.1}K/s", rate / 1_000.0);
                    } else {
                        let _ = write!(w, "{:.0}/s", rate);
                    }
                }),
        );
        pb.set_message(message.to_string());
        self.mp.add(pb)
    }

    /// Create a progress bar with a custom `ProgressStyle`.
    ///
    /// Useful for commands that need specialised templates (e.g. bytes
    /// throughput, custom keys).
    pub fn bar_with_style(&self, total: u64, style: ProgressStyle) -> ProgressBar {
        let pb = ProgressBar::new(total);
        pb.set_style(style);
        self.mp.add(pb)
    }

    /// Create an indeterminate spinner.
    pub fn spinner(&self, message: &str) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("  {spinner:.yellow} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(120));
        self.mp.add(pb)
    }

    /// Print a log line above the progress bar without disrupting it.
    ///
    /// When stderr is not a TTY this falls back to plain `eprintln!`.
    pub fn log(&self, msg: &str) {
        if self.is_tty {
            let _ = self.mp.println(msg);
        } else {
            eprintln!("{}", msg);
        }
    }

    /// Create a resource utilization status bar.
    ///
    /// The returned `ProgressBar` is a spinner-style bar at the top of the
    /// progress region. The caller should update it periodically via
    /// `set_message()` with the output of `ResourceGovernor::status_line()`.
    /// Use [`start_resource_ticker`] to automate this.
    pub fn resource_bar(&self) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("  {spinner:.dim} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(500));
        // Insert at position 0 so the resource bar is always at the top
        self.mp.insert(0, pb.clone());
        pb
    }

    /// Clear the progress region entirely.
    pub fn clear(&self) {
        let _ = self.mp.clear();
    }
}

impl Default for ProgressDisplay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_bar_and_finish() {
        let display = ProgressDisplay::new();
        let pb = display.bar(100, "testing");
        pb.inc(50);
        pb.finish_and_clear();
    }

    #[test]
    fn test_display_spinner_and_finish() {
        let display = ProgressDisplay::new();
        let pb = display.spinner("working");
        pb.finish_and_clear();
    }

    #[test]
    fn test_display_log() {
        let display = ProgressDisplay::new();
        display.log("hello from test");
    }

    #[test]
    fn test_display_clear() {
        let display = ProgressDisplay::new();
        let _pb = display.bar(10, "x");
        display.clear();
    }
}
