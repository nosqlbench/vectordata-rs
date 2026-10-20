// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Progress for a check that reads a whole file.
//!
//! Most checks read headers and counts and are over before a line
//! could be drawn. A check that scans every record — the ground-truth
//! bounds check over a 10-million-row index — takes long enough that a
//! silent run reads as a hung one, so it reports through a
//! [`ScanSink`]: the terminal in a run, a `Vec<String>` in a test.

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

/// Where a scanning check reports how far it has got.
pub trait ScanSink {
    /// `done` of `total` records of the scan `label` are read. Called
    /// once at the start, after every block, and once at the end with
    /// `done == total` (or where the scan stopped).
    fn report(&mut self, label: &str, done: u64, total: u64);
}

/// A test's sink: every report as a line.
impl ScanSink for Vec<String> {
    fn report(&mut self, label: &str, done: u64, total: u64) {
        self.push(format!("{label}: {done}/{total}"));
    }
}

/// The terminal: one line redrawn at most every 250 ms on a tty, a
/// line at every tenth otherwise, and always the last.
pub struct StderrSink {
    tty: bool,
    redraw: Duration,
    last_draw: Option<Instant>,
    last_tenth: Option<u64>,
    open: Option<String>,
}

impl Default for StderrSink {
    fn default() -> Self {
        Self { tty: std::io::stderr().is_terminal(), redraw: Duration::from_millis(250), last_draw: None, last_tenth: None, open: None }
    }
}

impl ScanSink for StderrSink {
    fn report(&mut self, label: &str, done: u64, total: u64) {
        let finished = total == 0 || done >= total;
        if self.open.as_deref() != Some(label) {
            self.open = Some(label.to_string());
            self.last_draw = None;
            self.last_tenth = None;
        }
        let mut err = std::io::stderr();
        if self.tty {
            let due = self.last_draw.is_none_or(|t| t.elapsed() >= self.redraw);
            if due || finished {
                let _ = write!(err, "\r  {label}: {done}/{total}\x1b[K");
                self.last_draw = Some(Instant::now());
            }
            if finished {
                let _ = writeln!(err);
                self.open = None;
            }
        } else {
            let tenth = (done * 10).checked_div(total).unwrap_or(10);
            if self.last_tenth != Some(tenth) || finished {
                let _ = writeln!(err, "  {label}: {done}/{total}");
                self.last_tenth = Some(tenth);
            }
            if finished {
                self.open = None;
            }
        }
        let _ = err.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capturing sink hears the start, every block and the end, in
    /// order, with the label it was given.
    #[test]
    fn a_vec_sink_keeps_every_report() {
        let mut lines: Vec<String> = Vec::new();
        lines.report("scanning x", 0, 4);
        lines.report("scanning x", 2, 4);
        lines.report("scanning x", 4, 4);
        assert_eq!(lines, ["scanning x: 0/4", "scanning x: 2/4", "scanning x: 4/4"]);
    }

    /// The terminal sink draws on a tty when the redraw interval has
    /// passed and always at the end, where a plain stream gets a line
    /// at every tenth; either way the label closes at the end and a new
    /// label starts its cadence afresh.
    #[test]
    fn the_stderr_sink_paces_by_tty_and_closes_the_label() {
        let mut tty = StderrSink { tty: true, redraw: Duration::from_secs(3600), last_draw: None, last_tenth: None, open: None };
        tty.report("scan", 0, 10);
        let first = tty.last_draw.expect("the first report draws");
        tty.report("scan", 5, 10);
        assert_eq!(tty.last_draw, Some(first), "within the interval nothing is redrawn");
        tty.report("scan", 10, 10);
        assert!(tty.last_draw > Some(first), "the end always draws");
        assert!(tty.open.is_none(), "the label closes at the end");

        let mut plain = StderrSink { tty: false, redraw: Duration::ZERO, last_draw: None, last_tenth: None, open: None };
        plain.report("scan", 0, 100);
        assert_eq!(plain.last_tenth, Some(0));
        plain.report("scan", 9, 100);
        assert_eq!(plain.last_tenth, Some(0), "9% is still the first tenth");
        plain.report("scan", 10, 100);
        assert_eq!(plain.last_tenth, Some(1));
        plain.report("other", 0, 0);
        assert!(plain.open.is_none(), "an empty scan opens and closes in one report");
    }
}
