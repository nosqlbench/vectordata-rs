// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Plain-text [`UiSink`] for CLI mode (non-TUI).
//!
//! Renders progress bars as single-line status updates to stderr using
//! carriage-return overwriting. Log messages go to stdout. This provides
//! CLI-mode progress feedback without requiring a terminal UI.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use super::event::{ProgressId, UiEvent};
use super::sink::UiSink;

/// Tracked progress bar state for CLI rendering.
struct BarState {
    label: String,
    unit: String,
    total: u64,
    position: u64,
    created: Instant,
    last_render: Instant,
}

/// Sink that writes plain text to stdout and renders progress bars to stderr.
pub struct PlainSink {
    next_id: AtomicU32,
    bars: Mutex<HashMap<ProgressId, BarState>>,
}

impl PlainSink {
    pub fn new() -> Self {
        PlainSink {
            next_id: AtomicU32::new(0),
            bars: Mutex::new(HashMap::new()),
        }
    }

    /// Render the current state of a bar to stderr.
    fn render_bar(bar: &mut BarState) {
        let elapsed = bar.created.elapsed().as_secs_f64();
        let pct = if bar.total > 0 {
            (bar.position as f64 / bar.total as f64 * 100.0) as u32
        } else {
            0
        };

        let rate_str = if elapsed > 0.5 && bar.position > 0 {
            let rps = bar.position as f64 / elapsed;
            if rps >= 1_000_000.0 {
                format!(" {:.1}M {}/s", rps / 1_000_000.0, bar.unit)
            } else if rps >= 1_000.0 {
                format!(" {:.1}K {}/s", rps / 1_000.0, bar.unit)
            } else if rps >= 1.0 {
                format!(" {:.0} {}/s", rps, bar.unit)
            } else if rps > 0.0 {
                format!(" {:.1}s/{}", 1.0 / rps, bar.unit)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let eta_str = if elapsed > 1.0 && bar.position > 0 && bar.total > bar.position {
            let rps = bar.position as f64 / elapsed;
            let remaining = (bar.total - bar.position) as f64 / rps;
            if remaining >= 3600.0 {
                format!(" eta {}h{}m", remaining as u64 / 3600, (remaining as u64 % 3600) / 60)
            } else if remaining >= 60.0 {
                format!(" eta {}m{}s", remaining as u64 / 60, remaining as u64 % 60)
            } else {
                format!(" eta {}s", remaining as u64)
            }
        } else {
            String::new()
        };

        let pos_str = format_count(bar.position);
        let total_str = format_count(bar.total);

        let _ = eprint!(
            "\r{} [{}/{}] {}%{}{}    ",
            bar.label, pos_str, total_str, pct, rate_str, eta_str,
        );
        let _ = std::io::stderr().flush();
        bar.last_render = Instant::now();
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

impl Default for PlainSink {
    fn default() -> Self {
        Self::new()
    }
}

impl UiSink for PlainSink {
    fn send(&self, event: UiEvent) {
        match event {
            UiEvent::ProgressCreate { id, total, label, unit, .. } => {
                let mut bars = self.bars.lock().unwrap();
                let now = Instant::now();
                bars.insert(id, BarState {
                    label, unit, total, position: 0,
                    created: now, last_render: now,
                });
            }

            UiEvent::ProgressUpdate { id, position } => {
                let mut bars = self.bars.lock().unwrap();
                if let Some(bar) = bars.get_mut(&id) {
                    bar.position = position;
                    // Throttle: render at 4Hz (250ms)
                    if bar.last_render.elapsed().as_millis() >= 250 {
                        Self::render_bar(bar);
                    }
                }
            }

            UiEvent::ProgressInc { id, delta } => {
                let mut bars = self.bars.lock().unwrap();
                if let Some(bar) = bars.get_mut(&id) {
                    bar.position += delta;
                    if bar.last_render.elapsed().as_millis() >= 500 {
                        Self::render_bar(bar);
                    }
                }
            }

            UiEvent::ProgressMessage { id, message } => {
                let mut bars = self.bars.lock().unwrap();
                if let Some(bar) = bars.get_mut(&id) {
                    bar.label = message;
                }
            }

            UiEvent::ProgressFinish { id } => {
                let mut bars = self.bars.lock().unwrap();
                if let Some(mut bar) = bars.remove(&id) {
                    bar.position = bar.total;
                    Self::render_bar(&mut bar);
                    let _ = eprintln!(); // newline after final render
                }
            }

            UiEvent::ResourceStatus { .. }
            | UiEvent::SetContext { .. }
            | UiEvent::SetStepYaml { .. }
            | UiEvent::SuspendBegin
            | UiEvent::SuspendEnd
            | UiEvent::Clear => {}

            UiEvent::Log { message } => {
                let mut out = std::io::stdout().lock();
                let _ = writeln!(out, "{}", message);
            }

            UiEvent::Emit { text } => {
                let mut out = std::io::stdout().lock();
                let _ = write!(out, "{}", text);
            }

            UiEvent::EmitLn { text } => {
                let mut out = std::io::stdout().lock();
                let _ = writeln!(out, "{}", text);
            }
        }
    }

    fn next_progress_id(&self) -> ProgressId {
        ProgressId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_sink_ids_are_unique() {
        let sink = PlainSink::new();
        let a = sink.next_progress_id();
        let b = sink.next_progress_id();
        assert_ne!(a, b);
    }

    #[test]
    fn plain_sink_progress_no_panic() {
        // Verify no panic on progress lifecycle events.
        let sink = PlainSink::new();
        let id = sink.next_progress_id();
        sink.send(UiEvent::ProgressCreate {
            id,
            kind: super::super::event::ProgressKind::Bar,
            total: 100,
            label: "test".into(),
            unit: "rec".into(),
        });
        sink.send(UiEvent::ProgressUpdate { id, position: 50 });
        sink.send(UiEvent::ProgressFinish { id });
    }
}
