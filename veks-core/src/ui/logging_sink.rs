// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Unified logging via the standard `log` crate.
//!
//! All pipeline logging flows through `log::info!()`, `log::debug!()`, etc.
//! The [`install_logger`] function sets up a global logger that routes
//! messages to two destinations:
//!
//! - **TUI display**: `Info` and above are forwarded to the `UiSink` as
//!   `UiEvent::Log` events.
//! - **Persistent file**: All levels are appended to `.cache/run.log`
//!   (when a file path is provided).
//!
//! `UiHandle::log()` is a thin wrapper around `log::info!()` — there
//! is no separate `UiEvent::Log` code path in application code.

use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::event::UiEvent;
use super::sink::UiSink;

/// Install the global `log` crate logger that routes messages to the
/// TUI sink and optionally to a persistent log file.
///
/// - `sink`: The UI sink that receives `Info`+ messages as `UiEvent::Log`
/// - `log_path`: If `Some`, all levels are appended to this file
///
/// Call once per process, typically during pipeline startup. Subsequent
/// calls are ignored (the `log` crate only allows one global logger).
pub fn install_logger(sink: Arc<dyn UiSink>, log_path: Option<&Path>) {
    let file_writer = log_path.and_then(|p| {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .ok()?;
        let writer = Mutex::new(std::io::BufWriter::with_capacity(8192, file));
        // Session header
        {
            let mut w = writer.lock().unwrap();
            let _ = writeln!(w, "\n=== veks run {} ===",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
            let _ = w.flush();
        }
        Some(writer)
    });

    let logger = Box::leak(Box::new(PipelineLogger { sink, file_writer }));
    let _ = log::set_logger(logger);
    log::set_max_level(log::LevelFilter::Trace);
}

/// The global logger. Routes log events to the TUI sink and/or a file.
struct PipelineLogger {
    sink: Arc<dyn UiSink>,
    file_writer: Option<Mutex<std::io::BufWriter<std::fs::File>>>,
}

impl log::Log for PipelineLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        // Write all levels to the persistent log file
        if let Some(ref fw) = self.file_writer {
            if let Ok(mut w) = fw.lock() {
                let ts = chrono::Local::now().format("%H:%M:%S");
                let _ = writeln!(w, "[{}] {:5} {} — {}",
                    ts,
                    record.level(),
                    record.target(),
                    record.args(),
                );
            }
        }

        // Forward Info and above to the TUI sink, unless the message
        // came from UiHandle::log() (target "ui") — those are already
        // sent directly to the sink to work even without a logger.
        if record.level() <= log::Level::Info && record.target() != "ui" {
            self.sink.send(UiEvent::Log {
                message: format!("{}", record.args()),
            });
        }
    }

    fn flush(&self) {
        if let Some(ref fw) = self.file_writer {
            if let Ok(mut w) = fw.lock() {
                let _ = w.flush();
            }
        }
    }
}
