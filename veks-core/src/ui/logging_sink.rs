// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Unified logging: TUI display + persistent file log.
//!
//! [`install_logger`] opens the log file, writes a session header, and
//! returns a shared writer that [`UiHandle`] stores. Every call to
//! `UiHandle::log()` writes to both the TUI sink and the file in lock-step.
//!
//! The `log` crate is also configured so that `log::info!()` etc. from
//! library code reach the file, but `UiHandle::log()` does NOT depend on
//! `log::set_logger` succeeding — the file write is direct.

use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::event::UiEvent;
use super::sink::UiSink;

/// Shared handle to the persistent log file.
///
/// Stored inside [`UiHandle`] so that `log()` writes in lock-step with
/// TUI display, without depending on `log::set_logger`.
pub type LogFileWriter = Arc<Mutex<std::io::BufWriter<std::fs::File>>>;

/// Combined log writers: plain text (run.log) + JSONL (run.jsonl).
///
/// Both are written synchronously from `UiHandle::log()`.
#[derive(Clone)]
pub struct LogWriters {
    pub text: LogFileWriter,
    pub jsonl: LogFileWriter,
}

/// Open the persistent log files, write session headers, and install a
/// global `log` crate logger.
///
/// Returns the writers for direct use by `UiHandle::log()`.
///
/// - `sink`: The UI sink for `log::info!`+ forwarding
/// - `log_path`: If `Some`, plain text is appended to this file
/// - `jsonl_path`: If `Some`, JSONL is appended to this file
///
/// Call once per process, typically during pipeline startup.
pub fn install_logger(sink: Arc<dyn UiSink>, log_path: Option<&Path>, jsonl_path: Option<&Path>) -> Option<LogWriters> {
    let text_writer = log_path.and_then(|p| {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .ok()?;
        let writer = Arc::new(Mutex::new(std::io::BufWriter::with_capacity(8192, file)));
        {
            let mut w = writer.lock().unwrap();
            let _ = writeln!(w, "\n=== veks run {} ===",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
            let _ = w.flush();
        }
        Some(writer)
    });

    let jsonl_writer = jsonl_path.and_then(|p| {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .ok()?;
        let writer = Arc::new(Mutex::new(std::io::BufWriter::with_capacity(8192, file)));
        {
            let ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
            let mut w = writer.lock().unwrap();
            let _ = writeln!(w, r#"{{"type":"session","ts":"{}","message":"veks run"}}"#, ts);
            let _ = w.flush();
        }
        Some(writer)
    });

    let writers = match (text_writer, jsonl_writer) {
        (Some(text), Some(jsonl)) => Some(LogWriters { text, jsonl }),
        (Some(text), None) => {
            // Create a dummy JSONL writer that discards
            Some(LogWriters { text: text.clone(), jsonl: text })
        }
        _ => None,
    };

    // Install global log crate logger for log::info!() etc. from library code.
    // This is best-effort — UiHandle::log() writes directly and doesn't need it.
    let logger = Box::leak(Box::new(PipelineLogger {
        sink,
        file_writer: writers.as_ref().map(|w| w.text.clone()),
        jsonl_writer: writers.as_ref().map(|w| w.jsonl.clone()),
    }));
    match log::set_logger(logger) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("WARNING: failed to install log crate logger: {e}");
        }
    }
    log::set_max_level(log::LevelFilter::Trace);

    writers
}

/// Write a message to both log files (plain text + JSONL) with timestamp.
///
/// Called from `UiHandle::log()` in lock-step with TUI display.
pub fn write_to_log(writers: &LogWriters, message: &str) {
    let ts = chrono::Local::now();

    // Plain text
    if let Ok(mut w) = writers.text.lock() {
        let _ = writeln!(w, "[{}] INFO  ui — {}", ts.format("%H:%M:%S"), message);
        let _ = w.flush();
    }

    // JSONL
    if let Ok(mut w) = writers.jsonl.lock() {
        // Escape JSON string: backslashes, quotes, control chars
        let escaped = message.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        let _ = writeln!(w, r#"{{"ts":"{}","level":"INFO","message":"{}"}}"#,
            ts.format("%Y-%m-%dT%H:%M:%S"), escaped);
        let _ = w.flush();
    }
}

/// The global logger. Routes `log` crate events to the TUI sink and files.
struct PipelineLogger {
    sink: Arc<dyn UiSink>,
    file_writer: Option<LogFileWriter>,
    jsonl_writer: Option<LogFileWriter>,
}

impl log::Log for PipelineLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        let ts = chrono::Local::now();
        let message = format!("{}", record.args());

        // Write all levels to the persistent plain text log file.
        if let Some(ref fw) = self.file_writer {
            if let Ok(mut w) = fw.lock() {
                let _ = writeln!(w, "[{}] {:5} {} — {}",
                    ts.format("%H:%M:%S"),
                    record.level(),
                    record.target(),
                    message,
                );
                let _ = w.flush();
            }
        }

        // Write all levels to the JSONL log file.
        if let Some(ref jw) = self.jsonl_writer {
            if let Ok(mut w) = jw.lock() {
                let escaped = message.replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                let _ = writeln!(w, r#"{{"ts":"{}","level":"{:5}","target":"{}","message":"{}"}}"#,
                    ts.format("%Y-%m-%dT%H:%M:%S"),
                    record.level(),
                    record.target(),
                    escaped,
                );
                let _ = w.flush();
            }
        }

        // Forward Info+ to the TUI sink, unless the message came from
        // UiHandle::log() (target "ui") — those are already sent directly.
        if record.level() <= log::Level::Info && record.target() != "ui" {
            self.sink.send(UiEvent::Log { message });
        }
    }

    fn flush(&self) {
        if let Some(ref fw) = self.file_writer {
            if let Ok(mut w) = fw.lock() { let _ = w.flush(); }
        }
        if let Some(ref jw) = self.jsonl_writer {
            if let Ok(mut w) = jw.lock() { let _ = w.flush(); }
        }
    }
}
