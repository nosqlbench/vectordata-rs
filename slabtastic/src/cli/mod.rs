// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI module for the `slab` file maintenance tool.
//!
//! This module implements the `slab` binary, a command-line interface for
//! creating, inspecting, and transforming slabtastic files. Each
//! subcommand lives in its own child module and is dispatched by
//! [`run`].
//!
//! ## Subcommands
//!
//! | Command       | Description                                              |
//! |---------------|----------------------------------------------------------|
//! | `analyze`     | Display file structure and statistics                    |
//! | `check`       | Validate a slab file for structural errors               |
//! | `get`         | Retrieve records by ordinal (hex, base64, or raw output) |
//! | `import`      | Import records from external formats (CSV, JSON, etc.)   |
//! | `export`      | Export records to external formats or another slab file   |
//! | `append`      | Append records from stdin or a source file               |
//! | `rewrite`     | Repack a slab file with new page settings                |
//! | `explain`     | Display block diagrams of page layouts                   |
//! | `namespaces`  | List all namespaces in a slab file                       |
//! | `completions` | Generate shell completions (bash, zsh, fish, etc.)       |
//!
//! ## Helpers
//!
//! - [`ProgressReporter`] — optional stderr progress line for long-running
//!   commands.
//! - [`make_writer_config`] — build a [`WriterConfig`] from optional CLI
//!   flags, falling back to defaults.
//! - [`write_with_buffer_rename`] — atomic write via a `.buffer` temp file
//!   followed by rename.

pub mod append;
pub mod check;
pub mod explain;
pub mod export;
pub mod get;
pub mod import;
pub mod analyze;
pub mod namespaces;
pub mod ordinal_range;
pub mod rewrite;

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::builder::ValueHint;
use clap::{Parser, Subcommand};
use clap_complete::Shell;
use crate::WriterConfig;

/// slab — slabtastic file maintenance tool
#[derive(Parser)]
#[command(name = "slab", version, about = "Slabtastic file maintenance tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Command {
    /// Display file structure and statistics.
    Analyze {
        /// Path to the slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Number of records/pages to sample for statistics.
        #[arg(long)]
        samples: Option<usize>,
        /// Percentage of records/pages to sample (0.0–100.0).
        #[arg(long)]
        sample_percent: Option<f64>,
        /// Namespace to operate on (uses default if omitted).
        #[arg(long)]
        namespace: Option<String>,
    },
    /// Check a slabtastic file for structural errors.
    Check {
        /// Path to the slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Namespace to check (checks default namespace if omitted).
        #[arg(long)]
        namespace: Option<String>,
    },
    /// Retrieve records by ordinal.
    Get {
        /// Path to the slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Ordinals to retrieve (individual or range specifiers).
        ordinals: Vec<String>,
        /// Output raw bytes instead of hex dump.
        #[arg(long)]
        raw: bool,
        /// Output bytes as hex with a space between each byte.
        #[arg(long)]
        as_hex: bool,
        /// Output bytes as base64.
        #[arg(long)]
        as_base64: bool,
        /// Namespace to operate on (uses default if omitted).
        #[arg(long)]
        namespace: Option<String>,
    },
    /// Rewrite a slabtastic file: reorder by ordinal and repack to new page settings.
    Rewrite {
        /// Input slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        input: String,
        /// Output slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        output: String,
        /// Ordinal range to rewrite (e.g., "100", "[5,10)", "0..99").
        #[arg(long)]
        range: Option<String>,
        /// Preferred page size in bytes.
        #[arg(long)]
        preferred_page_size: Option<u32>,
        /// Minimum page size in bytes.
        #[arg(long)]
        min_page_size: Option<u32>,
        /// Enable page alignment.
        #[arg(long)]
        page_alignment: bool,
        /// Show progress on stderr.
        #[arg(long)]
        progress: bool,
        /// Namespace to operate on (uses default if omitted).
        #[arg(long)]
        namespace: Option<String>,
    },
    /// Append records from stdin or a source file to an existing slab file.
    Append {
        /// Path to the existing slabtastic file to append to.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Optional source file to read records from (one per line).
        /// If omitted, reads from stdin.
        #[arg(long, value_hint = ValueHint::FilePath)]
        source: Option<String>,
        /// Preferred page size in bytes.
        #[arg(long)]
        preferred_page_size: Option<u32>,
        /// Minimum page size in bytes.
        #[arg(long)]
        min_page_size: Option<u32>,
        /// Enable page alignment.
        #[arg(long)]
        page_alignment: bool,
        /// Show progress on stderr.
        #[arg(long)]
        progress: bool,
        /// Namespace to operate on (uses default if omitted).
        #[arg(long)]
        namespace: Option<String>,
    },
    /// Import data from an external file format into a slab file.
    Import {
        /// Path to the target slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Path to the source file to import from.
        #[arg(value_hint = ValueHint::FilePath)]
        source: String,
        /// Force newline-delimited text format.
        #[arg(long)]
        newline_terminated_records: bool,
        /// Force null-terminated binary format.
        #[arg(long)]
        null_terminated_records: bool,
        /// Force slab format.
        #[arg(long)]
        slab_format: bool,
        /// Force JSON format.
        #[arg(long)]
        json: bool,
        /// Force JSONL (newline-delimited JSON) format.
        #[arg(long)]
        jsonl: bool,
        /// Force CSV format.
        #[arg(long)]
        csv: bool,
        /// Force TSV format.
        #[arg(long)]
        tsv: bool,
        /// Force YAML format.
        #[arg(long)]
        yaml: bool,
        /// Skip records that fail to parse instead of aborting.
        #[arg(long)]
        skip_malformed: bool,
        /// Strip trailing newlines from records before storing.
        #[arg(long)]
        strip_newline: bool,
        /// Preferred page size in bytes.
        #[arg(long)]
        preferred_page_size: Option<u32>,
        /// Minimum page size in bytes.
        #[arg(long)]
        min_page_size: Option<u32>,
        /// Enable page alignment.
        #[arg(long)]
        page_alignment: bool,
        /// Show progress on stderr.
        #[arg(long)]
        progress: bool,
        /// Namespace to operate on (uses default if omitted).
        #[arg(long)]
        namespace: Option<String>,
    },
    /// Export content from a slab file to an external format.
    Export {
        /// Path to the slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Output file path (stdout if omitted).
        #[arg(long, value_hint = ValueHint::FilePath)]
        output: Option<String>,
        /// Output format: raw, text, cstrings, hex, slab (default: raw).
        #[arg(long, value_enum)]
        format: Option<export::ExportFormatArg>,
        /// Ordinal range to export (e.g., "100", "[5,10)", "0..99").
        #[arg(long)]
        range: Option<String>,
        /// Preferred page size in bytes (for slab output).
        #[arg(long)]
        preferred_page_size: Option<u32>,
        /// Minimum page size in bytes (for slab output).
        #[arg(long)]
        min_page_size: Option<u32>,
        /// Enable page alignment (for slab output).
        #[arg(long)]
        page_alignment: bool,
        /// Show progress on stderr.
        #[arg(long)]
        progress: bool,
        /// Namespace to operate on (uses default if omitted).
        #[arg(long)]
        namespace: Option<String>,
    },
    /// List all namespaces in a slab file.
    Namespaces {
        /// Path to the slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
    },
    /// Generate shell completions.
    Completions {
        /// Shell to generate completions for (bash, zsh, fish, elvish, powershell).
        #[arg(long, value_enum)]
        shell: Shell,
    },
    /// Display block diagrams of page layouts.
    Explain {
        /// Path to the slabtastic file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Show specific page indices (0-based).
        #[arg(long, num_args = 1..)]
        pages: Option<Vec<usize>>,
        /// Show only pages belonging to a namespace.
        #[arg(long)]
        namespace: Option<String>,
        /// Show pages overlapping an ordinal range.
        #[arg(long)]
        ordinals: Option<String>,
    },
}

/// Optional progress reporter that prints record counts to stderr.
///
/// When enabled, a background thread prints a status line every second
/// or every 1M records (whichever comes first). Call [`inc`](Self::inc)
/// for each record processed and [`finish`](Self::finish) when done.
pub struct ProgressReporter {
    counter: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
    done: Arc<AtomicU64>,
}

impl ProgressReporter {
    /// Create and start a progress reporter.
    ///
    /// If `enabled` is `false`, returns a no-op reporter whose
    /// [`inc`](Self::inc) and [`finish`](Self::finish) methods are free.
    pub fn new(enabled: bool) -> Self {
        let counter = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicU64::new(0));
        let handle = if enabled {
            let c = Arc::clone(&counter);
            let d = Arc::clone(&done);
            Some(std::thread::spawn(move || {
                let mut last_printed = 0u64;
                let mut last_time = Instant::now();
                loop {
                    std::thread::sleep(Duration::from_millis(100));
                    let current = c.load(Ordering::Relaxed);
                    let elapsed = last_time.elapsed();
                    if elapsed >= Duration::from_secs(1)
                        || current - last_printed >= 1_000_000
                    {
                        eprint!("\r\x1b[2K{current} records processed...");
                        let _ = std::io::stderr().flush();
                        last_printed = current;
                        last_time = Instant::now();
                    }
                    if d.load(Ordering::Relaxed) != 0 {
                        break;
                    }
                }
                let final_count = c.load(Ordering::Relaxed);
                eprintln!("\r\x1b[2K{final_count} records processed.");
            }))
        } else {
            None
        };
        ProgressReporter {
            counter,
            handle,
            done,
        }
    }

    /// Increment the record counter by one.
    #[inline]
    pub fn inc(&self) {
        self.counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Signal completion and wait for the reporter thread to finish.
    pub fn finish(self) {
        self.done.store(1, Ordering::Relaxed);
        if let Some(h) = self.handle {
            let _ = h.join();
        }
    }
}

/// Build a `WriterConfig` from optional CLI flags, falling back to defaults.
pub fn make_writer_config(
    preferred_page_size: Option<u32>,
    min_page_size: Option<u32>,
    page_alignment: bool,
) -> crate::Result<WriterConfig> {
    let defaults = WriterConfig::default();
    WriterConfig::new(
        min_page_size.unwrap_or(defaults.min_page_size),
        preferred_page_size.unwrap_or(defaults.preferred_page_size),
        defaults.max_page_size,
        page_alignment,
    )
}

/// Write to a `.buffer` suffixed temporary file, then rename to the
/// target path on success.
///
/// The closure `f` receives the buffer path and should write the output
/// file there. If `f` succeeds, the buffer file is renamed to `target`.
/// If `f` fails, the buffer file is removed (best-effort) and the error
/// is propagated.
pub fn write_with_buffer_rename<F>(
    target: &str,
    f: F,
) -> std::result::Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce(&str) -> std::result::Result<(), Box<dyn std::error::Error>>,
{
    let buffer_path = format!("{target}.buffer");
    match f(&buffer_path) {
        Ok(()) => {
            std::fs::rename(&buffer_path, target)?;
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&buffer_path);
            Err(e)
        }
    }
}

/// Return `true` when the target path does not yet exist, meaning the
/// operation will create a new file (as opposed to appending).
pub fn is_new_file(path: &str) -> bool {
    !Path::new(path).exists()
}

/// Print the shell snippet that registers dynamic completions for `slab`.
fn completions(shell: Shell) {
    match shell {
        Shell::Bash => print!(r#"source <(COMPLETE=bash slab)"#),
        Shell::Zsh => print!(r#"source <(COMPLETE=zsh slab)"#),
        Shell::Fish => print!(r#"COMPLETE=fish slab | source"#),
        Shell::Elvish => print!(r#"eval (COMPLETE=elvish slab | slurp)"#),
        Shell::PowerShell => {
            print!(r#"(& {{ $env:COMPLETE="powershell"; slab }}) | Invoke-Expression"#)
        }
        _ => eprintln!("Unsupported shell: {:?}", shell),
    }
}

/// Dispatch to the appropriate subcommand.
pub fn run(cli: Cli) -> std::result::Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Analyze {
            file,
            samples,
            sample_percent,
            namespace,
        } => analyze::run(&file, samples, sample_percent, &namespace)?,
        Command::Check { file, namespace } => check::run(&file, &namespace)?,
        Command::Get {
            file,
            ordinals,
            raw,
            as_hex,
            as_base64,
            namespace,
        } => get::run(&file, &ordinals, raw, as_hex, as_base64, &namespace)?,
        Command::Append {
            file,
            source,
            preferred_page_size,
            min_page_size,
            page_alignment,
            progress,
            namespace,
        } => append::run(
            &file,
            source.as_deref(),
            preferred_page_size,
            min_page_size,
            page_alignment,
            progress,
            &namespace,
        )?,
        Command::Import {
            file,
            source,
            newline_terminated_records,
            null_terminated_records,
            slab_format,
            json,
            jsonl,
            csv,
            tsv,
            yaml,
            skip_malformed,
            strip_newline,
            preferred_page_size,
            min_page_size,
            page_alignment,
            progress,
            namespace,
        } => import::run(
            &file,
            &source,
            newline_terminated_records,
            null_terminated_records,
            slab_format,
            json,
            jsonl,
            csv,
            tsv,
            yaml,
            skip_malformed,
            strip_newline,
            preferred_page_size,
            min_page_size,
            page_alignment,
            progress,
            &namespace,
        )?,
        Command::Export {
            file,
            output,
            format,
            range,
            preferred_page_size,
            min_page_size,
            page_alignment,
            progress,
            namespace,
        } => export::run(&export::ExportConfig {
            file: &file,
            output: output.as_deref(),
            format: format.map(export::ExportFormat::from),
            preferred_page_size,
            min_page_size,
            page_alignment,
            progress,
            namespace: &namespace,
            range: range.as_deref(),
        })?,
        Command::Rewrite {
            input,
            output,
            range,
            preferred_page_size,
            min_page_size,
            page_alignment,
            progress,
            namespace,
        } => rewrite::run(
            &input,
            &output,
            range.as_deref(),
            preferred_page_size,
            min_page_size,
            page_alignment,
            progress,
            &namespace,
        )?,
        Command::Completions { shell } => completions(shell),
        Command::Namespaces { file } => namespaces::run(&file)?,
        Command::Explain {
            file,
            pages,
            namespace,
            ordinals,
        } => explain::run(&file, &pages, &namespace, &ordinals)?,
    }
    Ok(())
}
