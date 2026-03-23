// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks explore` — interactive visualization and exploration commands.
//!
//! These are user-facing TUI/interactive tools that don't belong in the
//! pipeline command registry (they're not composable pipeline steps).
//! The source argument accepts either a local file path or a
//! `dataset:profile:facet` specifier from the catalog.

use std::ffi::OsStr;
use std::path::PathBuf;

use clap::{Args, Subcommand};
use clap_complete::engine::{ArgValueCompleter, CompletionCandidate};

/// Interactive data exploration and visualization
#[derive(Args)]
#[command(disable_help_subcommand = true)]
pub struct ExploreArgs {
    #[command(subcommand)]
    pub command: ExploreCommand,
}

#[derive(Subcommand)]
pub enum ExploreCommand {
    /// Interactive data exploration shell for vector files
    DataShell {
        /// Data source: local file path or dataset:profile:facet from catalog
        #[arg(add = ArgValueCompleter::new(source_completer))]
        source: String,

        /// Trailing args passed as command options
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Generate scatter/distribution plots for vectors
    Plot {
        /// Data source: local file path or dataset:profile:facet from catalog
        #[arg(add = ArgValueCompleter::new(source_completer))]
        source: String,

        /// Trailing args passed as command options
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Interactive PCA scatter plot — project vectors onto principal components
    Pca {
        /// Data source: local file path (fvec, mvec, dvec)
        #[arg(add = ArgValueCompleter::new(source_completer))]
        source: String,

        /// Number of vectors to sample for covariance estimation (default: 50000)
        #[arg(long, default_value = "50000")]
        sample: usize,

        /// Random seed for sampling
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Trailing args
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

/// Recognized data file extensions for explore commands.
const DATA_EXTENSIONS: &[&str] = &[
    "fvec", "fvecs", "ivec", "ivecs", "mvec", "mvecs",
    "bvec", "bvecs", "dvec", "dvecs", "svec", "svecs",
    "npy", "slab", "parquet",
];

/// Completion for the source argument: local data files + catalog datasets.
fn source_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let prefix = current.to_string_lossy();
    let mut candidates = Vec::new();

    // 1. Local files matching data extensions
    candidates.extend(local_file_candidates(&prefix));

    // 2. Catalog dataset:profile entries
    candidates.extend(catalog_candidates(&prefix));

    candidates
}

/// Find local files with recognized data extensions matching the prefix.
fn local_file_candidates(prefix: &str) -> Vec<CompletionCandidate> {
    let mut candidates = Vec::new();

    // Determine directory to scan and filename prefix
    let path = std::path::Path::new(prefix);
    let (dir, file_prefix) = if prefix.ends_with('/') || prefix.ends_with(std::path::MAIN_SEPARATOR) {
        (path.to_path_buf(), String::new())
    } else if let Some(parent) = path.parent() {
        let dir = if parent.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            parent.to_path_buf()
        };
        let fname = path.file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        (dir, fname)
    } else {
        (PathBuf::from("."), prefix.to_string())
    };

    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Skip hidden files
            if name_str.starts_with('.') {
                continue;
            }

            // Include directories (for navigation) and data files
            let entry_path = entry.path();
            if entry_path.is_dir() {
                if name_str.starts_with(&*file_prefix) {
                    let display = if dir == PathBuf::from(".") {
                        format!("{}/", name_str)
                    } else {
                        format!("{}/{}/", dir.display(), name_str)
                    };
                    candidates.push(CompletionCandidate::new(display));
                }
            } else if name_str.starts_with(&*file_prefix) {
                let ext = entry_path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                if DATA_EXTENSIONS.contains(&ext) {
                    let display = if dir == PathBuf::from(".") {
                        name_str.to_string()
                    } else {
                        format!("{}/{}", dir.display(), name_str)
                    };
                    candidates.push(
                        CompletionCandidate::new(display)
                            .help(Some(ext.to_string().into()))
                    );
                }
            }
        }
    }

    candidates
}

/// Find catalog dataset:profile entries matching the prefix.
fn catalog_candidates(prefix: &str) -> Vec<CompletionCandidate> {
    // Only search catalog if the prefix doesn't look like a file path
    if prefix.contains('/') || prefix.contains('.') {
        return Vec::new();
    }

    let prefix_lower = prefix.to_lowercase();
    let entries = crate::datasets::filter::completion_entries();
    let mut candidates = Vec::new();

    for entry in entries {
        let profiles = entry.profile_names();
        for profile in &profiles {
            let candidate = format!("{}:{}", entry.name, profile);
            if candidate.to_lowercase().starts_with(&prefix_lower)
                || entry.name.to_lowercase().starts_with(&prefix_lower)
            {
                let help = entry.layout.attributes.as_ref()
                    .and_then(|a| a.distance_function.as_deref())
                    .unwrap_or("");
                candidates.push(
                    CompletionCandidate::new(&candidate)
                        .help(Some(help.to_string().into()))
                );
            }
        }
    }

    candidates
}

/// Resolve a source specifier to a local file path.
///
/// Accepts:
/// - A local file path (returned as-is if it exists)
/// - A `dataset:profile` or `dataset:profile:facet` catalog specifier
///   (resolved via catalog, prebuffered if needed)
///
/// Default facet is `base_vectors` when not specified.
fn resolve_source(source: &str) -> PathBuf {
    // If it looks like a file path and exists, use directly
    let as_path = std::path::Path::new(source);
    if as_path.exists() {
        return as_path.to_path_buf();
    }

    // If it contains a slash or a recognized extension, treat as a path
    if source.contains('/') || source.contains('.') {
        eprintln!("Error: file not found: {}", source);
        std::process::exit(1);
    }

    // Parse as dataset:profile[:facet]
    let parts: Vec<&str> = source.split(':').collect();
    let (dataset_name, profile_name, facet_name) = match parts.len() {
        1 => (parts[0], "default", "base_vectors"),
        2 => (parts[0], parts[1], "base_vectors"),
        3 => (parts[0], parts[1], parts[2]),
        _ => {
            eprintln!("Error: invalid source specifier '{}'. Use dataset:profile[:facet]", source);
            std::process::exit(1);
        }
    };

    // Resolve via catalog
    let sources = crate::catalog::sources::CatalogSources::new().configure_default();
    if sources.is_empty() {
        eprintln!("Error: '{}' is not a local file and no catalogs are configured.", source);
        eprintln!("Create ~/.config/vectordata/catalogs.yaml or use a local file path.");
        std::process::exit(1);
    }

    let catalog = crate::catalog::resolver::Catalog::of(&sources);
    let entry = match catalog.find_exact(dataset_name) {
        Some(e) => e,
        None => {
            eprintln!("Error: dataset '{}' not found in catalog.", dataset_name);
            catalog.list_datasets(dataset_name);
            std::process::exit(1);
        }
    };

    let profile = match entry.layout.profiles.profile(profile_name) {
        Some(p) => p,
        None => {
            eprintln!("Error: profile '{}' not found in '{}'. Available: {}",
                profile_name, entry.name, entry.profile_names().join(", "));
            std::process::exit(1);
        }
    };

    let view = match profile.view(facet_name) {
        Some(v) => v,
        None => {
            eprintln!("Error: facet '{}' not found in {}:{}. Available: {}",
                facet_name, entry.name, profile_name, profile.view_names().join(", "));
            std::process::exit(1);
        }
    };

    let source_path = &view.source.path;
    let base_url = entry.path.rsplit_once('/').map(|(base, _)| base).unwrap_or("");

    // Check local cache first
    let cache_dir = dirs_cache_dir().join(&entry.name);
    let cached_path = cache_dir.join(source_path);

    if cached_path.exists() {
        eprintln!("Using cached: {}", cached_path.display());
        return cached_path;
    }

    // Download to cache
    let full_url = if source_path.starts_with("http://") || source_path.starts_with("https://") {
        source_path.clone()
    } else {
        format!("{}/{}", base_url, source_path)
    };

    if !full_url.starts_with("http://") && !full_url.starts_with("https://") {
        // Local source relative to catalog entry
        let local_src = std::path::Path::new(&full_url);
        if local_src.exists() {
            return local_src.to_path_buf();
        }
        eprintln!("Error: source not available: {}", full_url);
        std::process::exit(1);
    }

    eprintln!("Downloading {}:{} ({})...", entry.name, profile_name, facet_name);
    if let Some(parent) = cached_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    match crate::datasets::prebuffer::download_file(&full_url, &cached_path) {
        Ok(size) => {
            eprintln!("Downloaded {} bytes to {}", size, cached_path.display());
            cached_path
        }
        Err(e) => {
            eprintln!("Error: failed to download {}: {}", full_url, e);
            std::process::exit(1);
        }
    }
}

/// Get the configured vectordata cache directory from settings.yaml.
fn dirs_cache_dir() -> PathBuf {
    crate::pipeline::commands::config::configured_cache_dir()
}

/// Format-agnostic vector reader that returns f64 values.
///
/// Wraps any supported vector format (fvec, mvec, dvec) and converts
/// individual vectors to f64 on read. No bulk file conversion needed.
enum AnyVectorReader {
    F32(vectordata::io::MmapVectorReader<f32>),
    F16(vectordata::io::MmapVectorReader<half::f16>),
}

impl AnyVectorReader {
    /// Open a vector file, auto-detecting format from extension.
    fn open(path: &std::path::Path) -> Self {
        use vectordata::io::MmapVectorReader;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "fvec" | "fvecs" => {
                AnyVectorReader::F32(MmapVectorReader::<f32>::open_fvec(path).unwrap_or_else(|e| {
                    eprintln!("Error: failed to open {}: {}", path.display(), e);
                    std::process::exit(1);
                }))
            }
            "mvec" | "mvecs" => {
                AnyVectorReader::F16(MmapVectorReader::<half::f16>::open_mvec(path).unwrap_or_else(|e| {
                    eprintln!("Error: failed to open {}: {}", path.display(), e);
                    std::process::exit(1);
                }))
            }
            _ => {
                eprintln!("Error: unsupported format '.{}' for visualization. Use fvec or mvec.", ext);
                std::process::exit(1);
            }
        }
    }

    fn count(&self) -> usize {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => <vectordata::io::MmapVectorReader<f32> as VectorReader<f32>>::count(r),
            AnyVectorReader::F16(r) => <vectordata::io::MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(r),
        }
    }

    fn dim(&self) -> usize {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => <vectordata::io::MmapVectorReader<f32> as VectorReader<f32>>::dim(r),
            AnyVectorReader::F16(r) => <vectordata::io::MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(r),
        }
    }

    /// Read a single vector as f64 values. Converts from native format on the fly.
    fn get_f64(&self, index: usize) -> Option<Vec<f64>> {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => {
                r.get(index).ok().map(|v| v.iter().map(|&x| x as f64).collect())
            }
            AnyVectorReader::F16(r) => {
                r.get(index).ok().map(|v| v.iter().map(|x| x.to_f64()).collect())
            }
        }
    }

    /// Read a single vector as f32 values (for simsimd hot paths).
    /// For f32 sources this is zero-copy; for f16 sources this converts.
    fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => {
                r.get(index).ok().map(|v| v.to_vec())
            }
            AnyVectorReader::F16(r) => {
                r.get(index).ok().map(|v| v.iter().map(|x| x.to_f32()).collect())
            }
        }
    }
}

/// Dispatch a visualize subcommand.
///
/// Visualize commands create a standalone ratatui TUI for interactive
/// display. The user presses 'q' to exit.
pub fn run(args: ExploreArgs) {
    match args.command {
        ExploreCommand::DataShell { source, args: extra } => {
            run_pipeline_command(
                crate::pipeline::commands::analyze_explore::factory(),
                &source,
                &extra,
            );
        }
        ExploreCommand::Plot { source, args: extra } => {
            run_interactive_plot(&source, &extra);
        }
        ExploreCommand::Pca { source, sample, seed, args: _ } => {
            run_interactive_pca(&source, sample, seed);
        }
    }
}

/// Run a pipeline command with a plain (non-TUI) sink.
fn run_pipeline_command(
    mut cmd: Box<dyn crate::pipeline::command::CommandOp>,
    source: &str,
    extra_args: &[String],
) {
    use crate::pipeline::command::{Options, StreamContext, Status};
    use crate::pipeline::progress::ProgressLog;
    use crate::pipeline::resource::ResourceGovernor;
    use indexmap::IndexMap;

    let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let mut opts = Options::new();
    opts.set("source", source);
    for arg in extra_args {
        if let Some(kv) = arg.strip_prefix("--") {
            if let Some((k, v)) = kv.split_once('=') {
                opts.set(k, v);
            } else {
                opts.set(kv, "true");
            }
        }
    }

    let mut ctx = StreamContext {
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace: workspace.clone(),
        scratch: workspace.join(".scratch"),
        cache: workspace.join(".cache"),
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads: 0,
        step_id: String::new(),
        governor: ResourceGovernor::default_governor(),
        ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::PlainSink::new())),
        status_interval: std::time::Duration::from_secs(1),
    };

    let result = cmd.execute(&opts, &mut ctx);
    if result.status == Status::Error {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
}

/// Run an interactive ratatui plot viewer.
///
/// Loads vector data, computes histogram bins, then renders using
/// ratatui's `BarChart` widget in a standalone TUI. Press 'q' to exit,
/// left/right arrows to change dimension.
fn run_interactive_plot(source: &str, extra_args: &[String]) {
    use crossterm::{
        event::{self, Event, KeyCode},
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        execute,
    };
    use ratatui::{
        backend::CrosstermBackend,
        layout::{Constraint, Direction, Layout},
        style::{Color, Style},
        text::{Line, Span},
        widgets::{Bar, BarChart, BarGroup, Block, Borders, Paragraph},
        Terminal,
    };

    // Parse options
    let mut dims_str = "0".to_string();
    let mut sample: usize = 50_000;
    let mut num_bins: Option<usize> = None;
    for arg in extra_args {
        if let Some(kv) = arg.strip_prefix("--") {
            if let Some((k, v)) = kv.split_once('=') {
                match k {
                    "dimensions" => dims_str = v.to_string(),
                    "sample" => { sample = v.parse().unwrap_or(sample); }
                    "bins" => { num_bins = v.parse().ok(); }
                    _ => {}
                }
            }
        }
    }

    let source_path = resolve_source(source);
    let reader = AnyVectorReader::open(&source_path);

    let total = reader.count();
    let dim = reader.dim();
    let mut current_sample = sample.min(total);

    let initial_dims: Vec<usize> = dims_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let mut current_dim: usize = initial_dims.first().copied().unwrap_or(0).min(dim - 1);

    let mut dim_cache: std::collections::HashMap<usize, Vec<f64>> = std::collections::HashMap::new();
    let mut needs_resample = true;

    // Enter TUI
    enable_raw_mode().unwrap();
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).unwrap();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    let filename = source_path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut show_help = false;

    loop {
        // Resample if sample size changed
        if needs_resample {
            dim_cache.clear();
            let indices = sample_indices(total, current_sample, 42);
            let n = indices.len();

            eprintln!("  Sampling {} vectors across {} dimensions...", n, dim);
            let mut all_values: Vec<Vec<f64>> = vec![Vec::with_capacity(n); dim];
            let start = std::time::Instant::now();
            let mut last_progress = std::time::Instant::now();
            for (loaded, &i) in indices.iter().enumerate() {
                if let Some(v) = reader.get_f64(i) {
                    for d in 0..dim {
                        all_values[d].push(v[d]);
                    }
                }
                if last_progress.elapsed() >= std::time::Duration::from_millis(250) || loaded + 1 == n {
                    let pct = (loaded + 1) * 100 / n;
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = if elapsed > 0.0 { (loaded + 1) as f64 / elapsed } else { 0.0 };
                    eprint!("\r  Sampling: {}/{} ({}%) {:.0} vec/s   ", loaded + 1, n, pct, rate);
                    last_progress = std::time::Instant::now();
                }
            }
            eprintln!("\r  Sampled {} vectors in {:.1}s                    ", n, start.elapsed().as_secs_f64());
            for d in 0..dim {
                dim_cache.insert(d, std::mem::take(&mut all_values[d]));
            }
            needs_resample = false;
        }

        let values = dim_cache.get(&current_dim).cloned().unwrap_or_default();

        // Size bins to fill the available terminal width (minus borders)
        let term_width = crossterm::terminal::size().map(|(w, _)| w as usize).unwrap_or(80);
        let bins = num_bins.unwrap_or_else(|| {
            // Each bar takes 1 column with bar_gap(0); subtract 2 for borders
            (term_width.saturating_sub(4)).max(10)
        });

        // Compute histogram
        let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let bin_width = if range > 0.0 { range / bins as f64 } else { 1.0 };

        let mut counts = vec![0u64; bins];
        for &v in &values {
            let idx = ((v - min_val) / bin_width) as usize;
            let idx = idx.min(bins - 1);
            counts[idx] += 1;
        }

        let max_count = counts.iter().copied().max().unwrap_or(1);

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),
                    Constraint::Min(5),
                    Constraint::Length(1),
                ])
                .split(frame.area());

            // Title
            let title = format!(
                " {} — dim {}/{} — {}/{} sampled ({}%)",
                filename, current_dim, dim, current_sample, total,
                current_sample * 100 / total
            );
            frame.render_widget(
                Paragraph::new(Span::styled(title, Style::default().fg(Color::Cyan))),
                chunks[0],
            );

            if show_help {
                let help_text = vec![
                    Line::from(Span::styled(" Keyboard Shortcuts", Style::default().fg(Color::Cyan))),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(" ←  / h   ", Style::default().fg(Color::Yellow)),
                        Span::raw("Previous dimension"),
                    ]),
                    Line::from(vec![
                        Span::styled(" →  / l   ", Style::default().fg(Color::Yellow)),
                        Span::raw("Next dimension"),
                    ]),
                    Line::from(vec![
                        Span::styled(" + / =    ", Style::default().fg(Color::Yellow)),
                        Span::raw("Double sample size"),
                    ]),
                    Line::from(vec![
                        Span::styled(" -        ", Style::default().fg(Color::Yellow)),
                        Span::raw("Halve sample size"),
                    ]),
                    Line::from(vec![
                        Span::styled(" q  / Esc ", Style::default().fg(Color::Yellow)),
                        Span::raw("Quit"),
                    ]),
                    Line::from(vec![
                        Span::styled(" ?        ", Style::default().fg(Color::Yellow)),
                        Span::raw("Toggle this help"),
                    ]),
                ];
                frame.render_widget(
                    Paragraph::new(help_text)
                        .block(Block::default().borders(Borders::ALL).title(" Help ")),
                    chunks[1],
                );
            } else {
                // Bar chart
                // Label ~5 evenly spaced bins for readability
                let label_interval = (bins / 5).max(1);
                let bars: Vec<Bar> = counts.iter().enumerate().map(|(i, &count)| {
                    let label = if i % label_interval == 0 || i == bins - 1 {
                        format!("{:.2}", min_val + (i as f64 + 0.5) * bin_width)
                    } else {
                        String::new()
                    };
                    Bar::default()
                        .value(count)
                        .label(label.into())
                        .style(Style::default().fg(Color::Green))
                }).collect();

                let chart = BarChart::default()
                    .block(Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" Histogram — dimension {} ", current_dim)))
                    .data(BarGroup::default().bars(&bars))
                    .bar_gap(0)
                    .max(max_count);

                frame.render_widget(chart, chunks[1]);
            }

            // Footer
            let footer = format!(
                " ←/→ dim | +/- sample | q quit | ? help | {} of {} | [{:.2}, {:.2}]",
                current_sample, total, min_val, max_val
            );
            frame.render_widget(
                Paragraph::new(Span::styled(footer, Style::default().fg(Color::DarkGray))),
                chunks[2],
            );
        }).unwrap();

        // Handle input
        if event::poll(std::time::Duration::from_millis(100)).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('?') => { show_help = !show_help; }
                    KeyCode::Right | KeyCode::Char('l') => {
                        if current_dim + 1 < dim {
                            current_dim += 1;
                        }
                    }
                    KeyCode::Left | KeyCode::Char('h') => {
                        if current_dim > 0 {
                            current_dim -= 1;
                        }
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') if current_sample < total => {
                        current_sample = (current_sample * 2).min(total);
                        needs_resample = true;
                    }
                    KeyCode::Char('-') if current_sample > 1000 => {
                        current_sample = (current_sample / 2).max(1000);
                        needs_resample = true;
                    }
                    _ => {}
                }
            }
        }
    }

    // Restore terminal
    disable_raw_mode().unwrap();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
}

// ---------------------------------------------------------------------------
// PCA visualization
// ---------------------------------------------------------------------------

/// Cache file for PCA results: stores mean, eigenvectors, eigenvalues,
/// and projected points so recomputation is avoided.
const PCA_CACHE_FILE: &str = "pca_projection.bin";

/// Partition size for incremental PCA (number of vectors per partition).
const PCA_PARTITION_SIZE: usize = 1_000_000;

/// Interactive PCA scatter plot with incremental streaming.
///
/// Processes the vector file in partitions of `PCA_PARTITION_SIZE` vectors.
/// Each partition's partial statistics (count, mean, scatter contribution)
/// are cached in `.cache/pca/partition_N.bin`. The TUI updates after each
/// partition, showing a progressively refined scatter plot.
///
/// On subsequent runs, cached partitions are reloaded instantly and only
/// new/changed partitions are recomputed.
fn run_interactive_pca(source: &str, initial_sample: usize, seed: u64) {
    use crossterm::{
        event::{self, Event, KeyCode},
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        execute,
    };
    use ratatui::{
        backend::CrosstermBackend,
        layout::{Constraint, Direction, Layout},
        style::{Color, Style},
        text::{Line, Span},
        widgets::{Block, Borders, Paragraph, canvas::{Canvas, Points}},
        Terminal,
    };

    let source_path = resolve_source(source);
    let reader = AnyVectorReader::open(&source_path);
    let total = reader.count();
    let dim = reader.dim();
    let filename = source_path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let cache_dir = source_path.parent()
        .unwrap_or(std::path::Path::new("."))
        .join(".cache")
        .join("pca");
    let _ = std::fs::create_dir_all(&cache_dir);
    let cache_path = cache_dir.join(PCA_CACHE_FILE);

    // Enter TUI
    enable_raw_mode().unwrap();
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).unwrap();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    /// Number of eigenvalues to compute for the spectrum view.
    /// Each eigenvalue requires 30 full passes over the sampled data,
    /// so keep this reasonable. Users can increase sample size but
    /// this count stays fixed.
    const NUM_EIGENVALUES: usize = 10;

    let mut current_sample = initial_sample.min(total);
    let mut show_help = false;
    let aborted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    install_abort_handler(aborted.clone());
    let mut projected_3d: Vec<[f64; 3]> = Vec::new();
    let mut eigenvalues: Vec<f64> = vec![0.0; NUM_EIGENVALUES];
    let mut eigenvectors_top3: Vec<Vec<f64>> = Vec::new();
    let mut needs_compute = true;
    let mut rot_y: f64 = 0.0;
    let mut rot_x: f64 = 0.0;
    // View mode: 0=scatter, 1=eigenvalue spectrum, 2=cumulative variance, 3=loadings
    let mut view_mode: usize = 0;
    const NUM_VIEWS: usize = 4;

    // Try loading cache first
    if is_cache_fresh(&cache_path, &source_path) {
        if let Some(cached) = load_pca_3d_cache(&cache_path) {
            projected_3d = cached.0;
            eigenvalues = cached.1;
            current_sample = projected_3d.len();
            needs_compute = false;
        }
    }

    loop {
        if needs_compute {
            let indices = sample_indices(total, current_sample, seed);
            let n = indices.len();
            let start_time = std::time::Instant::now();

            // Phase 1: Compute mean with progress
            let mut mean = vec![0.0f64; dim];
            let mut was_aborted = false;
            {
                let mut partial_sum = vec![0.0f64; dim];
                let mut last_draw = std::time::Instant::now();
                for (done, &i) in indices.iter().enumerate() {
                    if done % 4096 == 0 {
                        if aborted.load(std::sync::atomic::Ordering::Relaxed) {
                            was_aborted = true;
                            break;
                        }
                        if crossterm::event::poll(std::time::Duration::ZERO).unwrap_or(false) {
                            if let Ok(crossterm::event::Event::Key(key)) = crossterm::event::read() {
                                if matches!(key.code,
                                    crossterm::event::KeyCode::Esc |
                                    crossterm::event::KeyCode::Char('q') |
                                    crossterm::event::KeyCode::Char('c')
                                ) {
                                    was_aborted = true;
                                    break;
                                }
                            }
                        }
                    }
                    if let Some(v) = reader.get_f64(i) {
                        for d in 0..dim {
                            partial_sum[d] += v[d];
                        }
                    }
                    if last_draw.elapsed() >= std::time::Duration::from_millis(250) || done + 1 == n {
                        let pct = (done + 1) * 100 / n;
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let rate = (done + 1) as f64 / elapsed;
                        let eta = if rate > 0.0 { (n - done - 1) as f64 / rate } else { 0.0 };
                        terminal.draw(|frame| {
                            let area = frame.area();
                            let lines = vec![
                                Line::from(Span::styled(
                                    format!("  PCA Phase 1/3: Computing mean — {}/{} ({}%)", done + 1, n, pct),
                                    Style::default().fg(Color::Yellow),
                                )),
                                Line::from(Span::styled(
                                    format!("  {:.0} vec/s — eta {:.0}s", rate, eta),
                                    Style::default().fg(Color::DarkGray),
                                )),
                            ];
                            frame.render_widget(Paragraph::new(lines), area);
                        }).unwrap();
                        last_draw = std::time::Instant::now();
                    }
                }
                for d in 0..dim {
                    mean[d] = partial_sum[d] / n as f64;
                }
            }

            if was_aborted { needs_compute = false; continue; }

            // Phase 2: Eigenvectors with progress
            let k = NUM_EIGENVALUES.min(dim);
            let (eigvecs, eigvals) = {
                let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
                let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);
                let mut eigenvalues_out: Vec<f64> = Vec::with_capacity(k);
                use simsimd::SpatialSimilarity;

                'eigen: for ki in 0..k {
                    if aborted.load(std::sync::atomic::Ordering::Relaxed) {
                        was_aborted = true;
                        break 'eigen;
                    }

                    // Draw phase 2 header immediately
                    terminal.draw(|frame| {
                        let area = frame.area();
                        let lines = vec![
                            Line::from(Span::styled(
                                format!("  PCA Phase 2/3: Computing eigenvector {}/{} ...", ki + 1, k),
                                Style::default().fg(Color::Yellow),
                            )),
                            Line::from(Span::styled(
                                format!("  {} iterations × {} vectors per component", 30, n),
                                Style::default().fg(Color::DarkGray),
                            )),
                        ];
                        frame.render_widget(Paragraph::new(lines), area);
                    }).unwrap();

                    let mut v: Vec<f64> = (0..dim).map(|d| ((d * 7 + 13) % 97) as f64 - 48.0).collect();
                    normalize(&mut v);

                    for iter in 0..30 {
                        let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                        let mut new_v = vec![0.0f64; dim];
                        let mut centered = vec![0.0f32; dim];

                        let mut last_draw = std::time::Instant::now();
                        for (done, &idx) in indices.iter().enumerate() {
                            // Check for abort every 4096 vectors via both signal and key poll
                            if done % 4096 == 0 {
                                if aborted.load(std::sync::atomic::Ordering::Relaxed) {
                                    was_aborted = true;
                                    break;
                                }
                                // Poll for Escape or Ctrl-C via crossterm
                                if crossterm::event::poll(std::time::Duration::ZERO).unwrap_or(false) {
                                    if let Ok(crossterm::event::Event::Key(key)) = crossterm::event::read() {
                                        if matches!(key.code,
                                            crossterm::event::KeyCode::Esc |
                                            crossterm::event::KeyCode::Char('q') |
                                            crossterm::event::KeyCode::Char('c')
                                        ) {
                                            was_aborted = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            if let Some(x) = reader.get_f32(idx) {
                                for d in 0..dim {
                                    centered[d] = x[d] - mean_f32[d];
                                }
                                let dot = <f32 as SpatialSimilarity>::dot(&centered, &v_f32)
                                    .unwrap_or(0.0) as f64;
                                for d in 0..dim {
                                    new_v[d] += centered[d] as f64 * dot;
                                }
                            }
                            if last_draw.elapsed() >= std::time::Duration::from_millis(250) {
                                let phase_pct = ((ki * 30 + iter) * 100) / (k * 30);
                                terminal.draw(|frame| {
                                    let area = frame.area();
                                    let lines = vec![
                                        Line::from(Span::styled(
                                            format!("  PCA Phase 2/3: Eigenvectors — component {}/{}, iteration {}/30",
                                                ki + 1, k, iter + 1),
                                            Style::default().fg(Color::Yellow),
                                        )),
                                        Line::from(Span::styled(
                                            format!("  Overall: {}% — {}/{} vectors this iteration",
                                                phase_pct, done + 1, n),
                                            Style::default().fg(Color::DarkGray),
                                        )),
                                    ];
                                    frame.render_widget(Paragraph::new(lines), area);
                                }).unwrap();
                                last_draw = std::time::Instant::now();
                            }
                        }

                        for d in 0..dim { new_v[d] /= n as f64; }
                        for prev in &eigenvectors {
                            let proj: f64 = new_v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                            for d in 0..dim { new_v[d] -= proj * prev[d]; }
                        }
                        normalize(&mut new_v);
                        v = new_v;
                    }

                    // Eigenvalue
                    let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                    let mut ev_sum = 0.0f64;
                    let mut centered = vec![0.0f32; dim];
                    for &idx in &indices {
                        if let Some(x) = reader.get_f32(idx) {
                            for d in 0..dim { centered[d] = x[d] - mean_f32[d]; }
                            let dot = <f32 as SpatialSimilarity>::dot(&centered, &v_f32)
                                .unwrap_or(0.0) as f64;
                            ev_sum += dot * dot;
                        }
                    }
                    eigenvalues_out.push(ev_sum / n as f64);
                    eigenvectors.push(v);
                }
                (eigenvectors, eigenvalues_out)
            };

            if was_aborted { needs_compute = false; continue; }

            // Phase 3: Project with progress
            terminal.draw(|frame| {
                let area = frame.area();
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        format!("  PCA Phase 3/3: Projecting {} vectors onto 3 components...", n),
                        Style::default().fg(Color::Yellow),
                    )),
                    area,
                );
            }).unwrap();
            projected_3d = project_vectors_3d(&reader, &indices, &mean, &eigvecs, dim);
            eigenvalues = eigvals;
            eigenvectors_top3 = eigvecs.into_iter().take(3).collect();

            let elapsed = start_time.elapsed().as_secs_f64();
            terminal.draw(|frame| {
                let area = frame.area();
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        format!("  PCA complete: {} vectors, {} components in {:.1}s", n, eigenvalues.len(), elapsed),
                        Style::default().fg(Color::Green),
                    )),
                    area,
                );
            }).unwrap();

            let _ = save_pca_3d_cache(&cache_path, &projected_3d, &eigenvalues, dim, total, &filename);
            needs_compute = false;
        }

        // Apply rotation to 3D points → 2D projection
        let projected_2d = rotate_and_project(&projected_3d, rot_y, rot_x);
        let (x_min, x_max, y_min, y_max) = compute_bounds(&projected_2d);
        let total_var: f64 = eigenvalues.iter().sum();
        let var_pcts: Vec<f64> = eigenvalues.iter()
            .map(|&v| if total_var > 0.0 { 100.0 * v / total_var } else { 0.0 })
            .collect();
        let coverage_pct = current_sample * 100 / total;

        let points_owned: Vec<(f64, f64)> = projected_2d;
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),
                    Constraint::Min(10),
                    Constraint::Length(1),
                ])
                .split(frame.area());

            let title = format!(
                " PCA: {} — {} dims — {}/{} ({}%) — PC1:{:.1}% PC2:{:.1}% PC3:{:.1}% — rot Y:{:.0}° X:{:.0}°",
                filename, dim, current_sample, total, coverage_pct,
                var_pcts[0], var_pcts[1], var_pcts.get(2).unwrap_or(&0.0),
                rot_y.to_degrees(), rot_x.to_degrees(),
            );
            frame.render_widget(
                Paragraph::new(Span::styled(title, Style::default().fg(Color::Cyan))),
                chunks[0],
            );

            let view_names = ["1:Scatter", "2:Eigenvalues", "3:Cumulative", "4:Loadings"];
            if show_help {
                let help_text = vec![
                    Line::from(Span::styled(" Keyboard Shortcuts", Style::default().fg(Color::Cyan))),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(" 1-4 / Tab", Style::default().fg(Color::Yellow)),
                        Span::raw("  Switch view (scatter/eigenvalues/cumulative/loadings)"),
                    ]),
                    Line::from(vec![
                        Span::styled(" ←/→      ", Style::default().fg(Color::Yellow)),
                        Span::raw("  Rotate around Y axis (scatter view)"),
                    ]),
                    Line::from(vec![
                        Span::styled(" ↑/↓      ", Style::default().fg(Color::Yellow)),
                        Span::raw("  Rotate around X axis (scatter view)"),
                    ]),
                    Line::from(vec![
                        Span::styled(" r        ", Style::default().fg(Color::Yellow)),
                        Span::raw("  Reset rotation"),
                    ]),
                    Line::from(vec![
                        Span::styled(" Space    ", Style::default().fg(Color::Yellow)),
                        Span::raw("  Double sample size and recompute"),
                    ]),
                    Line::from(vec![
                        Span::styled(" +        ", Style::default().fg(Color::Yellow)),
                        Span::raw("  Increase sample by 50%"),
                    ]),
                    Line::from(vec![
                        Span::styled(" a        ", Style::default().fg(Color::Yellow)),
                        Span::raw("  Use all vectors (full dataset)"),
                    ]),
                    Line::from(vec![
                        Span::styled(" q / Esc  ", Style::default().fg(Color::Yellow)),
                        Span::raw("  Quit"),
                    ]),
                    Line::from(vec![
                        Span::styled(" ?        ", Style::default().fg(Color::Yellow)),
                        Span::raw("  Toggle this help"),
                    ]),
                ];
                frame.render_widget(
                    Paragraph::new(help_text)
                        .block(Block::default().borders(Borders::ALL).title(" Help ")),
                    chunks[1],
                );
            } else {
                match view_mode {
                    0 => {
                        // 3D scatter plot
                        let canvas = Canvas::default()
                            .block(Block::default()
                                .borders(Borders::ALL)
                                .title(format!(" [{}] 3D Scatter — ←→↑↓ rotate ",
                                    view_names.iter().enumerate()
                                        .map(|(i, n)| if i == view_mode { format!("*{}*", n) } else { n.to_string() })
                                        .collect::<Vec<_>>().join(" | "))))
                            .x_bounds([x_min, x_max])
                            .y_bounds([y_min, y_max])
                            .paint(move |ctx| {
                                ctx.draw(&Points { coords: &points_owned, color: Color::Green });
                            });
                        frame.render_widget(canvas, chunks[1]);
                    }
                    1 => {
                        // Eigenvalue spectrum bar chart
                        use ratatui::widgets::{Bar, BarChart, BarGroup};
                        let max_ev = eigenvalues.iter().copied().fold(0.0f64, f64::max);
                        let scale = if max_ev > 0.0 { 1000.0 / max_ev } else { 1.0 };
                        let bars: Vec<Bar> = eigenvalues.iter().enumerate()
                            .filter(|&(_, v)| *v > 0.0)
                            .map(|(i, v)| {
                        let v = *v;
                                Bar::default()
                                    .value((v * scale) as u64)
                                    .label(format!("PC{}", i + 1).into())
                                    .style(Style::default().fg(if i < 3 { Color::Green } else { Color::DarkGray }))
                            })
                            .collect();
                        let chart = BarChart::default()
                            .block(Block::default()
                                .borders(Borders::ALL)
                                .title(" Eigenvalue Spectrum — top components highlighted "))
                            .data(BarGroup::default().bars(&bars))
                            .bar_gap(1)
                            .max((1000.0) as u64);
                        frame.render_widget(chart, chunks[1]);
                    }
                    2 => {
                        // Cumulative variance explained
                        let total_var_sum: f64 = eigenvalues.iter().sum();
                        let mut lines_text: Vec<Line> = Vec::new();
                        lines_text.push(Line::from(Span::styled(
                            " Cumulative Variance Explained",
                            Style::default().fg(Color::Cyan),
                        )));
                        lines_text.push(Line::from(""));
                        let mut cumulative = 0.0f64;
                        for (i, &ev) in eigenvalues.iter().enumerate() {
                            if ev <= 0.0 { break; }
                            cumulative += ev;
                            let pct = if total_var_sum > 0.0 { 100.0 * cumulative / total_var_sum } else { 0.0 };
                            let bar_len = (pct * 0.5) as usize; // 50 chars = 100%
                            let bar: String = "█".repeat(bar_len);
                            lines_text.push(Line::from(vec![
                                Span::styled(format!(" PC{:<3}", i + 1), Style::default().fg(Color::Yellow)),
                                Span::styled(format!(" {:>6.2}% ", pct), Style::default().fg(Color::White)),
                                Span::styled(bar, Style::default().fg(if i < 3 { Color::Green } else { Color::DarkGray })),
                            ]));
                        }
                        frame.render_widget(
                            Paragraph::new(lines_text)
                                .block(Block::default().borders(Borders::ALL).title(" Cumulative Variance ")),
                            chunks[1],
                        );
                    }
                    3 => {
                        // Component loadings — top contributing dimensions for PC1-3
                        let mut lines_text: Vec<Line> = Vec::new();
                        lines_text.push(Line::from(Span::styled(
                            " Top Dimension Loadings per Component",
                            Style::default().fg(Color::Cyan),
                        )));
                        lines_text.push(Line::from(""));
                        for (pc_idx, evec) in eigenvectors_top3.iter().enumerate() {
                            lines_text.push(Line::from(Span::styled(
                                format!(" PC{} (var {:.1}%):", pc_idx + 1, var_pcts[pc_idx]),
                                Style::default().fg(Color::Yellow),
                            )));
                            // Find top-10 dimensions by absolute loading
                            let mut indexed: Vec<(usize, f64)> = evec.iter().enumerate()
                                .map(|(d, &v)| (d, v.abs()))
                                .collect();
                            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                            for &(d, mag) in indexed.iter().take(10) {
                                let sign = if evec[d] >= 0.0 { "+" } else { "-" };
                                let bar_len = (mag * 40.0) as usize; // scale to ~40 chars
                                let bar: String = "█".repeat(bar_len.min(40));
                                lines_text.push(Line::from(vec![
                                    Span::styled(format!("   dim {:>4} ", d), Style::default().fg(Color::DarkGray)),
                                    Span::styled(format!("{}{:.4} ", sign, mag), Style::default().fg(Color::White)),
                                    Span::styled(bar, Style::default().fg(Color::Green)),
                                ]));
                            }
                            lines_text.push(Line::from(""));
                        }
                        frame.render_widget(
                            Paragraph::new(lines_text)
                                .block(Block::default().borders(Borders::ALL).title(" Component Loadings ")),
                            chunks[1],
                        );
                    }
                    _ => {}
                }
            }

            let can_increase = current_sample < total;
            let view_indicator: String = view_names.iter().enumerate()
                .map(|(i, n)| if i == view_mode { format!("[{}]", n) } else { n.to_string() })
                .collect::<Vec<_>>().join(" ");
            let footer = if can_increase {
                format!(" {} | Tab:next | Space:2x | q quit | ? help | {} pts", view_indicator, current_sample)
            } else {
                format!(" {} | Tab:next | q quit | ? help | {} pts (full)", view_indicator, current_sample)
            };
            frame.render_widget(
                Paragraph::new(Span::styled(footer, Style::default().fg(Color::DarkGray))),
                chunks[2],
            );
        }).unwrap();

        if event::poll(std::time::Duration::from_millis(100)).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('?') => { show_help = !show_help; }
                    // View switching
                    KeyCode::Tab | KeyCode::PageDown => { view_mode = (view_mode + 1) % NUM_VIEWS; }
                    KeyCode::BackTab | KeyCode::PageUp => { view_mode = (view_mode + NUM_VIEWS - 1) % NUM_VIEWS; }
                    KeyCode::Char('1') => { view_mode = 0; }
                    KeyCode::Char('2') => { view_mode = 1; }
                    KeyCode::Char('3') => { view_mode = 2; }
                    KeyCode::Char('4') => { view_mode = 3; }
                    // Rotation (scatter view only)
                    KeyCode::Left if view_mode == 0 => { rot_y -= 0.1; }
                    KeyCode::Right if view_mode == 0 => { rot_y += 0.1; }
                    KeyCode::Up if view_mode == 0 => { rot_x -= 0.1; }
                    KeyCode::Down if view_mode == 0 => { rot_x += 0.1; }
                    KeyCode::Char('r') => { rot_y = 0.0; rot_x = 0.0; }
                    KeyCode::Char(' ') if current_sample < total => {
                        current_sample = (current_sample * 2).min(total);
                        needs_compute = true;
                    }
                    KeyCode::Char('+') if current_sample < total => {
                        current_sample = (current_sample * 3 / 2).min(total);
                        needs_compute = true;
                    }
                    KeyCode::Char('a') if current_sample < total => {
                        current_sample = total;
                        needs_compute = true;
                    }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode().unwrap();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
}

// ---------------------------------------------------------------------------
// PCA computation
// ---------------------------------------------------------------------------

/// Sample `effective` indices from `total` using Fisher-Yates shuffle.
fn sample_indices(total: usize, effective: usize, seed: u64) -> Vec<usize> {
    if effective >= total {
        return (0..total).collect();
    }
    let mut rng = crate::pipeline::rng::seeded_rng(seed);
    use rand::Rng;
    let mut idx: Vec<usize> = (0..total).collect();
    for i in 0..effective {
        let j = rng.random_range(i..total);
        idx.swap(i, j);
    }
    idx.truncate(effective);
    idx
}

/// Compute the mean vector across sampled indices.
fn compute_mean(
    reader: &AnyVectorReader,
    indices: &[usize],
    dim: usize,
) -> Vec<f64> {
    let n = indices.len();

    let partial_sums: Vec<Vec<f64>> = indices
        .chunks(4096)
        .map(|chunk| {
            let mut sum = vec![0.0f64; dim];
            for &i in chunk {
                if let Some(v) = reader.get_f64(i) {
                    for d in 0..dim {
                        sum[d] += v[d];
                    }
                }
            }
            sum
        })
        .collect();

    let mut mean = vec![0.0f64; dim];
    for ps in &partial_sums {
        for d in 0..dim {
            mean[d] += ps[d];
        }
    }
    for d in 0..dim {
        mean[d] /= n as f64;
    }
    mean
}

/// Compute top-k eigenvectors of the covariance matrix using power iteration.
///
/// For each eigenvector:
/// 1. Start with a random vector
/// 2. Repeatedly multiply by the covariance matrix (implicitly, via the data)
/// 3. Normalize
/// 4. Deflate the covariance by subtracting the found eigenvector's contribution
///
/// The covariance-vector product is computed implicitly:
///   Cv = (1/n) Σ (x_i - μ)(x_i - μ)ᵀ v = (1/n) Σ (x_i - μ) · ((x_i - μ)ᵀ v)
/// This avoids materializing the d×d covariance matrix.
fn compute_top_eigenvectors(
    reader: &AnyVectorReader,
    indices: &[usize],
    mean: &[f64],
    dim: usize,
    k: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    use simsimd::SpatialSimilarity;

    let n = indices.len();
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut eigenvalues: Vec<f64> = Vec::with_capacity(k);

    // Pre-compute f32 mean for simsimd dot products
    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();

    for _ki in 0..k {
        // Initialize with a deterministic pseudo-random vector
        let mut v: Vec<f64> = (0..dim).map(|d| ((d * 7 + 13) % 97) as f64 - 48.0).collect();
        normalize(&mut v);

        // Power iteration: 30 iterations is more than enough for convergence
        for _iter in 0..30 {
            // Implicit covariance-vector product: Cv = (1/n) Σ (x-μ)((x-μ)·v)
            // Use simsimd for the (x-μ)·v dot product in the inner loop.
            let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            let partial_products: Vec<Vec<f64>> = indices
                .chunks(4096)
                .map(|chunk| {
                    let mut product = vec![0.0f64; dim];
                    let mut centered = vec![0.0f32; dim];
                    for &i in chunk {
                        if let Some(x) = reader.get_f32(i) {
                            // Center: centered = x - μ
                            for d in 0..dim {
                                centered[d] = x[d] - mean_f32[d];
                            }
                            // SIMD dot product: (x-μ)·v
                            let dot = <f32 as SpatialSimilarity>::dot(&centered, &v_f32)
                                .unwrap_or(0.0) as f64;
                            // Accumulate outer product contribution in f64
                            for d in 0..dim {
                                product[d] += centered[d] as f64 * dot;
                            }
                        }
                    }
                    product
                })
                .collect();

            // Merge partial products
            let mut new_v = vec![0.0f64; dim];
            for pp in &partial_products {
                for d in 0..dim {
                    new_v[d] += pp[d];
                }
            }
            for d in 0..dim {
                new_v[d] /= n as f64;
            }

            // Deflate: remove components along previously found eigenvectors
            for prev in &eigenvectors {
                let proj: f64 = new_v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for d in 0..dim {
                    new_v[d] -= proj * prev[d];
                }
            }

            normalize(&mut new_v);
            v = new_v;
        }

        // Compute eigenvalue: λ = vᵀ C v (via the same implicit product)
        let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let eigenvalue_sum: f64 = indices
            .chunks(4096)
            .map(|chunk| {
                let mut sum = 0.0f64;
                let mut centered = vec![0.0f32; dim];
                for &i in chunk {
                    if let Some(x) = reader.get_f32(i) {
                        for d in 0..dim {
                            centered[d] = x[d] - mean_f32[d];
                        }
                        let dot = <f32 as SpatialSimilarity>::dot(&centered, &v_f32)
                            .unwrap_or(0.0) as f64;
                        sum += dot * dot;
                    }
                }
                sum
            })
            .sum();

        let eigenvalue = eigenvalue_sum / n as f64;
        eigenvalues.push(eigenvalue);
        eigenvectors.push(v);
    }

    (eigenvectors, eigenvalues)
}

/// Project sampled vectors onto the top-k eigenvectors.
///
/// Uses simsimd for the dot products: (x - μ) · eigenvector.
fn project_vectors(
    reader: &AnyVectorReader,
    indices: &[usize],
    mean: &[f64],
    eigenvectors: &[Vec<f64>],
    dim: usize,
) -> Vec<(f64, f64)> {
    use simsimd::SpatialSimilarity;

    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
    let ev0_f32: Vec<f32> = eigenvectors[0].iter().map(|&x| x as f32).collect();
    let ev1_f32: Vec<f32> = eigenvectors[1].iter().map(|&x| x as f32).collect();

    let mut centered = vec![0.0f32; dim];
    indices
        .iter()
        .filter_map(|&i| {
            let x = reader.get_f32(i)?;
            for d in 0..dim {
                centered[d] = x[d] - mean_f32[d];
            }
            let pc1 = <f32 as SpatialSimilarity>::dot(&centered, &ev0_f32)
                .unwrap_or(0.0) as f64;
            let pc2 = <f32 as SpatialSimilarity>::dot(&centered, &ev1_f32)
                .unwrap_or(0.0) as f64;
            Some((pc1, pc2))
        })
        .collect()
}

/// Project sampled vectors onto the top-3 eigenvectors (3D).
fn project_vectors_3d(
    reader: &AnyVectorReader,
    indices: &[usize],
    mean: &[f64],
    eigenvectors: &[Vec<f64>],
    dim: usize,
) -> Vec<[f64; 3]> {
    use simsimd::SpatialSimilarity;

    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
    let ev0_f32: Vec<f32> = eigenvectors[0].iter().map(|&x| x as f32).collect();
    let ev1_f32: Vec<f32> = eigenvectors[1].iter().map(|&x| x as f32).collect();
    let ev2_f32: Vec<f32> = if eigenvectors.len() > 2 {
        eigenvectors[2].iter().map(|&x| x as f32).collect()
    } else {
        vec![0.0f32; dim]
    };

    let mut centered = vec![0.0f32; dim];
    indices
        .iter()
        .filter_map(|&i| {
            let x = reader.get_f32(i)?;
            for d in 0..dim {
                centered[d] = x[d] - mean_f32[d];
            }
            let pc1 = <f32 as SpatialSimilarity>::dot(&centered, &ev0_f32).unwrap_or(0.0) as f64;
            let pc2 = <f32 as SpatialSimilarity>::dot(&centered, &ev1_f32).unwrap_or(0.0) as f64;
            let pc3 = <f32 as SpatialSimilarity>::dot(&centered, &ev2_f32).unwrap_or(0.0) as f64;
            Some([pc1, pc2, pc3])
        })
        .collect()
}

/// Rotate 3D points around Y then X axes and project onto 2D (drop Z).
fn rotate_and_project(points_3d: &[[f64; 3]], rot_y: f64, rot_x: f64) -> Vec<(f64, f64)> {
    let (sy, cy) = rot_y.sin_cos();
    let (sx, cx) = rot_x.sin_cos();

    points_3d.iter().map(|[x, y, z]| {
        // Rotate around Y axis
        let x1 = x * cy + z * sy;
        let y1 = *y;
        let z1 = -x * sy + z * cy;
        // Rotate around X axis
        let x2 = x1;
        let y2 = y1 * cx - z1 * sx;
        // Project: drop Z (orthographic projection)
        (x2, y2)
    }).collect()
}

/// Magic bytes for 3D PCA cache format (distinguishes from old 2D format).
const PCA_3D_MAGIC: &[u8; 4] = b"PC3D";

/// Save 3D PCA projection to cache.
fn save_pca_3d_cache(
    path: &std::path::Path,
    projected: &[[f64; 3]],
    eigenvalues: &[f64],
    dim: usize,
    total: usize,
    filename: &str,
) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(PCA_3D_MAGIC).map_err(|e| e.to_string())?;
    let fname_bytes = filename.as_bytes();
    f.write_all(&(dim as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(total as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(fname_bytes.len() as u16).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(fname_bytes).map_err(|e| e.to_string())?;
    // Write 3 eigenvalues
    for ev in eigenvalues.iter().take(3) {
        f.write_all(&ev.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    for _ in eigenvalues.len()..3 {
        f.write_all(&0.0f64.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    f.write_all(&(projected.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    for p in projected {
        for &v in p {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Load 3D PCA projection from cache.
fn load_pca_3d_cache(path: &std::path::Path) -> Option<(Vec<[f64; 3]>, Vec<f64>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];
    let mut buf2 = [0u8; 2];

    // Check magic header
    f.read_exact(&mut buf4).ok()?;
    if &buf4 != PCA_3D_MAGIC {
        return None; // Old format or corrupt — trigger recomputation
    }

    f.read_exact(&mut buf4).ok()?;
    // dim (unused for loading)
    f.read_exact(&mut buf8).ok()?;
    // total (unused for loading)
    f.read_exact(&mut buf2).ok()?;
    let fname_len = u16::from_le_bytes(buf2) as usize;
    let mut fname_buf = vec![0u8; fname_len];
    f.read_exact(&mut fname_buf).ok()?;

    let mut eigenvalues = Vec::with_capacity(3);
    for _ in 0..3 {
        f.read_exact(&mut buf8).ok()?;
        eigenvalues.push(f64::from_le_bytes(buf8));
    }

    f.read_exact(&mut buf8).ok()?;
    let n_points = u64::from_le_bytes(buf8) as usize;
    if n_points > 100_000_000 { return None; }

    let mut projected = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        let mut p = [0.0f64; 3];
        for v in &mut p {
            f.read_exact(&mut buf8).ok()?;
            *v = f64::from_le_bytes(buf8);
        }
        projected.push(p);
    }

    Some((projected, eigenvalues))
}

/// Normalize a vector to unit length.
fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute axis bounds with 5% padding.
fn compute_bounds(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let x_pad = (x_max - x_min) * 0.05;
    let y_pad = (y_max - y_min) * 0.05;
    (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
}

// ---------------------------------------------------------------------------
// PCA cache
// ---------------------------------------------------------------------------

/// Install a SIGINT (Ctrl-C) handler that sets an abort flag.
fn install_abort_handler(flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    unsafe {
        let flag_ptr = std::sync::Arc::into_raw(flag) as *mut std::sync::atomic::AtomicBool;
        // Store the pointer in a static so the signal handler can access it
        ABORT_FLAG.store(flag_ptr as usize, std::sync::atomic::Ordering::SeqCst);
        libc::signal(libc::SIGINT, sigint_handler as *const () as libc::sighandler_t);
    }
}

static ABORT_FLAG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

extern "C" fn sigint_handler(_sig: libc::c_int) {
    let ptr = ABORT_FLAG.load(std::sync::atomic::Ordering::SeqCst);
    if ptr != 0 {
        let flag = unsafe { &*(ptr as *const std::sync::atomic::AtomicBool) };
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Per-partition statistics for incremental PCA.
struct PartitionStats {
    count: usize,
    mean: Vec<f64>,
}

/// Save partition statistics to a binary cache file.
fn save_partition_cache(path: &std::path::Path, stats: &PartitionStats) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(&(stats.count as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(stats.mean.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    for &v in &stats.mean {
        f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Load partition statistics from a binary cache file.
fn load_partition_cache(path: &std::path::Path) -> Option<PartitionStats> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];

    f.read_exact(&mut buf8).ok()?;
    let count = u64::from_le_bytes(buf8) as usize;

    f.read_exact(&mut buf4).ok()?;
    let dim = u32::from_le_bytes(buf4) as usize;

    let mut mean = vec![0.0f64; dim];
    for i in 0..dim {
        f.read_exact(&mut buf8).ok()?;
        mean[i] = f64::from_le_bytes(buf8);
    }

    Some(PartitionStats { count, mean })
}

/// Check if cache file is newer than the source file.
fn is_cache_fresh(cache: &std::path::Path, source: &std::path::Path) -> bool {
    let cache_mtime = std::fs::metadata(cache).ok().and_then(|m| m.modified().ok());
    let source_mtime = std::fs::metadata(source).ok().and_then(|m| m.modified().ok());
    match (cache_mtime, source_mtime) {
        (Some(c), Some(s)) => c > s,
        _ => false,
    }
}

/// Save PCA projection results to a binary cache file.
///
/// Format: [dim: u32][total: u64][filename_len: u16][filename: bytes]
///         [eigenvalue_0: f64][eigenvalue_1: f64]
///         [n_points: u64][pc1_0: f64][pc2_0: f64][pc1_1: f64][pc2_1: f64]...
fn save_pca_cache(
    path: &std::path::Path,
    projected: &[(f64, f64)],
    eigenvalues: &[f64],
    dim: usize,
    total: usize,
    filename: &str,
) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)
        .map_err(|e| format!("failed to create cache: {}", e))?;
    let fname_bytes = filename.as_bytes();
    f.write_all(&(dim as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(total as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(fname_bytes.len() as u16).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(fname_bytes).map_err(|e| e.to_string())?;
    for ev in eigenvalues.iter().take(2) {
        f.write_all(&ev.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    f.write_all(&(projected.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    for (x, y) in projected {
        f.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&y.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Load cached PCA projection.
/// Returns (projected_points, eigenvalues, dim, total, filename).
fn load_pca_cache(
    path: &std::path::Path,
) -> Option<(Vec<(f64, f64)>, Vec<f64>, usize, usize, String)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    let mut buf2 = [0u8; 2];

    f.read_exact(&mut buf4).ok()?;
    let dim = u32::from_le_bytes(buf4) as usize;

    f.read_exact(&mut buf8).ok()?;
    let total = u64::from_le_bytes(buf8) as usize;

    f.read_exact(&mut buf2).ok()?;
    let fname_len = u16::from_le_bytes(buf2) as usize;
    let mut fname_buf = vec![0u8; fname_len];
    f.read_exact(&mut fname_buf).ok()?;
    let filename = String::from_utf8(fname_buf).ok()?;

    let mut eigenvalues = Vec::with_capacity(2);
    for _ in 0..2 {
        f.read_exact(&mut buf8).ok()?;
        eigenvalues.push(f64::from_le_bytes(buf8));
    }

    f.read_exact(&mut buf8).ok()?;
    let n_points = u64::from_le_bytes(buf8) as usize;

    let mut projected = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        f.read_exact(&mut buf8).ok()?;
        let x = f64::from_le_bytes(buf8);
        f.read_exact(&mut buf8).ok()?;
        let y = f64::from_le_bytes(buf8);
        projected.push((x, y));
    }

    Some((projected, eigenvalues, dim, total, filename))
}
