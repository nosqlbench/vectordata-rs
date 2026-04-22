// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks explore values` — scrollable raw-values grid.
//!
//! Opens any vector source (local file or `dataset:profile:facet`) and
//! presents the values as a 2-D grid of ordinals × dimensions, with
//! arrow-key scrolling, configurable significant digits, and a 24-bit
//! diverging-color heatmap. Heatmap normalization scope is switchable
//! between per-vector (each row scaled independently), per-column
//! (each dim scaled across visible rows), global (all visible cells),
//! and off.
//!
//! Cells render in fixed columns sized to the widest formatted value
//! among the currently visible rows + a small slack, so the grid stays
//! aligned regardless of how the data ranges across visible ordinals.

use std::io;

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};

use super::palette::{Curve, Palette};
use super::shared::UnifiedReader;

/// How the user exits the values grid.
pub(super) enum Exit {
    /// User pressed `q` — exit the program.
    Quit,
    /// User pressed `Esc` — return to the dataset picker (or program if
    /// the source was specified on the CLI).
    Back,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NormScope {
    /// Each visible row is normalized to its own [min, max].
    PerVector,
    /// Each visible column is normalized across all visible rows.
    PerColumn,
    /// All visible cells are normalized together.
    Global,
    /// No coloring.
    Off,
}

impl NormScope {
    fn label(self) -> &'static str {
        match self {
            NormScope::PerVector => "per-vector",
            NormScope::PerColumn => "per-column",
            NormScope::Global   => "global",
            NormScope::Off      => "off",
        }
    }
    fn next(self) -> Self {
        match self {
            NormScope::PerVector => NormScope::PerColumn,
            NormScope::PerColumn => NormScope::Global,
            NormScope::Global    => NormScope::Off,
            NormScope::Off       => NormScope::PerVector,
        }
    }
}


struct State {
    source: String,
    count: usize,
    dim: usize,
    /// Topmost ordinal in the visible window.
    row_top: usize,
    /// Leftmost dim index in the visible window.
    col_left: usize,
    digits: u8,
    norm: NormScope,
    palette: Palette,
    curve: Curve,
    /// When `true`, each visible row is divided by its L2 norm before
    /// formatting. The underlying cache is unchanged — the
    /// transformation is applied per-frame, so toggling it back off
    /// returns to the raw values without a re-read. The L2 column
    /// continues to show the *original* norm (since after normalization
    /// every displayed row would read 1.000 there, which would be
    /// useless).
    normalized: bool,
    /// Optional jump-to-ordinal input buffer; `Some` when active.
    jump_buf: Option<String>,
    /// One-line transient status (errors, hints).
    status: Option<String>,
    /// Cached values for the currently visible row range.
    cache: Vec<Option<Vec<f64>>>,
    /// Ordinal of `cache[0]`.
    cache_top: usize,
}

impl State {
    fn new(source: &str, reader: &UnifiedReader, start: usize, digits: u8) -> Self {
        Self {
            source: source.to_string(),
            count: reader.count(),
            dim: reader.dim(),
            row_top: start.min(reader.count().saturating_sub(1)),
            col_left: 0,
            digits: digits.clamp(1, 6),
            norm: NormScope::PerVector,
            palette: Palette::BlueOrange,
            curve: Curve::Linear,
            normalized: false,
            jump_buf: None,
            status: None,
            cache: Vec::new(),
            cache_top: 0,
        }
    }

    fn ensure_cache(&mut self, reader: &UnifiedReader, rows: usize) {
        // Refill the row cache if the visible window has shifted. The
        // batch read path lets remote-backed sources satisfy the whole
        // visible window in one HTTP round instead of one per row.
        let want = rows.min(self.count.saturating_sub(self.row_top));
        if self.cache_top == self.row_top && self.cache.len() == want {
            return;
        }
        self.cache_top = self.row_top;
        self.cache = reader.get_f64_range(self.row_top, want);
    }
}

pub(super) fn run(source: &str, start: usize, digits: u8) -> Exit {
    let reader = UnifiedReader::open(source);
    let mut state = State::new(source, &reader, start, digits);

    let mut stdout = io::stdout();
    if execute!(stdout, EnterAlternateScreen).is_err()
        || enable_raw_mode().is_err()
    {
        eprintln!("failed to initialize terminal");
        return Exit::Quit;
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(e) => {
            let _ = disable_raw_mode();
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
            eprintln!("terminal init failed: {e}");
            return Exit::Quit;
        }
    };

    let exit = loop {
        let mut visible_rows = 0usize;
        let mut visible_cols = 0usize;
        let _ = terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2), // header
                    Constraint::Min(3),    // grid
                    Constraint::Length(2), // footer
                ])
                .split(area);
            render_header(frame, chunks[0], &state);
            let (rows, cols) = render_grid(frame, chunks[1], &mut state, &reader);
            visible_rows = rows;
            visible_cols = cols;
            render_footer(frame, chunks[2], &state);
        });

        // Wait for input (blocking — no animation, no background work).
        let ev = match event::read() {
            Ok(e) => e,
            Err(_) => continue,
        };
        if let Event::Key(k) = ev {
            // Jump-to-ordinal input mode swallows most keys.
            if state.jump_buf.is_some() {
                if handle_jump_key(&mut state, k.code) { continue; }
                continue;
            }
            // Snapshot the row window so we know whether to invalidate
            // the cache below — palette/curve/sig-digit cycling don't
            // need a re-read, so spending an HTTP round for them is
            // pure overhead.
            let prev_row_top = state.row_top;
            match (k.code, k.modifiers) {
                (KeyCode::Char('q'), _) => break Exit::Quit,
                (KeyCode::Esc, _)       => break Exit::Back,

                // Single-step scrolling: arrow keys + lowercase hjkl
                // (vim convention). h/l move the dim window left/right
                // by one; j/k move the row window down/up by one.
                // crossterm reports Shift+letter as the uppercase Char,
                // so 'j' vs 'J' are already distinct without a guard.
                (KeyCode::Down,  _) | (KeyCode::Char('j'), _) =>
                    state.row_top = (state.row_top + 1).min(state.count.saturating_sub(1)),
                (KeyCode::Up,    _) | (KeyCode::Char('k'), _) =>
                    state.row_top = state.row_top.saturating_sub(1),
                (KeyCode::Right, _) | (KeyCode::Char('l'), _) =>
                    state.col_left = (state.col_left + 1).min(state.dim.saturating_sub(1)),
                (KeyCode::Left,  _) | (KeyCode::Char('h'), _) =>
                    state.col_left = state.col_left.saturating_sub(1),

                // Page scrolling: PageUp/PageDown + uppercase HJKL.
                (KeyCode::PageDown, _) | (KeyCode::Char('J'), _) =>
                    state.row_top = (state.row_top + visible_rows.max(1)).min(state.count.saturating_sub(1)),
                (KeyCode::PageUp,   _) | (KeyCode::Char('K'), _) =>
                    state.row_top = state.row_top.saturating_sub(visible_rows.max(1)),
                (KeyCode::Char('L'), _) =>
                    state.col_left = (state.col_left + visible_cols.max(1)).min(state.dim.saturating_sub(1)),
                (KeyCode::Char('H'), _) =>
                    state.col_left = state.col_left.saturating_sub(visible_cols.max(1)),

                // Top / bottom of dataset. `[` and `]` are present on
                // every keyboard (including laptops without dedicated
                // Home/End and most international layouts), so prefer
                // them over Home/End. Home/End are still accepted as an
                // alias when available.
                (KeyCode::Char('['), _) | (KeyCode::Home, KeyModifiers::CONTROL) => state.row_top = 0,
                (KeyCode::Char(']'), _) | (KeyCode::End,  KeyModifiers::CONTROL) => state.row_top = state.count.saturating_sub(visible_rows.max(1)),

                // Vim-style first/last for dim navigation: `0` jumps to
                // dim 0, `$` jumps to the last dim. Keeps Home/End as
                // an alias for users who have them.
                (KeyCode::Char('0'), _) | (KeyCode::Home, _) => state.col_left = 0,
                (KeyCode::Char('$'), _) | (KeyCode::End, _)  => state.col_left = state.dim.saturating_sub(visible_cols.max(1)),

                (KeyCode::Char('+') | KeyCode::Char('='), _) => state.digits = (state.digits + 1).min(6),
                (KeyCode::Char('-') | KeyCode::Char('_'), _) => state.digits = state.digits.saturating_sub(1).max(1),
                (KeyCode::Char('n'), _) => state.norm = state.norm.next(),
                (KeyCode::Char('p'), _) => state.palette = state.palette.next(),
                (KeyCode::Char('c'), _) => state.curve = state.curve.next(),
                // Uppercase N toggles "show as L2-normalized". Keeps the
                // vim hjkl/HJKL keys free for navigation.
                (KeyCode::Char('N'), _) => state.normalized = !state.normalized,
                (KeyCode::Char('g'), _) => state.jump_buf = Some(String::new()),
                _ => {}
            }
            // Suppress stale per-keystroke status; only jump emits its own.
            state.status = None;
            // Only invalidate the row cache when the row window actually
            // moved. ensure_cache() also checks (cache_top, cache.len())
            // but clearing here is what ensures it reads fresh data when
            // the window is the same size but starts elsewhere.
            if state.row_top != prev_row_top {
                state.cache.clear();
            }
        }
    };

    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
    exit
}

fn handle_jump_key(state: &mut State, code: KeyCode) -> bool {
    match code {
        KeyCode::Esc => { state.jump_buf = None; true }
        KeyCode::Enter => {
            if let Some(buf) = state.jump_buf.take() {
                match buf.trim().parse::<usize>() {
                    Ok(n) if n < state.count => {
                        state.row_top = n;
                        state.cache.clear();
                    }
                    Ok(n) => state.status = Some(format!("ordinal {n} ≥ count {}", state.count)),
                    Err(_) => state.status = Some(format!("not a number: {buf:?}")),
                }
            }
            true
        }
        KeyCode::Backspace => {
            if let Some(buf) = state.jump_buf.as_mut() { buf.pop(); }
            true
        }
        KeyCode::Char(c) if c.is_ascii_digit() => {
            if let Some(buf) = state.jump_buf.as_mut() { buf.push(c); }
            true
        }
        _ => true,
    }
}

fn render_header(frame: &mut ratatui::Frame, area: Rect, state: &State) {
    let title = Line::from(vec![
        Span::styled("values ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(&state.source),
    ]);
    let sub = Line::from(vec![
        Span::styled("count=", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("{}", state.count)),
        Span::raw("  "),
        Span::styled("dim=", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("{}", state.dim)),
        Span::raw("  "),
        Span::styled("row=", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("{}", state.row_top)),
        Span::raw("  "),
        Span::styled("dim_col=", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("{}", state.col_left)),
        Span::raw("  "),
        Span::styled("sig=", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("{}", state.digits)),
        Span::raw("  "),
        Span::styled("scope=", Style::default().fg(Color::DarkGray)),
        Span::raw(state.norm.label()),
        Span::raw("  "),
        Span::styled("palette=", Style::default().fg(Color::DarkGray)),
        Span::raw(state.palette.label()),
        Span::raw("  "),
        Span::styled("curve=", Style::default().fg(Color::DarkGray)),
        Span::raw(state.curve.label()),
        Span::raw("  "),
        Span::styled("view=", Style::default().fg(Color::DarkGray)),
        Span::raw(if state.normalized { "L2-normalized" } else { "raw" }),
    ]);
    let p = Paragraph::new(vec![title, sub]);
    frame.render_widget(p, area);
}

fn render_footer(frame: &mut ratatui::Frame, area: Rect, state: &State) {
    let key = Style::default().fg(Color::Yellow);
    let help = Line::from(vec![
        Span::styled("hjkl/↑↓←→", key), Span::raw(" scroll  "),
        Span::styled("HJKL/PgUp/PgDn", key), Span::raw(" page  "),
        Span::styled("[ ]", key), Span::raw(" first/last row  "),
        Span::styled("0 $", key), Span::raw(" first/last dim  "),
        Span::styled("+/-", key), Span::raw(" sig  "),
        Span::styled("n", key), Span::raw(" scope  "),
        Span::styled("p", key), Span::raw(" palette  "),
        Span::styled("c", key), Span::raw(" curve  "),
        Span::styled("N", key), Span::raw(" L2-norm view  "),
        Span::styled("g", key), Span::raw(" jump  "),
        Span::styled("Esc", key), Span::raw(" back  "),
        Span::styled("q", key), Span::raw(" quit"),
    ]);
    let status = if let Some(buf) = state.jump_buf.as_ref() {
        Line::from(vec![
            Span::styled("jump to ordinal: ", Style::default().fg(Color::Cyan)),
            Span::raw(buf.clone()),
            Span::styled("_", Style::default().fg(Color::DarkGray)),
            Span::styled("   (Enter to go, Esc to cancel)", Style::default().fg(Color::DarkGray)),
        ])
    } else if let Some(s) = state.status.as_ref() {
        Line::from(Span::styled(s.clone(), Style::default().fg(Color::Red)))
    } else {
        Line::from("")
    };
    let p = Paragraph::new(vec![help, status]);
    frame.render_widget(p, area);
}

/// Returns `(visible_rows, visible_cols)` so the input loop can size
/// page jumps to the actual viewport.
fn render_grid(
    frame: &mut ratatui::Frame,
    area: Rect,
    state: &mut State,
    reader: &UnifiedReader,
) -> (usize, usize) {
    let block = Block::default().borders(Borders::ALL)
        .title(Span::styled(" data ", Style::default().fg(Color::DarkGray)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width < 12 || inner.height < 2 {
        return (0, 0);
    }

    // Reserve top row for dim labels.
    let body_height = (inner.height as usize).saturating_sub(1);
    state.ensure_cache(reader, body_height);
    let visible_rows = state.cache.len();

    // Row label column: width is the digit-width of the largest ordinal
    // we might display, plus a small separator.
    let max_ord = state.row_top.saturating_add(visible_rows).saturating_sub(1).max(state.count.saturating_sub(1));
    let row_label_w = format!("{}", max_ord).len() + 2;

    // L2 norm column: precompute every visible row's norm so we can
    // size the column to the widest formatted value. Norms are useful
    // for spotting unnormalized vectors at a glance and for reading
    // each row's "size" without doing arithmetic on the cells.
    let norms: Vec<Option<f64>> = state.cache.iter()
        .map(|opt| opt.as_ref().map(|v| {
            let s: f64 = v.iter().map(|x| x * x).sum();
            s.sqrt()
        }))
        .collect();
    let norm_cells: Vec<FormattedCell> = norms.iter()
        .map(|opt| match opt {
            Some(n) => FormattedCell::from_value(*n, state.digits as usize),
            None    => FormattedCell::placeholder(),
        })
        .collect();
    let mut norm_left_w = "L2".len();
    let mut norm_right_w = 0usize;
    for fc in &norm_cells {
        norm_left_w = norm_left_w.max(fc.left.chars().count());
        norm_right_w = norm_right_w.max(fc.right.chars().count());
    }
    let norm_total_w = (norm_left_w + norm_right_w).clamp(4, 16);
    // Re-derive the half widths inside the clamp envelope so render()
    // stays consistent if the cap kicked in.
    let norm_left_w = norm_left_w.min(norm_total_w);
    let norm_right_w = norm_total_w - norm_left_w;
    // Width from ordinal end to first data column = " " + norm + " ".
    let norm_block_w = 1 + norm_total_w + 1;

    // Provisional column width, used only to decide *how many* columns
    // fit. Real per-column widths are computed below from the actual
    // visible values so decimals line up. We want roughly the worst-case
    // width: signed integer digit + dot + sig digits + optional exponent
    // ("e-NN" = 4 chars). One trailing space separates columns.
    let provisional_cell_w = state.digits as usize + 8;
    let avail_w = (inner.width as usize).saturating_sub(row_label_w + norm_block_w);
    let visible_cols = (avail_w / (provisional_cell_w + 1))
        .max(1)
        .min(state.dim.saturating_sub(state.col_left));

    // Per-row scale factor for the "L2-normalized view" toggle. We
    // compute it from the same row norms used in the L2 column, so the
    // user can read the original magnitude there even while the cells
    // themselves render the unit-norm variant. Rows that are identically
    // zero stay at the original (zero) values rather than dividing by
    // zero — there's no meaningful normalized form for the zero vector.
    let scale: Vec<f64> = norms.iter()
        .map(|opt| match opt {
            Some(n) if state.normalized && *n > 0.0 => 1.0 / *n,
            _ => 1.0,
        })
        .collect();

    // Format every visible cell as (left-of-dot, right-of-dot) parts so
    // we can pad them independently — that's what makes decimal points
    // align across rows regardless of sign or magnitude. Non-finite
    // values render as a single token shown right-aligned in the cell.
    let formatted: Vec<Vec<FormattedCell>> = state.cache.iter().enumerate()
        .map(|(r, opt)| {
            let s = scale[r];
            (0..visible_cols).map(|c| {
                let dim = state.col_left + c;
                opt.as_ref()
                    .and_then(|v| v.get(dim).copied())
                    .map(|x| FormattedCell::from_value(x * s, state.digits as usize))
                    .unwrap_or(FormattedCell::placeholder())
            }).collect()
        })
        .collect();

    // Compute a single uniform `(left_w, right_w)` across every visible
    // column. Per-column widths would shift each column's offset every
    // time the user scrolls left/right (one column moves out, another
    // moves in, widths recompute), making the dim labels appear to jump
    // around — which the user explicitly called out as disorienting.
    // Uniform widths cost a few characters of extra space but keep
    // every label pinned to the same screen offset across scrolls.
    let mut left_w = 0usize;
    let mut right_w = 0usize;
    for row in &formatted {
        for cell in row.iter() {
            left_w = left_w.max(cell.left.chars().count());
            right_w = right_w.max(cell.right.chars().count());
        }
    }
    // Floor the widths at the worst-case integer-part for the current
    // sig setting (sign + 1 int digit + dot, fraction = sig digits)
    // so an all-zero column doesn't shrink relative to neighbours.
    left_w = left_w.max(2);                       // " 0" or "-0"
    right_w = right_w.max(state.digits as usize + 1); // ".dddd"
    let col_w = (left_w + right_w).clamp(4, 24);
    if left_w + right_w > col_w {
        right_w = col_w - left_w;
    }

    // Walls every N columns. Drawn as a "│" between cells whose dim is
    // a multiple of `wall_every`, plus a synchronized header tick. The
    // wall sits in the trailing-space slot, so it adds no extra width.
    let wall_every: usize = 8;

    // Render the dim-label row using the same per-column widths so the
    // labels sit above their values.
    let mut hdr_spans: Vec<Span> = Vec::with_capacity(visible_cols + 1);
    hdr_spans.push(Span::styled(format!("{:>w$}", "ord", w = row_label_w - 1), Style::default().fg(Color::DarkGray)));
    hdr_spans.push(Span::raw(" "));
    hdr_spans.push(Span::styled(format!("{:>w$}", "L2", w = norm_total_w), Style::default().fg(Color::DarkGray)));
    hdr_spans.push(Span::raw(" "));
    for c in 0..visible_cols {
        let dim = state.col_left + c;
        let label = format!("d{}", dim);
        let padded = format!("{:>w$}", label, w = col_w);
        hdr_spans.push(Span::styled(padded, Style::default().fg(Color::DarkGray)));
        // Header gets a tick wherever the body row gets a wall, so the
        // wall is read as a connected vertical line.
        let next_dim = dim + 1;
        let is_wall = c + 1 < visible_cols && next_dim % wall_every == 0;
        if is_wall {
            hdr_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
        } else {
            hdr_spans.push(Span::raw(" "));
        }
    }
    let hdr_line = Line::from(hdr_spans);
    frame.render_widget(Paragraph::new(hdr_line), Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 });

    // Compute scope-wide normalization bounds once. In L2-normalized
    // view the bounds must be computed on the post-scale values so the
    // heatmap matches what the cells actually show.
    let global_bounds = scaled_global_bounds(state, &scale, visible_cols);

    for r in 0..visible_rows {
        let y = inner.y + 1 + r as u16;
        if y >= inner.y + inner.height { break; }

        let ord = state.row_top + r;
        let mut row_spans: Vec<Span> = Vec::with_capacity(2 * visible_cols + 4);
        row_spans.push(Span::styled(
            format!("{:>w$}", ord, w = row_label_w - 1),
            Style::default().fg(Color::DarkGray),
        ));
        row_spans.push(Span::raw(" "));
        // L2 norm column. Slightly emphasized so it reads as metadata
        // distinct from the data columns.
        row_spans.push(Span::styled(
            norm_cells[r].render(norm_left_w, norm_right_w),
            Style::default().fg(Color::Cyan),
        ));
        row_spans.push(Span::raw(" "));

        let row = state.cache.get(r).and_then(|o| o.as_ref());
        let s = scale[r];
        let row_bounds = if state.norm == NormScope::PerVector {
            row.map(|v| slice_bounds(v.iter().map(|x| x * s)))
        } else {
            None
        };

        for c in 0..visible_cols {
            let dim = state.col_left + c;
            let raw_cell = row.and_then(|v| v.get(dim).copied());
            let scaled_cell = raw_cell.map(|x| x * s);
            let fc = &formatted[r][c];
            let text = fc.render(left_w, right_w);
            let style = match scaled_cell {
                Some(v) => {
                    let (lo, hi) = match state.norm {
                        NormScope::Off       => (0.0, 0.0),
                        NormScope::PerVector => row_bounds.unwrap_or((0.0, 0.0)),
                        NormScope::PerColumn => scaled_col_bounds(state, &scale, dim),
                        NormScope::Global    => global_bounds,
                    };
                    cell_style(v, lo, hi, state.norm, state.palette, state.curve)
                }
                None => Style::default().fg(Color::DarkGray),
            };
            row_spans.push(Span::styled(text, style));
            let next_dim = dim + 1;
            let is_wall = c + 1 < visible_cols && next_dim % wall_every == 0;
            if is_wall {
                row_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
            } else {
                row_spans.push(Span::raw(" "));
            }
        }
        let line = Line::from(row_spans);
        let row_rect = Rect { x: inner.x, y, width: inner.width, height: 1 };
        frame.render_widget(Paragraph::new(line), row_rect);
    }

    (visible_rows, visible_cols)
}

/// A value formatted as two halves so the decimal point can be aligned
/// per-column across rows. `left` includes the sign and integer part
/// (e.g. " 0", "-0", " 1"); `right` starts with the dot (e.g. ".0294")
/// or the exponent's "e..." for scientific values that have no integer
/// digit visible at this magnitude, or is empty for non-finite tokens.
struct FormattedCell {
    left: String,
    right: String,
    /// If set, this is a non-finite or placeholder token to render
    /// right-aligned within the column instead of split-aligned.
    one_shot: Option<String>,
}

impl FormattedCell {
    fn placeholder() -> Self {
        Self { left: String::new(), right: String::new(), one_shot: Some("—".to_string()) }
    }

    fn from_value(v: f64, sig: usize) -> Self {
        if !v.is_finite() {
            let token = if v.is_nan() { "NaN" } else if v < 0.0 { "-inf" } else { "+inf" };
            return Self { left: String::new(), right: String::new(), one_shot: Some(token.to_string()) };
        }
        let raw = format_value(v, sig);
        // Reserve a leading space for the sign on positives so signed
        // and unsigned values share the same int-side width.
        let signed = if raw.starts_with('-') { raw } else { format!(" {}", raw) };
        // Split at the first '.', keeping the dot with the right half.
        let (left, right) = match signed.find('.') {
            Some(i) => (signed[..i].to_string(), signed[i..].to_string()),
            None    => (signed, String::new()),
        };
        Self { left, right, one_shot: None }
    }

    fn render(&self, left_w: usize, right_w: usize) -> String {
        if let Some(tok) = &self.one_shot {
            return format!("{:>w$}", tok, w = left_w + right_w);
        }
        format!("{:>lw$}{:<rw$}", self.left, self.right, lw = left_w, rw = right_w)
    }
}

fn slice_bounds<I: Iterator<Item = f64>>(it: I) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in it {
        if !v.is_finite() { continue; }
        if v < lo { lo = v; }
        if v > hi { hi = v; }
    }
    if !lo.is_finite() || !hi.is_finite() { (0.0, 0.0) } else { (lo, hi) }
}

fn scaled_col_bounds(state: &State, scale: &[f64], dim: usize) -> (f64, f64) {
    slice_bounds(
        state.cache.iter().enumerate()
            .filter_map(|(r, o)| o.as_ref().map(|v| (r, v)))
            .filter_map(|(r, v)| v.get(dim).copied().map(|x| x * scale[r]))
    )
}

fn scaled_global_bounds(state: &State, scale: &[f64], visible_cols: usize) -> (f64, f64) {
    if state.norm != NormScope::Global { return (0.0, 0.0); }
    slice_bounds(
        state.cache.iter().enumerate()
            .filter_map(|(r, o)| o.as_ref().map(|v| (r, v)))
            .flat_map(|(r, v)| {
                let end = (state.col_left + visible_cols).min(v.len());
                let s = scale[r];
                v[state.col_left..end].iter().map(move |x| x * s)
            })
    )
}

fn format_value(v: f64, sig: usize) -> String {
    if !v.is_finite() {
        return if v.is_nan() { "NaN".to_string() } else if v < 0.0 { "-inf".to_string() } else { "+inf".to_string() };
    }
    if v == 0.0 { return format!("0.{}", "0".repeat(sig.saturating_sub(1))); }

    let abs = v.abs();
    // Use scientific notation when the magnitude is too small or too
    // large to display cleanly in fixed-point at the requested sig digits.
    if abs < 10f64.powi(-(sig as i32)) || abs >= 10f64.powi(sig as i32 + 2) {
        return format!("{:.*e}", sig.saturating_sub(1), v);
    }

    // sig-digit fixed-point: number of decimals = sig - integer digits.
    // Use the *rounded* magnitude to count integer digits, otherwise
    // values just under a power of ten (e.g. 0.99996) count as 0 int
    // digits → 4 decimals → "1.0000" while 1.0000 itself counts as 1
    // int digit → 3 decimals → "1.000". The post-rounding count keeps
    // sig digits honest across that boundary so neighbouring rows in
    // the L2 column don't render at different widths.
    let int_digits = {
        let initial = abs.log10().floor() as i32 + 1;
        let dec0 = (sig as i32 - initial).max(0) as usize;
        let scale = 10f64.powi(dec0 as i32);
        let rounded = (abs * scale).round() / scale;
        if rounded > 0.0 { rounded.log10().floor() as i32 + 1 } else { initial }
    };
    let dec = (sig as i32 - int_digits).max(0) as usize;
    format!("{:.*}", dec, v)
}

/// Map a value to a 24-bit cell style. The palette controls the hue
/// ramp (cold/neutral/hot), the curve reshapes magnitude before lookup,
/// and `Off`/degenerate-bounds yields a flat gray. Sequential palettes
/// like Cividis collapse the negative side onto the positive; divergent
/// palettes use the cold anchor for negatives and hot for positives.
fn cell_style(v: f64, lo: f64, hi: f64, scope: NormScope, palette: Palette, curve: Curve) -> Style {
    if scope == NormScope::Off || !(hi > lo) {
        return Style::default().fg(Color::Gray);
    }
    let mid = 0.5 * (lo + hi);
    let half = (hi - mid).max(mid - lo).max(f64::EPSILON);
    // t in [-1, 1].
    let t = ((v - mid) / half).clamp(-1.0, 1.0);
    let mag = curve.apply(t.abs());

    let (cold, neutral, hot) = palette.anchors();
    let target = if palette.is_sequential() || t >= 0.0 { hot } else { cold };
    let r = lerp_u8(neutral.0, target.0, mag);
    let g = lerp_u8(neutral.1, target.1, mag);
    let b = lerp_u8(neutral.2, target.2, mag);

    let mut style = Style::default().fg(Color::Rgb(r, g, b));
    if mag > 0.85 {
        style = style.add_modifier(Modifier::BOLD);
    }
    style
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    let af = a as f64;
    let bf = b as f64;
    (af + (bf - af) * t).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_value_ranges() {
        assert_eq!(format_value(1.2345, 4), "1.234");
        assert_eq!(format_value(-0.7891, 4), "-0.7891");
        assert_eq!(format_value(3.14159, 3), "3.14");
        assert_eq!(format_value(0.0, 4), "0.000");
        assert!(format_value(1.23e-9, 4).contains("e"));
        assert!(format_value(1.23e12, 3).contains("e"));
        assert_eq!(format_value(f64::NAN, 4), "NaN");
        assert_eq!(format_value(f64::INFINITY, 4), "+inf");
        assert_eq!(format_value(f64::NEG_INFINITY, 4), "-inf");
    }

    #[test]
    fn format_value_round_up_keeps_sig_digits() {
        // The L2-norm column on normalized vectors produces values
        // just below 1.0 due to floating-point. Without the
        // post-rounding `int_digits` pass these would render as
        // "1.0000" while exact 1.0 renders as "1.000", giving the
        // column ragged width.
        assert_eq!(format_value(0.99996, 4), "1.000");
        assert_eq!(format_value(1.0,     4), "1.000");
        // Same edge case at other powers of ten.
        assert_eq!(format_value(9.9996, 3), "10.0");
        assert_eq!(format_value(10.0,   3), "10.0");
        // Doesn't disturb genuinely sub-1 values.
        assert_eq!(format_value(0.5, 4), "0.5000");
        assert_eq!(format_value(0.0789, 4), "0.07890");
    }

    #[test]
    fn formatted_cell_decimal_alignment() {
        // Per-column decimal-point alignment: positive values get a
        // leading space so '-' on negatives doesn't shift the dot.
        let pos = FormattedCell::from_value(0.02937, 4);
        let neg = FormattedCell::from_value(-0.01779, 4);
        let sci = FormattedCell::from_value(7.834e-5, 4);
        let lw = pos.left.chars().count()
            .max(neg.left.chars().count())
            .max(sci.left.chars().count());
        let rw = pos.right.chars().count()
            .max(neg.right.chars().count())
            .max(sci.right.chars().count());
        let p = pos.render(lw, rw);
        let n = neg.render(lw, rw);
        let s = sci.render(lw, rw);
        // The dot must be at the same column index in every rendered cell.
        let dot = p.find('.').unwrap();
        assert_eq!(n.find('.').unwrap(), dot, "neg: {n:?} vs pos: {p:?}");
        assert_eq!(s.find('.').unwrap(), dot, "sci: {s:?} vs pos: {p:?}");
        // All renders are the same width.
        assert_eq!(p.chars().count(), n.chars().count());
        assert_eq!(p.chars().count(), s.chars().count());
    }

    #[test]
    fn formatted_cell_nonfinite_renders_in_field() {
        let nan = FormattedCell::from_value(f64::NAN, 4);
        let s = nan.render(3, 5); // total width 8
        assert_eq!(s.chars().count(), 8);
        assert!(s.trim() == "NaN");
    }

    #[test]
    fn lerp_endpoints() {
        assert_eq!(lerp_u8(0, 255, 0.0), 0);
        assert_eq!(lerp_u8(0, 255, 1.0), 255);
        assert_eq!(lerp_u8(100, 200, 0.5), 150);
    }

    #[test]
    fn bounds_skip_nonfinite() {
        let xs = [1.0f64, f64::NAN, -2.0, f64::INFINITY, 3.5];
        let (lo, hi) = slice_bounds(xs.into_iter());
        assert_eq!(lo, -2.0);
        assert_eq!(hi, 3.5);
    }

    #[test]
    fn cell_style_off_is_flat() {
        let s = cell_style(1.5, 0.0, 1.0, NormScope::Off, Palette::BlueOrange, Curve::Linear);
        assert_eq!(s.fg, Some(Color::Gray));
    }

    #[test]
    fn curve_endpoints_are_exact() {
        for c in [Curve::Linear, Curve::Sqrt, Curve::Square, Curve::Sigmoid] {
            assert!((c.apply(0.0) - 0.0).abs() < 1e-9, "{:?}.apply(0)", c);
            assert!((c.apply(1.0) - 1.0).abs() < 1e-9, "{:?}.apply(1)", c);
        }
        // Shape sanity: sqrt boosts low end above linear, square suppresses it.
        assert!(Curve::Sqrt.apply(0.25) > Curve::Linear.apply(0.25));
        assert!(Curve::Square.apply(0.25) < Curve::Linear.apply(0.25));
    }

    #[test]
    fn palette_anchors_distinguishable() {
        // Color-blind-safe palettes must produce visibly different
        // anchors at the cold and hot extremes (otherwise the diverging
        // signal collapses).
        for p in [Palette::BlueOrange, Palette::BlueYellow, Palette::BlueRed, Palette::Mono] {
            let (cold, _, hot) = p.anchors();
            let dr = (cold.0 as i32 - hot.0 as i32).abs();
            let dg = (cold.1 as i32 - hot.1 as i32).abs();
            let db = (cold.2 as i32 - hot.2 as i32).abs();
            assert!(dr + dg + db > 100, "{:?} anchors too close: cold={cold:?} hot={hot:?}", p);
        }
    }

    #[test]
    fn palette_cycle_visits_all() {
        let mut seen = std::collections::HashSet::new();
        let mut p = Palette::BlueOrange;
        // 7 palettes — cycle for one extra to verify it wraps.
        for _ in 0..8 {
            seen.insert(format!("{:?}", p));
            p = p.next();
        }
        assert_eq!(seen.len(), 7);
    }

    #[test]
    fn norm_scope_cycles() {
        let order = [
            NormScope::PerVector,
            NormScope::PerColumn,
            NormScope::Global,
            NormScope::Off,
            NormScope::PerVector,
        ];
        for w in order.windows(2) {
            assert_eq!(w[0].next(), w[1]);
        }
    }
}
