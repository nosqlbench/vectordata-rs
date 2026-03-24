// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Data shell — interactive and batch REPL for vector file exploration.

use std::path::PathBuf;

use super::shared::{UnifiedReader, is_local_source};

/// Available data-shell commands for autocomplete.
const SHELL_COMMANDS: &[&str] = &[
    "info", "get", "range", "head", "tail", "dist", "distance",
    "norm", "norms", "stats", "help", "quit", "exit",
];

/// Run data-shell in non-interactive batch mode.
///
/// Interprets positional args as semicolon-separated commands,
/// executes them against the vector file, prints results to stdout, exits.
pub(super) fn run_data_shell_batch(source: &str, commands: &str) {
    use crate::pipeline::commands::analyze_explore;

    let reader = UnifiedReader::open(source);
    let count = reader.count();
    let dim = reader.dim();

    let get_f64 = |i: usize| -> Vec<f64> {
        reader.get_f64(i).unwrap_or_default()
    };

    for cmd in commands.split(';') {
        let cmd = cmd.trim();
        if cmd.is_empty() || cmd == "quit" || cmd == "exit" { break; }
        let result = analyze_explore::execute_repl_command(cmd, &get_f64, count, dim);
        println!("{}", result);
    }
}

/// Run data-shell as an interactive ratatui TUI with input line and output scrollback.
pub(super) fn run_data_shell_interactive(source: &str) {
    use crossterm::{
        event::{self, Event, KeyCode, KeyModifiers},
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        execute,
    };
    use ratatui::{
        backend::CrosstermBackend,
        layout::{Constraint, Direction, Layout},
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::{Block, Borders, Paragraph, Wrap},
        Terminal,
    };
    use crate::pipeline::commands::analyze_explore;

    let reader = UnifiedReader::open(source);
    let count = reader.count();
    let dim = reader.dim();

    let filename = if is_local_source(source) {
        PathBuf::from(source).file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    } else {
        source.to_string()
    };

    let get_f64 = |i: usize| -> Vec<f64> {
        reader.get_f64(i).unwrap_or_default()
    };

    // State
    let mut input = String::new();
    let mut cursor_pos: usize = 0;
    let mut output_lines: Vec<(String, Color)> = Vec::new();
    let mut history: Vec<String> = Vec::new();
    let mut history_idx: Option<usize> = None;
    let mut scroll_offset: u16 = 0;
    let mut completion_candidates: Vec<String> = Vec::new();

    // Welcome message
    output_lines.push((format!("data-shell: {} ({} vectors, {} dims)", filename, count, dim), Color::Cyan));
    output_lines.push(("Type 'help' for commands, Tab for completion, Ctrl-D or 'quit' to exit.".into(), Color::DarkGray));
    output_lines.push((String::new(), Color::White));

    if let Err(e) = enable_raw_mode() {
        eprintln!("Error: cannot enter TUI mode: {}", e);
        eprintln!("Ensure stdout is connected to a terminal.");
        std::process::exit(1);
    }
    let mut stdout = std::io::stdout();
    if let Err(e) = execute!(stdout, EnterAlternateScreen) {
        let _ = disable_raw_mode();
        eprintln!("Error: cannot enter alternate screen: {}", e);
        std::process::exit(1);
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(e) => {
            let _ = disable_raw_mode();
            eprintln!("Error: cannot initialize terminal: {}", e);
            std::process::exit(1);
        }
    };

    loop {
        // Build completion candidates based on current input
        if !input.is_empty() {
            completion_candidates = SHELL_COMMANDS.iter()
                .filter(|c| c.starts_with(input.trim()))
                .map(|c| c.to_string())
                .collect();
        } else {
            completion_candidates.clear();
        }

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(5),      // output scrollback
                    Constraint::Length(1),    // completion hint
                    Constraint::Length(3),    // input box
                ])
                .split(frame.area());

            // Output scrollback
            let visible_height = chunks[0].height.saturating_sub(2) as usize;
            let total_lines = output_lines.len();
            let start = if total_lines > visible_height {
                total_lines - visible_height - scroll_offset as usize
            } else {
                0
            };
            let display_lines: Vec<Line> = output_lines[start..]
                .iter()
                .take(visible_height)
                .map(|(text, color)| Line::from(Span::styled(text.clone(), Style::default().fg(*color))))
                .collect();

            frame.render_widget(
                Paragraph::new(display_lines)
                    .block(Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" {} — {} vectors × {} dims ", filename, count, dim)))
                    .wrap(Wrap { trim: false }),
                chunks[0],
            );

            // Completion hint
            let hint = if completion_candidates.len() == 1 {
                format!(" Tab: {}", completion_candidates[0])
            } else if completion_candidates.len() > 1 {
                format!(" Matches: {}", completion_candidates.join(", "))
            } else {
                String::new()
            };
            frame.render_widget(
                Paragraph::new(Span::styled(hint, Style::default().fg(Color::DarkGray))),
                chunks[1],
            );

            // Input box with cursor
            let input_widget = Paragraph::new(Line::from(vec![
                Span::styled("explore> ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw(&input),
            ]))
            .block(Block::default().borders(Borders::ALL));
            frame.render_widget(input_widget, chunks[2]);

            // Position cursor
            let cursor_x = chunks[2].x + 1 + 9 + cursor_pos as u16; // border + "explore> " + pos
            let cursor_y = chunks[2].y + 1;
            frame.set_cursor_position((cursor_x.min(chunks[2].right() - 2), cursor_y));
        }).unwrap();

        // Handle input
        if event::poll(std::time::Duration::from_millis(50)).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                match key.code {
                    KeyCode::Enter => {
                        let cmd = input.trim().to_string();
                        if !cmd.is_empty() {
                            output_lines.push((format!("explore> {}", cmd), Color::Green));

                            if cmd == "quit" || cmd == "exit" {
                                break;
                            }

                            let result = analyze_explore::execute_repl_command(
                                &cmd, &get_f64, count, dim,
                            );
                            for line in result.lines() {
                                output_lines.push((line.to_string(), Color::White));
                            }
                            output_lines.push((String::new(), Color::White));

                            history.push(cmd);
                            history_idx = None;
                        }
                        input.clear();
                        cursor_pos = 0;
                        scroll_offset = 0;
                    }
                    KeyCode::Tab => {
                        // Autocomplete
                        if completion_candidates.len() == 1 {
                            input = format!("{} ", completion_candidates[0]);
                            cursor_pos = input.len();
                        }
                    }
                    KeyCode::Backspace => {
                        if cursor_pos > 0 {
                            input.remove(cursor_pos - 1);
                            cursor_pos -= 1;
                        }
                    }
                    KeyCode::Delete => {
                        if cursor_pos < input.len() {
                            input.remove(cursor_pos);
                        }
                    }
                    KeyCode::Left => {
                        if cursor_pos > 0 { cursor_pos -= 1; }
                    }
                    KeyCode::Right => {
                        if cursor_pos < input.len() { cursor_pos += 1; }
                    }
                    KeyCode::Home => { cursor_pos = 0; }
                    KeyCode::End => { cursor_pos = input.len(); }
                    KeyCode::Up => {
                        // History navigation
                        if !history.is_empty() {
                            let idx = match history_idx {
                                Some(0) => 0,
                                Some(i) => i - 1,
                                None => history.len() - 1,
                            };
                            input = history[idx].clone();
                            cursor_pos = input.len();
                            history_idx = Some(idx);
                        }
                    }
                    KeyCode::Down => {
                        if let Some(idx) = history_idx {
                            if idx + 1 < history.len() {
                                let new_idx = idx + 1;
                                input = history[new_idx].clone();
                                cursor_pos = input.len();
                                history_idx = Some(new_idx);
                            } else {
                                input.clear();
                                cursor_pos = 0;
                                history_idx = None;
                            }
                        }
                    }
                    KeyCode::PageUp => {
                        scroll_offset = (scroll_offset + 10).min(output_lines.len() as u16);
                    }
                    KeyCode::PageDown => {
                        scroll_offset = scroll_offset.saturating_sub(10);
                    }
                    KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        break; // Ctrl-D exits
                    }
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Ctrl-C clears current input
                        input.clear();
                        cursor_pos = 0;
                    }
                    KeyCode::Esc => {
                        break;
                    }
                    KeyCode::Char(c) => {
                        input.insert(cursor_pos, c);
                        cursor_pos += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode().unwrap();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
}
