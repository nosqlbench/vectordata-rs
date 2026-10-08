// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Terminal capability detection.
//!
//! Detects ANSI color support once at startup and exposes it globally.
//! All output code should use these functions rather than checking
//! `atty::is()` directly.

use std::sync::OnceLock;

/// Cached terminal capabilities, detected once at startup.
struct TermCaps {
    /// stdout is a TTY and supports ANSI escape codes.
    color: bool,
    /// stdin is a TTY (interactive input available).
    interactive: bool,
}

static CAPS: OnceLock<TermCaps> = OnceLock::new();

fn caps() -> &'static TermCaps {
    CAPS.get_or_init(|| {
        // Respect NO_COLOR (https://no-color.org/)
        let no_color = std::env::var_os("NO_COLOR").is_some();
        // Respect FORCE_COLOR for CI that supports it
        let force_color = std::env::var_os("FORCE_COLOR").is_some();

        let is_tty = atty::is(atty::Stream::Stdout);
        let color = if force_color {
            true
        } else if no_color {
            false
        } else {
            is_tty
        };

        TermCaps {
            color,
            interactive: atty::is(atty::Stream::Stdin),
        }
    })
}

/// Whether stdout supports ANSI color codes.
pub fn use_color() -> bool {
    caps().color
}

/// Whether stdin is interactive (TTY).
pub fn is_interactive() -> bool {
    caps().interactive
}

// ── ANSI escape helpers ──────────────────────────────────────────────────

/// ANSI SGR reset: clears all colour and style attributes.
pub const RESET: &str = "\x1b[0m";
/// ANSI bold (increased intensity).
pub const BOLD: &str = "\x1b[1m";
/// ANSI dim (decreased intensity).
pub const DIM: &str = "\x1b[2m";

/// ANSI red foreground.
pub const RED: &str = "\x1b[31m";
/// ANSI green foreground.
pub const GREEN: &str = "\x1b[32m";
/// ANSI yellow foreground.
pub const YELLOW: &str = "\x1b[33m";
/// ANSI blue foreground.
pub const BLUE: &str = "\x1b[34m";
/// ANSI cyan foreground.
pub const CYAN: &str = "\x1b[36m";

/// ANSI bold red foreground.
pub const BOLD_RED: &str = "\x1b[1;31m";
/// ANSI bold green foreground.
pub const BOLD_GREEN: &str = "\x1b[1;32m";
/// ANSI bold yellow foreground.
pub const BOLD_YELLOW: &str = "\x1b[1;33m";
/// ANSI bold cyan foreground.
pub const BOLD_CYAN: &str = "\x1b[1;36m";

/// Wrap text in an ANSI color code, or return it unchanged if color is off.
pub fn color(code: &str, text: &str) -> String {
    if use_color() {
        format!("{}{}{}", code, text, RESET)
    } else {
        text.to_string()
    }
}

/// Wrap text in bold, or return unchanged if color is off.
pub fn bold(text: &str) -> String {
    color(BOLD, text)
}

/// Green text (for success/OK).
pub fn green(text: &str) -> String {
    color(GREEN, text)
}

/// Bold green (for prominent success).
pub fn ok(text: &str) -> String {
    color(BOLD_GREEN, text)
}

/// Red text (for errors/failures).
pub fn red(text: &str) -> String {
    color(RED, text)
}

/// Bold red (for prominent errors).
pub fn fail(text: &str) -> String {
    color(BOLD_RED, text)
}

/// Yellow text (for warnings).
pub fn warn(text: &str) -> String {
    color(YELLOW, text)
}

/// Cyan text (for informational labels).
pub fn info(text: &str) -> String {
    color(CYAN, text)
}

/// Blue text (for private/authenticated content).
pub fn blue(text: &str) -> String {
    color(BLUE, text)
}

/// Dim text (for secondary/deemphasized content).
pub fn dim(text: &str) -> String {
    color(DIM, text)
}
