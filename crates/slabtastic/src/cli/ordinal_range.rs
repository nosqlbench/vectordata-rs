// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Ordinal range parser for CLI commands.
//!
//! Parses ordinal range specifiers into half-open `[start, end)` intervals.
//!
//! ## Valid forms
//!
//! - `n` — shorthand for `[0, n)` (first n ordinals)
//! - `m..n` — closed interval, equivalent to `[m, n+1)`
//! - `[m,n)` or `[m..n)` — half-open (m inclusive, n exclusive)
//! - `[m,n]` or `[m..n]` — closed (m inclusive, n inclusive)
//! - `(m,n)` or `(m..n)` — open (m exclusive, n exclusive)
//! - `(m,n]` or `(m..n]` — half-open (m exclusive, n inclusive)
//! - `[n]` — single ordinal, equivalent to `[n, n+1)`

/// Parse an ordinal range specifier into a half-open `[start, end)` interval.
///
/// Returns `(start, end)` where `start` is inclusive and `end` is exclusive.
pub fn parse_ordinal_range(s: &str) -> Result<(i64, i64), String> {
    let s = s.trim();

    // Single number: n → [0, n)
    if let Ok(n) = s.parse::<i64>()
        && !s.starts_with('[') && !s.starts_with('(') {
            return Ok((0, n));
        }

    // [n] — single ordinal
    if s.starts_with('[') && s.ends_with(']') && !s.contains(',') && !s.contains("..") {
        let inner = &s[1..s.len() - 1];
        let n: i64 = inner
            .parse()
            .map_err(|_| format!("invalid ordinal in [{inner}]"))?;
        return Ok((n, n + 1));
    }

    // Bracket forms: [m,n), [m,n], (m,n), (m,n], and .. variants
    let left_inclusive = s.starts_with('[');
    let right_inclusive = s.ends_with(']');

    if (s.starts_with('[') || s.starts_with('('))
        && (s.ends_with(')') || s.ends_with(']'))
    {
        let inner = &s[1..s.len() - 1];
        let (left, right) = if inner.contains(',') {
            let parts: Vec<&str> = inner.splitn(2, ',').collect();
            (parts[0].trim(), parts[1].trim())
        } else if inner.contains("..") {
            let parts: Vec<&str> = inner.splitn(2, "..").collect();
            (parts[0].trim(), parts[1].trim())
        } else {
            return Err(format!("invalid range syntax: {s}"));
        };

        let m: i64 = left
            .parse()
            .map_err(|_| format!("invalid left bound: {left}"))?;
        let n: i64 = right
            .parse()
            .map_err(|_| format!("invalid right bound: {right}"))?;

        let start = if left_inclusive { m } else { m + 1 };
        let end = if right_inclusive { n + 1 } else { n };
        return Ok((start, end));
    }

    // m..n — closed interval
    if s.contains("..") {
        let parts: Vec<&str> = s.splitn(2, "..").collect();
        let m: i64 = parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("invalid left bound: {}", parts[0]))?;
        let n: i64 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("invalid right bound: {}", parts[1]))?;
        return Ok((m, n + 1));
    }

    Err(format!("unrecognized ordinal range: {s}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` → `[0, n)` (first n ordinals).
    #[test]
    fn test_single_count() {
        assert_eq!(parse_ordinal_range("10").unwrap(), (0, 10));
    }

    /// `m..n` → `[m, n+1)` (closed interval).
    #[test]
    fn test_dotdot_range() {
        assert_eq!(parse_ordinal_range("5..10").unwrap(), (5, 11));
    }

    /// `[m,n)` → half-open interval.
    #[test]
    fn test_half_open_comma() {
        assert_eq!(parse_ordinal_range("[5,10)").unwrap(), (5, 10));
    }

    /// `[m,n]` → closed interval.
    #[test]
    fn test_closed_comma() {
        assert_eq!(parse_ordinal_range("[5,10]").unwrap(), (5, 11));
    }

    /// `(m,n)` → open interval.
    #[test]
    fn test_open_comma() {
        assert_eq!(parse_ordinal_range("(5,10)").unwrap(), (6, 10));
    }

    /// `(m,n]` → half-open interval (m exclusive, n inclusive).
    #[test]
    fn test_half_open_right_inclusive() {
        assert_eq!(parse_ordinal_range("(5,10]").unwrap(), (6, 11));
    }

    /// `[n]` → single ordinal.
    #[test]
    fn test_single_ordinal() {
        assert_eq!(parse_ordinal_range("[42]").unwrap(), (42, 43));
    }

    /// `[m..n)` → half-open with dotdot separator.
    #[test]
    fn test_half_open_dotdot() {
        assert_eq!(parse_ordinal_range("[5..10)").unwrap(), (5, 10));
    }

    /// Negative ordinals work.
    #[test]
    fn test_negative_ordinals() {
        assert_eq!(parse_ordinal_range("[-5,5)").unwrap(), (-5, 5));
    }
}
