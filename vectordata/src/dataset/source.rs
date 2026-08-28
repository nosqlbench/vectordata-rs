// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Data source specifications: [`DSSource`], [`DSInterval`], [`DSWindow`].
//!
//! A [`DSSource`] identifies a file path plus optional namespace and window
//! (record range). Sources can be specified as bare strings with inline
//! sugar notation or as explicit maps in YAML. The sugar parser
//! ([`parse_source_string`]) handles bracket/paren windows, colon-delimited
//! namespaces, and SI-suffixed numbers.
//!
//! ## Sugar forms
//!
//! | Form | Example |
//! |------|---------|
//! | Bare string | `"file.fvec"` |
//! | String + bracket window | `"file.fvec[0..1M]"` |
//! | String + paren window | `"file.fvec(0..1000)"` |
//! | String + mixed delimiters | `"file.fvec[0..1M)"` |
//! | String + namespace | `"file.slab:content"` |
//! | String + namespace + window | `"file.slab:ns:[0..1K]"` |
//! | Map with `source` | `{ source: "f.fvec" }` |
//! | Map with `path` | `{ path: "f.fvec" }` |
//! | Map with window | `{ source: "f.fvec", window: "0..1000" }` |

use std::fmt;

use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};

/// Data source with optional namespace and window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct DSSource {
    /// Path to the source file.
    pub path: String,
    /// Optional namespace within the source (e.g., slab page name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Window (record range) restriction.
    #[serde(skip_serializing_if = "DSWindow::is_empty")]
    pub window: DSWindow,
    /// The **edifying count** from a trailing `=<count>` suffix.
    ///
    /// Redundant by construction, and that is the point: it documents a
    /// cardinality where a reader would otherwise do arithmetic on
    /// seven-digit bounds, and it catches a typo in either bound. What
    /// it is checked against depends on whether a window is present —
    /// the window's length if so, the file's own cardinality if not.
    ///
    /// See `docs/design/srd-multifile-facet-shards.md`, SH-62.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared_count: Option<u64>,
}

/// A window is a list of intervals (empty = ALL data).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct DSWindow(pub Vec<DSInterval>);

/// Half-open interval `[min_incl, max_excl)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DSInterval {
    /// Inclusive lower bound.
    pub min_incl: u64,
    /// Exclusive upper bound.
    pub max_excl: u64,
}

impl DSWindow {
    /// Returns `true` if no intervals are specified (meaning all data).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for DSInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.min_incl, self.max_excl)
    }
}

impl fmt::Display for DSWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            write!(f, "ALL")
        } else {
            let parts: Vec<String> = self.0.iter().map(|i| i.to_string()).collect();
            write!(f, "[{}]", parts.join(", "))
        }
    }
}

impl fmt::Display for DSSource {
    /// Render back to the source-string grammar, faithfully enough to
    /// parse to an equal `DSSource`.
    ///
    /// That round trip is relied upon: it is how a view serializes back
    /// to YAML, and how the catalog loader hands a source to the shard
    /// model without a second parser.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)?;
        if let Some(ref ns) = self.namespace {
            write!(f, ":{}", ns)?;
        }
        if !self.window.is_empty() {
            // `DSWindow` brackets itself; bracketing again produced
            // `a.u8[[0..100]]`, which only parsed because the window
            // parser strips an extra layer.
            write!(f, "{}", self.window)?;
        }
        if let Some(n) = self.declared_count {
            write!(f, "={n}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Interval parsing
// ---------------------------------------------------------------------------

/// Format a count value to a compact profile name using SI/IEC suffixes.
///
/// Never uses decimal points. Uses the shortest integer-only representation,
/// trying both SI decimal (`t`, `g`, `m`, `k`) and IEC binary (`ti`, `gi`,
/// `mi`, `ki`) scales. For values that don't divide evenly at one scale,
/// a compound notation is used (e.g., `1g24m` = 1 billion + 24 million).
///
/// This function generates profile directory names on disk.
///
/// Examples:
/// - `1000000` → `1m`
/// - `1024000000` → `1g24m`
/// - `524288000000` → `524g288m`
/// - `1073741824` → `1gi` (exact binary)
/// - `2000000000` → `2g`
pub fn format_count_with_suffix(n: u64) -> String {
    if n == 0 {
        return "0".to_string();
    }

    let all_scales: &[(u64, &str)] = &[
        (1_000_000_000_000, "t"),
        (1u64 << 40, "ti"),
        (1_000_000_000, "g"),
        (1u64 << 30, "gi"),
        (1_000_000, "m"),
        (1u64 << 20, "mi"),
        (1_000, "k"),
        (1u64 << 10, "ki"),
    ];

    let remainder_scales: &[(u64, &str)] = &[
        (1_000_000_000, "g"),
        (1u64 << 30, "gi"),
        (1_000_000, "m"),
        (1u64 << 20, "mi"),
        (1_000, "k"),
        (1u64 << 10, "ki"),
    ];

    let mut best: Option<String> = None;

    for &(divisor, suffix) in all_scales {
        let quotient = n / divisor;
        if quotient == 0 || quotient > 999 {
            continue;
        }
        let remainder = n % divisor;

        let candidate = if remainder == 0 {
            format!("{}{}", quotient, suffix)
        } else {
            let mut rem_str = String::new();
            for &(rd, rs) in remainder_scales {
                if rd >= divisor {
                    continue;
                }
                if remainder >= rd && remainder.is_multiple_of(rd) {
                    rem_str = format!("{}{}", remainder / rd, rs);
                    break;
                }
            }
            if rem_str.is_empty() {
                continue;
            }
            format!("{}{}{}", quotient, suffix, rem_str)
        };

        if best.as_ref().is_none_or(|b| candidate.len() < b.len()) {
            best = Some(candidate);
        }
    }

    best.unwrap_or_else(|| n.to_string())
}

/// Parse a unit suffix and return the multiplier.
///
/// Supports simple suffixes matching Java's `UnitConversions.longCountFor`:
/// - `K`/`k` → ×1,000
/// - `M`/`m` → ×1,000,000
/// - `B`/`b`/`G`/`g` → ×1,000,000,000
/// - `T`/`t` → ×1,000,000,000,000
///
/// Also supports ISO decimal (SI) and binary (IEC) suffixes:
/// - `KB` → ×1,000, `KiB` → ×1,024
/// - `MB` → ×1,000,000, `MiB` → ×1,048,576
/// - `GB` → ×1,000,000,000, `GiB` → ×1,073,741,824
/// - `TB` → ×1,000,000,000,000, `TiB` → ×1,099,511,627,776
pub fn parse_number_with_suffix(s: &str) -> Result<u64, String> {
    parse_number_with_suffix_unit(s).map(|(value, _unit)| value)
}

/// Like [`parse_number_with_suffix`], additionally returning the
/// multiplier of the unit the user actually spelled: `1` for bare
/// digits, `1_000_000` for `118MB`/`118m`, the smallest term's unit
/// for compound forms (`1g24m` → `1_000_000`). Callers use it as the
/// value's implied granularity — e.g. the approximate matching of
/// `--with-data`, where a candidate displayed as `118MB` must match
/// the ~118.4 MB dataset it was derived from.
pub fn parse_number_with_suffix_unit(s: &str) -> Result<(u64, u64), String> {
    let s = s.replace('_', "");
    if s.is_empty() {
        return Err("empty number".to_string());
    }

    // Try compound format first: e.g., "1g24m" = 1 billion + 24 million.
    // Split at positions where a digit is followed by an alpha suffix which
    // is then followed by another digit (the start of the remainder term).
    if let Some(total_and_unit) = try_parse_compound(&s) {
        return Ok(total_and_unit);
    }

    // Try multi-char ISO suffixes first (case-insensitive for the letter,
    // but 'i' must be lowercase in binary suffixes: KiB, MiB, GiB, TiB).
    let (num_part, multiplier) = if let Some(n) = s.strip_suffix("TiB") {
        (n, 1u64 << 40)
    } else if let Some(n) = s.strip_suffix("GiB") {
        (n, 1u64 << 30)
    } else if let Some(n) = s.strip_suffix("MiB") {
        (n, 1u64 << 20)
    } else if let Some(n) = s.strip_suffix("KiB") {
        (n, 1u64 << 10)
    } else if let Some(n) = s.strip_suffix("TB").or_else(|| s.strip_suffix("tb")) {
        (n, 1_000_000_000_000u64)
    } else if let Some(n) = s.strip_suffix("GB").or_else(|| s.strip_suffix("gb")) {
        (n, 1_000_000_000u64)
    } else if let Some(n) = s.strip_suffix("MB").or_else(|| s.strip_suffix("mb")) {
        (n, 1_000_000u64)
    } else if let Some(n) = s.strip_suffix("KB").or_else(|| s.strip_suffix("kb")) {
        (n, 1_000u64)
    // Two-char IEC binary suffixes without "B" (used in profile names)
    } else if let Some(n) = s.strip_suffix("ti").or_else(|| s.strip_suffix("Ti")) {
        (n, 1u64 << 40)
    } else if let Some(n) = s.strip_suffix("gi").or_else(|| s.strip_suffix("Gi")) {
        (n, 1u64 << 30)
    } else if let Some(n) = s.strip_suffix("mi").or_else(|| s.strip_suffix("Mi")) {
        (n, 1u64 << 20)
    } else if let Some(n) = s.strip_suffix("ki").or_else(|| s.strip_suffix("Ki")) {
        (n, 1u64 << 10)
    } else {
        // Single-char suffixes
        match s.as_bytes().last() {
            Some(b'K' | b'k') => (&s[..s.len() - 1], 1_000u64),
            Some(b'M' | b'm') => (&s[..s.len() - 1], 1_000_000u64),
            Some(b'B' | b'b' | b'G' | b'g') => (&s[..s.len() - 1], 1_000_000_000u64),
            Some(b'T' | b't') => (&s[..s.len() - 1], 1_000_000_000_000u64),
            _ => (s.as_str(), 1u64),
        }
    };

    let n: u64 = num_part
        .parse()
        .map_err(|e| format!("invalid number '{}': {}", num_part, e))?;
    Ok((n * multiplier, multiplier))
}

/// Try to parse a compound suffix string like `1g24m` or `524g288m`.
///
/// Splits into terms where each term is `{digits}{suffix}`. Returns the
/// sum of all terms plus the smallest term's multiplier (the value's
/// granularity), or None if the string doesn't match compound format.
fn try_parse_compound(s: &str) -> Option<(u64, u64)> {
    // Must have at least two terms (digit-suffix-digit pattern)
    let bytes = s.as_bytes();
    if bytes.len() < 4 {
        return None;
    }

    // Check: does the string have digits, then alpha, then digits again?
    let has_compound = bytes
        .windows(2)
        .any(|w| w[0].is_ascii_alphabetic() && w[1].is_ascii_digit());
    if !has_compound {
        return None;
    }

    // Split into terms: scan for transitions from alpha to digit
    let mut terms: Vec<&str> = Vec::new();
    let mut term_start = 0;
    for i in 1..bytes.len() {
        if bytes[i].is_ascii_digit() && bytes[i - 1].is_ascii_alphabetic() {
            terms.push(&s[term_start..i]);
            term_start = i;
        }
    }
    terms.push(&s[term_start..]);

    if terms.len() < 2 {
        return None;
    }

    // Parse each term as a simple number+suffix
    let mut total = 0u64;
    let mut smallest_unit = u64::MAX;
    for term in &terms {
        // Each term must end with a suffix
        let term_bytes = term.as_bytes();
        if term_bytes.is_empty() || term_bytes.last()?.is_ascii_digit() {
            return None;
        }
        // Find where digits end and suffix begins
        let suffix_start = term_bytes.iter().position(|b| b.is_ascii_alphabetic())?;
        let num_part = &term[..suffix_start];
        let suffix = &term[suffix_start..];
        let n: u64 = num_part.parse().ok()?;
        let multiplier = match suffix.to_lowercase().as_str() {
            "ti" => 1u64 << 40,
            "gi" => 1u64 << 30,
            "mi" => 1u64 << 20,
            "ki" => 1u64 << 10,
            "t" => 1_000_000_000_000,
            "g" | "b" => 1_000_000_000,
            "m" => 1_000_000,
            "k" => 1_000,
            _ => return None,
        };
        total += n * multiplier;
        smallest_unit = smallest_unit.min(multiplier);
    }

    Some((total, smallest_unit))
}

/// Parse a single interval from a string.
///
/// Accepted forms:
/// - `"1M"` → `[0, 1_000_000)`
/// - `"0..1000"` → `[0, 1000)`
/// - `"[0..1000)"` → `[0, 1000)`
/// - `"0..1K"` → `[0, 1000)`
///
/// Symbolic open-ended ranges:
/// - `"[10k..]"` → `[10_000, MAX)` — all but the first 10k
/// - `"[..10k]"` → `[0, 10_001)` — first 10,001 elements (inclusive end)
/// - `"[..10k)"` → `[0, 10_000)` — first 10k elements (exclusive end)
/// - `"(10k..]"` → `[10_001, MAX)` — all but the first 10,001 (exclusive start)
/// - `"[..]"` → `[0, MAX)` — all elements
///
/// # Empty intervals are rejected
///
/// An interval that selects nothing is always a mistake, and accepting
/// one silently is worse than refusing it. The case that matters is a
/// comma where `..` belongs: `,` separates *intervals* and `..`
/// separates *bounds*, so `"0,1000"` parses as two intervals — a bare
/// number being shorthand for `0..N`, the first is `[0..0)`.
///
/// That used to be accepted, and the two consumers of a window then
/// disagreed about it: the reader built a `WindowedVectorReader` of
/// length zero and reported no records, while the precache path saw a
/// degenerate range, gave up on windowing, and downloaded the entire
/// file. A dataset written `base.fvec[0,1000)` — which is how anyone
/// used to slice notation would write it — got an empty view and a
/// full-size download, in silence.
fn parse_interval(s: &str) -> Result<DSInterval, String> {
    let s = s.trim();

    // Detect bracket types for inclusive/exclusive semantics
    let left_exclusive = s.starts_with('(');
    let right_inclusive = s.ends_with(']');

    // Strip optional surrounding brackets
    let inner = if s.starts_with('[') || s.starts_with('(') {
        &s[1..]
    } else {
        s
    };
    let inner = if inner.ends_with(')') || inner.ends_with(']') {
        &inner[..inner.len() - 1]
    } else {
        inner
    };
    let inner = inner.trim();

    if let Some((left, right)) = inner.split_once("..") {
        let left = left.trim();
        let right = right.trim();

        let mut min_incl = if left.is_empty() {
            0
        } else {
            parse_number_with_suffix(left)?
        };
        let mut max_excl = if right.is_empty() {
            u64::MAX
        } else {
            parse_number_with_suffix(right)?
        };

        // Exclusive start '(' → skip one more element
        if left_exclusive && !left.is_empty() {
            min_incl = min_incl.checked_add(1).ok_or("start overflow")?;
        }
        // Inclusive end ']' → include the boundary element
        if right_inclusive && !right.is_empty() {
            max_excl = max_excl.checked_add(1).ok_or("end overflow")?;
        }

        check_non_empty(DSInterval { min_incl, max_excl }, s)
    } else {
        // Single number = shorthand for 0..N
        let max_excl = parse_number_with_suffix(inner)?;
        check_non_empty(
            DSInterval {
                min_incl: 0,
                max_excl,
            },
            s,
        )
    }
}

/// Reject an interval that selects nothing, naming the likely cause.
///
/// The hint fires when the offending text carries no `..`, because that
/// is the comma-for-`..` mistake and the message is useless without
/// saying so.
fn check_non_empty(iv: DSInterval, raw: &str) -> Result<DSInterval, String> {
    if iv.max_excl > iv.min_incl {
        return Ok(iv);
    }
    if !raw.contains("..") {
        return Err(format!(
            "interval '{raw}' selects no records: a bare number means 0..N, so \
             '{raw}' is [0..{}). Bounds are separated by '..' and intervals by \
             ',' — did you mean a '..' here?",
            iv.max_excl
        ));
    }
    Err(format!(
        "interval '{raw}' selects no records: end ({}) must exceed start ({})",
        iv.max_excl, iv.min_incl
    ))
}

/// Parse a window specification from a string.
///
/// Can be a single interval or comma-separated list of intervals.
pub fn parse_window(s: &str) -> Result<DSWindow, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(DSWindow(vec![]));
    }

    // Strip outer brackets if present: "[0..1K, 2K..3K]"
    let inner = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len() - 1]
    } else {
        s
    };

    let intervals: Result<Vec<DSInterval>, String> = inner
        .split(',')
        .map(|part| parse_interval(part.trim()))
        .collect();
    Ok(DSWindow(intervals?))
}

// ---------------------------------------------------------------------------
// Source string sugar parsing
// ---------------------------------------------------------------------------

/// Parse a source string with sugar notation.
///
/// Forms:
/// - `"file.fvec"` -> path only
/// - `"file.fvec[0..1M]"` -> path + bracket window
/// - `"file.fvec(0..1000)"` -> path + paren window
/// - `"file.fvec[0..1M)"` -> path + mixed-delimiter window
/// - `"file.slab:content"` -> path + namespace
/// - `"file.slab:ns:[0..1K]"` -> path + namespace + window
///
/// The outer delimiters (`[`, `(`, `]`, `)`) are structural — they mark
/// the window region but do not affect interval bound semantics. The
/// inner content is parsed by `parse_window`/`parse_interval`.
pub fn parse_source_string(s: &str) -> Result<DSSource, String> {
    let s = s.trim();
    let (s, declared_count) = split_declared_count(s)?;

    // Check for window delimiters: brackets or parens, including mixed forms
    // like "[0..1M)", "(0..1M]", etc. The opening delimiter can be '[' or '('
    // and the closing delimiter can be ']' or ')'.
    //
    // The outer delimiters in source sugar are structural (they separate
    // the path from the window spec) — they are stripped before the inner
    // content is passed to `parse_window` / `parse_interval`.
    let ends_with_close = s.ends_with(']') || s.ends_with(')');
    if ends_with_close {
        // Find the opening delimiter — first '[' or '(' that could start a window
        let window_start = s.find('[').or_else(|| s.find('('));
        if let Some(start_pos) = window_start {
            let mut path_part = &s[..start_pos];
            let window_part = &s[start_pos + 1..s.len() - 1];

            // Strip trailing colon before delimiter (ns separator in "path:ns:[window]")
            if path_part.ends_with(':') {
                path_part = &path_part[..path_part.len() - 1];
            }

            let (path, namespace) = split_namespace(path_part);
            let window = parse_window(window_part)?;
            return Ok(DSSource {
                path: path.to_string(),
                namespace,
                window,
                declared_count,
            });
        }
    }

    // No window — check for namespace: "file.slab:content"
    let (path, namespace) = split_namespace(s);
    Ok(DSSource {
        path: path.to_string(),
        namespace,
        window: DSWindow::default(),
        declared_count,
    })
}

/// Split a trailing `=<count>` suffix off a source string.
///
/// `"a.u8=4194304"` → `("a.u8", Some(4194304))`
/// `"a.u8[0..1M]=1M"` → `("a.u8[0..1M]", Some(1_000_000))`
/// `"a.u8"` → `("a.u8", None)`
///
/// **A source carrying a query string is never split.** `=` is the
/// key/value separator inside a URL query, so `f.fvec?token=abc123`
/// would otherwise be read as a path with a count. The count form is
/// not lost to such a source: a window (`f.fvec?token=abc[0..1M]`)
/// declares the same cardinality, so the restriction costs nothing.
///
/// The suffix is also only taken when what follows `=` parses as a
/// count. Anything else is left as part of the path, so an unrelated
/// `=` cannot turn into a parse error.
fn split_declared_count(s: &str) -> Result<(&str, Option<u64>), String> {
    if s.contains('?') {
        return Ok((s, None));
    }
    let Some((head, tail)) = s.rsplit_once('=') else {
        return Ok((s, None));
    };
    if head.is_empty() || tail.is_empty() {
        return Ok((s, None));
    }
    match parse_number_with_suffix(tail) {
        Ok(n) if n > 0 => Ok((head.trim_end(), Some(n))),
        // Not a count — leave it in the path rather than failing, so an
        // `=` that means something else to some other consumer survives.
        _ => Ok((s, None)),
    }
}

/// Split a path string on the last colon that follows a dot-extension,
/// treating the part after the colon as a namespace.
///
/// `"file.slab:content"` -> `("file.slab", Some("content"))`
/// `"file.fvec"` -> `("file.fvec", None)`
/// `"../path/file.slab:ns"` -> `("../path/file.slab", Some("ns"))`
fn split_namespace(s: &str) -> (&str, Option<String>) {
    // Find the last colon. If the part before it contains a dot (extension),
    // treat it as namespace separator. Skip Windows drive letters like C:
    if let Some(colon_pos) = s.rfind(':') {
        let before = &s[..colon_pos];
        let after = &s[colon_pos + 1..];
        // Only treat as namespace if there's a dot before the colon (extension)
        // and the part after colon is non-empty and doesn't look like a path
        if before.contains('.')
            && !after.is_empty()
            && !after.contains('/')
            && !after.contains('\\')
        {
            return (before, Some(after.to_string()));
        }
    }
    (s, None)
}

// ---------------------------------------------------------------------------
// Serde implementations
// ---------------------------------------------------------------------------

impl<'de> Deserialize<'de> for DSInterval {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct IntervalVisitor;

        impl<'de> Visitor<'de> for IntervalVisitor {
            type Value = DSInterval;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an interval string like '0..1000' or a number")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<DSInterval, E> {
                parse_interval(v).map_err(de::Error::custom)
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<DSInterval, E> {
                Ok(DSInterval {
                    min_incl: 0,
                    max_excl: v,
                })
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<DSInterval, E> {
                if v < 0 {
                    return Err(de::Error::custom("interval bound cannot be negative"));
                }
                Ok(DSInterval {
                    min_incl: 0,
                    max_excl: v as u64,
                })
            }

            fn visit_map<M>(self, mut map: M) -> Result<DSInterval, M::Error>
            where
                M: de::MapAccess<'de>,
            {
                let mut min_incl: Option<u64> = None;
                let mut max_excl: Option<u64> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "min_incl" => {
                            min_incl = Some(map.next_value()?);
                        }
                        "max_excl" => {
                            max_excl = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                Ok(DSInterval {
                    min_incl: min_incl.unwrap_or(0),
                    max_excl: max_excl.ok_or_else(|| {
                        de::Error::custom("interval map must have 'max_excl' key")
                    })?,
                })
            }
        }

        deserializer.deserialize_any(IntervalVisitor)
    }
}

impl<'de> Deserialize<'de> for DSWindow {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct WindowVisitor;

        impl<'de> Visitor<'de> for WindowVisitor {
            type Value = DSWindow;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a window string, number, or list of intervals")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<DSWindow, E> {
                parse_window(v).map_err(de::Error::custom)
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<DSWindow, E> {
                Ok(DSWindow(vec![DSInterval {
                    min_incl: 0,
                    max_excl: v,
                }]))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<DSWindow, E> {
                if v < 0 {
                    return Err(de::Error::custom("window bound cannot be negative"));
                }
                Ok(DSWindow(vec![DSInterval {
                    min_incl: 0,
                    max_excl: v as u64,
                }]))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<DSWindow, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut intervals = Vec::new();
                while let Some(interval) = seq.next_element::<DSInterval>()? {
                    intervals.push(interval);
                }
                Ok(DSWindow(intervals))
            }
        }

        deserializer.deserialize_any(WindowVisitor)
    }
}

impl<'de> Deserialize<'de> for DSSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct SourceVisitor;

        impl<'de> Visitor<'de> for SourceVisitor {
            type Value = DSSource;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a source string or map with 'source'/'path' key")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<DSSource, E> {
                parse_source_string(v).map_err(de::Error::custom)
            }

            fn visit_map<M>(self, mut map: M) -> Result<DSSource, M::Error>
            where
                M: de::MapAccess<'de>,
            {
                let mut path: Option<String> = None;
                let mut namespace: Option<String> = None;
                let mut window: Option<DSWindow> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "source" | "path" => {
                            path = Some(map.next_value()?);
                        }
                        "namespace" | "ns" => {
                            namespace = Some(map.next_value()?);
                        }
                        "window" => {
                            window = Some(map.next_value()?);
                        }
                        _ => {
                            // Ignore unknown keys
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                let path = path.ok_or_else(|| {
                    de::Error::custom("source map must have 'source' or 'path' key")
                })?;

                Ok(DSSource {
                    path,
                    namespace,
                    window: window.unwrap_or_default(),
                    declared_count: None,
                })
            }
        }

        deserializer.deserialize_any(SourceVisitor)
    }
}

#[cfg(test)]
mod declared_count {
    use super::*;
    /// **`Display` and `parse_source_string` are inverses.**
    ///
    /// Relied upon in two places: a view serializes back to YAML through
    /// `Display`, and the catalog loader hands a parsed source to the
    /// shard model by rendering it and re-parsing — which is what lets
    /// both loaders share one realization instead of mirroring it
    /// (SH-90).
    #[test]
    fn display_and_parse_are_inverses() {
        for raw in [
            "a.u8",
            "a.u8[0..100]",
            "m.slab:ns",
            "m.slab:ns:[0..1K]",
            "a.u8=500",
            "a.u8[0..1M]=1M",
            "https://h/f.fvec",
            "b__0000.fvec",
        ] {
            let once = parse_source_string(raw).unwrap();
            let again = parse_source_string(&once.to_string()).unwrap();
            assert_eq!(once, again, "round trip changed {raw} (via {})", once);
        }
    }

    /// The rendered form brackets a window exactly once. It used to
    /// bracket twice — `a.u8[[0..100]]` — which parsed only because the
    /// window parser strips an extra layer, and which is what a
    /// serialized view wrote to disk.
    #[test]
    fn a_window_is_bracketed_once() {
        let s = parse_source_string("a.u8[0..100]").unwrap();
        assert_eq!(s.to_string(), "a.u8[0..100]");
    }

    /// The `=<count>` suffix parses off the end, with SI suffixes, and
    /// leaves the path intact (SH-61).
    #[test]
    fn a_bare_name_may_declare_its_cardinality() {
        let s = parse_source_string("corpus-part-a.u8=4194304").unwrap();
        assert_eq!(s.path, "corpus-part-a.u8");
        assert_eq!(s.declared_count, Some(4_194_304));
        assert!(s.window.is_empty(), "a count is not a window");
    }

    /// A windowed entry may also carry one, so a mistyped bound is
    /// caught against its own entry as well as against the series total
    /// (SH-62).
    #[test]
    fn a_window_and_a_count_coexist() {
        let s = parse_source_string("a.u8[0..1M]=1M").unwrap();
        assert_eq!(s.path, "a.u8");
        assert_eq!(s.window.0[0].min_incl, 0);
        assert_eq!(s.window.0[0].max_excl, 1_000_000);
        assert_eq!(s.declared_count, Some(1_000_000));
    }

    /// Namespaces still parse alongside a count.
    #[test]
    fn a_namespace_survives_a_count() {
        let s = parse_source_string("meta.slab:mnodes=500").unwrap();
        assert_eq!(s.path, "meta.slab");
        assert_eq!(s.namespace.as_deref(), Some("mnodes"));
        assert_eq!(s.declared_count, Some(500));
    }

    /// **A URL query string is never split on `=`.**
    ///
    /// `=` is the key/value separator inside a query, so a source like
    /// `f.fvec?token=abc` would otherwise read as a path with a count.
    /// The count form is not lost to such a source — a window declares
    /// the same cardinality — so the restriction costs nothing.
    #[test]
    fn a_query_string_is_not_mistaken_for_a_count() {
        let s = parse_source_string("https://h/f.fvec?token=12345").unwrap();
        assert_eq!(s.path, "https://h/f.fvec?token=12345");
        assert_eq!(s.declared_count, None, "the query value is not a count");

        // And such a source can still declare its cardinality by window.
        let w = parse_source_string("https://h/f.fvec?token=12345[0..1M]").unwrap();
        assert_eq!(w.window.0[0].max_excl, 1_000_000);
    }

    /// An `=` whose tail is not a count is left in the path rather than
    /// raising, so an unrelated `=` cannot become a parse error.
    #[test]
    fn a_non_count_suffix_stays_in_the_path() {
        for raw in ["weird=name.u8", "a.u8=", "a.u8=abc", "=5"] {
            let s = parse_source_string(raw).unwrap();
            assert_eq!(s.declared_count, None, "{raw} must not yield a count");
        }
    }

    /// Sources without a count report `None` — the field is additive and
    /// every existing spelling keeps its meaning.
    #[test]
    fn existing_spellings_are_unchanged() {
        for raw in [
            "base.fvec",
            "base.fvec[0..1M]",
            "m.slab:content",
            "m.slab:ns:[0..1K]",
        ] {
            let s = parse_source_string(raw).unwrap();
            assert_eq!(s.declared_count, None, "{raw}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_number_with_suffix --

    #[test]
    fn test_parse_plain_number() {
        assert_eq!(parse_number_with_suffix("1000").unwrap(), 1000);
    }

    #[test]
    fn test_parse_with_suffix_unit_reports_granularity() {
        // Bare digits → granularity 1 (exact).
        assert_eq!(parse_number_with_suffix_unit("1000").unwrap(), (1000, 1));
        // A spelled unit → that unit's multiplier.
        assert_eq!(
            parse_number_with_suffix_unit("118MB").unwrap(),
            (118_000_000, 1_000_000)
        );
        assert_eq!(
            parse_number_with_suffix_unit("12g").unwrap(),
            (12_000_000_000, 1_000_000_000)
        );
        assert_eq!(
            parse_number_with_suffix_unit("5KiB").unwrap(),
            (5_120, 1_024)
        );
        // Compound → the smallest term's unit is the granularity.
        assert_eq!(
            parse_number_with_suffix_unit("1g24m").unwrap(),
            (1_024_000_000, 1_000_000)
        );
    }

    #[test]
    fn test_parse_k_suffix() {
        assert_eq!(parse_number_with_suffix("1K").unwrap(), 1_000);
        assert_eq!(parse_number_with_suffix("5k").unwrap(), 5_000);
    }

    #[test]
    fn test_parse_m_suffix() {
        assert_eq!(parse_number_with_suffix("1M").unwrap(), 1_000_000);
        assert_eq!(parse_number_with_suffix("10m").unwrap(), 10_000_000);
    }

    #[test]
    fn test_parse_b_g_suffix() {
        assert_eq!(parse_number_with_suffix("1B").unwrap(), 1_000_000_000);
        assert_eq!(parse_number_with_suffix("1G").unwrap(), 1_000_000_000);
    }

    #[test]
    fn test_parse_underscore_separator() {
        assert_eq!(parse_number_with_suffix("1_000_000").unwrap(), 1_000_000);
        assert_eq!(parse_number_with_suffix("1_000").unwrap(), 1_000);
    }

    #[test]
    fn test_parse_t_suffix() {
        assert_eq!(parse_number_with_suffix("1T").unwrap(), 1_000_000_000_000);
        assert_eq!(parse_number_with_suffix("2t").unwrap(), 2_000_000_000_000);
    }

    #[test]
    fn test_parse_iso_decimal_suffixes() {
        assert_eq!(parse_number_with_suffix("1KB").unwrap(), 1_000);
        assert_eq!(parse_number_with_suffix("1kb").unwrap(), 1_000);
        assert_eq!(parse_number_with_suffix("1MB").unwrap(), 1_000_000);
        assert_eq!(parse_number_with_suffix("1GB").unwrap(), 1_000_000_000);
        assert_eq!(parse_number_with_suffix("1TB").unwrap(), 1_000_000_000_000);
    }

    #[test]
    fn test_parse_iso_binary_suffixes() {
        assert_eq!(parse_number_with_suffix("1KiB").unwrap(), 1_024);
        assert_eq!(parse_number_with_suffix("1MiB").unwrap(), 1_048_576);
        assert_eq!(parse_number_with_suffix("1GiB").unwrap(), 1_073_741_824);
        assert_eq!(parse_number_with_suffix("1TiB").unwrap(), 1_099_511_627_776);
    }

    #[test]
    fn test_parse_iso_with_multiplier() {
        assert_eq!(parse_number_with_suffix("4GiB").unwrap(), 4_294_967_296);
        assert_eq!(parse_number_with_suffix("16KiB").unwrap(), 16_384);
    }

    // -- parse_interval --

    #[test]
    fn test_parse_interval_range() {
        let i = parse_interval("0..1000").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1000);
    }

    #[test]
    fn test_parse_interval_with_suffix() {
        let i = parse_interval("0..1M").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1_000_000);
    }

    #[test]
    fn test_parse_interval_bracketed() {
        let i = parse_interval("[0..1000)").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1000);
    }

    #[test]
    fn test_parse_interval_single_number() {
        let i = parse_interval("1M").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1_000_000);
    }

    #[test]
    fn test_parse_interval_with_underscores() {
        let i = parse_interval("0..1_000").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1_000);
    }

    #[test]
    fn test_parse_interval_open_right() {
        // [10k..] → all but the first 10k
        let i = parse_interval("[10k..]").unwrap();
        assert_eq!(i.min_incl, 10_000);
        assert_eq!(i.max_excl, u64::MAX);
    }

    #[test]
    fn test_parse_interval_open_left_inclusive_right() {
        // [..10k] → first 10,001 elements (inclusive end)
        let i = parse_interval("[..10k]").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 10_001);
    }

    #[test]
    fn test_parse_interval_open_left_exclusive_right() {
        // [..10k) → first 10k elements (exclusive end)
        let i = parse_interval("[..10k)").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 10_000);
    }

    #[test]
    fn test_parse_interval_exclusive_left_open_right() {
        // (10k..] → all but the first 10,001
        let i = parse_interval("(10k..]").unwrap();
        assert_eq!(i.min_incl, 10_001);
        assert_eq!(i.max_excl, u64::MAX);
    }

    #[test]
    fn test_parse_interval_inclusive_end() {
        // [0..100] → [0, 101) inclusive end
        let i = parse_interval("[0..100]").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 101);
    }

    #[test]
    fn test_parse_interval_exclusive_start() {
        // (5..10) → [6, 10)
        let i = parse_interval("(5..10)").unwrap();
        assert_eq!(i.min_incl, 6);
        assert_eq!(i.max_excl, 10);
    }

    #[test]
    fn test_parse_interval_all() {
        // [..] → [0, MAX) — all elements
        let i = parse_interval("[..]").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, u64::MAX);
    }

    // -- parse_window --

    #[test]
    fn test_parse_window_single() {
        let w = parse_window("0..1000").unwrap();
        assert_eq!(w.0.len(), 1);
        assert_eq!(w.0[0].min_incl, 0);
        assert_eq!(w.0[0].max_excl, 1000);
    }

    #[test]
    fn test_parse_window_bracketed_list() {
        let w = parse_window("[0..1K, 2K..3K]").unwrap();
        assert_eq!(w.0.len(), 2);
        assert_eq!(w.0[0].max_excl, 1000);
        assert_eq!(w.0[1].min_incl, 2000);
        assert_eq!(w.0[1].max_excl, 3000);
    }

    /// **The comma-for-`..` hazard.** `,` separates intervals and `..`
    /// separates bounds, so `"0,1000"` is two intervals, not one — and
    /// because a bare number means `0..N`, the first of them is empty.
    ///
    /// Accepting that used to split the two consumers of a window: the
    /// reader clipped to zero records and the precache path gave up on
    /// windowing and fetched the whole file. Neither said anything.
    #[test]
    fn comma_where_dotdot_belongs_is_rejected() {
        let err = parse_window("0,1000").unwrap_err();
        assert!(err.contains("selects no records"), "{err}");
        assert!(
            err.contains(".."),
            "the message must name the separator: {err}"
        );

        // The same mistake inside the source-string suffix sugar.
        let err = parse_source_string("base.fvec[0,1000)").unwrap_err();
        assert!(err.contains("selects no records"), "{err}");

        // And the spelling that was meant still works.
        let ok = parse_source_string("base.fvec[0..1000)").unwrap();
        assert_eq!(ok.path, "base.fvec");
        assert_eq!(ok.window.0.len(), 1);
        assert_eq!(
            ok.window.0[0],
            DSInterval {
                min_incl: 0,
                max_excl: 1000
            }
        );
    }

    /// Any interval that selects nothing is refused, however it is
    /// spelled — a window naming no records is always a mistake, and
    /// the degenerate first interval above is only the common way to
    /// write one by accident.
    #[test]
    fn empty_intervals_are_rejected_however_spelled() {
        for spec in ["0", "5..5", "[5..5)", "(5..5]", "1000..10", "[..0)"] {
            assert!(
                parse_interval(spec).is_err(),
                "'{spec}' selects nothing and should be refused"
            );
        }
        // Adjacent non-empty forms still parse.
        for spec in ["1", "5..6", "[5..6)", "(4..5]", "[..1)"] {
            assert!(parse_interval(spec).is_ok(), "'{spec}' should still parse");
        }
    }

    /// A genuine multi-interval window is unaffected — the rejection is
    /// of empty intervals, not of the list form.
    #[test]
    fn multi_interval_windows_still_parse() {
        let w = parse_window("[0..1K, 2K..3K]").unwrap();
        assert_eq!(w.0.len(), 2);
    }

    #[test]
    fn test_parse_window_empty() {
        let w = parse_window("").unwrap();
        assert!(w.is_empty());
    }

    // -- parse_source_string --

    #[test]
    fn test_source_bare_string() {
        let s = parse_source_string("file.fvec").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert_eq!(s.namespace, None);
        assert!(s.window.is_empty());
    }

    #[test]
    fn test_source_bracket_window() {
        let s = parse_source_string("file.fvec[0..1M]").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert_eq!(s.namespace, None);
        assert_eq!(s.window.0.len(), 1);
        assert_eq!(s.window.0[0].max_excl, 1_000_000);
    }

    #[test]
    fn test_source_paren_window() {
        let s = parse_source_string("file.fvec(0..1000)").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert_eq!(s.window.0[0].max_excl, 1000);
    }

    #[test]
    fn test_source_namespace() {
        let s = parse_source_string("file.slab:content").unwrap();
        assert_eq!(s.path, "file.slab");
        assert_eq!(s.namespace.as_deref(), Some("content"));
        assert!(s.window.is_empty());
    }

    #[test]
    fn test_source_namespace_and_window() {
        let s = parse_source_string("file.slab:ns:[0..1K]").unwrap();
        assert_eq!(s.path, "file.slab");
        assert_eq!(s.namespace.as_deref(), Some("ns"));
        assert_eq!(s.window.0.len(), 1);
        assert_eq!(s.window.0[0].max_excl, 1000);
    }

    #[test]
    fn test_source_yaml_bare_string() {
        let s: DSSource = serde_yaml::from_str("\"file.fvec\"").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert!(s.window.is_empty());
    }

    #[test]
    fn test_source_yaml_map_with_source() {
        let yaml = r#"
source: f.fvec
"#;
        let s: DSSource = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(s.path, "f.fvec");
    }

    #[test]
    fn test_source_yaml_map_with_path() {
        let yaml = r#"
path: f.fvec
"#;
        let s: DSSource = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(s.path, "f.fvec");
    }

    #[test]
    fn test_source_yaml_map_with_window() {
        let yaml = r#"
source: f.fvec
window: "0..1000"
"#;
        let s: DSSource = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(s.path, "f.fvec");
        assert_eq!(s.window.0.len(), 1);
        assert_eq!(s.window.0[0].max_excl, 1000);
    }

    #[test]
    fn test_window_yaml_number() {
        let w: DSWindow = serde_yaml::from_str("1000000").unwrap();
        assert_eq!(w.0.len(), 1);
        assert_eq!(w.0[0].min_incl, 0);
        assert_eq!(w.0[0].max_excl, 1_000_000);
    }

    #[test]
    fn test_window_yaml_list() {
        let yaml = r#"
- "0..1000"
- "2000..3000"
"#;
        let w: DSWindow = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(w.0.len(), 2);
    }

    #[test]
    fn test_interval_yaml_string() {
        let i: DSInterval = serde_yaml::from_str("\"0..1K\"").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1000);
    }

    // -- mixed bracket forms in source sugar --

    #[test]
    fn test_source_mixed_bracket_paren_window() {
        // "[0..1M)" — half-open range in source sugar
        let s = parse_source_string("file.fvec[0..1M)").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert_eq!(s.window.0.len(), 1);
        assert_eq!(s.window.0[0].min_incl, 0);
        assert_eq!(s.window.0[0].max_excl, 1_000_000);
    }

    #[test]
    fn test_source_mixed_paren_bracket_window() {
        // "(0..1M]" — exclusive start, inclusive end
        let s = parse_source_string("file.fvec(0..1M]").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert_eq!(s.window.0.len(), 1);
        // Outer delimiters are structural — inner "0..1M" parses as default half-open
        assert_eq!(s.window.0[0].min_incl, 0);
        assert_eq!(s.window.0[0].max_excl, 1_000_000);
    }

    #[test]
    fn test_source_mixed_with_interpolated_number() {
        // Simulates post-interpolation: "base.mvec[0..407000000)"
        let s = parse_source_string("base.mvec[0..407000000)").unwrap();
        assert_eq!(s.path, "base.mvec");
        assert_eq!(s.window.0[0].min_incl, 0);
        assert_eq!(s.window.0[0].max_excl, 407_000_000);
    }

    #[test]
    fn test_source_mixed_with_namespace() {
        let s = parse_source_string("file.slab:ns:[0..1K)").unwrap();
        assert_eq!(s.path, "file.slab");
        assert_eq!(s.namespace.as_deref(), Some("ns"));
        assert_eq!(s.window.0[0].max_excl, 1000);
    }
}
