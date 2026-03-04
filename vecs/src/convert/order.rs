// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

/// Detected sort order for file names in a directory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Lexicographic and numerical order agree
    Consistent,
    /// Lexicographic order differs from numerical — using lexicographic
    Lexicographic,
    /// No numerical component found in filenames — using lexicographic
    NonNumeric,
}

impl fmt::Display for SortOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Consistent => write!(f, "lexicographic (consistent with numerical)"),
            Self::Lexicographic => {
                write!(f, "lexicographic (differs from numerical sort)")
            }
            Self::NonNumeric => write!(f, "lexicographic (no numerical component detected)"),
        }
    }
}

/// Detect whether filenames sort the same way lexicographically and numerically.
///
/// Extracts the first numeric substring from each filename for numerical comparison.
/// Returns the detected sort order, always using lexicographic sorting for actual file ordering.
pub fn detect_sort_order(filenames: &[String]) -> SortOrder {
    if filenames.len() <= 1 {
        return SortOrder::Consistent;
    }

    let mut lex_sorted = filenames.to_vec();
    lex_sorted.sort();

    // Try to extract numbers from filenames
    let numbers: Vec<Option<i64>> = lex_sorted.iter().map(|f| extract_number(f)).collect();

    if numbers.iter().all(|n| n.is_none()) {
        return SortOrder::NonNumeric;
    }

    // Check if the lexicographic order is consistent with numerical order
    let has_numbers: Vec<(i64, &str)> = numbers
        .iter()
        .zip(lex_sorted.iter())
        .filter_map(|(n, name)| n.map(|num| (num, name.as_str())))
        .collect();

    let is_sorted = has_numbers.windows(2).all(|w| w[0].0 <= w[1].0);

    if is_sorted {
        SortOrder::Consistent
    } else {
        SortOrder::Lexicographic
    }
}

/// Extract the first numeric substring from a filename
fn extract_number(filename: &str) -> Option<i64> {
    let stem = filename.rsplit('/').next().unwrap_or(filename);
    let mut num_str = String::new();
    let mut found = false;

    for ch in stem.chars() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
            found = true;
        } else if found {
            break;
        }
    }

    if found {
        num_str.parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_order() {
        let files: Vec<String> = vec![
            "file_001.npy".into(),
            "file_002.npy".into(),
            "file_003.npy".into(),
        ];
        assert_eq!(detect_sort_order(&files), SortOrder::Consistent);
    }

    #[test]
    fn test_inconsistent_order() {
        // Lexicographic: file_1, file_10, file_2
        // Numerical: file_1, file_2, file_10
        let files: Vec<String> = vec![
            "file_1.npy".into(),
            "file_10.npy".into(),
            "file_2.npy".into(),
        ];
        assert_eq!(detect_sort_order(&files), SortOrder::Lexicographic);
    }

    #[test]
    fn test_no_numbers() {
        let files: Vec<String> = vec!["alpha.npy".into(), "beta.npy".into()];
        assert_eq!(detect_sort_order(&files), SortOrder::NonNumeric);
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("file_123.npy"), Some(123));
        assert_eq!(extract_number("img_emb_0.npy"), Some(0));
        assert_eq!(extract_number("nonum.npy"), None);
    }
}
