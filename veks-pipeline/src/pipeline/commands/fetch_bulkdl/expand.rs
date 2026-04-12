// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! URL template expansion for bulk downloads.
//!
//! Given a URL containing `${token}` placeholders and a map of token
//! ranges, produces the Cartesian product of all token values as
//! `(url, filename)` pairs.

use std::collections::HashMap;

use super::config::TokenSpec;

/// Expand all `${token}` placeholders in a URL template.
///
/// Computes the Cartesian product of all token ranges and substitutes
/// each combination into the template. Returns `(expanded_url, local_filename)`
/// pairs where the filename is the last path segment of the URL.
pub fn expand_tokens(
    baseurl: &str,
    tokens: &HashMap<String, TokenSpec>,
) -> Vec<(String, String)> {
    // Collect token names and their value ranges
    let token_names: Vec<&String> = tokens.keys().collect();
    let ranges: Vec<Vec<i64>> = token_names
        .iter()
        .map(|name| {
            let spec = &tokens[*name];
            (spec.start..=spec.end).collect()
        })
        .collect();

    if token_names.is_empty() {
        // No tokens — single URL
        let filename = url_to_filename(baseurl);
        return vec![(baseurl.to_string(), filename)];
    }

    // Cartesian product of all token ranges
    let mut combos: Vec<Vec<i64>> = vec![vec![]];
    for range in &ranges {
        let mut new_combos = Vec::new();
        for combo in &combos {
            for val in range {
                let mut c = combo.clone();
                c.push(*val);
                new_combos.push(c);
            }
        }
        combos = new_combos;
    }

    let pad_widths: Vec<usize> = token_names
        .iter()
        .map(|name| tokens[*name].pad_width)
        .collect();

    combos
        .into_iter()
        .map(|vals| {
            let mut url = baseurl.to_string();
            for (i, name) in token_names.iter().enumerate() {
                let placeholder = format!("${{{}}}", name);
                let formatted = if pad_widths[i] > 0 {
                    format!("{:0>width$}", vals[i], width = pad_widths[i])
                } else {
                    vals[i].to_string()
                };
                url = url.replace(&placeholder, &formatted);
            }
            let filename = url_to_filename(&url);
            (url, filename)
        })
        .collect()
}

/// Extract the filename from a URL (last path segment, query string stripped).
pub fn url_to_filename(url: &str) -> String {
    url.rsplit('/')
        .next()
        .unwrap_or("download")
        .split('?')
        .next()
        .unwrap_or("download")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::commands::fetch_bulkdl::config::parse_token_spec;

    #[test]
    fn test_parse_token_spec() {
        let spec = parse_token_spec("[0..409]").unwrap();
        assert_eq!(spec.start, 0);
        assert_eq!(spec.end, 409);
    }

    #[test]
    fn test_parse_token_spec_negative() {
        let spec = parse_token_spec("[-5..5]").unwrap();
        assert_eq!(spec.start, -5);
        assert_eq!(spec.end, 5);
    }

    #[test]
    fn test_parse_token_spec_invalid() {
        assert!(parse_token_spec("0..409").is_none());
        assert!(parse_token_spec("[abc]").is_none());
    }

    #[test]
    fn test_url_to_filename() {
        assert_eq!(
            url_to_filename("https://example.com/data/file.npy"),
            "file.npy"
        );
        assert_eq!(
            url_to_filename("https://example.com/data/file.npy?token=abc"),
            "file.npy"
        );
    }

    #[test]
    fn test_expand_tokens_single() {
        let mut tokens = HashMap::new();
        tokens.insert("i".to_string(), TokenSpec { start: 0, end: 2, pad_width: 0 });
        let results = expand_tokens("https://example.com/img_emb_${i}.npy", &tokens);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "https://example.com/img_emb_0.npy");
        assert_eq!(results[0].1, "img_emb_0.npy");
        assert_eq!(results[2].0, "https://example.com/img_emb_2.npy");
    }

    #[test]
    fn test_expand_tokens_none() {
        let tokens = HashMap::new();
        let results = expand_tokens("https://example.com/single.npy", &tokens);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "single.npy");
    }

    #[test]
    fn test_expand_tokens_zero_padded() {
        let mut tokens = HashMap::new();
        tokens.insert("part".to_string(), TokenSpec { start: 0, end: 2, pad_width: 2 });
        let results = expand_tokens("https://example.com/part-${part}.dat", &tokens);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "https://example.com/part-00.dat");
        assert_eq!(results[1].0, "https://example.com/part-01.dat");
        assert_eq!(results[2].0, "https://example.com/part-02.dat");
    }

    #[test]
    fn test_parse_token_spec_zero_padded() {
        let spec = parse_token_spec("[00..31]").unwrap();
        assert_eq!(spec.start, 0);
        assert_eq!(spec.end, 31);
        assert_eq!(spec.pad_width, 2);

        let spec3 = parse_token_spec("[000..409]").unwrap();
        assert_eq!(spec3.pad_width, 3);

        // No padding
        let spec_plain = parse_token_spec("[0..409]").unwrap();
        assert_eq!(spec_plain.pad_width, 0);
    }
}
