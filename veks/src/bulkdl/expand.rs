// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::config::TokenSpec;

/// Expands all `${token}` placeholders in the URL template, producing
/// all combinations of (expanded_url, local_filename) pairs.
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

    combos
        .into_iter()
        .map(|vals| {
            let mut url = baseurl.to_string();
            for (i, name) in token_names.iter().enumerate() {
                let placeholder = format!("${{{}}}", name);
                url = url.replace(&placeholder, &vals[i].to_string());
            }
            let filename = url_to_filename(&url);
            (url, filename)
        })
        .collect()
}

/// Extracts the filename from a URL (last path segment)
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
    use crate::bulkdl::config::parse_token_spec;

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
        tokens.insert("i".to_string(), TokenSpec { start: 0, end: 2 });
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
}
