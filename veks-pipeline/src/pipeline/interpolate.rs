// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Variable interpolation engine for command stream pipelines.
//!
//! Supports the following substitution patterns:
//! - `${name}` — lookup in defaults map (includes variables.yaml)
//! - `${name:-fallback}` — default value if not found
//! - `${env:VAR}` — environment variable
//! - `${variables:name}` — explicitly from variables.yaml only (short form)
//! - `${variables.yaml:name}` — explicitly from variables.yaml only (long form)
//! - `$$` — literal `$` (escape). Use `$${name}` to produce `${name}` in the
//!   output without variable expansion. Needed for commands like `fetch bulkdl`
//!   whose token placeholders use `${token}` syntax.
//!
//! The qualified `variables:` / `variables.yaml:` prefixes are unambiguous even
//! on the command line — shells will not expand `${variables:foo}` or
//! `${variables.yaml:foo}` because they are not valid shell variable names.
//!
//! Implicit variables available in every context:
//! - `${dataset_dir}` — directory containing the dataset.yaml
//! - `${workspace}` — same as dataset_dir (alias)
//! - `${scratch}` — temporary scratch directory (`<workspace>/.scratch`)
//! - `${cache}` — reusable cache directory (`<workspace>/.cache`)

use std::path::Path;

use indexmap::IndexMap;

/// Interpolate `${...}` patterns in the given string.
///
/// Looks up variable names in `defaults` first, then checks for special
/// prefixes (`env:`). Supports `${name:-fallback}` default values.
///
/// Returns an error if a required variable is not found and has no fallback.
pub fn interpolate(
    input: &str,
    defaults: &IndexMap<String, String>,
    workspace: &Path,
) -> Result<String, String> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'$') {
            // Escaped dollar: `$$` produces a literal `$`.
            chars.next(); // consume second '$'
            result.push('$');
        } else if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_expr = String::new();
            let mut depth = 1;
            while let Some(c) = chars.next() {
                if c == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                } else if c == '{' {
                    depth += 1;
                }
                var_expr.push(c);
            }
            if depth != 0 {
                return Err(format!("unclosed variable expression in: {}", input));
            }
            let resolved = resolve_var(&var_expr, defaults, workspace)?;
            result.push_str(&resolved);
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

/// Resolve a single variable expression (the part between `${` and `}`).
fn resolve_var(
    expr: &str,
    defaults: &IndexMap<String, String>,
    workspace: &Path,
) -> Result<String, String> {
    // Split on `:-` for default value
    let (name, fallback) = if let Some(idx) = expr.find(":-") {
        (&expr[..idx], Some(&expr[idx + 2..]))
    } else {
        (expr, None)
    };

    // Check for env: prefix
    if let Some(env_var) = name.strip_prefix("env:") {
        return match std::env::var(env_var) {
            Ok(val) => Ok(val),
            Err(_) => match fallback {
                Some(fb) => Ok(fb.to_string()),
                None => Err(format!("environment variable '{}' not set", env_var)),
            },
        };
    }

    // Check for variables: or variables.yaml: prefix — resolve exclusively
    // from variables.yaml, bypassing defaults and implicit variables.
    let var_name = name.strip_prefix("variables.yaml:")
        .or_else(|| name.strip_prefix("variables:"));
    if let Some(var_name) = var_name {
        let vars = super::variables::load(workspace)
            .map_err(|e| format!("failed to load variables.yaml: {}", e))?;
        return match vars.get(var_name) {
            Some(val) => Ok(val.clone()),
            None => match fallback {
                Some(fb) => Ok(fb.to_string()),
                None => Err(format!(
                    "variable '{}' not found in variables.yaml",
                    var_name,
                )),
            },
        };
    }

    // Check implicit variables
    match name {
        "dataset_dir" | "workspace" => {
            return Ok(workspace.to_string_lossy().into_owned());
        }
        "scratch" => {
            return Ok(workspace.join(".scratch").to_string_lossy().into_owned());
        }
        "cache" => {
            return Ok(workspace.join(".cache").to_string_lossy().into_owned());
        }
        _ => {}
    }

    // Check defaults map
    if let Some(val) = defaults.get(name) {
        return Ok(val.clone());
    }

    // Use fallback or error
    match fallback {
        Some(fb) => Ok(fb.to_string()),
        None => Err(format!("variable '{}' not defined", name)),
    }
}

/// Interpolate all string values in a step's options map.
///
/// Non-string YAML values are converted to their string representation.
pub fn interpolate_options(
    options: &IndexMap<String, serde_yaml::Value>,
    defaults: &IndexMap<String, String>,
    workspace: &Path,
) -> Result<IndexMap<String, String>, String> {
    let mut resolved = IndexMap::new();
    for (key, value) in options {
        let raw = match value {
            serde_yaml::Value::String(s) => s.clone(),
            serde_yaml::Value::Number(n) => n.to_string(),
            serde_yaml::Value::Bool(b) => b.to_string(),
            serde_yaml::Value::Null => continue,
            other => format!("{:?}", other),
        };
        let interpolated = interpolate(&raw, defaults, workspace)
            .map_err(|e| format!("in option '{}': {}", key, e))?;
        resolved.insert(key.clone(), interpolated);
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> IndexMap<String, String> {
        let mut m = IndexMap::new();
        m.insert("seed".to_string(), "42".to_string());
        m.insert("metric".to_string(), "COSINE".to_string());
        m
    }

    #[test]
    fn test_simple_substitution() {
        let result = interpolate("seed=${seed}", &defaults(), Path::new("/data")).unwrap();
        assert_eq!(result, "seed=42");
    }

    #[test]
    fn test_multiple_vars() {
        let result =
            interpolate("${seed}-${metric}", &defaults(), Path::new("/data")).unwrap();
        assert_eq!(result, "42-COSINE");
    }

    #[test]
    fn test_fallback() {
        let result =
            interpolate("${missing:-default_val}", &defaults(), Path::new("/data")).unwrap();
        assert_eq!(result, "default_val");
    }

    #[test]
    fn test_missing_var_error() {
        let result = interpolate("${missing}", &defaults(), Path::new("/data"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not defined"));
    }

    #[test]
    fn test_workspace_implicit() {
        let result =
            interpolate("dir=${workspace}", &defaults(), Path::new("/my/data")).unwrap();
        assert_eq!(result, "dir=/my/data");
    }

    #[test]
    fn test_dataset_dir_implicit() {
        let result =
            interpolate("${dataset_dir}/file.fvec", &defaults(), Path::new("/my/data"))
                .unwrap();
        assert_eq!(result, "/my/data/file.fvec");
    }

    #[test]
    fn test_env_var() {
        // SAFETY: test-only, no other threads depend on this variable
        unsafe { std::env::set_var("VECS_TEST_VAR", "hello") };
        let result =
            interpolate("${env:VECS_TEST_VAR}", &defaults(), Path::new("/data")).unwrap();
        assert_eq!(result, "hello");
        unsafe { std::env::remove_var("VECS_TEST_VAR") };
    }

    #[test]
    fn test_env_var_with_fallback() {
        let result = interpolate(
            "${env:VECS_NONEXISTENT:-fallback}",
            &defaults(),
            Path::new("/data"),
        )
        .unwrap();
        assert_eq!(result, "fallback");
    }

    #[test]
    fn test_scratch_implicit() {
        let result =
            interpolate("${scratch}/tmp.fvec", &defaults(), Path::new("/my/data")).unwrap();
        assert_eq!(result, "/my/data/.scratch/tmp.fvec");
    }

    #[test]
    fn test_cache_implicit() {
        let result =
            interpolate("${cache}/cached.fvec", &defaults(), Path::new("/my/data")).unwrap();
        assert_eq!(result, "/my/data/.cache/cached.fvec");
    }

    #[test]
    fn test_no_vars() {
        let result = interpolate("plain text", &defaults(), Path::new("/data")).unwrap();
        assert_eq!(result, "plain text");
    }

    #[test]
    fn test_empty_string() {
        let result = interpolate("", &defaults(), Path::new("/data")).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_variables_yaml_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        super::super::variables::set_and_save(tmp.path(), "vector_count", "407314954").unwrap();

        let result = interpolate(
            "${variables.yaml:vector_count}",
            &defaults(),
            tmp.path(),
        ).unwrap();
        assert_eq!(result, "407314954");
    }

    #[test]
    fn test_variables_short_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        super::super::variables::set_and_save(tmp.path(), "dim", "512").unwrap();

        let result = interpolate(
            "${variables:dim}",
            &defaults(),
            tmp.path(),
        ).unwrap();
        assert_eq!(result, "512");
    }

    #[test]
    fn test_variables_prefix_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let result = interpolate(
            "${variables:nonexistent}",
            &defaults(),
            tmp.path(),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found in variables.yaml"));
    }

    #[test]
    fn test_variables_prefix_with_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let result = interpolate(
            "${variables:missing:-default_val}",
            &defaults(),
            tmp.path(),
        ).unwrap();
        assert_eq!(result, "default_val");
    }

    #[test]
    fn test_interpolate_options() {
        let mut opts = IndexMap::new();
        opts.insert(
            "seed".to_string(),
            serde_yaml::Value::String("${seed}".to_string()),
        );
        opts.insert(
            "count".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(100)),
        );

        let resolved = interpolate_options(&opts, &defaults(), Path::new("/data")).unwrap();
        assert_eq!(resolved.get("seed").unwrap(), "42");
        assert_eq!(resolved.get("count").unwrap(), "100");
    }

    #[test]
    fn test_dollar_escape() {
        let result = interpolate("url_$${number}.npy", &defaults(), Path::new("/data")).unwrap();
        assert_eq!(result, "url_${number}.npy");
    }

    #[test]
    fn test_dollar_escape_with_vars() {
        let result = interpolate(
            "https://host/$${number}.npy?seed=${seed}",
            &defaults(),
            Path::new("/data"),
        )
        .unwrap();
        assert_eq!(result, "https://host/${number}.npy?seed=42");
    }

    #[test]
    fn test_dollar_escape_standalone() {
        let result = interpolate("cost: $$5", &defaults(), Path::new("/data")).unwrap();
        assert_eq!(result, "cost: $5");
    }

    #[test]
    fn test_interpolate_options_with_escape() {
        let mut opts = IndexMap::new();
        opts.insert(
            "baseurl".to_string(),
            serde_yaml::Value::String(
                "https://host/img_$${number}.npy".to_string(),
            ),
        );

        let resolved = interpolate_options(&opts, &defaults(), Path::new("/data")).unwrap();
        assert_eq!(
            resolved.get("baseurl").unwrap(),
            "https://host/img_${number}.npy"
        );
    }
}
