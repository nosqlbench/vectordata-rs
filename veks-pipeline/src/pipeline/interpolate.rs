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
//! - `${cache}` — reusable cache directory (`<workspace>/.cache`)
//!
//! Path variables expand per [`PathBase`]: workspace-relative for values a
//! command will re-resolve against the workspace, joined for standalone
//! emitted artifacts.

use std::path::Path;

use indexmap::IndexMap;

/// How the implicit path variables (`${cache}`, `${workspace}`,
/// `${dataset_dir}`) expand.
///
/// The two modes exist because two consumers need different things from the
/// same template, and conflating them is what produced `<ws>/<ws>/.cache`
/// paths when a pipeline ran from a parent directory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathBase {
    /// Expand workspace-relative (`.cache`, `.`). Correct for values a
    /// command will receive as an option and re-resolve against the
    /// workspace itself — which is every executed step, and therefore also
    /// the staleness comparison and artifact projection that must match it.
    /// Independent of the invoking directory, so recorded options stay
    /// stable.
    WorkspaceRelative,
    /// Expand joined onto the workspace. Correct for standalone artifacts —
    /// emitted YAML or shell scripts — where nothing re-resolves the path
    /// afterwards, so it has to be complete on its own.
    Joined,
}

/// Interpolate `${...}` patterns in the given string.
///
/// Looks up variable names in `defaults` first, then checks for special
/// prefixes (`env:`). Supports `${name:-fallback}` default values.
/// `base` selects how the implicit path variables expand.
///
/// Returns an error if a required variable is not found and has no fallback.
pub fn interpolate(
    input: &str,
    defaults: &IndexMap<String, String>,
    workspace: &Path,
    base: PathBase,
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
            for c in chars.by_ref() {
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
            let resolved = resolve_var(&var_expr, defaults, workspace, base)?;
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
    base: PathBase,
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

    // Check implicit variables.
    //
    // These expand to **workspace-relative** paths, matching every literal
    // path in `dataset.yaml` (`profiles/base/base_vectors.fvecs` and the
    // like) and the `resolve_path(value, &ctx.workspace)` that each command
    // applies to its path options. Expanding them workspace-*joined*
    // instead produced a CWD-relative string that commands then re-based
    // onto the workspace, so running a dataset from a parent directory
    // wrote `<ws>/<ws>/.cache/...`; it only looked correct because the
    // second join is a no-op when the workspace is absolute or is `.`,
    // which is how datasets are usually run.
    //
    // Being workspace-relative also makes the expansion independent of the
    // invoking directory, so the resolved options recorded in provenance no
    // longer change purely because the pipeline was launched from somewhere
    // else. Note that this does not by itself make staleness
    // invocation-independent: the progress log still records each output at
    // its run-time-resolved path, so a step whose first run was launched
    // from a parent directory is still reported stale when re-run from
    // inside the dataset. That is a smaller, separate issue — a spurious
    // recompute rather than a write to the wrong place.
    match name {
        "dataset_dir" | "workspace" => {
            return Ok(match base {
                PathBase::WorkspaceRelative => ".".to_string(),
                PathBase::Joined => workspace.to_string_lossy().into_owned(),
            });
        }
        "cache" => {
            return Ok(match base {
                PathBase::WorkspaceRelative => ".cache".to_string(),
                PathBase::Joined => workspace.join(".cache").to_string_lossy().into_owned(),
            });
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
    base: PathBase,
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
        let interpolated = interpolate(&raw, defaults, workspace, base)
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
        let result = interpolate("seed=${seed}", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(result, "seed=42");
    }

    #[test]
    fn test_multiple_vars() {
        let result =
            interpolate("${seed}-${metric}", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(result, "42-COSINE");
    }

    #[test]
    fn test_fallback() {
        let result =
            interpolate("${missing:-default_val}", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(result, "default_val");
    }

    #[test]
    fn test_missing_var_error() {
        let result = interpolate("${missing}", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not defined"));
    }

    #[test]
    fn test_workspace_implicit() {
        // Joined for standalone emission, `.` for values a command will
        // re-resolve against the workspace itself.
        let joined =
            interpolate("dir=${workspace}", &defaults(), Path::new("/my/data"), PathBase::Joined)
                .unwrap();
        assert_eq!(joined, "dir=/my/data");
        let relative = interpolate(
            "dir=${workspace}",
            &defaults(),
            Path::new("/my/data"),
            PathBase::WorkspaceRelative,
        )
        .unwrap();
        assert_eq!(relative, "dir=.");
    }

    #[test]
    fn test_dataset_dir_implicit() {
        let joined = interpolate(
            "${dataset_dir}/file.fvec",
            &defaults(),
            Path::new("/my/data"),
            PathBase::Joined,
        )
        .unwrap();
        assert_eq!(joined, "/my/data/file.fvec");
        let relative = interpolate(
            "${dataset_dir}/file.fvec",
            &defaults(),
            Path::new("/my/data"),
            PathBase::WorkspaceRelative,
        )
        .unwrap();
        assert_eq!(relative, "./file.fvec");
    }

    /// The doubling regression: a workspace-relative expansion must stay
    /// independent of the workspace, because the command re-bases it. With
    /// the old joined-only behaviour this produced `datasets/d/.cache/...`,
    /// which `resolve_path` then turned into
    /// `datasets/d/datasets/d/.cache/...`.
    #[test]
    fn test_cache_expansion_survives_re_resolution() {
        let ws = Path::new("datasets/d");
        let value = interpolate(
            "${cache}/all_vectors.fvecs",
            &defaults(),
            ws,
            PathBase::WorkspaceRelative,
        )
        .unwrap();
        assert_eq!(value, ".cache/all_vectors.fvecs");
        // What a command does with the option value it receives.
        let resolved = ws.join(&value);
        assert_eq!(
            resolved,
            Path::new("datasets/d/.cache/all_vectors.fvecs"),
            "re-resolution must not double the workspace prefix"
        );
    }

    #[test]
    fn test_env_var() {
        // SAFETY: test-only, no other threads depend on this variable
        unsafe { std::env::set_var("VECS_TEST_VAR", "hello") };
        let result =
            interpolate("${env:VECS_TEST_VAR}", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(result, "hello");
        unsafe { std::env::remove_var("VECS_TEST_VAR") };
    }

    #[test]
    fn test_env_var_with_fallback() {
        let result = interpolate(
            "${env:VECS_NONEXISTENT:-fallback}",
            &defaults(),
            Path::new("/data"),
            PathBase::WorkspaceRelative,
        )
        .unwrap();
        assert_eq!(result, "fallback");
    }

    #[test]
    fn test_cache_implicit() {
        // Workspace-relative, and identical whatever the workspace is:
        // commands re-base path options onto the workspace themselves, so
        // expanding this workspace-joined would double the prefix.
        let result =
            interpolate("${cache}/cached.fvec", &defaults(), Path::new("/my/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(result, ".cache/cached.fvec");
        let relative =
            interpolate("${cache}/cached.fvec", &defaults(), Path::new("datasets/d"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(relative, ".cache/cached.fvec");
    }

    #[test]
    fn test_workspace_implicit_is_relative_self() {
        for ws in ["/my/data", "datasets/d", "."] {
            let result =
                interpolate("${workspace}/vectors.fvec", &defaults(), Path::new(ws), PathBase::WorkspaceRelative).unwrap();
            assert_eq!(result, "./vectors.fvec", "workspace {ws}");
            let alias =
                interpolate("${dataset_dir}/vectors.fvec", &defaults(), Path::new(ws), PathBase::WorkspaceRelative).unwrap();
            assert_eq!(alias, "./vectors.fvec", "dataset_dir {ws}");
        }
    }

    #[test]
    fn test_no_vars() {
        let result = interpolate("plain text", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(result, "plain text");
    }

    #[test]
    fn test_empty_string() {
        let result = interpolate("", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
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
            PathBase::WorkspaceRelative,
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
            PathBase::WorkspaceRelative,
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
            PathBase::WorkspaceRelative,
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
            PathBase::WorkspaceRelative,
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

        let resolved = interpolate_options(&opts, &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(resolved.get("seed").unwrap(), "42");
        assert_eq!(resolved.get("count").unwrap(), "100");
    }

    #[test]
    fn test_dollar_escape() {
        let result = interpolate("url_$${number}.npy", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(result, "url_${number}.npy");
    }

    #[test]
    fn test_dollar_escape_with_vars() {
        let result = interpolate(
            "https://host/$${number}.npy?seed=${seed}",
            &defaults(),
            Path::new("/data"),
            PathBase::WorkspaceRelative,
        )
        .unwrap();
        assert_eq!(result, "https://host/${number}.npy?seed=42");
    }

    #[test]
    fn test_dollar_escape_standalone() {
        let result = interpolate("cost: $$5", &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
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

        let resolved = interpolate_options(&opts, &defaults(), Path::new("/data"), PathBase::WorkspaceRelative).unwrap();
        assert_eq!(
            resolved.get("baseurl").unwrap(),
            "https://host/img_${number}.npy"
        );
    }
}
