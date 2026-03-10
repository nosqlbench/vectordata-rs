// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline variable store: `variables.yaml`.
//!
//! A simple key-value YAML file that sits next to `dataset.yaml`. Values are
//! loaded into the interpolation defaults map and can be referenced from step
//! options via `${name}` — the same syntax used for `upstream.defaults` and
//! implicit variables.
//!
//! Variables can be set by:
//! - The `set variable` command (expressions like `count:file.mvec`)
//! - Direct editing of `variables.yaml`
//!
//! Load priority (last wins):
//!   1. `upstream.defaults` in `dataset.yaml`
//!   2. `variables.yaml`
//!   3. CLI `--set key=value` overrides

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

/// Path to the variables file relative to a dataset directory.
pub fn variables_path(dataset_dir: &Path) -> PathBuf {
    dataset_dir.join("variables.yaml")
}

/// Load variables from `variables.yaml` if it exists.
///
/// Returns an empty map if the file does not exist.
pub fn load(dataset_dir: &Path) -> Result<IndexMap<String, String>, String> {
    let path = variables_path(dataset_dir);
    if !path.exists() {
        return Ok(IndexMap::new());
    }
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let map: BTreeMap<String, serde_yaml::Value> = serde_yaml::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {}", path.display(), e))?;
    let mut result = IndexMap::new();
    for (k, v) in map {
        let s = match v {
            serde_yaml::Value::String(s) => s,
            serde_yaml::Value::Number(n) => n.to_string(),
            serde_yaml::Value::Bool(b) => b.to_string(),
            serde_yaml::Value::Null => continue,
            other => format!("{:?}", other),
        };
        result.insert(k, s);
    }
    Ok(result)
}

/// Save variables to `variables.yaml`.
///
/// Uses a sorted BTreeMap for deterministic output.
pub fn save(dataset_dir: &Path, vars: &IndexMap<String, String>) -> Result<(), String> {
    let path = variables_path(dataset_dir);
    let sorted: BTreeMap<&str, &str> = vars.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let content = serde_yaml::to_string(&sorted)
        .map_err(|e| format!("failed to serialize variables: {}", e))?;
    let tmp_path = path.with_extension("yaml.tmp");
    std::fs::write(&tmp_path, &content)
        .map_err(|e| format!("failed to write {}: {}", tmp_path.display(), e))?;
    std::fs::rename(&tmp_path, &path)
        .map_err(|e| format!("failed to rename variables file: {}", e))?;
    Ok(())
}

/// Set a single variable and persist.
pub fn set_and_save(dataset_dir: &Path, name: &str, value: &str) -> Result<(), String> {
    let mut vars = load(dataset_dir)?;
    vars.insert(name.to_string(), value.to_string());
    save(dataset_dir, &vars)
}

/// Evaluate an expression to a string value.
///
/// Supported expressions:
/// - `count:<path>` — record count of a vector file (fvec/mvec/ivec) or slab
/// - `dim:<path>` — dimension of a vector file
/// - `<literal>` — used as-is
pub fn evaluate_expr(expr: &str, workspace: &Path) -> Result<String, String> {
    if let Some(path_str) = expr.strip_prefix("count:") {
        let path = resolve_path(path_str, workspace);
        count_file(&path)
    } else if let Some(path_str) = expr.strip_prefix("dim:") {
        let path = resolve_path(path_str, workspace);
        dim_file(&path)
    } else {
        Ok(expr.to_string())
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn count_file(path: &Path) -> Result<String, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "fvec" => {
            let reader = MmapVectorReader::<f32>::open_fvec(path)
                .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
            Ok((<MmapVectorReader<f32> as VectorReader<f32>>::count(&reader)).to_string())
        }
        "mvec" => {
            let reader = MmapVectorReader::<half::f16>::open_mvec(path)
                .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
            Ok((<MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&reader)).to_string())
        }
        "ivec" => {
            let reader = MmapVectorReader::<i32>::open_ivec(path)
                .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
            Ok((<MmapVectorReader<i32> as VectorReader<i32>>::count(&reader)).to_string())
        }
        "slab" => {
            let reader = slabtastic::SlabReader::open(path)
                .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
            Ok(reader.total_records().to_string())
        }
        _ => Err(format!(
            "cannot count records in '{}': unsupported extension '.{}'",
            path.display(), ext
        )),
    }
}

fn dim_file(path: &Path) -> Result<String, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "fvec" => {
            let reader = MmapVectorReader::<f32>::open_fvec(path)
                .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
            Ok((<MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader)).to_string())
        }
        "mvec" => {
            let reader = MmapVectorReader::<half::f16>::open_mvec(path)
                .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
            Ok((<MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&reader)).to_string())
        }
        "ivec" => {
            let reader = MmapVectorReader::<i32>::open_ivec(path)
                .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
            Ok((<MmapVectorReader<i32> as VectorReader<i32>>::dim(&reader)).to_string())
        }
        _ => Err(format!(
            "cannot get dimension for '{}': unsupported extension '.{}'",
            path.display(), ext
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent() {
        let vars = load(Path::new("/nonexistent")).unwrap();
        assert!(vars.is_empty());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut vars = IndexMap::new();
        vars.insert("vector_count".to_string(), "407314954".to_string());
        vars.insert("dim".to_string(), "512".to_string());
        save(tmp.path(), &vars).unwrap();

        let loaded = load(tmp.path()).unwrap();
        assert_eq!(loaded.get("vector_count").unwrap(), "407314954");
        assert_eq!(loaded.get("dim").unwrap(), "512");
    }

    #[test]
    fn test_set_and_save() {
        let tmp = tempfile::tempdir().unwrap();
        set_and_save(tmp.path(), "foo", "bar").unwrap();
        set_and_save(tmp.path(), "baz", "123").unwrap();

        let loaded = load(tmp.path()).unwrap();
        assert_eq!(loaded.get("foo").unwrap(), "bar");
        assert_eq!(loaded.get("baz").unwrap(), "123");
    }

    #[test]
    fn test_evaluate_literal() {
        let val = evaluate_expr("42", Path::new("/tmp")).unwrap();
        assert_eq!(val, "42");
    }

    #[test]
    fn test_evaluate_count_mvec() {
        use crate::pipeline::command::{Options, StreamContext};
        use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
        use crate::pipeline::command::CommandOp;
        use crate::pipeline::progress::ProgressLog;

        let tmp = tempfile::tempdir().unwrap();
        let mvec_path = tmp.path().join("test.mvec");

        // Generate 50 f16 vectors
        let mut opts = Options::new();
        opts.set("output", mvec_path.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("count", "50");
        opts.set("seed", "42");
        opts.set("type", "f16");

        let mut ctx = StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: tmp.path().to_path_buf(),
            scratch: tmp.path().join(".scratch"),
            cache: tmp.path().join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        };

        let mut op = GenerateVectorsOp;
        op.execute(&opts, &mut ctx);

        let count = evaluate_expr(
            &format!("count:{}", mvec_path.display()),
            tmp.path(),
        ).unwrap();
        assert_eq!(count, "50");

        let dim = evaluate_expr(
            &format!("dim:{}", mvec_path.display()),
            tmp.path(),
        ).unwrap();
        assert_eq!(dim, "8");
    }
}
