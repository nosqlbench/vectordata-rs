// Copyright (c) nosqlbench contributors
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

use crate::pipeline::element_type::ElementType;

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
    } else if let Some(rest) = expr.strip_prefix("scale:") {
        // scale:<integer>*<float> — multiply an integer by a fraction, truncate to u64.
        // Optionally append :roundN to round to N significant digits.
        // Examples:
        //   scale:369247459*0.5          → 184623729
        //   scale:369247459*0.5:round2   → 180000000
        //   scale:369247459*0.5:round3   → 185000000
        let (arith, round_digits) = if let Some(pos) = rest.rfind(":round") {
            let digits_str = &rest[pos + 6..];
            let digits: u32 = digits_str.parse().unwrap_or(2);
            (&rest[..pos], Some(digits))
        } else {
            (rest, None)
        };
        let parts: Vec<&str> = arith.splitn(2, '*').collect();
        if parts.len() != 2 {
            return Err(format!("scale: expected '<int>*<float>', got '{}'", arith));
        }
        let base: u64 = parts[0].trim().parse()
            .map_err(|_| format!("scale: invalid integer '{}'", parts[0]))?;
        let factor: f64 = parts[1].trim().parse()
            .map_err(|_| format!("scale: invalid float '{}'", parts[1]))?;
        let result = (base as f64 * factor) as u64;
        let result = if let Some(d) = round_digits {
            round_to_sig_digits(result, d)
        } else {
            result
        };
        Ok(result.to_string())
    } else {
        Ok(expr.to_string())
    }
}

/// Round a u64 to the nearest value with `digits` significant digits.
///
/// Picks whichever of round-up or round-down is closer. If the
/// round-down candidate is zero, uses round-up instead (zero doesn't
/// count as a valid rounded value for sizing purposes).
///
/// Examples (2 significant digits):
///   184623729 → 180000000
///   1500      → 1500 (already 2 sig digits)
///   99        → 99
///   5         → 5
///   0         → 0
fn round_to_sig_digits(value: u64, digits: u32) -> u64 {
    if value == 0 { return 0; }

    // Find the magnitude: 10^(num_digits_in_value - sig_digits)
    let num_digits = ((value as f64).log10().floor() as u32) + 1;
    if num_digits <= digits {
        return value; // already has ≤ digits significant digits
    }

    let divisor = 10u64.pow(num_digits - digits);
    let down = (value / divisor) * divisor;
    let up = down + divisor;

    // Don't round to zero
    if down == 0 {
        return up;
    }

    // Pick whichever is closer
    if value - down <= up - value {
        down
    } else {
        up
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn count_file(path: &Path) -> Result<String, String> {
    // Handle HDF5 paths with #dataset selector (e.g., "file.hdf5#train")
    let path_str = path.to_string_lossy();
    if let Some(hash_pos) = path_str.rfind('#') {
        let file_part = &path_str[..hash_pos];
        let file_path = Path::new(file_part);
        let file_ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if file_ext == "hdf5" || file_ext == "h5" {
            // Use the format-aware source reader which handles HDF5 datasets
            let format = veks_core::formats::VecFormat::Hdf5;
            let threads = 1;
            match veks_core::formats::reader::open_source(path, format, threads, None) {
                Ok(source) => {
                    return match source.record_count() {
                        Some(n) => Ok(n.to_string()),
                        None => Err(format!("cannot determine record count for HDF5 dataset '{}'", path.display())),
                    };
                }
                Err(e) => return Err(format!("failed to open HDF5 source {}: {}", path.display(), e)),
            }
        }
    }

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "slab" {
        let reader = slabtastic::SlabReader::open(path)
            .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
        return Ok(reader.total_records().to_string());
    }

    // Handle directories (e.g. npy, parquet) via format-aware source reader
    if path.is_dir() {
        if let Some(format) = veks_core::formats::VecFormat::detect(path) {
            match veks_core::formats::reader::open_source(path, format, 1, None) {
                Ok(source) => {
                    return match source.record_count() {
                        Some(n) => Ok(n.to_string()),
                        None => Err(format!("cannot determine record count for directory '{}'", path.display())),
                    };
                }
                Err(e) => return Err(format!("failed to open source {}: {}", path.display(), e)),
            }
        } else {
            return Err(format!("cannot count records in '{}': unrecognized directory format", path.display()));
        }
    }

    let etype = ElementType::from_path(path)
        .map_err(|_| format!(
            "cannot count records in '{}': unsupported extension '.{}'",
            path.display(), ext
        ))?;
    crate::dispatch_reader!(etype, path, reader => {
        use vectordata::VectorReader;
        Ok(VectorReader::<_>::count(&reader).to_string())
    })
}

fn dim_file(path: &Path) -> Result<String, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let etype = ElementType::from_path(path)
        .map_err(|_| format!(
            "cannot get dimension for '{}': unsupported extension '.{}'",
            path.display(), ext
        ))?;
    crate::dispatch_reader!(etype, path, reader => {
        use vectordata::VectorReader;
        Ok(VectorReader::<_>::dim(&reader).to_string())
    })
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
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

    #[test]
    fn test_round_to_sig_digits() {
        // 2 significant digits
        assert_eq!(round_to_sig_digits(184623729, 2), 180000000);
        assert_eq!(round_to_sig_digits(155000000, 2), 150000000); // tie goes down
        assert_eq!(round_to_sig_digits(149999999, 2), 150000000);
        assert_eq!(round_to_sig_digits(1500, 2), 1500);
        assert_eq!(round_to_sig_digits(99, 2), 99);
        assert_eq!(round_to_sig_digits(5, 2), 5);
        assert_eq!(round_to_sig_digits(0, 2), 0);

        // Don't round to zero
        assert_eq!(round_to_sig_digits(3, 2), 3);
        assert_eq!(round_to_sig_digits(49, 2), 49);

        // 3 significant digits
        assert_eq!(round_to_sig_digits(184623729, 3), 185000000);
        assert_eq!(round_to_sig_digits(1234, 3), 1230);

        // Large round_digits effectively disables rounding
        assert_eq!(round_to_sig_digits(184623729, 10), 184623729);
    }

    #[test]
    fn test_scale_with_round() {
        let val = evaluate_expr("scale:369247459*0.5:round2", Path::new("/tmp")).unwrap();
        let n: u64 = val.parse().unwrap();
        // 369247459 * 0.5 = 184623729, rounded to 2 sig digits = 180000000
        assert_eq!(n, 180000000);

        let val = evaluate_expr("scale:369247459*0.5:round3", Path::new("/tmp")).unwrap();
        let n: u64 = val.parse().unwrap();
        assert_eq!(n, 185000000);

        // No rounding
        let val = evaluate_expr("scale:369247459*0.5", Path::new("/tmp")).unwrap();
        let n: u64 = val.parse().unwrap();
        assert_eq!(n, 184623729);
    }
}
