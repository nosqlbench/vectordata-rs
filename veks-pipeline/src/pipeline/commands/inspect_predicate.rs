// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: inspect predicate ↔ metadata relationships.
//!
//! Given a predicate ordinal, renders the predicate, looks up matching
//! metadata ordinals from the metadata-indices file, and renders each
//! matching metadata record. Supports slab format (MNode/PNode) and
//! scalar formats (`.u8`, `.i8`, `.u16`, `.i16`, `.u32`, `.i32`,
//! `.u64`, `.i64`) as well as ivec metadata-indices.

use std::path::{Path, PathBuf};
use std::time::Instant;

use slabtastic::SlabReader;

use veks_core::formats::anode::ANode;
use veks_core::formats::anode_vernacular::{self, Vernacular};
use veks_core::formats::mnode::MNode;
use veks_core::formats::pnode::PNode;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::commands::compute_filtered_knn::PredicateIndices;

/// Pipeline command: inspect predicate ↔ metadata cross-reference.
pub struct InspectPredicateOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(InspectPredicateOp)
}

impl CommandOp for InspectPredicateOp {
    fn command_path(&self) -> &str {
        "analyze explain-predicates"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Inspect predicate ↔ metadata relationship via metadata-indices".into(),
            body: format!(
                r#"# analyze predicate

Inspect predicate ↔ metadata relationship via metadata-indices.

## Description

For a given predicate ordinal, renders the predicate from the predicates
file, looks up matching metadata ordinals from the metadata-indices file,
and renders each matching metadata record.

This illustrates the cross-reference computed by `compute predicates`:
which metadata records satisfy a given predicate.

Supports both slab format (MNode/PNode binary encoding) and scalar
formats (`.u8`, `.i8`, `.u16`, `.i16`, `.u32`, `.i32`, `.u64`, `.i64`).
Metadata-indices may be slab (packed i32 ordinals) or ivec.

## Options

{}

## How It Works

The command opens the predicates, metadata, and metadata-indices files.
Format is auto-detected from the file extension:

- **Slab** (`.slab`): records are decoded as PNode/MNode and rendered
  in the chosen vernacular (json, yaml, sql, cql, cddl, readout, display).
- **Scalar** (`.u8`, `.i8`, `.u16`, `.i16`, `.u32`, `.i32`, `.u64`, `.i64`):
  records are flat-packed integers at ordinal × element_size. Predicates
  are rendered as `field_0 == <value>`, metadata as `field_0 = <value>`.
- **Metadata-indices**: slab (packed i32) or ivec (variable-length xvec
  records), auto-detected from extension.

## Data Preparation Role

`inspect predicate` is the primary debugging tool for the filtered KNN
pipeline. After `compute predicates` builds the cross-reference between
predicates and metadata records, this command lets you verify the
mapping by picking a specific predicate ordinal and seeing exactly
which metadata records it matches. This is essential for confirming
that predicates were synthesized correctly and that the metadata-indices
mapping accurately reflects the intended filter semantics. When
filtered KNN queries return unexpected results, `inspect predicate` is
typically the first diagnostic step to determine whether the issue is
in predicate synthesis, metadata indexing, or query execution.

## Notes

- The ordinal indexes into the predicates file (and the corresponding
  metadata-indices record at the same ordinal).
- Use `limit` to cap the number of matching metadata records shown.
- Available vernaculars: json, yaml, sql, cql, cddl, readout, display.
  (Vernaculars only apply to slab format; scalar format has a fixed
  rendering.)
"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let predicates_str = match options.require("predicates") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let metadata_str = match options.require("metadata") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let keys_str = match options.require("metadata-indices") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let ordinal: usize = match options.require("ordinal") {
            Ok(s) => match s.parse() {
                Ok(n) => n,
                _ => return error_result(format!("invalid ordinal: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let vernacular_str = options.get("vernacular").unwrap_or("readout");
        let vernacular = match Vernacular::parse(vernacular_str) {
            Some(v) => v,
            None => return error_result(
                format!("unknown vernacular: '{}'. Use json, yaml, sql, cql, cddl, readout, display", vernacular_str),
                start,
            ),
        };
        let limit: usize = options
            .get("limit")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);

        let predicates_path = resolve_path(predicates_str, &ctx.workspace);
        let metadata_path = resolve_path(metadata_str, &ctx.workspace);
        let keys_path = resolve_path(keys_str, &ctx.workspace);

        let pred_ext = file_ext(&predicates_path);
        let meta_ext = file_ext(&metadata_path);

        // Open metadata-indices (slab or ivec)
        let keys_reader = match PredicateIndices::open(&keys_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open metadata-indices: {}", e), start),
        };

        // Read matching ordinals for this predicate
        let matching_ordinals = match keys_reader.get_ordinals(ordinal) {
            Ok(v) => v,
            Err(e) => return error_result(
                format!("read metadata-indices at ordinal {}: {}", ordinal, e),
                start,
            ),
        };

        // Render predicate
        ctx.ui.log(&format!(
            "── predicate [ordinal {}] ──────────────────────────────────",
            ordinal,
        ));

        if pred_ext == "slab" {
            let pred_reader = match SlabReader::open(&predicates_path) {
                Ok(r) => r,
                Err(e) => return error_result(format!("open predicates slab: {}", e), start),
            };
            let pred_bytes = match pred_reader.get(ordinal as i64) {
                Ok(data) => data,
                Err(e) => return error_result(
                    format!("read predicate at ordinal {}: {}", ordinal, e),
                    start,
                ),
            };
            let pnode = match PNode::from_bytes_named(&pred_bytes) {
                Ok(p) => p,
                Err(e) => return error_result(
                    format!("decode predicate at ordinal {}: {}", ordinal, e),
                    start,
                ),
            };
            let rendered = anode_vernacular::render(&ANode::PNode(pnode), vernacular);
            ctx.ui.log(&rendered);
        } else {
            // Scalar format
            match read_scalar_value(&predicates_path, ordinal) {
                Ok(val) => ctx.ui.log(&format!("field_0 == {}", val)),
                Err(e) => return error_result(
                    format!("read predicate at ordinal {}: {}", ordinal, e),
                    start,
                ),
            }
        }

        ctx.ui.log(&format!(
            "\n── matching metadata: {} record{} ──────────────────────────",
            matching_ordinals.len(),
            if matching_ordinals.len() == 1 { "" } else { "s" },
        ));

        // Render matching metadata records
        let show_count = matching_ordinals.len().min(limit);

        if meta_ext == "slab" {
            let meta_reader = match SlabReader::open(&metadata_path) {
                Ok(r) => r,
                Err(e) => return error_result(format!("open metadata slab: {}", e), start),
            };
            for (i, &meta_ord) in matching_ordinals.iter().take(limit).enumerate() {
                let meta_bytes = match meta_reader.get(meta_ord as i64) {
                    Ok(data) => data,
                    Err(e) => {
                        ctx.ui.log(&format!(
                            "  [{}] metadata ordinal {} — read error: {}",
                            i, meta_ord, e,
                        ));
                        continue;
                    }
                };
                let mnode = match MNode::from_bytes(&meta_bytes) {
                    Ok(m) => m,
                    Err(e) => {
                        ctx.ui.log(&format!(
                            "  [{}] metadata ordinal {} — decode error: {}",
                            i, meta_ord, e,
                        ));
                        continue;
                    }
                };
                let rendered = anode_vernacular::render(&ANode::MNode(mnode), vernacular);
                ctx.ui.log(&format!("  [{}] metadata ordinal {}:", i, meta_ord));
                ctx.ui.log(&format!("    {}", rendered.replace('\n', "\n    ")));
            }
        } else {
            // Scalar format
            for (i, &meta_ord) in matching_ordinals.iter().take(limit).enumerate() {
                match read_scalar_value(&metadata_path, meta_ord as usize) {
                    Ok(val) => {
                        ctx.ui.log(&format!("  [{}] metadata ordinal {}: field_0 = {}", i, meta_ord, val));
                    }
                    Err(e) => {
                        ctx.ui.log(&format!(
                            "  [{}] metadata ordinal {} — read error: {}",
                            i, meta_ord, e,
                        ));
                    }
                }
            }
        }

        if matching_ordinals.len() > limit {
            ctx.ui.log(&format!(
                "\n  ... and {} more (use limit= to show more)",
                matching_ordinals.len() - limit,
            ));
        }

        let message = format!(
            "predicate {} → {} matching metadata records (showing {})",
            ordinal, matching_ordinals.len(), show_count,
        );

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("predicates", "Path", true, None, "Predicates file (slab or scalar format)", OptionRole::Input),
            opt("metadata", "Path", true, None, "Metadata file (slab or scalar format)", OptionRole::Input),
            opt("metadata-indices", "Path", true, None, "Predicate-keys file (slab or ivec)", OptionRole::Input),
            opt("ordinal", "int", true, None, "Predicate ordinal to inspect", OptionRole::Config),
            opt("vernacular", "enum", false, Some("readout"), "Rendering format (slab only): json, yaml, sql, cql, cddl, readout, display", OptionRole::Config),
            opt("limit", "int", false, Some("20"), "Max matching metadata records to display", OptionRole::Config),
        ]
    }
}

/// Read a single scalar value at the given ordinal from a flat-packed file.
///
/// The element size is inferred from the file extension. Returns the value
/// as an i128 string for display.
fn read_scalar_value(path: &Path, ordinal: usize) -> Result<String, String> {
    use std::io::{Read, Seek, SeekFrom};

    let ext = file_ext(path);
    let (elem_size, signed) = match ext.as_str() {
        "u8" => (1, false),
        "i8" => (1, true),
        "u16" => (2, false),
        "i16" => (2, true),
        "u32" => (4, false),
        "i32" => (4, true),
        "u64" => (8, false),
        "i64" => (8, true),
        _ => return Err(format!("unsupported scalar extension '.{}'", ext)),
    };

    let offset = ordinal as u64 * elem_size as u64;
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    f.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("seek to ordinal {}: {}", ordinal, e))?;

    let mut buf = [0u8; 8];
    f.read_exact(&mut buf[..elem_size])
        .map_err(|e| format!("read ordinal {}: {}", ordinal, e))?;

    let val = match (elem_size, signed) {
        (1, false) => format!("{}", buf[0]),
        (1, true) => format!("{}", buf[0] as i8),
        (2, false) => format!("{}", u16::from_le_bytes([buf[0], buf[1]])),
        (2, true) => format!("{}", i16::from_le_bytes([buf[0], buf[1]])),
        (4, false) => format!("{}", u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])),
        (4, true) => format!("{}", i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])),
        (8, false) => format!("{}", u64::from_le_bytes(buf)),
        (8, true) => format!("{}", i64::from_le_bytes(buf)),
        _ => unreachable!(),
    };

    Ok(val)
}

/// Extract lowercase file extension.
fn file_ext(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        role,
}
}
