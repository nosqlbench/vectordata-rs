// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: inspect predicate ↔ metadata relationships.
//!
//! Given a predicate ordinal, renders the predicate from `predicates.slab`,
//! looks up matching metadata ordinals from `metadata-indices.slab`, and
//! renders each matching metadata record.  This illustrates the cross-
//! reference computed by `compute predicates`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt};
use slabtastic::SlabReader;

use crate::formats::anode::ANode;
use crate::formats::anode_vernacular::{self, Vernacular};
use crate::formats::mnode::MNode;
use crate::formats::pnode::PNode;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: inspect predicate ↔ metadata cross-reference.
pub struct InspectPredicateOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(InspectPredicateOp)
}

impl CommandOp for InspectPredicateOp {
    fn command_path(&self) -> &str {
        "inspect predicate"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Inspect predicate ↔ metadata relationship via metadata-indices".into(),
            body: format!(
                r#"# inspect predicate

Inspect predicate ↔ metadata relationship via metadata-indices.

## Description

For a given predicate ordinal, renders the predicate from the predicates slab,
looks up matching metadata ordinals from the metadata-indices slab, and renders
each matching metadata record in the specified vernacular.

This illustrates the cross-reference computed by `compute predicates`:
which metadata records satisfy a given predicate.

## Options

{}

## How It Works

The command opens three slab files: the predicates slab, the metadata
slab, and the metadata-indices slab. It reads the predicate record at
the given ordinal, decodes it as a PNode, and renders it in the chosen
vernacular. It then reads the metadata-indices record at the same
ordinal, which contains a packed array of i32 values representing the
metadata ordinals that satisfy this predicate. For each matching
metadata ordinal (up to the display limit), it reads and decodes the
corresponding metadata record as an MNode and renders it.

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

- The ordinal indexes into the predicates slab (and the corresponding
  metadata-indices record at the same ordinal).
- Use `limit` to cap the number of matching metadata records shown.
- Available vernaculars: json, yaml, sql, cql, cddl, readout, display.
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
        let ordinal: i64 = match options.require("ordinal") {
            Ok(s) => match s.parse() {
                Ok(n) => n,
                _ => return error_result(format!("invalid ordinal: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let vernacular_str = options.get("vernacular").unwrap_or("readout");
        let vernacular = match Vernacular::from_str(vernacular_str) {
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

        // Open slabs
        let pred_reader = match SlabReader::open(&predicates_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open predicates: {}", e), start),
        };
        let meta_reader = match SlabReader::open(&metadata_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open metadata: {}", e), start),
        };
        let keys_reader = match SlabReader::open(&keys_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open metadata-indices: {}", e), start),
        };

        // Read predicate at the given ordinal
        let pred_bytes = match pred_reader.get(ordinal) {
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

        // Read metadata-indices record (matching metadata ordinals)
        let keys_bytes = match keys_reader.get(ordinal) {
            Ok(data) => data,
            Err(e) => return error_result(
                format!("read metadata-indices at ordinal {}: {}", ordinal, e),
                start,
            ),
        };
        let matching_ordinals = read_ordinals(&keys_bytes);

        // Render predicate
        let pred_rendered = anode_vernacular::render(&ANode::PNode(pnode), vernacular);

        ctx.ui.log(&format!(
            "── predicate [ordinal {}] ──────────────────────────────────",
            ordinal,
        ));
        ctx.ui.log(&pred_rendered);
        ctx.ui.log(&format!(
            "\n── matching metadata: {} record{} ──────────────────────────",
            matching_ordinals.len(),
            if matching_ordinals.len() == 1 { "" } else { "s" },
        ));

        let show_count = matching_ordinals.len().min(limit);
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
            opt("predicates", "Path", true, None, "Predicates slab file", OptionRole::Input),
            opt("metadata", "Path", true, None, "Metadata slab file", OptionRole::Input),
            opt("metadata-indices", "Path", true, None, "Predicate-keys slab from compute predicates", OptionRole::Input),
            opt("ordinal", "int", true, None, "Predicate ordinal to inspect", OptionRole::Config),
            opt("vernacular", "enum", false, Some("readout"), "Rendering format: json, yaml, sql, cql, cddl, readout, display", OptionRole::Config),
            opt("limit", "int", false, Some("20"), "Max matching metadata records to display", OptionRole::Config),
        ]
    }
}

/// Read a metadata-indices slab record as a Vec of i32 base ordinals.
fn read_ordinals(data: &[u8]) -> Vec<i32> {
    let mut cursor = std::io::Cursor::new(data);
    let mut result = Vec::with_capacity(data.len() / 4);
    while let Ok(v) = cursor.read_i32::<LittleEndian>() {
        result.push(v);
    }
    result
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
