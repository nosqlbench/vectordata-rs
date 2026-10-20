// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The reference tables in `docs/sysref` that mirror a constant in the
//! code are held to it (SRD element access, §13.9): the element table
//! of `01-data-model.md` §1.2 is `ElementType::ALL` with each type's
//! spellings, and the facet table of `02-api.md` §2.8 is
//! `StandardFacet::PREFERRED_ORDER`. A type or facet added to the code
//! without its row, or a row that misspells an extension, fails here
//! rather than misleading a reader.

use std::path::{Path, PathBuf};

use vectordata::dataset::facet::StandardFacet;
use vectordata::{Container, ElementType};

/// The workspace's `docs/sysref`, two levels above this crate.
fn sysref(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/sysref").join(file)
}

/// The rows of the first markdown table after `heading`, each as its
/// trimmed cells, without the header and the rule.
fn table_after(doc: &str, heading: &str) -> Vec<Vec<String>> {
    let start = doc.find(heading).unwrap_or_else(|| panic!("{heading:?} is not in the document"));
    let mut lines = doc[start..].lines().skip(1).skip_while(|l| !l.starts_with('|'));
    let header = lines.next().expect("a table follows the heading");
    let rule = lines.next().expect("the table has a rule");
    assert!(rule.starts_with("|---") || rule.starts_with("|--"), "not a table rule: {rule} (after {header})");
    lines
        .take_while(|l| l.starts_with('|'))
        .map(|l| l.trim_matches('|').split('|').map(|c| c.trim().to_string()).collect())
        .collect()
}

/// `` `.f32vecs` `` as `.f32vecs`; `—` as `None`.
fn ext_cell(cell: &str) -> Option<String> {
    let cell = cell.trim_matches('`');
    (cell != "—").then(|| cell.to_string())
}

/// **§1.2 is `ElementType::ALL`, in order, with each type's size, its
/// canonical spelling per container and its legacy vec spelling.**
#[test]
fn the_element_table_in_the_docs_matches_all() {
    let doc = std::fs::read_to_string(sysref("01-data-model.md")).unwrap();
    let rows = table_after(&doc, "## 1.2 Element Types");
    assert_eq!(rows.len(), ElementType::ALL.len(), "one row per element type");
    for (row, ty) in rows.iter().zip(ElementType::ALL) {
        assert_eq!(row.len(), 6, "a row is type | size | scalar | vec | vvec | legacy: {row:?}");
        assert_eq!(row[0], ty.name(), "rows follow ElementType::ALL");
        assert_eq!(row[1], format!("{} B", ty.byte_width()), "{ty}");
        for (cell, container) in row[2..5].iter().zip([Container::Scalar, Container::Vec, Container::Vvec]) {
            assert_eq!(ext_cell(cell).as_deref(), Some(format!(".{}", ty.extension(container)).as_str()), "{ty} {container:?}");
        }
        let legacy = ty.legacy_extension(Container::Vec).map(|l| format!(".{l}"));
        assert_eq!(ext_cell(&row[5]), legacy, "{ty} legacy");
    }
}

/// **§2.8 is `StandardFacet::PREFERRED_ORDER`, in order, keyed by the
/// canonical YAML key**, and names each facet's code letter where it
/// has one.
#[test]
fn the_facet_table_in_the_docs_matches_preferred_order() {
    let doc = std::fs::read_to_string(sysref("02-api.md")).unwrap();
    let rows = table_after(&doc, "## 2.8 Facet Reference");
    assert_eq!(rows.len(), StandardFacet::PREFERRED_ORDER.len(), "one row per standard facet");
    for (row, facet) in rows.iter().zip(StandardFacet::PREFERRED_ORDER) {
        let key = row[1].split('`').nth(1).unwrap_or_else(|| panic!("no key in {row:?}"));
        assert_eq!(key, facet.key(), "rows follow PREFERRED_ORDER");
        let code = row[0].trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '—');
        match facet.code() {
            Some(c) => assert!(code.starts_with(c), "{key}: code {c} is spelled {:?}", row[0]),
            None => assert_eq!(code, "—", "{key} has no code letter"),
        }
    }
}
