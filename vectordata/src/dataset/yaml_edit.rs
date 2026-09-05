// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Textual edits of an author's `dataset.yaml`.
//!
//! Tags, a tag schema, and a format version are written into the file
//! line by line, preserving every comment and every other line: a
//! serializer round trip would drop what the author wrote around the
//! declarations (PS-19). Every edit is idempotent, so a run that
//! writes what is already there leaves the file byte-identical.

use indexmap::IndexMap;
use serde_yaml::Value as Yaml;

/// The YAML text of a tag value, as the reader will read it back.
///
/// A word is bare; a string that YAML would read as something else —
/// a number such as `1e-2`, a boolean, a null, or anything with
/// punctuation YAML gives meaning to — is single-quoted. A list is a
/// flow sequence. A map is refused: a selector cannot compare one
/// (PS-15), so a tag never holds one.
pub fn render_scalar(v: &Yaml) -> Result<String, String> {
    match v {
        Yaml::Null => Ok("~".to_string()),
        Yaml::Bool(b) => Ok(b.to_string()),
        Yaml::Number(n) => Ok(n.to_string()),
        Yaml::String(s) => Ok(if is_bare_word(s) {
            s.clone()
        } else {
            format!("'{}'", s.replace('\'', "''"))
        }),
        Yaml::Sequence(items) => {
            let parts = items
                .iter()
                .map(render_scalar)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("[{}]", parts.join(", ")))
        }
        Yaml::Mapping(_) => Err("a tag value cannot be a map".to_string()),
        Yaml::Tagged(t) => render_scalar(&t.value),
    }
}

/// A string YAML reads back as the same string when written bare:
/// made of name characters only, and not something YAML reads as a
/// number, a boolean, or a null — `10m` is bare, `1e-2` is not.
fn is_bare_word(s: &str) -> bool {
    if s.is_empty()
        || !s
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return false;
    }
    matches!(serde_yaml::from_str::<Yaml>(s), Ok(Yaml::String(back)) if back == s)
}

fn indent_of(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count()
}

fn is_blank_or_comment(line: &str) -> bool {
    let t = line.trim_start();
    t.is_empty() || t.starts_with('#')
}

/// The key a `key:` line declares, when the line is one.
fn key_of(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let (key, _) = t.split_once(':')?;
    if key.is_empty() || key.contains(' ') || key.starts_with('#') || key.starts_with('-') {
        return None;
    }
    Some(key)
}

/// What follows `key:` on a line, and the trailing comment, if any.
fn value_and_comment(line: &str) -> (String, String) {
    let (value, _, comment) = value_gap_comment(line);
    (value, comment)
}

/// What follows `key:` on a line: the value, the whitespace before a
/// trailing comment, and the comment itself — so an updated value can
/// keep the author's alignment.
fn value_gap_comment(line: &str) -> (String, String, String) {
    let t = line.trim_start();
    let rest = t.split_once(':').map(|(_, r)| r).unwrap_or("");
    let rest = rest.trim_start();
    // A comment starts at a `#` preceded by whitespace and outside quotes.
    let mut in_single = false;
    let mut in_double = false;
    let mut prev = ' ';
    for (i, c) in rest.char_indices() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double && prev.is_whitespace() => {
                let value = rest[..i].trim_end();
                return (
                    value.to_string(),
                    rest[value.len()..i].to_string(),
                    rest[i..].to_string(),
                );
            }
            _ => {}
        }
        prev = c;
    }
    (rest.trim_end().to_string(), String::new(), String::new())
}

/// `<indent><key>: <value>` with the line's own gap and comment kept;
/// the line itself when the value is already what it should be.
fn updated_line(line: &str, indent: &str, key: &str, rendered: &str) -> String {
    let (value, gap, comment) = value_gap_comment(line);
    if value == rendered {
        return line.to_string();
    }
    if comment.is_empty() {
        format!("{indent}{key}: {rendered}")
    } else {
        format!("{indent}{key}: {rendered}{gap}{comment}")
    }
}

/// State a format version (V-25): the existing line is replaced, or
/// the line is inserted after the file's leading comment block.
pub fn set_format_version(yaml: &str, version: u32) -> String {
    let mut lines: Vec<String> = yaml.lines().map(|l| l.to_string()).collect();
    let new_line = format!("format_version: {version}");
    if let Some(i) = lines
        .iter()
        .position(|l| indent_of(l) == 0 && key_of(l) == Some("format_version"))
    {
        lines[i] = updated_line(&lines[i], "", "format_version", &version.to_string());
    } else {
        let at = lines.iter().position(|l| !is_blank_or_comment(l)).unwrap_or(lines.len());
        lines.insert(at, new_line);
    }
    finish(lines, yaml)
}

/// Declare the tag schema (PS-19): the top-level `profile_tags:`
/// block, replaced in place or inserted before `profiles:`.
pub fn set_profile_tags_schema(
    yaml: &str,
    schema: &IndexMap<String, Yaml>,
) -> Result<String, String> {
    let mut lines: Vec<String> = yaml.lines().map(|l| l.to_string()).collect();
    let mut block: Vec<String> = vec!["profile_tags:".to_string()];
    for (k, v) in schema {
        block.push(format!("  {k}: {}", render_scalar(v)?));
    }
    if let Some(start) = lines
        .iter()
        .position(|l| indent_of(l) == 0 && key_of(l) == Some("profile_tags"))
    {
        let end = block_end(&lines, start, 0);
        lines.splice(start..end, block);
    } else {
        let profiles = lines
            .iter()
            .position(|l| indent_of(l) == 0 && key_of(l) == Some("profiles"))
            .ok_or_else(|| "dataset.yaml declares no `profiles:`".to_string())?;
        let mut insert = block;
        insert.push(String::new());
        lines.splice(profiles..profiles, insert);
    }
    Ok(finish(lines, yaml))
}

/// Write tags onto one profile's own `attributes:` (PS-19): existing
/// keys are updated in place, missing keys appended, and a profile
/// without the map gains one right under its name. Comments on every
/// line survive; a map value is refused.
pub fn set_profile_attributes(
    yaml: &str,
    profile: &str,
    attrs: &[(String, Yaml)],
) -> Result<String, String> {
    let mut lines: Vec<String> = yaml.lines().map(|l| l.to_string()).collect();
    let profiles = lines
        .iter()
        .position(|l| indent_of(l) == 0 && key_of(l) == Some("profiles"))
        .ok_or_else(|| "dataset.yaml declares no `profiles:`".to_string())?;
    let profiles_end = block_end(&lines, profiles, 0);
    let profile_line = (profiles + 1..profiles_end)
        .find(|&i| indent_of(&lines[i]) == 2 && key_of(&lines[i]) == Some(profile))
        .ok_or_else(|| format!("profile '{profile}' is not declared"))?;
    if !value_and_comment(&lines[profile_line]).0.is_empty() {
        return Err(format!(
            "profile '{profile}' is declared in flow form; write it as a block to tag it"
        ));
    }
    let profile_end = block_end(&lines, profile_line, 2);
    let attributes_line = (profile_line + 1..profile_end)
        .find(|&i| indent_of(&lines[i]) == 4 && key_of(&lines[i]) == Some("attributes"));

    match attributes_line {
        Some(a) => {
            let (flow, comment) = value_and_comment(&lines[a]);
            if flow.starts_with('{') {
                // Flow form: re-render the one line with the keys updated.
                let mut map: IndexMap<String, Yaml> = serde_yaml::from_str(&flow)
                    .map_err(|e| format!("profile '{profile}' attributes: {e}"))?;
                for (k, v) in attrs {
                    map.insert(k.clone(), v.clone());
                }
                let mut parts = Vec::new();
                for (k, v) in &map {
                    parts.push(format!("{k}: {}", render_scalar(v)?));
                }
                let rendered = if parts.is_empty() {
                    "{}".to_string()
                } else {
                    format!("{{ {} }}", parts.join(", "))
                };
                let _ = comment;
                lines[a] = updated_line(&lines[a], "    ", "attributes", &rendered);
            } else if !flow.is_empty() {
                return Err(format!(
                    "profile '{profile}' attributes is a scalar, not a map"
                ));
            } else {
                // Block form: children at indent 6.
                let block_end_i = block_end(&lines, a, 4);
                let mut last_child = a;
                let mut pending: Vec<(String, Yaml)> = Vec::new();
                for (k, v) in attrs {
                    let existing = (a + 1..block_end_i).find(|&i| {
                        indent_of(&lines[i]) == 6 && key_of(&lines[i]) == Some(k.as_str())
                    });
                    match existing {
                        Some(i) => {
                            lines[i] = updated_line(&lines[i], "      ", k, &render_scalar(v)?);
                        }
                        None => pending.push((k.clone(), v.clone())),
                    }
                }
                for i in (a + 1..block_end_i).rev() {
                    if !is_blank_or_comment(&lines[i]) {
                        last_child = i;
                        break;
                    }
                }
                let mut insert = Vec::new();
                for (k, v) in &pending {
                    insert.push(format!("      {k}: {}", render_scalar(v)?));
                }
                lines.splice(last_child + 1..last_child + 1, insert);
            }
        }
        None => {
            let mut insert = vec!["    attributes:".to_string()];
            for (k, v) in attrs {
                insert.push(format!("      {k}: {}", render_scalar(v)?));
            }
            lines.splice(profile_line + 1..profile_line + 1, insert);
        }
    }
    Ok(finish(lines, yaml))
}

/// Name a profile's parent (PL-12): `inherits: <parent>` written under
/// the profile's name when no `inherits:` line is present; a line
/// already there, whatever it names, is left alone.
pub fn set_profile_inherits(yaml: &str, profile: &str, parent: &str) -> Result<String, String> {
    let mut lines: Vec<String> = yaml.lines().map(|l| l.to_string()).collect();
    let profiles = lines
        .iter()
        .position(|l| indent_of(l) == 0 && key_of(l) == Some("profiles"))
        .ok_or_else(|| "dataset.yaml declares no `profiles:`".to_string())?;
    let profiles_end = block_end(&lines, profiles, 0);
    let profile_line = (profiles + 1..profiles_end)
        .find(|&i| indent_of(&lines[i]) == 2 && key_of(&lines[i]) == Some(profile))
        .ok_or_else(|| format!("profile '{profile}' is not declared"))?;
    if !value_and_comment(&lines[profile_line]).0.is_empty() {
        return Err(format!(
            "profile '{profile}' is declared in flow form; write it as a block to name its parent"
        ));
    }
    let profile_end = block_end(&lines, profile_line, 2);
    let present = (profile_line + 1..profile_end)
        .any(|i| indent_of(&lines[i]) == 4 && key_of(&lines[i]) == Some("inherits"));
    if !present {
        let rendered = render_scalar(&Yaml::from(parent))?;
        lines.insert(profile_line + 1, format!("    inherits: {rendered}"));
    }
    Ok(finish(lines, yaml))
}

/// The line after the last line of the block that starts at `start`,
/// whose children are indented deeper than `indent`. Blank and comment
/// lines inside the block belong to it; trailing ones do not.
fn block_end(lines: &[String], start: usize, indent: usize) -> usize {
    let mut end = start + 1;
    let mut last_content = start;
    while end < lines.len() {
        let l = &lines[end];
        if is_blank_or_comment(l) {
            end += 1;
            continue;
        }
        if indent_of(l) <= indent {
            break;
        }
        last_content = end;
        end += 1;
    }
    last_content + 1
}

fn finish(lines: Vec<String>, original: &str) -> String {
    let mut out = lines.join("\n");
    if original.ends_with('\n') || original.is_empty() {
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = "# an author's header\nformat_version: 2\nname: t\n\nprofiles:\n  default:\n    base_vectors: base.fvec  # the base\n  10m:\n    base_count: 10000000\n    attributes:\n      family: stratified   # planned\n    neighbor_indices: profiles/10m/gt.ivec\n  20m: # twenty\n    base_count: 20000000\n";

    /// **Tags land on the profile's own lines and nothing else moves**
    /// (PS-19): comments, spacing and every other line survive.
    #[test]
    fn tags_are_written_textually_and_idempotently() {
        let once = set_profile_attributes(
            YAML,
            "10m",
            &[
                ("size".into(), Yaml::from("10m")),
                ("family".into(), Yaml::from("stratified")),
                ("selectivity".into(), Yaml::from(0.01)),
            ],
        )
        .unwrap();
        assert!(once.contains("      family: stratified   # planned\n"), "{once}");
        assert!(once.contains("      family: stratified   # planned\n      size: 10m\n      selectivity: 0.01\n"), "{once}");
        assert!(once.contains("# an author's header\n"));
        assert!(once.contains("    base_vectors: base.fvec  # the base\n"));
        assert!(once.contains("  20m: # twenty\n"));
        let twice = set_profile_attributes(&once, "10m", &[("size".into(), Yaml::from("10m"))]).unwrap();
        assert_eq!(once, twice, "writing what is there changes nothing");
        // The result reads back with the tags in place.
        let cfg: crate::dataset::DatasetConfig = serde_yaml::from_str(&once).unwrap();
        let p = cfg.profiles.profile("10m").unwrap();
        assert_eq!(p.attributes["size"], Yaml::from("10m"));
        assert_eq!(p.attributes["selectivity"], Yaml::from(0.01));
    }

    /// A profile without the map gains one under its name; a flow-form
    /// map is re-rendered on its line; a map value is refused.
    #[test]
    fn a_missing_map_is_added_and_a_flow_map_is_rerendered() {
        let out = set_profile_attributes(YAML, "20m", &[("size".into(), Yaml::from("20m"))]).unwrap();
        assert!(out.contains("  20m: # twenty\n    attributes:\n      size: 20m\n    base_count: 20000000\n"), "{out}");

        let flow = "profiles:\n  default:\n    attributes: { size: 495m }  # tags\n    base_vectors: b.fvec\n";
        let out = set_profile_attributes(flow, "default", &[("predicates".into(), Yaml::from("mixed"))]).unwrap();
        assert_eq!(out, "profiles:\n  default:\n    attributes: { size: 495m, predicates: mixed }  # tags\n    base_vectors: b.fvec\n");

        let map: Yaml = serde_yaml::from_str("{a: 1}").unwrap();
        assert!(set_profile_attributes(YAML, "10m", &[("k".into(), map)]).is_err());
        assert!(set_profile_attributes(YAML, "nope", &[]).is_err());
    }

    /// **Values are written as the reader reads them**: a word bare, a
    /// number-like string quoted, a list as a flow sequence.
    #[test]
    fn values_render_as_they_read_back() {
        assert_eq!(render_scalar(&Yaml::from("uniform-2")).unwrap(), "uniform-2");
        assert_eq!(render_scalar(&Yaml::from("10m")).unwrap(), "10m");
        assert_eq!(render_scalar(&Yaml::from("495m")).unwrap(), "495m");
        assert_eq!(render_scalar(&Yaml::from("1e-2")).unwrap(), "'1e-2'");
        assert_eq!(render_scalar(&Yaml::from("true")).unwrap(), "'true'");
        assert_eq!(render_scalar(&Yaml::from(true)).unwrap(), "true");
        assert_eq!(render_scalar(&Yaml::from(0.1)).unwrap(), "0.1");
        let list: Yaml = serde_yaml::from_str("[0.1, 0.01]").unwrap();
        assert_eq!(render_scalar(&list).unwrap(), "[0.1, 0.01]");
        for text in ["uniform-2", "'1e-2'", "[0.1, 0.01]"] {
            let back: Yaml = serde_yaml::from_str(text).unwrap();
            assert_eq!(render_scalar(&back).unwrap(), text);
        }
    }

    /// **A parent is named once and never renamed** (PL-12): the line
    /// is inserted under the profile when absent and left alone when
    /// present, whatever it names.
    #[test]
    fn a_parent_is_named_once() {
        let out = set_profile_inherits(YAML, "10m", "default").unwrap();
        assert!(out.contains("  10m:\n    inherits: default\n    base_count: 10000000\n"), "{out}");
        assert_eq!(set_profile_inherits(&out, "10m", "default").unwrap(), out);
        let named = out.replace("    inherits: default\n", "    inherits: 5m\n");
        assert_eq!(set_profile_inherits(&named, "10m", "default").unwrap(), named, "a present line is kept");
        assert!(set_profile_inherits(YAML, "nope", "default").is_err());
    }

    /// The schema goes before `profiles:` once, and the version line is
    /// stated once, replacing what was there.
    #[test]
    fn schema_and_version_are_stated_once() {
        let mut schema = IndexMap::new();
        schema.insert("size".to_string(), Yaml::Null);
        schema.insert("predicates".to_string(), Yaml::Null);
        let out = set_profile_tags_schema(YAML, &schema).unwrap();
        assert!(out.contains("name: t\n\nprofile_tags:\n  size: ~\n  predicates: ~\n\nprofiles:\n"), "{out}");
        let again = set_profile_tags_schema(&out, &schema).unwrap();
        assert_eq!(out, again);
        schema.insert("selectivity".to_string(), Yaml::Null);
        let grown = set_profile_tags_schema(&again, &schema).unwrap();
        assert_eq!(grown.matches("profile_tags:").count(), 1);
        assert!(grown.contains("  selectivity: ~\n\nprofiles:\n"), "{grown}");

        let v3 = set_format_version(&grown, 3);
        assert_eq!(v3.matches("format_version").count(), 1);
        assert!(v3.starts_with("# an author's header\nformat_version: 3\nname: t\n"), "{v3}");
        // Version 3 states every parent (PL-6).
        let v3 = set_profile_inherits(&v3, "10m", "default").unwrap();
        let v3 = set_profile_inherits(&v3, "20m", "default").unwrap();
        let fresh = set_format_version("name: x\nprofiles:\n  default:\n    base_vectors: b\n", 1);
        assert!(fresh.starts_with("format_version: 1\nname: x\n"), "{fresh}");
        let cfg: crate::dataset::DatasetConfig = serde_yaml::from_str(&v3).unwrap();
        assert_eq!(cfg.format_version, 3);
        assert_eq!(cfg.profile_tags.keys().collect::<Vec<_>>(), vec!["size", "predicates", "selectivity"]);
    }
}
