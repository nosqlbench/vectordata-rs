// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Shared, per-command option definitions with a project-wide consistency
//! guarantee.
//!
//! This replaces the old global value-provider map. The rules it enforces:
//!
//! * **Per-command attachment.** Nothing is injected globally. An option only
//!   exists on a command that attaches it, and its value completer is attached
//!   to that command's node — never to commands that don't declare it.
//! * **DRY definition, one parse-shape per name.** The *parse* definition of an
//!   option ([`OptionDef`] — flag spelling, arity, value-takes-ness, value
//!   name) is shared. Many commands may attach the same named option, but for a
//!   given name it must be *physically the same definition*. Attaching a
//!   conflicting definition for an already-defined name is a **runtime-verifiable
//!   error** ([`OptionConflict`]). This makes "two commands spell `--at`
//!   differently" impossible.
//! * **Per-command value resolvers.** The *value* side (what counts as a valid
//!   value / what to complete) may legitimately differ between commands that
//!   share one parse definition — `datasets ping --at` and
//!   `datasets precache --at` parse `--at` identically but might resolve
//!   different catalog sets. So the resolver is a per-attachment callback, not
//!   part of the shared definition, and is not subject to the consistency
//!   check.
//!
//! A caller supplies both halves through one [`CommandOption`] trait value: the
//! DRY [`OptionDef`] plus the callbacks for this attachment point.

use std::collections::HashMap;

use crate::ValueProvider;

/// The parse-defining shape of an option — everything that determines *how the
/// option is parsed*, and nothing about *what its values are*.
///
/// Equality is structural: two attachments of the same option name must
/// produce equal `OptionDef`s, or the registry rejects them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionDef {
    /// Long flag, including leading dashes, e.g. `"--at"`. This is the unique
    /// key for the option across the whole project.
    pub name: String,
    /// Optional short flag, e.g. `'a'`.
    pub short: Option<char>,
    /// Whether the option takes a value (`--at <X>`) vs. a boolean flag.
    pub takes_value: bool,
    /// Whether the option may be repeated.
    pub multiple: bool,
    /// Value placeholder shown in help (e.g. `"URL"`), when `takes_value`.
    pub value_name: Option<String>,
    /// Help text. Part of the shared definition so the same option reads the
    /// same everywhere.
    pub help: Option<String>,
}

impl OptionDef {
    /// A value-taking option with the given long name.
    pub fn value(name: impl Into<String>) -> Self {
        OptionDef {
            name: name.into(),
            short: None,
            takes_value: true,
            multiple: false,
            value_name: None,
            help: None,
        }
    }

    /// A boolean flag with the given long name.
    pub fn flag(name: impl Into<String>) -> Self {
        OptionDef {
            name: name.into(),
            short: None,
            takes_value: false,
            multiple: false,
            value_name: None,
            help: None,
        }
    }

    pub fn short(mut self, c: char) -> Self {
        self.short = Some(c);
        self
    }
    pub fn multiple(mut self, yes: bool) -> Self {
        self.multiple = yes;
        self
    }
    pub fn value_name(mut self, n: impl Into<String>) -> Self {
        self.value_name = Some(n.into());
        self
    }
    pub fn help(mut self, h: impl Into<String>) -> Self {
        self.help = Some(h.into());
        self
    }

    /// The parse-distinguishing signature: the properties that actually change
    /// *how the option is parsed* — value-taking, repeatable, and the short
    /// spelling. Display-only fields (`value_name`, `help`) are excluded, so a
    /// flag that merely documents itself differently across commands is not a
    /// parse inconsistency.
    pub fn parse_sig(&self) -> (bool, bool, Option<char>) {
        (self.takes_value, self.multiple, self.short)
    }
}

/// A detected violation of "one parse-definition per option name": the same
/// option name parses differently in different commands (or disagrees with the
/// registered definition).
#[derive(Clone, Debug)]
pub struct ParseMismatch {
    pub name: String,
    /// Every place the name was observed, with its (differing) definition.
    pub occurrences: Vec<(String, OptionDef)>,
}

impl std::fmt::Display for ParseMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "option '{}' parses inconsistently across commands (must be one parse-definition):",
            self.name
        )?;
        for (path, def) in &self.occurrences {
            let (takes_value, multiple, short) = def.parse_sig();
            writeln!(
                f,
                "  {:<28} takes_value={takes_value} multiple={multiple} short={short:?}",
                if path.is_empty() { "<root>" } else { path }
            )?;
        }
        Ok(())
    }
}

/// One attachment of a shared option to a command: the DRY parse definition
/// plus this command's value resolver.
///
/// The same option name may be attached by many commands; they must all return
/// an equal [`OptionDef`] from [`definition`](CommandOption::definition), but
/// each may return its own [`value_resolver`](CommandOption::value_resolver).
pub trait CommandOption {
    /// The shared, parse-defining shape. Must be identical for a given option
    /// name across every command that attaches it.
    fn definition(&self) -> OptionDef;

    /// This command's value completer. `None` means no value completion
    /// (a boolean flag, or a value with no closed/known set). May differ
    /// between commands that share the same `definition()`.
    fn value_resolver(&self) -> Option<ValueProvider> {
        None
    }
}

/// Raised when an option name is attached with a definition that disagrees with
/// the one already recorded for that name. Surfacing this at attach time is the
/// runtime-verifiable guarantee that every uniquely-named option parses one way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionConflict {
    pub name: String,
    pub existing: OptionDef,
    pub attempted: OptionDef,
}

impl std::fmt::Display for OptionConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "option '{}' is already defined with a different parse structure; \
             every command attaching '{}' must use one identical definition.\n  \
             existing:  {:?}\n  attempted: {:?}",
            self.name, self.name, self.existing, self.attempted
        )
    }
}

impl std::error::Error for OptionConflict {}

/// The project-wide registry of option *definitions* (the "common structure").
///
/// It records one [`OptionDef`] per option name and rejects any attempt to
/// (re)define that name with a different shape. It does **not** store value
/// resolvers — those are per-command and attach to nodes directly.
#[derive(Default, Debug)]
pub struct OptionRegistry {
    defs: HashMap<String, OptionDef>,
}

impl OptionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `def` for its name, or verify it matches the existing record.
    ///
    /// * first time a name is seen → recorded.
    /// * seen again with an **equal** definition → accepted (shared use).
    /// * seen again with a **different** definition → [`OptionConflict`].
    pub fn define(&mut self, def: &OptionDef) -> Result<(), OptionConflict> {
        match self.defs.get(&def.name) {
            Some(existing) if existing != def => Err(OptionConflict {
                name: def.name.clone(),
                existing: existing.clone(),
                attempted: def.clone(),
            }),
            Some(_) => Ok(()),
            None => {
                self.defs.insert(def.name.clone(), def.clone());
                Ok(())
            }
        }
    }

    /// Validate a [`CommandOption`] attachment and yield its per-command value
    /// resolver. The definition is checked for consistency; the resolver (which
    /// may vary per command) is returned for the caller to attach to that
    /// command's node.
    pub fn attach(
        &mut self,
        opt: &dyn CommandOption,
    ) -> Result<(OptionDef, Option<ValueProvider>), OptionConflict> {
        let def = opt.definition();
        self.define(&def)?;
        Ok((def, opt.value_resolver()))
    }

    /// The canonical definition recorded for an option name, if any.
    pub fn get(&self, name: &str) -> Option<&OptionDef> {
        self.defs.get(name)
    }

    /// Number of distinct option names defined.
    pub fn len(&self) -> usize {
        self.defs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Runtime enforcement of "one parse-definition per option name" against the
    /// **actual parser**.
    ///
    /// `observed` is every `(command_path, OptionDef)` extracted from the real
    /// command/argument tree (e.g. the clap `Command`). For each option name the
    /// audit collects the distinct [`parse_sig`](OptionDef::parse_sig)s seen —
    /// plus the registered canonical definition, if any — and reports a
    /// [`ParseMismatch`] for every name that has more than one. This catches the
    /// case the registry's `define` alone cannot: two *commands* declaring the
    /// same flag with different arities, even though neither went through the
    /// shared definition.
    pub fn audit(&self, observed: &[(String, OptionDef)]) -> Vec<ParseMismatch> {
        let mut by_name: std::collections::BTreeMap<String, Vec<(String, OptionDef)>> =
            std::collections::BTreeMap::new();
        for (path, def) in observed {
            by_name
                .entry(def.name.clone())
                .or_default()
                .push((path.clone(), def.clone()));
        }
        let mut mismatches = Vec::new();
        for (name, mut occs) in by_name {
            let mut sigs: std::collections::HashSet<(bool, bool, Option<char>)> =
                occs.iter().map(|(_, d)| d.parse_sig()).collect();
            // Fold in the registered canonical definition so an observed flag
            // that disagrees with the shared definition is flagged too.
            if let Some(reg) = self.defs.get(&name)
                && sigs.insert(reg.parse_sig()) {
                    occs.push(("<registered>".to_string(), reg.clone()));
                }
            if sigs.len() > 1 {
                occs.sort_by(|a, b| a.0.cmp(&b.0));
                mismatches.push(ParseMismatch { name, occurrences: occs });
            }
        }
        mismatches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two commands sharing one named option, each with its own resolver.
    struct PingAt;
    impl CommandOption for PingAt {
        fn definition(&self) -> OptionDef {
            OptionDef::value("--at").value_name("URL").help("Pin to a catalog location")
        }
        fn value_resolver(&self) -> Option<ValueProvider> {
            Some(std::sync::Arc::new(|_p: &str, _c: &[&str]| vec!["ping-catalog".to_string()]))
        }
    }
    struct PrecacheAt;
    impl CommandOption for PrecacheAt {
        // Same parse definition as PingAt …
        fn definition(&self) -> OptionDef {
            OptionDef::value("--at").value_name("URL").help("Pin to a catalog location")
        }
        // … but a different value resolver. This is allowed.
        fn value_resolver(&self) -> Option<ValueProvider> {
            Some(std::sync::Arc::new(|_p: &str, _c: &[&str]| vec!["precache-catalog".to_string()]))
        }
    }
    // Same name, DIFFERENT parse definition — must be rejected.
    struct BadAt;
    impl CommandOption for BadAt {
        fn definition(&self) -> OptionDef {
            OptionDef::flag("--at") // boolean, not value-taking → inconsistent
        }
    }

    #[test]
    fn same_name_same_definition_attaches_from_multiple_commands() {
        let mut reg = OptionRegistry::new();
        let (d1, r1) = reg.attach(&PingAt).expect("first attach ok");
        let (d2, r2) = reg.attach(&PrecacheAt).expect("second attach (shared def) ok");
        assert_eq!(d1, d2, "shared definition");
        assert_eq!(reg.len(), 1, "one definition recorded for the shared name");
        // Resolvers differ per command — both present, distinct results.
        assert_eq!(r1.unwrap()("", &[]), vec!["ping-catalog"]);
        assert_eq!(r2.unwrap()("", &[]), vec!["precache-catalog"]);
    }

    #[test]
    fn same_name_conflicting_definition_is_a_runtime_error() {
        let mut reg = OptionRegistry::new();
        reg.attach(&PingAt).expect("first attach ok");
        // (The Ok type carries a non-Debug `ValueProvider`, so match rather
        // than `expect_err`.)
        let err = match reg.attach(&BadAt) {
            Err(e) => e,
            Ok(_) => panic!("conflicting --at must error"),
        };
        assert_eq!(err.name, "--at");
        assert!(err.to_string().contains("already defined with a different parse structure"));
    }

    #[test]
    fn distinct_names_coexist() {
        let mut reg = OptionRegistry::new();
        reg.define(&OptionDef::value("--at")).unwrap();
        reg.define(&OptionDef::value("--dataset")).unwrap();
        reg.define(&OptionDef::flag("--recursive")).unwrap();
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn audit_flags_same_name_with_different_arity_across_commands() {
        let reg = OptionRegistry::new();
        // `--at` taken as a repeatable value in one command, a single value in
        // another — the exact "different clap arities" the registry alone can't
        // catch.
        let observed = vec![
            ("datasets ping".to_string(), OptionDef::value("--at").multiple(true)),
            ("datasets precache".to_string(), OptionDef::value("--at").multiple(true)),
            ("config catalog remove".to_string(), OptionDef::value("--at")),
        ];
        let mismatches = reg.audit(&observed);
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].name, "--at");
        assert_eq!(mismatches[0].occurrences.len(), 3);
    }

    #[test]
    fn audit_passes_when_every_occurrence_parses_identically() {
        let mut reg = OptionRegistry::new();
        reg.define(&OptionDef::value("--at").multiple(true)).unwrap();
        let observed = vec![
            ("datasets ping".to_string(), OptionDef::value("--at").multiple(true)),
            ("datasets precache".to_string(), OptionDef::value("--at").multiple(true)),
        ];
        assert!(reg.audit(&observed).is_empty());
    }
}
