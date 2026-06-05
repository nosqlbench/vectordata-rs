// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The veks-completion CLI framework: argument parsing, help, and a command
//! definition model — the in-tree replacement for clap.
//!
//! This is the runtime half. A command's CLI is described by a [`CommandSpec`]
//! (subcommands, options, positionals); [`parse`] turns an `argv` slice into a
//! [`ParsedArgs`]; [`render_help`] renders `--help`. The derive macro
//! (`veks-completion-derive`) generates the [`CommandSpec`] and the typed
//! extraction from this same model, so one declaration drives parsing, help,
//! **and** completion (the completion [`CommandTree`](crate::CommandTree) is
//! built from the very same [`CommandSpec`]).
//!
//! The option's *parse-defining shape* is the existing [`OptionDef`] — shared
//! and consistency-checked across commands. Per-command facets that may
//! legitimately vary (required-ness, default value) live on [`OptionSpec`],
//! deliberately outside the shape that the consistency audit compares.

use std::collections::{BTreeMap, HashSet};

use crate::OptionDef;

/// One option as it appears on a specific command: the shared [`OptionDef`]
/// shape plus this command's required-ness, default, and value completer.
#[derive(Clone)]
pub struct OptionSpec {
    /// The shared, parse-defining shape (flag, short, arity, value name, help).
    pub def: OptionDef,
    /// Whether this command requires the option. May differ per command.
    pub required: bool,
    /// Default value applied when the option is absent (value options only).
    pub default: Option<String>,
    /// Per-command value completer (closed set, path, dynamic). The completion
    /// bridge attaches it to this command's node. It is *not* part of the parse
    /// shape or the consistency audit, so it may legitimately differ between
    /// commands that share the same `def`.
    pub value_completion: Option<crate::ValueProvider>,
}

impl std::fmt::Debug for OptionSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptionSpec")
            .field("def", &self.def)
            .field("required", &self.required)
            .field("default", &self.default)
            .field("value_completion", &self.value_completion.as_ref().map(|_| "<provider>"))
            .finish()
    }
}

impl OptionSpec {
    pub fn new(def: OptionDef) -> Self {
        OptionSpec { def, required: false, default: None, value_completion: None }
    }
    pub fn required(mut self, yes: bool) -> Self {
        self.required = yes;
        self
    }
    pub fn default(mut self, v: impl Into<String>) -> Self {
        self.default = Some(v.into());
        self
    }
    /// Attach a value completer for this option on this command.
    pub fn value_completion(mut self, provider: crate::ValueProvider) -> Self {
        self.value_completion = Some(provider);
        self
    }
    /// The canonical long token, e.g. `"--at"`.
    pub fn flag(&self) -> &str {
        &self.def.name
    }
}

/// A positional argument.
#[derive(Clone, Debug)]
pub struct PositionalSpec {
    /// Display name, e.g. `"DATASET"`.
    pub name: String,
    pub required: bool,
    /// Greedy trailing positional (collects the rest).
    pub multiple: bool,
    pub help: Option<String>,
}

impl PositionalSpec {
    pub fn new(name: impl Into<String>) -> Self {
        PositionalSpec { name: name.into(), required: false, multiple: false, help: None }
    }
    pub fn required(mut self, yes: bool) -> Self {
        self.required = yes;
        self
    }
    pub fn multiple(mut self, yes: bool) -> Self {
        self.multiple = yes;
        self
    }
    pub fn help(mut self, h: impl Into<String>) -> Self {
        self.help = Some(h.into());
        self
    }
}

/// A command and (recursively) its subcommands. The single source consumed by
/// the parser, the help renderer, and the completion-tree builder.
#[derive(Clone, Debug, Default)]
pub struct CommandSpec {
    pub name: String,
    pub about: Option<String>,
    pub aliases: Vec<String>,
    pub options: Vec<OptionSpec>,
    pub positionals: Vec<PositionalSpec>,
    pub subcommands: Vec<CommandSpec>,
    /// When true, a subcommand must be given (a group command).
    pub subcommand_required: bool,
    /// Free-form text appended after the options block in `--help` (examples,
    /// notes). From `#[command(after_help/after_long_help = …)]`.
    pub after_help: Option<String>,
}

impl CommandSpec {
    pub fn new(name: impl Into<String>) -> Self {
        CommandSpec { name: name.into(), ..Default::default() }
    }
    pub fn about(mut self, a: impl Into<String>) -> Self {
        self.about = Some(a.into());
        self
    }
    /// Add an alternate name this command also answers to (e.g. `ls` for `list`).
    pub fn alias(mut self, a: impl Into<String>) -> Self {
        self.aliases.push(a.into());
        self
    }
    pub fn after_help(mut self, a: impl Into<String>) -> Self {
        self.after_help = Some(a.into());
        self
    }
    pub fn option(mut self, o: OptionSpec) -> Self {
        self.options.push(o);
        self
    }
    pub fn positional(mut self, p: PositionalSpec) -> Self {
        self.positionals.push(p);
        self
    }
    pub fn subcommand(mut self, c: CommandSpec) -> Self {
        self.subcommands.push(c);
        self
    }

    /// Find an option by long token (with or without leading dashes) or short.
    fn find_long(&self, token: &str) -> Option<&OptionSpec> {
        let want = token.trim_start_matches('-');
        self.options.iter().find(|o| o.def.name.trim_start_matches('-') == want)
    }
    fn find_short(&self, c: char) -> Option<&OptionSpec> {
        self.options.iter().find(|o| o.def.short == Some(c))
    }
    fn find_subcommand(&self, name: &str) -> Option<&CommandSpec> {
        self.subcommands
            .iter()
            .find(|s| s.name == name || s.aliases.iter().any(|a| a == name))
    }
}

/// The result of parsing one command level. Values are kept as strings (the
/// derive macro performs typed conversion); repeatable options accumulate.
#[derive(Clone, Debug, Default)]
pub struct ParsedArgs {
    /// Boolean flags that were present (canonical long token, no dashes).
    flags: HashSet<String>,
    /// Value options: canonical long token (no dashes) → values in order.
    values: BTreeMap<String, Vec<String>>,
    /// Positional arguments in order.
    positionals: Vec<String>,
    /// The chosen subcommand and its own parsed args, if any.
    subcommand: Option<(String, Box<ParsedArgs>)>,
}

impl ParsedArgs {
    pub fn has_flag(&self, name: &str) -> bool {
        self.flags.contains(name.trim_start_matches('-'))
    }
    /// First value for an option, if present.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.values.get(name.trim_start_matches('-')).and_then(|v| v.first()).map(|s| s.as_str())
    }
    /// All values for a (repeatable) option.
    pub fn values(&self, name: &str) -> &[String] {
        const EMPTY: &[String] = &[];
        self.values.get(name.trim_start_matches('-')).map(|v| v.as_slice()).unwrap_or(EMPTY)
    }
    pub fn positionals(&self) -> &[String] {
        &self.positionals
    }
    pub fn subcommand(&self) -> Option<(&str, &ParsedArgs)> {
        self.subcommand.as_ref().map(|(n, p)| (n.as_str(), p.as_ref()))
    }
}

/// A parse failure, with enough context to render a useful message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    UnknownFlag { command: String, flag: String },
    MissingValue { command: String, flag: String },
    MissingRequiredOption { command: String, flag: String },
    MissingRequiredPositional { command: String, name: String },
    UnexpectedPositional { command: String, value: String },
    UnknownSubcommand { command: String, name: String },
    MissingSubcommand { command: String },
    /// A value failed to convert to the field's type (e.g. `--count abc` for a
    /// `usize`). Produced during typed extraction by the derive macro.
    InvalidValue { flag: String, value: String, message: String },
}

/// Implemented by `#[derive(VeksCli)]` types: a command/args struct or a
/// subcommand enum. Provides the [`CommandSpec`] (drives parse + help +
/// completion) and the typed extraction from a [`ParsedArgs`].
pub trait VeksCli: Sized {
    /// The full spec for this type used as a command named `name`.
    fn veks_command_spec(name: &str) -> CommandSpec;
    /// Add this type's options/positionals/subcommands to an existing spec
    /// (used by `#[command(flatten)]` and subcommand fields).
    fn veks_augment_spec(spec: CommandSpec) -> CommandSpec;
    /// Build `Self` from already-parsed args.
    fn veks_from_parsed(parsed: &ParsedArgs) -> Result<Self, ParseError>;
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnknownFlag { command, flag } =>
                write!(f, "{command}: unexpected option '{flag}'"),
            ParseError::MissingValue { command, flag } =>
                write!(f, "{command}: option '{flag}' requires a value"),
            ParseError::MissingRequiredOption { command, flag } =>
                write!(f, "{command}: required option '{flag}' not provided"),
            ParseError::MissingRequiredPositional { command, name } =>
                write!(f, "{command}: required argument <{name}> not provided"),
            ParseError::UnexpectedPositional { command, value } =>
                write!(f, "{command}: unexpected argument '{value}'"),
            ParseError::UnknownSubcommand { command, name } =>
                write!(f, "{command}: unknown subcommand '{name}'"),
            ParseError::MissingSubcommand { command } =>
                write!(f, "{command}: a subcommand is required"),
            ParseError::InvalidValue { flag, value, message } =>
                write!(f, "invalid value '{value}' for '{flag}': {message}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse `argv` (the words *after* the program name) against `spec`.
///
/// Grammar handled: `--long`, `--long=value`, `--long value`, `-s`, `-s value`,
/// `-s=value`, `--` (end-of-options), positionals, and nested subcommands.
/// Boolean options take no value; value options consume the next word (or the
/// `=`-suffix). Repeatable options accumulate. Defaults fill absent value
/// options; required options/positionals are checked after parsing.
pub fn parse(spec: &CommandSpec, argv: &[String]) -> Result<ParsedArgs, ParseError> {
    let mut out = ParsedArgs::default();
    let mut i = 0;
    let mut options_ended = false;

    while i < argv.len() {
        let arg = &argv[i];

        if !options_ended && arg == "--" {
            options_ended = true;
            i += 1;
            continue;
        }

        if !options_ended && arg.starts_with("--") {
            // --long or --long=value
            let body = &arg[2..];
            let (name, inline) = match body.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (body, None),
            };
            let opt = spec
                .find_long(name)
                .ok_or_else(|| ParseError::UnknownFlag { command: spec.name.clone(), flag: arg.clone() })?;
            let canon = opt.def.name.trim_start_matches('-').to_string();
            if !opt.def.takes_value {
                out.flags.insert(canon);
            } else {
                let value = match inline {
                    Some(v) => v,
                    None => {
                        i += 1;
                        argv.get(i)
                            .cloned()
                            .ok_or_else(|| ParseError::MissingValue { command: spec.name.clone(), flag: arg.clone() })?
                    }
                };
                out.values.entry(canon).or_default().push(value);
            }
            i += 1;
            continue;
        }

        if !options_ended && arg.starts_with('-') && arg.len() > 1 {
            // -s or -s=value or -sVALUE (single short; bundling not supported)
            let body = &arg[1..];
            let mut chars = body.chars();
            let short = chars.next().unwrap();
            let rest: String = chars.collect();
            let opt = spec
                .find_short(short)
                .ok_or_else(|| ParseError::UnknownFlag { command: spec.name.clone(), flag: arg.clone() })?;
            let canon = opt.def.name.trim_start_matches('-').to_string();
            if !opt.def.takes_value {
                out.flags.insert(canon);
            } else {
                let value = if let Some(stripped) = rest.strip_prefix('=') {
                    stripped.to_string()
                } else if !rest.is_empty() {
                    rest
                } else {
                    i += 1;
                    argv.get(i)
                        .cloned()
                        .ok_or_else(|| ParseError::MissingValue { command: spec.name.clone(), flag: arg.clone() })?
                };
                out.values.entry(canon).or_default().push(value);
            }
            i += 1;
            continue;
        }

        // A bare word. If this command has subcommands and no positional has
        // been consumed yet, treat it as a subcommand selector; otherwise it's
        // a positional.
        if !spec.subcommands.is_empty() && out.positionals.is_empty() {
            let sub = spec.find_subcommand(arg).ok_or_else(|| ParseError::UnknownSubcommand {
                command: spec.name.clone(),
                name: arg.clone(),
            })?;
            let sub_parsed = parse(sub, &argv[i + 1..])?;
            out.subcommand = Some((sub.name.clone(), Box::new(sub_parsed)));
            // A subcommand consumes the remainder.
            finalize(spec, &mut out)?;
            return Ok(out);
        }

        out.positionals.push(arg.clone());
        i += 1;
    }

    finalize(spec, &mut out)?;
    Ok(out)
}

/// Apply defaults, then validate required options/positionals and subcommand
/// presence.
fn finalize(spec: &CommandSpec, out: &mut ParsedArgs) -> Result<(), ParseError> {
    for opt in &spec.options {
        let canon = opt.def.name.trim_start_matches('-').to_string();
        let present = out.flags.contains(&canon) || out.values.contains_key(&canon);
        if !present {
            if let Some(def) = &opt.default {
                out.values.entry(canon.clone()).or_default().push(def.clone());
            } else if opt.required {
                return Err(ParseError::MissingRequiredOption {
                    command: spec.name.clone(),
                    flag: opt.def.name.clone(),
                });
            }
        }
    }

    // Positional arity: count required positionals satisfied.
    let required_positionals = spec.positionals.iter().filter(|p| p.required).count();
    if out.positionals.len() < required_positionals {
        let missing = &spec.positionals[out.positionals.len()];
        return Err(ParseError::MissingRequiredPositional {
            command: spec.name.clone(),
            name: missing.name.clone(),
        });
    }

    if spec.subcommand_required && out.subcommand.is_none() {
        return Err(ParseError::MissingSubcommand { command: spec.name.clone() });
    }

    Ok(())
}

/// Render `--help` text for a command spec.
/// Target line width for help text (clap's default for a non-tty).
const HELP_WIDTH: usize = 100;

/// Word-wrap `text` to `width`, honoring existing newlines as hard breaks.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for para in text.split('\n') {
        if para.trim().is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut cur = String::new();
        for word in para.split_whitespace() {
            if cur.is_empty() {
                cur.push_str(word);
            } else if width == 0 || cur.len() + 1 + word.len() <= width {
                cur.push(' ');
                cur.push_str(word);
            } else {
                lines.push(std::mem::take(&mut cur));
                cur.push_str(word);
            }
        }
        lines.push(cur);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Two-column (term, description) renderer with a capped left column and a
/// wrapped, hanging-indented right column — the shape clap uses for its
/// Commands/Arguments/Options blocks.
fn render_two_col(out: &mut String, rows: &[(String, String)]) {
    if rows.is_empty() {
        return;
    }
    let col = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(0).min(28);
    let help_width = HELP_WIDTH.saturating_sub(col + 4).max(20);
    for (left, help) in rows {
        let wrapped = wrap_text(help, help_width);
        let mut iter = wrapped.iter();
        let first = iter.next().map(|s| s.as_str()).unwrap_or("");
        if left.len() <= col {
            out.push_str(&format!("  {:<col$}  {}\n", left, first, col = col));
        } else {
            // Left entry overflows the column — put it on its own line.
            out.push_str(&format!("  {}\n", left));
            out.push_str(&format!("  {:<col$}  {}\n", "", first, col = col));
        }
        for cont in iter {
            out.push_str(&format!("  {:<col$}  {}\n", "", cont, col = col));
        }
    }
}

/// Render `--help` text for a command spec, formatted comparably to clap:
/// about, usage, aliases, commands, arguments, options (with an auto
/// `-h, --help`), and any `after_help`.
pub fn render_help(spec: &CommandSpec) -> String {
    let mut s = String::new();

    if let Some(about) = &spec.about {
        for line in wrap_text(about, HELP_WIDTH) {
            s.push_str(&line);
            s.push('\n');
        }
        s.push('\n');
    }

    // Usage line.
    s.push_str(&format!("Usage: {}", spec.name));
    if !spec.options.is_empty() {
        s.push_str(" [OPTIONS]");
    }
    for p in &spec.positionals {
        let token = if p.multiple {
            format!("[{}]...", p.name)
        } else if p.required {
            format!("<{}>", p.name)
        } else {
            format!("[{}]", p.name)
        };
        s.push(' ');
        s.push_str(&token);
    }
    if !spec.subcommands.is_empty() {
        s.push_str(" <COMMAND>");
    }
    s.push('\n');

    if !spec.aliases.is_empty() {
        s.push_str(&format!("\nAliases: {}\n", spec.aliases.join(", ")));
    }

    if !spec.subcommands.is_empty() {
        s.push_str("\nCommands:\n");
        let rows: Vec<(String, String)> = spec
            .subcommands
            .iter()
            .map(|c| {
                let name = if c.aliases.is_empty() {
                    c.name.clone()
                } else {
                    format!("{}, {}", c.name, c.aliases.join(", "))
                };
                (name, c.about.clone().unwrap_or_default())
            })
            .collect();
        render_two_col(&mut s, &rows);
    }

    if !spec.positionals.is_empty() {
        s.push_str("\nArguments:\n");
        let rows: Vec<(String, String)> = spec
            .positionals
            .iter()
            .map(|p| (format!("<{}>", p.name), p.help.clone().unwrap_or_default()))
            .collect();
        render_two_col(&mut s, &rows);
    }

    {
        s.push_str("\nOptions:\n");
        let mut rows: Vec<(String, String)> = spec
            .options
            .iter()
            .map(|o| {
                // Align long flags whether or not a short exists ("-x, " is 4 wide).
                let mut f = match o.def.short {
                    Some(sh) => format!("-{}, ", sh),
                    None => "    ".to_string(),
                };
                f.push_str(&o.def.name);
                if o.def.takes_value {
                    f.push_str(&format!(" <{}>", o.def.value_name.as_deref().unwrap_or("VALUE")));
                }
                (f, o.def.help.clone().unwrap_or_default())
            })
            .collect();
        rows.push(("-h, --help".to_string(), "Print help".to_string()));
        render_two_col(&mut s, &rows);
    }

    if let Some(after) = &spec.after_help {
        s.push('\n');
        s.push_str(after.trim_end());
        s.push('\n');
    }

    s
}

// ---------------------------------------------------------------------------
// Completion bridge: CommandSpec -> CommandTree
// ---------------------------------------------------------------------------

/// Build a completion [`CommandTree`](crate::CommandTree) from a
/// [`CommandSpec`] — the same spec that drives parsing and help. This replaces
/// walking a `clap::Command`: one definition now feeds parse + help + complete.
///
/// `resolvers` maps a flag's canonical long token (e.g. `"--at"`) to the value
/// completer to attach **per command** — only flags a command actually declares
/// receive one, so nothing leaks (the same property the option registry gives).
pub fn build_completion_tree(
    spec: &CommandSpec,
    resolvers: &std::collections::BTreeMap<String, crate::ValueProvider>,
) -> crate::CommandTree {
    let mut tree = crate::CommandTree::new(&spec.name);
    tree.root = spec_to_node(spec, resolvers);
    tree
}

fn spec_to_node(
    spec: &CommandSpec,
    resolvers: &std::collections::BTreeMap<String, crate::ValueProvider>,
) -> crate::Node {
    if spec.subcommands.is_empty() {
        let value_flags: Vec<&str> =
            spec.options.iter().filter(|o| o.def.takes_value).map(|o| o.def.name.as_str()).collect();
        let boolean_flags: Vec<&str> =
            spec.options.iter().filter(|o| !o.def.takes_value).map(|o| o.def.name.as_str()).collect();
        let mut node = crate::Node::leaf_with_flags(&value_flags, &boolean_flags);
        for o in &spec.options {
            if let Some(h) = &o.def.help {
                node = node.with_flag_help(&o.def.name, h);
            }
            if o.def.takes_value {
                // The option's own completer wins; otherwise fall back to a
                // shared resolver registered for that flag name.
                let provider = o
                    .value_completion
                    .clone()
                    .or_else(|| resolvers.get(&o.def.name).cloned());
                if let Some(p) = provider {
                    node = node.with_value_provider(&o.def.name, p);
                }
            }
        }
        node
    } else {
        let mut node = crate::Node::empty_group();
        for sub in &spec.subcommands {
            node = node.with_child(&sub.name, spec_to_node(sub, resolvers));
        }
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vopt(name: &str) -> OptionSpec {
        OptionSpec::new(OptionDef::value(name))
    }
    fn fopt(name: &str) -> OptionSpec {
        OptionSpec::new(OptionDef::flag(name))
    }

    fn datasets_ping() -> CommandSpec {
        CommandSpec::new("ping")
            .about("Ping a remote dataset")
            .option(vopt("--at").def_multiple())
            .option(OptionSpec::new(OptionDef::value("--dataset")).required(true))
            .option(OptionSpec::new(OptionDef::value("--profile")).default("default"))
    }

    // small helper so tests can flip the OptionDef.multiple bit inline
    impl OptionSpec {
        fn def_multiple(mut self) -> Self {
            self.def = self.def.multiple(true);
            self
        }
    }

    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_value_space_and_equals_forms() {
        let spec = datasets_ping();
        let p = parse(&spec, &argv(&["--dataset", "sift1m", "--at", "1"])).unwrap();
        assert_eq!(p.value("--dataset"), Some("sift1m"));
        assert_eq!(p.values("--at"), &["1".to_string()]);
        let p2 = parse(&spec, &argv(&["--dataset=sift1m"])).unwrap();
        assert_eq!(p2.value("--dataset"), Some("sift1m"));
    }

    #[test]
    fn repeatable_option_accumulates() {
        let spec = datasets_ping();
        let p = parse(&spec, &argv(&["--dataset", "d", "--at", "1", "--at", "2"])).unwrap();
        assert_eq!(p.values("--at"), &["1".to_string(), "2".to_string()]);
    }

    #[test]
    fn default_applies_when_absent() {
        let spec = datasets_ping();
        let p = parse(&spec, &argv(&["--dataset", "d"])).unwrap();
        assert_eq!(p.value("--profile"), Some("default"));
    }

    #[test]
    fn required_option_missing_errors() {
        let spec = datasets_ping();
        let err = parse(&spec, &argv(&["--at", "1"])).unwrap_err();
        assert_eq!(err, ParseError::MissingRequiredOption { command: "ping".into(), flag: "--dataset".into() });
    }

    #[test]
    fn unknown_flag_errors() {
        let spec = datasets_ping();
        let err = parse(&spec, &argv(&["--dataset", "d", "--nope"])).unwrap_err();
        assert!(matches!(err, ParseError::UnknownFlag { .. }));
    }

    #[test]
    fn boolean_flag_takes_no_value() {
        let spec = CommandSpec::new("list").option(fopt("--verbose"));
        let p = parse(&spec, &argv(&["--verbose"])).unwrap();
        assert!(p.has_flag("--verbose"));
        // The following word is a positional, not the flag's value.
        let p2 = parse(&spec, &argv(&["--verbose", "x"])).unwrap();
        assert_eq!(p2.positionals(), &["x".to_string()]);
    }

    #[test]
    fn double_dash_ends_options() {
        let spec = CommandSpec::new("run").option(fopt("--flag"));
        let p = parse(&spec, &argv(&["--", "--flag"])).unwrap();
        assert!(!p.has_flag("--flag"));
        assert_eq!(p.positionals(), &["--flag".to_string()]);
    }

    #[test]
    fn subcommand_dispatch_and_short_value() {
        let spec = CommandSpec::new("datasets")
            .subcommand(datasets_ping())
            .subcommand(
                CommandSpec::new("derive")
                    .option(OptionSpec::new(OptionDef::value("--output").short('o')).required(true)),
            );
        let p = parse(&spec, &argv(&["derive", "-o", "/tmp/out"])).unwrap();
        let (name, sub) = p.subcommand().unwrap();
        assert_eq!(name, "derive");
        assert_eq!(sub.value("--output"), Some("/tmp/out"));
    }

    #[test]
    fn unknown_subcommand_errors() {
        let spec = CommandSpec::new("datasets").subcommand(datasets_ping());
        let err = parse(&spec, &argv(&["frobnicate"])).unwrap_err();
        assert!(matches!(err, ParseError::UnknownSubcommand { .. }));
    }

    #[test]
    fn completion_tree_built_from_spec() {
        let spec = CommandSpec::new("veks").subcommand(
            CommandSpec::new("datasets")
                .subcommand(
                    CommandSpec::new("ping")
                        .option(vopt("--at").def_multiple())
                        .option(vopt("--dataset")),
                )
                .subcommand(CommandSpec::new("list").option(fopt("--verbose"))),
        );
        let mut resolvers: std::collections::BTreeMap<String, crate::ValueProvider> =
            std::collections::BTreeMap::new();
        resolvers.insert(
            "--at".to_string(),
            crate::fn_provider(|_p, _c| vec!["1".to_string(), "2".to_string()]),
        );
        let tree = build_completion_tree(&spec, &resolvers);

        // Per-command flags: ping has --at/--dataset, list has --verbose only.
        let ping_flags = crate::complete(&tree, &["veks", "datasets", "ping", "--"]);
        assert!(ping_flags.contains(&"--at".to_string()));
        assert!(ping_flags.contains(&"--dataset".to_string()));
        let list_flags = crate::complete(&tree, &["veks", "datasets", "list", "--"]);
        assert!(list_flags.contains(&"--verbose".to_string()));
        assert!(!list_flags.contains(&"--at".to_string()), "--at must not leak onto list");

        // Value completion: the --at resolver fires on ping.
        let at_vals = crate::complete(&tree, &["veks", "datasets", "ping", "--at", ""]);
        assert_eq!(at_vals, vec!["1".to_string(), "2".to_string()]);
    }
}

#[cfg(test)]
mod derive_tests {
    use crate::VeksCli;
    use veks_completion_derive::VeksCli;

    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[derive(VeksCli, Debug, PartialEq)]
    #[command(about = "Ping a remote dataset")]
    struct Ping {
        /// Catalog locations
        #[arg(long = "at")]
        at: Vec<String>,
        #[arg(long)]
        dataset: String,
        #[arg(long, default = "default")]
        profile: String,
        #[arg(long)]
        verbose: bool,
    }

    #[test]
    fn derive_struct_spec_and_extract() {
        let spec = Ping::veks_command_spec("ping");
        // spec carries the right shapes
        assert_eq!(spec.about.as_deref(), Some("Ping a remote dataset"));
        let p = crate::cli::parse(
            &spec,
            &argv(&["--dataset", "sift1m", "--at", "1", "--at", "2", "--verbose"]),
        )
        .unwrap();
        let ping = Ping::veks_from_parsed(&p).unwrap();
        assert_eq!(
            ping,
            Ping {
                at: vec!["1".into(), "2".into()],
                dataset: "sift1m".into(),
                profile: "default".into(),
                verbose: true,
            }
        );
    }

    #[test]
    fn derive_typed_conversion_and_default() {
        #[derive(VeksCli, Debug, PartialEq)]
        struct Run {
            #[arg(long, default = "4")]
            threads: usize,
            #[arg(long)]
            tag: Option<String>,
        }
        let spec = Run::veks_command_spec("run");
        let p = crate::cli::parse(&spec, &argv(&["--threads", "8"])).unwrap();
        let run = Run::veks_from_parsed(&p).unwrap();
        assert_eq!(run, Run { threads: 8, tag: None });
        // default applies
        let p2 = crate::cli::parse(&spec, &argv(&[])).unwrap();
        assert_eq!(Run::veks_from_parsed(&p2).unwrap().threads, 4);
        // bad value → InvalidValue
        let p3 = crate::cli::parse(&spec, &argv(&["--threads", "abc"])).unwrap();
        assert!(matches!(
            Run::veks_from_parsed(&p3),
            Err(crate::cli::ParseError::InvalidValue { .. })
        ));
    }

    #[derive(VeksCli, Debug, PartialEq)]
    enum Cmd {
        Ping(Ping),
        /// List datasets
        List {
            #[arg(long)]
            verbose: bool,
        },
    }

    #[test]
    fn derive_enum_subcommand_dispatch() {
        let spec = Cmd::veks_command_spec("veks");
        assert!(spec.subcommand_required);
        // tuple variant delegating to a struct
        let p = crate::cli::parse(&spec, &argv(&["ping", "--dataset", "d"])).unwrap();
        match Cmd::veks_from_parsed(&p).unwrap() {
            Cmd::Ping(ping) => assert_eq!(ping.dataset, "d"),
            _ => panic!("expected Ping"),
        }
        // named-field variant
        let p2 = crate::cli::parse(&spec, &argv(&["list", "--verbose"])).unwrap();
        assert_eq!(Cmd::veks_from_parsed(&p2).unwrap(), Cmd::List { verbose: true });
    }
}
