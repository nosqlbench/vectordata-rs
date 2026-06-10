// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `#[derive(VeksCli)]` — generates the [`veks_completion::VeksCli`] impl for a
//! command/args struct or a subcommand enum. From one annotated declaration it
//! produces the `CommandSpec` (which drives parsing, help, and completion) and
//! the typed extraction from a `ParsedArgs`. This is the in-tree replacement
//! for clap's derive.
//!
//! ## Structs (a command's arguments)
//! - a field with `#[arg(long)]`/`#[arg(short)]` is an **option**;
//! - a `bool` field is a **flag** (no value);
//! - a bare field (no `long`/`short`) is a **positional**;
//! - `Vec<T>` ⇒ repeatable, `Option<T>` ⇒ optional, bare `T` ⇒ required
//!   (unless `#[arg(default = "…")]`);
//! - `#[command(flatten)]` pulls in another `VeksCli` struct's options (DRY);
//! - `#[command(subcommand)]` on an enum-typed field wires up subcommands.
//!
//! ## Enums (subcommands)
//! Each variant becomes a subcommand: unit ⇒ no args, `V(ArgsStruct)` ⇒ delegate,
//! `V { … }` ⇒ inline fields.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TS2;
use quote::quote;
use syn::{
    punctuated::Punctuated, token::Comma, Attribute, Data, DeriveInput, Expr, ExprLit, Field,
    Fields, Lit, Meta, Type,
};

#[proc_macro_derive(VeksCli, attributes(arg, command))]
pub fn derive_veks_cli(input: TokenStream) -> TokenStream {
    let di = syn::parse_macro_input!(input as DeriveInput);
    let result = match &di.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => derive_struct(&di, &named.named),
            _ => Err(syn::Error::new_spanned(&di.ident, "VeksCli structs need named fields")),
        },
        Data::Enum(e) => derive_enum(&di, &e.variants),
        Data::Union(_) => Err(syn::Error::new_spanned(&di.ident, "VeksCli does not support unions")),
    };
    result.unwrap_or_else(|e| e.to_compile_error()).into()
}

// ---------------------------------------------------------------------------
// Attribute parsing
// ---------------------------------------------------------------------------

#[derive(Default)]
struct FieldAttrs {
    long: Option<String>,        // explicit long name (without dashes)
    has_long: bool,              // `long` was present (bare or with value)
    short: Option<char>,
    default: Option<String>,
    value_name: Option<String>,
    positional: bool,
    flatten: bool,
    subcommand: bool,
    help: Option<String>,
    /// Closed set of valid values from `#[arg(value_parser = ["a", "b"])]`.
    /// Drives a `closed_set` completion provider on the generated `OptionSpec`.
    value_parser: Vec<String>,
    /// A computed default from `#[arg(default_value_t = <expr>)]`. The expression
    /// yields the field type directly and is used as the fallback when the flag
    /// is absent (the option is then optional rather than required).
    default_expr: Option<TS2>,
}

fn collect_docs(attrs: &[Attribute]) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    for a in attrs {
        if a.path().is_ident("doc") {
            if let Meta::NameValue(nv) = &a.meta {
                if let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = &nv.value {
                    lines.push(s.value().trim().to_string());
                }
            }
        }
    }
    if lines.is_empty() {
        None
    } else {
        // Join consecutive doc lines into one paragraph; blank lines split.
        Some(lines.join(" ").trim().to_string())
    }
}

/// Extract `#[command(alias = "x")]` / `#[command(aliases = ["x", "y"])]`.
fn collect_aliases(attrs: &[Attribute]) -> Vec<String> {
    let mut out = Vec::new();
    for a in attrs {
        if a.path().is_ident("command") {
            let _ = a.parse_nested_meta(|meta| {
                if meta.path.is_ident("alias") {
                    if let Ok(v) = meta.value() {
                        if let Ok(s) = v.parse::<syn::LitStr>() {
                            out.push(s.value());
                        }
                    }
                } else if meta.path.is_ident("aliases") {
                    if let Ok(v) = meta.value() {
                        if let Ok(arr) = v.parse::<syn::ExprArray>() {
                            for elem in arr.elems {
                                if let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = elem {
                                    out.push(s.value());
                                }
                            }
                        }
                    }
                } else {
                    let _ = meta.value().and_then(|v| v.parse::<Expr>());
                }
                Ok(())
            });
        }
    }
    out
}

fn parse_field_attrs(field: &Field) -> syn::Result<FieldAttrs> {
    let mut fa = FieldAttrs { help: collect_docs(&field.attrs), ..Default::default() };
    for attr in &field.attrs {
        if attr.path().is_ident("arg") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("long") {
                    fa.has_long = true;
                    if let Ok(v) = meta.value() {
                        let s: syn::LitStr = v.parse()?;
                        fa.long = Some(s.value());
                    }
                } else if meta.path.is_ident("short") {
                    if let Ok(v) = meta.value() {
                        let c: syn::LitChar = v.parse()?;
                        fa.short = Some(c.value());
                    } else {
                        fa.short = Some('\0'); // sentinel: derive from long's first char
                    }
                } else if meta.path.is_ident("default") || meta.path.is_ident("default_value") {
                    let v = meta.value()?;
                    let s: syn::LitStr = v.parse()?;
                    fa.default = Some(s.value());
                } else if meta.path.is_ident("value_name") {
                    let v = meta.value()?;
                    let s: syn::LitStr = v.parse()?;
                    fa.value_name = Some(s.value());
                } else if meta.path.is_ident("positional") {
                    fa.positional = true;
                } else if meta.path.is_ident("default_value_t") {
                    // A computed default expression (e.g. a function call). Kept
                    // verbatim and used as the fallback in `veks_from_parsed`.
                    if let Ok(v) = meta.value() {
                        if let Ok(e) = v.parse::<Expr>() {
                            fa.default_expr = Some(quote! { #e });
                        }
                    }
                } else if meta.path.is_ident("value_parser") {
                    // `value_parser = ["a", "b", …]` declares a closed set of
                    // valid values → capture the literals for a completion
                    // provider. A non-array value_parser (a parser fn/path) is
                    // consumed and ignored (the field's `FromStr` does the work).
                    let v = meta.value()?;
                    if let Expr::Array(arr) = v.parse::<Expr>()? {
                        for elem in arr.elems {
                            if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = elem {
                                fa.value_parser.push(s.value());
                            }
                        }
                    }
                } else {
                    // Unknown arg keys (conflicts_with, value_hint, alias, …) are
                    // tolerated and ignored — consume any `= value` so parsing
                    // continues.
                    let _ = meta.value().and_then(|v| v.parse::<Expr>());
                }
                Ok(())
            })?;
        } else if attr.path().is_ident("command") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("flatten") {
                    fa.flatten = true;
                } else if meta.path.is_ident("subcommand") {
                    fa.subcommand = true;
                } else {
                    let _ = meta.value().and_then(|v| v.parse::<Expr>());
                }
                Ok(())
            })?;
        }
    }
    Ok(fa)
}

/// `#[command(about = "…")]` or the type's doc comment.
/// Read a `#[command(key = "literal")]` string from an attribute list (works
/// for both type-level and enum-variant attrs).
fn command_str_attr(attrs: &[Attribute], key: &str) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("command") {
            let mut out = None;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident(key) {
                    if let Ok(v) = meta.value() {
                        if let Ok(s) = v.parse::<syn::LitStr>() {
                            out = Some(s.value());
                        }
                    }
                } else {
                    let _ = meta.value().and_then(|v| v.parse::<Expr>());
                }
                Ok(())
            });
            if out.is_some() {
                return out;
            }
        }
    }
    None
}

/// The `.stability(…)` builder call from `#[command(stability = "…")]`, if any.
/// An unrecognized value falls back to the default (`Stable`) at runtime.
fn stability_call(attrs: &[Attribute]) -> TS2 {
    match command_str_attr(attrs, "stability") {
        Some(s) => quote! {
            .stability(::veks_completion::Stability::from_name(#s).unwrap_or_default())
        },
        None => quote! {},
    }
}

fn type_about(di: &DeriveInput) -> Option<String> {
    for attr in &di.attrs {
        if attr.path().is_ident("command") {
            let mut about = None;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("about") {
                    if let Ok(v) = meta.value() {
                        if let Ok(s) = v.parse::<syn::LitStr>() {
                            about = Some(s.value());
                        }
                    }
                } else {
                    // Tolerate clap-only type-level keys (disable_help_subcommand,
                    // arg_required_else_help, …) — consume any `= value`.
                    let _ = meta.value().and_then(|v| v.parse::<Expr>());
                }
                Ok(())
            });
            if about.is_some() {
                return about;
            }
        }
    }
    collect_docs(&di.attrs)
}

/// `#[command(after_help = "…")]` / `#[command(after_long_help = "…")]` on any
/// attribute list (a type or an enum variant).
fn after_help_from_attrs(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("command") {
            let mut out = None;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("after_long_help") || meta.path.is_ident("after_help") {
                    if let Ok(v) = meta.value() {
                        if let Ok(s) = v.parse::<syn::LitStr>() {
                            out = Some(s.value());
                        }
                    }
                } else {
                    let _ = meta.value().and_then(|v| v.parse::<Expr>());
                }
                Ok(())
            });
            if out.is_some() {
                return out;
            }
        }
    }
    None
}

fn type_after_help(di: &DeriveInput) -> Option<String> {
    after_help_from_attrs(&di.attrs)
}

/// The `.about(…).after_help(…)` builder calls for a type's command spec.
fn about_after_help_calls(di: &DeriveInput) -> TS2 {
    let about = match type_about(di) {
        Some(a) => quote! { .about(#a) },
        None => quote! {},
    };
    let after = match type_after_help(di) {
        Some(h) => quote! { .after_help(#h) },
        None => quote! {},
    };
    let stability = stability_call(&di.attrs);
    quote! { #about #after #stability }
}

// ---------------------------------------------------------------------------
// Type shape helpers
// ---------------------------------------------------------------------------

fn type_is_bool(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("bool"))
}

/// If `ty` is `Wrapper<Inner>` (e.g. `Vec<T>`, `Option<T>`), return `Inner`.
fn generic_inner<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    if let Type::Path(p) = ty {
        let seg = p.path.segments.last()?;
        if seg.ident == wrapper {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                    return Some(inner);
                }
            }
        }
    }
    None
}

fn ident_to_long(ident: &syn::Ident) -> String {
    ident.to_string().trim_start_matches("r#").replace('_', "-")
}

fn pascal_to_kebab(ident: &syn::Ident) -> String {
    let s = ident.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i != 0 {
                out.push('-');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Field codegen (shared by structs and named enum variants)
// ---------------------------------------------------------------------------

/// Returns (spec-building statements operating on `__spec`, field initializers
/// `ident: expr` reading from `__p`).
fn process_fields(fields: &Punctuated<Field, Comma>) -> syn::Result<(Vec<TS2>, Vec<TS2>)> {
    let mut spec_stmts: Vec<TS2> = Vec::new();
    let mut inits: Vec<TS2> = Vec::new();
    let mut positional_index: usize = 0;

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let ty = &field.ty;
        let fa = parse_field_attrs(field)?;

        // ---- flatten / subcommand: delegate to another VeksCli type ----
        if fa.flatten {
            spec_stmts.push(quote! {
                __spec = <#ty as ::veks_completion::VeksCli>::veks_augment_spec(__spec);
            });
            inits.push(quote! {
                #ident: <#ty as ::veks_completion::VeksCli>::veks_from_parsed(__p)?
            });
            continue;
        }
        if fa.subcommand {
            if let Some(inner) = generic_inner(ty, "Option") {
                // Optional subcommand: present → `Some(Inner)`, absent → `None`.
                // The inner enum's augment marks the subcommand required; relax
                // that here so the parent command may stand alone.
                spec_stmts.push(quote! {
                    __spec = <#inner as ::veks_completion::VeksCli>::veks_augment_spec(__spec);
                    __spec.subcommand_required = false;
                });
                inits.push(quote! {
                    #ident: if __p.subcommand().is_some() {
                        ::std::option::Option::Some(
                            <#inner as ::veks_completion::VeksCli>::veks_from_parsed(__p)?,
                        )
                    } else {
                        ::std::option::Option::None
                    }
                });
            } else {
                spec_stmts.push(quote! {
                    __spec = <#ty as ::veks_completion::VeksCli>::veks_augment_spec(__spec);
                });
                inits.push(quote! {
                    #ident: <#ty as ::veks_completion::VeksCli>::veks_from_parsed(__p)?
                });
            }
            continue;
        }

        let help_call = match &fa.help {
            Some(h) => quote! { .help(#h) },
            None => quote! {},
        };

        // ---- boolean flag ----
        if type_is_bool(ty) {
            let long = fa.long.clone().unwrap_or_else(|| ident_to_long(ident));
            let dashed = format!("--{long}");
            let short_call = short_call(&fa, &long);
            spec_stmts.push(quote! {
                __spec = __spec.option(::veks_completion::OptionSpec::new(
                    ::veks_completion::OptionDef::flag(#dashed) #short_call #help_call
                ));
            });
            inits.push(quote! { #ident: __p.has_flag(#dashed) });
            continue;
        }

        let is_option = fa.has_long || fa.short.is_some();

        if is_option && !fa.positional {
            // ---- value option ----
            let long = fa.long.clone().unwrap_or_else(|| ident_to_long(ident));
            let dashed = format!("--{long}");
            let short_call = short_call(&fa, &long);
            let value_name_call = match &fa.value_name {
                Some(v) => quote! { .value_name(#v) },
                None => quote! {},
            };
            // Closed-set completion from `#[arg(value_parser = [..])]`: the same
            // declaration that constrains parsing also feeds tab-completion.
            let value_completion_call = if fa.value_parser.is_empty() {
                quote! {}
            } else {
                let vals = &fa.value_parser;
                quote! { .value_completion(::veks_completion::closed_set(&[#(#vals),*])) }
            };

            if let Some(inner) = generic_inner(ty, "Vec") {
                let parse = parse_expr(inner, &dashed);
                spec_stmts.push(quote! {
                    __spec = __spec.option(::veks_completion::OptionSpec::new(
                        ::veks_completion::OptionDef::value(#dashed).multiple(true) #short_call #value_name_call #help_call
                    ) #value_completion_call);
                });
                inits.push(quote! {
                    #ident: {
                        let mut __v = ::std::vec::Vec::new();
                        for __s in __p.values(#dashed) { __v.push(#parse); }
                        __v
                    }
                });
            } else if let Some(inner) = generic_inner(ty, "Option").filter(|i| generic_inner(i, "Vec").is_some()) {
                // Option<Vec<T>>: an optional repeatable. `None` when the flag is
                // absent, `Some(vec)` when one or more values were given.
                let vec_inner = generic_inner(inner, "Vec").unwrap();
                let parse = parse_expr(vec_inner, &dashed);
                spec_stmts.push(quote! {
                    __spec = __spec.option(::veks_completion::OptionSpec::new(
                        ::veks_completion::OptionDef::value(#dashed).multiple(true) #short_call #value_name_call #help_call
                    ) #value_completion_call);
                });
                inits.push(quote! {
                    #ident: {
                        let mut __v = ::std::vec::Vec::new();
                        for __s in __p.values(#dashed) { __v.push(#parse); }
                        if __v.is_empty() {
                            ::std::option::Option::None
                        } else {
                            ::std::option::Option::Some(__v)
                        }
                    }
                });
            } else if let Some(inner) = generic_inner(ty, "Option") {
                let parse = parse_expr(inner, &dashed);
                spec_stmts.push(quote! {
                    __spec = __spec.option(::veks_completion::OptionSpec::new(
                        ::veks_completion::OptionDef::value(#dashed) #short_call #value_name_call #help_call
                    ) #value_completion_call);
                });
                inits.push(quote! {
                    #ident: match __p.value(#dashed) {
                        ::std::option::Option::Some(__s) => ::std::option::Option::Some(#parse),
                        ::std::option::Option::None => ::std::option::Option::None,
                    }
                });
            } else if let Some(def_expr) = &fa.default_expr {
                // bare T with a computed default: optional, falls back to the
                // expression when the flag is absent.
                let parse = parse_expr(ty, &dashed);
                spec_stmts.push(quote! {
                    __spec = __spec.option(::veks_completion::OptionSpec::new(
                        ::veks_completion::OptionDef::value(#dashed) #short_call #value_name_call #help_call
                    ) #value_completion_call);
                });
                inits.push(quote! {
                    #ident: match __p.value(#dashed) {
                        ::std::option::Option::Some(__s) => #parse,
                        ::std::option::Option::None => #def_expr,
                    }
                });
            } else {
                // bare T: required unless a default is given
                let parse = parse_expr(ty, &dashed);
                match &fa.default {
                    Some(def) => {
                        spec_stmts.push(quote! {
                            __spec = __spec.option(::veks_completion::OptionSpec::new(
                                ::veks_completion::OptionDef::value(#dashed) #short_call #value_name_call #help_call
                            ).default(#def) #value_completion_call);
                        });
                        inits.push(quote! {
                            #ident: { let __s = __p.value(#dashed).unwrap_or(#def); #parse }
                        });
                    }
                    None => {
                        spec_stmts.push(quote! {
                            __spec = __spec.option(::veks_completion::OptionSpec::new(
                                ::veks_completion::OptionDef::value(#dashed) #short_call #value_name_call #help_call
                            ).required(true) #value_completion_call);
                        });
                        inits.push(quote! {
                            #ident: {
                                let __s = __p.value(#dashed).ok_or_else(||
                                    ::veks_completion::cli::ParseError::MissingRequiredOption {
                                        command: ::std::string::String::new(),
                                        flag: #dashed.to_string(),
                                    })?;
                                #parse
                            }
                        });
                    }
                }
            }
            continue;
        }

        // ---- positional ----
        let name = fa.value_name.clone().unwrap_or_else(|| ident.to_string().to_uppercase());
        let idx = positional_index;
        positional_index += 1;

        if let Some(inner) = generic_inner(ty, "Vec") {
            let parse = parse_expr(inner, &name);
            spec_stmts.push(quote! {
                __spec = __spec.positional(::veks_completion::PositionalSpec::new(#name).multiple(true) #help_call);
            });
            inits.push(quote! {
                #ident: {
                    let mut __v = ::std::vec::Vec::new();
                    for __s in __p.positionals().iter().skip(#idx) {
                        let __s = __s.as_str();
                        __v.push(#parse);
                    }
                    __v
                }
            });
        } else if let Some(inner) = generic_inner(ty, "Option") {
            let parse = parse_expr(inner, &name);
            spec_stmts.push(quote! {
                __spec = __spec.positional(::veks_completion::PositionalSpec::new(#name) #help_call);
            });
            inits.push(quote! {
                #ident: match __p.positionals().get(#idx) {
                    ::std::option::Option::Some(__s) => { let __s = __s.as_str(); ::std::option::Option::Some(#parse) },
                    ::std::option::Option::None => ::std::option::Option::None,
                }
            });
        } else {
            let parse = parse_expr(ty, &name);
            spec_stmts.push(quote! {
                __spec = __spec.positional(::veks_completion::PositionalSpec::new(#name).required(true) #help_call);
            });
            inits.push(quote! {
                #ident: {
                    let __s = __p.positionals().get(#idx).map(|__x| __x.as_str()).ok_or_else(||
                        ::veks_completion::cli::ParseError::MissingRequiredPositional {
                            command: ::std::string::String::new(),
                            name: #name.to_string(),
                        })?;
                    #parse
                }
            });
        }
    }

    Ok((spec_stmts, inits))
}

/// `.short('x')` builder call, deriving from the long's first char for the
/// bare `#[arg(short)]` sentinel.
fn short_call(fa: &FieldAttrs, long: &str) -> TS2 {
    match fa.short {
        Some('\0') => match long.chars().next() {
            Some(c) => quote! { .short(#c) },
            None => quote! {},
        },
        Some(c) => quote! { .short(#c) },
        None => quote! {},
    }
}

/// `<Inner as FromStr>::from_str(__s)` mapped into a `ParseError::InvalidValue`.
/// `__s` must be a `&str` in scope.
fn parse_expr(inner: &Type, flag: &str) -> TS2 {
    quote! {
        <#inner as ::std::str::FromStr>::from_str(__s).map_err(|__e|
            ::veks_completion::cli::ParseError::InvalidValue {
                flag: #flag.to_string(),
                value: __s.to_string(),
                message: __e.to_string(),
            })?
    }
}

// ---------------------------------------------------------------------------
// Struct & enum impls
// ---------------------------------------------------------------------------

fn derive_struct(di: &DeriveInput, fields: &Punctuated<Field, Comma>) -> syn::Result<TS2> {
    let name = &di.ident;
    let about_call = about_after_help_calls(di);
    let (spec_stmts, inits) = process_fields(fields)?;

    Ok(quote! {
        impl ::veks_completion::VeksCli for #name {
            fn veks_command_spec(__name: &str) -> ::veks_completion::CommandSpec {
                let __spec = ::veks_completion::CommandSpec::new(__name) #about_call;
                Self::veks_augment_spec(__spec)
            }
            fn veks_augment_spec(mut __spec: ::veks_completion::CommandSpec) -> ::veks_completion::CommandSpec {
                #(#spec_stmts)*
                __spec
            }
            fn veks_from_parsed(__p: &::veks_completion::ParsedArgs)
                -> ::std::result::Result<Self, ::veks_completion::cli::ParseError>
            {
                ::std::result::Result::Ok(#name { #(#inits),* })
            }
        }
    })
}

fn derive_enum(
    di: &DeriveInput,
    variants: &Punctuated<syn::Variant, Comma>,
) -> syn::Result<TS2> {
    let name = &di.ident;
    let about_call = about_after_help_calls(di);

    let mut sub_specs: Vec<TS2> = Vec::new();
    let mut match_arms: Vec<TS2> = Vec::new();

    for variant in variants {
        let vident = &variant.ident;
        let cmd_name = pascal_to_kebab(vident);
        let vabout = collect_docs(&variant.attrs);
        let vabout_call = match vabout {
            Some(a) => quote! { .about(#a) },
            None => quote! {},
        };
        let vafter_call = match after_help_from_attrs(&variant.attrs) {
            Some(h) => quote! { .after_help(#h) },
            None => quote! {},
        };
        let aliases = collect_aliases(&variant.attrs);
        let alias_calls = quote! { #(.alias(#aliases))* };
        let stability = stability_call(&variant.attrs);
        let vabout_call = quote! { #vabout_call #vafter_call #alias_calls #stability };

        match &variant.fields {
            Fields::Unit => {
                sub_specs.push(quote! {
                    __spec = __spec.subcommand(::veks_completion::CommandSpec::new(#cmd_name) #vabout_call);
                });
                match_arms.push(quote! {
                    ::std::option::Option::Some((#cmd_name, _)) => ::std::result::Result::Ok(Self::#vident),
                });
            }
            Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1 => {
                let inner = &unnamed.unnamed.first().unwrap().ty;
                sub_specs.push(quote! {
                    __spec = __spec.subcommand(
                        <#inner as ::veks_completion::VeksCli>::veks_command_spec(#cmd_name) #vabout_call
                    );
                });
                match_arms.push(quote! {
                    ::std::option::Option::Some((#cmd_name, __sub)) =>
                        ::std::result::Result::Ok(Self::#vident(
                            <#inner as ::veks_completion::VeksCli>::veks_from_parsed(__sub)?
                        )),
                });
            }
            Fields::Named(named) => {
                let (spec_stmts, inits) = process_fields(&named.named)?;
                sub_specs.push(quote! {
                    __spec = __spec.subcommand({
                        let mut __spec = ::veks_completion::CommandSpec::new(#cmd_name) #vabout_call;
                        #(#spec_stmts)*
                        __spec
                    });
                });
                // For the dispatch arm, fields read from the subcommand's parsed
                // args (`__p` inside `veks_from_parsed`).
                match_arms.push(quote! {
                    ::std::option::Option::Some((#cmd_name, __p)) =>
                        ::std::result::Result::Ok(Self::#vident { #(#inits),* }),
                });
            }
            Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    vident,
                    "VeksCli enum variants must be unit, single-field tuple, or named",
                ));
            }
        }
    }

    Ok(quote! {
        impl ::veks_completion::VeksCli for #name {
            fn veks_command_spec(__name: &str) -> ::veks_completion::CommandSpec {
                let __spec = ::veks_completion::CommandSpec::new(__name) #about_call;
                Self::veks_augment_spec(__spec)
            }
            fn veks_augment_spec(mut __spec: ::veks_completion::CommandSpec) -> ::veks_completion::CommandSpec {
                __spec.subcommand_required = true;
                #(#sub_specs)*
                __spec
            }
            fn veks_from_parsed(__top: &::veks_completion::ParsedArgs)
                -> ::std::result::Result<Self, ::veks_completion::cli::ParseError>
            {
                match __top.subcommand() {
                    #(#match_arms)*
                    ::std::option::Option::Some((__other, _)) =>
                        ::std::result::Result::Err(::veks_completion::cli::ParseError::UnknownSubcommand {
                            command: ::std::string::String::new(),
                            name: __other.to_string(),
                        }),
                    ::std::option::Option::None =>
                        ::std::result::Result::Err(::veks_completion::cli::ParseError::MissingSubcommand {
                            command: ::std::string::String::new(),
                        }),
                }
            }
        }
    })
}
