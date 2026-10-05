// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Shared vocabulary for the AAA model: data-plane **actions** and their
//! aggregate **classes**, management-plane **privilege levels**, namespace
//! **listability**, and the crate error type. These map directly onto the
//! privilege tree in `docs/design/vecd-daemon.md` (§ *The privilege
//! tree*).

use std::fmt;
use std::str::FromStr;

/// A base data-plane action. The four actions are the leaves of the
/// privilege tree's *action axis*; HTTP methods map onto them
/// (GET/HEAD→READ, PUT→WRITE, DELETE→DELETE, governance→ADMIN).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Action {
    Read,
    Write,
    Delete,
    Admin,
}

impl Action {
    const READ: u8 = 1 << 0;
    const WRITE: u8 = 1 << 1;
    const DELETE: u8 = 1 << 2;
    const ADMIN: u8 = 1 << 3;

    fn bit(self) -> u8 {
        match self {
            Action::Read => Self::READ,
            Action::Write => Self::WRITE,
            Action::Delete => Self::DELETE,
            Action::Admin => Self::ADMIN,
        }
    }

    /// Lower-case canonical name (as used in `roles.actions` and the CLI).
    pub fn name(self) -> &'static str {
        match self {
            Action::Read => "read",
            Action::Write => "write",
            Action::Delete => "delete",
            Action::Admin => "admin",
        }
    }
}

impl FromStr for Action {
    type Err = VecdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "read" => Ok(Action::Read),
            "write" => Ok(Action::Write),
            "delete" => Ok(Action::Delete),
            "admin" => Ok(Action::Admin),
            other => Err(VecdError::Usage(format!("unknown action '{other}'"))),
        }
    }
}

/// A set of [`Action`]s, stored as a bitmask. This is the unit the cone
/// collects (union) and narrows (intersect).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct ActionSet(u8);

impl ActionSet {
    pub const EMPTY: ActionSet = ActionSet(0);
    /// Every action — the authority an owner holds at the apex of a cone.
    pub const ALL: ActionSet =
        ActionSet(Action::READ | Action::WRITE | Action::DELETE | Action::ADMIN);

    pub fn of(actions: &[Action]) -> Self {
        let mut s = ActionSet::EMPTY;
        for a in actions {
            s.0 |= a.bit();
        }
        s
    }

    pub fn contains(self, a: Action) -> bool {
        self.0 & a.bit() != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Union — additive collection of privilege.
    pub fn union(self, other: ActionSet) -> ActionSet {
        ActionSet(self.0 | other.0)
    }

    /// Intersection — the narrowing every ceiling applies.
    pub fn intersect(self, other: ActionSet) -> ActionSet {
        ActionSet(self.0 & other.0)
    }

    /// Iterate the contained actions in canonical order.
    pub fn iter(self) -> impl Iterator<Item = Action> {
        [Action::Read, Action::Write, Action::Delete, Action::Admin]
            .into_iter()
            .filter(move |a| self.contains(*a))
    }

    /// Comma-joined canonical names (`"read,write"`), or `"-"` if empty —
    /// the form stored in `roles.actions` and shown by `ping`.
    pub fn to_csv(self) -> String {
        if self.is_empty() {
            return "-".to_string();
        }
        self.iter().map(|a| a.name()).collect::<Vec<_>>().join(",")
    }

    /// Parse a comma-separated action list (`"read,write,delete"`).
    pub fn parse_csv(s: &str) -> Result<Self, VecdError> {
        let mut set = ActionSet::EMPTY;
        for tok in s.split(',') {
            let tok = tok.trim();
            if tok.is_empty() || tok == "-" {
                continue;
            }
            set = set.union(ActionSet::of(&[tok.parse()?]));
        }
        Ok(set)
    }
}

/// An aggregate **class** — the shorthand a grant names instead of a bag
/// of actions. Each implies all classes below it (`curate` ⊃ `maintain` ⊃
/// `publish` ⊃ `read`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Read,
    Publish,
    Maintain,
    Curate,
}

impl Class {
    /// The action set this class expands to.
    pub fn actions(self) -> ActionSet {
        match self {
            Class::Read => ActionSet::of(&[Action::Read]),
            Class::Publish => ActionSet::of(&[Action::Read, Action::Write]),
            Class::Maintain => ActionSet::of(&[Action::Read, Action::Write, Action::Delete]),
            Class::Curate => ActionSet::ALL,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Class::Read => "read",
            Class::Publish => "publish",
            Class::Maintain => "maintain",
            Class::Curate => "curate",
        }
    }
}

impl FromStr for Class {
    type Err = VecdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "read" | "reader" => Ok(Class::Read),
            "publish" | "publisher" => Ok(Class::Publish),
            "maintain" | "maintainer" => Ok(Class::Maintain),
            "curate" | "curator" | "all" => Ok(Class::Curate),
            other => Err(VecdError::Usage(format!(
                "unknown role class '{other}' (read|publish|maintain|curate)"
            ))),
        }
    }
}

/// Management-plane privilege level — a strict ladder; no principal may
/// grant or assume a level above its own.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    User = 0,
    Operator = 1,
    Admin = 2,
    Superuser = 3,
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Level::User => "user",
            Level::Operator => "operator",
            Level::Admin => "admin",
            Level::Superuser => "superuser",
        }
    }

    pub const ALL: [Level; 4] = [Level::User, Level::Operator, Level::Admin, Level::Superuser];
}

impl FromStr for Level {
    type Err = VecdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "user" => Ok(Level::User),
            "operator" => Ok(Level::Operator),
            "admin" => Ok(Level::Admin),
            "superuser" => Ok(Level::Superuser),
            other => Err(VecdError::Usage(format!(
                "unknown level '{other}' (user|operator|admin|superuser)"
            ))),
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Who may *see a namespace exists* — independent of object-level access.
/// An owner may narrow but never widen past what the admin set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Listable {
    /// Anyone, authenticated or not.
    Public,
    /// Any caller presenting a valid token.
    Known,
    /// Only principals with some binding/ownership in the subtree (default).
    Grantees,
}

impl Listable {
    pub fn name(self) -> &'static str {
        match self {
            Listable::Public => "public",
            Listable::Known => "known",
            Listable::Grantees => "grantees",
        }
    }
}

impl FromStr for Listable {
    type Err = VecdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "public" => Ok(Listable::Public),
            "known" => Ok(Listable::Known),
            "grantees" => Ok(Listable::Grantees),
            other => Err(VecdError::Usage(format!(
                "unknown listability '{other}' (public|known|grantees)"
            ))),
        }
    }
}

/// Default per-namespace storage cap: 50 TiB — large enough that quotas
/// are always *present* (enforcement can tighten incrementally) without
/// blocking normal use.
pub const DEFAULT_QUOTA_BYTES: u64 = 50 * 1024 * 1024 * 1024 * 1024;

/// Default token lifetime when `--expires` is omitted: 90 days.
pub const DEFAULT_TOKEN_TTL_SECS: i64 = 90 * 24 * 60 * 60;

/// Hard maximum token lifetime, enforced regardless of config: 365 days.
pub const MAX_TOKEN_TTL_SECS: i64 = 365 * 24 * 60 * 60;

/// The standard system privilege: write into an over-quota namespace.
pub const PRIV_IGNORE_QUOTAS: &str = "IGNORE-QUOTAS";

/// The crate-wide error. The CLI maps [`VecdError::Usage`] to exit code 2
/// and everything else to 1, mirroring `vectordata push`'s convention.
#[derive(thiserror::Error, Debug)]
pub enum VecdError {
    /// Caller error — bad arguments, unknown name, precondition the user
    /// can fix. Exit code 2.
    #[error("{0}")]
    Usage(String),
    /// Operational failure — I/O, DB, backend, transport. Exit code 1.
    #[error("{0}")]
    Operational(String),
    #[error(transparent)]
    Db(#[from] rusqlite::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl VecdError {
    pub fn usage(m: impl Into<String>) -> Self {
        VecdError::Usage(m.into())
    }
    pub fn op(m: impl Into<String>) -> Self {
        VecdError::Operational(m.into())
    }
    /// The process exit code this error maps to.
    pub fn exit_code(&self) -> i32 {
        match self {
            VecdError::Usage(_) => 2,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_nesting_expands_actions() {
        assert!(Class::Read.actions().contains(Action::Read));
        assert!(!Class::Read.actions().contains(Action::Write));
        assert!(Class::Publish.actions().contains(Action::Write));
        assert!(!Class::Publish.actions().contains(Action::Delete));
        assert!(Class::Maintain.actions().contains(Action::Delete));
        assert!(!Class::Maintain.actions().contains(Action::Admin));
        assert_eq!(Class::Curate.actions(), ActionSet::ALL);
    }

    #[test]
    fn action_set_union_and_intersect() {
        let rw = ActionSet::of(&[Action::Read, Action::Write]);
        let r = ActionSet::of(&[Action::Read]);
        assert_eq!(rw.intersect(r), r);
        assert_eq!(r.union(ActionSet::of(&[Action::Delete])).to_csv(), "read,delete");
        assert!(ActionSet::EMPTY.is_empty());
    }

    #[test]
    fn action_set_csv_roundtrip() {
        let s = ActionSet::parse_csv("read, write ,delete").unwrap();
        assert_eq!(s, Class::Maintain.actions());
        assert_eq!(ActionSet::parse_csv("-").unwrap(), ActionSet::EMPTY);
        assert_eq!(s.to_csv(), "read,write,delete");
    }

    #[test]
    fn levels_are_ordered() {
        assert!(Level::Superuser > Level::Admin);
        assert!(Level::Admin > Level::Operator);
        assert!(Level::Operator > Level::User);
    }
}
