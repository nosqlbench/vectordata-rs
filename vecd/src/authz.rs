// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The authorization model: an indexed, reloadable **control-plane
//! snapshot** and the **privilege cone** that resolves it.
//!
//! Per-request authorization never touches the DB on the hot path — it
//! runs against an in-memory [`Snapshot`] the daemon swaps atomically on a
//! live reload. The cone (`docs/design/vecd-daemon.md` § *The privilege
//! cone*) is: **union to collect, intersect to narrow** —
//!
//! ```text
//! allowed(C, K) = ( ⋃ bindings applying to C over prefixes covering K  ∪  owner-apex )
//!                 ∩ token_ceiling(C, K)
//!                 with group (PUBLIC/KNOWN) grants gated by the ancestry ceiling
//! ```
//!
//! - **Owner apex** — the owner of any covering namespace holds every
//!   action there (owner may be a user or a system role `@level`).
//! - **Token ceiling** — a token may carry a profile (a `(class, scope)`
//!   subset); the session is intersected with it, so a read-only token
//!   never writes even for an owner.
//! - **Ancestry ceiling (narrow-only openness)** — a `PUBLIC`/`KNOWN`
//!   audience is admitted at `K` only if every *existing* covering
//!   ancestor namespace also admits that group (has a binding for it).
//!   The auto-seeded root (`""`) is exempt. Named user/owner grants are
//!   **not** subject to this — it gates only the group audiences.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::db::{ControlPlane, NamespaceRow};
use crate::model::{Action, ActionSet, Class, Level, Listable};

/// One `(class, scope)` entry of a token's or named profile's access
/// profile. `scope` is a namespace path (`""` = whole server).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileEntry {
    pub class: String,
    pub scope: String,
}

/// A token's parsed access profile (`None` = full issuer authority).
pub type Profile = Vec<ProfileEntry>;

/// The principal making a request, after authentication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Caller {
    /// No (or no valid) token — belongs to `PUBLIC` only.
    Anonymous,
    /// A valid token resolved to its issuing user, narrowed by the token's
    /// profile.
    User {
        name: String,
        level: Level,
        /// `None` = the user's full authority; `Some` = the token ceiling.
        profile: Option<Profile>,
        /// The token's mandatory description (shown at every usage point).
        token_desc: String,
        token_id: i64,
    },
}

impl Caller {
    pub fn is_authenticated(&self) -> bool {
        matches!(self, Caller::User { .. })
    }
    pub fn name(&self) -> Option<&str> {
        match self {
            Caller::User { name, .. } => Some(name),
            Caller::Anonymous => None,
        }
    }
    pub fn token_desc(&self) -> Option<&str> {
        match self {
            Caller::User { token_desc, .. } => Some(token_desc),
            Caller::Anonymous => None,
        }
    }
    pub fn level(&self) -> Option<Level> {
        match self {
            Caller::User { level, .. } => Some(*level),
            Caller::Anonymous => None,
        }
    }
}

/// One namespace as a caller sees it — for `ping`/`whoami`.
#[derive(Clone, Debug)]
pub struct NsAccess {
    pub path: String,
    /// Effective actions after the cone + token narrowing.
    pub actions: ActionSet,
    pub listable: Listable,
    /// Whether the caller owns this namespace.
    pub owner: bool,
}

/// A namespace as the snapshot sees it.
#[derive(Clone, Debug)]
pub struct Ns {
    pub path: String,
    pub owner: String,
    pub backend_config: Option<String>,
    pub active: bool,
    pub listable: Listable,
    pub quota_bytes: u64,
    pub ttl_seconds: Option<i64>,
}

impl Ns {
    fn from_row(r: &NamespaceRow) -> Self {
        Ns {
            path: r.path.clone(),
            owner: r.owner.clone(),
            backend_config: r.backend_config.clone(),
            active: r.active,
            listable: r.listable.parse().unwrap_or(Listable::Grantees),
            quota_bytes: r.quota_bytes.max(0) as u64,
            ttl_seconds: r.ttl_seconds,
        }
    }
}

/// What a valid token resolves to — its issuing principal and ceiling.
/// Held in the snapshot so authentication never hits the DB on the hot
/// path.
#[derive(Clone, Debug)]
pub struct TokenInfo {
    pub token_id: i64,
    pub user_name: String,
    pub level: Level,
    pub description: String,
    pub profile: Option<Profile>,
    pub expires_at: i64,
}

/// The indexed, immutable control-plane snapshot used for per-request
/// decisions. Cheap to build, atomically swapped on reload.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub generation: i64,
    /// user name → level
    user_levels: HashMap<String, Level>,
    /// token sha256 hash (hex) → resolved token info
    tokens: HashMap<String, TokenInfo>,
    /// bindings, kept flat (the binding count is small relative to objects)
    bindings: Vec<Binding>,
    /// namespace path → namespace
    namespaces: HashMap<String, Ns>,
    /// backend config name → row (routing target)
    backends: HashMap<String, crate::db::BackendRow>,
    /// principal → granted system privileges
    system_privileges: HashMap<String, HashSet<String>>,
}

#[derive(Clone, Debug)]
struct Binding {
    principal: String,
    actions: ActionSet,
    scope: String,
}

impl Snapshot {
    /// Build an indexed snapshot from a raw control-plane read.
    pub fn build(cp: &ControlPlane) -> Self {
        let roles: HashMap<String, ActionSet> = cp
            .roles
            .iter()
            .map(|r| (r.name.clone(), ActionSet::parse_csv(&r.actions).unwrap_or(ActionSet::EMPTY)))
            .collect();

        let user_levels: HashMap<String, Level> = cp
            .users
            .iter()
            .filter(|u| !u.disabled)
            .map(|u| (u.name.clone(), u.level.parse().unwrap_or(Level::User)))
            .collect();

        // id → (name, level), restricted to enabled users — a token for a
        // disabled user resolves to nothing.
        let by_id: HashMap<i64, (String, Level)> = cp
            .users
            .iter()
            .filter(|u| !u.disabled)
            .map(|u| (u.id, (u.name.clone(), u.level.parse().unwrap_or(Level::User))))
            .collect();

        let tokens: HashMap<String, TokenInfo> = cp
            .tokens
            .iter()
            .filter_map(|t| {
                let (user_name, level) = by_id.get(&t.user_id)?.clone();
                let profile = t.profile.as_deref().and_then(|s| {
                    serde_json::from_str::<Profile>(s).ok()
                });
                Some((
                    t.token_hash.clone(),
                    TokenInfo {
                        token_id: t.id,
                        user_name,
                        level,
                        description: t.description.clone(),
                        profile,
                        expires_at: t.expires_at,
                    },
                ))
            })
            .collect();

        let bindings = cp
            .bindings
            .iter()
            .map(|b| Binding {
                principal: b.principal.clone(),
                actions: roles.get(&b.role).copied().unwrap_or(ActionSet::EMPTY),
                scope: normalize(&b.namespace_path),
            })
            .collect();

        let namespaces = cp
            .namespaces
            .iter()
            .map(|n| (normalize(&n.path), Ns::from_row(n)))
            .collect();

        let backends = cp.backends.iter().map(|b| (b.name.clone(), b.clone())).collect();

        let mut system_privileges: HashMap<String, HashSet<String>> = HashMap::new();
        for p in &cp.system_privileges {
            system_privileges.entry(p.principal.clone()).or_default().insert(p.privilege.clone());
        }

        Snapshot {
            generation: cp.generation,
            user_levels,
            tokens,
            bindings,
            namespaces,
            backends,
            system_privileges,
        }
    }

    /// A backend config row by name.
    pub fn backend(&self, name: &str) -> Option<&crate::db::BackendRow> {
        self.backends.get(name)
    }

    /// Look up a token by its sha256 hash (hex). Returns the resolved info
    /// regardless of expiry — the caller ([`crate::auth`]) checks `now`
    /// against `expires_at` so the reason ("expired") is distinguishable.
    pub fn lookup_token(&self, token_hash: &str) -> Option<&TokenInfo> {
        self.tokens.get(token_hash)
    }

    /// The privilege level of a known, enabled user (for management-plane
    /// checks).
    pub fn user_level(&self, name: &str) -> Option<Level> {
        self.user_levels.get(name).copied()
    }

    pub fn namespace(&self, path: &str) -> Option<&Ns> {
        self.namespaces.get(&normalize(path))
    }

    pub fn namespaces(&self) -> impl Iterator<Item = &Ns> {
        self.namespaces.values()
    }

    /// The deepest namespace record covering `key` (its access/storage
    /// home). Always at least the root, which is seeded.
    pub fn resolve_ns(&self, key: &str) -> Option<&Ns> {
        let key = normalize(key);
        self.namespaces
            .values()
            .filter(|n| covers(&n.path, &key))
            .max_by_key(|n| n.path.len())
    }

    /// Does this principal hold a system privilege (e.g. `IGNORE-QUOTAS`)?
    pub fn has_system_privilege(&self, caller: &Caller, privilege: &str) -> bool {
        match caller {
            Caller::Anonymous => false,
            Caller::User { name, level, .. } => {
                if let Some(set) = self.system_privileges.get(name) {
                    if set.contains(privilege) {
                        return true;
                    }
                }
                // Superusers implicitly hold all system privileges.
                *level >= Level::Superuser
            }
        }
    }

    /// The effective action set a caller has on `key` — the full cone.
    pub fn allowed(&self, caller: &Caller, key: &str) -> ActionSet {
        let key = normalize(key);

        // Covering namespaces (existing records whose path covers the key).
        let covering: Vec<&Ns> =
            self.namespaces.values().filter(|n| covers(&n.path, &key)).collect();

        // ── collect (union) ──────────────────────────────────────────
        let mut collected = ActionSet::EMPTY;

        // Owner apex: the owner of any covering namespace holds all actions.
        for ns in &covering {
            if self.owner_matches(&ns.owner, caller) {
                collected = collected.union(ActionSet::ALL);
            }
        }

        // Named (user) bindings — not subject to the ancestry ceiling.
        if let Some(user) = caller.name() {
            for b in &self.bindings {
                if b.principal == user && covers(&b.scope, &key) {
                    collected = collected.union(b.actions);
                }
            }
        }

        // Group audiences — gated by the ancestry ceiling.
        if self.group_admitted("PUBLIC", &covering) {
            collected = collected.union(self.group_actions("PUBLIC", &key));
        }
        if caller.is_authenticated() && self.group_admitted("KNOWN", &covering) {
            collected = collected.union(self.group_actions("KNOWN", &key));
        }

        // ── narrow (intersect): the token ceiling ────────────────────
        if let Caller::User { profile: Some(profile), .. } = caller {
            collected = collected.intersect(profile_actions(profile, &key));
        }

        collected
    }

    /// True iff `caller` may perform `action` on `key`.
    pub fn can(&self, caller: &Caller, action: Action, key: &str) -> bool {
        self.allowed(caller, key).contains(action)
    }

    /// Namespaces this caller may *see* (per `listable`), each with the
    /// caller's effective actions and whether they own it. Plus the count
    /// of namespaces hidden from them. Powers `vectordata ping`/`whoami`.
    /// The root (`""`) is omitted — it is a structural container.
    pub fn visible_to(&self, caller: &Caller) -> (Vec<NsAccess>, usize) {
        let mut visible = Vec::new();
        let mut hidden = 0usize;
        for ns in self.namespaces.values() {
            if ns.path.is_empty() {
                continue;
            }
            let actions = self.allowed(caller, &ns.path);
            let owner = self.owner_matches(&ns.owner, caller);
            let can_see = match ns.listable {
                Listable::Public => true,
                Listable::Known => caller.is_authenticated(),
                Listable::Grantees => owner || !actions.is_empty(),
            };
            if can_see {
                visible.push(NsAccess { path: ns.path.clone(), actions, listable: ns.listable, owner });
            } else {
                hidden += 1;
            }
        }
        visible.sort_by(|a, b| a.path.cmp(&b.path));
        (visible, hidden)
    }

    /// Does an owner string (a user name or a system role `@level`) match
    /// this caller?
    fn owner_matches(&self, owner: &str, caller: &Caller) -> bool {
        match caller {
            Caller::Anonymous => false,
            Caller::User { name, level, .. } => {
                if let Some(role_level) = owner.strip_prefix('@') {
                    // System-role ownership: governed collectively by the
                    // privilege level (and anyone above it).
                    match role_level.parse::<Level>() {
                        Ok(req) => *level >= req,
                        Err(_) => false,
                    }
                } else {
                    owner == name
                }
            }
        }
    }

    /// Union of a group's binding actions over prefixes covering `key`.
    fn group_actions(&self, group: &str, key: &str) -> ActionSet {
        let mut acc = ActionSet::EMPTY;
        for b in &self.bindings {
            if b.principal == group && covers(&b.scope, key) {
                acc = acc.union(b.actions);
            }
        }
        acc
    }

    /// The ancestry ceiling: a group is admitted at the key only if every
    /// *existing, non-root* covering namespace also admits the group (has
    /// a binding for it).
    fn group_admitted(&self, group: &str, covering: &[&Ns]) -> bool {
        covering.iter().all(|ns| {
            if ns.path.is_empty() {
                return true; // root is exempt
            }
            self.bindings.iter().any(|b| b.principal == group && b.scope == ns.path)
        })
    }
}

/// The actions a profile grants at `key` (union of class actions over
/// entries whose scope covers the key).
fn profile_actions(profile: &Profile, key: &str) -> ActionSet {
    let mut acc = ActionSet::EMPTY;
    for e in profile {
        if covers(&normalize(&e.scope), key) {
            if let Ok(class) = e.class.parse::<Class>() {
                acc = acc.union(class.actions());
            }
        }
    }
    acc
}

/// Normalize a namespace path / key: trim surrounding slashes. The empty
/// string is the root.
pub fn normalize(p: &str) -> String {
    p.trim_matches('/').to_string()
}

/// Does `scope` cover `key`? (`scope` is an ancestor-or-self prefix.)
pub fn covers(scope: &str, key: &str) -> bool {
    let scope = scope.trim_matches('/');
    let key = key.trim_matches('/');
    scope.is_empty() || key == scope || key.starts_with(&format!("{scope}/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{BindingRow, NamespaceRow, RoleRow, SystemPrivRow, UserRow};
    use crate::model::PRIV_IGNORE_QUOTAS;

    /// A builder for control-plane fixtures so the worked examples read
    /// like the design doc.
    #[derive(Default)]
    struct Fix {
        cp: ControlPlane,
    }
    impl Fix {
        fn new() -> Self {
            let mut f = Fix::default();
            for (name, actions) in [
                ("read", "read"),
                ("publish", "read,write"),
                ("maintain", "read,write,delete"),
                ("curate", "read,write,delete,admin"),
            ] {
                f.cp.roles.push(RoleRow {
                    name: name.into(),
                    actions: actions.into(),
                    builtin: true,
                });
            }
            f
        }
        fn user(mut self, name: &str, level: Level) -> Self {
            let id = self.cp.users.len() as i64 + 1;
            self.cp.users.push(UserRow {
                id,
                name: name.into(),
                disabled: false,
                level: level.name().into(),
                password_hash: None,
            });
            self
        }
        fn ns(mut self, path: &str, owner: &str) -> Self {
            self.cp.namespaces.push(NamespaceRow {
                path: path.into(),
                owner: owner.into(),
                backend_config: Some("b".into()),
                active: true,
                listable: "grantees".into(),
                quota_bytes: 1 << 40,
                ttl_seconds: None,
            });
            self
        }
        fn bind(mut self, principal: &str, role: &str, ns: &str) -> Self {
            self.cp.bindings.push(BindingRow {
                principal: principal.into(),
                role: role.into(),
                namespace_path: ns.into(),
            });
            self
        }
        fn syspriv(mut self, principal: &str, priv_: &str) -> Self {
            self.cp
                .system_privileges
                .push(SystemPrivRow { principal: principal.into(), privilege: priv_.into() });
            self
        }
        fn snap(self) -> Snapshot {
            Snapshot::build(&self.cp)
        }
    }

    fn anon() -> Caller {
        Caller::Anonymous
    }
    fn user(name: &str, level: Level) -> Caller {
        Caller::User {
            name: name.into(),
            level,
            profile: None,
            token_desc: "test".into(),
            token_id: 1,
        }
    }
    fn user_scoped(name: &str, level: Level, profile: Profile) -> Caller {
        Caller::User { name: name.into(), level, profile: Some(profile), token_desc: "t".into(), token_id: 1 }
    }
    fn entry(class: &str, scope: &str) -> ProfileEntry {
        ProfileEntry { class: class.into(), scope: scope.into() }
    }

    #[test]
    fn ex1_public_dataset_anonymous_pull() {
        let s = Fix::new().ns("datasets/glove-100", "@admin").bind("PUBLIC", "read", "datasets/glove-100").snap();
        assert!(s.can(&anon(), Action::Read, "datasets/glove-100/base.fvec"));
        assert!(!s.can(&anon(), Action::Write, "datasets/glove-100/base.fvec"));
    }

    #[test]
    fn ex2_known_only() {
        let s = Fix::new()
            .user("alice", Level::User)
            .ns("datasets/internal", "@admin")
            .bind("KNOWN", "read", "datasets/internal")
            .snap();
        assert!(!s.can(&anon(), Action::Read, "datasets/internal/x"));
        assert!(s.can(&user("alice", Level::User), Action::Read, "datasets/internal/x"));
    }

    #[test]
    fn ex3_private_by_default() {
        let s = Fix::new()
            .user("alice", Level::User)
            .user("bob", Level::User)
            .ns("users/alice/scratch", "alice")
            .snap();
        assert!(!s.can(&user("bob", Level::User), Action::Read, "users/alice/scratch/x"));
        assert!(s.can(&user("alice", Level::User), Action::Write, "users/alice/scratch/x"));
        assert!(s.can(&user("alice", Level::User), Action::Admin, "users/alice/scratch/x"));
    }

    #[test]
    fn ex5_token_narrows_below_user() {
        // Alice owns datasets/glove (all actions) but the token is read-only.
        let s = Fix::new().user("alice", Level::User).ns("datasets/glove", "alice").snap();
        let ro = user_scoped("alice", Level::User, vec![entry("read", "datasets/glove")]);
        assert!(s.can(&ro, Action::Read, "datasets/glove/x"));
        assert!(!s.can(&ro, Action::Write, "datasets/glove/x"));
        // The user (full authority) could write.
        assert!(s.can(&user("alice", Level::User), Action::Write, "datasets/glove/x"));
    }

    #[test]
    fn ex6_union_collects_ceiling_bounds() {
        let s = Fix::new()
            .user("bob", Level::User)
            .ns("datasets", "@admin")
            .ns("datasets/bobset", "@admin")
            .bind("KNOWN", "read", "datasets")
            .bind("bob", "publish", "datasets/bobset")
            .snap();
        // Full authority: union {READ (KNOWN)} ∪ {READ,WRITE (bob)} = {R,W}.
        let bob = user("bob", Level::User);
        assert!(s.can(&bob, Action::Read, "datasets/bobset/x"));
        assert!(s.can(&bob, Action::Write, "datasets/bobset/x"));
        // Read-only token narrows the session to {READ}.
        let ro = user_scoped("bob", Level::User, vec![entry("read", "datasets")]);
        assert!(s.can(&ro, Action::Read, "datasets/bobset/x"));
        assert!(!s.can(&ro, Action::Write, "datasets/bobset/x"));
    }

    #[test]
    fn ex7_ancestry_ceiling_narrow_only() {
        // datasets/ exists and is closed (no PUBLIC binding); datasets/glove
        // grants reader→PUBLIC. The child binding is inert.
        let s = Fix::new()
            .user("alice", Level::User)
            .ns("datasets", "@admin")
            .ns("datasets/glove", "@admin")
            .bind("PUBLIC", "read", "datasets/glove")
            .snap();
        assert!(!s.can(&anon(), Action::Read, "datasets/glove/x"));
        // Opening datasets/ too makes it effective.
        let s2 = Fix::new()
            .ns("datasets", "@admin")
            .ns("datasets/glove", "@admin")
            .bind("PUBLIC", "read", "datasets")
            .bind("PUBLIC", "read", "datasets/glove")
            .snap();
        assert!(s2.can(&anon(), Action::Read, "datasets/glove/x"));
        // A named grant on the child is NOT subject to the umbrella.
        let s3 = Fix::new()
            .user("alice", Level::User)
            .ns("datasets", "@admin")
            .ns("datasets/glove", "@admin")
            .bind("alice", "read", "datasets/glove")
            .snap();
        assert!(s3.can(&user("alice", Level::User), Action::Read, "datasets/glove/x"));
    }

    #[test]
    fn system_role_owner_matches_by_level() {
        let s = Fix::new()
            .user("op", Level::Operator)
            .user("adm", Level::Admin)
            .ns("infra", "@admin")
            .snap();
        // An admin governs an @admin-owned namespace; an operator does not.
        assert!(s.can(&user("adm", Level::Admin), Action::Admin, "infra/x"));
        assert!(!s.can(&user("op", Level::Operator), Action::Read, "infra/x"));
    }

    #[test]
    fn ignore_quotas_privilege() {
        let s = Fix::new().user("drainer", Level::User).syspriv("drainer", PRIV_IGNORE_QUOTAS).snap();
        assert!(s.has_system_privilege(&user("drainer", Level::User), PRIV_IGNORE_QUOTAS));
        assert!(!s.has_system_privilege(&user("nobody", Level::User), PRIV_IGNORE_QUOTAS));
        // Superuser holds it implicitly.
        assert!(s.has_system_privilege(&user("root", Level::Superuser), PRIV_IGNORE_QUOTAS));
    }
}
