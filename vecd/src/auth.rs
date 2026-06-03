// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Authentication: minting bearer tokens, hashing them for storage, and
//! resolving a presented `Authorization: Bearer …` to a [`Caller`].
//!
//! Tokens are opaque secrets shown **once** at creation and stored only as
//! their SHA-256 hash. A request with no token is [`Caller::Anonymous`]
//! (the `PUBLIC` group); a presented token that is unknown or expired is a
//! hard authentication failure (the server answers 401), distinct from a
//! permitted-but-anonymous request.

use sha2::{Digest, Sha256};

use crate::authz::{Caller, Snapshot};

/// Token plaintext prefix — purely cosmetic, so a leaked secret is
/// recognizable as a vecd key (matches the `vd_…` convention the client's
/// credential store uses).
const TOKEN_PREFIX: &str = "vd_";

/// Why authentication of a *presented* token failed. (No token at all is
/// not a failure — it is anonymous access.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    /// The hash matched no token.
    Unknown,
    /// The token is past its `expires_at`.
    Expired,
}

impl AuthError {
    pub fn message(self) -> &'static str {
        match self {
            AuthError::Unknown => "unknown or revoked token",
            AuthError::Expired => "token has expired",
        }
    }
}

/// Mint a fresh token: returns `(plaintext, hash)`. The plaintext is shown
/// to the user once; only the hash is persisted.
pub fn generate_token() -> (String, String) {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let plaintext = format!("{TOKEN_PREFIX}{}", hex::encode(bytes));
    let hash = hash_token(&plaintext);
    (plaintext, hash)
}

/// The SHA-256 hash (hex) under which a token is stored and looked up.
pub fn hash_token(plaintext: &str) -> String {
    let mut h = Sha256::new();
    h.update(plaintext.as_bytes());
    hex::encode(h.finalize())
}

/// Resolve a presented bearer credential against the snapshot.
///
/// - `bearer == None` → [`Caller::Anonymous`].
/// - a known, unexpired token → [`Caller::User`] narrowed by its profile.
/// - anything else → `Err` (the server maps this to 401).
///
/// `now` is epoch seconds (injected so this stays pure / testable).
pub fn authenticate(
    snap: &Snapshot,
    bearer: Option<&str>,
    now: i64,
) -> Result<Caller, AuthError> {
    let Some(token) = bearer else {
        return Ok(Caller::Anonymous);
    };
    let hash = hash_token(token);
    let Some(info) = snap.lookup_token(&hash) else {
        return Err(AuthError::Unknown);
    };
    if info.expires_at <= now {
        return Err(AuthError::Expired);
    }
    Ok(Caller::User {
        name: info.user_name.clone(),
        level: info.level,
        profile: info.profile.clone(),
        token_desc: info.description.clone(),
        token_id: info.token_id,
    })
}

/// Extract the bearer token from an `Authorization` header value, if it is
/// a well-formed `Bearer <token>`.
pub fn bearer_from_header(value: Option<&str>) -> Option<&str> {
    let v = value?.trim();
    let rest = v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer "))?;
    let rest = rest.trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{ControlPlane, TokenRow, UserRow};

    fn snapshot_with_token(expires_at: i64, disabled: bool) -> (Snapshot, String) {
        let (plaintext, hash) = generate_token();
        let mut cp = ControlPlane::default();
        cp.users.push(UserRow {
            id: 1,
            name: "alice".into(),
            disabled,
            level: "user".into(),
            password_hash: None,
        });
        cp.tokens.push(TokenRow {
            id: 1,
            user_id: 1,
            token_hash: hash,
            description: "ci key".into(),
            profile: None,
            expires_at,
        });
        (Snapshot::build(&cp), plaintext)
    }

    #[test]
    fn no_token_is_anonymous() {
        let (snap, _) = snapshot_with_token(i64::MAX, false);
        assert!(matches!(authenticate(&snap, None, 100).unwrap(), Caller::Anonymous));
    }

    #[test]
    fn valid_token_resolves_to_user() {
        let (snap, tok) = snapshot_with_token(1000, false);
        let caller = authenticate(&snap, Some(&tok), 100).unwrap();
        match caller {
            Caller::User { name, token_desc, .. } => {
                assert_eq!(name, "alice");
                assert_eq!(token_desc, "ci key");
            }
            _ => panic!("expected user"),
        }
    }

    #[test]
    fn expired_token_is_rejected() {
        let (snap, tok) = snapshot_with_token(1000, false);
        assert_eq!(authenticate(&snap, Some(&tok), 1000), Err(AuthError::Expired));
        assert_eq!(authenticate(&snap, Some(&tok), 2000), Err(AuthError::Expired));
    }

    #[test]
    fn unknown_token_is_rejected() {
        let (snap, _) = snapshot_with_token(1000, false);
        assert_eq!(authenticate(&snap, Some("vd_deadbeef"), 100), Err(AuthError::Unknown));
    }

    #[test]
    fn disabled_user_token_is_unknown() {
        // A disabled user is excluded from the snapshot index entirely.
        let (snap, tok) = snapshot_with_token(1000, true);
        assert_eq!(authenticate(&snap, Some(&tok), 100), Err(AuthError::Unknown));
    }

    #[test]
    fn bearer_parsing() {
        assert_eq!(bearer_from_header(Some("Bearer vd_x")), Some("vd_x"));
        assert_eq!(bearer_from_header(Some("bearer  vd_y ")), Some("vd_y"));
        assert_eq!(bearer_from_header(Some("Basic abc")), None);
        assert_eq!(bearer_from_header(Some("Bearer ")), None);
        assert_eq!(bearer_from_header(None), None);
    }
}
