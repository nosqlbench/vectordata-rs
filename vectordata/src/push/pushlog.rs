// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `pushlog.jsonl` — the single primary provenance artifact for a
//! pushed dataset, modeled as an append-only event log.
//!
//! Every push is *bracketed* by two events sharing one monotonically
//! increasing `seq`: a `begin` (written first, declaring the intent of
//! a half-complete update) and a `complete` (written last, marking the
//! version stable for download). The presence of an unmatched `begin`
//! is the in-flux signal and the crash tombstone all at once — there is
//! no separate marker file.
//!
//! The log is also the *single provenance*: a local copy must converge
//! to the remote (be an ancestor of, or equal to, it) before a push may
//! proceed.
//!
//! See `docs/design/push-command.md` — *`pushlog.jsonl` is an event
//! log*, *Provenance convergence*, and *Upload versioning*.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Name of the provenance log at the publish root.
pub const PUSHLOG_FILE: &str = "pushlog.jsonl";

/// A single overwritten object recorded on a `begin` event.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Overwrite {
    pub key: String,
    pub old_digest: String,
    pub new_digest: String,
}

/// One line of the event log.
///
/// Internally tagged on `event` so a line is a flat JSON object exactly
/// as shown in the design doc.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "lowercase")]
pub enum Event {
    /// Opens a push: declares intent and the fingerprints it will set.
    Begin {
        seq: u64,
        ts: String,
        actor: String,
        cmd: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        overwrites: Vec<Overwrite>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        added: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        deletes: Vec<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        sums: BTreeMap<String, String>,
        tool_version: String,
    },
    /// Closes a push: the version is stable for download. Echoes the
    /// per-directory `sums` so a tail-only reader is self-contained.
    Complete {
        seq: u64,
        ts: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        sums: BTreeMap<String, String>,
    },
    /// Closes an open `begin` *without* completing it: the interrupted
    /// push at `seq` is abandoned (e.g. the source changed and the push
    /// was re-driven fresh under a later seq). Like `complete`, it makes
    /// the seq no longer "open".
    Abort {
        seq: u64,
        ts: String,
        actor: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

impl Event {
    /// The sequence number this event belongs to.
    pub fn seq(&self) -> u64 {
        match self {
            Event::Begin { seq, .. }
            | Event::Complete { seq, .. }
            | Event::Abort { seq, .. } => *seq,
        }
    }

    /// Is this a `begin` event?
    pub fn is_begin(&self) -> bool {
        matches!(self, Event::Begin { .. })
    }
}

/// An in-memory view of a `pushlog.jsonl`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Log {
    pub events: Vec<Event>,
}

/// How a local log relates to the remote (authoritative) log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Convergence {
    /// Identical histories.
    Equal,
    /// Local is the remote plus `extra` trailing events the remote
    /// lacks — recoverable; carry the extras up after acknowledgement.
    LocalAhead { extra: usize },
    /// Remote is the local plus `extra` trailing events — divergent;
    /// the local must re-sync before pushing.
    RemoteAhead { extra: usize },
    /// Neither is a prefix of the other — forked histories.
    Diverged { common: usize },
}

impl Log {
    /// Parse a `pushlog.jsonl` document (one JSON event per line).
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut events = Vec::new();
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let ev: Event = serde_json::from_str(line)
                .map_err(|e| format!("malformed pushlog line {}: {e}", n + 1))?;
            events.push(ev);
        }
        Ok(Log { events })
    }

    /// Render as JSONL (one event per line, trailing newline).
    pub fn render(&self) -> String {
        let mut out = String::new();
        for ev in &self.events {
            out.push_str(&serde_json::to_string(ev).expect("event serializes"));
            out.push('\n');
        }
        out
    }

    /// The sequence number a new push should use: one past the highest
    /// `seq` present, or 1 for an empty log.
    pub fn next_seq(&self) -> u64 {
        self.events.iter().map(Event::seq).max().map_or(1, |m| m + 1)
    }

    /// The `seq` of the latest `complete` — the stable version, if any.
    pub fn stable_version(&self) -> Option<u64> {
        self.events
            .iter()
            .rev()
            .find(|e| matches!(e, Event::Complete { .. }))
            .map(Event::seq)
    }

    /// The per-directory `sums` fingerprint of the latest `complete` —
    /// the content fingerprint of the current stable version, if any.
    pub fn stable_sums(&self) -> Option<&BTreeMap<String, String>> {
        self.events.iter().rev().find_map(|e| match e {
            Event::Complete { sums, .. } => Some(sums),
            _ => None,
        })
    }

    /// If the log's last event is a `begin` with no matching `complete`
    /// or `abort`, an update is open (in progress or crashed). Returns
    /// its `(seq, actor, ts)`.
    pub fn open_update(&self) -> Option<(u64, String, String)> {
        match self.events.last() {
            Some(Event::Begin { seq, actor, ts, .. }) => {
                Some((*seq, actor.clone(), ts.clone()))
            }
            _ => None,
        }
    }

    /// The trailing open `begin` event, if the log ends with one.
    pub fn trailing_open_begin(&self) -> Option<&Event> {
        match self.events.last() {
            Some(e @ Event::Begin { .. }) => Some(e),
            _ => None,
        }
    }

    /// The committed history: the log with any trailing open `begin`
    /// removed. Convergence is judged against this, since an open begin
    /// is an in-flight/crashed push, not settled history.
    pub fn committed(&self) -> Log {
        if matches!(self.events.last(), Some(Event::Begin { .. })) {
            Log { events: self.events[..self.events.len() - 1].to_vec() }
        } else {
            self.clone()
        }
    }

    /// Classify this (local) log against the `remote` log for
    /// convergence. See [`Convergence`].
    pub fn classify(&self, remote: &Log) -> Convergence {
        let common = self
            .events
            .iter()
            .zip(remote.events.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let (l, r) = (self.events.len(), remote.events.len());
        if common == l && common == r {
            Convergence::Equal
        } else if common == l {
            Convergence::RemoteAhead { extra: r - common }
        } else if common == r {
            Convergence::LocalAhead { extra: l - common }
        } else {
            Convergence::Diverged { common }
        }
    }

    /// The events present locally but not on the remote, given that the
    /// remote is a strict prefix (the `LocalAhead` case). Used to carry
    /// the missing tail up.
    pub fn tail_after(&self, prefix_len: usize) -> &[Event] {
        &self.events[prefix_len.min(self.events.len())..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn begin(seq: u64) -> Event {
        Event::Begin {
            seq,
            ts: "Mon, 01 Jan 2024 00:00:00 GMT".into(),
            actor: "u@h".into(),
            cmd: "vectordata push".into(),
            message: None,
            overwrites: vec![],
            added: vec![],
            deletes: vec![],
            sums: BTreeMap::new(),
            tool_version: "test".into(),
        }
    }
    fn complete(seq: u64) -> Event {
        Event::Complete { seq, ts: "Mon, 01 Jan 2024 00:00:01 GMT".into(), sums: BTreeMap::new() }
    }

    #[test]
    fn jsonl_roundtrip_and_tag() {
        let log = Log { events: vec![begin(1), complete(1)] };
        let text = log.render();
        assert!(text.lines().next().unwrap().contains("\"event\":\"begin\""));
        assert_eq!(Log::parse(&text).unwrap(), log);
    }

    #[test]
    fn next_seq_and_stable_version() {
        let mut log = Log::default();
        assert_eq!(log.next_seq(), 1);
        assert_eq!(log.stable_version(), None);
        log.events.push(begin(1));
        // open begin: stable is still none, next seq is 2
        assert_eq!(log.stable_version(), None);
        assert_eq!(log.next_seq(), 2);
        log.events.push(complete(1));
        assert_eq!(log.stable_version(), Some(1));
        assert_eq!(log.next_seq(), 2);
    }

    #[test]
    fn open_update_detected_only_on_trailing_begin() {
        let mut log = Log { events: vec![begin(1), complete(1)] };
        assert!(log.open_update().is_none());
        log.events.push(begin(2));
        let (seq, actor, _) = log.open_update().unwrap();
        assert_eq!(seq, 2);
        assert_eq!(actor, "u@h");
    }

    #[test]
    fn convergence_cases() {
        let base = Log { events: vec![begin(1), complete(1)] };
        // equal
        assert_eq!(base.classify(&base), Convergence::Equal);
        // local ahead (local has an extra push)
        let local_ahead = Log {
            events: vec![begin(1), complete(1), begin(2), complete(2)],
        };
        assert_eq!(local_ahead.classify(&base), Convergence::LocalAhead { extra: 2 });
        // remote ahead → divergent
        assert_eq!(base.classify(&local_ahead), Convergence::RemoteAhead { extra: 2 });
        // forked: same first event, different second
        let forked_a = Log { events: vec![begin(1), complete(1)] };
        let mut forked_b_ev = vec![begin(1)];
        forked_b_ev.push(complete(2)); // different second event
        let forked_b = Log { events: forked_b_ev };
        assert!(matches!(forked_a.classify(&forked_b), Convergence::Diverged { common: 1 }));
    }
}
