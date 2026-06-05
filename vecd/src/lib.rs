// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vecd` — the vectordata endpoint daemon.
//!
//! `vecd` is the production server behind a `vectordata` `https://`
//! publish/pull endpoint: an **AAA gateway** in front of object storage.
//! It authenticates callers, authorizes each push/pull against owners and
//! role bindings (the privilege *cone*), routes the request to the
//! namespace's storage backend, and enforces the conditional-write
//! contract that `vectordata push`'s single-provenance guarantee depends
//! on — while remaining a transparent drop-in for the REST object
//! protocol the client already speaks.
//!
//! The crate is split so the request-handling core ([`store`], [`authz`],
//! [`auth`], [`namespace`], [`backend`], [`db`]) is **synchronous and
//! runtime-free** — directly unit-testable, mirroring the way
//! `vectordata::push::execute` is transport-agnostic. [`server`] is the
//! thin `axum`/`tokio` shell that marshals each request onto that core.
//!
//! This is the **Phase 1** surface from `docs/design/vecd-daemon.md`
//! (decision #11): `serve` + TLS, namespaces, backend configs
//! (`local`/`s3`/`mem`), users + tokens (mandatory expiry + access
//! profiles), roles/bindings, the authn/authz cone, live reload, and DB
//! backup. Stasis/cleanup, the daemon lifecycle, introspection, and the
//! client-side `login`/`ping`/`backup` land in Phase 2.

pub mod auth;
pub mod authz;
pub mod backend;
pub mod db;
pub mod model;
pub mod namespace;
pub mod admin;
pub mod backup;
pub mod cli;
pub mod config;
pub mod credentials;
pub mod daemon;
pub mod lifetime;
pub mod ratelimit;
pub mod server;
pub mod session;
pub mod store;
pub mod upload;

pub use model::{Action, ActionSet, Class, Level, Listable, VecdError};
