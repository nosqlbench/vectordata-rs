# vecd

The `vectordata` endpoint daemon: a self-hosted gateway that publishes
vector datasets to the people and pipelines that read them.

`vecd` is an authentication, authorization and accounting gateway in front of object
storage. It serves the REST object protocol that
[`vectordata`](https://crates.io/crates/vectordata) already speaks for
`push` and pull, so a `vectordata` client works against it unchanged. On
every request it:

- authenticates the caller by bearer token (tokens always expire and
  carry an access profile);
- authorizes the push or pull against namespace owners and role
  bindings;
- routes it to the namespace's storage backend (`local`, `s3` or `mem`);
- enforces the conditional-write contract that `vectordata datasets push`'s
  single-provenance guarantee depends on, while streaming bytes through
  without re-hashing them.

```bash
cargo install vecd

vecd config auto                     # write a local-only config (127.0.0.1:8443)
vecd init --superuser root           # one-time: control-plane DB + a superuser token
vecd start                           # self-daemonizing; or `vecd serve` in the foreground
vecd backends add store --kind local --endpoint "local:$PWD/vecd-objects" --active
vecd status
```

Then add users, namespaces and role bindings (`vecd users`, `vecd ns`,
`vecd bind`), as the quickstart below walks through. Clients log in with
`vectordata login <url>`, publish with `vectordata datasets push`, and
read the endpoint like any other catalog.

`vecd` is supported on Linux. Its `start`/`stop`/`status` lifecycle
self-daemonizes through Unix process control, and release binaries are
published for Linux only.

## Documentation

- [Introduction and 2-minute quickstart](https://github.com/nosqlbench/vectordata-rs/blob/main/docs/guides/vecd-intro.md)
- [Deploying: binary, Docker, systemd](https://github.com/nosqlbench/vectordata-rs/blob/main/deploy/vecd/README.md)
- [End-to-end tutorial](https://github.com/nosqlbench/vectordata-rs/blob/main/docs/tutorials/vecd-end-to-end/README.md)
- [Design](https://github.com/nosqlbench/vectordata-rs/blob/main/docs/design/vecd-daemon.md)

The library half of the crate (request handling, authorization,
backends) is synchronous and runtime-free, so it is directly testable
and embeddable; `server` is the thin `axum`/`tokio` shell over it. See
the crate documentation.

License: Apache-2.0
