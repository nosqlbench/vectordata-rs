# vecd — what it is, and a 2-minute quickstart

**`vecd`** is the server that lets you **publish vectordata datasets for others
to read** — a private, self-hosted gateway that sits between `vectordata` /
`veks` clients and your object storage. It handles **authentication**
(bearer tokens), **authorization** (per-namespace access control), and
**versioned, atomic publication**, then serves datasets over plain HTTP so any
`vectordata` client can discover and read them.

You don't need vecd to *use* vectordata — local and S3 datasets work without a
server. You want vecd when you need to **share datasets across a team or
network** with access control, instead of handing out raw bucket credentials.

## The shape of it

```
   vectordata / veks              vecd                    storage backend
   (clients & CLI)            (this server)             (local dir, S3, mem)
  ┌──────────────┐    HTTP    ┌────────────────────┐     ┌───────────────┐
  │  read  /     │  ───────▶  │  authenticate      │ ──▶ │   objects     │
  │  push        │  (bearer   │  authorize (cone)  │     │  (versioned   │
  │  datasets    │   token)   │  version + serve   │ ◀── │   snapshots)  │
  └──────────────┘  ◀───────  └────────────────────┘     └───────────────┘
                                control plane: a SQLite DB of users,
                                tokens, namespaces, grants, and versions
```

Five concepts carry the whole model:

- **Backend** — a named storage connection (`local:<dir>`, `s3://bucket/prefix`,
  or `mem:` for tests). Where object bytes actually live.
- **Namespace** — a path prefix (e.g. `datasets/`) bound to one backend and
  one owner, with its own quota, TTL, and access grants. Everything under the
  prefix lives here.
- **Token** — an opaque bearer secret a client sends as
  `Authorization: Bearer …`. Tokens expire (90 days by default) and carry a
  narrowed subset of their issuer's authority.
- **Grant / the access cone** — `vecd bind --to <principal> --role <role> --ns
  <namespace>` opens access. `PUBLIC` means unauthenticated; a grant on a
  namespace governs everything beneath it.
- **Version** — every push creates an atomic, content-addressed snapshot;
  readers always see a consistent version.

For the mental model behind each, see **[vecd concepts](./vecd-concepts.md)**;
for the precise rules, the [design doc](../design/vecd-daemon.md).

## 2-minute quickstart

Get the binaries (or build from the workspace — `cargo build -p vecd -p veks -p
vectordata --bins`):

```bash
cargo install --path vecd
cargo install --path veks
cargo install --path vectordata
```

Stand up a local server and publish a dataset anyone on the box can read:

```bash
# 1. Create a config with safe defaults (bind 127.0.0.1:8443 — LOCAL ONLY —
#    and a data dir). vecd requires a config before it will run; `config auto`
#    writes one and confirms. (`--yes` skips the prompt.)
vecd config auto

# 2. Initialize the control-plane DB and mint a superuser token. init also
#    auto-logs-in the superuser (stores the token in the config dir's
#    credentials.json and names the file) — it's the irrecoverable admin key.
vecd init --superuser root

# 3. Start the daemon (binds the local-only address from the config).
vecd start

# 4. Register a storage backend, a user, a namespace, and access grants.
vecd backends add store --kind local --endpoint "local:$PWD/vecd-objects" --active
vecd users add alice --level user
vecd ns add datasets --owner alice --backend-config store --active
vecd bind --to alice  --role curate --ns datasets   # alice can publish
vecd bind --to PUBLIC --role reader --ns datasets    # anyone can read

# 5. Mint a push token for alice (printed once — capture it). Add --json to
#    save a token record {token,user,id,expires_at} you can feed straight to
#    --token below (vecd tokens create … --json > alice.json).
vecd tokens create --user alice --description "quickstart push key" --expires 30d
```

> **Single-shot standup:** `vecd init --config-import setup.yaml` (or `.json`,
> or a directory) seeds the config and creates the DB in one command — handy
> for provisioning. `vecd config …` manages the config — `auto` (defaults),
> `get` (read whole/one value, any format), `set` (set a value or replace from
> a file); `--conf <dir>` points at an alternate config location. See the
> [config reference](./vecd-config.md).

Now push a dataset and read it back. (Need a dataset to push? Generate a toy
one with `veks` — see the [end-to-end tutorial](../tutorials/vecd-end-to-end/),
which scripts this whole flow.)

```bash
# Authenticate once — the token is stored (keyed by endpoint) and used
# automatically by later reads and pushes, so no --token flag below.
# --token accepts the literal token, a bare-token file, or a --json record
# (e.g. --token alice.json), and records which user the token is for.
vectordata login http://127.0.0.1:8443/ --token "<TOKEN-FROM-STEP-5>"

# Make a throwaway example dataset to publish — base + query vectors and a
# minimal dataset.yaml, in one step (no external data needed). Point
# `veks prepare bootstrap` + `veks run` at it later to add derived facets.
veks generate example-dataset --target ./my-dataset

# Publish a dataset directory as version 1 of datasets/mydata.
vectordata datasets push ./my-dataset --to http://127.0.0.1:8443/datasets/mydata/

# Read it back — anonymously, via the PUBLIC reader grant.
vectordata datasets precache http://127.0.0.1:8443/datasets/mydata/dataset.yaml
```

The `vecd` CLI is a client too: `vecd login <endpoint> --token …` then
`vecd whoami <endpoint>` checks your access (token stored under the config dir).
Check on the server any time: `vecd status`, `vecd log` (access log),
`curl localhost:8443/healthz`, `curl localhost:8443/metrics`.

## Going to production

The local-only default keeps a bare `vecd start` safe. To serve over the
network, set a non-loopback `bind` **and** configure TLS — vecd prints a
cleartext warning if you expose a public address without it:

```
# in $VECD_CONFIG/vecd.conf
bind     = 0.0.0.0:8443
tls_cert = /etc/vecd/cert.pem
tls_key  = /etc/vecd/key.pem
```

(Or terminate TLS at a reverse proxy in front of a loopback-bound vecd.) All
operator settings live in `vecd.conf` — every key mirrors a `vecd serve` flag;
see the **[vecd.conf reference](./vecd-config.md)** and the
**[deployment guide](../../deploy/vecd/README.md)** (Docker, systemd, TLS).

## Where to go next

- **[End-to-end tutorial](../tutorials/vecd-end-to-end/)** — the full loop as
  runnable scripts: server, toy dataset, upload, catalog, client exploration.
- **[Rate-limits tutorial](../tutorials/vecd-rate-limits/)** — per-connection
  vs per-client bandwidth caps and concurrent-chunk download scaling.
- **[Design doc](../design/vecd-daemon.md)** — the complete architecture: the
  authorization cone, version/session semantics, storage and integrity model.
