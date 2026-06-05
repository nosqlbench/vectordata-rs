# vecd concepts

The mental model behind vecd's commands — enough to know *why* a setup looks
the way it does, without reading the full [design doc](../design/vecd-daemon.md)
(which has the precise rules and worked examples). Four ideas:
[namespaces](#namespaces), [backends](#backends),
[identities & access](#identities--access), and
[versions & sessions](#versions--sessions).

## Namespaces

A **namespace** is a path prefix that owns everything beneath it. `datasets/`
governs the dataset at `datasets/glove/…` and the catalog at
`datasets/catalog.yaml` alike. Each namespace carries:

- a **backend** — where its object bytes are stored (see below);
- an **owner** — a user who holds all authority over the subtree (the apex of
  the access cone);
- a **quota** (bytes; default 50 TiB) and an optional **TTL** (objects auto-
  expire into "stasis" after it);
- **visibility** and **role bindings** — who can see and do what.

Namespaces nest into an **access tree**: a binding or owner on a parent covers
the whole subtree, and (for the public audiences) a child can never be more
open than its parent. Create one with `vecd ns add <path> --owner <user>
--backend-config <backend>`; "private" is just a namespace with nothing opened
up.

## Backends

A **backend config** is a named storage connection — *where* bytes live,
decoupled from *who* may touch them (that's the namespace's job). Register one
with `vecd backends add <name> --kind <kind> --endpoint <uri> --active`:

| Kind | Endpoint | Use |
|------|----------|-----|
| `local` | `local:/var/lib/vecd/objects` | single host or NFS-backed; atomic temp+rename |
| `s3` | `s3://bucket/prefix` | cloud-native (optional `--region`, `--endpoint-url`, `--aws-profile`) |
| `mem` | `mem:<id>` | in-process, ephemeral — tests only |

One backend can serve many namespaces; a namespace points at exactly one. An
endpoint may have only one active config at a time.

## Identities & access

vecd separates two planes on purpose:

- the **management plane** (create users, mint tokens, define/bind roles, set
  ownership/TTL, read logs) — governed by a user's global **privilege level**;
- the **data plane** (read/write/delete objects) — governed by **roles** bound
  to principals on **scopes**, plus ownership.

**Privilege levels** are a strict ladder — `superuser` > `admin` > `operator` >
`user` — and no one can grant or assume a level above their own. Note the local
`vecd` admin CLI is **superuser-by-filesystem-access**: it operates directly on
the SQLite control-plane DB, so holding that file *is* the authority. Levels
primarily gate token-bearing API principals.

**Tokens** are opaque bearer secrets sent as `Authorization: Bearer <token>`.
They expire (90 days by default, 365 max), are stored only as hashes (the
plaintext is shown once), and carry a **narrowed subset** of their issuer's
authority — a read-only token can never write, even for an owner.

**Roles** are nesting aggregate classes on the action axis, each implying the
ones below it:

| Role | Implies | Actions |
|------|---------|---------|
| `read` | — | READ |
| `publish` | `read` | READ, WRITE |
| `maintain` | `publish` | READ, WRITE, DELETE |
| `curate` (≡ `all`) | `maintain` | READ, WRITE, DELETE, ADMIN |

You name the class, not the bag of actions, and a class granted on a namespace
covers its whole subtree. Two built-in groups are always present: **`PUBLIC`**
(every caller, authenticated or not) and **`KNOWN`** (any caller with a valid
token). There is no `PRIVATE` — privacy is the default that falls out when
nothing is opened.

```bash
vecd bind --to PUBLIC --role reader   --ns datasets/glove/    # world-readable
vecd bind --to KNOWN  --role reader   --ns datasets/internal/ # any logged-in user
vecd bind --to alice  --role curate   --ns datasets/          # alice owns/curates
# private: bind nothing to PUBLIC/KNOWN — owner-only.
```

### The access cone

Why "cone"? Authority flows down from an owner and is only ever *narrowed* —
never amplified. A request is allowed when the action is in the **union** of
all bindings that apply to the caller over prefixes covering the key, **and**
it survives three ceilings:

- **Token ceiling** — the session is intersected with the token's action
  subset.
- **Delegation ceiling** — a binding can only grant actions the granting
  principal already holds, so sub-delegations only narrow.
- **Ancestry ceiling** — a `PUBLIC`/`KNOWN` audience is admitted to a key only
  if it's admitted on *every* covering ancestor. So a child is never more open
  than its parent; a `reader→PUBLIC` grant under a closed parent is inert until
  the parent is opened too. (Explicit user/owner grants aren't subject to this
  umbrella.)

Collect by union within a principal's grants; narrow by the ceilings. That's
the whole model — see the design doc's worked examples for the corner cases.
Inspect the live grants any time with `vecd bindings [--ns <prefix>]`.

## Versions & sessions

Object writes come in two flavors:

- A plain **`push`** (or a lone `PUT`) is an immediate per-object write with
  CAS (compare-and-set via `If-Match`/`If-None-Match`) and quota enforcement.
  What's live is the set of objects under the namespace — see
  `vecd objects <ns>`.
- An **upload session** wraps a set of writes into a **transactional
  publication**: changes stage invisibly to readers, then a single atomic
  pointer flip publishes a new, content-addressed **version** of the namespace.
  Readers always see a consistent version; a half-finished publish is never
  visible. Session versions show under `vecd versions <ns>`.

Integrity is content-addressed: object identity is a content key and a version
has a manifest hash, so a version names exactly the bytes it published. (vecd
itself never hashes object *content* for the client's integrity guarantee —
that's the client's separate `SHA256SUMS` round-trip; vecd's ETags are opaque
envelope tags.)

## See also

- [Intro & quickstart](./vecd-intro.md) · [vecd.conf reference](./vecd-config.md)
- [End-to-end tutorial](../tutorials/vecd-end-to-end/) (these concepts in action)
- [Daemon design doc](../design/vecd-daemon.md) (the precise rules)
