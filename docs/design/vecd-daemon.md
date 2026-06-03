# `vecd` — the vectordata endpoint daemon (design draft)

**Status:** draft / proposal. No code yet. This is the security-model and
architecture spec for a new workspace crate, `vecd`, to be reviewed
before implementation — the same spec-then-build rhythm used for the push
command. AAA decisions are load-bearing, so they get nailed here first.

## What problem this solves

`vectordata push` and the read path already speak a clean object-REST
protocol over HTTPS (`GET`/`HEAD`/`PUT`/`DELETE`, bearer auth,
`If-Match`/`If-None-Match`, `ETag`). Today there is no *server* on the
other end that does **access control** — the in-repo `objectstore.rs`
test mock is an anonymous, in-memory toy that proved the contract.

`vecd` is the production server: the **AAA gateway** in front of the
object store. It authenticates callers, authorizes each push/pull against
owners and permissions, enforces data lifetimes, and records an access
log — while remaining a *transparent drop-in* for the HTTPS endpoint that
`push`/pull already target. Nothing in the client changes for the happy
path; a `.publish_url` of `https://vecd-host/datasets/glove-100/` plus a
bearer token Just Works.

(Why a new daemon rather than an off-the-shelf system? See the
[build-vs-buy footnote](#footnote-build-vs-buy--why-build-vecd) — the
field surveyed, and why none of it covers the load-bearing middle.)

```
veks publish ─▶ vectordata push ─▶ HTTPS (PUT/GET/…) ─▶ ┌──────────────┐
                vectordata read  ─▶ HTTPS (GET/HEAD)  ─▶ │     vecd     │
                                                          │  AAA gateway │
                                                          │  ├─ authn (tokens)
                                                          │  ├─ authz (roles/owners)
                                                          │  ├─ lifetimes (TTL → stasis, admin purge)
                                                          │  ├─ access log
                                                          │  └─ blob store + SQLite
                                                          └──────────────┘
```

## Non-negotiable: stay a faithful endpoint

`vecd` must satisfy the exact contract `HttpsTransport` (and the read
path) rely on, or it breaks the guarantees we just hardened:

- `GET`/`HEAD` → body / size, with an **`ETag`** and accurate
  **`Content-Length`** (the client's truncation guard checks it).
- `PUT` honoring **`If-None-Match: *`** (412 if present) and
  **`If-Match: "<etag>"`** (412 on mismatch). This is mandatory — push's
  preflight conditional-write probe *refuses any endpoint that ignores
  it* (failure mode I10). `vecd` must pass that probe.
- `DELETE` → remove; the **`ETag` is the object's content-key** (a
  canonical descriptive-metadata hash — see *Hashing & addressing* — not a
  whole-content hash). Clients treat ETags as opaque, so this stays
  compatible with push's existing CAS round-trip.
- Bearer auth via `Authorization: Bearer <token>` (the same token push
  takes from `--token` / `$VECTORDATA_PUSH_TOKEN`).

The provenance/checksum artifacts (`.publish_url`, `pushlog.jsonl`,
`SHA256SUMS`) are ordinary objects to `vecd` — but it is *provenance-aware*
where that buys something (ownership binding from `.publish_url`,
accounting from `pushlog.jsonl`; see below).

## Hashing & addressing — a terminology invariant

Two distinct hashing notions are named precisely throughout this spec and
**never conflated**:

- **"content-addressed" / "content-key" ≡ a *canonical descriptive-
  metadata hash*.** A content-key is a sha256 over a **canonical
  serialization of an object's descriptive metadata** (size, element
  type/shape, the object's full-content fingerprint, and other canonical
  descriptors) — **not** a hash of the whole byte stream. It is the
  store's *address* for an object and the basis of dedup and the CAS
  ETag. The descriptors are **deterministically derived from the object
  itself** (size, type/shape, the embedded full-content fingerprint —
  *not* its location or time), so a content-key is **content-determining
  and global**: byte-identical objects share one content-key everywhere
  (enabling dedup and content-addressed fetch), while it remains, by
  construction, a metadata hash. Physically the bytes live under the
  logical-mirrored path `<backend>/<key>/<content-key>`; the same
  content-key may therefore appear under several keys, and a fetch *by*
  content-key (`/-/blob/<content-key>`) is resolved through vecd's
  content-key index in the DB.
- **"full content hashing via <method>" ≡ hashing the whole byte
  stream.** Wherever byte-level integrity or resumable verification is
  meant, the spec says so explicitly *and names the method* — in practice
  **full content hashing via a merkle tree** (vectordata's `.mref`/`.mrkl`
  chunk tree), used for verified, chunk-resumable reads and as the
  full-content fingerprint embedded in a content-key.
- **`manifest_hash`** is likewise a *canonical metadata hash* — a sha256
  over the version's canonical manifest (`key → content-key`), not over
  content bytes.

So: **addressing, dedup, and ETag/CAS run on metadata hashes**
(content-keys, manifest hashes); **byte-level integrity runs on full
content hashing via a merkle tree.** Any "hash" in this document resolves
to exactly one of these two, stated explicitly. (The client-side
`SHA256SUMS` is, by contrast, *full content hashing via `sha256sum`* — a
separate, explicitly-whole-file artifact push already produces.)

## AAA model

### Identities & credentials (Authentication)

- **users** — a named principal (`id`, `name`, `created`, `disabled`).
- **tokens (API keys)** — opaque bearer secrets, **stored only as SHA-256
  hashes**, the plaintext shown **once** at creation. Every token:
  - **expires.** `expires_at` is **mandatory for every token, no
    exceptions** — there are no non-expiring credentials. Default 90 days,
    **hard maximum 365 days** (config-tunable, but the max is enforced);
    service accounts rotate via `token issue`. Expired tokens are
    rejected.
  - **carries an access profile** (next subsection) — by default the full
    authority of its issuing user, or a *selected subset* for a delegated
    key.
  - has a **mandatory description**, shown at **every usage point** (access
    log, `ping`/`whoami`, audit) so a key's purpose is always visible.
  - is a **capability, not an account**: it acts under its issuing
    principal's identity (narrowed by its profile), and **its holder need
    not be a known principal** — you can hand the key to anyone.
  Auth = `sha256(presented) → token_hash → token`, rejecting disabled
  users and expired tokens.
- **unauthenticated** — a request with no token belongs to the `PUBLIC`
  group only (see *Standard groups*); it gets exactly what `PUBLIC`
  bindings allow (e.g. public read), nothing more.

### Authorization — RBAC with privilege levels

Two planes, deliberately separated:

- the **management plane** (create users, mint tokens, define roles, bind
  roles, assign ownership/TTL, run cleanup, read logs) — governed by a user's
  global **privilege level**;
- the **data plane** (read/write/delete objects) — governed by **roles**
  bound to principals on **scopes**, plus ownership.

#### Privilege levels (management plane)

Every user has one global privilege level — a strict ladder; no one may
grant or assume a level above their own:

| Level | Management capability |
|---|---|
| `superuser` | everything, including managing other admins and server-wide settings |
| `admin` | manage users at or below their level, tokens, role bindings, ownership, and TTL **within scopes they administer**; cannot elevate |
| `operator` | run `cleanup`, read logs, set TTL; no user/token/role management |
| `user` | data plane only — no management |

The **local `vecd` admin CLI is superuser-by-filesystem-access**: it
operates directly on the SQLite DB, so holding the DB file *is* the
authority (matching "command line tools local to the sqlite database").
Privilege levels therefore primarily gate **token-bearing API
principals** — and, if an authenticated admin HTTP API is later exposed
(decision point), gate that too.

**System privileges** are named, server-wide capabilities orthogonal to
levels and roles, grantable to a principal for special operations. The
first is **`IGNORE-QUOTAS`** (write into an over-quota namespace; see
*Quotas*); more can be added without disturbing the role/level model.

#### The privilege tree (roles as nesting aggregate classes)

Privilege is a **well-structured tree along two axes** — *actions* and
*scopes* — and the built-in roles are **aggregate shorthand classes that
imply everything they contain** on both axes.

**Action axis** — the base actions are `READ`→GET/HEAD, `WRITE`→PUT,
`DELETE`→DELETE, `ADMIN`→delegate bindings/ownership/TTL. The built-in
classes are a **strict nesting chain**, each implying all below it:

| Class | Implies | Actions |
|---|---|---|
| `read` | — | READ |
| `publish` | `read` | READ, WRITE |
| `maintain` | `publish` | READ, WRITE, DELETE |
| `curate` (a.k.a. `all`) | `maintain` | READ, WRITE, DELETE, ADMIN |

So granting `maintain` *is* granting `publish` and `read` — you name the
class, not the bag of actions. (Custom classes via
`vecd roles add NAME --actions …`; the four above are the canonical
shorthands.)

**Scope axis** — a class granted on a namespace **implies the same class
on the entire subtree beneath it** (bindings cover descendants). So
`curate @ team/` ⇒ curate over `team/**`. "Aggregate classes imply all
contained within their scope hierarchy" is exactly this: pick a class at a
node and it covers everything under that node — bounded, of course, by the
cone's ceilings (token, delegation, ancestry).

A privilege is thus a point `(class, scope)` in the tree that *expands*
down both axes; an **access profile** (used by tokens, below) is just a
selected set of such points.

#### Role bindings & ownership

- **role_bindings** — `(principal, role, namespace_path)`; principal is a
  user or a standard group (`PUBLIC`/`KNOWN`); the scope is a namespace
  path (`datasets/glove-100/`, or `""` for the whole server) and the
  binding applies to that subtree. A binding expands to its role's action
  set over that scope.
- **Ownership (governance) vs holding (write privilege)** — two distinct
  things, deliberately separated to avoid the owner-vs-holder ambiguity:
  - **Ownership is governance, and it inherits down the tree.** An
    object's owner defaults to its namespace's owner; a child namespace's
    owner defaults to its parent's — ownership **transits from the
    parent** unless explicitly set. An owner holds `curator` (full
    governance: bind roles, delete, set TTL) over what they own and may
    delegate within it.
  - An owner may be a **user or a system role** (e.g. `@admin`), so
    shared/infrastructure namespaces are governed *collectively*. A
    **system role** denotes the set of users at a given privilege level
    (`@admin` = all admins, `@operator` = all operators, …); membership
    follows level, so governance isn't pinned to one person.
  - **Holding is a granted privilege and never confers ownership.** A
    principal *granted* WRITE on an area (a "holder") may write there, but
    the objects they write are owned by the **inherited owner** (the
    namespace / role), not by the writer. Writing ≠ owning.
  - **Per-user private areas** come from giving a user their *own*
    namespace (`users/alice/` owned by `alice`), not from writes
    auto-claiming ownership.
- **Resolution** is the privilege cone defined just below (union of
  applicable bindings + ownership, intersected with the token and
  delegation ceilings); no explicit deny in v1 (a documented future
  extension).
- **No claim-on-first-write.** Ownership is assigned (namespaces, at
  `ns add`) and inherited (objects and child namespaces, from their
  parent) — never acquired by the act of writing. An explicit per-object
  owner override is possible but rare.

A plain "grant" is just binding a role (built-in or ad-hoc) to a
principal on a scope — the CLI keeps `grant`/`revoke` convenience verbs
that map `--read/--write/...` onto a role.

#### Standard groups: `PUBLIC` and `KNOWN`

Two built-in groups are always present and visible to every authenticated
user; you bind roles to them exactly like a user:

| Group | Who is in it | Auth required |
|---|---|---|
| `PUBLIC` | every caller, authenticated or not | none |
| `KNOWN` | every caller presenting a valid token | a valid token |

There is **no `PRIVATE` group** — privacy is the *default that falls out
of the cone*. A scope with nothing bound to `PUBLIC`/`KNOWN` is reachable
only by its owner and any specifically-bound users; "private" is simply
"nothing was opened up." Idiomatically:

- world-readable: `vecd bind --to PUBLIC --role reader --ns datasets/glove-100/`
- any logged-in user: `vecd bind --to KNOWN --role reader --ns datasets/internal/`
- private: bind nothing to `PUBLIC`/`KNOWN` — owner-only by default.

#### The privilege cone: union to collect, intersect to narrow

Privilege flows down from an owner (the apex) and is only ever *narrowed*
by delegation, tokens, and ancestry — never amplified. For a caller `C`,
key `K`, action `A`:

```
allowed(C, K, A)  ⇔
      A ∈ ( ⋃ actions of bindings that apply to C over prefixes covering K )   -- collect (union)
  AND A ∈ token_scope(C)                                                        -- narrow: token ceiling
  AND ( C reached via a group ⇒ that group is admitted on every ancestor of K ) -- narrow: ancestry ceiling
```

A binding *applies to* `C` if its principal is `C`'s user, or `KNOWN`
(and `C` is authenticated), or `PUBLIC`; the owner of a covering prefix
implicitly holds all actions (the apex). Three ceilings make it a cone:

- **Token ceiling:** a token may carry an action subset of its user's
  authority; the session is always intersected with it (a read-only token
  can never write, even for an owner).
- **Delegation ceiling:** a binding can only grant actions the *granting*
  principal itself holds on that scope — `ADMIN` can't manufacture
  privilege it lacks, so sub-delegations only narrow.
- **Ancestry ceiling (narrow-only openness):** a `PUBLIC`/`KNOWN` audience
  is admitted to `K` only if it is admitted on **every** covering
  ancestor — group openness is the *meet* down the tree. So a child can
  never be more open than its parent: a `reader→PUBLIC` binding under a
  closed parent is inert until the parent is opened too. (Explicit
  user/owner grants are not subject to this umbrella — it gates the
  *group* audiences.) A private parent therefore guarantees a private
  subtree.
- Within a principal's own applicable bindings it is a **union**
  (additive), but every union is bounded by the ceilings above.

Privacy needs no special case: nothing bound to `PUBLIC`/`KNOWN` ⇒ only
the owner is inside the cone ⇒ the resource is private. Management-plane
authority (privilege level) is a separate axis — it governs who may
*administer* and never widens data-plane actions.

#### Worked examples

1. **Public dataset, anonymous pull.** `reader→PUBLIC` on
   `datasets/glove-100/`. Anonymous `GET`: the `PUBLIC` binding applies to
   everyone ⇒ `{READ}`; no token ⇒ no narrowing → **allowed** (transparent
   public pull, unchanged read client).
2. **Known-only.** `reader→KNOWN` on `datasets/internal/`. Anonymous
   `GET`: no `PUBLIC` binding and the caller isn't `KNOWN` ⇒ no applicable
   actions → **404/401**. Alice with a token: `KNOWN` applies ⇒ `{READ}`
   → **allowed**.
3. **Private by default.** `users/alice/scratch/` owned by Alice, nothing
   bound to `PUBLIC`/`KNOWN`. Bob (authenticated): no applicable binding
   and not the owner → **404**. Alice: owner ⇒ all actions → **allowed**.
   No `PRIVATE` flag was needed.
4. **Delegation can't amplify.** The owner delegated only `publisher`
   (R,W) to Bob on `team/bob/`. Bob binding `maintainer` (which adds
   DELETE) to a teammate is capped by the delegation ceiling — he can't
   grant DELETE he doesn't hold, so the teammate's effective set excludes
   it.
5. **Token narrows below the user.** Alice owns `datasets/glove/` (all
   actions) and mints a **read-only** token. A push with it:
   `{R,W,D,ADMIN} ∩ token{R} = {R}` → the `PUT` is **403**, though
   Alice-the-user could write.
6. **Union collects, ceiling still bounds.** Bob has `reader→KNOWN` on
   `datasets/` and `publisher→bob` on `datasets/bobset/`. On
   `datasets/bobset/x` the union is `{READ,WRITE}`; with a read-only token
   the `∩{READ}` ceiling reduces the session to `{READ}`.
7. **Ancestry ceiling (narrow-only).** `datasets/` has no `PUBLIC`/`KNOWN`
   binding (closed); `datasets/glove/` has `reader→PUBLIC`. Anonymous
   `GET datasets/glove/x`: the `PUBLIC` audience isn't admitted on the
   `datasets/` ancestor, so the child binding is **inert** → **404**.
   Opening it requires a `PUBLIC` (or `KNOWN`) binding on `datasets/` too
   — a private parent keeps the whole subtree private. (An explicit
   `reader→alice` on the child *would* work — the umbrella gates groups,
   not named grants.)

### Data lifetimes — expire to *stasis*, never auto-delete

Expiry is **non-destructive**. A timer never physically removes data; it
moves it to **stasis** and hands the decision to an admin.

- A namespace may carry a **default TTL** (cascades). **The unit of
  lifecycle is the committed version** (the dataset snapshot — see *Upload
  sessions*): a version gets `expires_at = committed_at + ttl`. This keeps
  "wholly-uploaded dataset" and "expiry" aligned — a dataset expires as a
  coherent whole, not file-by-file.
- **On expiry → stasis.** When a version's `expires_at` passes, the
  sweeper sets `state = stasis` (records `stasis_at`): it becomes
  **invisible to clients** (the namespace no longer resolves it; `GET`/
  `HEAD` → `410 Gone`) while **all bytes and manifest rows are retained**.
  Nothing is deleted.
- **Admin cleanup queue.** Stasis versions surface as **pending cleanup
  tasks** (`vecd cleanup list`); per dataset the admin either:
  - **extend** — set a new lifecycle and restore it as the current version
    (`vecd cleanup extend <target> --duration D`), or
  - **purge** — physically delete the version's no-longer-referenced
    content (`vecd cleanup purge <target>`), the *only* path that removes
    bytes, always explicit. (Content shared with a live version via the
    content-addressed substructure is never purged.)
- **Lifecycle is advertised in headers.** A live version under a lifecycle
  limit serves **`X-Vecd-Expires: <rfc3339>`** + `X-Vecd-Lifecycle: live`;
  a stasis version answers `410` with `X-Vecd-Lifecycle: stasis`. Versions
  with no lifecycle limit carry no such header.
- Provenance objects (`.publish_url`, `pushlog.jsonl`) live inside the
  version like any other key, so they enter stasis with the dataset as a
  whole — never piecemeal; a live dataset never loses its history.

### Accounting (audit)

- **access_log** — `(ts, principal, token_desc, action, key, status,
  bytes, remote_addr)` for *every* request, including denials and auth
  failures (and, when acted via a delegated key, that key's mandatory
  description). Complements `pushlog.jsonl` (which records
  *writes/versions*); the access log records *all* access, incl. pulls.
- **Retention is bounded with optional cold archive.** Rows past a
  configurable window (default 90 days) or size cap are pruned; before
  pruning they can be **append-archived to the backup bucket**, so the DB
  (and its encrypted snapshots) stay bounded while full history survives
  in cold storage. Pruning is a best-effort write (it does not bump
  `auth_generation`).
- `vecd` can ingest a pushed `pushlog.jsonl` to attribute versions to
  owners for richer accounting — optional, post-v1.

#### Access-profile tokens & delegated keys

A token's **access profile** is the set of `(class, scope)` privileges it
may exercise — a selection over the privilege tree. By default a token's
profile is its user's full authority; minting one with a **deliberately
narrowed profile** is how "give someone a key scoped to exactly this"
works.

- **Delegated issuance.** Any principal may mint a token whose profile is
  a **subset of their own** authority (the delegation ceiling) with an
  expiry ≤ the max lifetime, and hand it to someone else; the holder then
  acts with exactly that profile. *Example:* Alice, who curates
  `datasets/`, issues a 7-day key with profile
  `{ read datasets/, publish datasets/scratch/ }` for a collaborator.
- **Effective authority = profile ∩ issuer's *live* authority.** The
  profile is a ceiling intersected with what the issuer holds *now* — so a
  key can never outgrow its issuer, and if Alice's access is later
  narrowed or revoked, every key she issued narrows with it (revocation is
  transitive, the cone holds).
- **Two issuance paths:**
  - admin, local-to-DB: `vecd tokens create --user NAME --profile "<sel>" --expires DUR`;
  - **delegated, over the API:** an authenticated `POST /tokens` (the
    issuer's own bearer credential) requests a profile ≤ its authority +
    an expiry and receives the new key **once** — surfaced client-side as
    `vectordata token issue --profile "<sel>" --expires DUR` (using your
    `login` session). `<sel>` uses the shorthand classes, e.g.
    `read datasets/glove, publish datasets/scratch`.
- Revoke with `vecd tokens revoke ID` (admin) or the issuer revoking their
  own over the API. `ping`/`whoami` shows a session's profile so a holder
  can see exactly what their key allows.

#### Named privilege profiles (reusable, parameterized)

An access profile can be **designed, named, and persisted** as a
reusable **privilege profile** — a template for minting tokens, so common
delegations aren't hand-specified each time.

- A profile is a named set of `(class, scope)` entries whose scopes may
  contain **interpolated placeholders** — *token positions* like
  `{dataset}` — filled in when a token is created from it.
- *Example:* profile `collaborator` =
  `read datasets/{dataset}/, publish datasets/{dataset}/scratch/`. Issuing
  `--from collaborator --set dataset=glove-100` expands it to a concrete
  profile over `datasets/glove-100/…`.
- Expansion is still **bounded by the issuer's live authority** (the
  delegation ceiling): a template can't grant what the issuer lacks;
  placeholders only *parameterize*, never widen.
- Profiles are persisted (`vecd profiles add`, or over the API by a
  privileged user) and consumed by `vecd tokens create --from …` /
  `vectordata token issue --from …`; the **concrete expansion** is what's
  stored on the issued token (and shown by `ping`/`whoami`).

## Namespaces — views, storage, and the access tree

A **namespace** is the single indirection that connects the *front-end
view* (the URL prefix a client addresses, and the access policy on it) to
the *back-end storage* (where the bytes actually live). Everything —
ownership, role bindings, TTL, and the storage backend — attaches to
namespaces. **A namespace path *is* a scope:** the `scope_prefix` used
throughout the AAA model is a namespace path.

### The tree

Namespaces form a `/`-delimited hierarchy (`""` is the root;
`datasets`, `datasets/glove-100`, `teamA/projX` …). Each node carries:

| Field | Meaning | Who sets it |
|---|---|---|
| `owner` | the **user or system role** that owns this subtree and may delegate within it | admin (at create / reassign) |
| `backend config` | reference to a named **backend config** (the storage connection), or **none** (config-only) | **admin only** |
| `active` | whether active storage is enabled here | **admin only** |
| `listable` | who may *see this namespace exists*: `public` / `known` / `grantees` | admin (owner may narrow) |
| `ttl` | default **version** lifetime here (cascades) | owner or admin |
| `quota_bytes` | storage cap (default 50 TB, cascades) | admin (owner may narrow) |
| role bindings | data-plane grants on this subtree (see AAA) | owner (within) / admin |

### Backend configs (named storage connections)

A namespace doesn't embed a raw storage URI — it **references a named
backend config**. A backend config is a first-class, reusable connection:
a `kind` (`local`/`s3`/`mem`), the physical `endpoint` (the `local:<dir>`
or `s3://bucket/prefix` extent), and the connection details to reach it
(`endpoint_url`, `region`, a credentials reference). This decouples *what
a namespace is* from *how its storage is reached* — configure a connection
once, point one or more namespaces at it, rotate credentials in one place.

**Exclusivity rule — one endpoint, one active config.** A given physical
`endpoint` may appear in **at most one *active* backend config** (a
partial-unique index `WHERE active`). Two active configs aliasing the same
bucket/dir would each believe they own it — racing on conditional writes,
ownership, and provenance, exactly the single-writer assumption `vecd`
relies on. Inactive configs may retain an endpoint (for standby/rotation),
but only one can be live at a time. Activating a second config on a
claimed endpoint is refused until the first is deactivated.

**Credentials.** A backend config references creds rather than inlining
secrets in the control-plane DB: an AWS profile / IAM role / env by
default. If a secret must be stored, it goes in the **encrypted config
file** (the same mechanism as the DB key — see *Configuration*), never
inline in the DB. (`local`/`mem` need none.)

### Storage mapping (incl. S3)

For a request on `/<path>/<rest>`, `vecd` resolves the **nearest
ancestor-or-self namespace that is `active` and references an active
backend config**, then stores/serves the object at
`config.endpoint + (key relative to that namespace)`:

- namespace `datasets` → config `vd-s3` (`s3://vd-bucket/ds/`, active):
  `GET /datasets/glove/base.fvec` is served from
  `s3://vd-bucket/ds/glove/base.fvec` — `vecd` applies AAA, then proxies
  to/from S3 with that config's credentials. `vecd` is an **AAA gateway in
  front of S3**.
- namespace `teamA` → config `team-local` (`local:/srv/vecd/teamA/`).
- The conditional-write contract (and push's probe) is upheld by **vecd's
  DB**, not the backend (see *Storage & schema*), so an S3 config works
  whether or not the store offers native conditional `PutObject`.

### Supported backends

All backends sit behind one `Backend` trait — `head`/`get`/`put`
(conditional)/`delete`/`etag`/`list` — so namespaces, AAA, and lifetimes
are backend-agnostic. The built-in set:

| Backend | URI | Notes |
|---|---|---|
| **Local filesystem** | `local:/dir` | atomic temp+rename; any mounted FS (incl. NFS) |
| **S3 / S3-compatible** | `s3://bucket/prefix` | one backend covers **AWS S3 *and* MinIO, Cloudflare R2, Backblaze B2, Wasabi, Ceph RGW, GCS's XML/S3 API** via a per-backend `--endpoint-url` + region (plain byte store; CAS is vecd's DB) |
| **Memory** | `mem:<id>` | in-process, **ephemeral — lost on restart** (never for durable data); needs a unique `<id>` to satisfy endpoint-exclusivity; for tests and short-lived/scratch namespaces |

"S3" here means *the S3 API*, so the one backend reaches the bulk of the
cloud-object-store landscape just by pointing `endpoint_url` at the
provider. Genuinely-different APIs are left as **extension points** on
the same trait, deliberately deferred:

- **native GCS / Azure Blob** — only needed for features their S3-compat
  facades lack; most deployments use the S3 path. *Deferred (decision #8).*
- **HTTPS passthrough / federation** — a backend that proxies to another
  `vecd` or a remote object store, enabling read-through mirrors and
  chained gateways. *Future.*

Backends are **plain byte stores** — they need not support conditional
writes. **vecd is the CAS authority**: each object's **content-key** (a
canonical descriptive-metadata hash — see *Hashing & addressing*) lives in
the transactional **version manifest** (`version_objects`), and vecd
enforces `If-Match`/`If-None-Match` against the staged manifest
(begin → check content-key → write bytes → record content-key → commit),
so the single-provenance guarantee holds over *any* backend, including
`mem` and stores with no conditional-write API. The price is that vecd
owns DB↔backend consistency on each write (same begin/commit discipline
push uses).

### Hierarchy: cascading defaults, and parents as real extents

- **Config cascades down.** `ttl`, `listable`, and role bindings set on a
  parent apply to descendants unless overridden. An **empty parent**
  (no backend, `active=false`) is exactly a *defaults container*: it holds
  no bytes but supplies inherited policy to the namespaces beneath it.
- **Parents can also be active extents.** A parent may itself have a
  backend and `active=true` and hold objects directly — being a container
  and a storage extent are independent. (Resolution still picks the
  nearest active backend, so a child with its own backend overrides the
  parent's for its subtree.)
- **Backend resolution and security narrowing are both ancestor-walks:**
  storage takes the *nearest* active backend; the privilege cone composes
  bindings down the same chain (union of grants, bounded by the ceilings).

### Ownership & delegation within a namespace

- The namespace **owner is the apex of the cone for that subtree** —
  and may be a **user or a system role** (a privilege-level group like
  `@admin`/`@operator`; named user-groups are a future addition), so
  shared subtrees are governed collectively. The owner holds all
  data-plane actions there, may bind roles to users/`PUBLIC`/`KNOWN`
  within it, set its `ttl`, and create **child namespaces** (config-only
  by default, inheriting the parent's backend config *and* its owner)
  and reassign ownership of those children within the subtree.
- **Ownership inherits; writing is granted.** Child namespaces and the
  objects within inherit the owner from their parent; a user who is merely
  *granted WRITE* (a holder) writes objects that remain owned by the
  inherited owner — never auto-claiming ownership.
- Delegation is **bounded to the subtree and to actions the owner holds**
  (the delegation ceiling) — an owner can neither escape their namespace
  nor manufacture privilege.
- **Admin/superuser retains the storage and discoverability levers**:
  assigning/changing the `backend config`, toggling `active`, reassigning
  a namespace `owner`, and setting `listable`/`quota` (who can see the
  namespace exists, and its storage cap). Rationale: backends provision real infrastructure (S3 buckets,
  disk) and discoverability is server policy — both above an owner's pay
  grade. An admin may delegate `listable`-narrowing to the owner.

### Visibility of namespaces

`listable` governs *discovery* (does the namespace appear in `ping`/list
output), independent of object-level access:

- `public` — anyone sees it exists; `known` — any authenticated user;
  `grantees` (default) — only principals with some binding/ownership in
  the subtree.
- An owner may *narrow* their namespace's listability but not widen it
  past what the admin set (the same narrowing-only rule as visibility).

### Per-user home namespaces

`vecd users add alice` **auto-provisions a home namespace** `users/alice/`
owned by `alice`, bound to a configured default backend config, private by
default (nothing bound to `PUBLIC`/`KNOWN`). This makes the per-user
privacy model turnkey — every user has a private area immediately, without
an admin hand-creating one. Controlled by a config setting (the default
home backend config) and a toggle to disable auto-provisioning.

### Quotas

Every active-storage namespace carries a **`quota_bytes`** (cascading;
admin-set, owner may narrow) with a **default of 50 TB** — large enough
that quotas are always *present* (so enforcement can be tightened
incrementally) without blocking normal use. Usage is the **sum of unique
stored content (deduped by content-key) across all *retained* versions**
of the subtree — so keeping version history consumes quota, and
`cleanup purge` of stasis versions reclaims it.

- A write that would push a namespace's usage **over quota** is refused
  with `507 Insufficient Storage` (+ `X-Vecd-Quota`).
- **Exception:** a principal holding the **`IGNORE-QUOTAS`** system
  privilege may still write into an over-quota namespace (for admin
  remediation / draining), audited like any write.

### Removing namespaces & moving backends

- **Removal cascades to stasis** (never an abrupt delete — matching the
  lifecycle philosophy): `vecd ns remove team/old` moves the namespace's
  versions to **stasis** (they show up in the cleanup queue) and retires
  the namespace record; bytes are removed only by an explicit
  `vecd cleanup purge team/old`. The removal is reversible until purge.
- **Changing the backend config triggers a background migration.**
  `vecd ns set team/ --backend-config new` copies the **current version's**
  content old→new as a background job; reads keep serving from the old
  backend until the copy finishes, then the namespace's backend pointer
  flips atomically (DB txn). An endpoint is released from its old config
  only once nothing references it (the one-endpoint-one-active-config
  rule). Progress shows in `vecd ns show`.

## Upload sessions & transactional publication

The **front-end logical name** a client addresses (`<namespace>/<key>`) is
**indirected** from the **physical extent** where the bytes actually land.
That indirection is what lets an upload present *transactionally*: a
downloader resolving a logical name only ever sees a **wholly-uploaded
version**, never a half-written one.

- **A version is the unit of publication.** Each namespace (dataset) has
  a **current committed version** — a manifest mapping logical keys →
  object content. Reads resolve `namespace → current version → key`, so a
  GET reflects exactly one committed version.
- **Uploads run in a session (the push `begin`).** A session **stages**
  its objects into a *new, not-yet-current* version — written to the
  backend but invisible to readers. Mid-upload, downloaders keep seeing
  the prior committed version.
- **Commit is an atomic pointer flip (the push `complete`).** In a single
  DB transaction (vecd is the CAS authority — A1), vecd publishes the new
  version (manifest = prior version + the session's adds/overwrites −
  deletes) and advances the namespace's **current-version pointer**.
  Before the txn: readers see the old version; after: the new one. There
  is no torn intermediate. This **closes the push "in-flux window"** — the
  pointer flip *is* the atomicity the push spec deferred to "version-
  prefixed publication + pointer flip."
- **Aborted/incomplete sessions never publish.** A session that never
  commits (crash, error) leaves a staged version that no pointer
  references; it's reaped like any orphan. Downloaders are unaffected.
- **Retention & rollback.** Prior committed versions are retained
  (subject to lifecycle/stasis), so an admin can **roll back** by
  re-pointing the namespace at an earlier version — and the version list
  *is* the dataset's history (aligned with `pushlog.jsonl`).

**How `vecd` demarcates a session (pushlog-driven).** `vecd` reads the
`pushlog.jsonl` events as the session signal — so plain push gets
transactional multi-object publication with **no client change** (the
`begin`/`complete` it already writes *are* the boundaries):

- a `PUT pushlog.jsonl` whose appended tail is a **`begin`** opens a
  **staging version** for that namespace (its `If-None-Match`/`If-Match`
  is evaluated against the *committed* pushlog, so the CAS / single-
  provenance guarantee still holds);
- subsequent object PUTs (and per-dir `SHA256SUMS`) **stage into that open
  version**, invisible to readers;
- the `PUT pushlog.jsonl` whose tail is the matching **`complete`**
  **commits** — the atomic pointer flip publishes the version;
- a write to a namespace with **no open session** (a lone raw PUT, no
  pushlog bracket) auto-commits as a one-object version, so non-push
  writers still work (without whole-dataset atomicity).

(An explicit session API for non-push clients is a possible future
addition — not required, since the pushlog already brackets sessions.)

**"Delete" is a manifest omission, not a physical erase.** push
`--delete` (orphan removal) is recorded as the new committed version's
manifest **omitting** those keys — a *logical* delete: prior versions
still reference the content, and the bytes are reclaimed only by
`cleanup purge`. So deletes are versioned and reversible like everything
else.

**Storage layout — logical name as the default extent, with a keyed
substructure, and version hashes in the tree.** A back-end path **mirrors
its logical front-end name by default** (`<backend>/<namespace-relative
key>/…`), and under each object name sits a **keyed substructure, present
in every case** — the content keyed by its **content-key** (a canonical
descriptive-metadata hash; see *Hashing & addressing* — **not** a
whole-content hash). So `datasets/glove/base.fvec` lives at
`<backend>/datasets/glove/base.fvec/<content-key>`. The bucket/dir stays
**human-navigable** (you can see which logical object an extent belongs
to), while the keyed layer lets **multiple versions of the same object
coexist** and lets the manifest + current-version pointer select the live
one; identical content across versions of an object dedups within that
object's substructure.

The **copy-on-write tree includes the version hash** as a first-class
node: each committed version's **manifest is stored under its version hash
(the `manifest_hash`)** — e.g. `<backend>/@v/<version-hash>/manifest` — so
versions are themselves hash-named, immutable nodes. A version is a
**manifest** of `logical key → content-key`; a session uploads only the
content-keys it changed and references existing ones for unchanged keys,
so a new version is a cheap new hash-node that shares prior content. Both
layers are thus hash-addressed — **content by content-key, versions by
version hash** — which is exactly what makes the tree copy-on-write,
dedup-friendly, and cleanly resumable. The current-version pointer (and
the canonical manifest) also live in vecd's DB, which is why commit is a
single atomic transaction.

### Version addressing & integrity

- **Every committed version is tagged and hashed.** It gets a monotonic
  `seq`, a **tag** (a user label like `v1.2`, else the default `v<seq>`),
  and a **`manifest_hash`** — a sha256 over the version's metadata (the
  sorted `key → content-key` manifest) — so a version is integrity-checked
  and citable as a whole.
- **The logical URI carries an optional version selector.** A dataset is
  addressed as `…/<namespace>/` ⇒ **`@latest`** (the current pointer), or
  pinned as `…/<namespace>/@<selector>/…` where `<selector>` is a `tag`,
  `v<seq>`, or a `manifest_hash` prefix. `latest` is a **reserved tag**
  that always resolves to `current_version`, giving a uniform
  always-newest access pattern.
- **Omitting the version means "latest, automatically."** Resolving a
  bare logical name accepts *latest-version* semantics (you get whatever
  is current); pin `@<tag>` for reproducibility. `GET`/`HEAD` and
  `ping`/`whoami` echo the resolved version (`X-Vecd-Version: v3` +
  `X-Vecd-Manifest: <hash>`).
- push assigns the tag (`push --tag v1.2`); pull/read selects it
  (`…/@v1.2/`) or omits for latest.

## Storage & schema

- **Content**: written through the resolved **namespace backend** (see
  *Namespaces*) at `<backend>/<logical key>/<content_key>` (the logical
  name mirrored, with the keyed content substructure beneath — see *Upload
  sessions*) — `local:<dir>` (atomic temp+rename, the I19/I20 lesson) or
  `s3://bucket/prefix` (vecd's own AWS creds). The DB and pidfile live
  under `<data_dir>`; `local` backends default under `<data_dir>/objects/`.
- **SQLite** via `rusqlite 0.31` (bundled), the workspace's established
  choice. Schema sketch:

```sql
meta(key PRIMARY KEY, value)                   -- schema_version, auth_generation (see Concurrency)
users(id, name UNIQUE, created, disabled, level, password_hash NULL)   -- level: user|operator|admin|superuser
tokens(id, user_id→users, token_hash UNIQUE, description, profile NULL, created, expires_at, last_used)
                                               -- description MANDATORY (shown at every usage point); user_id = the issuing principal it acts as;
                                               -- profile: selected (class,scope) set ≤ issuer authority (NULL = full authority); expires_at mandatory
roles(name PRIMARY KEY, actions, builtin)      -- actions: subset of READ|WRITE|DELETE|ADMIN
profiles(name PRIMARY KEY, owner, spec, created)   -- named privilege profile: (class,scope-with-{placeholders}) template; positions filled at token-create
system_privileges(principal, privilege, granted_by, created, PRIMARY KEY(principal, privilege))   -- e.g. IGNORE-QUOTAS
backends(name PRIMARY KEY, kind, endpoint, endpoint_url NULL, region NULL, creds_ref NULL, active, created)
                                               -- kind: local|s3|mem; partial UNIQUE(endpoint) WHERE active = 1
namespaces(path PRIMARY KEY, owner, backend_config→backends NULL, active, listable, quota_bytes, current_version→versions NULL, ttl_seconds NULL, created)
                                               -- owner: user id or system role (@admin); quota_bytes default 50 TB; current_version = published pointer (NULL until first commit)
versions(id PRIMARY KEY, namespace_path→namespaces, seq, tag, manifest_hash, state, created, committed_at NULL, expires_at NULL, stasis_at NULL)
                                               -- seq auto; tag UNIQUE per ns (default 'v<seq>'); manifest_hash = sha256 over version metadata; reads resolve current_version (= 'latest')
version_objects(version_id→versions, key, content_key, size, owner NULL, PRIMARY KEY(version_id, key))
                                               -- manifest: logical key → content_key (canonical descriptive-metadata hash); physical = <backend>/<key>/<content_key>; owner NULL = inherit ns owner
role_bindings(id, principal, role→roles, namespace_path→namespaces, created_by, created)  -- principal: user id | 'PUBLIC' | 'KNOWN'
access_log(id, ts, principal, token_desc NULL, action, key, status, bytes, remote_addr)   -- token_desc: the key's description, when acted via a delegated key
```

The DB runs in **WAL mode** with a `busy_timeout` (see *Concurrency*), so
the daemon and the local admin CLI can share it safely. Every
**control-plane write** — authz *and* namespace/backend/quota/profile/
privilege config — bumps `meta.auth_generation` in the same transaction;
that is the live-reload signal the daemon watches (see *Concurrency*).

## Concurrency — safe multi-process SQLite sharing & live reload

The daemon and the local admin CLI work on the same `vecd.db`
concurrently: the daemon reads authz on every request and appends to
`access_log` + version metadata, while the CLI mutates users/tokens/roles/
bindings/namespaces. The sharing convention is deliberately minimal —
no lock files, no bespoke IPC:

- **WAL + busy_timeout.** `PRAGMA journal_mode=WAL` gives many concurrent
  readers alongside a single writer; a `PRAGMA busy_timeout` (a few
  seconds) makes a writer briefly *wait* for the write lock instead of
  erroring. Every mutation is a transaction with `foreign_keys=ON`. That
  is the whole multi-process story.

- **Live reload — the daemon noticing admin changes without a restart.**
  The daemon holds an in-memory **control-plane snapshot** — everything a
  request needs to *authorize and route* but not the object content:
  users, tokens, roles, bindings, system privileges, profiles,
  **namespaces (incl. backend config, active, listable, quota,
  current-version pointer)**, and backend configs — so per-request
  decisions never hit the DB on the hot path. To stay current:
  1. every CLI write to **any of that control-plane config** bumps
     **`meta.auth_generation`** (a monotonic integer) *inside its
     transaction* — authz changes *and* namespace/backend/quota/profile
     changes alike, so a `vecd ns set --backend-config …` reloads routing
     just as `vecd bind …` reloads authz;
  2. the daemon cheaply polls **`PRAGMA data_version`** — a built-in that
     changes whenever *another connection* commits — on a ~1 s tick (and
     opportunistically). It's a single in-memory read, essentially free,
     and answers "did anyone write?" with no query;
  3. on a `data_version` change it reads `meta.auth_generation`; if that
     advanced, it reloads the snapshot and **atomically swaps** it in.

  So an admin's `vecd bind …` / `vecd ns set …` takes effect server-wide
  within ~1 s, with **zero hot-path DB cost** and no restart.
  `data_version` is the cheap gate, `auth_generation` the precise signal,
  the snapshot the fast path. Best-effort writes (`last_used`,
  `access_log`) deliberately do *not* bump `auth_generation`, so they
  never trigger a reload storm.

## Database durability & backup

The SQLite DB holds the entire **control plane** — users, tokens (hashed),
roles, namespaces, ownership, version metadata, the access log. (The
*data plane* — object content — is separately and independently
backup-able by a client; see *Introspection & off-system backup*. The two
together are a full backup.) The control-plane DB is the one piece of
state `vecd` alone owns, so it can **automatically back it up to a private
S3 bucket**:

- **Consistent snapshots, no downtime.** Backups use SQLite's **online
  backup API** (a page-by-page live copy, safe under WAL while the daemon
  keeps serving; `VACUUM INTO` is the one-shot equivalent) into a temp
  snapshot, which is then uploaded. No request is blocked.
- **Private destination**, separate from every namespace backend and
  never reachable through the front-end view:
  `vecd serve --db-backup s3://vecd-private/backups/ [--backup-interval 1h]
  [--backup-retain 24]`. Snapshots are timestamped; `--backup-retain`
  keeps the last N (or a time window) and prunes older ones.
- **Triggers:** the configured interval, clean shutdown, and on demand
  (`vecd backup now`) — driven by the same background task as the stasis
  sweeper.
- **Encryption — at the SQLite layer.** Rather than a bespoke
  encrypt-before-upload step, the control-plane DB is itself encrypted
  using a supported SQLite mechanism (the official **SQLite Encryption
  Extension (SEE)** or **SQLCipher**, to the extent SQLite / its official
  extensions support). The snapshot is therefore **encrypted by
  construction** — backups are just the already-encrypted DB file, so the
  bytes at rest in S3 are never readable without the key. The encryption
  key is read from the **vecd config file** (see *Configuration*), never
  passed on the command line; S3 SSE may still be layered on as
  defense-in-depth.
- **Restore** is explicit: `vecd restore <snapshot-uri> [--data-dir DIR]`
  fetches a snapshot and installs it as the active DB, refusing to clobber
  a running daemon (stop first). `vecd backup list` enumerates snapshots.
- The backup reuses the S3 backend's client/credential machinery but is
  **internal daemon state, not a namespace** — no view, no AAA binding,
  invisible to clients.

```
vecd backup now | list                               # manual snapshot / list
vecd restore <snapshot-uri> [--data-dir DIR]         # install a snapshot (daemon stopped)
```
plus `serve` flags `--db-backup`, `--backup-interval`, `--backup-retain`
(the DB encryption key comes from the config file; optional S3 SSE is a
config setting, not a headline flag).

## Introspection & off-system backup (client-driven)

The store's **layout and model are inspectable** by a sufficiently
privileged client, and the data structures are **self-consistent and
incremental by design** — so a `vectordata` client with enough access can
mirror the *entire* store off-system over the ordinary HTTPS API, with no
server-side cooperation beyond authorization. This is distinct from the
control-plane DB snapshot (above): that backs up *who-can-do-what*; this
backs up *the data and its structure*.

### Why the model already supports this

The versioned, content-addressed, manifest layout *is* an incremental,
verifiable export format:

- **Append-only versions.** Each namespace's versions have a monotonic
  `seq`; nothing is mutated in place, so "what's new since seq N" is a
  cheap, well-defined delta.
- **Content-addressed dedup.** Content is keyed by its **content-key**
  (a canonical descriptive-metadata hash); a client that already has a
  content-key never re-fetches it — incremental pulls move only
  genuinely-new bytes.
- **Self-verifying.** Each version carries a `manifest_hash` (a canonical
  metadata hash) and each object's content-key pins its descriptor; the
  bytes themselves are checked by **full content hashing via the merkle
  tree** (embedded in the descriptor) — so a mirror verifies both
  structure and bytes end-to-end without trusting the transport.
- **Consistent snapshots.** Because reads resolve a committed version (the
  pointer never exposes a partial one), an export of "version V" is always
  a coherent whole.

### Introspection API (privileged)

A principal sees structure to the extent it can read it; the physical
*layout* detail is gated to **privileged** callers (admin/operator, or a
read-broad backup principal):

- `GET /-/namespaces` — namespaces visible to the caller, with their
  config (owner, backend config, listable, quota, current version) to the
  extent allowed.
- `GET /-/versions/<ns>` — the version list (`seq`, `tag`, `manifest_hash`,
  `state`, `expires_at`), the append-only history.
- `GET /<ns>/@<ver>/-/manifest` — a version's **manifest**: every
  `key → content-key (canonical descriptive-metadata hash) + size`. This
  is the unit a backup walks.
- content is fetched by the normal `GET /<ns>/@<ver>/<key>`; an optional
  content-addressed `GET /-/blob/<content-key>` lets a mirror fetch a blob
  once and dedup across keys/versions/namespaces.

All of it is authorized by the same cone — **you can export exactly what
you can read** — so a "backup principal" is simply a user granted broad
`read` over the tree (or an admin/operator).

### The client-driven backup

`vectordata backup <url> --to <dest> [--since <version>] [--incremental]`
walks the introspection API and mirrors **content + manifests + version
metadata + namespace structure** (for the scope it can read) into `dest`
— a faithful, content-addressed, restorable copy. `dest` may be a local
`file://` layout or another `vecd`, so the mirror is directly restorable
by pushing it back. (Access *policy* — users/bindings/owners — travels in
the control-plane DB snapshot, not this data export; a restore
re-establishes datasets and an admin re-applies policy.)

**Resumable from a partial state (copy-on-write mirror + download-state
tracking).** A backup of a large store will be interrupted; resuming must
never re-fetch what's already local. The mirror is a **versioned,
copy-on-write, content-addressed tree** (the same shape as a `vecd`
backend): content blobs are write-once under their content-key, **each
version is an immutable node named by its version hash**
(`@v/<version-hash>`), versions are append-only, and a new version shares
unchanged content-keys with its predecessors — so a partial mirror is
always a coherent intermediate, not a corrupt one. A version-hash node
either exists complete or not at all, which is what makes resume clean. The client keeps **download-state tracking** so a re-run of
`vectordata backup` resumes automatically:

- **content-key presence = done.** A content-key already present *and
  verified* — its bytes pass **full content hashing via the merkle tree**
  (the merkle root embedded in the content-key's descriptor) — is skipped;
  this reuses vectordata's existing merkle-verified chunked download
  cache, so a *partially*-fetched large object resumes at **chunk
  granularity**, not from zero.
- **version commit is COW and last.** A mirrored version is marked
  **complete only once all its content-keys are present** (its manifest
  written last, like `vecd`'s own begin→stage→commit). So an interrupted
  backup leaves fully-present prior versions + one in-progress version;
  resume finishes the in-progress one and continues.
- **incremental diff.** Resume re-lists remote versions, compares against
  the mirror's completed-version set, and fetches only the missing
  versions' missing content-keys — `--since <version>` bounds it further.

So an off-system backup is **incremental, resumable, and self-verifying**,
driven wholly by the client.

```
vectordata backup  <url> --to <dest> [--incremental]   # mirror readable store off-system
vectordata restore <src> --to <url>                    # push a mirror back into a vecd
```

## HTTP server & request pipeline

`axum 0.8` + `tokio`. Middleware order per request:

```
log span → authenticate (token → principal) → authorize (action × key vs roles/owners)
         → handler (resolve version; object op: honor If-Match/If-None-Match, compute ETag) → access_log
```

- `authorize` maps method→action, applies the privilege cone (bindings +
  ownership, bounded by token/delegation/ancestry ceilings) for the key,
  and 401s (no/invalid token) or 403s (authenticated but unpermitted)
  early — before touching blobs.
- **Version resolution:** the path's version selector (`@<tag>`/`@v<seq>`/
  `@<hash>`, default `@latest`) picks the version; `GET`/`HEAD` echo the
  resolved `X-Vecd-Version: <tag>` + `X-Vecd-Manifest: <manifest_hash>`.
- **Lifecycle headers / stasis:** a live version under a lifecycle limit is
  served with `X-Vecd-Expires: <rfc3339>` + `X-Vecd-Lifecycle: live`; a
  version in stasis is **not** served — `GET`/`HEAD` → `410 Gone` with
  `X-Vecd-Lifecycle: stasis` — until an admin extends or purges it.
- The handler is the `objectstore.rs` mock grown up: real blob store,
  real ETags, conditional writes **honored** (passes push's probe).
- `GET /healthz` (unauthenticated) for `status`/liveness.
- `POST /auth/token` (unauthenticated) — exchange credentials
  (`{user, password}`) for a freshly minted bearer token; drives
  `vectordata login`. Only available for users with a `password_hash`
  set (`vecd users passwd`). Tokens it issues are recorded in `tokens`
  like any other.
- `POST /tokens` (authenticated) — **delegated key issuance**: an
  authenticated principal mints a token with an access profile ≤ its own
  authority and an expiry ≤ the max, returned **once**. Drives
  `vectordata token issue`. `DELETE /tokens/<id>` revokes a token the
  caller issued (or owns).
- `GET /-/whoami` (authenticated, or anonymous) — returns the caller's
  effective access: principal, level, and the role bindings / owned
  scopes that apply. Backs `vectordata ping <url>` (see *Endpoint
  capability probe*).
- **Listing:** `vecd` exposes a listing endpoint
  `GET /<prefix>?list` (authz: `READ` on the prefix) returning the keys +
  etags under it. Generic HTTPS still has no portable list verb, so a
  **`vecd`-aware client transport** uses this endpoint to enable push
  `--delete` (orphan removal) over `https://vecd-host/…`; against a
  non-`vecd` `https` endpoint, `--delete` remains unsupported as before.
- **Introspection / export:** `GET /-/namespaces`, `GET /-/versions/<ns>`,
  `GET /<ns>/@<ver>/-/manifest`, and content-addressed
  `GET /-/blob/<content-key>` expose the layout/model to a privileged
  reader for client-driven off-system backup (see *Introspection &
  off-system backup*). Authorized by the same cone — you export what you
  can read.

## Transparent push **and** pull

- **Push** works unchanged: bearer token + the conditional/ETag contract
  `vecd` honors.
- **Public pull** works unchanged: a `reader→PUBLIC` binding on the
  namespace means the existing (no-token) read path fetches transparently.
- **Private pull** needs the *read* client to present a token. The
  vectordata read transport is anonymous today, so the build adds
  read-side bearer auth resolved by origin from `--token`/
  `$VECTORDATA_TOKEN`/the `login` store (*decided, #4*; Phase 2). Public
  read keeps working with no token.

## Client credentials & login (`vectordata login`)

For push *and* pull to use `vecd` transparently — no `--token` on every
call — the vectordata client stores per-endpoint credentials locally.

```
vectordata login  <url> [--user NAME] [--token TOKEN]   # establish a local API key for an endpoint
vectordata logout <url>
vectordata whoami [<url>]                                # stored identity + capabilities (see ping)
vectordata login --list                                  # endpoints with stored credentials
vectordata token issue <url> --description "…" --expires DUR \
            ( --profile "<sel>" | --from PROFILE [--set pos=val …] )   # mint a delegated key (≤ your access) to hand to someone
vectordata token revoke <url> <id>                       # revoke a key you issued
```

- **Interactive (default):** `vectordata login https://vecd-host/` prompts
  for username + password (no echo), calls `POST /auth/token`, and stores
  the returned bearer token bound to the endpoint origin.
- **Pre-issued token:** `--token` (or paste when prompted) stores an
  admin-minted token (`vecd tokens create`) with no password exchange —
  for token-only deployments.

**Local credential store** — `~/.config/vectordata/credentials.toml`, mode
`0600`, keyed by endpoint origin:

```toml
["https://vecd-host"]
token   = "vd_…"          # bearer secret (file is 0600)
user    = "alice"         # informational
expires = "2026-…"        # if the token carries an expiry
```

**Token resolution (push and pull)** — first present wins:

1. explicit `--token` (push) / call-site override;
2. `$VECTORDATA_PUSH_TOKEN` (push) / `$VECTORDATA_TOKEN`;
3. the stored credential for the request's origin (`vectordata login`);
4. none → anonymous.

This closes the **private-pull** gap: the read transport gains the same
bearer-from-store lookup, so a logged-in user pulls private datasets
transparently while public datasets keep working anonymously. (That
read-side auth hook is the small vectordata-side change this requires.)

## Endpoint capability probe (`vectordata ping <url>`)

`vectordata ping <url>` — a datasource URL **with no dataset or profile** —
answers *"what does my access let me see and do here?"* It resolves the
stored/`--token` credential for the URL, calls `vecd`'s `GET /-/whoami`,
and prints the reflected authorization view:

```
$ vectordata ping https://vecd-host/
endpoint:  https://vecd-host/        (vecd 1.x)
identity:  alice                     level: user
visible namespaces:
  datasets/             KNOWN    read
  datasets/glove-100/   PUBLIC   read
  teamA/                (owner)  read,write,delete,admin
  users/alice/          (owner)  read,write,delete,admin
hidden from you:        3 namespaces
token scope:            read,write   (session narrowed below your full authority)
```

- Lists the **namespaces visible** to the caller (per `listable`), the
  **effective actions** in each (after the cone + token narrowing), and
  which are owned — the user-facing reflection of exactly what the server
  enforces.
- **Graceful fallback:** against a non-`vecd` endpoint (plain S3 / static
  host with no `/-/whoami`), `ping` degrades to a reachability +
  anonymous-read probe and says so. Useful against any datasource,
  richest against `vecd`.
- Distinct from `vectordata datasets ping name:profile` (which probes a
  specific dataset's facets); the bare-URL form probes *access*.

## CLI surface (with dynamic completions)

Mirrors the `vectordata` binary's `clap` + `clap_complete` dynamic-
completion idiom (a one-line `eval "$(vecd completions)"` wrapper; no
frozen script).

```
vecd init   [--data-dir DIR]                 # create DB + schema, mint first superuser token
vecd serve  [--bind ADDR] [--data-dir DIR] [--tls-cert F --tls-key F]
vecd start | stop | status | restart         # daemon lifecycle (pidfile under data-dir)

vecd users  add NAME [--level user|operator|admin|superuser] [--password]   # also auto-provisions home ns users/NAME/
            | list | level NAME LEVEL | passwd NAME | disable NAME | enable NAME | remove NAME
vecd tokens create --user NAME --description "…" --expires DUR
            ( --profile "read datasets/glove, publish datasets/scratch"      # ad-hoc (class,scope) subset, or
            | --from PROFILE [--set pos=val …] )                             # expand a named privilege profile, filling placeholders
            # description mandatory; profile ≤ user authority; --expires mandatory (≤ 365d)
            | list [--user NAME] | revoke ID
vecd profiles add NAME --spec "read datasets/{dataset}/, publish datasets/{dataset}/scratch/"   # named, persisted, parameterized
            | list | show NAME | remove NAME      # {placeholders} are interpolated at token-create time
vecd roles  list | show NAME | add NAME --actions read,write[,delete,admin] | remove NAME
vecd backends add NAME --kind s3|local|mem --endpoint s3://bucket/prefix|local:DIR
            [--endpoint-url URL] [--region R] [--aws-profile P] [--active]   # named storage connection
            | list | show NAME | set NAME [--active|--no-active|--endpoint-url …|--region …|--aws-profile …] | remove NAME
vecd ns     add PATH --owner NAME [--backend-config NAME] [--active] [--listable public|known|grantees] [--ttl DUR] [--quota SIZE]
            | list | show PATH | set PATH [--owner …|--backend-config …|--active|--no-active|--listable …|--ttl DUR|--quota SIZE] | remove PATH
vecd bind   --to NAME|PUBLIC|KNOWN --role ROLE --ns PATH      # data-plane RBAC binding on a namespace subtree
vecd unbind --to NAME|PUBLIC|KNOWN [--role ROLE] --ns PATH
vecd grant/revoke …                          # convenience aliases that map --read/--write/… to a role on --ns
vecd priv   grant IGNORE-QUOTAS --to NAME | revoke IGNORE-QUOTAS --to NAME   # system privileges
vecd cleanup list                            # pending tasks: objects/datasets in stasis (expired)
            | extend TARGET --duration D     # re-lifecycle + restore visibility
            | purge  TARGET                  # physical delete (the ONLY thing that removes bytes)
vecd log    [--tail N] [--user NAME] [--key K]
vecd completions [--shell SH]
```

Namespace `backend`/`active`/`listable`/`owner` edits are **admin-only**;
a namespace owner uses `ns add` (child, config-only), `bind`/`unbind`, and
`ns set --ttl`/`--listable` (narrowing) **within their own subtree**.

#### DB-backed dynamic completions

The `vecd` CLI uses the same `clap_complete` dynamic engine as the
`vectordata` binary (`eval "$(vecd completions)"`, no frozen script), but
its value completers **query the local SQLite DB at completion time** so
management is tab-driven against live state:

| Argument | Completes from |
|---|---|
| `--to`, `--user`, `NAME` | `users.name` (plus `PUBLIC`/`KNOWN` where a principal is accepted) |
| `--role ROLE` | `roles.name` |
| `--ns PATH` | `namespaces.path` (the namespace tree) |
| `--backend-config NAME` | `backends.name` (named storage connections) |
| `--from PROFILE` / `--set pos=` | `profiles.name`, then its placeholder positions |
| `--level LEVEL` | the privilege-level enum |
| `--endpoint` | `local:` / `s3://` scheme hints |
| `revoke ID` | the user's `tokens` ids/labels |

Each completer opens the DB **read-only** (a fast indexed scan) and is a
no-op when the DB is absent. Because the admin CLI is local-to-the-DB,
these run with the host operator's authority.

## Daemon lifecycle — **decided: self-daemonizing**

`vecd start` daemonizes itself: double-fork, `setsid`, redirect std fds
(to a log under the data-dir), and write `vecd.pid`. `vecd stop` sends
`SIGTERM` to the pid (graceful drain, then clean the pidfile); `vecd
status` checks pid liveness + `GET /healthz`; `restart` = stop+start.
`vecd serve` remains the foreground workhorse (what `start` execs into,
and what `systemd`/containers run directly). Adds a small daemonize step
(unix); Windows service support is out of scope for v1.

Original options, for the record:

- **A — self-daemonizing**: `start` double-forks, writes a pidfile;
  `stop` sends `SIGTERM`; `status` checks the pid + `/healthz`.
  Self-contained, classic Unix.
- **B — spawn-and-supervise (recommended)**: `serve` is the only
  long-running entry point; `start` spawns `vecd serve` as a detached
  child and records the pid; `stop`/`status` act on the pidfile. No
  daemonize crate, portable, and composes with `systemd`/`supervisord`
  (which just run `vecd serve`).

Recommendation: **B**, with `serve` as the real workhorse.

## TLS — **decided: terminate in `vecd` (rustls)**

`vecd` terminates TLS itself so it is a turnkey `https://` endpoint
(`rustls` via `axum-server`/`tokio-rustls`):

- `serve --tls-cert <pem> --tls-key <pem>` serves HTTPS; `--tls-self-signed`
  mints an ephemeral cert for dev.
- with no cert flags, `serve` listens **plain HTTP** — for local use or
  when an operator prefers to front it with a reverse proxy.

So a single binary covers both the standalone-`https` and
behind-a-proxy deployments; the cost accepted is the `rustls` dependency
and cert/key handling in `vecd`.

## Configuration

`vecd` reads a config file from a default config directory —
`$HOME/.config/vecd/` (overridable via `--config` / `$VECD_CONFIG`) —
holding settings an operator shouldn't pass on the command line:

- the **DB encryption key** (SEE/SQLCipher; see *Database durability &
  backup*);
- default `--data-dir`, bind address, and TLS cert/key paths;
- the `--db-backup` destination + its S3 credentials reference, interval,
  and retention.

CLI flags override the file; the config dir and any secret-bearing file
are created `0700`/`0600`. (`vecd init` writes a starter config.)

## Crate layout

`vecd` is its **own workspace crate** (`vecd/`, binary + lib), added to
the root `Cargo.toml` members. Dependency direction stays acyclic:
`veks → vectordata`, and `vecd → vectordata` (firm) `→ veks` (as needed)
— `vecd` is downstream of both, so no cycle.

```
vecd/Cargo.toml         # deps: axum 0.8, tokio (full), rusqlite 0.31 (bundled),
                        #       clap 4 + clap_complete (dynamic), sha2, hex, aws-sdk-s3, rustls,
                        #       vectordata (path) — reuse PUBLISH_FILE/PUSHLOG_FILE/CHECKSUMS_FILE
                        #       constants + pushlog parsing for provenance-awareness;
                        #       veks (path) — shared producer-side helpers (e.g. .publish_url /
                        #       check semantics) where it keeps server↔producer in lockstep
src/main.rs             # CLI dispatch + dynamic completions (mirrors bin/vectordata.rs)
src/db.rs               # SQLite schema + migrations + queries; WAL + auth_generation
src/auth.rs             # token hashing, authenticate(), principal resolution
src/authz.rs            # roles/bindings/owners/levels, the privilege cone, cached snapshot + live reload
src/namespace.rs        # namespace tree, resolution (backend + cascading config)
src/session.rs          # upload sessions, version manifests, atomic commit (pointer flip)
src/backend/mod.rs      # Backend trait (head/get/put-conditional/delete/etag/list)
src/backend/{local,s3,mem}.rs   # the built-in backends
src/server.rs           # axum app, middleware pipeline, handlers, /auth/token, /tokens, /-/whoami, ?list, /healthz, TLS
src/lifetime.rs         # TTL resolution + stasis sweep + cleanup (extend/purge); X-Vecd-Expires
src/backup.rs           # SQLite online-backup → private S3 (interval/retain/restore)
src/admin.rs            # users/tokens/roles/bind/ns/backends/profiles/priv/cleanup/log/backup commands
src/daemon.rs           # start/stop/status/pidfile (self-daemonizing)
```

`vecd` depends on `vectordata` for the sentinel-name constants and
`pushlog` parsing (keeping server and client in lockstep on the provenance
artifacts), and on `veks` for the producer-side helpers it shares — kept
to only what's actually used, so the server stays lean.

## Security considerations

- Tokens hashed at rest (SHA-256); plaintext shown once; constant-time
  compare on lookup. Every token expires (≤ 365 d). The control-plane DB
  itself is encrypted (SEE/SQLCipher; key in the config file).
- Authn/authz happen *before* any blob I/O; denials are logged.
- The CAS / single-provenance guarantee is enforced in **vecd's
  transactional DB** (not the backend), so it holds over any backend.
- **Rate limiting / abuse protection in vecd:** repeated failed auth (bad
  token/password) per source is throttled with exponential backoff →
  temporary block (logged); an optional per-principal request cap returns
  `429`. So vecd is safe to expose without relying solely on a fronting
  WAF (a proxy can still add its own layer).
- **Key normalization:** object keys are normalized and **confined to
  their namespace's backend extent** — `..`, absolute, or
  boundary-escaping keys are rejected, so one namespace can never read or
  write into another's storage.
- The DB, blob root, and config dir are `0700`/`0600`; the daemon holds
  the TLS private key + DB key in memory only. Backup snapshots are
  encrypted by construction (the DB is encrypted).
- Accounting is append-only; `vecd log` is read-only. Retention is
  bounded with optional cold archive (see *Accounting*).
- **Observability:** a `GET /metrics` (Prometheus) endpoint — request
  rates, auth failures, quota/stasis counts — is noted as a near-term
  add (Phase 2), distinct from the audit log.

## Decisions to confirm before building

1. ✅ **TLS** — *decided:* terminate in `vecd` (rustls), plain HTTP when no
   certs; deployable standalone or behind a proxy.
2. ✅ **Daemon model** — *decided:* self-daemonizing (`start` double-forks +
   pidfile; `serve` is the foreground workhorse).
3. ✅ **Ownership** — *decided:* ownership is governance that **inherits
   from the parent** (no claim-on-first-write); a namespace may be owned by
   a **system role**; **writing is a granted privilege that never confers
   ownership** (owner ≠ holder). Per-user privacy via per-user namespaces.
4. ✅ **Read-side auth** — *decided:* add bearer-token auth to the
   vectordata read transport now (resolves `--token`/`$VECTORDATA_TOKEN`/
   the `login` store by origin); enables private pulls + `ping`'s authed
   readout. Public datasets still pull anonymously.
5. ✅ **Nesting policy** — *decided:* **narrow-only**. A child can never be
   more open than its parent; a `PUBLIC`/`KNOWN` audience is admitted only
   where every ancestor admits it (the ancestry ceiling). A private parent
   guarantees a private subtree.
6. ✅ **CAS authority** — *decided (revises the earlier "refuse
   non-conforming backends"):* **vecd's transactional DB is the
   conditional-write authority**, not the backend. vecd records each
   object's content-key (a canonical descriptive-metadata hash) in the
   version manifest and enforces `If-Match`/`If-None-Match` inside the
   write transaction (the
   SQLite txn is the serialization point). Backends are therefore **plain byte stores —
   any backend works, none need conditional-write support**. vecd keeps
   DB↔backend consistent per write (begin→write-bytes→commit ordering,
   the same discipline push uses).
7. ✅ **`--delete` over `vecd`** — *decided:* expose a listing endpoint
   (`GET /<prefix>?list`) and a `vecd`-aware client transport so push
   `--delete` works over `https://vecd-host/…`; generic non-`vecd` https
   stays no-list.
8. ✅ **Backends** — *decided:* `local` + `s3`(-compatible) + `mem` in v1;
   native GCS/Azure and HTTPS passthrough/federation are deferred trait
   extensions.
9. ✅ **DB backup encryption** — *decided:* encrypt at the SQLite layer
   (SEE/SQLCipher, to the extent SQLite/official extensions support), key
   in the config file (`$HOME/.config/vecd/`); snapshots are encrypted by
   construction. SSE optional as defense-in-depth.
10. ✅ **Token lifetime** — *decided:* every token expires, **no
    exceptions**; default 90d, **hard max 365d** (config-tunable, max
    enforced). Service accounts rotate via `token issue`.
11. ✅ **Scope** — *decided:* **phased, core gateway first.**
    - **Phase 1** (an authenticated, conditional-write-honoring gateway
      push/pull work against end-to-end): `serve` + rustls, namespaces,
      backend configs (`local`/`s3`/`mem`), users + tokens (mandatory
      expiry + access profiles), roles/`bind`, the authn/authz cone,
      live reload, DB backup.
    - **Phase 2** (operability + client UX): stasis/`cleanup`, daemon
      `start`/`stop`/`status`, `log`, the listing + introspection
      endpoints (+ push `--delete`), and the client side —
      `vectordata login`/`ping`/`token issue` + read-side bearer auth, and
      the resumable client-driven `vectordata backup`/`restore`.

## Implementation status

**Phase 1 is built and green** (the `vecd/` workspace crate, bin + lib).
The request-handling core is **synchronous and runtime-free** — `db`,
`model`, `backend` (`local`/`mem`/`s3`), `authz` (the privilege cone),
`auth`, `namespace`, `store` (the DB-as-CAS-authority object store) — with
`server` a thin `axum`/`tokio` shell that marshals each request onto it via
`spawn_blocking`, plus `admin` (the control-plane CRUD library), `backup`
(`VACUUM INTO` → file/s3), `config`, and `cli`/`main` (clap + dynamic
completions). Live reload is wired (poll `PRAGMA data_version` →
`auth_generation` → atomic snapshot swap).

Delivered against this Phase 1 list: `serve` (+ rustls TLS path),
namespaces, backend configs, users + tokens (mandatory expiry + access
profiles + named parameterized profiles), roles + `bind`/`grant`, the
authn/authz cone (all seven worked examples are unit tests), live reload,
and DB backup. The acceptance criterion — the **real `vectordata push`
engine** pushing/pulling end-to-end against `vecd`, conditional-write probe
and begin→complete CAS chain included — is covered by an in-process
integration test (`tests/push_against_vecd.rs`) and a compiled-binary
integration test (`tests/cli_end_to_end.rs`).

The object identity / ETag is the **content-key** per the terminology
invariant — a sha256 over a canonical descriptor `{size, full-content
merkle root}`, **not** a whole-content hash — so it is content-determining
and dedup-ready while remaining a metadata hash (a unit test pins that it
is *not* `sha256(bytes)`). The one Phase-1 simplification, called out
honestly: `objects` is a **flat per-namespace manifest**, so the COW
`versions`/`version_objects` tree, tags, and `@latest`/`@tag` addressing
land in Phase 2. The `s3` backend, the TLS path, and SQLCipher encryption
are implemented but not yet integration-tested.

**Phase 2 progress.** The server-side surface is now substantially
complete and tested (67 tests):

- **Introspection/listing** — `GET /-/whoami`, `GET /-/namespaces`
  (operator-gated), `GET /<prefix>?list` (server side of push `--delete`).
- **Daemon lifecycle** — self-daemonizing `start`/`stop`/`status`/
  `restart` (pidfile, `vecd.addr`), draining on SIGINT/SIGTERM.
- **COW version tree + transactional sessions** — the pushlog `begin`
  opens a staging manifest (COW-initialised from the live manifest, begin
  `deletes` applied), object PUTs stage invisibly to readers, and
  `complete` commits in one transaction (atomic pointer flip + immutable
  version snapshot). This closes push's in-flux window with no client
  change. Writes outside a session mutate the live manifest with per-object
  CAS (so the conditional-write probe never pollutes version history).
- **Version addressing + introspection** — `@latest`/`@v<seq>`/`@<tag>`/
  `@<hash>` selectors, `X-Vecd-Version`/`X-Vecd-Manifest` headers,
  `GET /-/versions/<ns>`, `GET /<ns>/@<ver>/-/manifest`.
- **Stasis lifecycle** — a background sweeper moves expired committed
  versions to stasis (non-destructive, rolling each namespace back to its
  newest survivor); `410 Gone` + `X-Vecd-Lifecycle` headers; `vecd cleanup
  list`/`extend`/`purge` (purge the only path that removes bytes,
  respecting content-key sharing).
- **Delegated token API** — `POST /auth/token` (password grant, drives
  `vectordata login`), `POST /tokens` (delegated profile keys, drives
  `vectordata token issue`), `DELETE /tokens/<id>`; a freshly minted key is
  usable immediately (synchronous snapshot reload).
- **Abuse throttle + metrics** — per-source auth-failure backoff → `429`;
  `GET /metrics` (Prometheus counters).

**Client side (`vectordata` crate) — built and tested, additively.** Every
addition kept the frozen contract green (see *Client contract* above):

- **Per-origin credential store** (`credentials.rs`) +
  `vectordata login`/`logout`/`whoami`/`login --list` over `POST
  /auth/token`; token resolution precedence (`--token`/env/store/none).
- **`vectordata ping <url>`** over `GET /-/whoami`, with graceful fallback
  on non-`vecd` hosts.
- **`vectordata token issue`/`revoke`** over `POST /tokens` / `DELETE
  /tokens/<id>`.
- **Read-side bearer auth** — the read transport attaches a token resolved
  by origin; private pulls work, and public/anonymous reads are unchanged
  (proven by the frozen `http_storage.rs` plus a dedicated private-pull
  test through the public reader API).
- **push `--delete` over vecd** — the push `https` transport's `list()`
  uses `GET <prefix>?list`, enabling orphan removal against a `vecd`
  endpoint while staying unsupported against a generic host.
- **Resumable `backup`/`restore`** — `backup` mirrors readable namespaces →
  versions → manifests → content into a content-addressed COW tree
  (`blobs/<content-key>` write-once; per-version `manifest.json` written
  last as the completion marker), resumable and incremental; `restore`
  republishes a mirror's latest state into a target `vecd`. A cross-store
  round-trip test (A → mirror → fresh B) passes.

Still outstanding: the content-addressed `GET /-/blob/<content-key>`
shortcut (a global content-key index — `backup` currently fetches by
`@<ver>/<key>`, which works without it); full version-history *restore*
replay (latest-state restore is implemented); and integration coverage for
the `s3` backend, TLS, and SQLCipher (all implemented, exercised only by
unit/build paths).

## Client contract — stabilized tests & additive integration

The `vectordata`-side support for `vecd` is built **incrementally and
additively**: every new client behavior layers onto a `vecd` endpoint that
is already implemented and tested, and the existing client↔endpoint
contract is **frozen** — the tests below must stay green through every
addition. New behavior is opt-in and never changes an existing call site
or the anonymous defaults.

**Frozen contract (must not regress):**

- **REST object contract push relies on** — `vectordata/tests/push_https.rs`:
  `https_transport_verbs_and_conditional_put`, `push_over_https_end_to_end`,
  `https_auth_failure_then_success_with_token`,
  `push_refuses_endpoint_that_ignores_conditional_writes`.
- **push ↔ vecd** — `vecd/tests/push_against_vecd.rs`:
  `push_succeeds_and_is_retrievable`, `push_without_token_is_rejected`,
  `read_only_token_cannot_push`, and (the read-path invariant)
  **`public_binding_enables_anonymous_pull`**; plus
  `vecd/tests/cli_end_to_end.rs::cli_init_serve_push_pull`.
- **Read/pull path (anonymous)** — `vectordata/tests/http_storage.rs` (the
  cached/chunked/merkle-verified read stack) and
  `vectordata/tests/chunked_http_stress.rs` (chunk-resumable download). These
  are the guardrail that **read-side auth stays opt-in**: public reads keep
  working with no token, and the resume machinery the client backup reuses
  does not regress. (`catalog_knn_entries_fallback.rs` covers HTTP catalog
  discovery on the same path.)
- **Push engine internals** — the `vectordata` `push::` unit tests
  (add/skip/overwrite, resume, abort, CAS probe, SHA256SUMS oracle); the
  `--delete` work must not regress them.

**Additive integration plan** — each row ships with its own incremental
test and re-runs the frozen set green:

| New `vectordata` behavior | Builds on (vecd, already tested) | Guardrail preserved |
|---|---|---|
| `login`/`logout`/`whoami`/`--list` + per-origin credential store | `POST /auth/token` | (new `~/.config/vectordata` file) |
| `ping <url>` | `GET /-/whoami` | graceful fallback on non-vecd hosts |
| `token issue`/`revoke` | `POST /tokens`, `DELETE /tokens/<id>` | — |
| **read-side bearer auth** (resolve token by origin) | private namespaces | `public_binding_enables_anonymous_pull` + all of `http_storage.rs` |
| push `--delete` over vecd | `GET /<prefix>?list` | the `push::` engine tests |
| `backup`/`restore` (resumable mirror) | `/-/versions`, `/-/manifest`, `@<ver>` reads | `chunked_http_stress.rs` |

The only change touching an existing hot path is the read-side auth hook;
its guardrail (`http_storage.rs`, anonymous reads) is exactly what forces it
to remain additive — a token is resolved by request origin, and its absence
falls back to anonymous, unchanged.

**Token resolution precedence** (push and pull), first present wins:
explicit `--token` / call-site override → `$VECTORDATA_PUSH_TOKEN`
(push) / `$VECTORDATA_TOKEN` → the stored credential for the request origin
(`vectordata login`) → none (anonymous).

## Design-review resolutions

A structured review surfaced these tensions, gaps, and missing essentials
beyond the numbered decisions; all are now folded into the spec:

- **A1 CAS authority** → vecd's transactional DB (not the backend); any
  backend works.
- **A2 delegated-key identity** → a *capability* bound to the issuer, with
  a **mandatory description shown at every usage point**; the holder need
  not be a known principal.
- **A3 live reload** → the hot-reloaded snapshot covers *all* control-plane
  config (authz + namespace/backend/quota/profile), each bumping
  `auth_generation`.
- **A4 writer attribution** → the access log (principal + key description)
  plus the version a write belongs to provide it; no extra field.
- **B1 namespace removal** → cascades to stasis (reversible until purge).
- **B2 backend change** → background migration, then atomic pointer flip.
- **B3 `mem`** → unique `mem:<id>`, ephemeral (never durable).
- **B5 key normalization** → keys confined to their namespace's extent.
- **B6 terminology** → `@role` owners denote level-groups, distinct from
  action-roles.
- **C1 home namespaces** → auto-provisioned per user.
- **C2 quotas** → per-namespace, 50 TB default; over-quota needs
  `IGNORE-QUOTAS`.
- **C3 rate limiting** → in-vecd auth-failure throttling + optional req
  caps.
- **C4 audit retention** → bounded window + optional cold archive.
- **C5 HA** → single-host v1 (SQLite); Postgres-backed multi-instance is
  future.
- **C6 observability** → `GET /metrics` (Prometheus) as a Phase-2 add.

Structural additions made *during* review: **named backend configs** with
one-endpoint-one-active-config exclusivity; **transactional upload
sessions** (logical↔physical indirection, atomic pointer-flip publication,
content-addressed keyed substructure beneath the mirrored logical name);
**versioning** (tags + `manifest_hash`, `@latest`/`@tag` addressing,
latest-by-default); and **named parameterized privilege profiles**
(`{placeholder}` token positions).

## Footnote: build vs. buy — why build `vecd`

Before committing to a new daemon we surveyed the strongest open-source,
freely-available systems that might cover this functional surface
off-the-shelf. The conclusion: **no single system covers it, and the
gaps fall in the load-bearing middle, not the edges.** The honest posture
is *build the control plane, buy the backend* — vecd's `kind: s3|local|mem`
abstraction already "buys" the commodity byte-storage tier; what it builds
is the thin AAA + namespace + transactional-versioning plane that the
existing client already speaks natively.

**The discriminating requirement surface** (the axes that matter):

- **R1** REST object gateway honoring conditional writes (`If-Match` /
  `If-None-Match`) — `push`'s single-provenance CAS guarantee rides on it.
- **R2** namespace→backend indirection (hierarchical, cascading config,
  one-endpoint-one-config).
- **R3** bearer-token RBAC *cone* — privilege tree (read⊂publish⊂maintain⊂
  curate), levels, roles, `PUBLIC`/`KNOWN` groups, ownership, intersection
  narrowing.
- **R4** delegated expiring scoped keys + parameterized profiles
  (`{placeholder}` token positions).
- **R5** transactional versioned publication (pushlog-driven session,
  atomic pointer-flip).
- **R6** metadata-hash manifests + version tags + `@latest`/`@tag`, COW
  tree keyed by version hash.
- **R7** non-destructive stasis lifecycle (expire→hidden, admin
  extend/purge, `X-`header signal).
- **R8** hierarchical quotas + `IGNORE-QUOTAS` system privilege.
- **R9** single-binary SQLite ops, multi-process safe, live config reload.
- **R10** client-driven incremental/resumable/self-verifying off-system
  backup + object-store introspection.
- **R11** native fit with the existing push/pull + pushlog/SHA256SUMS
  provenance (no protocol impedance).

**The five strongest contenders, and where each falls down:**

1. **lakeFS** (Apache-2.0, Go) — git-for-data: atomic commits, branches,
   merges, zero-copy branching over S3/GCS/Azure, *plus* an S3 gateway.
   Closest match on **R5/R6**. But granular RBAC is a lakeFS-Cloud /
   enterprise feature — OSS auth is thin (no delegated scoped expiring
   keys, no privilege cone, no profile placeholders → misses **R3/R4**);
   deployment needs Go server **+ an external KV store** (Postgres/Dynamo),
   so no single-binary-SQLite story (**R9**); no stasis lifecycle (**R7**)
   or per-namespace backend indirection (**R2**); its versioning model is
   its own, so coupling to pushlog/SHA256SUMS provenance is impedance, not
   fit (**R11**). *The one tempting "buy" — lakeFS as the versioning
   substrate — still loses: we'd use almost none of its branch/merge
   surface (our model is immutable, tagged, atomically-published
   snapshots) while inheriting all of its operational weight and bypassing
   its auth entirely. Impedance cost > implementation cost.*
2. **Harbor / OCI registries (Zot)** (CNCF, Apache-2.0) — content-addressed
   blobs, manifests, **tags**, project=namespace, **RBAC**, **per-project
   quotas**, **retention**, replication: strong on **R3/R6/R8** and partial
   **R7**. Disqualifier is the **protocol** — clients must speak the OCI
   distribution protocol, not plain REST `PUT`/`GET`; it does not front
   arbitrary S3 backends as namespaces (**R2**), gives no whole-tree atomic
   publication (**R5** partial), and has zero native fit with push/pull
   (**R11**). Adopting it means reframing every dataset as an OCI artifact
   and rewriting the client transport.
3. **MinIO** (**AGPLv3** — licensing friction for a tool we ship, Go) — S3
   server with IAM policies, bucket versioning, ILM expiry. But ILM
   *deletes* (our stasis never deletes — **R7**); gateway mode was removed,
   so no namespace→backend indirection (**R2**); no transactional dataset
   publication (**R5**), no delegated profile tokens (**R4**), no pushlog
   fit (**R11**). Best read as a **candidate backend behind vecd**, not a
   replacement for it.
4. **S3 gateways — versitygw / s3proxy / Zenko CloudServer** — the only
   contenders that natively do **R2** (versitygw even keeps per-bucket
   metadata in a `--meta-bucket`). But auth is S3 SigV4 access/secret keys,
   not a bearer-token RBAC cone (**R3/R4**), and none do versioning,
   manifests, tags, stasis, or quotas (**R5–R8**). They solve exactly one
   of eleven axes.
5. **Lightweight S3 stores — SeaweedFS (Apache-2.0) / Garage (Rust, single
   binary) / Ceph RGW** — raw S3 with access-key auth (**R1** only). Garage
   is the operational cousin of what we want (Rust, single binary, ~512 MB,
   zero deps) but *explicitly lacks versioning, object lock, and
   lifecycle*. Like MinIO, these are **backend candidates**, not
   control-plane replacements.

*(Out-of-class and rejected: DVC = client-side git+remote, no server AAA;
Pachyderm = needs Kubernetes; Quilt = AWS-centric catalog.)*

**Coverage matrix** (✓ full · ◑ partial/conditional · ✗ none):

| | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **lakeFS** | ◑ | ✗ | ◑¹ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ◑ | ✗ |
| **Harbor/Zot** | ✗² | ✗ | ✓ | ◑ | ◑ | ✓ | ◑ | ✓ | ✗ | ✗ | ✗ |
| **MinIO** | ✓ | ✗ | ◑ | ◑ | ✗ | ◑ | ◑³ | ◑ | ✗ | ✗ | ✗ |
| **versitygw/s3proxy** | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◑ | ✗ | ✗ |
| **SeaweedFS/Garage** | ◑⁴ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◑ | ✗ | ✗ |
| **`vecd` (proposed)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

¹ enterprise/Cloud only · ² OCI protocol, not REST · ³ ILM deletes, not
stasis · ⁴ Garage has no versioning.

**Verdict.** The field splits into *backends* (MinIO, SeaweedFS, Garage,
Ceph — R1 and nothing above it; already "bought" as vecd backends) and
*partial control planes* that each nail one cluster and structurally miss
the rest (lakeFS owns versioning but not AAA-or-ops; Harbor owns
content-addressing+RBAC+quota but speaks the wrong protocol and can't front
S3 namespaces; the gateways own backend-indirection alone). The combination
that *defines* vecd — bearer-token RBAC cone + delegated expiring
profile-tokens + namespace→backend indirection + pushlog-native
transactional versioning addressable as `@latest`/`@tag` + non-destructive
stasis + single-binary-SQLite ops + zero-impedance fit with the existing
push/pull/SHA256SUMS provenance — exists in no off-the-shelf system. To
"buy" it you would assemble *versitygw (R2) + lakeFS (R5/R6) + an external
IdP/OPA (R3/R4) + custom lifecycle glue (R7) + custom backup tooling (R10)*
and **still** not get **R11** — the seam where the assembled stack would
leak. So: **buy the backend, build the control plane**, because the closest
adoptable substitute brings more integration surface than the control plane
itself.

*Sources surveyed: [lakeFS](https://github.com/treeverse/lakeFS) /
[docs](https://docs.lakefs.io/); [Harbor](https://github.com/goharbor/harbor)
and the [OCI distribution spec](https://github.com/opencontainers/distribution-spec/blob/main/spec.md);
MinIO [versioning](https://github.com/minio/minio/blob/master/docs/bucket/versioning/README.md)
and [ILM](https://github.com/minio/minio/blob/master/docs/bucket/lifecycle/README.md);
[versitygw](https://github.com/versity/versitygw/wiki/S3-Backend),
[s3proxy](https://github.com/gaul/s3proxy),
[Zenko CloudServer](https://github.com/scality/cloudserver);
[SeaweedFS](https://github.com/seaweedfs/seaweedfs) and
[Garage](https://rilavek.com/resources/self-hosted-s3-compatible-object-storage-2026);
[DVC/Pachyderm](https://www.pachyderm.com/blog/data-versioning-comparing-dvc-with-pachyderm/).*
