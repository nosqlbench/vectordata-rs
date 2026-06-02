# `vectordata push` — design draft

**Status:** draft / proposal. Nothing here is implemented yet. This document
describes the command we want, why it is shaped the way it is, and how it rides
on the abstractions that already exist (the read-side `Storage`/transport
factoring, the `.publish_url` binding, and the `veks publish` flow) without
duplicating them or confusing users about which verb to reach for.

## What problem this solves

`vectordata` is the *consumer* CLI: it reads, describes, caches, and explores
datasets that live behind a URL. Today the only way to put a dataset *up* at a
URL is `veks publish` — the producer toolkit's full-directory S3 sync, which
runs the whole build/check suite, can delete remote objects, and sweeps an
entire publish root that may hold many datasets.

That is the right tool when you are *producing* a dataset. It is too much tool
when you already have a dataset in a known-good state and just want to make it
reachable. `vectordata push` is that low-effort verb:

> "I have a good dataset (or a directory of files) right here. Put it at this
> remote so vectordata can read it later, over whatever transport the URL
> implies, and don't let me clobber someone else's data by accident."

The design goal is *low ceremony for the happy path, hard stops for the
dangerous ones.*

## Positioning vs. `veks publish` (so users aren't confused)

| | `veks publish` | `vectordata push` |
|---|---|---|
| Lives in | `veks` (producer toolkit) | `vectordata` (consumer CLI) |
| Scope | a whole publish root, many datasets | one already-good dataset, or one ad-hoc dir |
| Builds / regenerates | yes (part of the produce loop) | never — push only moves bytes |
| Validation | full check suite | confirm "known-good", then move |
| Transports | S3 only (AWS CLI) | scheme-dispatched: `s3://`, `https://`, `file://` |
| Can delete remote | `--delete` | no (push is additive) |
| Overwrite policy | timestamp/size sync | refused unless an audit message is supplied |

They deliberately **share** two things so the mental model stays single:

1. The **`.publish_url` binding** (the file that ties local data to a remote
   endpoint), and
2. The **known-good check semantics**.

The intended end state is that `veks publish` becomes "build + check + `push`
the whole root", delegating the byte-moving and the binding/audit rules to the
machinery defined here. That consolidation is out of scope for the first cut
but the interfaces below are drawn so it is a later refactor, not a rewrite.

## Command surface

```
vectordata push [PATH] [--to URL] [-m MESSAGE] [--raw] [--dry-run] [options]
```

`PATH` defaults to `.`. CLI flags and the on-disk `.publish_url` /
`dataset.yaml` keys are congruent mirrors — every knob reachable from the
command line is also expressible in the persisted files, and vice versa
(see *Binding* and *Auth*).

| Flag | Mirror | Meaning |
|---|---|---|
| `[PATH]` | — | dataset dir, catalog dir, or ad-hoc dir to push (default `.`) |
| `--to URL` | `.publish_url` | target endpoint; if both present they must agree |
| `-m, --message TEXT` | remote `pushlog.jsonl` | required iff the push would overwrite remote bytes |
| `--raw` | — | ad-hoc mode: push every file verbatim, no shape validation |
| `--checksums MODE` | `SHA256SUMS` | `auto` (default: recompute if stale) or `keep` (use existing, fail if stale) |
| `--dry-run` | — | resolve, validate, and print the full plan; touch nothing |
| `--profile NAME` | `AWS_PROFILE` | AWS profile for S3 credentials |
| `--endpoint-url URL` | `AWS_ENDPOINT_URL` | S3-compatible endpoint override |
| `--token TOKEN` | `VECTORDATA_PUSH_TOKEN` | bearer token for generic `https://` |
| `--concurrency N` | `VECTORDATA_HTTP_RUNTIMES` | parallel upload streams (default 4) |
| `--no-check` | — | skip known-good validation (discouraged; never skips binding/overwrite rules) |
| `-y, --yes` | — | skip the interactive confirmation |

Non-goals for the first cut (call them out so they aren't mistaken for bugs):
remote deletion, partial/range re-upload of a changed file (we re-put whole
objects), and signed *download* URLs (read access stays anonymous/public as it
is today).

## The three source modes

`push` accepts exactly three shapes of source. The mode is auto-detected from
the directory; the only time the user must say `--raw` is the ad-hoc case,
because we refuse to silently ship unstructured bytes.

### 1. Structured dataset — `dataset.yaml` present

The strongest mode. "Known-good" means, reusing `veks check` semantics:

- `dataset.yaml` parses and resolves (`DatasetConfig::load_and_resolve`);
- required publishability attributes are present and true:
  `is_zero_vector_free`, `is_duplicate_vector_free`;
- every facet file declared by every profile exists on disk;
- each facet's merkle sidecar (`*.mrkl`) is present and verifies against the
  file — this is what lets the read side serve the data with chunk
  verification, and what we reuse for cheap overwrite detection (below);
- each directory level has a current `SHA256SUMS` (or one is generated under
  `--checksums auto`) — see *Content checksums*.

### 2. Catalog map — `knn_entries.yaml` present

The legacy flat `"dataset:profile" → {facet: path}` map. "Known-good" is the
lighter check: every referenced facet path exists and opens as a valid
xvec/ivec. No attribute requirements (the format predates them). Merkle
sidecars are generated on the fly if absent so the pushed copy is
read-verifiable.

### 3. Ad-hoc directory — `--raw`

Push every regular file under `PATH` verbatim, preserving the relative tree.
No shape validation, no facet model — for arbitrary blobs (model files,
notes, derived artifacts) that should live next to a dataset. The binding,
overwrite-gating, and audit-log rules still apply in full; `--raw` relaxes
*shape* checks, never *safety* checks.

## Binding: `.publish_url` is the contract

The `.publish_url` file (already used by `veks`; see
`veks/src/check/publish_url.rs`) is the single source of truth for "where does
this data belong." `push` both **honors** and **persists** it.

```
# <source>/.publish_url
s3://my-bucket/datasets/glove-100/
```

Resolution rules:

- **No `.publish_url`, no `--to`** → error. We refuse to invent a destination.
  The error tells the user exactly how to bind:
  `echo 'https://host/path/' > .publish_url`.
- **`.publish_url` present, no `--to`** → use it.
- **`--to` present, no `.publish_url`** → use `--to` and *write*
  `.publish_url` into the source so the binding is persisted with the data.
- **Both present and they disagree** → **conflict, hard stop.** The local data
  is already bound to a different endpoint; re-binding is a deliberate act, not
  a flag the user trips over. (A future `vectordata rebind` can own that.)

This generalizes the existing binding in two ways the current `veks` code does
not yet support, both required by this command:

1. **Scheme set.** `SUPPORTED_SCHEMES` grows from `["s3"]` to
   `["s3", "https", "http", "file"]`. The parser/validator
   (`parse_publish_url`, `find_publish_file`) is otherwise reused unchanged.
2. **Shared home.** The binding code moves to a place both crates depend on
   (a small shared module / the `vectordata` crate) so `veks publish` and
   `vectordata push` cannot drift in how they read the same file. Until that
   move lands, `vectordata` re-exports the `veks` functions.

### Remote-side conflict detection

The user's brief: *"a user who tries to push into a path which already has a
`.publish_url` should be told when there is a conflict."*

Every publish root we write carries a **self-describing** `.publish_url` on the
remote (its own canonical URL) plus an identity marker — for a structured
dataset that is the `name:` from `dataset.yaml`; for catalog/ad-hoc it is a
content-derived id stamped at first push. Before transferring, `push` does a
cheap remote read of the target's `.publish_url`:

- **Absent** → first push to a fresh path. We create the binding. Fine.
- **Present and its identity matches our source** → normal re-push. Proceed to
  overwrite analysis.
- **Present and its identity differs** → **conflict, hard stop.** The remote
  path is already owned by a different dataset. We name both identities and
  refuse — pushing here would mean overwriting an unrelated dataset's root,
  which `-m` alone should not authorize.

## Overwrite protection and the remote update log

The brief: *"when a user might overwrite remote data, disallow the push unless
they provide a command that goes in a persistent update log on the remote."*

**Overwrite detection.** For each object we are about to put, we compare
against what is already at the remote:

- New object (no remote counterpart) → *additive*, always allowed.
- Identical bytes (matching size + content digest; we use the merkle root for
  facet files, ETag/size otherwise) → *skip*, nothing to do.
- Different bytes at an existing key → *overwrite*, gated.

### `pushlog.jsonl` is an event log

`pushlog.jsonl` is the single primary provenance artifact for a dataset — it
lives at the publish root and is never relegated to a side prefix. It is an
**append-only event log**, not a one-record-per-push journal. Every push is
*bracketed* by two events that share one monotonically increasing `seq`:

```json
{"event":"begin","seq":42,"ts":"2026-06-01T18:22:04Z","actor":"jshook@host","cmd":"vectordata push --to s3://my-bucket/datasets/glove-100/ -m \"regen neighbors after dedup\"","message":"regen neighbors after dedup","overwrites":[{"key":"profiles/1m/neighbor_indices.ivec","old_digest":"…","new_digest":"…"}],"added":["profiles/2m/…"],"sums":{"":"sha256:…","profiles/1m":"sha256:…"},"tool_version":"1.2.2"}
{"event":"complete","seq":42,"ts":"2026-06-01T18:23:10Z","sums":{"":"sha256:…","profiles/1m":"sha256:…"}}
```

- A **`begin`** event is the first thing an uploader writes — it commits, up
  front, the intent of a half-complete update: the resolved invocation (`cmd`,
  reproducible), the human justification (`message`, the *why*), the planned
  `overwrites`/`added`, and the per-directory `SHA256SUMS` digests the push
  *will* establish. Its presence on the remote means **"`seq` N is being
  written; the data is in flux."**
- A **`complete`** event is the last thing the uploader writes, once every
  object is up. It marks **"`seq` N is stable for download,"** and echoes the
  per-directory `sums` so a reader that fetches only the *tail* of the log has a
  self-contained fingerprint of the stable version without scanning backward.

This is what replaces a separate in-flight marker file: the in-flux signal, the
commit point, and the crash tombstone are all just events (or the *absence* of
the closing event) in the one log everyone already reads. Downloaders are
covered under *Upload versioning* below.

**The gate.** If the plan contains **any** overwrite, the push is refused unless
`-m/--message` is supplied; the message rides on the `begin` event. Additive-only
pushes still emit `begin`/`complete`, but do not *require* a message.

### Provenance convergence — one history per dataset

The log is not just an audit trail; it is the **single provenance** of a
dataset, and `push` enforces that there is exactly one. The source directory
keeps its own `pushlog.jsonl` (persisted with the data, alongside
`.publish_url`), and the remote keeps the authoritative copy. The local log
must *converge* to the remote: before writing its `begin` event, `push` fetches
the remote `pushlog.jsonl` and compares it event-for-event against the local log
(excluding the events it is about to append). The local must be an **ancestor
of, or equal to,** the remote. Three cases, by design strict:

- **Remote equals local** → histories are in sync. The push proceeds (subject to
  the open-update check below).
- **Remote is local + more (remote ahead)** → **divergent provenance.** Someone
  else has pushed since this source last synced; proceeding would fork the
  single history. The push is **refused**, the remote-only delta is shown, and
  the user is told the next steps (re-sync the local copy / log, then reconcile
  the actual data) before any retry.
- **Local is local + more (local ahead)** → the local log has events the remote
  lacks. This is recoverable — the remote is simply behind. The user gets a
  warning and, **after explicit acknowledgement,** the push proceeds and carries
  the missing local events up to the remote.

**Open-update check.** Independently of the ancestor comparison, if the remote
log's tail is a `begin` with no matching `complete`, an update is *in progress
or did not finish*. A second uploader must **not** start: it would interleave
two pushes into one provenance. `push` refuses with the open `seq`, its actor,
and its timestamp. The unmatched `begin` is exactly the crash tombstone — and is
precisely where the deferred partial-failure handling will hook in: a later push
inspects the open event, reconciles (re-drive to `complete`, or record an
explicit `abort`), and only then proceeds. Starting from this strict invariant
is intentional — it forces each error corner case (a partially failed transfer
that leaves remote bytes inconsistent with the log, say) to be handled
explicitly with air-tight coverage as it is actually encountered, rather than
papered over by a lenient merge.

Mechanics: the log is append-only by intent, and object stores don't append, so
each event write is read-modify-write. The convergence compare is the *semantic*
guard (content-based: ancestor / divergent / ahead / open); an `If-Match` ETag
on every put-back is the *atomic* guard, so a racing uploader that slips an event
in between our fetch and our write is caught by the precondition failure and
forced to re-read and re-classify rather than silently overwrite. `--dry-run`
performs the convergence and open-update checks and prints the `begin`/`complete`
pair that *would* be written, without writing anything.

## Content checksums

`push` materializes a standard, externally-verifiable checksum file for the
content it ships. This is separate from the internal `.mrkl` merkle sidecars:
the merkle tree is the *streaming, chunk-level read-time* verification artifact;
the checksum file is the *whole-file, directory-level, tool-interoperable*
provenance artifact that any user can verify with stock coreutils.

### Algorithm: SHA-256

We use **SHA-256**, emitted in `sha256sum`-compatible form. The efficiency
trade-off lands here decisively:

- The repo already commits to SHA-256 everywhere it hashes content — the merkle
  scheme is `sha2 = "0.10"` (`vectordata/src/merkle/mod.rs`). SHA-1 is not even
  a workspace dependency. Using SHA-256 means one hash family across the
  internal merkle and the external checksum file, and zero new dependencies;
  SHA-1 would *add* one.
- The classic "SHA-1 is faster" argument is moot for this workload. Dataset
  facet files are large (vector blobs, often GiB), so checksumming is
  IO-bound, not CPU-bound; and on any modern x86-64 (SHA-NI) or ARMv8 (crypto
  extensions) host SHA-256 is hardware-accelerated to roughly parity with SHA-1
  anyway. SHA-1 wins only marginally, only on old CPUs, only in pure software —
  none of which describes the case here.
- SHA-1 is deprecated for integrity/provenance use; shipping it on a brand-new
  feature would be a poor signal even though we don't depend on collision
  resistance for corruption detection.

We **generate** the digests natively via the existing `sha2::Sha256` (no
shelling out), but **format** the file in the normative
`sha256sum -c`-compatible layout so an external consumer can verify it with the
stock tool:

```
# <dir>/SHA256SUMS
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  base_vectors.fvec
a1b2c3...                                                          query_vectors.fvec
```

### One file per directory level

Following normative practice, a single `SHA256SUMS` file is generated **per
directory level**, listing exactly the regular files at that level
(non-recursive — subdirectories carry their own `SHA256SUMS`). For a structured
dataset this means one at the dataset root and one inside each `profiles/<p>/`
directory. The `SHA256SUMS` file never lists itself, the `.mrkl`/`.mref`
sidecars, `.publish_url`, or `pushlog.jsonl` — only content.

### Freshness: the mtime invariant

A `SHA256SUMS` file is considered **current** iff its mtime is **greater than or
equal to** the mtime of every file it describes, and its file set exactly
matches the content files present at that level. If any described file is newer,
or files were added/removed, the checksum file is **stale**.

- **Default (`--checksums auto`)** — a stale `SHA256SUMS` is recomputed before
  the push (which also brings its mtime current). This is the happy path.
- **Override (`--checksums keep`)** — do not recompute; the user owns checksum
  generation (e.g. they produced `SHA256SUMS` with an external pipeline). If it
  is stale under the invariant, `push` **hard-stops** rather than shipping
  inconsistent content. The override changes *who computes*, never *whether the
  invariant must hold*.

### The checksum file is content, permanently

Once a directory has a `SHA256SUMS` (locally or on the remote), it is part of
the content from that point forward. Two hard rules follow, enforced in the
transfer planner:

1. **You may never push a content change into a directory that has a
   `SHA256SUMS` without also pushing an updated, current `SHA256SUMS`.** If the
   plan touches any content file in such a directory, a current checksum file
   for that directory must be in the plan too. Under `auto` this is automatic
   (recompute); under `keep` a stale file fails the push. This holds in `--raw`
   mode as well — the moment a remote directory carries a `SHA256SUMS`, ad-hoc
   pushes into it are bound by the same rule.
2. **The `SHA256SUMS` object participates in overwrite detection and the
   pushlog.** Its SHA-256 digest is recorded in the `begin`/`complete` events
   (per touched directory), so a completed version pins an exact content
   fingerprint set, and overwriting it goes through the same `-m` gate as any
   other content.

## Upload versioning via pushlog events

A push touches many objects but object stores commit one object at a time, so
there is a window in which an observer resolving the dataset could otherwise see
a torn state — new data files against an old `SHA256SUMS`, or vice versa. Rather
than bolt a separate marker file onto the side, the `begin`/`complete` events in
`pushlog.jsonl` (above) *are* the versioning mechanism: they make that window
**bracketed, ordered, and detectable** in the one artifact every downloader
already reads.

### Version = pushlog sequence

We do not invent a parallel version counter. **The dataset version is the `seq`
of the latest `complete` event.** Each push owns one `seq`, carried on both its
`begin` and `complete`. Because the events name the `SHA256SUMS` digest of every
directory the push established, version `N` *is* a precise content fingerprint of
the whole dataset at that point. The pushlog is already the provenance ledger
(see convergence, above); it doubles as the version history for free.

### The commit point and the ordering discipline

A version becomes official at exactly one moment: **the `complete` event lands**
(the `If-Match`-guarded append). Everything between `begin` and `complete` is the
in-flux window. The ordering is therefore:

```
1. append the begin event to the remote pushlog   ← announces seq N is in flux
2. upload content objects; within each directory, the SHA256SUMS goes LAST
3. refresh the remote .publish_url binding if needed
4. append the complete event to the remote pushlog ← atomic instant version N goes live
5. mirror both events into the local pushlog
```

Writing each directory's `SHA256SUMS` last makes the checksum file the local
commit signal for that directory: until it lands, a strict reader validating
against checksums sees the prior consistent set; once it lands, the new files it
names are all already present (they were uploaded first). The `complete` event is
the *global* commit signal across the whole push.

### What a downloader sees

A downloader needs only the pushlog — or just its tail (object stores serve a
range GET, so the last few KiB suffice):

- **Tail is a `complete` (seq N)** → the dataset is stable at version N. Verify
  downloaded files against the `sums` digests carried on that event.
- **Tail is a `begin` (seq N) with no matching `complete`** → an update to N is
  in progress (or crashed). The stable version is the prior `complete` (seq <
  N); pin to it and ignore any object newer than that commit, or wait and
  re-read. The downloader is *never* forced to guess from object timestamps —
  the log says, in band, whether the data is settled.

On a failed push the `complete` is simply never written, so the open `begin`
remains as the crash tombstone (handled in the open-update check, above). An
explicit `abort` event is a future refinement for cleanly cancelled pushes; the
unmatched `begin` already covers the safety case.

### Stronger atomicity, if we ever need it

The scheme above is "bracketed and detectable," not "truly atomic": the
`begin`/`complete` envelope plus checksums-last ordering lets a careful reader
avoid torn state, but a reader that ignores the log entirely still could observe
one. If full atomicity is ever required, the heavier alternative is
**version-prefixed publication** — upload version `N` under an immutable
`v<seq>/` prefix and flip a tiny mutable `latest` pointer as the single atomic
swap, with old versions retained for rollback. That changes the read-side layout
(readers indirect through the pointer) and multiplies storage, so it is noted as
a future option rather than the v1 default; the `seq`/event versioning here is
forward-compatible with it.

## Transports: dispatch on the URL scheme

There is a read-side transport factoring already
(`docs/design/storage_transport_factoring.md`): a single `open(source)`
dispatches `s3://`/`https://`/local to the right `Storage` variant, and no
caller picks a transport by hand. Push needs the **write** mirror of that,
which does not exist yet. We introduce a `PushTransport` trait with one
implementation per scheme, selected by the same scheme-dispatch the reader
uses — so "use whatever transport the URL suggests" is automatic, not a flag.

```rust
// write-side mirror of the read-side ChunkedTransport
trait PushTransport {
    fn head(&self, rel: &str) -> Result<Option<RemoteObject>>; // None = absent
    fn get(&self, rel: &str) -> Result<Vec<u8>>;               // for pushlog/.publish_url
    fn put(&self, rel: &str, body: &Path) -> Result<()>;       // create/overwrite
    fn put_conditional(&self, rel: &str, body: &[u8], if_match: Option<&str>) -> Result<()>;
}
struct RemoteObject { size: u64, digest: Digest } // ETag or merkle root
```

| Scheme | Implementation | Existence / overwrite check |
|---|---|---|
| `s3://`, virtual-hosted `https://*.s3.*` | S3 `PutObject` / `HeadObject` (AWS SDK; AWS CLI acceptable for v1, matching `veks`) | `HeadObject` size + ETag |
| generic `https://`, `http://` | REST: `PUT <base>/<rel>`, `HEAD`, `GET`, conditional `If-Match` | `HEAD` size + ETag |
| `file://`, bare local path | filesystem copy preserving the tree | `stat` size + digest |

The `s3://` ↔ `https://bucket.s3.region.amazonaws.com/` normalization already
in `vectordata/src/transport/mod.rs` is reused so the local binding can be
`s3://…` while the actual HTTP verbs go to the virtual-hosted host. A generic
`https://` host that is *not* S3-shaped uses plain REST `PUT` semantics and
expects an object-store gateway that honors them; this is documented as the
server contract, and a non-conforming endpoint surfaces as a clear transport
error rather than silent partial state.

## Authentication: automatic when present, explicit errors when not

Auth is resolved per-transport from ambient credentials first, flags/env as
override. The principle: *if the environment already has what the endpoint
needs, the happy path requires no auth flags at all.*

- **S3** — the standard AWS credential chain (env vars, `~/.aws/credentials`,
  `AWS_PROFILE`/`--profile`, SSO, IAM role). `--endpoint-url` for
  S3-compatible stores.
- **Generic HTTPS** — bearer token from `--token` or `VECTORDATA_PUSH_TOKEN`,
  sent as `Authorization: Bearer …`. No token → anonymous attempt.
- **file://** — filesystem permissions.

Error contract (no silent failures, per project posture):

| Condition | Message shape |
|---|---|
| No credentials available for an endpoint that needs them | `push: <endpoint> requires credentials; set AWS_PROFILE / --profile (S3) or VECTORDATA_PUSH_TOKEN / --token (https)` |
| Credentials present but rejected (401/403) | `push: authentication failed for <endpoint> (HTTP 403) — check that <profile/token> can write <key>` |
| Endpoint doesn't honor the write contract (e.g. PUT 405) | `push: <endpoint> does not accept object PUT (HTTP 405); not a writable object store?` |

Auth is checked with a cheap preflight (one `HEAD`/`HeadObject` against the
publish root) before any bytes move, so an auth problem fails fast and before
the binding or log is touched.

## End-to-end flow

```
1. Resolve PATH and source mode (dataset.yaml | knn_entries.yaml | --raw).
2. Resolve the destination:
     - read .publish_url (walk up) and/or --to; reconcile or hard-stop on disagreement.
     - if only --to given, stage a .publish_url write into the source.
3. Known-good validation for the mode (skippable with --no-check; never skips steps 5–9).
4. Checksums: per touched dir, check SHA256SUMS freshness (mtime ≥ all described files).
     - --checksums auto → recompute stale files.  --checksums keep → stale is a STOP.
     - any dir with a SHA256SUMS whose content changed MUST ship a current SHA256SUMS.
5. Select PushTransport from the URL scheme. Auth preflight (HEAD root) → fail fast.
6. Remote binding check: read remote .publish_url → absent | match | CONFLICT(stop).
7. Build the transfer plan: per file → add | skip | overwrite (digest compare).
     - any overwrite and no -m  → STOP with the list of would-be-overwritten keys.
8. Provenance convergence: fetch remote pushlog.jsonl, compare to local:
     - equal            → proceed.
     - remote ahead     → DIVERGENT, STOP; show remote-only delta and next steps.
     - local ahead      → WARN; proceed only after acknowledgement, carrying local-only events up.
     - remote tail = open begin (no complete) → STOP; update in progress at seq N by <actor>.
9. Confirm (unless -y). --dry-run prints plan + checksum actions + convergence verdict
   + the begin/complete pair (incl. seq) that would be written, and exits here.
10. Commit, in order:
     a. append the begin event to the remote pushlog (seq N) ← announces in-flux
        (If-Match guarded; precondition failure → re-read, re-run step 8, retry).
     b. upload content objects (concurrency M); within each dir, SHA256SUMS LAST.
     c. write/refresh remote .publish_url.
     d. append the complete event to the remote pushlog ← atomic instant version N goes live.
     e. on failure before (d): leave the open begin as the seq-N tombstone.
11. Mirror both events into the local pushlog; persist the local .publish_url binding.
12. Report: version (seq), N added, M overwritten, K skipped, destination URL.
```

Exit codes: `0` success (incl. dry-run), `1` operational failure (transport,
auth, partial transfer — leaves an open `begin` event as the seq-N tombstone),
`2` usage / binding conflict / overwrite-without-`-m` / divergent provenance /
open update in progress / stale checksums under `keep` (the "you need to do
something different" class).

## Open questions for review

1. **Identity for catalog/ad-hoc roots.** `dataset.yaml` gives a natural
   `name:`; ad-hoc needs a stamped id. Content hash of the first manifest, or
   a UUID written into the remote `.publish_url`?
2. **AWS SDK vs. AWS CLI for v1.** `veks publish` shells out to the AWS CLI.
   Matching that is the fastest path and keeps credential behavior identical;
   a native SDK removes the external dependency and gives us real `If-Match`
   conditional puts. Leaning SDK for the conditional-put requirement.
3. **Consolidation timing.** Do we land `push` standalone first and refactor
   `veks publish` to delegate later, or build the shared binding/transport
   module up front? The interfaces here assume the latter is cheap.

