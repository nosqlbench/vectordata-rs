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

## Failure modes

This is the exhaustive catalogue of what can go wrong in a push, how the
system detects it, and how it responds. The guiding principle is the one
stated up top — *low ceremony for the happy path, hard stops for the
dangerous ones* — refined by experience into three rules:

1. **Never silently change or lose remote data.** Anything destructive
   (overwrite, delete) is gated behind `-m` and recorded; anything
   ambiguous is refused, not guessed.
2. **Never silently skip work.** A flag that can't be honored (e.g.
   `--delete` on a transport that can't list) errors loudly.
3. **Always be resumable.** Every multi-object write is bracketed so a
   failure leaves a well-defined, recoverable state, never a torn one
   that looks complete.

Exit codes: **0** success (including `--dry-run` and the no-op
fast-forward); **1** operational failure (transport, auth, I/O, a
concurrency race); **2** the "you must do something different" class
(usage, binding/provenance conflicts, ungated destructive change).
Failures in classes A–E are detected *before any remote write*, so they
leave the remote untouched.

### A. Source resolution & validation (exit 2; remote untouched)

| Trigger | Response | Test |
|---|---|---|
| `PATH` is not a directory | refuse | — |
| No `dataset.yaml`/`knn_entries.yaml` and no `--raw` | refuse, tell user to pass `--raw` | `no_recognized_mode_without_raw_is_refused` |
| Structured: `dataset.yaml` unparseable | refuse | — |
| Structured: `is_zero_vector_free`/`is_duplicate_vector_free` missing or not `true` | refuse, naming the attribute | `structured_missing_attributes_…` |
| Catalog: `knn_entries.yaml` unparseable | refuse | — |
| Catalog: a referenced facet file is missing on disk | refuse, naming the file | `catalog_mode_end_to_end_and_missing_file_…` |

`--no-check` bypasses this whole class (the binding/provenance/gate
rules below still apply); covered by the `…no_check bypasses` assertion.

### B. Binding (exit 2; remote untouched)

| Trigger | Response | Test |
|---|---|---|
| No `.publish_url` (here or in a parent) and no `--to` | refuse — "no destination", with the `echo … > .publish_url` hint | `no_destination_is_refused` |
| `.publish_url` malformed / unsupported scheme | hard stop | `malformed_publish_url_is_a_hard_stop` |
| Local `.publish_url` and `--to` disagree | conflict — re-binding is deliberate | `binding_conflict_is_refused` |
| Remote root already bound to a *different* endpoint | remote conflict | `remote_bound_elsewhere_is_a_conflict` |

### C. Checksums (exit 2 under `keep`; `auto` self-heals)

| Trigger | Response | Test |
|---|---|---|
| `--checksums keep` + missing `SHA256SUMS` | refuse | `keep_policy_refuses_stale_checksums` |
| `--checksums keep` + stale `SHA256SUMS` | refuse, with the staleness reason | (same) |
| `--checksums auto` + stale/missing | recompute (default happy path) | overwrite/`generate` tests |

Residual risk: freshness is judged by the mtime invariant, anchored to
the newest *described file's* mtime (not `now()`, which jitters against
the filesystem clock on some hosts). On a filesystem with coarse (≥1 s)
mtime granularity, a content change in the *same second* as the last
checksum generation is invisible to the heuristic. Documented limit; the
remote `SHA256SUMS` (regenerated each `auto` push) still bounds it in
practice.

### D. Provenance & convergence (exit 2, except the recoveries)

Convergence is judged against the remote's **committed** history (the
log minus any trailing open `begin`).

| Trigger | Response | Test |
|---|---|---|
| Remote committed-ahead, **and** local content differs from the remote head | refuse — divergent provenance | `divergent_provenance_is_refused` |
| Remote committed-ahead, **but** local content reproduces the remote head | **fast-forward** the local log, report up-to-date (exit 0) | `crash_after_complete_before_mirror_fast_forwards` |
| Forked (neither log a prefix of the other) | refuse — manual reconciliation | `forked_provenance_is_refused` |
| Local ahead of remote committed | proceed; carry the local-only tail up | `local_ahead_carries_its_tail_up` |
| Open `begin` on remote, intended `sums` **match** ours | **resume** that seq (idempotent finish) | `resume_finishes_an_interrupted_push` |
| Open `begin`, `sums` differ, no `--abort-incomplete` | refuse, offering resume-from-source or `--abort-incomplete` | `open_push_with_changed_source_…`, `open_update_tombstone_blocks` |
| Open `begin`, `sums` differ, `--abort-incomplete` | record `abort`, push fresh at next seq | `open_push_with_changed_source_…` |

### E. Destructive-change gate (exit 2; remote untouched)

| Trigger | Response | Test |
|---|---|---|
| Plan overwrites existing content, no `-m` | refuse, listing the keys | `overwrite_without_message_is_blocked` |
| `--delete` would remove orphans, no `-m` | refuse, listing the keys | `delete_removes_orphans_gated_by_message` |

A resume skips this gate — the interrupted push was already authorized
when its `begin` was written. A `SHA256SUMS` updating to reflect a purely
*additive* change does not trip it.

### F. Transport & operational (exit 1)

| Trigger | Response | Test |
|---|---|---|
| Credentials missing / rejected (S3 chain, https bearer) | fail fast at the preflight HEAD; actionable message | `https_auth_failure_then_success_with_token` |
| Endpoint doesn't accept object `PUT` (HTTP 405) | error — "not a writable object store?" | (https transport) |
| `aws` CLI absent from `PATH` (s3) | error with install hint | — |
| `--delete` on a generic `https://` endpoint (no listing) | error loudly — orphan cleanup unsupported here | `https_transport_verbs_…` (`list` errs) |
| Generic I/O / protocol error | surfaced as operational | — |
| **Concurrency race**: another writer changed `pushlog.jsonl` between our read and our conditional write (`If-Match` precondition fails) | error — "re-run to re-converge"; no silent clobber | `transport_conditional_put_enforces_if_match`, `https_transport_verbs_…` |

### G. Partial failure & crash recovery

The commit order is `[abort] → begin → upload (SHA256SUMS last per dir)
→ .publish_url → complete → delete`. `begin` is written *before* any
object, so any failure between `begin` and `complete` leaves an open
`begin` tombstone and a recoverable state — never a torn "looks done".

| Where the push dies | State left | Recovery on rerun |
|---|---|---|
| After `begin`, before any upload | open `begin`; no objects | resume re-uploads everything, writes `complete` |
| Mid-upload | open `begin`; some objects present | resume re-uploads only what's missing/differing (idempotent), writes `complete` |
| After content, before `SHA256SUMS` | open `begin`; content up, sums stale/absent | resume uploads sums + `complete` |
| After `complete`, before local mirror | committed remote; local behind | fast-forward (D, row 2) — no wedge |
| During the `abort` write (abort-then-fresh) | open `begin` still present | rerun re-evaluates the open begin from scratch |
| Confirmation declined, or non-interactive without `-y` | nothing written | exit 2 |

End-to-end proof that a *real* mid-upload fault leaves a resumable
tombstone (not only a manufactured state):
`real_midupload_failure_leaves_resumable_tombstone` injects a read-only
remote directory, confirms the push dies with the open `begin` recorded
and the object absent, then clears the fault and verifies the rerun
resumes the same seq to completion.

### H. Residual risks (acknowledged, not fully eliminated)

- **The in-flux window is detectable, not atomic.** Between `begin` and
  `complete` a reader that consults the log (or reads `SHA256SUMS`-last
  per directory) sees the prior stable version; a reader that ignores the
  log entirely could observe a torn set. Full atomicity would need
  version-prefixed publication + a pointer flip (noted as future work).
- **Deletion is not transactional with the version.** Orphans are removed
  *after* `complete`, so a reader pinned to a prior version may lose files
  that version referenced. `--delete` is opt-in and `-m`-gated precisely
  because it is destructive across versions.
- **Multi-object upload is not rolled back on failure** — it is *resumed*.
  The remote may hold a partial set between a crash and the next push;
  that set is reconciled, never presented as complete (no `complete`
  event exists for it).
- **Same-second checksum staleness** on coarse-granularity filesystems
  (see C).
- **S3 transport spawns one `aws` process per object** — a throughput
  limit, not a correctness issue; a native SDK is the future swap.
- **Untestable in CI here**: live-endpoint S3 auth (the https-bearer
  equivalent *is* covered against the in-repo object-store mock).

### I. Causal shell — second-order failure modes

Classes A–H are the *surface* failures: what the user sees and how the
command responds. This section is the next ring outward — the underlying
causes and the failures that live *beneath the mechanisms* A–H rely on.
Each row is tagged: **[mitigated]** (handled or structurally prevented),
**[residual]** (real but accepted, documented limit), or **[gap]** (a
hardening we should do; not yet implemented).

#### Beneath A (source scan) — what the walk assumes about the tree

| Deeper cause | Surface failure it produces | Status |
|---|---|---|
| A symlink in the source tree (`file_type()` reports neither file nor dir) | symlinked content is **silently omitted** from the publish set — violates "never silently skip" | **[mitigated]** — scan now refuses symlinks with a clear error (`scan_refuses_symlinks`) |
| Symlink loop / cycle | (currently moot — symlinks skipped, so no recursion) | **[mitigated]** by the skip above |
| Non-UTF-8 filename | `to_string_lossy` would mangle the key (U+FFFD), so the published name ≠ the real byte name | **[mitigated]** — scan now refuses non-UTF-8 names (`scan_refuses_non_utf8_names`) |
| Filename containing whitespace/newline | the line-based `SHA256SUMS` format isn't injective — the entry round-trips wrong, corrupting that dir's manifest | **[mitigated]** — coreutils `\`-escaping of newline/backslash names (`escapes_names_with_backslash_or_newline`) |
| `is_zero_vector_free: true` asserted but untrue | validation trusts the *metadata*, never verifies the content — a mislabeled dataset publishes as "known-good" | **[residual]** — by design veks owns content verification |
| A file is read for its SHA-256, then **changes before upload** (TOCTOU) | remote bytes ≠ the digest recorded in `SHA256SUMS` → silent integrity violation | **[mitigated]** — `(len,mtime)` snapshot at hash time is re-checked before any write; a change aborts pre-commit (`changed_since_detects_mid_push_mutation`). Same-second same-size edit on coarse fs remains the documented limit |

#### Beneath B (binding) — URL identity & ancestry

| Deeper cause | Surface failure | Status |
|---|---|---|
| Binding equality is string comparison of the normalized URL | case / trailing-slash variants within a scheme → **false conflict or false match** | **[mitigated]** — normalize now lowercases scheme+host and fixes the trailing slash (`canonicalizes_scheme_and_host_case`). Cross-scheme aliases (`s3://` vs its `https` virtual-host form) and percent-encoding remain **[residual]** |
| `.publish_url` walk-up crosses into an unintended ancestor / mount | binds to the wrong remote | **[residual]** — logical-path walk limits but doesn't eliminate |
| Two sibling sources both covered by one ancestor `.publish_url` | they collide on one root `pushlog.jsonl`/manifest → interleaved provenance | **[residual]** — whole-hierarchy delegation makes this rare |

#### Beneath D/F (provenance & the wire) — trust in fetched bytes

| Deeper cause | Surface failure | Status |
|---|---|---|
| A `GET` of `pushlog.jsonl` returns a valid **prefix** (trailing line dropped by a truncated/reset response) | fewer events parsed → **misclassified convergence** (looks like remote is behind) → possible wrong overwrite/diverge verdict | **[mitigated]** — `get` verifies body length against `Content-Length` (and reqwest already errors on a premature EOF); a short body errors instead of parsing |
| The endpoint **silently ignores** `If-Match`/`If-None-Match` (older S3-compatible stores) | the CAS guard no-ops → two racing pushers clobber the pushlog and **lose events** — the single-provenance guarantee breaks | **[mitigated]** — a preflight conditional-write probe refuses any endpoint that ignores the precondition (`push_refuses_endpoint_that_ignores_conditional_writes`) |
| `pushlog.jsonl` grows without bound | every push reads + rewrites the whole log (O(n) each, O(n²) over history); large logs cost memory/bandwidth and widen the CAS race window | **[residual]** — no compaction yet |
| The entire skip/resume/fast-forward logic trusts SHA-256 collision resistance | a collision → wrong "skip"/false match | **[residual]** — accepted cryptographic assumption |
| Local `pushlog.jsonl` corrupted/hand-edited but still parses | mis-converges (e.g. spurious "forked") | **[residual]** — recover by deleting the local log → fast-forward heals |

#### Beneath F (transport) — the moving parts under "operational error"

| Deeper cause | Surface failure | Status |
|---|---|---|
| S3 region redirect (301/307) on a **PUT** to a wrong-region virtual host | reqwest may not replay the request body → upload fails or misbehaves (the read path tolerates GET redirects; the write path is less forgiving) | **[mitigated]** — a redirect on PUT now surfaces a clear "use the correct regional endpoint or s3://" error instead of an opaque failure; using an `s3://` binding avoids it entirely |
| `aws` CLI error classified by **stderr substring** ("not found", "accessdenied") | a different CLI version/locale → an auth error miscategorized as generic (wrong exit code + worse message) | **[residual]** — fragile; structured `--output json` errors would be better |
| A killed large `aws s3 cp` (multipart) | orphaned incomplete multipart parts → **storage-cost leak**, no visible object | **[residual]** — relies on a bucket lifecycle rule to reap |
| Transient DNS/TLS/timeout on a write | the push fails with no automatic retry (the read path has `RetryPolicy`; the write path has none) | **[deferred — judgement]** resume already recovers any failed push, and auto-retrying a *conditional* put risks turning a lost-response success into a spurious precondition failure; favoring correctness over convenience, retry is deliberately not added |

#### Beneath G (crash recovery) — gaps between the bracketed writes

| Deeper cause | Surface failure | Status |
|---|---|---|
| Crash **between** the `abort` write and the fresh `begin` (abort-then-fresh) | remote is `[…, begin(N), abort(N)]`, local behind, new content differs from the (stale) stable head → classified divergent → **wedge** until the user syncs the local pushlog | **[residual]** — recoverable (the error says how), but two non-atomic writes |
| Crash mid-write of the **local** mirror on `file://` | torn/half-written local `pushlog.jsonl` | **[mitigated]** — local mirror + binding writes are now temp-file + atomic `rename`; and as a backstop, deleting the local log → fast-forward heals |
| Crash mid-write of a remote `SHA256SUMS` on `file://` | truncated remote manifest → bad overwrite oracle | **[mitigated]** — the `file://` transport now writes via temp + atomic `rename`, so objects are never torn (matching S3/https whole-object PUT atomicity) |
| The injected fault never clears (read-only dir stays read-only) | each rerun re-attempts and re-fails; never progresses | **[mitigated]** — no corruption, just no progress; the tombstone keeps it resumable |

#### Cross-cutting assumption

The single-provenance guarantee rests on **one logical writer per publish
root, serialized by the pushlog's conditional write**. Concurrent writers
are serialized (the loser gets `PreconditionFailed` and retries) *only if
the store honors conditional writes* (see the [gap] above). The
file:///in-memory-mock tests exercise the conditional path; real-store
behaviors (multipart ETags, region redirects, eventual consistency) are
not yet exercised against a live endpoint.

## Failure modes by aspect

The same modes (surface classes A–H and the causal shell I/X), cross-cut
by four questions: **who owns robustness** for it, **how much control the
user has** to avoid it (None/Low/Med/High), **whether there is a remedy**
once it happens, and the **design cost to eliminate** it entirely
(None = already handled; Small/Med/Large otherwise). IDs reuse the
section labels above; the `I*`/`X1` rows are the causal-shell items in
their listed order.

| ID | Failure (brief) | Owns robustness | User control | Remedy once hit | Design cost to eliminate |
|---|---|---|---|---|---|
| A1 | PATH not a directory | User | High | fix the path | None (handled) |
| A2 | no mode + no `--raw` | User | High | add a descriptor or `--raw` | None |
| A3 | `dataset.yaml` unparseable | User / veks | High | fix the file | None |
| A4 | publishable attrs missing/false | User / veks | High | set attrs, or `--no-check` | None |
| A5 | `knn_entries.yaml` unparseable | User | High | fix the file | None |
| A6 | catalog references a missing file | User | High | provide the file | None |
| B1 | no destination (no binding, no `--to`) | User | High | write `.publish_url` or pass `--to` | None |
| B2 | malformed/unsupported `.publish_url` | User | High | fix the URL | None |
| B3 | local `.publish_url` vs `--to` disagree | User | High | edit binding or drop `--to` | Small (a `rebind` verb) |
| B4 | remote already bound elsewhere | User + remote | Med | choose another path / coordinate | Med (ownership model) |
| C1 | `keep` + missing `SHA256SUMS` | User | High | use `auto`, or generate sums | None |
| C2 | `keep` + stale `SHA256SUMS` | User | High | use `auto`, or regenerate | None |
| D1 | divergent (remote ahead, content differs) | User + remote | Med | re-sync local log, reconcile, re-push | Med (reconcile tool) |
| D3 | forked provenance | User + remote | Low | manual reconciliation | Large (multi-writer merge) |
| D6 | open begin, intent differs | User | High | resume from source, or `--abort-incomplete` | None (remedies built in) |
| E1 | overwrite without `-m` | User | High | supply `-m` | None (intentional gate) |
| E2 | delete without `-m` | User | High | supply `-m` | None (intentional gate) |
| F1 | credentials missing/rejected | User + env | High | set creds / token | None (clear error) |
| F2 | endpoint won't accept `PUT` (405) | User + remote | Med | point at a writable object store | None–Small |
| F3 | `aws` CLI absent | User / env | High | install it | Small (native SDK) |
| F4 | `--delete` on a non-listable `https` | User + transport | High | drop `--delete`, or use s3/file | Med (often impossible) |
| F5 | transient I/O / protocol error | transport / network | Low | re-run (resumes) | Small (write retry) |
| F6 | CAS race (precondition fails) | push + concurrent user | Low | re-run (re-converges) | None (guard working) |
| I1 | symlink in tree silently dropped | push (scan) | Low | restructure tree; notice the gap | Small (warn/resolve/refuse) |
| I2 | non-UTF-8 filename mangled | push + OS | Med | rename | Small (byte-preserve/reject) |
| I3 | filename whitespace breaks `SHA256SUMS` | push (format) + user | Med | rename | Small (escape/reject) |
| I4 | publishability attrs asserted but untrue | User / veks | High | honesty / content verify upstream | Med (push-side verify) |
| I5 | content changes between hash and upload (TOCTOU) | push + user | Med | don't mutate mid-push; re-push | Med (verify-after / lock) |
| I6 | URL equality false match/conflict | push (binding) | Low | make URLs textually identical | Small (canonicalize) |
| I7 | walk-up binds wrong ancestor | user (layout) + push | Med | place `.publish_url` correctly | Small |
| I8 | two sources share one root | user (layout) | Med | separate publish roots | Med |
| I9 | truncated pushlog fetch → misclassify | transport + push | None | (dangerous: may act silently) re-run | Small (Content-Length guard) |
| I10 | store silently ignores conditional writes | remote store + push | Low | use a conforming store | Small (probe at preflight) |
| I11 | `pushlog.jsonl` unbounded growth | push (design) | None | none (slow degrade) | Med (compaction/snapshot) |
| I12 | SHA-256 collision assumption | crypto | None | none | Large (and unwarranted) |
| I13 | local pushlog corrupt but parses | user / crash | High | delete local log → fast-forward | None |
| I14 | S3 region redirect on `PUT` | transport + user | High | set region / use aws transport | Small (handle redirect) |
| I15 | fragile `aws` stderr classification | push (transport) | None | none (worse message only) | Small (structured errors) |
| I16 | orphaned multipart on killed upload | transport + remote | Low | bucket lifecycle rule | Small / external |
| I17 | no write-side retry | push (transport) | Low | re-run (resumes) | Small (RetryPolicy on writes) |
| I18 | crash between `abort` and fresh `begin` | push (non-atomic writes) | None | sync local log → re-run | Med (atomic, or auto-sync) |
| I19 | torn local mirror write (crash) | push + OS | None | delete local log → fast-forward | Small (atomic rename) |
| I20 | torn remote `SHA256SUMS` on `file://` | push + OS | None | replace file / re-push dir | Small (atomic rename) |
| I21 | injected fault never clears | environment | Med | fix the fault → re-run resumes | None (stays resumable) |
| X1 | >1 logical writer per root | design assumption | Med | serialize pushers | Large (multi-writer) |

### What the matrix shows

- **The happy-path guardrails cost nothing to keep correct.** Every
  *user-controllable* mode (A, B1–B3, C, E, most of F, D6) is High-control
  with a clear remedy and **None** design cost — they are working as
  intended: refuse early, tell the user exactly what to change.
- **The residual risk concentrates in the transport/remote layer** the
  push command doesn't own (I9, I10, I14–I17, F5) — almost all **Small**
  fixes (a fetch-length guard, a conditional-write probe, write retries,
  region handling, structured `aws` errors). These are the highest
  value-per-effort hardening and the place to spend next.
- **The "None user control" rows are the ones to design against**, since
  the user can't avoid them: I9 (silent misclassification — the most
  dangerous, Small to fix), I10 (silent guarantee loss, Small to detect),
  I11/I18 (Med), and the crash-torn-write trio I19/I20 (Small, atomic
  rename). Notably most are still **Small**.
- **Only genuinely multi-writer concurrency (D3, X1) needs Large change.**
  Everything else is None/Small/Med — the single-writer, single-provenance
  model is cheap to harden and only expensive to abandon.

### Hardening status (implemented)

The matrix's "design cost to eliminate" column is the *pre-hardening*
estimate. The following have since been implemented (the per-mode rows in
§I carry the specifics and test names):

- **Closed:** I1 (refuse symlinks), I2 (refuse non-UTF-8 names), I3
  (coreutils-escaped manifest names), I5 (TOCTOU `(len,mtime)` recheck),
  I6 (scheme+host URL canonicalization), I9 (`Content-Length` fetch
  guard), I10 (conditional-write preflight probe), I14 (clear PUT-redirect
  error), I18 (RemoteAhead fast-forward generalized to abandoned/abort
  tails), I19/I20 (atomic temp+rename for local writes).
- **Deferred by judgement, with rationale:** I17 (no write-side retry —
  resume already recovers; auto-retrying conditional puts risks
  correctness). I11 (pushlog compaction — no correctness impact, future).
  I15 (`aws` stderr classification — message quality only). I16 (orphaned
  multipart — bucket-lifecycle territory). I12 (SHA-256 collision —
  accepted cryptographic assumption).
- **Still residual (large or out-of-model):** cross-scheme URL aliasing
  (part of I6), D3/X1 multi-writer forks, live-endpoint S3 auth (untested
  here, behavior in place).

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

