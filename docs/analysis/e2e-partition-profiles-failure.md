# Root-cause analysis: `e2e_partition_profiles_full_pipeline` failure

**Status: FIXED & VALIDATED.** Was a pre-existing, deterministic defect on
`main` (HEAD `fd6e1df`), not introduced by the vecd/vectordata work. Root
cause was **two-layered** (both in the postfilter E-facet path). Both
layers are now fixed; the test passes and the full workspace is green
(`cargo test --workspace --no-fail-fast`: **2445 passed, 0 failed**). This
document replaces the earlier hand-wave ("pre-existing, unrelated") with
the actual causal chain and the implemented fix.

---

## 1. Symptom

`cargo test -p veks --test e2e_pipeline e2e_partition_profiles_full_pipeline`
fails at pipeline step 24 of 25:

```
Pipeline failed: step 'compute-postfiltered-knn' failed:
  open ground-truth <dataset>/neighbor_indices.ivec:
  IO error: No such file or directory (os error 2)
```

The test bootstraps a partitioned dataset (`required_facets =
"BQGDMPRFO"`, `partition_oracles = true`) and runs the full pipeline via
the `veks` binary.

## 2. An honest correction to the earlier triage

Two earlier claims need correcting:

1. **"It passed in the first workspace run."** It did not. `cargo test`
   is **fail-fast across test binaries by default** — it stops at the
   first failing binary. The first `cargo test --workspace` halted at
   `check_and_import` (the `compute_predicates` bug), so `e2e_pipeline`
   was never reached. Each fix simply *uncovered the next* already-failing
   binary (`check_and_import` → `completion_double_tab` →
   `e2e_pipeline`). These were serial discoveries, not new regressions.

2. **"Pre-existing, unrelated — leave it."** "Pre-existing" is a
   provenance fact, not an explanation. It *is* pre-existing (proven below)
   but it is a real, specific, reproducible bug worth understanding.

**Proof of pre-existing:** built and ran the test in a clean `git
worktree` at pristine HEAD (`fd6e1df`, none of this session's changes) —
it fails **identically**. Reverting the only non-test change made this
session (the `gen_predicate_keys` `survey`-input fix) also leaves the
failure identical.

## 3. Investigation method

The test uses a `tempfile::TempDir` that is deleted on drop (even on
panic), so the on-disk state vanishes. A temporary diagnostic
(`std::mem::forget(tmp)` on failure) kept the dataset directory for
inspection. That, plus tracing producers/consumers of the
`neighbor_indices` facet, revealed the chain below. The diagnostic and all
experimental edits have been reverted.

## 4. Causal layer 1 — wrong ground-truth path in `compute-postfiltered-knn`

**What's on disk after the run** (the `default` profile):

```
profiles/default/neighbor_indices.ivecs        ← produced (note: profile dir, .ivecs)
profiles/default/neighbor_distances.fvecs
profiles/default/prefiltered_neighbor_indices.ivec
profiles/default/metadata_indices.ivvecs
```

**What the step asked for:** `<dataset>/neighbor_indices.ivec` — wrong on
**both** axes:

| | produced | requested |
|---|---|---|
| directory | `profiles/default/` | dataset root |
| extension | `.ivecs` (plural) | `.ivec` (singular) |

**Where the bug is** — `veks/src/prepare/import.rs`, the
`compute-postfiltered-knn` step (≈ lines 1982–2021):

```rust
let mut e_opts = vec![
    ("ground-truth".into(), "neighbor_indices.ivec".into()),   // ← bug (line ~2001)
    ("metadata-indices".into(), slots.metadata...predicate_indices.path().into()),
    ("indices".into(), "postfiltered_neighbor_indices.ivec".into()),
];
if slots.knn... {
    e_opts.push(("ground-truth-distances".into(), "neighbor_distances.fvec".into())); // ← bug (~2009)
    ...
}
```

The step is `per_profile: true`. For such steps, **bare relative *output*
paths are auto-prefixed with the profile directory** (that's why the
producer's `output: "neighbor_indices.ivecs"` lands in
`profiles/default/`), but a bare relative **input** path is resolved
against the **workspace root**. Reading a per-profile facet as an input
therefore requires the explicit `${profile_dir}` token.

**The sibling step proves the correct idiom** — `verify-knn-partition`
(also `per_profile`) reads the same facet correctly
(`import.rs` line ~2114):

```rust
("indices".into(), "${profile_dir}neighbor_indices.ivecs".into()),
("distances".into(), "${profile_dir}neighbor_distances.fvecs".into()),
```

### Why `compute-prefiltered-knn` succeeds but `compute-postfiltered-knn` fails

They are siblings in the same `slots.filtered_knn` block, but:

- **prefiltered** computes filtered KNN directly from `base` + `query` +
  `predicates`. It does **not** read `neighbor_indices` at all, and its
  only bare-relative paths are *outputs* (auto-prefixed). Immune.
- **postfiltered** *derives* `E = G ∩ R`, so it **must read `G`
  (`neighbor_indices`)** as an input — through the mis-wired path. The one
  step that reads that facet is the one that breaks.

### Empirical confirmation

Temporarily changing the two lines to
`"${profile_dir}neighbor_indices.ivecs"` and
`"${profile_dir}neighbor_distances.fvecs"` makes step 24
(`compute-postfiltered-knn`) **succeed** — it produces
`profiles/default/postfiltered_neighbor_indices.ivec` and
`postfiltered_neighbor_distances.fvec`. The failure then moves to step 25
(layer 2). Layer 1 is therefore confirmed.

## 5. Causal layer 2 — `verify-postfiltered-knn` under-specifies its inputs

With layer 1 patched, step 25 fails:

```
step 'verify-postfiltered-knn' failed: verify postfiltered-knn-consolidated:
  anchor profile 'default' ... is missing required facet:
  - metadata_results
```

**Where it comes from** — the verifier
(`PostfilteredKnnConsolidated`) declares its anchor inputs in
`veks-pipeline/src/pipeline/dataset_lookup.rs` (≈ lines 219–222):

```rust
Self::PostfilteredKnnConsolidated => &[
    ("ground-truth",    "neighbor_indices"),
    ("metadata-indices", "metadata_results"),
],
```

Validation **skips** a facet check *only if the step passes the
corresponding `--<option>` explicitly* (documented at lines 198–202).

But the `verify-postfiltered-knn` step in `import.rs` passes **no input
options at all** — only `sample`, `seed`, `output`:

```rust
options: vec![
    ("sample".into(),  "50".into()),
    ("seed".into(),    "${seed}".into()),
    ("output".into(),  "${cache}/verify_postfiltered_knn.json".into()),
],
```

So validation falls back to requiring the anchor profile to *declare* a
facet literally named **`metadata_results`** — which it never does. The
predicate/metadata-index file in this pipeline is declared as
**`metadata_indices`** (`profiles/default/metadata_indices.ivvecs`).
(Contrast `verify-prefiltered-knn` just above it, which passes explicit
`base`/`query`/`metadata`/`predicates` options and so never hits this.)

Layer 2 is thus a second, independent wiring defect:

- the `verify-postfiltered-knn` step doesn't forward the inputs it needs
  (`--ground-truth`, `--metadata-indices`, `--indices`); **and/or**
- there is **facet-name drift** for the predicate-index artifact —
  `metadata_results` (what the verifier looks up) vs `metadata_layout`
  (`PredicateResults` anchor, line 226) vs `metadata_indices` (what's
  actually produced). The same underlying file is referred to by three
  different canonical facet names across the codebase.

A layer 3 may exist behind layer 2 (the verifier's own re-derivation
correctness) — not reached until layer 2 is fixed.

## 6. Why no other test catches this

The post-filter **E facet** is only emitted when `slots.filtered_knn` is
materialized *and* the dataset is laid out per-profile (partitioned). The
`e2e_partition_profiles_full_pipeline` test is the **only** test that
requests `BQGDMPRFO` with `partition_oracles = true`, so it is the only
end-to-end exercise of the postfilter compute+verify chain. That chain has
**never passed end-to-end** — both defects were shipped behind a test that
was itself masked by `cargo`'s fail-fast behavior.

## 7. The implemented fix

The initial instinct — hardcode `${profile_dir}neighbor_indices.ivecs` in
`import.rs` — was rejected: it re-pins a single extension and bypasses the
facet system. The codebase **already** has the right abstraction:
`dataset_lookup::resolve_path_option` → `lookup_facet`, used by every
`verify_*` command, which reads the profile's *declared* facet path with
`.ivecs`/`.ivec` locator tolerance. The defect was that the postfilter
steps bypassed it. Both layers are fixed by routing through it:

**Layer 1 — `compute-postfiltered-knn` resolves G/D as facets.**
- `compute_postfiltered_knn.rs`: resolve `ground-truth` via
  `resolve_path_option(ctx, options, "ground-truth", "neighbor_indices")`
  (and `ground-truth-distances` → `"neighbor_distances"` when a distances
  output is requested); mark the `ground-truth` schema option
  `required: false` so facet fallback is allowed (mirroring `verify_*`).
- `veks/src/prepare/import.rs`: stop hardcoding `ground-truth` /
  `ground-truth-distances` on the step — the command resolves them
  per-profile (correct path **and** extension tolerance for extant
  datasets).

**Layer 2 — fix the postfilter verifier's R anchor facet.**
- `dataset_lookup.rs`: `PostfilteredKnnConsolidated` anchored R on
  `metadata_results` (a `.slab` facet that can never locate the file). The
  R input is the per-query predicate-match **index** stored as
  `metadata_indices.{ivvecs,…}`, which the resolver locates under the
  `metadata_layout` canonical (`canonical_basename_for` +
  `canonical_extensions_for`). Corrected the anchor to
  `("metadata-indices", "metadata_layout")`.

**Validation:** `e2e_partition_profiles_full_pipeline` passes; full
workspace `cargo test --workspace --no-fail-fast` = **2445 passed, 0
failed**. No layer 3.

## 7a. Would extension tolerance (`.ivecs` vs `.ivec`) have fixed it?

**No.** The mismatch is on two independent axes, and the extension is the
*secondary* one:

| axis | produced | requested |
|---|---|---|
| **directory** | `profiles/default/` | dataset **root** |
| extension | `.ivecs` (plural, canonical) | `.ivec` (singular) |

The failing open was `<dataset>/neighbor_indices.ivec` — at the **root**,
which held no `neighbor_indices.*` under *either* extension. An
extension-agnostic open at the root would still find nothing. The dominant
defect is the missing `${profile_dir}` prefix; the directory fix is
necessary and sufficient for layer 1, and the canonical path already uses
the plural form that exists on disk.

The singular/plural angle is therefore a **red herring for this failure** —
but it's a real, independent backward-compat concern. The correct fix
must:

1. wire the **canonical plural** path (`${profile_dir}neighbor_indices.ivecs`),
   which fixes the actual bug; and
2. **retain the ability to read the singular `.ivec`/`.fvec` form** for
   extant datasets — as extension tolerance at the file-resolution point,
   *not* by reverting the wiring to the singular literal. (Fixing only the
   directory but keeping the singular literal would still fail here, since
   the produced file is `.ivecs`.)

## 8. Naming debt — RESOLVED (Stage 1)

The deeper inconsistency (the R facet wearing three names —
`metadata_indices` key, `metadata_results` canonical, `metadata_layout`
locator) has since been fixed. `metadata_results` is now the single
canonical R facet, owning its file under both the canonical and legacy
`metadata_indices` names; the `metadata_layout => metadata_indices` hack is
removed; `lookup_facet` is alias-aware; and the interim postfiltered anchor
(which leaned on `metadata_layout`) is repointed to `metadata_results`.
Full workspace green (2445 passed, 0 failed).

This is part of a broader cleanup — the corrected facet↔file model, the
`path#namespace` locator, and the plan to store the metadata **layout**
schema as a namespace inside both the content and results slabs (with
content↔results compatibility verified by a byte-for-byte schema match) —
recorded in **`docs/design/metadata-facets-and-layout-namespace.md`**.
Stage 1 (naming correctness) is done; Stage 2 (writing the layout namespace
+ the compatibility check) is the remaining feature work there.
