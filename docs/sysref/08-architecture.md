# 8. Architecture

Internal design of the workspace, pipeline engine, CLI framework,
and UI layer.

---

## 8.0 Design Principles

- **Performance at scale** — zero-copy I/O (mmap), SIMD distance
  kernels, resource-aware operations governed by `ResourceGovernor`
- **Storage/logic decoupling** — slab files store opaque bytes;
  interpretation happens in the ANode codec layer above
- **Location transparency** — same API for local and remote datasets;
  Merkle trees + HTTP Range requests make remote feel local
- **Declarative pipelines** — `dataset.yaml` defines the full
  processing graph; the engine handles ordering, caching, resumption

### Technology stack

| Role | Crate |
|------|-------|
| CLI | clap v4 |
| I/O | memmap2, reqwest |
| Concurrency | std::thread::scope, rayon |
| SIMD distance | simsimd |
| Progress | indicatif |
| TUI | ratatui |

---

## 8.1 Workspace Structure

```
vectordata-rs/
├── vectordata/        Access library (consumer crate)
│   ├── io.rs          VectorReader, VvecReader, open_vec, open_vvec
│   ├── view.rs        TestDataView trait, profile facet access
│   ├── typed_access.rs  TypedReader with width negotiation
│   ├── dataset/       dataset.yaml parsing, profiles, catalogs
│   ├── catalog/       Catalog resolver and sources
│   ├── merkle/        Merkle hash trees (.mref/.mrkl)
│   ├── transport/     HTTP transport with Range requests
│   └── cache/         Read-through cache with Merkle verification
│
├── veks/              CLI application
│   ├── main.rs        Clap CLI, subcommand dispatch
│   ├── cli/           Shell completion engine
│   ├── datasets/      Consumer commands (list, prebuffer, probe, cache)
│   ├── prepare/       Producer commands (bootstrap, import, stratify)
│   ├── publish/       S3 publishing
│   ├── check/         Preflight checks (integrity, merkle, extraneous)
│   └── explore/       Interactive TUI: dataset picker, unified analytics
│                      (norms/distances/eigenvalues/PCA), shell REPL,
│                      values grid (scrollable ordinal × dim, with
│                      heatmap palettes, sig-digit control, L2-norm
│                      column, and L2-normalized view toggle)
│
├── veks-pipeline/     Pipeline engine + commands
│   ├── pipeline/
│   │   ├── mod.rs     DAG resolution, step execution, variable sync
│   │   ├── runner.rs  Step lifecycle, freshness, fingerprinting
│   │   ├── commands/  50+ command implementations
│   │   └── resource.rs  Resource governance
│
├── veks-anode/        Wire formats (MNode, PNode, ANode)
├── veks-core/         Shared utilities (term colors, filters, formats)
├── veks-completion/   Shell completion library
├── veks-io/           Standalone vector I/O (readers, writers, mmap)
└── slabtastic/        Page-aligned record container
```

---

## 8.2 Command Framework

Every pipeline command implements the `CommandOp` trait:

```rust
pub trait CommandOp: Send {
    fn command_path(&self) -> &str;           // e.g., "compute knn"
    fn command_doc(&self) -> CommandDoc;       // built-in markdown documentation
    fn describe_options(&self) -> Vec<OptionDesc>;
    fn describe_resources(&self) -> Vec<ResourceDesc>;
    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult;
}
```

### CommandDoc

Every command carries its own documentation as compiled-in markdown:

```rust
CommandDoc {
    summary: "one-line description".into(),
    body: "# command name\n\n## Description\n\n...".into(),
}
```

Surfaced via `veks pipeline <command> --help` and the `?` key in
the TUI. No external doc files — documentation is code.

### Options and Resources

```rust
OptionDesc {
    name: "source",
    type_name: "Path",
    required: true,
    default: None,
    description: "Input vector file",
    role: OptionRole::Input,
}

ResourceDesc {
    name: "mem",
    description: "Vector data buffers",
    adjustable: false,
}
```

### CommandDoc

Every command carries compiled-in markdown documentation:

```rust
CommandDoc {
    summary: "one-line for completions".into(),
    body: "# command\n\n## Description\n\n...".into(),
}
```

Accessed via `veks help <command>` or `--help`. Options and resources
tables are auto-generated from `describe_options()` and
`describe_resources()`. Test suite verifies every command has non-empty
docs mentioning all declared options.

### StreamContext

The execution environment passed to every command:

```rust
pub struct StreamContext {
    pub workspace: PathBuf,          // dataset directory
    pub dataset_name: String,
    pub ui: UiHandle,                // progress + logging
    pub governor: ResourceGovernor,  // memory/thread budgets
    pub defaults: HashMap<String, String>,  // pipeline variables
    pub threads: usize,
    pub cache: PathBuf,              // .cache/ directory
}
```

---

## 8.3 Resource Governance

### Problem

Unbounded memory growth during large-scale operations (e.g.,
predicate evaluation on LAION-400M: 400M metadata records ×
predicate count → OOM). The resource governor prevents this.

### Architecture

```
ResourceGovernor
├── memory budget (from system RAM or --resources mem=4G)
├── thread budget  (from CPU count or --resources threads=16)
└── per-step resource declarations (describe_resources)
```

### Governance protocol

1. Command declares resource requirements via `describe_resources()`
2. Governor allocates budget before step execution
3. Command queries `ctx.governor.current_or("mem", default)` for limits
4. Governor logs utilization after completion

### Operating bands

Memory pressure is classified into bands that drive governor decisions:

| Band | RSS % of ceiling | Governor action |
|------|-----------------|----------------|
| UNDERUSED | < 70% | May increase allocations toward ceiling |
| NOMINAL | 70–85% | Allocations stable |
| CAUTION | 85–90% | Begin reducing allocations |
| THROTTLE | 90–95% | Commands told to slow down |
| EMERGENCY | > 95% | Aggressive reclamation to prevent OOM |

### Strategies

| Strategy | Behavior |
|----------|----------|
| `maximize` (default) | Use as much budget as pressure allows |
| `conservative` | Start near floor, grow slowly |
| `fixed` | Lock at midpoint, no adjustment |

### Failure mitigation

| Scenario | Mitigation |
|----------|------------|
| RSS exceeds budget | Reduce batch size, switch to streaming |
| Thread contention | Governor caps thread pool size |
| Mmap pressure | Advise sequential/random as appropriate |
| Page cache thrashing | Detect storage type, adjust prefetch |

---

## 8.4 UI Eventing Layer

The pipeline decouples computation from display via an event-based
UI architecture.

### Event flow

```
Command code → UiHandle → UiEvent → UiSink (trait object)
                                         ├── PlainSink (non-TTY)
                                         ├── RatatuiSink (rich TUI)
                                         └── TestSink (unit tests)
```

### UiHandle

The facade used by command code:

```rust
// Progress bars
let pb = ctx.ui.bar(total, "processing");
pb.set_position(42);
pb.inc(1);
pb.finish();

// Logging
ctx.ui.log("message");

// Step log buffer (captured to dataset.log)
ctx.ui.clear_step_log();
// ... step runs ...
let lines = ctx.ui.drain_step_log();
```

### Event types

| Category | Events |
|----------|--------|
| Progress | Create, Update, Inc, SetMessage, Finish |
| Text | Log, Emit, EmitLn |
| Status | ResourceStatus, SetContext, Suspend, Clear |

### RatatuiSink

The rich terminal display:

- Multi-progress-bar layout with resource charts
- Keyboard shortcuts: `q` quit, `?` help, `/` info, `Tab` cycle views
- 30fps redraw with batch coalescing (multiple events per frame)
- Thread-safe: sink runs on its own thread, events queued via channel
- Downsampling: high-frequency progress updates throttled to display rate

### PlainSink

Non-interactive output (piped, CI):

- Progress bars rendered as periodic percentage lines
- No ANSI escape codes
- All log output to stdout/stderr

### TestSink

Deterministic output capture for unit tests:

- Events stored in `Vec<UiEvent>`
- No threading, no timing dependencies
- Assertions against captured event sequence

---

## 8.5 Facet Swimlane

The pipeline superset viewed as a swimlane diagram. Each facet code
occupies a vertical lane. Steps flow top-to-bottom. Cross-lane arrows
show data dependencies.

```
  B           Q           G           D           M           P           R           F
  ─           ─           ─           ─           ─           ─           ─           ─
  │           │           │           │           │           │           │           │
  count       │           │           │           │           │           │           │
  │           │           │           │           │           │           │           │
  prepare     │           │           │           │           │           │           │
  │           │           │           │           │           │           │           │
  shuffle ────┼───────────┤           │           │           │           │           │
  │           │           │           │           │           │           │           │
  extract ────┤  extract  │           │           │           │           │           │
  │           │  queries  │           │           │           │           │           │
  │           │           │           │           │           │           │           │
  │           ├───────────┤  compute  │           │           │           │           │
  │           │           │  knn ─────┤           │           │           │           │
  │           │           │           │           │           │           │           │
  │           │           │           │  generate │           │           │           │
  │           │           │           │  metadata │           │           │           │
  │           │           │           │     │     │           │           │           │
  │           │           │           │     │     │  generate │           │           │
  │           │           │           │     │     │  predicates           │           │
  │           │           │           │     │     │     │     │           │           │
  │           │           │           │     ├─────┼─────┤  evaluate      │           │
  │           │           │           │     │     │     │  predicates ───┤           │
  │           │           │           │     │     │     │     │          │           │
  ├───────────┼───────────┤           │     │     │     │     ├──────────┤  compute  │
  │           │           │           │     │     │     │     │          │  filtered │
  │           │           │           │     │     │     │     │          │  knn ─────┤
  │           │           │           │     │     │     │     │          │           │
  ├───────────┤           │           │     │     │     │     │          │  partition │
  │           │           │           │     │     │     │     │          │  profiles  │
  │           │           │           │     │     │     │     │          │     │      │
  │           │           │           │     │     │     │     │          │  ── re-expand ──
  │           │           │           │     │     │     │     │          │  compute-knn  │
  │           │           │           │     │     │     │     │          │  (per label)  │
```

When `partition_oracles = true`, the pipeline has three phases:

1. **Phase 1** (core): generate data, compute default-profile KNN,
   evaluate predicates, compute filtered KNN

2. **Phase 2** (partition preparation): `partition-profiles` extracts
   base vectors per label and registers profiles in dataset.yaml

3. **Phase 3** (partition KNN): pipeline re-loads dataset.yaml, sees
   new partition profiles, re-expands `per_profile` templates
   (compute-knn, verify-knn) for each partition. The same compute-knn
   code path with SimSIMD + batching + caching runs for each partition.
   Already-completed steps skip via freshness checks.

### Step activation conditions

| Step | Active when |
|------|------------|
| count-vectors | always |
| prepare-vectors | !no_dedup |
| generate-shuffle | seed != 0 |
| extract-base | needs_base_extract |
| extract-queries | queries materialized |
| compute-knn | G materialized |
| generate-metadata | M synthesized |
| generate-predicates | P synthesized |
| evaluate-predicates | M + P present |
| compute-filtered-knn | R + B + Q present |
| scan-zeros | prepare skipped |
| scan-duplicates | prepare skipped |
| partition-profiles | partition_oracles + M + F present |

To derive a specific pipeline configuration, remove steps whose
conditions are not met. The remaining steps form the DAG.

---

## 8.6 Pipeline DAG Configurations

Common configurations arising from different input combinations:

| Config | Inputs | Steps |
|--------|--------|-------|
| B only | base vectors | count → scan → vvec-index → merkle → catalog |
| B self-search | base (self_search) | count → prepare → shuffle → extract-B/Q → compute-knn → verify → ... |
| B+Q | base + queries | count → prepare → extract-B → extract-Q → compute-knn → verify → ... |
| B+Q+GT | base + queries + ground truth | count → scan-zeros → scan-dups → (symlinks) → verify-knn → ... |
| B+Q+GT+synth | above + synthesize_metadata | above + generate-M → generate-P → evaluate-P → verify-sqlite → compute-filtered → verify-filtered → ... |
| above + partitions | above + partition_oracles | above + partition-profiles → ... |

### Facet inference rules

| Inputs provided | Facets inferred |
|----------------|----------------|
| B | B |
| B + Q | B Q G D |
| B + Q + GT | B Q G (D if distances provided) |
| B + Q + M | B Q G D M P R F |
| B + Q + GT + synthesize | B Q G M P R F |

### Variable reference rules

| Variable | Set by | Available after |
|----------|--------|----------------|
| `${vector_count}` | count-vectors | step 1 |
| `${clean_count}` | prepare-vectors | dedup complete |
| `${base_count}` | count-base | extraction complete |
| `${seed}` | defaults section | always |
| `${cache}` | runner | always (.cache/ path) |
| `${profile_dir}` | per_profile expansion | per_profile steps |
