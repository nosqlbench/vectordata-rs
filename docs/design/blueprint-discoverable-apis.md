# Blueprint — making a library's canonical API discoverable

**What this is:** a project-independent method for making sure that a
person reading a library's published docs, or a coding agent reading
its registry copy, finds the one right call for each common task — and
for keeping that true as the code changes.

**Where it comes from:** the vectordata work recorded in
`srd-api-discoverability.md`, which applied it to one crate. That SRD
holds the project-specific requirements; this file keeps only what
carries over. Examples are Rust, because that is where it was proven,
but every mechanism has an equivalent elsewhere (noted where it
matters).

## 1. The failure this prevents

A consumer needs a capability. The library has it. The consumer
concludes that it does not, and writes their own — usually worse, and
always duplicated. It happens through four predictable paths:

1. **The first primitive that works ends the search.** Nothing on that
   primitive says a better call exists.
2. **Location and shape signal "not for you".** The real call lives in a
   module documented as CLI plumbing, takes command-line-shaped strings,
   returns an exit code, or prints. A reader reasonably takes it for
   internal.
3. **A near miss on the obvious type.** A method that does almost the
   job reads as the feature with a piece missing, which implies the
   piece does not exist anywhere.
4. **Visible behaviour is never traced back.** The CLI visibly does the
   thing; nothing invites the reader to look at what the command calls.

Each path is a property of the library, so each is fixable in the
library. None is fixed by writing more prose.

## 2. Principles

1. **One canonical call per task.** Name the tasks. For each, exactly
   one recommended call; every overlapping public item says so and
   names it.
2. **The call lives on the type the caller already holds.** If callers
   hold a `View`, fetching lives on `View`, not in a helper module.
3. **Library first, CLI on top.** Every command is a thin adapter over a
   library function that returns a value and reports progress through a
   sink the caller passes. The library never picks an exit code, never
   prints, never prompts, never exits.
4. **Shapes tell the truth.** A library function takes typed input and
   returns `Result<Report, Error>`. If it looks like a command, readers
   will treat it as one.
5. **Traps are designed out, or stated where they bite.** Anything that
   silently costs a lot (a whole-file download, a full scan, a network
   round trip per item) is either made impossible, made explicit through
   a parameter or a sink, or stated in the first doc line of the call
   that triggers it.
6. **Discoverability is tested.** Links resolve, public items are
   documented, the task index names items that exist, the CLI map
   covers every command, and the set of overlapping entry points is
   budgeted. All of it fails the build when it drifts.
7. **Write for two readers.** A human on the rendered docs, and an agent
   that greps names and reads the crate root, module heads and any
   guide shipped in the package. Put the index where the agent looks
   first.

## 3. API shape — do this before writing docs

Documentation cannot rescue a misleading API; it can only describe it.
Fix the shape first.

- **Split plan from execute** for anything expensive. `plan(request) →
  Plan` opens, resolves and costs everything without side effects;
  `Plan::execute(sink) → Report` does the work. The canonical call is
  the two composed. Callers get the cost up front, CLIs get a natural
  `--plan`/`--dry-run` that runs the same code path minus the effects,
  and refusals (unknown inputs, unaffordable requests, consent needed)
  happen for the whole run before any of it starts.
- **Progress through a sink the caller passes.** A trait with one
  method taking an event enum, implemented by a silent sink, a terminal
  renderer, and — via a blanket impl — any closure:

  ```rust
  pub trait Progress { fn on_event(&mut self, e: &Event<'_>); }
  impl<F: FnMut(&Event<'_>)> Progress for F { fn on_event(&mut self, e: &Event<'_>) { self(e) } }
  pub struct Silent;
  ```

  Make the events cumulative and clamped (bytes so far *for this item*,
  never past its planned total), so consumers do not reimplement
  summing across ranges or clamping overshoot. Ship the CLI's renderer
  as a library type writing to any `Write`, so a capturing test can
  read it back and a library caller can show the same display.
- **Diagnostics are values.** "Did you mean", skipped entries,
  unreadable sources: return them in the error or on a `diagnostics()`
  accessor. The CLI prints them; a library caller decides. A function
  that exits the process from library code is a bug.
- **Typed errors for the cases callers branch on** (`UnknownItem {
  suggestions }`, `Ambiguous { matches }`, `Refused { why, cost }`),
  with messages that name the fix.
- **Keep the old entry points working, as thin wrappers.** Reimplement
  each duplicate over the canonical call, mark it deprecated with a
  note naming the replacement, and never remove it in the release that
  introduces the replacement. Pin each with a test that is explicitly
  allowed to use the deprecated form.
- **Accept the vocabulary users already have.** If items have aliases in
  configuration files, accept them in the API too. If the domain has
  several words for the operation ("fetch", "download", "precache",
  "prefetch"), pick one for the call and mention the others in its
  docs, so a search for any of them lands there.

## 4. Documentation artifacts

1. **Crate root (or package index) in this order:**
   - what it is and who calls it — one paragraph;
   - **Common tasks** — a table of *task | call | example*, each call
     a checked link, each example a snippet or an `examples/` file;
   - **Layers** — the stack from the outside in, saying which layers
     applications use;
   - **Traps** — each surprising behaviour, linked to the item that has
     it;
   - **Feature flags** — what each gates, and explicitly what does
     *not* need it.
2. **Module heads.** The first line of every public module says
   `Library API.`, `CLI support.` or `Internal.` A module that is both
   names its library items. Keep each module's docs in one place — in
   Rust, docs on both the `mod` declaration and inside the file are
   concatenated, render twice, and resolve links in different scopes.
3. **First lines of non-canonical items.** `Low-level: <what it does>.
   Most callers want [<canonical>], which <what it adds>.` Deprecated
   duplicates: `Deprecated: use [<canonical>] …`. Don't hide useful
   public items; hiding is for items that must be public for internal
   reasons only.
4. **Examples** for every Common-tasks row longer than a snippet,
   compiled in CI.
5. **A short README** that points at the Common tasks index instead of
   repeating it. A README that repeats the API goes stale first — in
   the vectordata case it was calling methods that no longer existed.
6. **An agent guide shipped in the package** (`AGENTS.md`):
   - *Do this* — "to do X, call Y", with full paths, mirroring Common
     tasks;
   - *Do not hand-roll* — the capabilities people reimplement, each
     with what the library already handles;
   - *CLI ↔ library* — one row per command, naming the call behind it
     (the single most effective row in the file);
   - *Traps*;
   - a paragraph for consumers to paste into their own agent
     instructions.
7. **Reference docs** for the layer model and the CLI map, and design
   records for each "one canonical path" decision, so the next
   addition is weighed against it.

## 5. Enforcement — make drift fail the build

| Property | Mechanism (Rust) | Elsewhere |
|---|---|---|
| Links in docs and guide resolve | `cargo doc` with `-D warnings`; render the guide as a doc module, `#[doc = include_str!("../AGENTS.md")] pub mod _agents {}`, so its links are checked too | doc builder in strict mode; link checker over the guide |
| Public items documented | `#![warn(missing_docs)]` per crate, promoted by `-D warnings` in the doc build | docstring coverage linter |
| Snippets compile | doctests, `no_run` for network, never `ignore`; `cargo build --examples` | doctest runner, examples in CI |
| Task index names real items | every Common-tasks row must contain a link (test parses the table), so rustdoc checks it; named example files must exist | parse the table, resolve names by import |
| Every command has a map row | test walks the real command tree (the parser's spec, not `--help` text) and diffs it against the guide's table, both ways | same, against the CLI framework's command registry |
| Library code does not print or exit | test reads the source, strips test-only code and comments, and fails on print/exit calls in any file under a module labelled `Library API.`, with a short allow-list of renderers, each with a reason, that fails if an entry goes stale | same, with the language's print/exit primitives |
| Module heads carry a label | test reads every module's first doc line | same |
| Overlap budget | test lists every public fetch-like (task-like) entry point by name with its role and whether it is a deprecated duplicate; new names fail until budgeted; deprecated-ness must match | same, by public symbol |
| Guide ships in the package | test runs `cargo package --list` and requires `AGENTS.md` and `examples/` | inspect the built artifact |
| Agents actually find it (optional) | a checked-in list of task prompts; before a release, a fresh agent given only the package must answer each with the listed call | same |

Two notes on writing these tests:

- **Prove each guard fails.** Insert the violation it exists for (a
  print in a library file, an undocumented item), watch it fail, then
  restore the file. A guard that has never failed has not been tested.
- **Prefer the real artifact to a copy.** Walk the CLI's parser spec,
  not its help text; read the shipped guide, not a duplicate list in
  the test.

## 6. Order of work

1. **API shape** (§3). Without it the docs only describe the problem.
2. **Common tasks index and examples, with their checks.** This fixes
   discovery even before every shape change lands.
3. **Agent guide and its packaging check.** Cheapest, but it only
   works when read, so it backs up the first two rather than replacing
   them.
4. **The remaining guards** — module labels, print scanner, overlap
   budget — once there is something stable for them to protect.

## 7. What the vectordata pass turned up beyond its plan

Doing this properly is also an audit. Expect to find, as this pass did:

- a reported trap that was real and worse than described (opening a
  reader on a remote variable-length file downloaded all of it to build
  an index), now refused with a message naming the fix;
- a name mismatch between the configuration key, the canonical name and
  what the API reported for the same item;
- doc blocks detached from their item by later insertions, so rustdoc
  attached them to whatever came next — documenting the wrong item and
  leaving the right one bare;
- comments that contradicted the code they described;
- a zero-copy fast path that read the wrong file for sharded data;
- library modules that printed, prompted or exited the process;
- a CLI command that reimplemented a library capability instead of
  calling it.

Fix the code each one points at, not just its documentation.

## 8. Templates

Common tasks row:

```text
| Fetch chosen items with progress | [`View::fetch`] with [`Request::items`] | `examples/fetch_with_progress.rs` |
```

First doc line of a non-canonical item:

```text
/// Low-level: <what it does>. Most callers want [`<canonical>`], which <what it adds>.
```

Module head:

```text
//! <Library API. | CLI support. | Internal.> <one sentence>. Applications use [`<item>`]; the rest supports `<binary> <cmd>`.
```

`AGENTS.md`:

```text
# <crate> for agents
## Do this
- <task>: [`<full::path>`](crate::<full::path>) — <note>
## Do not hand-roll
- <capability> — <what the library already handles>; use `<path>`
## CLI ↔ library
| Command | Library call |
|---|---|
| `<binary> <cmd>` | [`<full::path>`](crate::<full::path>) |
## Traps
- <behaviour> — <how to avoid it>
## For your own project's agent instructions
> Before writing <capabilities> against `<crate>`, read its `AGENTS.md` …
```
