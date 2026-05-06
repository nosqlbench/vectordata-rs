# 11a. Completion Functional Spec — MetricsQL × Bash

A hyper-contextual functional specification for tab-completion
behavior in the `veks-completion` crate, with MetricsQL as the
primary worked grammar. Companion to **§11. Shell Completions**
(architecture / API surface) — this doc covers the *behavioral
contract*: for every cursor position the user can be in, what
candidates must be produced, and how those candidates must be
shaped so the shell's own heuristics don't corrupt the result.

This doc exists because tab completion has two distinct sources
of behavior — the engine (which decides *what* to suggest) and
the shell (which decides *how the suggestion is integrated*) —
and the contract between them is non-obvious. Bug-by-bug
iteration converges slowly. The spec below is the converged
target.

---

## A. Bash / Readline Behavior Catalog

These are the shell-side forces our engine must work *with*. They
are not documented in one place in bash itself; what follows is
the empirically observable contract we depend on.

### A.1 Word Splitting (`COMP_WORDBREAKS`)

Bash uses `COMP_WORDBREAKS` to split the line into "words" before
calling our completion function. Default is
`" \t\n\"'><=;|&(:"`. The "current word" is the slice of the line
between the most recent wordbreak before the cursor and the
cursor itself.

Bash uses this same set when **splicing** the candidate back into
the line: `COMPREPLY` entries replace the current word. So the
candidate must be shaped to include whatever the user already
typed *back to the most recent wordbreak* — otherwise the splice
truncates it.

Our engine sets `COMP_WORDBREAKS=$' \t\n<>;|&'` locally in the
hook. Stripped: `' " = ( :`. Reasons:
- `' "` — let shell-wrapper quotes survive intact.
- `=` — keep `key=value` as one token.
- `(` — keep `func(arg` as one token (nested calls).
- `:` — keep `[5m:1m]` subquery step as one token.

### A.2 Quote-Balance Tracking (Readline)

**Independent of `COMP_WORDBREAKS`**, readline tracks open quote
state as it scans the line. When the user invokes completion
inside an open `'` or `"` and the chosen candidate is a *single
match*, readline appends the closing quote to "fix" the line. It
also adds a trailing space.

This is the source of the bug:

```
Line:    metricsql query 'abs(http_request_duration_secon
Engine:  'abs(http_request_duration_seconds_bucket           ← single candidate
Bash:    metricsql query 'abs(http_request_duration_seconds_bucket' 
                                                            ↑↑
                                                            readline auto-closed
                                                            + trailing space
```

There is **no `compopt` flag that disables this** for arbitrary
non-filename completion. The only reliable mitigations are:
1. Return ≥2 candidates with a common prefix → no single-match,
   no auto-substitution, no auto-close.
2. Return a candidate that already includes the closing quote
   (when the expression is genuinely complete — see TERMINAL
   intent in §B.4).
3. Use `bind -x` (a different completion mechanism that bypasses
   readline's match logic entirely).

We use **(1) and (2)**. (3) would require a much larger hook and
is reserved for a future iteration if needed.

### A.3 Single-Match Auto-Substitution + Space

When `COMPREPLY` has exactly one entry that strictly extends the
current word, bash substitutes it and (unless `-o nospace`) adds
a trailing space. We use `complete -o nospace` globally so no
candidate gets an unwanted trailing space.

But: even with `-o nospace`, the **quote-close from §A.2** still
fires. `nospace` controls only the literal trailing space, not
readline's quote balancing.

### A.4 Multi-Match Common-Prefix Completion

When `COMPREPLY` has ≥2 entries:
- First tab: bash extends the current word to the longest common
  prefix shared by all entries. No quote-closing. No trailing
  space.
- Second tab: bash displays the list of entries.

This is the regime we want to be in for any "open context, more
typing expected" scenario (which is most of MetricsQL completion).

### A.5 `compopt` Options That Matter

| Option       | Effect                                                       | Used? |
|--------------|--------------------------------------------------------------|-------|
| `-o nospace` | Don't append a space after a single-match completion.        | YES   |
| `-o nosort`  | Don't alphabetise — preserve `COMPREPLY` order.              | YES   |
| `-o noquote` | Don't quote completed words. **Filenames only** — useless to us. | no |
| `-o filenames` | Treat candidates as filenames (adds `/` to dirs, etc.).    | no    |
| `-o plusdirs` | Also add directory matches.                                 | no    |
| `-o default` | Fall back to default filename completion if no matches.      | no    |

### A.6 Hook Lifecycle

```
user types … <TAB>
  ↓
bash populates COMP_LINE, COMP_POINT, COMP_WORDS, COMP_CWORD
  ↓
bash invokes _<app>_complete (our function)
  ↓
hook sets local IFS, local COMP_WORDBREAKS
  ↓
hook calls binary with COMP_LINE + COMP_POINT
  ↓
binary returns candidate strings on stdout, one per line
  ↓
hook assigns COMPREPLY=(…)
  ↓
bash splices COMPREPLY[0] (single match) or completes common prefix
```

Note: the binary does **not** see COMP_WORDS — it reparses
COMP_LINE itself. This insulates us from differences between
shells (zsh, fish) that tokenise differently.

---

## B. Completion Intent Taxonomy

(Repeated from §11.13 for self-containment, expanded here.)

Every candidate fits one of four intents. Intent governs:
- Whether the candidate ends in an open-context character (`(`,
  `{`, `[`, operator) or a closed-context one.
- Whether the cursor lands inside or outside the new context.
- Whether the engine should emit ≥2 candidates (to suppress
  readline auto-close — see §A.2).
- Whether the outer wrapper quote may legitimately close.

### B.1 OPEN

Opens a new sub-context. Cursor lands inside. Examples:

| Trigger context           | Candidate            | Why OPEN              |
|---------------------------|----------------------|-----------------------|
| Function name typed       | `delta(`             | enters arg list       |
| After metric name         | `up{`                | enters label matcher  |
| After metric name         | `up[`                | enters range selector |
| After `[5m`               | `[5m:`               | enters subquery step  |
| Aggregation modifier      | `by(`                | enters label list     |

Required: must NEVER be a single candidate when wrapper quote is
open (would trigger §A.2). Pair with at least one alternative
continuation.

### B.2 APPEND

Extends the current context with a token whose semantics are
"the user can keep going from here". Examples:

| Trigger context             | Candidate                |
|-----------------------------|--------------------------|
| Top of expression           | `up` (a metric name)     |
| Inside `{`                  | `job=` (label key + op)  |
| Inside `[`                  | `5m` (a duration)        |
| After comparison op         | `bool`                   |
| After `}` or `)`            | `+`, `and`, `offset`     |

Required: candidate is a token only — no trailing space, no
trailing context-opener. The user types more or moves on.

### B.3 CLOSE

Terminates a context but the expression continues. Examples:

| Trigger context               | Candidate           |
|-------------------------------|---------------------|
| Inside `delta(metric`         | `delta(metric)`     |
| Inside `up{job="x"`           | `up{job="x"}`       |
| Inside `up[5m`                | `up[5m]`            |

Required: candidate ends with the closer. If there's an outer
context still open (e.g., the wrapper quote), it must NOT close
— this is just closing the inner context.

### B.4 TERMINAL

The user's *whole expression* is complete. The shell wrapper
quote (if any) may close. Examples:

| Trigger context                                          | Candidate                       |
|----------------------------------------------------------|---------------------------------|
| Inside `'abs(metric_name`                                | `'abs(metric_name)'`            |
| Inside `"up{job="prom"`                                  | `"up{job=\"prom\"}"`            |
| Inside `'rate(http_requests_total[5m])` (already closed) | `'rate(http_requests_total[5m])'` |

Required: candidate includes the matching outer wrapper closer.
Always offered alongside OPEN/APPEND alternatives so the user
can choose: "I'm done" vs "I want to keep going".

### B.5 The Multi-Candidate Rule

> **For any context where the wrapper quote is open, the engine
> must emit ≥2 candidates whenever it has at least one
> grammar-valid suggestion.**

This is the core defense against §A.2 auto-close. The rule is
not "always emit 2"; it's "emit grammar-valid continuations
covering the natural next steps". For a metric-name completion
inside a function call, that means: bare metric (APPEND),
metric+`{` (OPEN labels), metric+`[` (OPEN range), metric+`)`
(CLOSE function), and the wrapper-closing TERMINAL form. Five
candidates from one logical match — and bash treats them as a
multi-match, displays the list, no auto-close.

---

## C. MetricsQL Grammar Context Catalog

For every cursor position the user can be in, this section lists
the grammar context, the intents we should emit, the candidate
shapes, and any quirks. Where the current implementation is
**incomplete**, it's flagged with **GAP**.

Notation: `<cursor>` marks the cursor; `…` means "any prefix".

### C.1 Top of Expression — Empty Prefix

```
metricsql query '<cursor>
metricsql query 'sum(<cursor>     # arg position, also top-of-expr
```

Tap 1 (cold): metric names only — APPEND set, "what data do I
have?".
Tap 2 (rapid): + functions — OPEN set, "what builders exist?".
Tap 3+: + keywords (`WITH`, `with`) — full vocabulary.

**Each metric should expand into the multi-candidate set:**
- `metric_name`        — APPEND
- `metric_name{`       — OPEN (label match)
- `metric_name[`       — OPEN (range selector)
- `metric_name)`       — CLOSE (only if `paren_depth > 0`)
- `'metric_name'`      — TERMINAL (only if wrapper open and the
                          metric stands alone as a complete
                          expression — i.e., `paren_depth == 0`)

Each function should expand into:
- `function(`          — OPEN (almost always; the call is the entry)
- `function()`         — CLOSE+TERMINAL combo (rarely useful;
                          omit for now)

**GAP**: current implementation emits only the bare metric and
the bare `function(`. Single-candidate cases trigger §A.2.

### C.2 Top of Expression — With Prefix

```
metricsql query 'abs<cursor>
metricsql query 'http_<cursor>
```

Same as C.1 but filter both metric_names and METRICSQL_FUNCTIONS
by the prefix. Multi-candidate rule still applies — for a
unique-match prefix like `abs`, emit `abs(` plus alternatives
(see §C.1 expansion above).

**GAP**: same as C.1.

### C.3 Inside Function Call — Arg Position

```
metricsql query 'sum(<cursor>
metricsql query 'rate(http_<cursor>
metricsql query 'histogram_quantile(0.95, <cursor>
```

If function is in `METRICSQL_SCALAR_FIRST_ARG` and arg index is
0 → emit nothing (it's a literal number).

Otherwise → recurse into Top-of-Expression (§C.1/§C.2) but with
`paren_depth > 0` so the metric-name expansion includes
`metric_name)` (CLOSE) candidates.

**GAP**: current implementation falls through to §C.1/§C.2 but
without the `)` continuation. The user's reported bug:

```
Line:    metricsql query 'abs(http_request_duration_secon
Want:    candidates including:
           'abs(http_request_duration_seconds_bucket{
           'abs(http_request_duration_seconds_bucket[
           'abs(http_request_duration_seconds_bucket)
           'abs(http_request_duration_seconds_bucket)'      # TERMINAL
Got:     'abs(http_request_duration_seconds_bucket          # single → auto-close
```

### C.4 Inside Label Matcher `{…}` — Key Position

```
metricsql query 'up{<cursor>
metricsql query 'up{job="x", <cursor>
```

For each label key: emit four operator variants:
- `job=`   `job!=`   `job=~`   `job!~`

These ARE the multi-candidate set (4 per key × N keys), so the
multi-candidate rule is satisfied automatically.

**Gap**: minor — could also emit `}` (CLOSE) and `,` (continuation
in same matcher). Today we don't.

### C.5 Inside Label Matcher `{…}` — Value Position (No Quote Open)

```
metricsql query 'up{job=<cursor>
```

Emit each label value wrapped in `"…"`:
- `"prometheus"`  `"node_exporter"`  …

Multi-candidate rule satisfied (N values).

**Gap**: should also emit `}` (CLOSE — close matcher with no
value), and `,` is N/A (no value yet). Today no `}` candidate.

### C.5b Inside `{…}` — Between Matchers (After Closed Value)

```
metricsql query 'up{job="prometheus"<cursor>
metricsql query 'avg(http_requests{job="x"<cursor>
```

The cursor sits right after a *closed* `"…"` value pair (or
after a numeric literal). The grammar permits only `,` (next
matcher) or `}` (close matcher) here. **NOT** another label key
— without a separator, that would be a syntax error.

Required candidates:
- `,` — APPEND, next matcher
- `}` — CLOSE the matcher
- `})` — CLOSE matcher + enclosing func call (when `bs.paren > 0`)
- `}<wrapper>` — TERMINAL (when matcher close is the only outer
  scope still open and wrapper is open)
- `})<wrapper>` — CLOSE all + TERMINAL (when matcher + one func
  call are the only outer scopes and wrapper is open)

**Was a P0 bug (now fixed)**: branch (3)'s default arm computed
`target_start = label_key_start_in_brace`, which points at the
position right after the `{`. Splicing a label-key candidate
there would replace the existing `key="value"` pair. Bash's
common-prefix completion across many candidates would then
truncate the line to `…{` — destroying the user's typed value.
Discriminator: `brace_between_matchers(before)` — see §H.1.



```
metricsql query 'up{job="prom<cursor>
```

Emit each matching value as a bare string (no quote — the open
quote is already there). For unique-match cases, multi-candidate
rule must be enforced. Options:
- Add `"` (TERMINAL of the value, leaves user inside `{…}`).
- Add `"}` (CLOSE both string and matcher).
- Add `","` (CLOSE string, then `,`, then re-open for next value
  — actually next *key*).

**GAP**: emits only the bare-string variants; if there's only one
matching value (e.g. `'up{job="prom`) → single match → auto-close
of wrapper. Need at least the `"` continuation as a second
candidate.

### C.7 Inside Range Selector `[…]`

```
metricsql query 'rate(up[<cursor>
metricsql query 'rate(up[5<cursor>
```

Emit time-unit candidates: `s`, `m`, `h`, `d`, `w`, `y`, `i`,
`ms`. Filtered by the (digits-stripped) ident prefix.

After a duration unit is typed, also emit `:` (subquery step
opener). Today we do this.

**Multi-candidate rule**: 8 units → no auto-close issue.

### C.8 Inside Subquery Step `[5m:…]`

```
metricsql query 'rate(up[5m:<cursor>
metricsql query 'rate(up[5m:1<cursor>
```

Same time-unit set. Today we do this.

### C.9 Aggregation Modifier Group `by(…)` / `without(…)` / `on(…)` / `ignoring(…)` / `group_left(…)` / `group_right(…)`

```
metricsql query 'sum by (<cursor>
metricsql query 'a + on(<cursor>
```

Emit label keys, comma-separated. Today emits bare keys.

**Gap**: when only one key matches the prefix, single-match
auto-close fires. Need at least `,` and `)` as alternatives.

### C.10 After `@` Modifier

```
metricsql query 'rate(up[5m]) @ <cursor>
```

Emit `start()`, `end()`. Two candidates → no auto-close. ✓

### C.11 After `offset` Keyword

```
metricsql query 'rate(up[5m]) offset <cursor>
```

Emit time-unit suggestions. Multi-candidate set. ✓

### C.12 After Comparison Op (`> < >= <= == !=`)

```
metricsql query 'up > <cursor>
```

Emit `bool` modifier + top-of-expression set (functions +
metrics). Multi-candidate. ✓ (assuming top-of-expression returns
multi-candidate set per §C.1).

### C.13 After Aggregation Function (`sum`, `avg`, `count`, …)

```
metricsql query 'sum <cursor>
```

Emit `by`, `without`, `(`. Today emits `by`/`without`.

**Gap**: missing `(` (open paren, alternate placement: `sum(expr)`).
Minor; user can type `(` themselves.

### C.14 After `)`, `}`, `]` (Post-Expression Positions)

```
metricsql query 'up{job="x"} <cursor>
metricsql query 'rate(up[5m]) <cursor>
metricsql query 'up[5m] <cursor>
```

Emit:
- Binary operators: `+`, `-`, `*`, `/`, `%`, `^`, `==`, `!=`,
  `<`, `>`, `<=`, `>=`, `and`, `or`, `unless`
- Modifiers: `offset`, `@`, `keep_metric_names`
- Aggregation modifiers: `by`, `without` (only after `)`)
- Range selector `[` (only after `}`)
- Wrapper-close TERMINAL: `'…)'`, `'…}'`, `'…]'` — single-quote
  closers when wrapper is open and the expression is now
  syntactically complete.

Today: emits the binops + modifiers but NOT the TERMINAL
closer.

**GAP**: TERMINAL closer (the user's "I'm done" signal). Without
it, the user has to manually type the closing wrapper quote
without engine help — and for users learning the grammar, this
is the moment they're least sure whether the expression *is*
complete.

### C.15 Inside Open Wrapper Quote at End-of-Expression

```
metricsql query 'up<cursor>
```

The metric is a complete expression on its own. Tap should
offer:
- `up`              — APPEND, keep going
- `up{`             — OPEN, add labels
- `up[`             — OPEN, range
- `up'`             — TERMINAL, done

Today: emits only `up`.

**GAP**: same as §C.1 — multi-candidate expansion needed.

### C.16 WITH Templates

```
metricsql query 'WITH (<cursor>
metricsql query 'WITH (commonExpr = …) <cursor>
```

Today: not handled — `WITH` is in `METRICSQL_KEYWORDS` but no
context-aware completion follows.

**GAP**: full WITH-template completion (template name → `=` →
expression → `,`/`)` → top-of-expression). Sizeable; defer to a
follow-up unless WITH usage is a priority.

---

## D. Cross-Cutting Behaviors

### D.1 The Wrapper-Quote Lifecycle

```
Stage 1   metricsql query <cursor>            no wrapper yet
Stage 2   metricsql query '<cursor>           wrapper open, expression empty
Stage 3   metricsql query '<expr><cursor>     wrapper open, building
Stage 4   metricsql query '<complete>'<cursor> wrapper closed, expression done
```

Engine behavior must differ by stage:
- Stage 1 → top-of-expression candidates with NO wrapper prefix.
- Stage 2/3 → all candidates carry the leading `'`.
- Stage 3 → multi-candidate rule applies (avoid auto-close).
- Stage 3 → TERMINAL candidates (`'…)'`, `'…'`) are explicit
  user-facing options.
- Stage 4 → bash sees a closed expression; engine offers
  post-expression suggestions OUTSIDE the wrapper (next
  argument, redirection, EOL).

### D.2 The "Single Logical Match" Trap

Whenever a context yields exactly one grammar-valid candidate
(e.g., a unique metric prefix, a unique function prefix,
exactly one matching label value), bash goes into single-match
mode (§A.2). Inside an open wrapper, that means auto-close.

**Required engine behavior**: enforce the multi-candidate rule
(§B.5) by *expanding* a unique grammar match into its natural
continuation set (intent variants per §B.1–B.4).

### D.3 The Splice-Prefix Invariant

Every candidate must start with `raw_line[shell_word_start..target_start]`.
The `splice_candidate` helper enforces this. With raw-mode
WORDBREAKS (§A.1), `shell_word_start` is the byte after the
last `<space|tab|newline|<|>|;|||&>` before the cursor.

For typical metricsql lines (`metricsql query '…`), this means
the wrapper quote and everything after it is part of every
candidate. Tests assert exact splice-ready strings.

### D.4 Tap Tier Layering for Subtree Providers

Subtree providers can layer their output by tap count:
- Tap 1 → primary candidate set.
- Tap 2 → primary + secondary (e.g., metrics + functions).
- Tap 3 → all (metrics + functions + keywords).

Engine support: `complete_rotating_with_raw` passes the full
`tap_count` to subtree providers (not the modulo-by-children
version), and `handle_complete_env` caps tap_count at
`SUBTREE_PROVIDER_MAX_TAPS = 3` so taps actually advance.

Required at provider level: each tier should still respect the
multi-candidate rule per context.

### D.5 Cursor-Inside-Existing-Token Editing

```
metricsql query 'up{j<cursor>ob="x"}
```

User edits the middle of an existing label key. Today: the
engine sees `up{j` as the before-cursor and `ob="x"}` as the
after-cursor; the candidate set is "label keys starting with
`j`" — but the splice-target is wrong (it would replace `up{j`
with `up{job=`, leaving `ob="x"}` as garbage).

**GAP**: the splice currently doesn't account for what's after
the cursor. For mid-token edits, the engine should either:
- Replace the entire token under the cursor (consume `j` and
  `ob` both).
- Or insert without consuming after-cursor text.

Today this case is broken silently. Listed for future work.

### D.6 Continuation From a Just-Completed Candidate

After the user accepts an OPEN candidate (e.g., `'delta(`), the
cursor lands at the end. Their next tab should immediately
offer arg-position candidates without requiring any typed
prefix. This is §C.3 — and it works today, *provided* the
multi-candidate rule fixes §C.3's auto-close issue first.

### D.7 Mixed Contexts

`metricsql` is a subcommand of a larger CLI. Outside `query`,
completion follows the regular CommandTree path. Inside
`query`, the metricsql provider takes over. The hand-off is
clean (subtree provider mechanism) but:

**Gap**: argv-position completion outside `query` is not
exhaustively spec'd here — see §11.6 for argv-parser behavior.
Cross-reference both when adding new top-level commands.

---

## E. Gap Summary (Implementation Punch List)

Severity tags: **P0** = currently broken in user-visible ways;
**P1** = correctness gap, no user has hit it yet; **P2** =
nice-to-have polish. Tick (✓) = fixed and covered by tests.

| #  | Gap                                                              | §    | Sev | Status |
|----|------------------------------------------------------------------|------|-----|--------|
| 1  | Single-candidate auto-close on metric name in func arg           | C.3  | P0  | ✓ |
| 2  | Single-candidate auto-close on top-of-expression unique match    | C.1/C.2 | P0 | ✓ |
| 3  | Single-candidate auto-close on unique label value (open string)  | C.6  | P0  | ✓ |
| 4  | TERMINAL (wrapper-close) candidates not offered post-expression  | C.14, C.15 | P1 | ✓ |
| 5  | Single-candidate auto-close in `by(…)` / `on(…)` etc.            | C.9  | P1  | (covered by §F.2 ghost) |
| 6  | Mid-token edit splice is wrong                                   | D.5  | P1  | open |
| 7  | `}` continuation in label-key/value position                     | C.4, C.5 | P2 | open |
| 8  | `(` after aggregation function                                   | C.13 | P2  | open |
| 9  | WITH-template completion                                         | C.16 | P2  | open |
| 10 | **Context discrimination — see §H**                              | H    | P0/P1 | partial (agg-func + by-modifier fixed) |

---

## F. Implementation Prescription (P0 Fixes)

### F.0 The Anti-Auto-Close Toolkit — Options Compared

The §A.2 auto-close trap is the single biggest correctness
issue. Below are every known technique to break the
single-match condition, ranked from "most semantic" to "most
hacky", with rationale for which one we adopt where.

| # | Technique                          | When it fits                        | Cost                                     | Where used |
|---|------------------------------------|-------------------------------------|------------------------------------------|------------|
| 1 | **Grammar-valid expansion**        | Context has natural continuations   | Per-context engine work; teaches grammar | §F.1, §F.4 |
| 2 | **Ghost-prefix candidate**         | Unique grammar match, no useful expansions | Tiny: one extra string                   | §F.2 (fallback)        |
| 3 | **Gratuitous sentinel** (e.g., `#`)  | Nothing else applies                | Visible noise in candidate list          | §F.2.bis (last resort) |
| 4 | `bind -x` mechanism                | We need total control over splice   | Heavy hook rewrite, separate bash branch | not now    |
| 5 | Pre-quote the candidate            | `compopt -o filenames`+filename rules | Adds `/` to dirs, breaks every other case | not now    |

**Ordering rule for engine code**: try (1) first per context.
If (1) yields exactly one candidate, fall back to (2). If (2)
also degenerates to one entry (e.g., the user's typed prefix
exactly equals the only completion), fall back to (3) — the
gratuitous sentinel.

The sentinel approach is intentional last-resort because it
shows up in the user's tab list as visible noise; the
grammar-valid and ghost approaches are invisible (the ghost is
the user's own typing; the grammar variants are useful).

### F.0.bis Progressive Disclosure

Emitting every grammar-valid continuation on every tap clutters
the candidate list:

```
metricsql query <TAB>
http_request_duration_seconds_bucket   http_requests_total       …  up
http_request_duration_seconds_bucket[  http_requests_total[      …  up[
http_request_duration_seconds_bucket{  http_requests_total{      …  up{
```

The user wanted the names view first, then the continuations on
a subsequent action. Tap-tier semantics provide this naturally:

| Condition                              | Tap 1               | Tap 2 (rapid)       |
|----------------------------------------|---------------------|---------------------|
| Multiple matching metrics              | names only          | names + continuations + functions |
| Partial prefix, **unique** metric match | name + continuations (immediately) | + functions |
| Typed ident **==** a metric name       | name + continuations (immediately) | + functions |

Rationale for the immediate-continuations exceptions:

- **`ident == metric` (full match)**: the user has typed the
  full metric name. Their next decision IS what comes after.
- **`is_unique_match` (partial but unambiguous)**: the typed
  prefix matches exactly one metric. The user has effectively
  pinned it down even mid-typing. Crucially, this case ALSO
  fixes a usability bug: with just a single bare-name candidate
  + ghost-prefix (§F.2), bash's longest-common-prefix algorithm
  computes the common prefix as = the typed text (the ghost
  doesn't extend it), so single-tab advancement gets stuck.
  Emitting continuations gives bash a richer common prefix
  (the full metric name) AND the multi-candidate set — single
  tab advances, second tab shows alternatives.
- **Multi-match with partial prefix**: continuations would
  multiply the candidate list (N_metrics × 5) while the user is
  still narrowing. Tap 1 stays clean; tap 2 reveals the full
  set.

This is implemented in `expand_metric_continuations` via the
gating predicate:

```rust
let show_continuations =
    ident == metric || is_unique_match || pp.tap_count >= 2;
```

The anti-auto-close defenses (§F.2 ghost, §F.2.bis sentinel)
still apply at all tap levels — progressive disclosure does not
weaken the readline-auto-close guarantee.

### F.1.bis The LCP-Advancement Rule

The §F.0/§F.2 multi-candidate guarantee solves *auto-close*
but creates a follow-on problem: when the candidate set is
just `[real, ghost]` and the ghost equals the typed text (or
even the smart `real - 1`), bash's longest-common-prefix
algorithm advances the line by 0 or 1 characters. The user
sees the menu but the line isn't completed — they still have
to type the missing chars manually.

**The categorical rule**: every per-context emission must
yield ≥2 candidates whose **longest common prefix extends to
a useful landing point** (typically the next syntactic
boundary the user wants to be at).

| Context                   | LCP must reach                | How emitted                                    |
|---------------------------|-------------------------------|-------------------------------------------------|
| Function name (unique)    | `func(`                       | `func(` + `func()` (`emit_function_candidates`) |
| Metric name (unique)      | `metric` (then continuations) | `expand_metric_continuations` emits 4–5 variants |
| Label key                 | `key=` (or operator variant)  | 4 operator variants per key                     |
| Label value               | `value`                       | bare + close-string + close-matcher variants    |
| Modifier-list label key   | `(label`                      | one per matching key (multi-match by default)   |

For function names, `func()` is the deliberate second
candidate: it's syntactically valid metricsql (nullary call
form), shares the `func(` prefix with the primary candidate,
and extends LCP past the entry-point paren — single tab
advances the user all the way to "ready to type args inside
the parens". The user sees both options; the `()` form is
rarely what they want, but it's harmless.

This is implemented via `emit_function_candidates`, applied
uniformly at every function-emission site (§C.1, §C.2, §C.3,
§C.12). New emission sites added in the future MUST use this
helper rather than the bare `func(` pattern.

### F.1 Multi-Candidate Expansion for Metric Names

Replace the metric-name emission in branch (6) of
`complete_metricsql` with an expansion helper:

```rust
fn expand_metric_continuations(
    pp: &PartialParse,
    target_start: usize,
    metric: &str,
    paren_depth: i32,
    wrapper_quote: Option<char>,
) -> Vec<String> {
    let mut out = vec![
        pp.splice_candidate(target_start, metric),                    // APPEND bare
        pp.splice_candidate(target_start, &format!("{}{{", metric)),  // OPEN labels
        pp.splice_candidate(target_start, &format!("{}[", metric)),   // OPEN range
    ];
    if paren_depth > 0 {
        out.push(pp.splice_candidate(target_start, &format!("{}{})", metric, "")));  // CLOSE func
    }
    if let Some(q) = wrapper_quote {
        if paren_depth > 0 {
            // CLOSE func + TERMINAL wrapper
            out.push(pp.splice_candidate(target_start, &format!("{}){}", metric, q)));
        } else {
            // TERMINAL wrapper directly
            out.push(pp.splice_candidate(target_start, &format!("{}{}", metric, q)));
        }
    }
    out
}
```

Apply at every metric-name emission site:
- §C.1 / §C.2 top-of-expression
- §C.3 function-arg recursion (paren_depth > 0)
- §C.12 after-comparison-op recursion

`wrapper_quote` is derived by re-reading the wrapper detection
that branch (1) already does at the top of `complete_metricsql`
— surface it as a value the rest of the function can see.

### F.2 Ghost-Prefix Fallback (Function Names + Anywhere Else Single-Match Survives)

When a context yields exactly one candidate after F.1's
grammar-valid expansion (e.g., a unique function-name prefix
like `'delt` → `'delta(`), append a "ghost" candidate equal to
the user's typed-so-far:

```
COMPREPLY:  'delta(    'delt
```

Bash sees two entries with common prefix `'delt`, completes the
common prefix only (no advancement, no auto-close), and on a
second tab shows both for the user to pick.

The ghost is **invisible noise** to the user — they see their
own typing as one of two "options" and the real candidate as
the other. They press tab once more, see the list, choose.

Encapsulate as a final-pass helper applied to every context
branch's return value:

```rust
/// Ensure ≥2 candidates whenever the wrapper quote is open,
/// so bash readline doesn't auto-close on a single match.
/// Step 1: if exactly one candidate, append a ghost equal to
/// the typed prefix. Step 2: if still one (the typed prefix
/// already equals the candidate), call `enforce_via_sentinel`
/// (§F.2.bis).
fn enforce_multi_candidate(
    pp: &PartialParse,
    out: &mut Vec<String>,
    wrapper_quote: Option<char>,
) {
    if wrapper_quote.is_none() || out.len() >= 2 { return; }
    let typed = pp.shell_current_word();
    if !typed.is_empty() && !out.iter().any(|c| c == typed) {
        out.push(typed.to_string());
        return;
    }
    enforce_via_sentinel(pp, out);
}
```

Apply at the end of `complete_metricsql`'s every return path.

### F.2.bis Gratuitous Sentinel — Last Resort

If (a) F.1 grammar expansion produced one candidate, AND (b)
F.2 ghost-prefix degenerated (the user's typed prefix exactly
equals the sole candidate, so the ghost would dedupe), bash
will still auto-close. Last resort: emit a deliberately
non-completion sentinel as the second candidate.

Rules for the sentinel:
- Must NOT be a valid grammar continuation (so the user can't
  accidentally accept it).
- Must NOT share a prefix with the real candidate beyond what
  the user already typed (so bash's common-prefix completion
  doesn't extend into the sentinel).
- Should be visually identifiable as "this is a placeholder, not
  a real option".

Recommended: a single `#` (MetricsQL doesn't use `#`; visually
short; clearly not a completion):

```rust
fn enforce_via_sentinel(pp: &PartialParse, out: &mut Vec<String>) {
    let typed = pp.shell_current_word();
    let sentinel = format!("{typed}#options");
    if !out.iter().any(|c| c == &sentinel) {
        out.push(sentinel);
    }
}
```

The user sees:
```
'delta(    'delta(#options
```

The `#options` entry is gibberish if accepted (would produce
`'delta(#options` on the line — also a no-op for metricsql
parsing since `#` isn't valid syntax there), but its presence
forces bash into multi-match mode. The user ignores it and
picks the real one.

This is intentionally ugly so we have motivation to prefer F.1
expansion or F.2 ghost wherever possible.

### F.3 TERMINAL Candidates Post-Expression

In branches §C.14 (after `)`, `}`, `]`), if a wrapper quote is
open AND the expression is syntactically complete (paren depth
0, brace depth 0, bracket depth 0), append:

```rust
out.push(pp.splice_candidate(pp.cursor_offset, &wrapper_quote.to_string()));
```

### F.4 Open-String Label Value (§C.6) Multi-Candidate

For label-value completions inside an open `"…`, after emitting
the bare-value candidates, append `"` (close-string) so the user
can pick "this value" or "I'm done with this string".

When inside the wrapper as well, also append `"}` (close string
+ matcher) and `"}'` (close all and TERMINAL wrapper).

### F.5 Tests

Each P0 fix gets a regression test:

```rust
#[test]
fn metric_name_in_func_arg_emits_multi_candidate_set() {
    let cands = engine_run("metricsql query 'abs(http_request_duration_secon");
    // Must contain ≥2 candidates so bash doesn't auto-close.
    assert!(cands.len() >= 2,
        "expected multi-candidate to suppress auto-close: {cands:?}");
    let want_open_labels = "'abs(http_request_duration_seconds_bucket{";
    let want_close_func = "'abs(http_request_duration_seconds_bucket)";
    let want_terminal   = "'abs(http_request_duration_seconds_bucket)'";
    assert!(cands.iter().any(|s| s == want_open_labels));
    assert!(cands.iter().any(|s| s == want_close_func));
    assert!(cands.iter().any(|s| s == want_terminal));
}
```

---

## G. Non-Goals

- **zsh / fish first-class support**: out of scope for this
  spec. The engine is shell-agnostic at the API level (it takes
  raw line + cursor); only the hook script and `DEFAULT_BASH_WORDBREAKS`
  are bash-specific. A zsh hook can be added later by mirroring
  §11.12 with zsh's `_arguments` mechanism.
- **Inline syntax error reporting**: the engine completes; it
  doesn't lint. If the user types invalid metricsql, candidates
  may be empty or strange — that's acceptable.
- **Completion of literal numbers**: scalar-arg positions in
  `histogram_quantile`, `topk`, `quantile`, etc. emit no
  candidates by design.
- **Server-side schema introspection**: the catalog trait
  (`MetricsqlCatalog`) is provided by the embedder. The engine
  doesn't reach out to a Prometheus instance.

---

## H. Context Discrimination — A Class of Bug

Many completion bugs share one shape: **the same lexical content
has different semantic meaning depending on bracket / paren
state**, and the engine's branches don't discriminate. The
fix pattern is always the same: gate the branch on a bracket-state
predicate that distinguishes the meanings.

### H.1 The Pattern

| Lexical content      | Meaning A                       | Meaning B                       | Discriminator                                    |
|----------------------|---------------------------------|----------------------------------|--------------------------------------------------|
| `agg_func` precedes  | alternate-placement modifiers   | function args                    | `bs.paren > 0` AND `enclosing_function_call == agg_func` → meaning B |
| `by`/`without`/`on`  | label-key list (modifier)       | the keyword itself, no list yet  | `bs.paren > 0` → meaning A; else don't fire     |
| `metric_name` typed  | continuations (`{`, `[`, …)     | partial prefix, still narrowing  | `ident == metric_name` OR `tap_count >= 2` → meaning A |
| `)` is last sig char | end of inner expr, more to come | end of full expression, terminal | `bs.paren == 0 && bs.brace == 0 && bs.bracket == 0` AND wrapper open → emit TERMINAL |
| `=` after key inside `{` | value-quote-start position  | typo of `==` operator outside `{`| `bs.brace > 0` AND trigger char is `=`/`~` → meaning A |
| Inside `{…}` brace   | fresh label-key position (after `{` or `,`) | between-matchers (after closed `"value"`) | `brace_between_matchers(before)` — segment from most recent `{`/`,` ends with closed `"` or digit AND contains `=` → meaning B |

### H.2 The Audit Rule

For every branch in `complete_metricsql` that conditions on a
**lexical signal** (`preceding_keyword`, `preceding_ident`,
`last_significant_char`, `trigger_char_of`), ask: **does this
signal have any other plausible meaning depending on bracket
state?** If yes, add the discriminator to the branch's guard.

Adding to a branch is safer than not adding: a missed
discriminator silently routes the wrong context to the wrong
candidate set (the `'avg(` bug). Over-discriminating just
falls through to the next branch (usually top-of-expression),
which is rarely catastrophic.

### H.2.bis Categorical Helpers vs. Per-Branch Logic

When the same emit-pattern shows up in multiple branches, factor
it into a helper applied uniformly. This makes future audits
easier (one place to inspect) and prevents the
discrimination-bug class from recurring per-branch.

Current helpers:

| Helper                              | Replaces                                  | Applied to                          |
|-------------------------------------|-------------------------------------------|-------------------------------------|
| `expand_metric_continuations`       | per-branch metric-name `format!`          | every metric-name emission site     |
| `emit_function_candidates`          | per-branch `format!("{f}(")` pattern      | every function-name emission site   |
| `append_close_continuations`        | per-branch `)` + `,` + TERMINAL inline    | every post-`)`/`}`/`]` branch       |
| `enforce_multi_candidate`           | per-branch ghost-candidate appending      | final-pass on every return path     |
| `cursor_past_closed_wrapper`        | (none — new safeguard)                    | engine entry, emits nothing if true |

Rule: when adding a new emission point that fits one of these
patterns, USE THE HELPER. New ad-hoc `format!("{f}(")` is a
regression — it loses the LCP-advancement guarantee.

### H.3 Known Discrimination Gaps (P1)

| Trigger                        | Current behavior              | Should also offer        |
|--------------------------------|-------------------------------|--------------------------|
| `sum (` (space before paren)   | `(by` / `(without`            | metric names + functions (ambiguous: alt-placement OR func-call args) |
| `sum by (job) ` (after mod list) | binops + AGGR_MODIFIERS     | `(` to start the agg's arg expression |
| `sum by ` (no paren yet)       | falls through to top-of-expr  | `(` as the natural next char |
| Inside `(` of any function     | top-of-expression             | `)` as a CLOSE option (already in `expand_metric_continuations` for metric matches; missing for the empty/function-only case) |

Each of these is "context A's branch fires, but the candidate
set should also include some of context B's candidates because
the user's next legal move is in B".

### H.4 Why This Class Recurs

The engine is **branch-driven**: a series of `if` clauses each
matching one context. Each clause assumes its trigger is
unambiguous. But MetricsQL grammar is layered (functions,
modifiers, expressions, lists, all using parens), and bracket
state is the only thing that disambiguates. Lexical signals
alone — preceding ident, last char, trigger char — can mean
multiple things simultaneously.

**The branch ordering becomes load-bearing**: if branch (4)
fires on a lexical signal that branch (5) would also have
matched, branch (4) wins. When branch (4)'s guard is too loose,
its win is wrong.

The fix pattern (always): tighten the guard with `bs.paren`,
`bs.brace`, `bs.bracket`, or `enclosing_function_call`. Or,
when the right answer is "both", refactor the branch to
*append* candidates rather than `return` them, then let
subsequent branches add more.

This last option (append-then-fallthrough) is a structural
improvement that would make the engine more robust to this
class of bug. Today every branch `return`s; a future refactor
could make per-context emission additive.

---

## I. Verification Strategy

When implementing the P0 fixes from §F:

1. **Unit tests** in `providers.rs` cover candidate-shape
   correctness. The `engine_run` helper exercises the full
   engine path (matches what bash invokes).
2. **`apply_substitution` round-trip** asserts that taking a
   candidate, splicing it via `shell_word_start`, produces the
   exact line the user expects to see.
3. **Multi-candidate count assertions** (`cands.len() >= 2`)
   prevent regressions of §A.2-class auto-close bugs.
4. **Manual interactive verification**: `eval "$(./target/debug/examples/metricsql completions)"`
   then test the user's reported scenarios in real bash. The
   tests verify candidate shape; only interactive testing
   verifies that bash *integrates* them as expected.

A complete PR should include all four checks for any
context-class change.
