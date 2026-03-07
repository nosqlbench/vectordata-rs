<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 08 — UI Eventing Layer

## 8.1 Overview

All user-facing output in veks flows through a **UI-agnostic eventing layer**.
Pipeline code never writes directly to stdout, constructs ANSI escape
sequences, or imports a rendering framework. Instead, every visual action is
expressed as an algebraic `UiEvent` value dispatched to a `UiSink` trait
object. Concrete sink implementations decide how — or whether — to render
each event.

This design decouples the pipeline engine and all commands from any specific
terminal library, enabling:

- **Plain-text output** for pipes, CI, and non-TTY environments.
- **Rich terminal rendering** (e.g. ratatui) when stdout is a TTY.
- **Deterministic testing** via an in-memory sink that records events for
  assertion.
- **Future backends** (web dashboard, log aggregation, GUI) without touching
  any pipeline code.

## 8.2 Architecture

```
 ┌──────────────────────────────────┐
 │  Pipeline code (commands, runner)│
 │                                  │
 │  ctx.ui.bar(100, "importing")    │
 │  ctx.ui.log("step 3 complete")   │
 │  pb.inc(1)                       │
 └───────────┬──────────────────────┘
             │  UiEvent values
             ▼
 ┌──────────────────────────────────┐
 │          UiHandle                │
 │    (facade over Arc<dyn UiSink>) │
 │    Clone-cheap, Send + Sync      │
 └───────────┬──────────────────────┘
             │  send(UiEvent)
             ▼
 ┌──────────────────────────────────┐
 │         dyn UiSink               │
 │                                  │
 │  ┌────────────┐ ┌─────────────┐  │
 │  │ PlainSink  │ │  TestSink   │  │
 │  │ (stdout)   │ │ (Vec<Event>)│  │
 │  └────────────┘ └─────────────┘  │
 │  ┌─────────────────────────────┐ │
 │  │  RatatuiSink (planned)      │ │
 │  │  Viewport::Inline rendering │ │
 │  └─────────────────────────────┘ │
 └──────────────────────────────────┘
```

Source modules:

| Module | Purpose |
|--------|---------|
| `ui/event.rs` | `UiEvent` enum, `ProgressId`, `ProgressKind` |
| `ui/sink.rs` | `UiSink` trait |
| `ui/handle.rs` | `UiHandle` facade, `ProgressHandle` RAII guard |
| `ui/plain_sink.rs` | Non-TTY sink (stdout text, no progress) |
| `ui/test_sink.rs` | Test harness sink (records events) |
| `ui/ratatui_sink.rs` | TTY sink (ratatui `Viewport::Inline` rendering) |
| `ui/mod.rs` | Module root, re-exports, and `auto_ui_handle()` |

## 8.3 Event Algebra

All visual primitives are expressed as variants of the `UiEvent` enum.
Events are `Send`, `Clone`, and carry no rendering state.

### 8.3.1 Progress lifecycle events

| Variant | Fields | Semantics |
|---------|--------|-----------|
| `ProgressCreate` | `id: ProgressId`, `kind: ProgressKind`, `total: u64`, `label: String` | Allocate a new progress indicator. `kind` is `Bar` (determinate, known total) or `Spinner` (indeterminate). |
| `ProgressUpdate` | `id: ProgressId`, `position: u64` | Set the absolute position of an existing indicator. |
| `ProgressInc` | `id: ProgressId`, `delta: u64` | Increment the position by `delta`. |
| `ProgressMessage` | `id: ProgressId`, `message: String` | Update the trailing label/message on an indicator. |
| `ProgressFinish` | `id: ProgressId` | Mark the indicator as complete and remove it from the display. |

### 8.3.2 Text output events

| Variant | Fields | Semantics |
|---------|--------|-----------|
| `Log` | `message: String` | A log line that scrolls above the progress region. Rich backends insert it above the fixed area; plain backends write a line to stdout. |
| `Emit` | `text: String` | Raw text with no trailing newline. |
| `EmitLn` | `text: String` | Raw text with a trailing newline. |

### 8.3.3 Status and lifecycle events

| Variant | Fields | Semantics |
|---------|--------|-----------|
| `ResourceStatus` | `line: String` | Update the resource-utilization status line (CPU, memory, I/O). |
| `SuspendBegin` | *(none)* | Begin a batch — suppress intermediate redraws until `SuspendEnd`. |
| `SuspendEnd` | *(none)* | End a batch — perform a single atomic redraw. |
| `Clear` | *(none)* | Clear the entire progress region. |

### 8.3.4 ProgressId

`ProgressId(u32)` is an opaque, `Copy`, `Eq`, `Hash` handle that identifies
a progress indicator within a session. It is allocated by `UiSink::next_progress_id()`
and referenced in all subsequent progress events. Being `Copy`, it can be
shared freely across threads without cloning an `Arc`.

### 8.3.5 ProgressKind

```rust
pub enum ProgressKind {
    Bar,      // determinate — known total, rendered as a gauge
    Spinner,  // indeterminate — no known total, rendered as a spinner
}
```

## 8.4 Sink Trait

The `UiSink` trait is the sole rendering interface. It is `Send + Sync` so
that an `Arc<dyn UiSink>` can be shared across threads (background resource
monitor, parallel segment processing, async download tasks).

```rust
pub trait UiSink: Send + Sync {
    /// Dispatch a single UI event.
    fn send(&self, event: UiEvent);

    /// Allocate a fresh ProgressId unique within this sink.
    fn next_progress_id(&self) -> ProgressId;
}
```

### REQ-UI-01: No rendering in pipeline code

Pipeline code, commands, the runner, and all non-UI modules MUST NOT import
any rendering library (indicatif, ratatui, crossterm, etc.) directly. All
visual output MUST flow through `UiSink::send()` via the `UiHandle` facade.

### REQ-UI-02: Sink implementations

Every `UiSink` implementation MUST handle all `UiEvent` variants. Unknown
or irrelevant variants SHOULD be silently ignored (not panic). Sinks MUST
be safe to call from any thread at any time.

### REQ-UI-03: Id uniqueness

`next_progress_id()` MUST return monotonically increasing, unique ids within
a single sink instance. Implementations use `AtomicU32` with `Relaxed`
ordering for lock-free allocation.

## 8.5 Handle Facade

### 8.5.1 UiHandle

`UiHandle` wraps `Arc<dyn UiSink>` and provides the ergonomic API that all
pipeline code uses. It is cheap to clone (just an `Arc` bump).

```rust
pub struct UiHandle {
    sink: Arc<dyn UiSink>,
}
```

Methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `bar(total, label)` | `ProgressHandle` | Create a determinate progress bar. |
| `spinner(label)` | `ProgressHandle` | Create an indeterminate spinner. |
| `log(message)` | — | Emit a `Log` event. |
| `emit(text)` | — | Emit raw text (no newline). |
| `emitln(text)` | — | Emit raw text with newline. |
| `resource_status(line)` | — | Update the resource status line. |
| `suspend_begin()` | — | Suppress redraws until `suspend_end()`. |
| `suspend_end()` | — | End batch, trigger atomic redraw. |
| `clear()` | — | Clear the entire progress region. |
| `inc_by_id(id, delta)` | — | Increment a progress indicator by `ProgressId`. |
| `set_message_by_id(id, msg)` | — | Update a progress message by `ProgressId`. |
| `finish_by_id(id)` | — | Finish a progress indicator by `ProgressId`. |

The `*_by_id` methods exist for cross-thread sharing patterns where multiple
tasks need to update the same progress bar. The `ProgressId` is `Copy`, so it
can be sent to any thread without cloning an `Arc` handle.

### 8.5.2 ProgressHandle

`ProgressHandle` is an owned RAII guard for a single progress indicator.
It sends a `ProgressFinish` event on drop if `finish()` was not already
called, preventing leaked indicators.

```rust
pub struct ProgressHandle {
    id: ProgressId,
    sink: Arc<dyn UiSink>,
    finished: AtomicBool,
}
```

Methods:

| Method | Description |
|--------|-------------|
| `id()` | Return the underlying `ProgressId` for cross-thread sharing. |
| `set_position(u64)` | Set absolute position. |
| `inc(u64)` | Increment position by delta. |
| `set_message(impl Into<String>)` | Update the trailing message. |
| `finish()` | Explicitly finish the indicator (idempotent via `AtomicBool`). |

`finish()` takes `&self` (not `&mut self`) using an `AtomicBool` swap, so
that callers can finish an immutably-bound handle without requiring
mutability.

## 8.6 Concrete Sinks

### 8.6.1 PlainSink

Used when stdout is a pipe or file, in standalone CLI commands (`veks import`,
`veks convert`), and as the default for pipeline execution.

Behavior:

| Event category | Action |
|----------------|--------|
| Progress (`Create`, `Update`, `Inc`, `Message`, `Finish`) | Silently ignored — no progress bars in piped output. |
| `Log`, `Emit`, `EmitLn` | Written to stdout with appropriate newline handling. |
| `ResourceStatus`, `SuspendBegin`, `SuspendEnd`, `Clear` | Silently ignored. |

### 8.6.2 TestSink

Used in unit tests. Records all events into a `Mutex<Vec<UiEvent>>` for
post-hoc assertion.

Helper methods:

| Method | Description |
|--------|-------------|
| `events()` | Snapshot of all recorded events. |
| `len()` / `is_empty()` | Event count. |
| `clear()` | Reset the event log. |
| `log_messages()` | Extract only `Log` event messages. |

### 8.6.3 RatatuiSink

Rich terminal rendering using ratatui's `Viewport::Inline` mode.

Source: `ui/ratatui_sink.rs`

Architecture:
- Events are sent over `mpsc::channel` to a dedicated render thread that
  owns the ratatui `Terminal<CrosstermBackend<Stdout>>`.
- The `RatatuiSink` itself holds only a `Sender` and `AtomicU32`, making
  it `Send + Sync` without locking the terminal.
- Rate-limited redraws (50ms minimum interval) prevent terminal flooding.

Behavior:

| Event category | Action |
|----------------|--------|
| `ProgressCreate` | Track bar state; insert into ordered list; mark dirty. |
| `ProgressUpdate`, `ProgressInc`, `ProgressMessage` | Update tracked state; mark dirty. |
| `ProgressFinish` | Remove from tracked state; mark dirty. |
| `Log` | `insert_before()` above the inline progress region. |
| `Emit`, `EmitLn` | Combined into lines, inserted above the progress region. |
| `ResourceStatus` | Rendered as a dim status line below progress bars. |
| `SuspendBegin` | Suppress redraws until `SuspendEnd`. |
| `SuspendEnd` | End batch, trigger single atomic redraw. |
| `Clear` | Remove all tracked bars and status. |

Selection: `auto_ui_handle()` in `ui/mod.rs` selects `RatatuiSink` when
stdout is a TTY, falling back to `PlainSink` on initialization failure.

## 8.7 Requirements

### REQ-UI-04: Fixed progress region

All within-step progress indicators MUST render in a fixed region at the
bottom of the terminal (when a TTY-capable sink is active). Updates
overwrite in place — they MUST NOT emit newlines that scroll the terminal.
Log messages scroll above the fixed region.

### REQ-UI-05: Auto-finish on drop

`ProgressHandle` MUST send `ProgressFinish` on drop if `finish()` was not
already called. This prevents leaked indicators when a function returns
early or panics.

### REQ-UI-06: Thread safety

`UiHandle` and `ProgressHandle` use `&self` methods exclusively (no `&mut
self`). The underlying sink is behind `Arc<dyn UiSink>` with `Send + Sync`
bounds. Progress indicators can be updated from any thread using either:

- The `ProgressHandle` itself (when the handle's lifetime is scoped to one
  thread).
- The `*_by_id` methods on `UiHandle` (when multiple threads need to update
  the same indicator — the `ProgressId` is `Copy`).

### REQ-UI-07: No scrolling counters

Repeated counter, percentage, or ETA updates MUST use progress indicators
(bar or spinner), never sequential log lines. One-shot informational
messages (e.g., "opening base vectors: foo.fvec") are allowed as `Log`
events since they appear only once.

### REQ-UI-08: StreamContext integration

`StreamContext` provides a `ui: UiHandle` field. The runner creates the
handle with an appropriate sink before step execution begins and passes it
to every command via the context. Between steps, the runner sends `Clear`
and emits step-transition messages as `Log` events.

For standalone CLI commands (`veks import`, `veks convert`, `veks bulkdl`)
that bypass the pipeline runner, the command creates its own `UiHandle`
wrapping a `PlainSink`.

### REQ-UI-09: Batch redraw suppression

When creating multiple progress indicators in quick succession (e.g.,
pipeline setup), the caller SHOULD bracket the creates with
`suspend_begin()` / `suspend_end()` to suppress per-event redraws and
perform a single atomic redraw at the end. This prevents visual flicker
and ghost-line artifacts.

### REQ-UI-10: Multi-phase progress

Commands with multiple sequential phases (e.g., `compute knn` has planning,
prefetching, computing, merging) create a new `ProgressHandle` for each
phase. The prior handle is dropped (auto-finishing it), and a log line
summarizing the completed phase scrolls above. The new bar appears on the
fixed progress line.

## 8.8 Usage Patterns

### Pipeline command

```rust
fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
    let pb = ctx.ui.bar(total, "importing records");
    for record in source {
        sink.write_record(ordinal, &record);
        pb.inc(1);
    }
    pb.finish();
    // ...
}
```

### Cross-thread progress sharing (async downloads)

```rust
let pb = ui.bar(total, &dataset.name);
let pb_id = pb.id();  // ProgressId is Copy

for (url, filename) in pending {
    let ui = ui.clone();
    tokio::spawn(async move {
        // ... download logic ...
        ui.inc_by_id(pb_id, 1);  // safe from any thread
    });
}
```

### Standalone CLI command

```rust
let ui = UiHandle::new(Arc::new(PlainSink::new()));
let pb = if let Some(total) = record_count {
    ui.bar(total, "converting records")
} else {
    ui.spinner("converting records")
};
```

### Test assertions

```rust
let sink = Arc::new(TestSink::new());
let ui = UiHandle::new(sink.clone());

let pb = ui.bar(100, "testing");
pb.set_position(50);
pb.finish();

let events = sink.events();
assert!(matches!(events[0], UiEvent::ProgressCreate { .. }));
assert!(matches!(events[1], UiEvent::ProgressUpdate { position: 50, .. }));
assert!(matches!(events[2], UiEvent::ProgressFinish { .. }));
// drop does not double-finish
drop(pb);
assert_eq!(sink.events().len(), 3);
```

## 8.9 Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-UI-01 | Done | No pipeline code imports a rendering library. indicatif fully removed. |
| REQ-UI-02 | Done | `PlainSink`, `TestSink`, and `RatatuiSink` implemented. |
| REQ-UI-03 | Done | `AtomicU32` id allocation in both sinks. |
| REQ-UI-04 | Done | `RatatuiSink` uses `Viewport::Inline` with fixed progress region. `auto_ui_handle()` selects it when stdout is a TTY. |
| REQ-UI-05 | Done | `ProgressHandle::Drop` sends `ProgressFinish` via `AtomicBool` guard. |
| REQ-UI-06 | Done | All methods take `&self`; `Arc<dyn UiSink>` shared across threads. |
| REQ-UI-07 | Done | All scrolling counter patterns replaced with progress bars. |
| REQ-UI-08 | Done | `StreamContext.ui` field; standalone commands use inline `UiHandle`. |
| REQ-UI-09 | Done | `SuspendBegin`/`SuspendEnd` events defined; rendering TBD in `RatatuiSink`. |
| REQ-UI-10 | Done | Multi-phase commands drop prior handle, log summary, create new bar. |
