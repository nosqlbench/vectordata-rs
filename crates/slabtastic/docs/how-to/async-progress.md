# How to Use Background Tasks with Progress Polling

Slabtastic provides `SlabTask<T>` for long-running operations that execute
on a background thread. The caller gets a handle to poll progress and
eventually collect the result.

## The SlabTask / SlabProgress model

```text
Caller thread                 Background thread
─────────────                 ─────────────────
task = start_async(...)  ──>  spawns thread
                              work begins
task.progress().completed()   tracker.inc()
task.progress().fraction()    tracker.inc()
task.is_done() == false       ...
                              tracker.mark_done()
task.is_done() == true
result = task.wait()     <──  thread joins, returns Result<T>
```

- `SlabProgress` provides `total()`, `completed()`, `is_done()`, and
  `fraction()` — all thread-safe atomic reads.
- `SlabTask::wait()` consumes the task and blocks until the thread finishes.

## Async read example

```rust
use slabtastic::SlabReader;
use std::path::PathBuf;

let sink: Vec<u8> = Vec::new();
let task = SlabReader::read_to_sink_async(
    PathBuf::from("data.slab"),
    sink,
    |count| eprintln!("callback: {count} records read"),
);

while !task.is_done() {
    let p = task.progress();
    eprintln!("progress: {}/{}", p.completed(), p.total());
    std::thread::sleep(std::time::Duration::from_millis(100));
}
let count = task.wait().expect("read succeeded");
```

## Async write example

```rust
use slabtastic::{SlabWriter, WriterConfig};
use std::path::PathBuf;

let records = (0..50_000).map(|i| format!("r{i}").into_bytes());
let task = SlabWriter::write_from_iter_async(
    PathBuf::from("output.slab"),
    WriterConfig::default(),
    records,
    |count| eprintln!("callback: wrote {count} records"),
);

while !task.is_done() {
    eprintln!("written: {}", task.progress().completed());
    std::thread::sleep(std::time::Duration::from_millis(50));
}
let total = task.wait().expect("write succeeded");
```

## Notes

- `progress().total()` is set lazily — for reads, it is known after the
  pages page is scanned; for writes from an iterator, it is set once the
  iterator is exhausted.
- The `on_complete` callback runs on the background thread, not the caller.
- If the background thread panics, `wait()` will propagate the panic.
