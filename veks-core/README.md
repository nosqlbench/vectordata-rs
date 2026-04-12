# veks-core

Shared utilities for the vectordata-rs workspace.

- `formats` — `VecFormat` enum, extension detection, element sizes
- `filters` — file exclusion rules (underscore prefix, hidden files)
- `term` — terminal colors and formatting (bold, red, green, dim)
- `ui` — `UiHandle`, `ProgressHandle`, UI event types, sink trait
- `paths` — relative path display utilities

This crate has no heavy dependencies. It provides the common types
used across `veks`, `veks-pipeline`, and `vectordata`.
