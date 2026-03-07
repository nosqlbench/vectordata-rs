# Observability and Documentation

`vectordata-rs` is designed for processing multi-hundred-GB datasets, where visibility into the status of long-running operations and ease of access to command documentation are critical.

## Integrated Progress Monitoring

Every pipeline command that processes large datasets uses real-time progress indicators. This provides the user with clear feedback on:
- **Throughput**: Current data processing rate (e.g., vectors per second or MB/s).
- **Completion Percentage**: Progress towards the final artifact.
- **Estimated Time Remaining (ETA)**: A dynamic estimate based on current throughput and remaining work.

These progress displays are implemented using the `indicatif` crate and are carefully integrated with the pipeline engine to ensure they are visible without cluttering the logs.

## Self-Documenting Commands

To prevent documentation from falling out of sync with the code, every pipeline command carries its own documentation directly within the Rust binary.

### CommandDoc Struct
Each command implements the `command_doc()` trait method, which returns a `CommandDoc` containing:
- **Summary**: A one-line plain-text description used in shell completions.
- **Body**: Full Markdown documentation, including descriptions, options tables, and examples.

### Automatic Options and Resources
The `Options` and `Resources` tables are generated automatically from the same trait methods (`describe_options()` and `describe_resources()`) used to parse arguments and govern system resources. This ensures that the documentation is always accurate and complete.

### Accessing Documentation
Users can access this documentation directly through the `veks` CLI:
- `veks help <command>`: Displays the long-form Markdown documentation.
- `veks pipeline <group> <cmd> --help`: Displays the standard CLI help derived from the command documentation.

## Test-Enforced Completeness

To maintain high documentation standards, a workspace test suite verifies that:
- Every command provides a non-empty summary and body.
- The body mentions every option declared in the code.
- If a command declares resource usage, it must include a corresponding "Resources" section in its documentation.

By making documentation a first-class citizen in the code, `vectordata-rs` ensures that users always have access to accurate, up-to-date information directly from the tools they are using.
