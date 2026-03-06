<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Resource Governance

Pipeline commands that process multi-hundred-GB datasets can saturate system
resources — memory, I/O, CPU — causing lockups or OOM kills. Resource
governance is the mechanism that prevents this by dynamically adjusting
resource allocations at runtime.

## The ResourceGovernor

A `ResourceGovernor` component monitors system state and acts as the sole
entity that adjusts resource allocations. No command modifies its own
allocation directly. Instead, commands interact with the governor through
read-only views:

| Method             | Purpose                                              |
|--------------------|------------------------------------------------------|
| `current()`        | Read the current resource allocation snapshot         |
| `request()`        | Submit a request for additional resources             |
| `checkpoint()`     | Report current usage so the governor can reassess     |
| `should_throttle()`| Ask whether the command should slow down or yield     |

## ResourceBudget

Users specify allowable resource ranges via the `--resources` flag:

```
veks pipeline run --resources 'mem:25%-50%,threads:4-8' --steps '...'
```

Each range declares the minimum and maximum that the governor may allocate.
The governor stays within these bounds while reacting to actual system
pressure.

## Operating bands

The governor classifies current memory pressure into five bands relative to
the user-specified memory ceiling:

| Band       | Range   | Behavior                                        |
|------------|---------|------------------------------------------------|
| UNDERUSED  | < 70%   | Governor may increase allocations               |
| NOMINAL    | 70–85%  | Allocations remain stable                        |
| CAUTION    | 85–90%  | Governor begins reducing allocations             |
| THROTTLE   | 90–95%  | Commands are told to slow down via `should_throttle()` |
| EMERGENCY  | > 95%   | Governor aggressively reclaims resources          |

## Strategies

Three built-in strategies control how the governor behaves within the budget:

- **maximize** (default) — Use as much of the budget as system pressure
  allows. Grows allocations toward the ceiling when headroom exists.
- **conservative** — Start near the minimum and grow slowly. Prefers
  stability over throughput.
- **fixed** — Lock allocations at the minimum. No dynamic adjustment.

## Per-command resource declarations

Each pipeline command declares its resource needs via the
`describe_resources()` method on the `CommandOp` trait. This tells the
governor what kinds of resources the command uses (memory buffers, thread
pools, temporary files) and their expected ranges.

All 37 commands that process dataset-scale files must comply with resource
governance. A command that ignores `should_throttle()` or allocates outside
its declared budget is considered non-compliant.
