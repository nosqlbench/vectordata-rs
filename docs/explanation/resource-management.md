# Resource Management

Pipeline commands that process multi-hundred-GB datasets can easily saturate system resources (memory, I/O, threads), leading to system lockups or Out-Of-Memory (OOM) kills. **Resource Governance** is the mechanism that prevents this by dynamically adjusting resource allocations at runtime based on real-time system pressure.

## The ResourceGovernor

A `ResourceGovernor` component monitors the system state and acts as the central authority for resource allocation. Pipeline commands do not manage their own budgets; instead, they interact with the governor through a read-only view.

### Key Responsibilities
- **Allocation Snapshots**: Provide commands with the latest view of allowed resource usage (e.g., memory in MB, thread counts).
- **Resource Requests**: Accept and evaluate requests for additional resources from commands.
- **System Monitoring**: Track global system pressure (CPU, memory, disk I/O) to inform allocation decisions.
- **Throttling**: Signal to commands when they should slow down or yield to other processes to maintain system stability.

## Resource Budgeting

Users specify allowable resource ranges via the `--resources` flag:

```bash
veks pipeline run --resources 'mem:25%-50%,threads:4-8' --steps '...'
```

The governor stays within these bounds while reacting to actual system pressure. Each range declares the minimum and maximum that the governor may allocate.

## Operating Bands

The governor classifies current memory pressure into several bands:

| Band | Behavior |
|------|----------|
| **UNDERUSED** | Below 70% pressure; the governor may increase allocations toward the ceiling. |
| **NOMINAL** | 70–85% pressure; allocations remain stable. |
| **CAUTION** | 85–90% pressure; the governor begins reducing allocations. |
| **THROTTLE** | 90–95% pressure; commands are told to slow down via `should_throttle()`. |
| **EMERGENCY** | Above 95% pressure; the governor aggressively reclaims resources to prevent OOM. |

## Strategies

Three built-in strategies control how the governor behaves within the budget:
- **Maximize (Default)**: Use as much of the budget as system pressure allows.
- **Conservative**: Start near the minimum and grow slowly, prioritizing stability over throughput.
- **Fixed**: Lock allocations at the minimum with no dynamic adjustment.

## Per-Command Resource Declarations

Every pipeline command that processes dataset-scale files must be **resource-aware**. Commands declare their resource requirements (memory buffers, thread pools, temporary files) through the `describe_resources()` method. This allows the governor to make informed decisions about how to distribute the available budget across multiple concurrent operations.
