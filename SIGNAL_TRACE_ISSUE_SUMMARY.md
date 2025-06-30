# Signal Trace Generation Issue Analysis

## Problem
Signal traces are not being generated when running a signal generation command.

## Root Cause Identified

The issue is in `/src/core/coordinator/topology.py` at lines 158-160:

```python
# For signal generation, always set up MultiStrategyTracer regardless of container tracing
# MultiStrategyTracer handles signal storage independently from event tracing
if mode == 'signal_generation' and use_sparse:
    from ..events.tracer_setup import setup_multi_strategy_tracer
    setup_multi_strategy_tracer(topology, context, tracing_config)
```

The tracer is only set up when **both** conditions are met:
1. `mode == 'signal_generation'`
2. `use_sparse == True` (from `execution.trace_settings.use_sparse_storage`)

## Why Traces Might Not Be Generated

1. **Missing `use_sparse_storage` flag**: If the config doesn't have `execution.trace_settings.use_sparse_storage: true`, the tracer won't be set up.

2. **Wrong mode**: If the topology mode isn't exactly `'signal_generation'`, the tracer won't be set up.

3. **Config structure**: The expected config structure is:
   ```yaml
   execution:
     enable_event_tracing: true
     trace_settings:
       use_sparse_storage: true  # This MUST be present
   ```

## Implementation Details

### Tracer Classes
- **MultiStrategyTracer** (`/src/core/events/observers/multi_strategy_tracer.py`): Default tracer
- **StreamingMultiStrategyTracer** (`/src/core/events/observers/streaming_multi_strategy_tracer.py`): Used for large runs (>2000 bars) or when explicitly requested

### Tracer Selection Logic
From `tracer_setup.py`:
```python
max_bars = context['config'].get('max_bars', 0)
is_signal_generation = context.get('mode') == 'signal_generation'
use_streaming = max_bars > 2000 or context['config'].get('streaming_tracer', False) or is_signal_generation
```

### Recent Changes
The git log shows a recent commit about "0-based indexing":
- Commit: `507ad35a` - "Fix MultiStrategyTracer to use 0-based indexing matching source files"

This change ensures bar indices match the source data files (0-based) instead of 1-based counting.

## Solution

To enable signal tracing, ensure your config includes:

```yaml
execution:
  enable_event_tracing: true
  trace_settings:
    use_sparse_storage: true  # REQUIRED for signal generation
    # Optional settings:
    write_interval: 0         # For streaming tracer
    write_on_changes: 0       # For streaming tracer
```

## Additional Notes

1. The tracer attaches to the root event bus and listens for:
   - `BAR` events (to track bar count)
   - `SIGNAL` events (from strategies)
   - `CLASSIFICATION` events (from classifiers)

2. Signal storage is hierarchical:
   - `traces/[strategy_type]/[strategy_id].parquet`
   - Sparse format (only stores changes)

3. A `strategy_index.parquet` file is created with metadata for all strategies

4. The tracer creates a `metadata.json` file in the workspace root with run information