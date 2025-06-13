# Strategy-Level Signal Tracing Implementation

## Overview

We've implemented strategy-level signal tracing as requested. This new architecture traces signals at the source (strategy containers) rather than the destination (portfolio containers), making signal generation and replay much simpler.

## Key Changes

### 1. New StrategySignalTracer Class
Created `/src/core/events/observers/strategy_signal_tracer.py`:
- Traces signals at the strategy level
- Each strategy gets its own sparse signal file
- Stores only signal changes with bar indices
- Includes strategy parameters in metadata

### 2. Container Updates
Updated `/src/core/containers/container.py`:
- Added `_setup_strategy_signal_tracing()` method
- Strategy containers now create tracers after initialization
- Each strategy within a container gets its own tracer
- Tracers are flushed during container cleanup

### 3. Directory Organization
Signals are now stored in a cleaner structure:
```
workspaces/
├── {workflow_id}/
│   └── {run_id}/
│       └── strategies/
│           ├── signals_{strategy_id1}_{timestamp}.json
│           └── signals_{strategy_id2}_{timestamp}.json
```

## Benefits

1. **Simpler Signal Replay**: Portfolios can subscribe to specific strategy signals by ID
2. **Better Organization**: Each strategy's signals are in separate files
3. **Parameter Tracking**: Strategy parameters are stored with signals
4. **Efficient Storage**: Uses temporal sparse storage (only changes)
5. **Performance Metrics**: Includes signal statistics and performance tracking

## Usage

### Configuration
Enable strategy signal tracing in your YAML config:
```yaml
execution:
  enable_event_tracing: true
  trace_settings:
    use_sparse_storage: true
    container_settings:
      'strategy*':
        enabled: true
```

### Signal File Format
Each strategy signal file contains:
```json
{
  "metadata": {
    "run_id": "strategy_id",
    "total_bars": 100,
    "total_changes": 10,
    "compression_ratio": 0.1,
    "signal_statistics": { ... },
    "strategy_parameters": { ... }
  },
  "changes": [
    {
      "idx": 0,      // bar index
      "ts": "...",   // timestamp
      "sym": "SPY",  // symbol
      "val": 1,      // signal value (-1, 0, 1)
      "strat": "...",// strategy ID
      "px": 100.0    // price at signal
    }
  ]
}
```

## Example Test

See `test_direct_strategy_trace.py` for a working example that demonstrates:
- Creating a strategy signal tracer
- Publishing signals
- Viewing compression statistics
- Saving signals to disk

## Next Steps

1. **Signal Replay**: Implement signal replay that reads strategy signal files
2. **Grid Search Integration**: Use with parameter grid search for optimization
3. **Signal Aggregation**: Combine signals from multiple strategies
4. **Performance Analysis**: Analyze strategy performance from signal files