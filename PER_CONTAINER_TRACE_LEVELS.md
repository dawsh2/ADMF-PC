# Per-Container Trace Levels in ADMF-PC

## Overview

The trace level system is designed to be **per-container**, not global. Each container type gets appropriate settings based on its role and the selected trace level. This enables fine-grained control over memory usage and performance.

## The MINIMAL Trace Level

The `minimal` trace level implements exactly what was requested:
- Portfolio containers track **only open trades**
- When trades close, metrics are updated (persist in memory)
- Event traces are removed from memory after trade completion
- Absolute minimum memory usage while maintaining full metrics

### How It Works

1. **MetricsEventTracer** with `retention_policy='trade_complete'`:
   - Stores events in `active_trades` dict by trade ID (no max limit)
   - When a position closes, calls `_cleanup_completed_trade_events()`
   - Removes all events for that trade from memory

2. **StreamingMetrics** (always in memory):
   - Calculates metrics incrementally
   - No historical data needed
   - Maintains persistent metrics:
     - Sharpe ratio, total return
     - Win rate (via winning_trades/losing_trades counters)
     - Total P&L, gross profit/loss
     - Current value
   - **Tradeoff**: Max drawdown = 0 (requires equity curve)

3. **Memory Usage Pattern**:
   ```
   Open Trade:  [Metrics] + [Trade Events]
   Close Trade: [Updated Metrics] (events deleted)
   ```

## Per-Container Settings by Trace Level

### MINIMAL (Optimization/Production)
```yaml
portfolio_*:
  enabled: true
  max_events: 0  # No limit - retention policy handles cleanup
  retention_policy: trade_complete
  results:
    streaming_metrics: true
    store_trades: false
    store_equity_curve: false

# ALL other containers disabled
data_*, feature_*, strategy_*, risk_*, execution_*:
  enabled: false  # No tracing needed
```

### NORMAL (Development)
```yaml
portfolio_*:
  enabled: true
  max_events: 10000
  retention_policy: trade_complete
  results:
    streaming_metrics: true
    store_trades: true  # Keep trade history
    store_equity_curve: true  # Track equity
    snapshot_interval: 100

data_*:
  enabled: true
  max_events: 1000

feature_*:
  enabled: true
  max_events: 5000

strategy_*:
  enabled: true
  max_events: 1000

execution_*:
  enabled: true
  max_events: 5000
```

### DEBUG (Full Analysis)
```yaml
portfolio_*:
  enabled: true
  max_events: 50000
  retention_policy: sliding_window  # Keep everything
  results:
    streaming_metrics: true
    store_trades: true
    store_equity_curve: true
    snapshot_interval: 10

*:  # All containers
  enabled: true
  max_events: 50000
  retention_policy: sliding_window
```

### NONE (Zero Overhead)
```yaml
# All tracing disabled
# Containers still maintain their own state/metrics
```

## Container-Specific Behaviors

### Portfolio Containers
- Always track metrics (even with trace_level: none)
- Use MetricsEventTracer for memory-efficient metric calculation
- Trade retention based on trace level

### Data/Feature Containers
- Stateful but don't need event history
- Disabled in MINIMAL to save memory
- Enabled in NORMAL/DEBUG for debugging

### Strategy/Risk Containers
- Stateless services - minimal benefit from tracing
- Disabled in MINIMAL
- Light tracing in NORMAL for debugging

### Execution Container
- Needs some tracking for order lifecycle
- Always gets at least minimal tracing
- More events in NORMAL/DEBUG

## Memory Efficiency

Example with 100 portfolio containers running 1-year backtest:

| Trace Level | Memory Usage | What's Stored |
|-------------|--------------|---------------|
| NONE | ~100KB | Basic container state only |
| MINIMAL | ~200KB-1MB | Persistent metrics + open trades only (dynamic) |
| NORMAL | ~100MB | + Trade history + equity curve |
| DEBUG | ~1GB | Full event history |

With MINIMAL, memory usage scales with number of open positions, not total trades:
- 10 open positions × 10 events each = ~100 events in memory
- 1000 closed trades = 0 events in memory (metrics updated, events deleted)

## Configuration Examples

### Simple Usage
```yaml
workflow: parameter_optimization
trace_level: minimal  # Automatic per-container settings
```

### Override Specific Container
```yaml
trace_level: minimal

# Override just one container
execution:
  trace_settings:
    container_settings:
      portfolio_main:
        max_events: 5000  # More for debugging
```

### Custom Pattern
```yaml
execution:
  trace_settings:
    container_settings:
      # Different settings for different portfolio groups
      portfolio_conservative_*:
        max_events: 500
      portfolio_aggressive_*:
        max_events: 2000
```

## Best Practices

1. **Use MINIMAL for optimization**: Thousands of portfolios with minimal memory
2. **Use NORMAL for development**: Good balance of visibility and performance
3. **Use DEBUG sparingly**: Only when you need full event history
4. **Override specific containers**: When you need to debug just one component

## Implementation Details

The system works through several layers:

1. **Trace Level Presets** (`trace_levels.py`):
   - Define per-container settings for each level
   - Include both trace config and results config

2. **Container Setup** (`container.py`):
   - `_should_enable_tracing()` checks container-specific settings
   - `_setup_metrics()` configures MetricsEventTracer

3. **MetricsEventTracer** (`metrics.py`):
   - Implements retention policies
   - Manages trade lifecycle tracking
   - Calculates streaming metrics

This design ensures each container type gets exactly the tracing it needs, no more, no less.