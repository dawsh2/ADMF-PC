# Strategy-Level Signal Tracing (Refactored)

## Overview

As requested, we've refactored the strategy-level signal tracing to reuse existing infrastructure instead of creating a new class. Strategy containers now use the same `SparsePortfolioTracer` that portfolio containers use, just configured differently.

## Key Changes

### 1. Reused SparsePortfolioTracer
- No new classes created
- Strategy containers use `SparsePortfolioTracer` with strategy-specific configuration
- The tracer is injected into strategy containers during initialization

### 2. Container Updates
Updated `/src/core/containers/container.py`:
```python
def _setup_strategy_signal_tracing(self):
    """Setup signal tracing for strategy containers using SparsePortfolioTracer."""
    # ... 
    self._strategy_tracer = SparsePortfolioTracer(
        container_id=f"strategies_{self.container_id}",
        workflow_id=workflow_id,
        managed_strategies=strategy_ids,  # Track signals from these strategies
        storage_config=storage_config,
        portfolio_container=self  # Pass self as the container
    )
```

### 3. Benefits of This Approach
- **No new code**: Uses existing, tested infrastructure
- **Consistent behavior**: Same tracing logic for both portfolios and strategies
- **Simpler maintenance**: One tracer implementation to maintain
- **Flexible**: Can trace signals at either strategy or portfolio level

## Configuration

Enable strategy signal tracing in your topology configuration:
```yaml
execution:
  enable_event_tracing: true
  trace_settings:
    use_sparse_storage: true
    container_settings:
      'strategy*':  # Enable for strategy containers
        enabled: true
```

## Directory Structure
```
workspaces/
├── {workflow_id}/
│   └── {run_id}/
│       ├── strategies_{container_id}/  # Strategy container traces
│       │   └── signals_*.json
│       └── portfolio_{container_id}/   # Portfolio container traces
│           └── signals_*.json
```

## How It Works

1. When a strategy container is created with tracing enabled, it sets up a `SparsePortfolioTracer`
2. The tracer monitors SIGNAL events from the strategies within that container
3. Only signal changes are stored (temporal sparse storage)
4. Strategy parameters are captured in the metadata
5. Files are flushed to disk during container cleanup

## Next Steps

1. **Signal Replay**: Read strategy signal files and replay to portfolios
2. **Multi-Strategy Containers**: Each container can trace multiple strategies
3. **Parameter Grid Search**: Use with optimization workflows
4. **Performance Analysis**: Analyze strategy performance from signal files