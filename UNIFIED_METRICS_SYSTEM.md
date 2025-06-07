# Unified Event-Based Metrics System

## Executive Summary

We've unified the metrics tracking and event tracing systems into a single coherent approach where **event tracing IS the metrics system**. This eliminates the need for parallel systems and provides memory-efficient performance tracking with smart retention policies.

## Key Innovation: MetricsEventTracer

The `MetricsEventTracer` class processes events to calculate metrics in real-time, then retains or discards events based on configurable retention policies:

```python
# In src/core/containers/metrics.py
class MetricsEventTracer:
    """
    Event tracer optimized for metrics calculation.
    
    Processes events to calculate metrics then discards them based on
    retention policy. This provides a unified system where event tracing
    IS the metrics system.
    """
```

### Retention Policies

1. **`trade_complete`** (Default)
   - Only retains events for active trades
   - Events are cleaned up when trades complete
   - Most memory efficient
   - Perfect for production and optimization phases

2. **`sliding_window`**
   - Keeps last N events in a circular buffer
   - Configurable via `max_events` parameter
   - Good for debugging and final validation
   - Provides bounded memory usage

3. **`minimal`**
   - Aggressive cleanup after each trade
   - Only keeps events for positions with open trades
   - Minimal memory footprint
   - Ideal for large parameter sweeps

## Memory-Efficient Metrics Calculation

The system uses **Welford's algorithm** for numerically stable calculation of statistics without storing all data points:

```python
class StreamingMetrics:
    """Calculate performance metrics without storing full history."""
    
    def update_portfolio_value(self, value: float, timestamp: Optional[datetime] = None):
        # Updates running statistics efficiently
        # O(1) space complexity, O(1) time complexity
```

### Key Metrics Tracked
- Sharpe ratio (calculated incrementally)
- Total return and drawdowns
- Win rate and profit factor
- Trade statistics
- Portfolio value snapshots (optional)

## Phase-Specific Configuration

Different workflow phases can optimize their memory usage:

### Grid Search Phase (Minimal Memory)
```yaml
results_override:
  retention_policy: minimal    # Aggressive cleanup
  max_events: 100             # Very small buffer
  collection:
    streaming_metrics: true
    store_trades: false       # No trade storage
    store_equity_curve: false # No equity curve
```

### Final Validation Phase (Full Analysis)
```yaml
results_override:
  retention_policy: sliding_window  # Keep more events
  max_events: 5000                 # Larger buffer
  collection:
    streaming_metrics: true
    store_trades: true
    store_equity_curve: true      # Full equity curve
    snapshot_interval: 1          # Every bar
```

## Container Integration

The unified system is automatically set up for portfolio containers:

```python
def _setup_metrics(self):
    """Setup event-based metrics tracking."""
    # Creates MetricsEventTracer that processes events
    self.streaming_metrics = MetricsEventTracer(tracer_config)
    
    # Subscribe to all relevant events
    for event_type in event_types:
        self.event_bus.subscribe(event_type, self.streaming_metrics.trace_event)
```

## Results Collection

The system provides different levels of results based on configuration:

```python
def get_results(self) -> Dict[str, Any]:
    """Get complete results including metrics, trades, and equity curve."""
    return {
        'metrics': self.get_metrics(),        # Always available
        'trades': self.completed_trades,      # If store_trades=true
        'equity_curve': self.equity_curve     # If store_equity_curve=true
    }
```

## Memory Guarantees

1. **Bounded Memory Usage**: All retention policies guarantee bounded memory
2. **Automatic Cleanup**: Events are cleaned up based on retention policy
3. **Configurable Limits**: Max events and memory thresholds are configurable
4. **Smart Storage**: Automatic switching between memory and disk storage

## Migration Benefits

Moving from separate metrics and tracing systems to unified approach:

### Before (Two Systems)
- Event tracing system storing all events
- Separate metrics calculation storing portfolio history
- Duplicate event processing
- Complex configuration

### After (Unified System)
- Single event processing pipeline
- Smart retention based on need
- Unified configuration
- Memory-efficient by design

## Example: Complete Configuration

```yaml
# Global defaults (conservative)
results:
  retention_policy: trade_complete
  max_events: 1000
  collection:
    streaming_metrics: true
    store_trades: true
    store_equity_curve: false
    snapshot_interval: 100

# Phase-specific overrides
phases:
  - name: optimization
    results_override:
      retention_policy: minimal
      max_events: 100
      
  - name: validation
    results_override:
      retention_policy: sliding_window
      max_events: 5000
      collection:
        store_equity_curve: true
        snapshot_interval: 1
```

## Implementation Status

✅ **Completed**:
- StreamingMetrics class with Welford's algorithm
- MetricsEventTracer with retention policies
- Container integration in container.py
- YAML schema updates
- Example configuration

✅ **Key Files Updated**:
- `/src/core/containers/metrics.py` - New unified implementation
- `/src/core/containers/container.py` - Updated to use MetricsEventTracer
- `/src/core/config/schemas.py` - Added retention_policy and max_events
- `/config/example_with_results.yaml` - Shows phase-specific configuration

## Summary

The unified event-based metrics system provides:

1. **Memory Efficiency**: Smart retention policies minimize memory usage
2. **Performance**: Streaming algorithms calculate metrics without storing history
3. **Flexibility**: Different phases can optimize for their specific needs
4. **Simplicity**: One system instead of two parallel systems
5. **Observability**: Events provide both metrics and debugging capability

This approach aligns perfectly with the event-driven architecture while solving the memory constraints for large-scale optimization workflows.