# Event Tracing Implementation Summary

## Overview
Event tracing has been successfully implemented for single-phase workflows in the ADMF-PC system. This allows complete event flow tracking, performance monitoring, and debugging capabilities.

## Changes Made

### 1. WorkflowManager Enhancement (`src/core/coordinator/topology.py`)
- Added tracing support in `_create_universal_topology` method
- Check for `tracing.enabled` in workflow configuration
- Create `TracedEventBus` instead of regular `EventBus` when tracing is enabled
- Initialize `EventTracer` with correlation ID and max events configuration
- Replace container event buses with `TracedEventBus` instances

### 2. TracedEventBus Fix (`src/core/events/tracing/traced_event_bus.py`)
- Fixed constructor parameter from `name` to `container_id` to match parent class
- Fixed publish method to check `self.tracer is not None` instead of `if self.tracer`
- Fixed logging to use `self.container_id` instead of `self.name`

### 3. Container Event Bus Replacement
- Replace event buses on all containers (data, feature, portfolio, execution) with TracedEventBus
- Ensure tracers are attached before event wiring
- Copy existing subscriptions when replacing event buses
- Wire container connections after event bus replacement

### 4. Trace Summary Integration
- Added `get_trace_summary()` method to WorkflowManager
- Include trace summary in workflow results (both final_results and metadata)
- Log trace summary statistics after workflow completion

## Configuration

To enable event tracing in a workflow, add the following to your YAML configuration:

```yaml
tracing:
  enabled: true
  max_events: 10000  # Optional, defaults to 10000
```

## Usage Example

```yaml
# config/test_tracing_backtest.yaml
type: backtest

tracing:
  enabled: true
  max_events: 10000

mode: backtest
symbols: ['SPY']
start_date: '2023-01-01'
end_date: '2023-01-10'

backtest:
  data:
    source: csv
    file_path: ./data/SPY.csv
  # ... rest of config
```

## Trace Summary Output

When tracing is enabled, the workflow results will include a trace summary with:
- Total events traced
- Event counts by type
- Container counts (which containers emitted events)
- Latency statistics
- Event sequence range

Example trace summary:
```json
{
  "correlation_id": "workflow_20250605_133238",
  "total_events": 10000,
  "event_counts": {
    "1": 10000  // EventType.BAR
  },
  "container_counts": {
    "SPY_1d_data": 10000
  },
  "latency_stats": {},
  "sequence_range": {
    "first": 1,
    "last": 10000
  }
}
```

## Benefits

1. **Complete Event Lineage**: Track the flow of events through the system
2. **Performance Monitoring**: Identify bottlenecks and latency issues
3. **Debugging**: Understand event causation chains and system behavior
4. **Pattern Discovery**: Analyze event patterns for optimization opportunities

## Future Enhancements

1. Add more detailed event type mapping in trace summary
2. Include event causation chains in summary
3. Add latency percentile statistics
4. Create visualization tools for event traces
5. Add event filtering capabilities