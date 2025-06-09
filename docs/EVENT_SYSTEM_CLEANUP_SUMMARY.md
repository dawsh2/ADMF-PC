# Event System Cleanup Summary

## What We Did

We successfully unified the event system by eliminating the parallel implementations and creating a single source of truth for events and event handling.

### 1. Enhanced the Core Event Class

Added built-in correlation support to the base Event class:
- `correlation_id: Optional[str]` - Track related events (e.g., signal→order→fill)
- `causation_id: Optional[str]` - Track what event caused this one
- `sequence_number: Optional[int]` - Order within correlation group

The Event class now auto-generates correlation_id if not provided, using container_id and timestamp.

### 2. Created Unified EventBus

Merged TracedEventBus functionality into the base EventBus:
- Optional tracing via `enable_tracing(trace_config)`
- Automatic correlation_id propagation
- Automatic causation_id setting for nested events
- Performance tracking when tracing is enabled
- Zero overhead when tracing is disabled

Key methods added:
- `enable_tracing()` / `disable_tracing()`
- `set_correlation_id()`
- `get_tracer_summary()`
- `trace_causation_chain()`

### 3. Updated EventTracer

Modified EventTracer to work with regular Events instead of TracedEvent:
- Enhances existing Event metadata instead of creating new objects
- Stores Events directly in memory
- Calculates latency from trace_timing metadata

### 4. Deleted Duplicate Code

Removed:
- `TracedEvent` class - no longer needed
- `TracedEventBus` class - functionality merged into EventBus
- Updated all imports to remove references

### 5. Updated Container Integration

Containers now use the unified approach:
- EventBus with optional tracing (no separate tracer object)
- `_setup_tracing()` calls `event_bus.enable_tracing()`
- Simplified trace management

### 6. Updated MetricsEventTracer

Now uses built-in correlation_id:
- Primary: Uses `event.correlation_id` if available
- Fallback: Extracts order_id from payload (backward compatibility)
- Works seamlessly with or without tracing enabled

## Key Benefits

1. **Single Source of Truth**
   - One Event class with built-in correlation
   - One EventBus with optional tracing
   - No confusion about which implementation to use

2. **Zero Overhead When Disabled**
   - Tracing only active when explicitly enabled
   - No performance impact for normal runs
   - Correlation_id always available for metrics

3. **Cleaner Architecture**
   - No duplicate implementations
   - Clear separation of concerns
   - Easier to understand and maintain

4. **Better Correlation Tracking**
   - Correlation_id is a first-class field
   - Automatic propagation through event chains
   - Works for both metrics and tracing

## Configuration

### Minimal Run (No Tracing)
```yaml
execution:
  enable_event_tracing: false  # No tracing overhead
```

### Debug Run (Full Tracing)
```yaml
execution:
  enable_event_tracing: true
  trace_settings:
    trace_dir: ./traces
    max_events: 50000
```

## Migration Notes

- All existing code continues to work
- Events without correlation_id get one auto-generated
- MetricsEventTracer falls back to order_id extraction if no correlation_id
- No breaking changes to public APIs

## Next Steps

1. Update event creation throughout the system to propagate correlation_ids
2. Ensure strategies create new correlation_ids for each signal
3. Propagate correlation_id through order→fill chain
4. Consider adding correlation_id to portfolio updates

The event system is now much cleaner and provides a solid foundation for both metrics collection and optional debug tracing.