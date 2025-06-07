# Event System Cleanup Checklist

## Executive Summary

The event system has become spaghetti'd with three parallel implementations:
1. Base Event/EventBus (no correlation)
2. TracedEvent/TracedEventBus (full tracing)
3. MetricsEventTracer (manual correlation extraction)

This violates our single source of truth principle. We need ONE canonical implementation.

## Goal

Create a single, unified event system with:
- **One Event class** with built-in correlation_id
- **One EventBus class** with optional tracing
- **No duplicate implementations**

## Phase 1: Update Core Event Structure

### 1.1 Enhance Event Class
- [x] Add `correlation_id: Optional[str]` field to Event dataclass
- [x] Add `causation_id: Optional[str]` field for event chains
- [x] Add `sequence_number: Optional[int]` for ordering
- [x] Update event factory functions to accept correlation_id
- [x] Update Event.__post_init__ to generate correlation_id if needed

File: `src/core/types/events.py`

## Phase 2: Create Unified EventBus

### 2.1 Merge TracedEventBus functionality into EventBus
- [x] Add `_tracer: Optional[EventTracer] = None` field
- [x] Add `enable_tracing(trace_config: Dict[str, Any])` method
- [x] Add `disable_tracing()` method
- [x] Update `publish()` to optionally trace if tracer is enabled
- [x] Add `_current_correlation_id` for automatic correlation propagation
- [x] Add tracing helper methods (get_trace_summary, etc) that return None if not tracing

File: `src/core/events/event_bus.py`

### 2.2 Update publish method logic
- [x] If tracer enabled: trace event with timing
- [x] If processing another event: set causation_id
- [x] Always execute normal publish logic

## Phase 3: Update EventTracer

### 3.1 Make EventTracer work with regular Events
- [x] Change all `TracedEvent` references to `Event`
- [x] Remove TracedEvent creation, work with Events directly
- [x] Update trace_event to enhance existing Event metadata
- [x] Ensure correlation_id is propagated to event if not set

File: `src/core/events/tracing/event_tracer.py`

## Phase 4: Delete Duplicate Code

### 4.1 Files to DELETE completely
- [x] `src/core/events/tracing/traced_event.py` (content removed, imports updated)
- [x] `src/core/events/tracing/traced_event_bus.py` (content removed, imports updated)
- [x] Any other enhanced/improved event files

### 4.2 Update imports in remaining tracing files
- [x] `src/core/events/tracing/__init__.py` - Remove TracedEvent, TracedEventBus
- [ ] `src/core/events/tracing/unified_tracer.py` - Update if it exists
- [ ] Other tracing files - Update to use Event instead of TracedEvent

## Phase 5: Update Container Integration

### 5.1 Container uses unified EventBus
- [x] Container already uses EventBus - no change needed!
- [x] Update `_setup_tracing()` to call `event_bus.enable_tracing()` instead of creating separate tracer
- [x] Remove standalone event_tracer field

File: `src/core/containers/container.py`

## Phase 6: Update MetricsEventTracer

### 6.1 Use built-in correlation_id
- [x] Update `_store_trade_event()` to use `event.correlation_id`
- [x] Remove manual order_id extraction logic (kept as fallback)
- [x] Update type hints from Event to Event (no change, but verify)
- [x] Ensure it works with or without tracing enabled

File: `src/core/containers/metrics.py`

## Phase 7: Update Event Creation

### 7.1 Propagate correlation throughout system
- [ ] Strategy signals include correlation_id
- [ ] Order events maintain correlation_id
- [ ] Fill events maintain correlation_id
- [ ] Portfolio updates maintain correlation_id

Files: Various strategy and execution files

## Phase 8: Testing and Validation

### 8.1 Verify functionality
- [ ] Test basic event publishing without tracing
- [ ] Test with tracing enabled
- [ ] Test correlation_id propagation
- [ ] Test MetricsEventTracer with new correlation
- [ ] Ensure no performance degradation when tracing disabled

### 8.2 Clean up any remaining references
- [ ] Search for TracedEvent references
- [ ] Search for TracedEventBus references
- [ ] Update any documentation

## Phase 9: Ensure Config-Based Tracing Works

### 9.1 Verify tracing config flows through orchestration
- [x] Config YAML supports per-container tracing settings
- [x] Coordinator passes tracing config to Sequencer (applies trace level presets)
- [x] Sequencer passes tracing config to TopologyBuilder
- [x] TopologyBuilder creates containers with correct tracing settings
- [x] Container._setup_tracing() reads config and enables tracing appropriately

### 9.2 Add default trace levels for common scenarios
- [x] Define default trace config for optimization workflows (minimal tracing)
- [x] Define default trace config for debugging (full tracing)
- [x] Define default trace config for production (no tracing)
- [x] Support trace level presets: 'none', 'minimal', 'normal', 'debug'
- [x] Map trace levels to appropriate max_events and retention settings

Example trace level presets:
```yaml
# Optimization workflow - minimal memory usage
trace_level: minimal
# Maps to:
# - Portfolio containers: Only track open trades (trade_complete retention)
# - Metrics persist, events deleted after trade closes  
# - Other containers: Tracing disabled
# - Absolute minimum memory usage

# Debugging - full visibility
trace_level: debug  
# Maps to:
# - All containers: max_events: 50000
# - Full event history retained
# - Trade history and equity curves stored
# - Maximum visibility for debugging

# Production - no overhead
trace_level: none
# Maps to:
# - Tracing disabled completely
# - Containers maintain own metrics
```

Note: Trace levels are **per-container**, not global. Each container type
gets appropriate settings based on its role.

## Configuration

### Example: Minimal (No Tracing)
```yaml
execution:
  enable_event_tracing: false  # No overhead
```

### Example: Debug (Full Tracing)
```yaml
execution:
  enable_event_tracing: true
  trace_settings:
    trace_dir: ./traces
    max_events: 50000
    trace_pattern: "ORDER|FILL|SIGNAL"  # Optional filtering
```

## Success Criteria

1. **One Event class** - No TracedEvent
2. **One EventBus class** - No TracedEventBus
3. **Correlation always available** - Even without tracing
4. **Tracing is optional** - No performance hit when disabled
5. **Clean imports** - No enhanced/improved variants

## Implementation Notes

- **Fast and destructive** - Delete duplicates immediately
- **No gradual migration** - Update all at once
- **No backward compatibility** - Fix all breaks immediately
- **No wrappers** - Direct implementation only

## Order of Implementation

1. Update Event class (add fields)
2. Update EventBus (add optional tracing)
3. Update EventTracer (use Event not TracedEvent)
4. Delete duplicate files
5. Update imports everywhere
6. Test the system
7. Update MetricsEventTracer
8. Verify correlation flow

This is a breaking change that will touch many files, but results in a much cleaner, single source of truth for events.