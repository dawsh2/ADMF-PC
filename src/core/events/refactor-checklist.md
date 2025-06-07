# Event System Refactor Checklist

## Pre-Implementation Reading
Before starting the refactor, read these files to understand the current architecture:

### Required Reading
- [ ] `src/core/containers/container.py` - Container implementation
- [ ] `src/core/containers/factory.py` - Container creation patterns
- [ ] `src/core/containers/protocols.py` - Container interfaces
- [ ] `src/core/communication/factory.py` - Adapter creation
- [ ] `src/core/coordinator/topology.py` - Topology building
- [ ] `src/core/coordinator/topology_helpers.py` - Topology patterns

### Understand Current Event Flow
- [ ] How containers currently create and use EventBus
- [ ] How adapters subscribe to events
- [ ] How tracing is currently configured
- [ ] How metrics are tracked

## Phase 0: Container Isolation (Preliminary Work)

### Move Isolation to Container Module
- [ ] Create `src/core/containers/isolation.py` with simplified EventIsolationManager
- [ ] Remove global singleton pattern
- [ ] Focus on container lifecycle management
- [ ] Implement container-scoped event bus creation
- [ ] Add thread-local container context support

### Container Factory Updates
- [ ] Update `container.py` to use isolation manager
- [ ] Each container gets isolated event bus from isolation manager
- [ ] Container factory passes isolation manager to containers
- [ ] Topology builder manages the isolation manager instance

## Phase 1: Core Event System Refactor

### 1.1 Clean EventBus Implementation
- [ ] Create new `src/core/events/bus.py` with pure pub/sub EventBus
- [ ] No tracing logic in EventBus
- [ ] Support for observer attachment/detachment
- [ ] Basic metrics (event count, error count)
- [ ] Thread-safe for single container use

### 1.2 Protocol Definitions
- [ ] Create `src/core/events/protocols.py` with:
  - [ ] EventObserverProtocol
  - [ ] EventTracerProtocol
  - [ ] EventStorageProtocol
  - [ ] EventFilterProtocol
  - [ ] EventTypeRegistryProtocol

### 1.3 Event Types and Registry
- [ ] Update `src/core/events/types.py` with existing Event class
- [ ] Add EventTypeRegistry for type mapping
- [ ] Define type transformation rules (BAR → FEATURE → SIGNAL → ORDER → FILL)
- [ ] Container role inference patterns
- [ ] Event validation helpers

### 1.4 Observer Implementations
- [ ] Create `src/core/events/observers/` directory
- [ ] Implement EventTracer observer
- [ ] Implement MetricsObserver (adapts existing MetricsEventTracer)
- [ ] Implement MemoryMonitorObserver
- [ ] Create observer factory functions

### 1.5 Storage Backends
- [ ] Create `src/core/events/storage/` directory
- [ ] Implement MemoryEventStorage with configurable retention
- [ ] Implement DiskEventStorage with compression
- [ ] Support hierarchical storage (root/container structure)
- [ ] Implement retention policies (all, trade_complete, sliding_window, minimal)

## Phase 2: Container Integration

### 2.1 Container Updates
- [ ] Update Container protocol to support observer configuration
- [ ] Add `_setup_event_observers()` method to container.py
- [ ] Container creates observers based on config, NOT topology builder
- [ ] Preserve existing event bus usage patterns

### 2.2 Topology Builder Updates
- [ ] TopologyBuilder passes tracing_config to containers (not tracer objects!)
- [ ] Configuration includes:
  - [ ] Which containers to trace
  - [ ] Memory limits per container
  - [ ] Storage backend selection
  - [ ] Retention policies
- [ ] Topology builder creates isolation manager and passes to container factory

### 2.3 Configuration Schema
- [ ] Update configuration schemas to include observer settings
- [ ] Container-specific trace configuration
- [ ] Memory limits and storage options
- [ ] Event type filtering options

Example configuration:
```python
{
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {
            'trace_id': 'backtest_20240106_123456',
            'trace_dir': './traces',
            'container_settings': {
                'portfolio_*': {
                    'observers': ['metrics', 'tracer'],
                    'retention_policy': 'trade_complete',
                    'max_events': 10000,
                    'storage_backend': 'memory'
                },
                'strategy_*': {
                    'observers': ['tracer'],
                    'events_to_trace': ['SIGNAL'],
                    'storage_backend': 'disk'
                }
            }
        }
    }
}
```

## Phase 3: Advanced Features

### 3.1 Semantic Events (Additive)
- [ ] Create `src/core/events/semantic/` directory
- [ ] Implement semantic event protocols
- [ ] Add semantic event classes (TradingSignal, FillEvent, etc.)
- [ ] Support both legacy and semantic events simultaneously
- [ ] Conversion helpers between formats

### 3.2 Type Flow Validation
- [ ] Implement type flow validator using EventTypeRegistry
- [ ] Validate routes at topology creation time
- [ ] Runtime validation of event flows
- [ ] Warning system for type mismatches

### 3.3 Memory Monitoring
- [ ] Implement container memory observer
- [ ] Track memory by event type
- [ ] Detect memory leaks (positions not freed)
- [ ] Emit memory warnings as events

## Phase 4: Migration and Testing

### 4.1 Backward Compatibility
- [ ] Ensure existing code continues to work
- [ ] MetricsEventTracer wrapper for observer pattern
- [ ] Legacy event support alongside semantic events
- [ ] Gradual migration path documented

### 4.2 Testing
- [ ] Unit tests for all observer implementations
- [ ] Integration tests for container + observers
- [ ] Performance tests for event throughput
- [ ] Memory leak tests with observers

### 4.3 Documentation
- [ ] Update module READMEs
- [ ] Add observer usage examples
- [ ] Document configuration options
- [ ] Migration guide from old system

## Implementation Notes

### Container Instantiation Flow
1. TopologyBuilder creates topology definition with tracing_config
2. Container factory receives config including trace settings
3. Container factory creates isolation manager
4. Each container:
   - Gets isolated event bus from isolation manager
   - Creates observers based on its specific config
   - Attaches observers to its event bus
   - No changes to existing event publishing code!

### Key Design Decisions
- **Isolation moves to containers module** - It's about container lifecycle, not events
- **Containers create their own observers** - Based on config passed from topology
- **No breaking changes** - Existing event flow continues to work
- **Composition everywhere** - Observers, storage, filters all composable
- **Configuration-driven** - Behavior determined by config, not code

### Success Criteria
- [ ] Existing tests pass without modification
- [ ] Event tracing configurable per container
- [ ] Memory-aware retention policies work
- [ ] Hierarchical storage structure created
- [ ] No performance regression
- [ ] Clean separation of concerns

## Post-Implementation Cleanup
- [ ] Remove old tracing code from EventBus
- [ ] Delete enhanced_* files if any exist
- [ ] Update all imports to use new structure
- [ ] Run full test suite
- [ ] Performance profiling with observers