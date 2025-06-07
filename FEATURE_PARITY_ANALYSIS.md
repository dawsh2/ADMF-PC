# Feature Parity Analysis: Original vs Declarative Coordinator System

## File Locations

| Component | Original (Non-Declarative) | Declarative Version |
|-----------|---------------------------|--------------------|
| Coordinator | `/src/core/coordinator/coordinator.py` | `/src/core/coordinator/coordinator_declarative.py` |
| Sequencer | `/src/core/coordinator/sequencer.py` | `/src/core/coordinator/sequencer_declarative.py` |
| Topology | `/src/core/coordinator/topology.py` | `/src/core/coordinator/topology_declarative.py` |

## Executive Summary

**DO NOT PROCEED WITH DEPRECATION** - The declarative versions are missing critical production features that would break the system.

## Detailed Feature Comparison

### 1. Coordinator: coordinator.py vs coordinator_declarative.py

| Feature | Original ✅ | Declarative ❌ | Impact |
|---------|------------|----------------|--------|
| Workflow discovery | ✅ | ❌ Missing | Can't auto-discover workflows |
| Composable workflows (iteration/branching) | ✅ | ❌ Missing | No adaptive workflows |
| Component discovery | ✅ | ❌ Missing | Can't find available components |
| Trace level presets | ✅ | ❌ Missing | No easy trace configuration |
| Deep config merging | ✅ | ❌ Missing | Config inheritance broken |
| Primary metric extraction | ✅ | ❌ Missing | Can't compare results |
| Store full results option | ✅ | ❌ Missing | Memory management broken |

### 2. Sequencer: sequencer.py vs sequencer_declarative.py

| Feature | Original ✅ | Declarative ❌ | Impact |
|---------|------------|----------------|--------|
| Container lifecycle management | ✅ | ❌ Mock only | **CRITICAL: No actual execution** |
| Memory/disk/hybrid storage | ✅ | ❌ Missing | **CRITICAL: OOM on large backtests** |
| Streaming metrics collection | ✅ | ❌ Missing | **CRITICAL: No results collected** |
| Error recovery with cleanup | ✅ | ❌ Missing | Resources leak on errors |
| Results directory organization | ✅ | ❌ Missing | No organized output |
| Container role detection | ✅ | ❌ Missing | Can't find data containers |
| Execution modes (backtest/optimization) | ✅ | ❌ Mock only | Only returns fake data |

### 3. Topology: topology.py vs topology_declarative.py

**File Locations:**
- Non-Declarative: `/Users/daws/ADMF-PC/src/core/coordinator/topology.py`
- Declarative: `/Users/daws/ADMF-PC/src/core/coordinator/topology_declarative.py`

| Feature | Non-Declarative ✅ | Declarative ⚠️ | Impact |
|---------|-------------------|----------------|--------|
| Module-based creation | ✅ | Pattern-based | Different approach (OK) |
| Direct topology imports | ✅ | YAML pattern loading | More flexible (OK) |
| Nested trace_settings structure | ✅ | ⚠️ Different structure | Inconsistent with containers |
| Container-specific trace settings | ✅ | ❌ Missing | Can't configure per container |
| ContainerTracingMixin example | ✅ | ❌ Missing | No guidance for containers |
| Handles 'adapters' in return | ✅ | 'routes' instead | Terminology change |

## Critical Missing Features

### 1. Container Lifecycle (MOST CRITICAL)
```python
# Original sequencer.py has full lifecycle:
# Phase 1: Initialize all containers
# Phase 2: Start all containers  
# Phase 3: Run execution (stream data)
# Phase 4: Collect results while running
# Phase 5: Stop containers
# Phase 6: Cleanup (triggers result saves)

# Declarative has:
return {
    'success': True,
    'containers_executed': len(topology.get('containers', {})),
    'metrics': {
        'sharpe_ratio': 1.5,  # HARDCODED!
        'total_return': 0.15,  # FAKE DATA!
        'max_drawdown': 0.08   # NOT REAL!
    }
}
```

### 2. Memory Management (CRITICAL)
```python
# Original handles three modes:
if results_storage == 'disk':
    # Save everything, return only path
elif results_storage == 'hybrid':  
    # Save large data, keep summary
else:  # 'memory'
    # Keep everything (risky for large runs)

# Declarative: NONE of this exists
```

### 3. Results Collection (CRITICAL)
```python
# Original collects from containers:
for container_id, container in containers.items():
    if hasattr(container, 'streaming_metrics'):
        container_results = container.streaming_metrics.get_results()
        
# Declarative: No collection at all
```

## Feature Parity Checklist

### Must Have Before Migration:
- [ ] Real container lifecycle management in declarative sequencer
- [ ] Memory/disk/hybrid storage modes  
- [ ] Streaming metrics collection from containers
- [ ] Error recovery with proper cleanup
- [ ] Results directory organization
- [ ] Composable workflow support (iteration/branching)
- [ ] Component/workflow discovery
- [ ] Trace level presets
- [ ] Deep config merging
- [ ] Fix topology.py trace_settings structure

### Nice to Have:
- [ ] ContainerTracingMixin documentation
- [ ] Primary metric extraction
- [ ] Store full results option

## Topology Implementation Deep Dive

The topology implementations have fundamentally different approaches:

### Non-Declarative (topology.py):
- **Imports specific topology modules**: `from .topologies import create_backtest_topology`
- **Clean trace_settings structure**: Properly nested under `config['execution']['trace_settings']`
- **Container-specific settings**: Passes through `container_settings` from tracing config
- **ContainerTracingMixin**: Shows how containers should handle their own tracing
- **Returns**: `{'containers': {...}, 'adapters': [...], 'metadata': {...}}`

### Declarative (topology_declarative.py):
- **Loads patterns from YAML**: More flexible, data-driven approach
- **Different trace structure**: Adds to `config['execution']` but not nested properly
- **Missing container settings**: Doesn't pass through container-specific trace settings
- **No mixin example**: No guidance on container tracing implementation
- **Returns**: `{'containers': {...}, 'routes': [...], 'metadata': {...}}` (terminology change)

The declarative approach is more powerful but needs the trace settings structure fixed to match what containers expect.

## Risk Assessment

**Migrating now would:**
1. **Break all backtests** - Returns fake data instead of running
2. **Cause OOM crashes** - No memory management for large runs
3. **Lose all results** - No metrics collection from containers
4. **Leak resources** - No proper cleanup on errors
5. **Break workflows** - No composable/iterative workflows

## Recommendation

**DO NOT MIGRATE YET**. The declarative approach is architecturally sound but the implementation is incomplete. Options:

1. **Complete the implementation** - Add all missing features (2-3 days work)
2. **Hybrid approach** - Use declarative patterns but keep original execution
3. **Gradual migration** - Start with simple cases, keep original for complex

The declarative system needs to be a **superset** of features, not a subset.