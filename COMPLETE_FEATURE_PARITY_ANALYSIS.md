# Complete Feature Parity Analysis: Non-Declarative vs Declarative Systems

## File Locations

| Component | Non-Declarative | Declarative |
|-----------|-----------------|-------------|
| Coordinator | `/src/core/coordinator/coordinator.py` | `/src/core/coordinator/coordinator_declarative.py` |
| Sequencer | `/src/core/coordinator/sequencer.py` | `/src/core/coordinator/sequencer_declarative.py` |
| Topology | `/src/core/coordinator/topology.py` | `/src/core/coordinator/topology_declarative.py` |

## Executive Summary

**CRITICAL: DO NOT DEPRECATE NON-DECLARATIVE VERSIONS**

The declarative versions are missing essential production features:
1. **No actual execution** - Sequencer returns mock data
2. **No memory management** - Would cause OOM on large runs
3. **No container lifecycle** - Containers never properly managed
4. **No result collection** - Real results never gathered
5. **No composable workflows** - Missing iteration/branching

---

## 1. COORDINATOR COMPARISON

### Core Features

| Feature | Non-Declarative ✅ | Declarative ❌ | Impact |
|---------|-------------------|----------------|--------|
| **Component Discovery** | ✅ Auto-discovers workflows/sequences | ❌ Manual pattern loading | Can't find available components |
| **Composable Workflows** | ✅ Full iteration/branching support | ❌ None | No adaptive workflows |
| **Default Config Merging** | ✅ Deep merge with defaults | ❌ None | Config inheritance broken |
| **Workflow IDs** | ✅ Unique execution tracking | ❌ None | Can't track executions |
| **Event Tracing** | ✅ EventTracer integration | ❌ None | No debugging capability |
| **Trace Level Presets** | ✅ Configurable trace levels | ❌ None | No easy trace config |
| **Primary Metric Extraction** | ✅ For comparisons | ❌ None | Can't compare results |
| **Phase Ordering** | ✅ Topological sort | ❌ Assumes pre-ordered | Dependency errors possible |

### Advanced Features

#### Non-Declarative Composable Workflow Support:
```python
# Supports iteration
should_continue() -> bool
modify_config_for_next() -> Dict

# Supports branching
get_branches() -> List[WorkflowBranch]

# Tracks iterations and branches
'iteration_results': [...],
'branch_results': {...}
```

#### Declarative Conditional Execution (Limited):
```yaml
conditions:
  - type: metric_threshold
    phase: optimization
    metric: sharpe_ratio
    threshold: 1.5
```

### What Declarative Adds:
- ✅ YAML pattern loading
- ✅ Template variable resolution
- ✅ File output support
- ✅ Complex condition types
- ✅ Workflow-level outputs

---

## 2. SEQUENCER COMPARISON

### Critical Missing Features

| Feature | Non-Declarative ✅ | Declarative ❌ | Impact |
|---------|-------------------|----------------|--------|
| **Container Lifecycle** | ✅ Full 6-phase lifecycle | ❌ Mock execution only | **NO ACTUAL EXECUTION** |
| **Memory Management** | ✅ memory/disk/hybrid modes | ❌ None | **OOM ON LARGE RUNS** |
| **Result Collection** | ✅ From streaming metrics | ❌ Returns fake data | **NO REAL RESULTS** |
| **Error Recovery** | ✅ Proper cleanup | ❌ None | **RESOURCE LEAKS** |
| **Data Streaming** | ✅ Streams through containers | ❌ None | **NO DATA FLOW** |

### Container Lifecycle (Non-Declarative):
```python
# Complete lifecycle management:
1. Initialize all containers
2. Start all containers
3. Execute (stream data)
4. Collect results (while running!)
5. Stop containers (reverse order)
6. Cleanup containers (triggers saves!)
```

### Declarative Mock Execution:
```python
# THIS IS ALL IT DOES:
return {
    'success': True,
    'metrics': {
        'sharpe_ratio': 1.5,      # FAKE!
        'total_return': 0.15,     # FAKE!
        'max_drawdown': 0.08      # FAKE!
    }
}
```

### Result Storage (Non-Declarative):
```
./results/
  {workflow_id}/
    {phase_name}/
      containers/
        {container_id}_results.json
      aggregate_results.json
      all_trades.json
      phase_summary.json
```

### What Declarative Adds:
- ✅ Pattern-based sequences
- ✅ Multiple iteration types
- ✅ Config modifiers
- ✅ Sub-sequences support
- ✅ Template resolution

---

## 3. TOPOLOGY COMPARISON

### Implementation Approach

| Feature | Non-Declarative ✅ | Declarative ⚠️ | Impact |
|---------|-------------------|----------------|--------|
| **Creation Method** | Module imports | Pattern loading | Different but OK |
| **Flexibility** | Fixed topologies | Dynamic patterns | Better in declarative |
| **Trace Settings** | Proper nesting | ⚠️ Different structure | May confuse containers |
| **Container Settings** | ✅ Pass-through | ❌ Missing | Can't configure per-container |
| **Mixin Example** | ✅ ContainerTracingMixin | ❌ None | No implementation guide |

### Trace Configuration Structure

#### Non-Declarative (Correct):
```python
config['execution']['trace_settings'] = {
    'trace_id': ...,
    'trace_dir': ...,
    'max_events': ...,
    'container_settings': {...}  # Per-container config
}
```

#### Declarative (Inconsistent):
```python
config['execution']['enable_event_tracing'] = True
config['execution']['trace_settings'] = {
    # Missing proper nesting
    # Missing container_settings
}
```

### What Declarative Adds:
- ✅ YAML pattern support
- ✅ Component creation
- ✅ Route creation
- ✅ Behavior application
- ✅ Foreach loops
- ✅ Pattern matching

---

## 4. INTEGRATION GAPS

### To Make Declarative Functional:

1. **Sequencer Needs**:
   - Import `_execute_topology()` from non-declarative
   - Implement `_collect_phase_results()`
   - Add `_save_results_to_disk()`
   - Remove mock returns
   - Add container lifecycle

2. **Coordinator Needs**:
   - Add component discovery
   - Implement composable workflows
   - Add default config merging
   - Add workflow ID tracking
   - Integrate event tracing

3. **Topology Needs**:
   - Fix trace_settings structure
   - Add container_settings pass-through
   - Include ContainerTracingMixin

---

## 5. RISK ASSESSMENT

### If We Deprecate Now:

1. **All Backtests Break** ❌
   - Returns fake data instead of running
   - No actual container execution

2. **Memory Crashes** ❌
   - No disk/hybrid storage
   - Large backtests will OOM

3. **No Results** ❌
   - Streaming metrics ignored
   - No result collection

4. **Resource Leaks** ❌
   - No cleanup on errors
   - Containers left running

5. **No Advanced Workflows** ❌
   - No iteration support
   - No branching support
   - No adaptive strategies

---

## 6. RECOMMENDATION

### Option 1: Complete Implementation (Recommended)
Add missing features to declarative versions:
- Real topology execution
- Container lifecycle management
- Memory management modes
- Result collection
- Error recovery
- Composable workflows

**Estimated effort**: 3-5 days

### Option 2: Hybrid Approach
Use declarative patterns with non-declarative execution:
- Keep pattern-based configuration
- Delegate to non-declarative for execution
- Best of both worlds

**Estimated effort**: 1-2 days

### Option 3: Gradual Migration
- Keep both versions
- Use declarative for simple cases
- Use non-declarative for production
- Migrate features incrementally

**Estimated effort**: Ongoing

---

## 7. CONCLUSION

The declarative approach offers better flexibility and maintainability through:
- YAML-based configuration
- Pattern composition
- Template resolution
- Conditional execution

However, it currently lacks the **core execution engine** that makes the system functional. The non-declarative versions contain years of production-hardened code for:
- Container lifecycle management
- Memory optimization
- Error recovery
- Result collection
- Performance optimization

**DO NOT DEPRECATE** until declarative versions have feature parity or use a hybrid approach that leverages both systems' strengths.