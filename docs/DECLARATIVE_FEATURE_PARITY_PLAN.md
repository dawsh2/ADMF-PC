# Declarative System Feature Parity Implementation Plan

## Overview
This plan details the steps needed to bring the declarative coordinator system to full feature parity with the imperative system.

## Phase 1: Core Execution Engine (Priority: CRITICAL)

### 1.1 Replace Mock Execution in sequencer_declarative.py

**Current State (Line 424-436)**:
```python
# Mock execution for now
return {
    'success': True,
    'containers_executed': len(topology.get('containers', {})),
    'metrics': {
        'sharpe_ratio': 1.5,
        'total_return': 0.15,
        'max_drawdown': 0.08
    }
}
```

**Required Implementation**:
```python
def _execute_single_topology(self, topology_mode: str,
                           config: Dict[str, Any],
                           phase_config: PhaseConfig,
                           context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single topology with full container lifecycle."""
    
    # Build topology
    topology_definition = self._build_topology_definition(
        topology_mode, config, phase_config, context
    )
    topology = self.topology_builder.build_topology(topology_definition)
    
    # Execute with proper lifecycle
    execution_result = self._execute_topology(topology, phase_config, context)
    
    # Collect and process results
    phase_results = self._collect_phase_results(topology)
    
    # Handle storage modes
    return self._process_results(
        execution_result, phase_results, phase_config, context
    )
```

### 1.2 Add Container Lifecycle Management

**Required Methods**:
```python
def _execute_topology(self, topology: Dict[str, Any], 
                     phase_config: PhaseConfig, 
                     context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute topology with 6-phase lifecycle."""
    # Phase 1: Initialize
    # Phase 2: Start
    # Phase 3: Execute (stream data)
    # Phase 4: Collect results
    # Phase 5: Stop (reverse order)
    # Phase 6: Cleanup (triggers saves)

def _run_topology_execution(self, topology: Dict[str, Any],
                           phase_config: PhaseConfig,
                           context: Dict[str, Any]) -> Dict[str, Any]:
    """Run actual execution based on topology mode."""
    # Handle: backtest, signal_generation, optimization

def _collect_phase_results(self, topology: Dict[str, Any]) -> Dict[str, Any]:
    """Collect results from streaming metrics."""
```

### 1.3 Memory Management Implementation

**Add Storage Modes**:
```python
def _process_results(self, execution_result: Dict[str, Any],
                    phase_results: Dict[str, Any],
                    phase_config: PhaseConfig,
                    context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle memory/disk/hybrid storage modes."""
    
    results_storage = phase_config.config.get('results_storage', 'memory')
    
    if results_storage == 'disk':
        # Save everything to disk, return only paths
        results_path = self._save_results_to_disk(phase_results, phase_config, context)
        return {
            'results_saved': True,
            'results_path': results_path,
            'summary': self._create_summary(phase_results),
            'aggregate_metrics': phase_results.get('aggregate_metrics', {})
        }
    elif results_storage == 'hybrid':
        # Save large data to disk, keep summaries in memory
        results_path = self._save_results_to_disk(phase_results, phase_config, context)
        return {
            'results_saved': True,
            'results_path': results_path,
            'summary': self._create_summary(phase_results),
            'aggregate_metrics': phase_results.get('aggregate_metrics', {}),
            'phase_results': self._extract_essential_data(phase_results)
        }
    else:  # 'memory'
        # Keep everything in memory (risky for large runs)
        return {
            'phase_results': phase_results,
            'aggregate_metrics': phase_results.get('aggregate_metrics', {})
        }
```

## Phase 2: Coordinator Enhancements

### 2.1 Add Component Discovery to coordinator_declarative.py

**Required Implementation**:
```python
def _discover_components(self):
    """Auto-discover workflows, sequences, and strategies."""
    self.discovered_workflows = self._discover_workflows()
    self.discovered_sequences = self._discover_sequences()
    self.discovered_strategies = self._discover_strategies()
    self.discovered_classifiers = self._discover_classifiers()

def _discover_workflows(self) -> Dict[str, Any]:
    """Scan for workflow patterns and classes."""
    # Check YAML patterns
    # Check Python modules
    # Merge discovered components
```

### 2.2 Implement Composable Workflows

**Add Support for**:
```python
class ComposableWorkflowSupport:
    def should_continue(self) -> bool:
        """Check if workflow should continue iterating."""
        
    def modify_config_for_next(self) -> Dict[str, Any]:
        """Modify config for next iteration."""
        
    def get_branches(self) -> List[WorkflowBranch]:
        """Get workflow branches for parallel execution."""
```

### 2.3 Deep Config Merging

**Implementation**:
```python
def _merge_configs(self, base: Dict, override: Dict) -> Dict:
    """Deep merge configurations with proper inheritance."""
    # Recursive merge
    # Handle lists appropriately
    # Preserve override precedence
```

### 2.4 Event Tracing Integration

**Add EventTracer Support**:
```python
def _setup_event_tracing(self, config: Dict[str, Any]) -> Optional[EventTracer]:
    """Setup event tracing if enabled."""
    if config.get('execution', {}).get('enable_event_tracing', False):
        from ..events.tracing import EventTracer
        return EventTracer(config['execution']['trace_settings'])
    return None
```

## Phase 3: Topology Improvements

### 3.1 Fix Trace Settings Structure in topology_declarative.py

**Current (Incorrect)**:
```python
config['execution']['enable_event_tracing'] = True
config['execution']['trace_settings'] = {...}
```

**Required (Correct)**:
```python
config['execution']['trace_settings'] = {
    'trace_id': tracing_config.get('trace_id'),
    'trace_dir': tracing_config.get('trace_dir', './traces'),
    'max_events': tracing_config.get('max_events', 10000),
    'container_settings': tracing_config.get('container_settings', {})
}
```

### 3.2 Add Container Settings Pass-through

**Implementation**:
```python
def _build_context(self, pattern: Dict[str, Any], config: Dict[str, Any], 
                  tracing_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build evaluation context with proper trace settings."""
    # ... existing code ...
    
    # Pass through container-specific settings
    if 'container_settings' in tracing_config:
        config['execution']['trace_settings']['container_settings'] = \
            tracing_config['container_settings']
```

## Phase 4: Testing & Validation

### 4.1 Create Feature Parity Tests

```python
# test_declarative_parity.py
class TestDeclarativeParity:
    def test_container_lifecycle(self):
        """Verify all 6 phases execute correctly."""
        
    def test_memory_management(self):
        """Test memory/disk/hybrid modes."""
        
    def test_result_collection(self):
        """Verify streaming metrics collection."""
        
    def test_error_recovery(self):
        """Test cleanup on errors."""
```

### 4.2 Migration Tests

```python
def test_imperative_to_declarative_migration():
    """Run same config on both systems, compare results."""
    imperative_result = run_imperative(config)
    declarative_result = run_declarative(config)
    assert_results_equivalent(imperative_result, declarative_result)
```

## Implementation Order

### Week 1: Core Execution
1. Replace mock execution with real container lifecycle
2. Implement _execute_topology with 6 phases
3. Add _run_topology_execution for different modes
4. Implement _collect_phase_results

### Week 2: Memory & Storage
1. Add memory management modes
2. Implement _save_results_to_disk
3. Add result aggregation
4. Test with large datasets

### Week 3: Coordinator Features
1. Add component discovery
2. Implement composable workflows
3. Add deep config merging
4. Integrate event tracing

### Week 4: Testing & Polish
1. Fix topology trace settings
2. Add comprehensive tests
3. Document migration path
4. Performance optimization

## Success Criteria

The declarative system has feature parity when:

1. ✅ All backtests produce identical results to imperative
2. ✅ Memory usage matches imperative system
3. ✅ Error recovery works correctly
4. ✅ All tests pass
5. ✅ Performance is within 10% of imperative

## Risk Mitigation

1. **Keep imperative system intact** - No changes until declarative is proven
2. **Incremental testing** - Test each feature as implemented
3. **A/B testing** - Run both systems in parallel initially
4. **Rollback plan** - Can revert to imperative at any time

## Next Steps

1. Start with `_execute_single_topology` replacement
2. Copy container lifecycle logic from imperative
3. Add proper result collection
4. Test with simple backtest first
5. Gradually add remaining features