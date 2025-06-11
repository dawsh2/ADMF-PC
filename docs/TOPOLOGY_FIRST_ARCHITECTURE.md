# Topology-First Architecture Proposal

## Current Problem

The system currently requires workflows as the base concept, with topologies being auto-wrapped as single-phase workflows. This is backwards from natural composability.

## Proposed Architecture

### 1. Base Level: Topology Execution

```python
# Simple, direct topology execution
coordinator.run_topology('signal_generation', config={
    'data_source': 'file',
    'data_path': 'SPY.csv',
    'symbols': ['SPY'],
    'strategies': [{'type': 'momentum', 'params': {...}}]
})
```

### 2. Sequence Level: Iteration Over Topologies

```python
# Run topology with different data windows
coordinator.run_sequence('walk_forward', {
    'topology': 'backtest',
    'window_size': 252,
    'step_size': 21,
    'config': {...}
})
```

### 3. Workflow Level: Composition of Sequences

```python
# Compose multiple topology+sequence combinations
coordinator.run_workflow('research_pipeline', {
    'phases': [
        {
            'name': 'signal_generation',
            'topology': 'signal_generation',
            'sequence': 'single_pass'
        },
        {
            'name': 'parameter_optimization', 
            'topology': 'backtest',
            'sequence': 'parameter_sweep'
        },
        {
            'name': 'validation',
            'topology': 'backtest',
            'sequence': 'walk_forward'
        }
    ]
})
```

## Implementation Changes

### 1. Add `run_topology` method to Coordinator

```python
def run_topology(self, topology_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a topology directly without workflow wrapper.
    
    This is the base execution primitive - just runs a topology once with given config.
    """
    # Build topology
    topology_def = {
        'mode': topology_name,
        'config': config,
        'metadata': self._build_metadata()
    }
    
    # Add tracing if configured
    if config.get('execution', {}).get('enable_event_tracing'):
        topology_def['tracing_config'] = self._extract_trace_config(config)
    
    # Build and execute
    topology = self.topology_builder.build_topology(topology_def)
    return self._execute_topology_direct(topology, config)
```

### 2. Refactor `run_workflow` to use `run_topology`

```python
def run_workflow(self, workflow_definition: Union[str, Dict[str, Any]], 
                workflow_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute a workflow (composition of topologies + sequences).
    """
    # If just a topology name, delegate to run_topology
    if isinstance(workflow_definition, str) and workflow_definition in self.topology_patterns:
        return self.run_topology(workflow_definition, {})
    
    # Otherwise handle as full workflow...
    # Phases would internally call run_topology with sequence handling
```

### 3. Natural CLI Usage

```bash
# Direct topology execution (most common for debugging/testing)
python main.py --topology signal_generation --config config.yaml

# Sequence execution (for systematic analysis)
python main.py --sequence walk_forward --topology backtest --config config.yaml

# Full workflow (for complex pipelines)
python main.py --workflow research_pipeline --config config.yaml
```

## Benefits

1. **Natural Composability**: Build from simple (topology) to complex (workflow)
2. **Better Debugging**: Can test topologies in isolation
3. **Clearer Mental Model**: Topology = wiring, Sequence = iteration, Workflow = composition
4. **No Magic Wrapping**: Everything is explicit
5. **Gradual Complexity**: Users can start with topologies and grow into workflows

## Migration Path

1. Add `run_topology` method
2. Keep current `run_workflow` for compatibility
3. Internally refactor to use topology-first approach
4. Update examples to show natural progression
5. Eventually deprecate auto-wrapping behavior