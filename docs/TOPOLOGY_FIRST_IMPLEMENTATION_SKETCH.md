# Topology-First Implementation Sketch

## Minimal Changes for Maximum Impact

### 1. Add to Coordinator.__init__
```python
from .topology_runner import TopologyRunner

class Coordinator:
    def __init__(self):
        # ... existing init ...
        
        # Add topology runner
        self.topology_runner = TopologyRunner(
            topology_builder=self.topology_builder
        )
```

### 2. Add run_topology Method
```python
def run_topology(self, topology_name: str, 
                config: Dict[str, Any],
                execution_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute a topology directly without workflow wrapping.
    
    This is the most basic execution - just runs the topology once.
    """
    # Validate topology exists
    if topology_name not in self.topology_patterns:
        raise ValueError(f"Unknown topology: {topology_name}")
    
    # Delegate to topology runner
    return self.topology_runner.run_topology(
        topology_name=topology_name,
        config=config,
        execution_id=execution_id
    )
```

### 3. Update run_workflow for Cleaner Logic
```python
def run_workflow(self, workflow_definition: Union[str, Dict[str, Any]], 
                workflow_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute a workflow, with special handling for direct topology requests."""
    
    # If it's a string, check if it's a topology
    if isinstance(workflow_definition, str):
        # Direct topology execution if no workflow exists
        if (workflow_definition not in self.workflow_patterns and 
            workflow_definition in self.topology_patterns):
            logger.info(f"Executing topology '{workflow_definition}' directly")
            return self.run_topology(workflow_definition, {})
    
    # ... rest of existing workflow logic ...
```

### 4. Simple Backwards-Compatible Config Handling
```python
# In run_workflow, when processing config dict:
if isinstance(workflow_definition, dict):
    config = workflow_definition
    
    # Check for direct topology request
    if 'topology' in config and 'workflow' not in config and 'phases' not in config:
        topology_name = config.pop('topology')
        logger.info(f"Executing topology '{topology_name}' from config")
        return self.run_topology(topology_name, config)
    
    # ... existing workflow logic ...
```

## Testing the Implementation

### Test 1: Direct Topology Execution
```python
def test_direct_topology_execution():
    coordinator = Coordinator()
    
    result = coordinator.run_topology('signal_generation', {
        'data_source': 'file',
        'data_path': 'test_data.csv',
        'symbols': ['SPY'],
        'strategies': [{
            'type': 'momentum',
            'params': {'period': 20}
        }]
    })
    
    assert result['success']
    assert result['topology'] == 'signal_generation'
    assert 'execution_id' in result
```

### Test 2: Config-Based Topology
```python
def test_config_based_topology():
    coordinator = Coordinator()
    
    # Config with topology field
    config = {
        'topology': 'backtest',
        'data': {...},
        'strategies': [...]
    }
    
    result = coordinator.run_workflow(config)
    
    # Should execute as topology, not wrapped workflow
    assert result['topology'] == 'backtest'
    assert 'phases' not in result  # Not a workflow
```

### Test 3: Backwards Compatibility
```python
def test_backwards_compatibility():
    coordinator = Coordinator()
    
    # Old-style config (no topology or workflow specified)
    config = {
        'data': {...},
        'strategies': [...]
    }
    
    result = coordinator.run_workflow(config)
    
    # Should still work with default workflow
    assert result['workflow'] == 'simple_backtest'
```

## Migration Path

### Phase 1: Add Functionality (No Breaking Changes)
1. Add TopologyRunner class
2. Add run_topology method to Coordinator
3. Update docs with examples

### Phase 2: Update Examples
1. Show topology-first approach in tutorials
2. Update example configs
3. Add CLI support for --topology

### Phase 3: Deprecate Auto-Wrapping
1. Add deprecation warning for auto-wrapping
2. Encourage explicit topology or workflow specification
3. Update all tests to use new approach

### Phase 4: Clean Architecture (Major Version)
1. Remove auto-wrapping logic
2. Require explicit topology/workflow specification
3. Simplify Coordinator logic

## End Result

A much cleaner architecture where:
- **Topologies** are the atomic unit of execution
- **Sequences** provide iteration over topologies
- **Workflows** compose multiple sequenced topologies
- Each level builds naturally on the previous one
- No magic or auto-wrapping needed