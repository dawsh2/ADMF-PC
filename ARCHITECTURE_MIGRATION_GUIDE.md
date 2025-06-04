# Architecture Migration Guide

## Migrating to Standard Pattern-Based Architecture

This guide helps migrate existing code to comply with the mandatory pattern-based architecture standard.

## Overview

The standard architecture enforces:
1. **Container Factory** (`core/containers/factory.py`) - Container creation only
2. **Communication Factory** (`core/communication/factory.py`) - Communication adapters only  
3. **Workflow Manager** (`coordinator/workflows/workflow_manager.py`) - Pattern orchestration

## Migration Steps

### Step 1: Identify Non-Compliant Code

Look for these anti-patterns:

```bash
# Find files that mix responsibilities
find src/ -name "*.py" -exec grep -l "create.*container.*adapter\|create.*adapter.*container" {} \;

# Find duplicate pattern definitions
find src/ -name "*.py" -exec grep -l "backtest.*pattern\|signal_generation.*pattern" {} \;

# Find files with problematic names
find src/ -name "*enhanced*" -o -name "*improved*" -o -name "*advanced*"
```

### Step 2: Files to Migrate

#### High Priority (Remove These)

```
src/core/coordinator/workflows/factory.py          # DUPLICATE - Remove
src/core/coordinator/workflows/workflow_factory.py  # DUPLICATE - Remove  
src/core/coordinator/workflows/containers_pipeline.py # WRAPPER - Remove
```

#### Medium Priority (Refactor These)

```
src/core/coordinator/workflows/container_factories.py # CONSOLIDATE - Functions should move to core factory or be deprecated
```

#### Low Priority (Update These)

```
src/core/coordinator/workflows/__init__.py         # UPDATE - Export standard interfaces
src/core/coordinator/coordinator.py               # UPDATE - Use WorkflowManager
```

### Step 3: Migration Commands

#### Remove Duplicate Files
```bash
rm -f src/core/coordinator/workflows/factory.py
rm -f src/core/coordinator/workflows/workflow_factory.py
rm -f src/core/coordinator/workflows/containers_pipeline.py
```

#### Update Imports
```bash
# Replace old imports with standard ones
find src/ -name "*.py" -exec sed -i 's/from \.workflows\.factory import/from ..containers.factory import/g' {} \;
find src/ -name "*.py" -exec sed -i 's/from \.workflows\.workflow_factory import/from .workflow_manager import/g' {} \;
```

### Step 4: Code Migration Examples

#### Before (Non-Compliant)
```python
# OLD: Mixed responsibilities
from .workflows.factory import create_backtest_with_communication

def execute_backtest(config):
    # Creates both containers AND communication - WRONG!
    everything = create_backtest_with_communication(config)
    return everything.run()
```

#### After (Compliant)
```python
# NEW: Separated responsibilities
from .workflows.workflow_manager import WorkflowManager

async def execute_backtest(config):
    # Delegates to appropriate factories - CORRECT!
    workflow_manager = WorkflowManager()
    result = await workflow_manager.execute(config)
    return result
```

#### Before (Duplicate Pattern Definition)
```python
# workflows/patterns.py - DUPLICATE!
BACKTEST_PATTERN = {
    'containers': ['data', 'strategy', 'risk'],
    'communication': [...]
}

# coordinator/patterns.py - DUPLICATE!
BACKTEST_WORKFLOW = {
    'containers': ['data', 'strategy', 'risk'],
    'adapters': [...]
}
```

#### After (Single Definition)
```python
# workflows/workflow_manager.py - SINGLE SOURCE OF TRUTH!
self._workflow_patterns = {
    'simple_backtest': {
        'container_pattern': 'simple_backtest',    # → Container Factory
        'communication_config': [...]              # → Communication Factory
    }
}
```

### Step 5: Update Container Registrations

Ensure all container types are registered with the global factory:

```python
# In __init__.py or module startup
from core.containers.factory import register_container_type
from .container_factories import CONTAINER_FACTORIES

# Register all container types
for role, factory_func in CONTAINER_FACTORIES.items():
    register_container_type(role, factory_func)
```

### Step 6: Update Communication Patterns

Move communication configurations to WorkflowManager:

```python
# OLD: Communication config in container patterns
container_pattern = ContainerPattern(
    name="backtest",
    structure={...},
    communication_config={...}  # WRONG!
)

# NEW: Communication config in WorkflowManager
def _get_simple_backtest_communication(self, containers):
    return [{
        'name': 'backtest_pipeline',
        'type': 'pipeline',
        'event_flow': [...]
    }]
```

## Validation

### Automated Checks
```python
def validate_architecture_compliance():
    """Check architecture compliance."""
    errors = []
    
    # Check no communication config in container patterns
    container_patterns = get_all_container_patterns()
    for pattern in container_patterns:
        if hasattr(pattern, 'communication_config'):
            errors.append(f"Pattern {pattern.name} has communication config")
    
    # Check no container creation in communication factory
    comm_factory_code = read_file('core/communication/factory.py')
    if 'create_container' in comm_factory_code:
        errors.append("Communication factory creates containers")
    
    # Check workflow patterns are only defined in WorkflowManager
    workflow_files = find_files_with_pattern_definitions()
    if len(workflow_files) > 1:
        errors.append(f"Workflow patterns defined in multiple files: {workflow_files}")
    
    return errors
```

### Manual Checks
- [ ] No communication config in container patterns
- [ ] No container creation in communication factory
- [ ] Workflow patterns defined only in WorkflowManager
- [ ] All factories have single responsibility
- [ ] No duplicate pattern definitions

## Testing Migration

### Unit Tests
```python
def test_container_factory_only_creates_containers():
    factory = get_global_factory()
    container = factory.compose_pattern('simple_backtest', {})
    
    # Should create containers
    assert isinstance(container, ComposableContainer)
    
    # Should NOT create adapters
    assert not hasattr(container, 'adapters')
    assert not hasattr(container, 'communication_config')

def test_communication_factory_only_creates_adapters():
    factory = AdapterFactory()
    config = [{'name': 'test', 'type': 'pipeline'}]
    
    adapters = factory.create_adapters_from_config(config, {})
    
    # Should create adapters
    assert len(adapters) == 1
    
    # Should NOT create containers
    assert not hasattr(adapters[0], 'child_containers')

def test_workflow_manager_orchestrates_both():
    manager = WorkflowManager()
    config = WorkflowConfig(workflow_type=WorkflowType.BACKTEST)
    
    result = await manager.execute(config, ExecutionContext())
    
    # Should have created both containers and communication
    assert result.success
    assert 'container_structure' in result.metadata
    assert len(manager.active_adapters) > 0
```

### Integration Tests
```python
def test_end_to_end_backtest():
    """Test complete backtest workflow using standard architecture."""
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        data_config={'source': 'csv', 'file_path': 'test.csv'},
        backtest_config={'initial_capital': 10000}
    )
    
    manager = WorkflowManager()
    result = await manager.execute(config, ExecutionContext())
    
    assert result.success
    assert 'portfolio' in result.final_results
    assert result.final_results['portfolio']['final_value'] > 0
```

## Rollback Plan

If migration causes issues:

1. **Revert changes**: Use git to revert to pre-migration state
2. **Keep old imports**: Temporarily maintain old import paths
3. **Gradual migration**: Migrate one workflow pattern at a time
4. **Parallel implementation**: Run old and new side-by-side during transition

## Timeline

- **Week 1**: Remove duplicate factory files
- **Week 2**: Update imports and basic refactoring  
- **Week 3**: Migrate container registrations
- **Week 4**: Move communication patterns to WorkflowManager
- **Week 5**: Testing and validation
- **Week 6**: Documentation updates

## Success Criteria

✅ All workflow patterns defined in one place (WorkflowManager)  
✅ Container factory only creates containers  
✅ Communication factory only creates adapters  
✅ No duplicate pattern definitions  
✅ All tests pass  
✅ Architecture compliance validation passes  

## Support

For migration issues:
1. Check this guide first
2. Review `STANDARD_PATTERN_ARCHITECTURE.md`
3. Ask for clarification with specific examples
4. Test changes incrementally