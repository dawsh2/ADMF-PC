# Refactored Coordinator Architecture

This directory contains the refactored coordinator system that follows Protocol + Composition principles.

## Architecture Overview

### Clean Separation of Concerns

1. **Coordinator** (`coordinator.py`)
   - The workflow manager
   - Manages workflow patterns (discovered via decorators)
   - Handles result streaming
   - Manages distributed execution
   - ALWAYS delegates to Sequencer

2. **Sequencer** (`sequencer.py`)
   - Executes ALL workflows (even single phase)
   - Owns TopologyBuilder
   - Manages phase transitions
   - Handles checkpointing

3. **TopologyBuilder** (`topology_builder.py`)
   - ONLY builds topologies
   - Single public method: `build_topology()`
   - No workflow logic, no execution

### Key Features

- **Protocol-Based Design**: No inheritance, uses protocols for clean interfaces
- **Composition**: Components own their dependencies
- **Decorator Discovery**: Workflows discovered automatically via `@workflow` decorator
- **Unified Execution**: Everything goes through Sequencer (no special cases)
- **Result Streaming**: Coordinator manages streaming with multiple format support
- **Distributed Ready**: Support for multi-sequencer execution

### Workflow Discovery

Workflows are discovered automatically using decorators:

```python
from coordinator_refactor import workflow

@workflow(
    name='my_workflow',
    description='My custom workflow',
    tags=['custom', 'example']
)
def my_workflow():
    return {
        'phases': [
            {'name': 'phase1', 'topology': 'backtest'},
            {'name': 'phase2', 'topology': 'analysis'}
        ]
    }
```

### Usage Example

```python
from coordinator_refactor import Coordinator

# Create coordinator
coordinator = Coordinator()

# Execute workflow
result = await coordinator.execute_workflow({
    'workflow': 'adaptive_ensemble',
    'symbols': ['SPY', 'QQQ'],
    'distributed_execution': {
        'enabled': True,
        'num_workers': 4
    }
})
```

### Composing Workflows

Use the WorkflowComposer to build complex workflows:

```python
from coordinator_refactor import WorkflowComposer

composer = WorkflowComposer()

# Walk-forward = repeated backtest
walk_forward = composer.repeat('backtest', times=12, config={
    'window_size': 180,
    'step_size': 30
})
```

## Migration from Old Architecture

See the migration guide in the parent directory for step-by-step instructions on migrating from the old mixed-responsibility architecture to this clean Protocol + Composition design.

## Benefits

1. **Single Responsibility**: Each component does ONE thing
2. **No Circular Dependencies**: Clean dependency graph
3. **Easy Testing**: Test each component in isolation
4. **Extensibility**: Easy to add new workflows and topologies
5. **Clear Execution Flow**: Predictable, debuggable flow
