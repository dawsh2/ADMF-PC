# STANDARD: Pattern-Based Architecture

## Official Architecture Standard for ADMF-PC

**Status**: MANDATORY  
**Version**: 1.0  
**Date**: January 2025

## Core Principle

**Each workflow pattern is defined in exactly ONE place, with clear separation of responsibilities between factories and orchestrators.**

## Architecture Components

### 1. Container Factory (`core/containers/factory.py`)

**Responsibility**: Container creation and hierarchy ONLY

```python
# ✅ CORRECT: Container structure only
simple_backtest = ContainerPattern(
    name="simple_backtest",
    structure={
        "root": {"role": "backtest", "children": {...}}
    },
    required_capabilities={"data.historical", "execution.backtest"},
    default_config={...}
)

# ❌ WRONG: Don't include communication config here
simple_backtest = ContainerPattern(
    name="simple_backtest", 
    structure={...},
    communication_config={...}  # NO! This belongs in WorkflowManager
)
```

**What it provides**:
- Container patterns (structure definitions)
- Container creation (`compose_pattern()`)
- Container registry
- Automatic feature inference

**What it does NOT provide**:
- Communication setup
- Event routing
- Adapter configuration

### 2. Communication Factory (`core/communication/factory.py`)

**Responsibility**: Communication adapters ONLY

```python
# ✅ CORRECT: Creates adapters from configuration
adapter = factory.create_adapter('pipeline', {
    'type': 'pipeline',
    'containers': ['data', 'strategy', 'risk'],
    'event_flow': [...]
})

# ❌ WRONG: Don't create containers here
adapter = factory.create_pipeline_with_containers({
    'containers_to_create': [...]  # NO! Use container factory
})
```

**What it provides**:
- Adapter creation (`create_adapter()`)
- Adapter registry
- Network validation
- Standard adapter types (pipeline, broadcast, hierarchical, selective)

**What it does NOT provide**:
- Container creation
- Workflow orchestration
- Pattern definitions

### 3. Workflow Manager (`coordinator/workflows/workflow_manager.py`)

**Responsibility**: Orchestration according to named patterns

```python
# ✅ CORRECT: Single pattern definition
self._workflow_patterns = {
    'simple_backtest': {
        'description': 'Simple backtest workflow',
        'container_pattern': 'simple_backtest',    # → Container Factory
        'communication_config': [                  # → Communication Factory
            {
                'name': 'backtest_pipeline',
                'type': 'pipeline',
                'event_flow': [...]
            }
        ]
    }
}
```

**What it provides**:
- Workflow pattern definitions (SINGLE SOURCE OF TRUTH)
- Orchestration (coordinates container + communication factories)
- Standard communication configs for each pattern
- Workflow execution lifecycle

**What it does NOT provide**:
- Direct container creation (delegates to container factory)
- Direct adapter creation (delegates to communication factory)

## Standard Workflow Patterns

### Pattern: `simple_backtest`
- **Container Pattern**: `simple_backtest` (from container factory)
- **Communication**: Linear pipeline
- **Event Flow**: Data → Features → Strategy → Risk → Execution → Portfolio
- **Use Case**: Basic single-strategy backtesting

### Pattern: `full_backtest`
- **Container Pattern**: `full_backtest` (from container factory)  
- **Communication**: Hierarchical with feedback loops
- **Event Flow**: Includes classifier, complex routing
- **Use Case**: Multi-strategy with regime detection

### Pattern: `signal_generation`
- **Container Pattern**: `signal_generation` (from container factory)
- **Communication**: Linear pipeline to signal capture
- **Event Flow**: Data → Features → Strategy → SignalCapture
- **Use Case**: Signal generation for later replay

### Pattern: `signal_replay`
- **Container Pattern**: `signal_replay` (from container factory)
- **Communication**: Replay pipeline
- **Event Flow**: SignalReplay → Risk → Execution → Portfolio
- **Use Case**: Ensemble optimization

## Implementation Rules

### ✅ DO:

1. **Define each pattern once** in WorkflowManager
2. **Delegate container creation** to container factory
3. **Delegate communication setup** to communication factory
4. **Use composition** over inheritance
5. **Follow single responsibility** principle

### ❌ DON'T:

1. **Mix container creation with communication** setup
2. **Duplicate pattern definitions** across multiple files
3. **Create "enhanced" versions** of existing patterns
4. **Embed communication config** in container patterns
5. **Create containers** in communication factory

## Migration Path

### For Existing Code:

1. **Identify duplicate patterns** across files
2. **Extract pattern definitions** to WorkflowManager
3. **Update imports** to use standard factories
4. **Remove duplicate factory files**

### For New Patterns:

1. **Add container structure** to container factory (if needed)
2. **Add communication config** to WorkflowManager
3. **Register pattern** in WorkflowManager._workflow_patterns
4. **Test integration** end-to-end

## Enforcement

### Code Review Checklist:

- [ ] No communication config in container patterns
- [ ] No container creation in communication factory
- [ ] Pattern defined once in WorkflowManager
- [ ] Uses delegation to appropriate factories
- [ ] Follows naming conventions (no "enhanced_" prefixes)

### Automated Checks:

```python
# Add to pre-commit hooks
def validate_architecture():
    # Check container patterns don't have communication config
    # Check communication factory doesn't create containers
    # Check no duplicate pattern definitions
    pass
```

## Examples

### ✅ CORRECT Usage:

```python
# In coordinator
workflow_manager = WorkflowManager()
result = await workflow_manager.execute(config)

# Internally:
# 1. Determines pattern: "simple_backtest"
# 2. Creates containers via: container_factory.compose_pattern("simple_backtest")
# 3. Sets up communication via: adapter_factory.create_adapters_from_config(comm_config)
# 4. Orchestrates execution
```

### ❌ INCORRECT Usage:

```python
# DON'T mix responsibilities
container = ContainerFactory.create_with_communication(...)  # NO!
adapter = CommunicationFactory.create_with_containers(...)   # NO!

# DON'T duplicate patterns
backtest_factory.create_backtest(...)      # Pattern defined here
workflow_manager.create_backtest(...)      # AND here - NO!
```

## Benefits

1. **Single Source of Truth**: Each pattern defined once
2. **Clear Separation**: Each component has one responsibility
3. **Testability**: Can test container creation and communication independently
4. **Maintainability**: Changes to patterns happen in one place
5. **Extensibility**: Easy to add new patterns or modify existing ones
6. **Reusability**: Same containers can be used with different communication patterns

## Compliance

**This is the mandatory architecture standard for ADMF-PC.** All new code must follow this pattern, and existing code should be migrated to comply.

**Any deviations must be explicitly approved and documented with rationale.**