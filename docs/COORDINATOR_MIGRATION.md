# Enhanced Coordinator Migration Guide

This guide helps you migrate from the traditional Coordinator to the Enhanced Coordinator with composable container support.

## Overview

The Enhanced Coordinator provides **backward compatibility** while adding powerful new composable container capabilities. You can:

1. **Keep existing code unchanged** (traditional mode)
2. **Gradually migrate** to composable containers 
3. **Use hybrid approaches** (mix traditional + composable)

## Quick Start

### Before (Traditional Coordinator)
```python
from src.core.coordinator.coordinator import Coordinator
from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType

coordinator = Coordinator()
config = WorkflowConfig(workflow_type=WorkflowType.BACKTEST, ...)
result = await coordinator.execute_workflow(config)
```

### After (Enhanced Coordinator - Backward Compatible)
```python
from src.core.coordinator.enhanced_coordinator import EnhancedCoordinator
from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType

# Drop-in replacement
coordinator = EnhancedCoordinator(enable_composable_containers=False)
config = WorkflowConfig(workflow_type=WorkflowType.BACKTEST, ...)
result = await coordinator.execute_workflow(config)  # Same API!
```

### After (Enhanced Coordinator - With Composable Containers)
```python
from src.core.coordinator.enhanced_coordinator import EnhancedCoordinator

# Enable composable containers
coordinator = EnhancedCoordinator(enable_composable_containers=True)
config = WorkflowConfig(
    workflow_type=WorkflowType.BACKTEST,
    parameters={'container_pattern': 'full_backtest'},  # NEW!
    ...
)
result = await coordinator.execute_workflow(config, execution_mode="composable")
```

## Migration Strategies

### 1. **No Migration Required** (Traditional Mode)

Keep your existing code exactly as-is:

```python
# This continues to work unchanged
coordinator = EnhancedCoordinator(enable_composable_containers=False)
```

**Benefits**: Zero code changes, same behavior
**When to use**: You're satisfied with current functionality

### 2. **Gradual Migration** (Auto Mode)

Let the coordinator choose the best execution mode:

```python
coordinator = EnhancedCoordinator(enable_composable_containers=True)

# Coordinator automatically chooses traditional vs composable
result = await coordinator.execute_workflow(config, execution_mode="auto")
```

**Benefits**: Automatic optimization, no config changes needed
**When to use**: You want better performance without changing configs

### 3. **Selective Migration** (Hybrid Mode)

Use composable containers for specific workflows:

```python
coordinator = EnhancedCoordinator(enable_composable_containers=True)

# Simple backtests: use traditional
if is_simple_backtest(config):
    result = await coordinator.execute_workflow(config, execution_mode="traditional")
else:
    # Complex optimization: use composable
    config.parameters['container_pattern'] = 'full_backtest'
    result = await coordinator.execute_workflow(config, execution_mode="composable")
```

**Benefits**: Best of both worlds, gradual transition
**When to use**: You want to test composable containers on specific use cases

### 4. **Full Migration** (Composable Mode)

Migrate everything to composable containers:

```python
coordinator = EnhancedCoordinator(enable_composable_containers=True)

# All workflows use composable containers
result = await coordinator.execute_workflow(config, execution_mode="composable")
```

**Benefits**: Maximum flexibility and performance
**When to use**: You want to leverage all new capabilities

## Container Pattern Migration

### Simple Backtest
```python
# Before: Basic backtest with WorkflowConfig
config = WorkflowConfig(
    workflow_type=WorkflowType.BACKTEST,
    backtest_config={'strategy': 'momentum', 'initial_capital': 100000}
)

# After: Use simple_backtest pattern
config = WorkflowConfig(
    workflow_type=WorkflowType.BACKTEST,
    parameters={'container_pattern': 'simple_backtest'},
    optimization_config={
        'strategies': [{'type': 'momentum', 'parameters': {...}}]
    },
    backtest_config={'initial_capital': 100000}
)
```

### Full Backtest with Classifiers
```python
# Before: Complex config in optimization_config
config = WorkflowConfig(
    workflow_type=WorkflowType.OPTIMIZATION,
    optimization_config={
        'classifiers': [...],
        'risk_profiles': [...],
        'strategies': [...]
    }
)

# After: Use full_backtest pattern (same config!)
config = WorkflowConfig(
    workflow_type=WorkflowType.BACKTEST,  # Can use BACKTEST instead of OPTIMIZATION
    parameters={'container_pattern': 'full_backtest'},
    optimization_config={
        'classifiers': [...],  # Same structure
        'risk_profiles': [...],  # Same structure  
        'strategies': [...]  # Same structure
    }
)
```

### Signal Generation
```python
# Before: Not easily supported
# After: Built-in pattern
config = WorkflowConfig(
    workflow_type=WorkflowType.ANALYSIS,
    parameters={'container_pattern': 'signal_generation'},
    analysis_config={
        'mode': 'signal_generation',
        'output_path': './signals/'
    }
)
```

## Benefits of Migration

### Performance Benefits
- **Single data pass**: Indicators computed once, shared across all strategies
- **Parallel execution**: Multiple strategies/containers run simultaneously  
- **Memory efficiency**: Streaming results, configurable memory limits
- **Faster optimization**: Signal replay patterns 10-100x faster than recomputation

### Flexibility Benefits
- **Pattern library**: Pre-built patterns for common use cases
- **Custom patterns**: Define your own container arrangements
- **Easy scaling**: Add/remove container levels without code changes
- **Protocol-based**: Zero inheritance, clean testing

### Operational Benefits
- **Reproducible results**: Identical container creation every time
- **Better debugging**: Clear container hierarchy and event flow
- **Resource management**: Per-container limits and monitoring
- **Clean isolation**: No state leakage between backtests

## Migration Checklist

### ✅ Phase 1: Setup
- [ ] Update imports to use `EnhancedCoordinator`
- [ ] Enable composable containers: `enable_composable_containers=True`
- [ ] Test existing workflows in "auto" mode
- [ ] Verify results match traditional coordinator

### ✅ Phase 2: Simple Migration
- [ ] Migrate simple backtests to `simple_backtest` pattern
- [ ] Migrate complex backtests to `full_backtest` pattern
- [ ] Add container patterns to workflow configs
- [ ] Test performance improvements

### ✅ Phase 3: Advanced Features
- [ ] Implement signal generation workflows
- [ ] Use signal replay for optimization
- [ ] Create custom container patterns
- [ ] Leverage parallel execution for large optimizations

### ✅ Phase 4: Optimization
- [ ] Tune container resource limits
- [ ] Optimize indicator computation
- [ ] Implement streaming results
- [ ] Monitor memory usage and performance

## Common Migration Issues

### Import Errors
```python
# Issue: Missing composable container imports
# Solution: Ensure all container types are registered
from src.execution.containers import register_execution_containers
register_execution_containers()
```

### Configuration Mismatch
```python
# Issue: Traditional config doesn't map to container pattern
# Solution: Use configuration mapping helpers
from src.core.coordinator.enhanced_coordinator import execute_backtest_workflow

result = await execute_backtest_workflow(
    config=traditional_config,
    container_pattern="full_backtest"
)
```

### Performance Regressions
```python
# Issue: Composable containers slower than expected
# Solution: Check pattern selection and resource limits
config.parameters.update({
    'container_pattern': 'simple_backtest',  # Use simpler pattern
    'max_indicators': 50,  # Limit indicator computation
    'max_memory_mb': 1024  # Set resource limits
})
```

## Examples

See `src/core/coordinator/integration_example.py` for complete working examples of:

- Traditional vs Composable execution
- Different container patterns  
- Custom pattern creation
- Multi-pattern workflows
- Auto mode selection

## Getting Help

1. **Start with auto mode**: Let the coordinator choose execution mode
2. **Use simple patterns first**: Begin with `simple_backtest` and `full_backtest`
3. **Check the examples**: `integration_example.py` covers most use cases
4. **Monitor performance**: Compare execution times and memory usage
5. **Test incrementally**: Migrate one workflow type at a time

The Enhanced Coordinator is designed to make migration seamless while unlocking powerful new capabilities for complex trading system development.