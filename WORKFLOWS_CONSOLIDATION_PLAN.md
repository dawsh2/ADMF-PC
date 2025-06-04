# Workflows Directory Consolidation Plan

## Current Structure Analysis

The workflows directory contains ~5200 lines of code with significant redundancy:

### Identified Redundancies

1. **Duplicate Backtest Implementations**
   - `workflows/backtest.py` (290 lines) - Full backtest workflow
   - `workflows/modes/backtest.py` (133 lines) - Backtest mode configuration
   - Both define `BacktestConfig` class with identical fields

2. **Duplicate Signal Replay Implementations**
   - `workflows/signal_replay.py` (325 lines) - Signal capture/replay components
   - `workflows/modes/signal_replay.py` (115 lines) - Signal replay mode
   - Different approaches to same functionality

3. **Container Factory Redundancy**
   - `container_factories.py` (871 lines) - Main factory functions
   - `containers_pipeline.py` (289 lines) - Pipeline wrapper that imports from container_factories
   - Essentially containers_pipeline just wraps container_factories

4. **Multi-Strategy Files**
   - `multi_strategy_config.py` (370 lines) - Configuration classes
   - `multi_strategy_aggregation.py` (381 lines) - Aggregation logic
   - `strategy_coordinator.py` (288 lines) - Coordination logic
   - Could be consolidated into a single multi_strategy module

## Consolidation Strategy

### 1. Merge Modes into Main Workflows
Combine the modes/ directory implementations with their main counterparts:

```
workflows/
├── backtest.py         # Merge backtest.py + modes/backtest.py
├── signal_generation.py # Keep modes/signal_generation.py (no duplicate)
├── signal_replay.py    # Merge signal_replay.py + modes/signal_replay.py
```

**Benefits:**
- Single source of truth for each workflow type
- Eliminates duplicate `BacktestConfig` definitions
- Reduces confusion about which implementation to use

### 2. Simplify Container Creation
Keep only `container_factories.py` and remove `containers_pipeline.py`:

```python
# In container_factories.py, add:
def create_container(role: ContainerRole, config: Dict[str, Any]) -> Container:
    """Universal container creation function."""
    factory = get_container_factory(role)
    if factory:
        return factory(config)
    raise ValueError(f"No factory for role: {role}")
```

**Benefits:**
- Removes unnecessary wrapper layer
- Single entry point for container creation
- Maintains composability

### 3. Consolidate Multi-Strategy Components
Merge into a single `multi_strategy.py` module:

```python
# multi_strategy.py - contains:
# - Configuration classes from multi_strategy_config.py
# - Aggregation logic from multi_strategy_aggregation.py  
# - Coordinator from strategy_coordinator.py
```

**Benefits:**
- Related functionality in one place
- Easier to understand multi-strategy flow
- Reduces file count by 2

### 4. Create Composable Workflow Patterns
Instead of separate workflow classes, use composable patterns:

```python
# workflow_patterns.py
class WorkflowPatterns:
    @staticmethod
    def backtest_pattern(config) -> List[Container]:
        """Standard backtest: Data → Indicator → Strategy → Risk → Execution"""
        return [
            create_container(ContainerRole.DATA, config.data),
            create_container(ContainerRole.INDICATOR, config.indicator),
            create_container(ContainerRole.STRATEGY, config.strategy),
            create_container(ContainerRole.RISK, config.risk),
            create_container(ContainerRole.EXECUTION, config.execution),
        ]
    
    @staticmethod
    def signal_generation_pattern(config) -> List[Container]:
        """Signal generation: Data → Indicator → Strategy → SignalCapture"""
        return [
            create_container(ContainerRole.DATA, config.data),
            create_container(ContainerRole.INDICATOR, config.indicator),
            create_container(ContainerRole.STRATEGY, config.strategy),
            create_container(ContainerRole.SIGNAL_CAPTURE, config.capture),
        ]
    
    @staticmethod
    def signal_replay_pattern(config) -> List[Container]:
        """Signal replay: SignalReplay → Risk → Execution"""
        return [
            create_container(ContainerRole.SIGNAL_REPLAY, config.replay),
            create_container(ContainerRole.RISK, config.risk),
            create_container(ContainerRole.EXECUTION, config.execution),
        ]
```

**Benefits:**
- Reusable patterns
- Easy to compose custom workflows
- Clear separation of concerns

## Implementation Steps

1. **Merge duplicate files**
   - Combine backtest implementations
   - Combine signal replay implementations
   - Remove containers_pipeline.py

2. **Consolidate multi-strategy files**
   - Create single multi_strategy.py
   - Migrate all functionality
   - Update imports

3. **Create workflow_patterns.py**
   - Define standard patterns
   - Make patterns composable
   - Document usage

4. **Update imports**
   - Fix all import statements
   - Update __init__.py exports
   - Test all workflows

## Expected Results

### Before: 15 files, ~5200 lines
```
workflows/
├── __init__.py                    (63)
├── backtest.py                   (290)
├── container_factories.py        (871)
├── containers_pipeline.py        (289)
├── feature_hub_workflow.py       (584)
├── multi_strategy_aggregation.py (381)
├── multi_strategy_config.py      (370)
├── optimization_workflows.py     (930)
├── signal_replay.py              (325)
├── strategy_coordinator.py       (288)
├── walk_forward_workflow.py      (496)
└── modes/
    ├── __init__.py                (19)
    ├── backtest.py               (133)
    ├── signal_generation.py       (80)
    └── signal_replay.py          (115)
```

### After: 8 files, ~3500 lines (32% reduction)
```
workflows/
├── __init__.py                    
├── backtest.py                   (~350 lines - merged)
├── container_factories.py        (~900 lines - enhanced)
├── multi_strategy.py             (~800 lines - consolidated)
├── optimization_workflows.py     (930 lines - kept)
├── signal_generation.py          (80 lines - from modes/)
├── signal_replay.py              (~400 lines - merged)
├── walk_forward_workflow.py      (496 lines - kept)
└── workflow_patterns.py          (~150 lines - new)
```

## Benefits Summary

1. **32% code reduction** through eliminating duplication
2. **Clearer structure** with one implementation per workflow type
3. **Better composability** through workflow patterns
4. **Easier maintenance** with consolidated multi-strategy logic
5. **Consistent APIs** across all workflow types