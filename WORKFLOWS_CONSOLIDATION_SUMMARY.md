# Workflows Consolidation Summary

## Key Insight
The user correctly identified that we were creating duplicate factory functionality in `src/core/coordinator/workflows/` when `src/core/containers/factory.py` already provides comprehensive factory infrastructure.

## What Was Done

### 1. Created WorkflowManager Bridge
Created `src/core/coordinator/workflows/workflow_manager.py` that:
- Acts as a bridge between the Coordinator and the existing container factory
- Leverages `core.containers.factory` instead of duplicating functionality
- Provides both `WorkflowManager` and `ComposableWorkflowManagerPipeline` interfaces
- Handles workflow execution using the core factory's pattern composition

### 2. Updated Coordinator
Modified `src/core/coordinator/coordinator.py` to:
- Import WorkflowManager from `workflows.workflow_manager` instead of looking for a non-existent `manager.py`
- Use the new bridge that properly leverages the core factory

### 3. Updated Workflows Package
Modified `src/core/coordinator/workflows/__init__.py` to:
- Export the new WorkflowManager as the primary interface
- Import and re-export core factory functions (`get_global_factory`, `compose_pattern`, etc.)
- Mark duplicate container factories as deprecated but keep them for backward compatibility

## Benefits of This Approach

1. **No Duplication**: Uses existing `core.containers.factory` infrastructure
2. **Single Source of Truth**: Container patterns defined in one place
3. **Leverages Existing Features**: 
   - Pattern registry with pre-defined patterns (full_backtest, signal_generation, etc.)
   - Automatic feature inference from strategy configurations
   - Container composition and hierarchy building
4. **Clean Architecture**: Clear separation between:
   - Core factory infrastructure (containers/factory.py)
   - Workflow orchestration (coordinator/)
   - Container implementations (execution/, strategy/, etc.)

## Existing Factory Features We Can Now Use

From `core.containers.factory.py`:
- **ContainerRegistry**: Manages container types and capabilities
- **ContainerFactory**: Creates and composes containers
- **PatternManager**: Loads/saves patterns from YAML
- **Built-in Patterns**:
  - `full_backtest`: Complete backtest workflow
  - `signal_generation`: Signal generation only
  - `signal_replay`: Signal replay for ensemble optimization
  - `simple_backtest`: Simplified backtest with peer containers
- **Automatic Feature Inference**: Infers required indicators from strategy configurations

## Migration Path

For existing code using the duplicate factories:
1. Import from core: `from core.containers.factory import compose_pattern`
2. Use pattern-based composition: `container = compose_pattern('simple_backtest', config)`
3. The WorkflowManager handles this transparently for coordinator-based workflows

## Files That Can Be Removed (Future Cleanup)

Once migration is complete, these files contain duplicate functionality:
- `src/core/coordinator/workflows/factory.py` (the one we started creating)
- `src/core/coordinator/workflows/workflow_factory.py` 
- Eventually: `src/core/coordinator/workflows/container_factories.py` (871 lines of duplication)
- `src/core/coordinator/workflows/containers_pipeline.py` (thin wrapper, can be removed)

## Next Steps

1. Test the new WorkflowManager with existing workflows
2. Gradually migrate code to use core factory directly
3. Remove deprecated container factory files
4. Add any missing container types to the core registry