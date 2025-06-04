# Namespace Consolidation Guide

## Overview
This guide addresses the systematic cleanup of duplicate/conflicting implementations that arise during the Protocol + Composition refactoring process.

## Problem
During the migration from inheritance to Protocol + Composition, we end up with:
- Multiple implementations of the same functionality with "smelly" names (`*_refactored`, `*_enhanced`, `*_pipeline`)
- Namespace conflicts (multiple `ContainerRole` enums)
- Canonical vs non-canonical file confusion
- Risk of deleting newly established P+C implementations

## Approach: Systematic Consolidation

### Phase 1: Identify Canonical Files (CURRENT)
**Goal**: Determine which files are truly canonical by analyzing runtime usage

**Method**: Use the canonical run script to trace what's actually being used:
```bash
python main.py --config config/multi_strategy_test.yaml --bars 50
```

**Key Insights**:
- Runtime logs show which files are imported and used
- Error messages reveal missing dependencies and namespace conflicts
- Import traces show the actual dependency graph

### Phase 2: Map Namespace Conflicts
**Goal**: Create a mapping of conflicting implementations

**Current Conflicts Identified**:
1. **ContainerRole Enum Conflict**:
   - `src/core/containers/composable.py:ContainerRole` (composition engine)
   - `src/core/containers/container.py:ContainerRole` (P+C factories)
   - **Solution**: Consolidate into single canonical enum

2. **Workflow Manager Conflict**:
   - `composable_workflow_manager.py` (54KB, main)
   - `composable_workflow_manager_pipeline.py` (23KB, actively used!)
   - `composable_workflow_manager_nested.py` (14KB, unused)

3. **Container Implementation Conflict**:
   - `containers_pipeline.py` (11KB, P+C refactored - CANONICAL)
   - `containers_refactored.py` (7KB, experimental)

4. **Factory Conflict**:
   - `container_factories.py` (24KB, P+C implementation - CANONICAL)
   - `container_factories_pipeline.py` (95KB, experimental variant)

### Phase 3: Consolidation Strategy

#### Rule 1: Preserve P+C Implementations
**NEVER DELETE**:
- `container_factories.py` (our P+C foundation)
- `containers_pipeline.py` (our P+C container system)
- `new_backtest_container_factory.py` (working P+C example)

#### Rule 2: Consolidate Namespaces Before Deletion
**Process**:
1. Identify which enum/class is imported by more files
2. Migrate all references to the most-used version
3. Delete the duplicate
4. Test canonical run script

#### Rule 3: Use Runtime Validation
**Before any deletion**:
```bash
# Test that canonical functionality still works
python main.py --config config/multi_strategy_test.yaml --bars 50

# Test P+C factory system
python -c "from src.execution.new_backtest_container_factory import create_simple_momentum_backtest; create_simple_momentum_backtest()"
```

## Current Status

### ✅ Established P+C Foundation:
- `container_factories.py` - 11 factory functions
- `containers_pipeline.py` - composition-based containers  
- `new_backtest_container_factory.py` - working example
- All inheritance violations identified and mapped

### 🔧 Active Namespace Conflicts:
1. **ContainerRole enum conflict** - BEING RESOLVED
   - Consolidated enum in `composable.py`
   - Updated imports to use single source

### 🚫 Files Safe to Delete (After Consolidation):
- `composable_workflow_manager_nested.py` (unused variant)
- `containers_refactored.py` (experimental, superseded by P+C version)
- `container_factories_pipeline.py` (experimental variant)
- `backtest_container_factory_traced.py` (debugging variant)

## Next Steps

1. **Complete ContainerRole consolidation** 
2. **Test canonical run script passes**
3. **Systematically remove non-canonical files**
4. **Continue with inheritance elimination**

## Warning Signs to Stop
- Canonical run script fails
- P+C factory functions break
- Import errors in container system
- Test failures in Protocol + Composition code

**Golden Rule**: If in doubt, don't delete. Better to have duplicates than break the P+C foundation.