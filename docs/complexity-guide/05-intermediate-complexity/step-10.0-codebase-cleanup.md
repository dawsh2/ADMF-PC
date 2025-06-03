# Step 10.0: Critical Codebase Cleanup

## 🚨 CRITICAL: Complete This Before Any Other Work

The codebase currently contains **~50+ duplicate implementations** that violate core ADMF-PC principles. This technical debt is severely limiting development velocity and must be addressed before proceeding with any other features.

## 🎯 Objectives

1. Remove ALL inheritance-based implementations
2. Consolidate to ONE canonical implementation of each concept
3. Enforce Protocol + Composition architecture
4. Delete all legacy and duplicate files
5. Establish clear architectural boundaries

## 📋 Current Violations

### Inheritance Violations (NEVER use inheritance!)
- `BaseComposableContainer` with 20+ inheriting classes
- `BaseClassifier` with multiple inheriting classifiers
- 17 files using ABC (Abstract Base Classes)
- Multiple Base* classes throughout codebase

### Duplicate Implementations
- **4 container implementation files** (containers.py, containers_fixed.py, containers_nested.py, containers_pipeline.py)
- **3 workflow managers** (composable_workflow_manager*.py)
- **3 backtest brokers** (backtest_broker.py, improved_*, refactored_*)
- **4 execution engines** (execution_engine.py, improved_*, backtest_*, simple_*)
- **3 order managers** (order_manager.py, improved_*, step2_*)
- **4 portfolio implementations** (portfolio_state.py, risk_portfolio.py, improved_*, step2_*)
- **6 factory implementations** scattered across modules

### Naming Violations
- Files prefixed with `improved_`, `step2_`, `_fixed`, `_refactored`
- Legacy directory `data_OLD` still present
- Multiple versions of the same concept

## 📊 Additional Findings from Previous Analysis

### Container System Migration Status
- **UniversalScopedContainer** still used in **26 files** (major migration needed)
- Strategy module, risk management, and execution engines still on old system
- Three different container architectures coexisting

### Type System Chaos
- **5 different type systems** trying to solve pydantic dependency issues:
  - `simple_types.py` (dataclass-based, no dependencies)
  - `types.py` (pydantic wrapper)
  - `types_no_pydantic.py` (compatibility layer)
  - `minimal_types.py` (duck typing)
  - `core/types.py` (domain enums)

### Naming Anti-Patterns
- Files with adjective prefixes indicating non-canonical implementations:
  - `enhanced_container.py`
  - `minimal_bootstrap.py`
  - `simple_types.py`
  - `composable_workflow_manager.py`

## 🧹 Cleanup Plan by Module

### Phase 1: Core Infrastructure (Week 1)

#### 1. Type System Consolidation (FIRST PRIORITY)
```bash
# KEEP ONLY:
src/core/coordinator/simple_types.py  # Rename to types.py after migration
src/core/types.py  # Domain enums (merge into above)

# DELETE:
src/core/coordinator/types.py  # Pydantic wrapper
src/core/coordinator/types_no_pydantic.py  # Compatibility layer
src/core/minimal_types.py  # Duck typing

# MIGRATION:
- Update ALL imports to use simple_types
- Merge domain enums from core/types.py
- Remove pydantic dependencies completely
```

#### 2. Container System (MAJOR MIGRATION - 26 FILES)
```bash
# CURRENT STATE:
- UniversalScopedContainer: Used in 26 files
- BaseComposableContainer: Used in execution containers
- Enhanced container: Used in 1 file

# TARGET STATE:
src/core/containers/protocols.py  # ALL container protocols
src/core/containers/container.py  # ONE canonical implementation

# DELETE AFTER MIGRATION:
src/core/containers/universal.py  # 26 files need migration!
src/core/containers/enhanced_container.py
src/core/containers/composable.py  # BaseComposableContainer
src/core/containers/minimal_bootstrap.py
src/execution/containers.py
src/execution/containers_fixed.py
src/execution/containers_nested.py
```

**Container Migration Steps:**
1. Create comprehensive test suite for all 26 files using UniversalScopedContainer
2. Define complete container protocol in protocols.py
3. Implement ONE canonical container using composition
4. Migrate files in batches:
   - Batch 1: Data loaders and streamers
   - Batch 2: Strategy components
   - Batch 3: Risk management
   - Batch 4: Execution engines
5. Update execution/containers_pipeline.py to use protocols
6. Delete all old container implementations

#### 3. Event System
```bash
# KEEP ONLY:
src/core/events/event_bus.py
src/core/events/types.py

# INTEGRATE AS CAPABILITIES:
src/core/events/tracing/*  # Add tracing as event bus capability, not separate class
```

#### 4. Coordinator/Workflow (Remove Adjective Prefixes)
```bash
# KEEP ONLY:
src/core/coordinator/coordinator.py
src/core/coordinator/workflow_manager.py  # Merge both composable versions

# DELETE:
src/core/coordinator/composable_workflow_manager.py
src/core/coordinator/composable_workflow_manager_pipeline.py
src/core/coordinator/composable_workflow_manager_nested.py
```

### Phase 2: Execution Module (Week 2)

#### 1. Brokers
```bash
# DETERMINE BEST, KEEP ONE:
# Review and pick most complete implementation

# DELETE OTHERS:
src/execution/backtest_broker.py
src/execution/backtest_broker_refactored.py
src/execution/improved_backtest_broker.py
```

#### 2. Engines
```bash
# KEEP:
src/execution/backtest_engine.py  # If most complete

# DELETE:
src/execution/execution_engine.py
src/execution/improved_execution_engine.py
src/execution/simple_backtest_engine.py
```

#### 3. Order Management
```bash
# KEEP ONE:
# Review and pick most complete

# DELETE:
src/execution/order_manager.py
src/execution/improved_order_manager.py
src/risk/step2_order_manager.py
```

### Phase 3: Risk Module (Week 3)

#### 1. Portfolio State
```bash
# ANALYZE AND KEEP ONE:
src/risk/portfolio_state.py

# DELETE:
src/risk/risk_portfolio.py
src/risk/improved_risk_portfolio.py
src/risk/step2_portfolio_state.py
```

#### 2. Remove ALL step2_ files
```bash
# DELETE ALL:
src/risk/step2_*
```

### Phase 4: Strategy Module (Week 4)

#### 1. Classifier System
```bash
# CONVERT TO PROTOCOLS:
src/strategy/classifiers/classifier.py  # Remove ABC, use Protocol

# UPDATE ALL CLASSIFIERS:
# Remove inheritance, use composition
```

#### 2. Container Consolidation
```bash
# KEEP:
src/strategy/classifiers/classifier_container.py

# DELETE:
src/strategy/classifiers/enhanced_classifier_container.py
```

### Phase 5: Factory Consolidation (Week 5)

```bash
# CREATE ONE CANONICAL:
src/core/containers/factory.py  # Using protocols only

# DELETE ALL OTHERS:
src/core/containers/backtest/factory.py
src/execution/backtest_container_factory.py
src/execution/backtest_container_factory_traced.py
src/risk/step2_container_factory.py
src/execution/execution_module_factory.py
```

### Phase 6: Final Cleanup (Week 6)

1. Remove `data_OLD` directory entirely
2. Remove all ABC imports and usage
3. Update all imports throughout codebase
4. Run comprehensive tests
5. Update documentation

## 📋 Files Requiring Migration from UniversalScopedContainer

### Strategy Module
- [ ] `src/strategy/components/indicator_hub.py`
- [ ] `src/strategy/components/indicators.py`
- [ ] `src/strategy/strategies/*.py` (all strategy files)
- [ ] `src/strategy/optimization/*.py` (optimization components)

### Risk Module
- [ ] `src/risk/risk_container.py`
- [ ] `src/risk/portfolio_state.py`
- [ ] `src/risk/position_sizing.py`
- [ ] `src/risk/risk_limits.py`

### Execution Module
- [ ] `src/execution/backtest_engine.py`
- [ ] `src/execution/execution_engine.py`
- [ ] `src/execution/order_manager.py`
- [ ] `src/execution/market_simulation.py`

### Other Modules
- [ ] `src/data/handlers.py`
- [ ] `src/data/loaders.py`
- [ ] Various test files
- [ ] Documentation and examples

## ✅ Implementation Checklist

### Week 1: Core Infrastructure
- [ ] Consolidate 5 type systems into 1
- [ ] Create canonical container protocol
- [ ] Start migration of 26 files from UniversalScopedContainer
- [ ] Remove adjective prefixes from filenames
- [ ] Merge workflow managers into one

### Week 2: Execution Module
- [ ] Consolidate to one broker implementation
- [ ] Consolidate to one engine implementation
- [ ] Clean up order management
- [ ] Remove all "improved_" files

### Week 3: Risk Module
- [ ] Consolidate portfolio implementations
- [ ] Remove all "step2_" files
- [ ] Update to protocol-based design

### Week 4: Strategy Module
- [ ] Convert classifiers to protocols
- [ ] Remove BaseClassifier
- [ ] Consolidate container implementations

### Week 5: Factories and Integration
- [ ] Create one canonical factory
- [ ] Remove all duplicate factories
- [ ] Update all factory usage

### Week 6: Final Pass
- [ ] Remove data_OLD directory
- [ ] Remove all ABC usage
- [ ] Update all imports
- [ ] Run full test suite
- [ ] Update documentation

## 🧪 Testing Strategy

### 1. Create Baseline Tests
Before ANY deletions, create tests that verify current functionality:
```python
# tests/cleanup/test_baseline_functionality.py
# Capture current behavior before refactoring
```

### 2. Incremental Testing
After each module cleanup:
- Run baseline tests
- Verify no functionality lost
- Update tests for new structure

### 3. Integration Testing
After each phase:
- Run full integration tests
- Verify cross-module communication
- Check performance metrics

## 🎯 Success Criteria

### Code Quality Metrics
- [ ] ZERO inheritance (no class X(Y) except Protocol)
- [ ] ONE implementation per concept
- [ ] NO files with version prefixes (improved_, step2_, etc.)
- [ ] NO Abstract Base Classes (ABC)
- [ ] ALL containers use protocols

### Functionality Preservation
- [ ] All existing tests pass
- [ ] No performance regression
- [ ] All workflows execute correctly
- [ ] Event flow unchanged

### Architecture Compliance
- [ ] 100% Protocol + Composition
- [ ] Complete container isolation
- [ ] Event-driven communication only
- [ ] Configuration-driven behavior

## 🚨 Common Pitfalls

### 1. Premature Deletion
**Problem**: Deleting files before understanding dependencies
**Solution**: Map all dependencies first, create tests, then delete

### 2. Feature Loss
**Problem**: Losing functionality when consolidating
**Solution**: Document all features in each implementation before merging

### 3. Import Breakage
**Problem**: Breaking imports throughout codebase
**Solution**: Use automated refactoring tools, comprehensive grep

### 4. Hidden Dependencies
**Problem**: Code relying on inheritance behavior
**Solution**: Convert to explicit composition before removing base classes

## 📊 Tracking Progress

### Metrics to Track
1. Number of duplicate files remaining
2. Inheritance usage count
3. ABC import count
4. Test coverage
5. Performance benchmarks

### Daily Checklist
- [ ] Document files to be removed
- [ ] Create/update tests
- [ ] Perform refactoring
- [ ] Run test suite
- [ ] Update imports
- [ ] Commit with clear message

## 🎯 End State

After completion:
- **ONE** implementation of each concept
- **ZERO** inheritance (except Protocol)
- **ZERO** duplicate files
- **100%** protocol-based architecture
- **Clean** module boundaries

## 📚 Resources

- [Protocol + Composition Guide](../../architecture/03-PROTOCOL-COMPOSITION.md)
- [Container Architecture](../../architecture/02-CONTAINER-HIERARCHY.md)
- [ADMF-PC Principles](../../CLAUDE.md)
- [Style Guide](../../standards/STYLE-GUIDE.md)

## 🚀 Next Steps

Once cleanup is complete:
1. Implement event tracing on clean codebase
2. Add multi-portfolio support
3. Continue with complexity guide steps

**Remember**: This cleanup is not optional - it's critical for the project's future maintainability and development velocity.