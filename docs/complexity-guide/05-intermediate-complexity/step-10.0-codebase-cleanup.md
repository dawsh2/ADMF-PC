# Step 10.0: Critical Codebase Cleanup

## 🚨 CRITICAL: Complete This Before Any Other Work

The codebase currently contains **~50+ duplicate implementations** that violate core ADMF-PC principles. This technical debt is severely limiting development velocity and must be addressed before proceeding with any other features.

## 🎯 Objectives

1. Remove ALL inheritance-based implementations
2. Consolidate to ONE canonical implementation of each concept
3. Enforce Protocol + Composition architecture
4. Delete all legacy and duplicate files
5. Establish clear architectural boundaries

## 📋 Detailed Violations Analysis (50 Total Files)

### Critical Multiple Implementations (17 files - HIGHEST PRIORITY)

#### Backtest Brokers (3 implementations)
- ✅ `src/execution/backtest_broker.py` ← **KEEP as canonical**
- ❌ `src/execution/backtest_broker_refactored.py` ← DELETE after merging features
- ❌ `src/execution/improved_backtest_broker.py` ← DELETE after merging features

#### Execution Engines (3 implementations)
- ✅ `src/execution/execution_engine.py` ← **KEEP as canonical**
- ❌ `src/execution/improved_execution_engine.py` ← DELETE after merging features
- ❌ `src/execution/simple_backtest_engine.py` ← DELETE after merging features

#### Order Managers (2 implementations)
- ✅ `src/execution/order_manager.py` ← **KEEP as canonical**
- ❌ `src/execution/improved_order_manager.py` ← DELETE after merging features
- ❌ `src/risk/step2_order_manager.py` ← DELETE after merging features

#### Market Simulation (2 implementations)
- ✅ `src/execution/market_simulation.py` ← **KEEP as canonical**
- ❌ `src/execution/improved_market_simulation.py` ← DELETE after merging features

#### Workflow Managers (3 implementations)
- ✅ `src/core/coordinator/composable_workflow_manager.py` ← **KEEP as canonical**
- ❌ `src/core/coordinator/composable_workflow_manager_nested.py` ← MERGE into canonical
- ❌ `src/core/coordinator/composable_workflow_manager_pipeline.py` ← MERGE into canonical

#### Container Factories (4 implementations)
- ✅ `src/execution/backtest_container_factory.py` ← **KEEP as canonical**
- ❌ `src/execution/backtest_container_factory_traced.py` ← DELETE, add tracing as capability
- ❌ `src/execution/new_backtest_container_factory.py` ← DELETE after reviewing features
- ❌ `src/risk/step2_container_factory.py` ← DELETE after merging features

### Adjective Prefix Violations (15 files)

#### execution/ module violations
- ❌ `src/execution/improved_backtest_broker.py` (covered above)
- ❌ `src/execution/improved_dependency_injection.py` ← DELETE, merge into canonical
- ❌ `src/execution/improved_execution_engine.py` (covered above)
- ❌ `src/execution/improved_market_simulation.py` (covered above)
- ❌ `src/execution/improved_order_manager.py` (covered above)

#### risk/ module violations
- ❌ `src/risk/improved_capabilities.py` ← DELETE, merge into `capabilities.py`
- ❌ `src/risk/improved_risk_portfolio.py` ← DELETE, merge into `portfolio_state.py`
- ❌ `src/risk/optimized_signal_flow.py` ← DELETE, merge into `signal_flow.py`

#### strategy/ module violations
- ❌ `src/strategy/enhanced_strategy_container.py` ← DELETE, merge capabilities into canonical

#### core/ module violations
- ❌ `src/core/config/simple_validator.py` ← DELETE, merge into `schema_validator.py`
- ❌ `src/core/containers/minimal_bootstrap.py` ← DELETE, merge into `bootstrap.py`
- ❌ `src/core/logging/simple_test.py` ← DELETE (test file, likely outdated)

#### strategy/strategies/ violations
- ❌ `src/strategy/strategies/simple_trend.py` ← REVIEW: may be legitimate simple strategy

#### data_OLD/ violations
- ❌ `src/data_OLD/simple_loader.py` ← DELETE (entire directory marked for deletion)

### Step/Version Suffix Violations (11 files)

#### step2_ prefix (5 files - ALL IN RISK MODULE)
- ❌ `src/risk/step2_container_factory.py` ← DELETE after merging features
- ❌ `src/risk/step2_order_manager.py` ← DELETE after merging features  
- ❌ `src/risk/step2_portfolio_state.py` ← DELETE after merging features
- ❌ `src/risk/step2_position_sizer.py` ← DELETE after merging features
- ❌ `src/risk/step2_risk_limits.py` ← DELETE after merging features

#### _refactored suffix (3 files)
- ❌ `src/execution/backtest_broker_refactored.py` (covered above)
- ❌ `src/execution/containers_refactored.py` ← DELETE, merge into canonical
- ❌ `src/strategy/optimization/walk_forward_refactored.py` ← DELETE, merge into `walk_forward.py`

#### _v3 suffix (1 file)
- ❌ `src/core/logging/test_logging_v3.py` ← DELETE (outdated test file)

#### _OLD directory (7 files)
- ❌ `src/data_OLD/` ← **DELETE ENTIRE DIRECTORY** after confirming no needed features

### Additional Container Issues

#### Execution container files (4 implementations)
- ✅ `src/execution/containers_pipeline.py` ← **KEEP as canonical**
- ❌ `src/execution/containers_refactored.py` ← DELETE after merging features

#### Config validation (2 implementations)
- ✅ `src/core/config/schema_validator.py` ← **KEEP as canonical**
- ❌ `src/core/config/simple_validator.py` ← DELETE after merging features

### Legacy Files with _fixed suffix (2 files)
- ❌ `src/data_OLD/FIXED_EXAMPLES/handler_fixed.py` ← DELETE (in _OLD directory)
- ❌ `src/data_OLD/FIXED_EXAMPLES/protocols_fixed.py` ← DELETE (in _OLD directory)

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

#### 1. Brokers - CONSOLIDATE TO 1 CANONICAL
```bash
# KEEP CANONICAL:
src/execution/backtest_broker.py  # Base implementation

# MERGE FEATURES FROM AND DELETE:
src/execution/backtest_broker_refactored.py    # Review refactoring improvements
src/execution/improved_backtest_broker.py      # Review improvements

# ACTION: Review "improved" and "refactored" for features to merge into canonical
```

#### 2. Execution Engines - CONSOLIDATE TO 1 CANONICAL  
```bash
# KEEP CANONICAL:
src/execution/execution_engine.py  # Primary engine

# MERGE FEATURES FROM AND DELETE:
src/execution/improved_execution_engine.py     # Review improvements
src/execution/simple_backtest_engine.py       # Review if any simplifications needed

# ACTION: Merge any valuable features into canonical execution_engine.py
```

#### 3. Order Management - CONSOLIDATE TO 1 CANONICAL
```bash
# KEEP CANONICAL:
src/execution/order_manager.py  # Primary order manager

# MERGE FEATURES FROM AND DELETE:
src/execution/improved_order_manager.py        # Review improvements
src/risk/step2_order_manager.py               # Review step2 features

# ACTION: Merge risk-specific features into canonical order_manager.py
```

#### 4. Market Simulation - CONSOLIDATE TO 1 CANONICAL
```bash
# KEEP CANONICAL:
src/execution/market_simulation.py  # Base simulation

# MERGE FEATURES FROM AND DELETE:
src/execution/improved_market_simulation.py    # Review improvements

# ACTION: Merge improvements into canonical market_simulation.py
```

#### 5. Container Factories - CONSOLIDATE TO 1 CANONICAL
```bash
# KEEP CANONICAL:
src/execution/backtest_container_factory.py  # Base factory

# MERGE FEATURES FROM AND DELETE:
src/execution/backtest_container_factory_traced.py  # Add tracing as capability
src/execution/new_backtest_container_factory.py     # Review "new" features
src/risk/step2_container_factory.py                 # Merge risk-specific features

# ACTION: Add tracing as configurable capability, not separate class
```

#### 6. Additional Execution Cleanup
```bash
# DELETE THESE FILES (merge features into canonical versions):
src/execution/improved_dependency_injection.py      # Merge into dependency injection
src/execution/containers_refactored.py             # Merge into containers_pipeline.py
```

### Phase 3: Risk Module (Week 3) - MASSIVE STEP2_ CLEANUP

#### 1. Portfolio State - CONSOLIDATE TO 1 CANONICAL
```bash
# KEEP CANONICAL:
src/risk/portfolio_state.py  # Primary portfolio implementation

# MERGE FEATURES FROM AND DELETE:
src/risk/improved_risk_portfolio.py           # Review improvements
src/risk/step2_portfolio_state.py            # Review step2 features

# DELETE DEPRECATED:
src/risk/risk_portfolio.py                   # Old implementation
```

#### 2. COMPLETE STEP2_ CLEANUP (5 FILES TO DELETE)
```bash
# DELETE ALL STEP2_ FILES after merging features:
src/risk/step2_container_factory.py          # Merge factory features
src/risk/step2_order_manager.py              # Merge order management features
src/risk/step2_portfolio_state.py            # Merge portfolio features (covered above)
src/risk/step2_position_sizer.py             # Merge into position_sizing.py
src/risk/step2_risk_limits.py                # Merge into risk_limits.py
```

#### 3. Additional Risk Module Cleanup
```bash
# CONSOLIDATE CAPABILITIES:
src/risk/capabilities.py                     # KEEP canonical
src/risk/improved_capabilities.py            # DELETE after merging improvements

# CONSOLIDATE SIGNAL FLOW:
src/risk/signal_flow.py                      # KEEP canonical  
src/risk/optimized_signal_flow.py            # DELETE after merging optimizations
```

### Phase 4: Strategy Module (Week 4)

#### 1. Strategy Container Cleanup
```bash
# CONSOLIDATE STRATEGY CONTAINERS:
# Keep canonical classifier container
src/strategy/classifiers/classifier_container.py      # KEEP canonical

# DELETE ENHANCED VERSION:
src/strategy/enhanced_strategy_container.py           # DELETE after merging capabilities
```

#### 2. Optimization Module Cleanup
```bash
# CONSOLIDATE WALK FORWARD:
src/strategy/optimization/walk_forward.py             # KEEP canonical
src/strategy/optimization/walk_forward_refactored.py  # DELETE after merging refactorings
```

#### 3. Strategy Verification
```bash
# REVIEW THIS FILE - may be legitimate:
src/strategy/strategies/simple_trend.py               # REVIEW: legitimate simple strategy vs naming violation
```

### Phase 5: Core Module Final Cleanup (Week 5)

#### 1. Config Module Cleanup
```bash
# CONSOLIDATE CONFIG VALIDATION:
src/core/config/schema_validator.py           # KEEP canonical
src/core/config/simple_validator.py           # DELETE after merging features
```

#### 2. Container Module Cleanup  
```bash
# CONSOLIDATE BOOTSTRAP:
src/core/containers/bootstrap.py              # KEEP canonical
src/core/containers/minimal_bootstrap.py      # DELETE after merging features
```

#### 3. Logging Module Cleanup
```bash
# DELETE OUTDATED TEST:
src/core/logging/simple_test.py               # DELETE (outdated test file)
src/core/logging/test_logging_v3.py           # DELETE (version-suffixed test)
```

### Phase 6: Root Directory Organization (Week 6)

#### 1. Create tmp/ Directory Structure
```bash
# CREATE ORGANIZED tmp/ STRUCTURE:
mkdir -p tmp/{debug,analysis,reports,prototypes,logs,scratch}

# MOVE ROOT DIRECTORY CHAOS TO tmp/:
# Move 22 test files to tmp/debug/
mv test_*.py tmp/debug/

# Move 20 log files to tmp/logs/  
mv *.log tmp/logs/

# Move 13 analysis/script files to tmp/analysis/
mv *audit*.py tmp/analysis/
mv debug_*.py tmp/analysis/
mv analyze_*.py tmp/analysis/
mv fix_*.py tmp/analysis/

# Move 7 HTML reports to tmp/reports/
mv heatmap_*.html tmp/reports/

# Move status documents to tmp/reports/
mv *_SUMMARY.md tmp/reports/
mv *_PLAN.md tmp/reports/
mv *_IMPLEMENTATION*.md tmp/reports/

# DELETE temporary files:
rm *~ \#*\#
```

#### 2. Root Directory Whitelist (Target: 147 → 15 files)
```bash
# KEEP ONLY THESE IN ROOT:
README.md                    # Fix from README.MD
CLAUDE.md                   # LLM guidelines
STYLE.md                    # Style guide  
requirements.txt            # Dependencies
main.py                     # Entry point
config/                     # Configuration directory
src/                        # Source code
docs/                       # Documentation
tests/                      # Test suites
logs/                       # Organized logs
reports/                    # Organized reports
examples/                   # Example files
scripts/                    # Organized scripts
data/                       # Data files
tmp/                        # ALL temporary work
```

### Phase 7: Final Cleanup (Week 7)

#### 1. Delete data_OLD Directory (7 files)
```bash
# DELETE ENTIRE DIRECTORY:
rm -rf src/data_OLD/                          # Contains 7 legacy files including:
# - src/data_OLD/simple_loader.py
# - src/data_OLD/FIXED_EXAMPLES/handler_fixed.py  
# - src/data_OLD/FIXED_EXAMPLES/protocols_fixed.py
# - And 4 other legacy data files
```

#### 2. Final Import Updates
```bash
# UPDATE ALL IMPORTS throughout codebase for deleted files
# REMOVE ALL ABC imports and usage  
# VERIFY NO BROKEN REFERENCES to deleted files
```

#### 3. Comprehensive Testing
```bash
# RUN FULL TEST SUITE after all deletions
# VERIFY ALL WORKFLOWS STILL FUNCTION
# CHECK PERFORMANCE BENCHMARKS unchanged
```

#### 4. Documentation Updates
```bash
# UPDATE MODULE README files
# MARK CANONICAL STATUS in docstrings
# UPDATE ARCHITECTURE DIAGRAMS if needed
```

## 📋 Files Requiring Migration from UniversalScopedContainer

### Strategy Module
- [ ] `src/strategy/components/feature_hub.py`
- [ ] `src/strategy/components/features.py`
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

## ✅ Detailed Implementation Checklist

### Week 1: Core Infrastructure
- [ ] Consolidate 5 type systems into 1
- [ ] Create canonical container protocol  
- [ ] Start migration of 26 files from UniversalScopedContainer
- [ ] Merge 3 workflow managers into `composable_workflow_manager.py`
- [ ] Remove adjective prefixes from core module filenames

### Week 2: Execution Module (17 files to consolidate)
**Brokers (3→1):**
- [ ] Keep `backtest_broker.py` as canonical
- [ ] Merge features from `backtest_broker_refactored.py` and delete
- [ ] Merge features from `improved_backtest_broker.py` and delete

**Engines (3→1):**
- [ ] Keep `execution_engine.py` as canonical
- [ ] Merge features from `improved_execution_engine.py` and delete
- [ ] Merge features from `simple_backtest_engine.py` and delete

**Order Managers (3→1):**  
- [ ] Keep `order_manager.py` as canonical
- [ ] Merge features from `improved_order_manager.py` and delete
- [ ] Merge features from `step2_order_manager.py` and delete

**Market Simulation (2→1):**
- [ ] Keep `market_simulation.py` as canonical
- [ ] Merge features from `improved_market_simulation.py` and delete

**Container Factories (4→1):**
- [ ] Keep `backtest_container_factory.py` as canonical
- [ ] Add tracing as capability from `backtest_container_factory_traced.py`
- [ ] Merge features from `new_backtest_container_factory.py` and delete
- [ ] Merge features from `step2_container_factory.py` and delete

**Additional Cleanup:**
- [ ] Delete `improved_dependency_injection.py` after merging
- [ ] Delete `containers_refactored.py` after merging

### Week 3: Risk Module (8 files to consolidate)
**Portfolio State (3→1):**
- [ ] Keep `portfolio_state.py` as canonical
- [ ] Merge features from `improved_risk_portfolio.py` and delete
- [ ] Merge features from `step2_portfolio_state.py` and delete
- [ ] Delete deprecated `risk_portfolio.py`

**Complete STEP2_ Cleanup (5 files):**
- [ ] Delete `step2_container_factory.py` after merging features
- [ ] Delete `step2_order_manager.py` after merging features
- [ ] Delete `step2_position_sizer.py` after merging into `position_sizing.py`
- [ ] Delete `step2_risk_limits.py` after merging into `risk_limits.py`

**Additional Risk Cleanup:**
- [ ] Delete `improved_capabilities.py` after merging into `capabilities.py`
- [ ] Delete `optimized_signal_flow.py` after merging into `signal_flow.py`

### Week 4: Strategy Module (3 files to consolidate)
- [ ] Delete `enhanced_strategy_container.py` after merging capabilities
- [ ] Delete `walk_forward_refactored.py` after merging into `walk_forward.py`
- [ ] Review `simple_trend.py` - determine if legitimate strategy or naming violation

### Week 5: Core Module Final Cleanup (5 files)
- [ ] Delete `simple_validator.py` after merging into `schema_validator.py`
- [ ] Delete `minimal_bootstrap.py` after merging into `bootstrap.py`
- [ ] Delete `simple_test.py` (outdated test file)
- [ ] Delete `test_logging_v3.py` (version-suffixed test)

### Week 6: Final Pass (7+ files)
- [ ] Delete entire `data_OLD/` directory (7 files)
- [ ] Remove all ABC imports and usage throughout codebase
- [ ] Update all imports for deleted files
- [ ] Run comprehensive test suite
- [ ] Update documentation and mark canonical status
- [ ] Verify no broken references to deleted files

### Final Success Metrics
- [ ] **ZERO files with adjective prefixes** (improved_, enhanced_, simple_, minimal_, optimized_)
- [ ] **ZERO files with version suffixes** (_refactored, _v3, _fixed) 
- [ ] **ZERO files with step prefixes** (step2_)
- [ ] **ZERO legacy directories** (data_OLD/)
- [ ] **ONE canonical implementation per concept**
- [ ] **All 50 violations resolved**

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