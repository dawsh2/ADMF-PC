# Duplicate Implementations and ADMF-PC Violations Audit

## Overview
This audit identifies duplicate/competing implementations that violate ADMF-PC's core principles of Protocol + Composition (no inheritance), Container Architecture (complete isolation), and Event-Driven Design.

## Critical Violations

### 1. BaseComposableContainer Inheritance Hierarchy (MAJOR VIOLATION)
**Location**: `/src/core/containers/composable.py`
- **Problem**: `BaseComposableContainer` is a base class that multiple containers inherit from
- **Violates**: "NEVER use inheritance - Only Protocol + Composition"
- **Files using inheritance**:
  - `/src/execution/containers.py` - 7 containers inherit from BaseComposableContainer
  - `/src/execution/containers_fixed.py` - 3 containers inherit
  - `/src/execution/containers_nested.py` - 3 containers inherit
  - `/src/execution/containers_pipeline.py` - 7 containers inherit
- **Solution**: Replace with protocol-based composition using `ComposableContainerProtocol`

### 2. Multiple Container Implementations (MAJOR DUPLICATION)
**Execution containers with 4 competing versions**:
- `/src/execution/containers.py` (62KB) - Original implementation
- `/src/execution/containers_fixed.py` (17KB) - "Fixed" version
- `/src/execution/containers_nested.py` (17KB) - Nested version
- `/src/execution/containers_pipeline.py` (71KB) - Pipeline version

**Each file implements the same containers differently**:
- BacktestContainer (in 2 files)
- DataContainer (in 2 files)
- IndicatorContainer (in 2 files)
- StrategyContainer (in 4 files)
- RiskContainer (in 4 files)
- PortfolioContainer (in 4 files)
- ExecutionContainer (in 2 files)

### 3. Multiple Workflow Manager Implementations (DUPLICATION)
**Three competing workflow managers**:
- `/src/core/coordinator/composable_workflow_manager.py` (54KB)
- `/src/core/coordinator/composable_workflow_manager_nested.py` (13KB)
- `/src/core/coordinator/composable_workflow_manager_pipeline.py` (22KB)

### 4. Multiple Event Bus Implementations
**Event bus implementations found in**:
- `/src/core/events/event_bus.py` - Main implementation
- `/src/core/events/tracing/traced_event_bus.py` - Traced version
- `/src/core/containers/enhanced_container.py` - Enhanced version
- `/src/core/communication/integration.py` - Integration version

### 5. Multiple Broker Implementations (DUPLICATION)
**Three backtest broker implementations**:
- `/src/execution/backtest_broker.py`
- `/src/execution/backtest_broker_refactored.py`
- `/src/execution/improved_backtest_broker.py`

### 6. Multiple Engine Implementations (DUPLICATION)
**Execution engines**:
- `/src/execution/execution_engine.py`
- `/src/execution/improved_execution_engine.py`
- `/src/execution/backtest_engine.py`
- `/src/execution/simple_backtest_engine.py`

**Signal engines**:
- `/src/execution/signal_generation_engine.py`
- `/src/execution/signal_replay_engine.py`

### 7. Multiple Order Manager Implementations (DUPLICATION)
- `/src/execution/order_manager.py`
- `/src/execution/improved_order_manager.py`
- `/src/risk/step2_order_manager.py`

### 8. Multiple Portfolio Implementations (DUPLICATION)
- `/src/risk/portfolio_state.py`
- `/src/risk/risk_portfolio.py`
- `/src/risk/improved_risk_portfolio.py`
- `/src/risk/step2_portfolio_state.py`

### 9. Multiple Factory Implementations (DUPLICATION)
**Container factories**:
- `/src/core/containers/factory.py`
- `/src/core/containers/backtest/factory.py`
- `/src/execution/backtest_container_factory.py`
- `/src/execution/backtest_container_factory_traced.py`
- `/src/risk/step2_container_factory.py`
- `/src/execution/execution_module_factory.py`

### 10. Classifier Inheritance Hierarchy (VIOLATION)
**Location**: `/src/strategy/classifiers/`
- **Problem**: `BaseClassifier` abstract base class with inheritance
- **Inheritors**:
  - `DummyClassifier(BaseClassifier)`
  - `HMMClassifier(BaseClassifier)`
  - `PatternClassifier(BaseClassifier)`
- **Solution**: Replace with protocol-based design

### 11. Multiple Classifier Container Implementations
- `/src/strategy/classifiers/classifier_container.py` - Original
- `/src/strategy/classifiers/enhanced_classifier_container.py` - Enhanced version

### 12. "Improved" and "Step2" Prefixed Files (DUPLICATION PATTERN)
Multiple files with "improved_", "step2_", or similar prefixes indicating iterative development:
- `improved_backtest_broker.py`
- `improved_execution_engine.py`
- `improved_order_manager.py`
- `improved_risk_portfolio.py`
- `step2_order_manager.py`
- `step2_portfolio_state.py`
- `step2_container_factory.py`

## Recommended Actions

### 1. Eliminate BaseComposableContainer
- Remove inheritance-based `BaseComposableContainer`
- Use only `ComposableContainerProtocol` for type contracts
- Implement containers through composition, not inheritance

### 2. Choose ONE Container Implementation
- Keep only `/src/execution/containers_pipeline.py` (most complete)
- Delete containers.py, containers_fixed.py, containers_nested.py
- Ensure it uses protocols, not inheritance

### 3. Consolidate Workflow Managers
- Keep only one workflow manager (recommend pipeline version)
- Delete the other two implementations

### 4. Single Event Bus
- Keep only `/src/core/events/event_bus.py`
- If tracing needed, add as decorator/wrapper, not separate implementation

### 5. One Broker Implementation
- Keep the most recent/complete broker implementation
- Delete all "improved" and "refactored" versions

### 6. Consolidate Engines
- One execution engine
- One backtest engine (if different from execution)
- Remove "simple" and "improved" variants

### 7. Single Portfolio State
- Choose one portfolio implementation
- Remove all variants and "step2" versions

### 8. Protocol-Based Classifiers
- Replace BaseClassifier ABC with a protocol
- Convert all classifiers to use composition

### 9. Remove Version-Named Files
- No "improved_", "step2_", "_fixed", "_refactored" files
- Either update the original or replace it entirely

### 10. One Factory Pattern
- Single factory implementation using protocols
- Remove specialized factories unless absolutely necessary

### 13. ABC (Abstract Base Class) Usage (VIOLATION)
**17 files using ABC pattern instead of Protocols**:
- `/src/core/coordinator/managers.py`
- `/src/core/infrastructure/validation.py`
- `/src/risk/position_sizing.py`
- `/src/risk/risk_limits.py`
- `/src/strategy/classifiers/classifier.py`
- `/src/strategy/signal_aggregation.py`
- And 11 more files

**Problem**: ABC forces inheritance, violates protocol-based design

### 14. Legacy Code Directory
**Location**: `/src/data_OLD/`
- Contains 12 files of old data handling implementation
- Includes ABC-based protocols, handlers, loaders
- Should be completely removed

### 15. Manager Pattern Duplication
Multiple "manager" implementations suggesting centralized control:
- `composable_workflow_manager.py` (3 versions)
- `backtest_manager.py`
- `order_manager.py` (3 versions)
- `subscription_manager.py`
- `log_manager.py`

## Summary
The codebase contains approximately **50+ duplicate implementations** that need consolidation. The main violations are:
1. Inheritance-based design (BaseComposableContainer, BaseClassifier, ABC usage)
2. Multiple implementations of the same concept (4x containers, 3x workflow managers, etc.)
3. Iterative development files not cleaned up ("improved_", "step2_", "_fixed")
4. Protocol + Inheritance mixed approaches
5. Legacy directories still present (data_OLD)

All of these violate the core ADMF-PC principles:
- **"NEVER use inheritance - Only Protocol + Composition"**
- **"NEVER create duplicate files - modify canonical files directly"**
- **"Configuration > Code"**