# ADMF-PC Cleanup and BACKTEST.MD Implementation Plan

## Phase 1: Cleanup Duplicate Functionality

### 1.1 Coordinator Module Cleanup
**Keep:**
- `src/core/coordinator/coordinator.py` - Main implementation
- `src/core/coordinator/protocols.py` - Clean interfaces
- `src/core/coordinator/managers.py` - Workflow managers

**Remove:**
- `minimal_coordinator.py` - Temporary workaround
- `yaml_coordinator.py` - Functionality should be in main coordinator
- `simple_backtest_manager.py` - Use main backtest_manager.py
- `backtest_manager.py` - Merge best parts into managers.py
- `execution_modes.py` - Move to execution module

**Consolidate Types:**
- Keep `types.py` as main types file
- Remove `simple_types.py`, `types_no_pydantic.py`
- Move types from `src/core/minimal_types.py` to proper location

### 1.2 Bootstrap Cleanup
**Keep:**
- `src/core/containers/bootstrap.py` - Main implementation

**Remove:**
- `minimal_bootstrap.py` - Temporary workaround

### 1.3 Execution Module Cleanup
**Keep:**
- `src/execution/backtest_engine.py` - Main backtest engine
- `src/execution/backtest_broker.py` - Main broker implementation
- `src/execution/protocols.py` - Clean interfaces

**Remove:**
- `simple_backtest_engine.py` - Temporary implementation
- `backtest_broker_refactored.py` - Merge improvements into main
- `src/backtest/backtest_engine.py` - Duplicate in wrong location

**Review:**
- `backtest_container_factory.py` - May be the right approach for BACKTEST.MD

### 1.4 Data Module Cleanup
**Keep:**
- `src/data/loaders.py` - Main data loading infrastructure

**Remove:**
- `simple_loader.py` - Merge train/test split functionality into main

### 1.5 Classifier Module Cleanup
**Keep:**
- `src/strategy/classifiers/classifier_container.py` - Base implementation

**Remove:**
- `enhanced_classifier_container.py` - Merge enhancements into main

### 1.6 Test File Cleanup
**Remove all duplicate test files and keep only:**
- One comprehensive test per module
- Integration tests in `tests/integration/`
- Unit tests in `tests/unit/`

**Remove all these test files from root:**
- All `test_*.py` files should be in `tests/` directory
- All `run_*.py` files should be in `examples/` or `scripts/`

### 1.7 Example File Consolidation
**Create:**
- `examples/` directory for all example files

**Move:**
- All `example_*.py` files to `examples/`
- All `run_*.py` files to `examples/`

## Phase 2: Implement BACKTEST.MD Architecture

### 2.1 Create Core Container Architecture
```
src/core/containers/
├── protocols.py          # Container protocols
├── backtest/
│   ├── __init__.py
│   ├── factory.py       # BacktestContainerFactory
│   ├── container.py     # BacktestContainer
│   └── patterns.py      # Three patterns (Full, Signal Replay, Signal Gen)
├── lifecycle.py         # Container lifecycle management
└── factory.py           # Base factory patterns
```

### 2.2 Implement Three Backtest Patterns

1. **Full Backtest Pattern**
   - Create `FullBacktestContainer`
   - Implement complete hierarchy: Data → Indicators → Classifiers → Strategies → Risk → Execution

2. **Signal Replay Pattern**
   - Create `SignalReplayContainer`
   - Implement: Signal Logs → Ensemble → Risk → Execution

3. **Signal Generation Pattern**
   - Create `SignalGenerationContainer`
   - Implement: Data → Indicators → Classifiers → Strategies → Analysis

### 2.3 Implement Shared Components
- `IndicatorHub` - Compute once, share across all strategies
- `DataStreamer` - Unified data streaming interface
- `EventBus` - Scoped event system per container

### 2.4 Update Coordinator
- Remove all execution logic from coordinator
- Coordinator only orchestrates workflow phases
- Use factory pattern to create appropriate containers

## Phase 3: Migration Strategy

### 3.1 Incremental Migration
1. Start with `SignalGenerationContainer` (simplest)
2. Then implement `SignalReplayContainer`
3. Finally implement full `BacktestContainer`

### 3.2 Maintain Compatibility
- Keep main.py working throughout migration
- Add feature flags for new vs old implementation
- Comprehensive testing at each step

## Phase 4: Final Structure

### Target Directory Structure
```
src/
├── core/
│   ├── coordinator/
│   │   ├── __init__.py
│   │   ├── coordinator.py    # Main orchestrator
│   │   ├── protocols.py      # Coordinator protocols
│   │   ├── managers.py       # Workflow managers
│   │   └── types.py         # Coordinator types
│   ├── containers/
│   │   ├── __init__.py
│   │   ├── protocols.py     # Container protocols
│   │   ├── factory.py       # Base factory
│   │   ├── lifecycle.py     # Lifecycle management
│   │   ├── bootstrap.py     # Bootstrap system
│   │   └── backtest/        # Backtest-specific containers
│   └── events/              # Event system
├── execution/
│   ├── __init__.py
│   ├── protocols.py         # Execution protocols
│   ├── engine.py           # Execution engine
│   ├── broker.py           # Broker implementation
│   └── analysis/           # Signal analysis
├── strategy/
│   ├── indicators/         # Indicator hub
│   ├── classifiers/        # Regime classifiers
│   └── strategies/         # Trading strategies
├── risk/
│   ├── protocols.py        # Risk protocols
│   ├── manager.py          # Risk manager
│   └── portfolio.py        # Portfolio management
└── data/
    ├── protocols.py        # Data protocols
    ├── loaders.py          # Data loading
    └── streamers.py        # Data streaming
```

## Implementation Priority

1. **Immediate (Week 1)**
   - Clean up duplicate files
   - Consolidate types
   - Move tests to proper directories

2. **Short Term (Week 2-3)**
   - Implement container factories
   - Create three backtest patterns
   - Update coordinator to use factories

3. **Medium Term (Week 4-5)**
   - Implement IndicatorHub
   - Complete event bus scoping
   - Full integration testing

4. **Long Term (Week 6+)**
   - Performance optimization
   - Cloud deployment readiness
   - Documentation and examples

## Success Criteria

1. **No Duplicate Files**: One canonical implementation per functionality
2. **Clean Architecture**: Clear separation of concerns per BACKTEST.MD
3. **Three Patterns Work**: Full, Signal Replay, and Signal Generation
4. **Standardized Creation**: Every container created identically
5. **Massive Parallelization**: Can run 1000s of backtests in parallel
6. **Clean Event Flow**: No circular dependencies
7. **Protocol-Based**: No deep inheritance chains

## Notes

- Start with cleanup to avoid building on shaky foundation
- Each phase should maintain working system
- Comprehensive tests at each step
- Document decisions as we go