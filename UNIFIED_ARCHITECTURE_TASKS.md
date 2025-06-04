# Unified Stateless Architecture Implementation Tasks

## Overview
Transform ADMF-PC to use a single universal topology with stateless components, eliminating pattern detection complexity and reducing containers by 60%.

## Phase 1: Foundation - Define Protocols (Week 1)

### 1.1 Create Protocol Definitions
- [ ] File: `src/core/protocols.py` - Add stateless component protocols
  - [ ] `StatelessStrategy` protocol for pure function signal generation
  - [ ] `StatelessClassifier` protocol for pure function regime detection  
  - [ ] `StatelessRiskValidator` protocol for pure function risk validation
  - [ ] Ensure protocols are minimal - just the essential methods

### 1.2 Define Stateful Container Roles
- [ ] File: `src/core/containers/README.md` - Document which containers remain stateful
  - [ ] Data container: streaming position, timeline, cache
  - [ ] FeatureHub container: indicator calculations, caching
  - [ ] Portfolio container: position tracking, P&L, history
  - [ ] Execution container: order lifecycle, fill management

## Phase 2: Enhance WorkflowManager (Week 1-2)

### 2.1 Add Universal Topology Creation
- [ ] File: `src/core/coordinator/topology.py` - Enhance WorkflowManager
  - [ ] Add `_create_universal_topology()` method
  - [ ] Always create same 4 container types regardless of config
  - [ ] Create stateless components (not containers) for strategies/classifiers
  - [ ] Add `_expand_parameter_combinations()` for portfolio creation
  - [ ] Add `_calculate_parameter_grid()` to detect all parameter combinations
  - [ ] Create one portfolio container per parameter combination
  - [ ] Map each portfolio to its strategy/classifier/risk combo via combo_id

### 2.2 Add Mode Detection
- [ ] File: `src/core/coordinator/topology.py` - Add mode handling
  - [ ] Add `_determine_mode()` method to detect workflow mode
  - [ ] Add WorkflowMode enum (BACKTEST, SIGNAL_GENERATION, SIGNAL_REPLAY)
  - [ ] Update `execute()` to route to mode-specific execution

### 2.3 Add Universal Adapter Wiring
- [ ] File: `src/core/coordinator/topology.py` - Add adapter creation
  - [ ] Add `_create_universal_adapters()` method
  - [ ] Always create same 4 adapters:
    - BroadcastAdapter: feature_hub → strategies/classifiers
    - RoutingAdapter: strategies → portfolios (by combo_id)
    - PipelineAdapter: portfolios → execution
    - BroadcastAdapter: execution → portfolios

## Phase 2.5: Ensure Multi-Phase Compatibility (Week 1-2)

### 2.5.1 Update Sequencer for Base Modes
- [ ] File: `src/core/coordinator/sequencer.py` - Work with base modes
  - [ ] Update `execute_phases()` to use WorkflowManager with modes
  - [ ] Each phase specifies its execution mode (backtest/signal_gen/replay)
  - [ ] Preserve checkpointing and phase inheritance
  - [ ] Maintain result aggregation across phases

### 2.5.2 Define Composite Workflow Types
- [ ] File: `src/core/types/workflow.py` - Add composite types
  - [ ] Keep WorkflowType.OPTIMIZATION (multi-phase parameter search)
  - [ ] Keep WorkflowType.WALK_FORWARD (rolling window validation)
  - [ ] Add clear distinction: base modes vs composite workflows
  - [ ] Document that composite workflows use sequences of base modes

## Phase 3: Implement Execution Modes (Week 2)

### 3.1 Full Backtest Mode
- [ ] File: `src/core/coordinator/topology.py` - Add backtest execution
  - [ ] Add `_execute_full_pipeline()` method
  - [ ] Run complete pipeline: data → features → signals → orders → fills
  - [ ] Ensure all events flow through entire pipeline

### 3.2 Signal Generation Mode
- [ ] File: `src/core/coordinator/topology.py` - Add signal generation
  - [ ] Add `_execute_signal_generation()` method
  - [ ] Stop pipeline after signal generation
  - [ ] Add signal storage mechanism
  - [ ] No execution container activation

### 3.3 Signal Replay Mode
- [ ] File: `src/core/coordinator/topology.py` - Add signal replay
  - [ ] Add `_execute_signal_replay()` method
  - [ ] Load stored signals
  - [ ] Skip strategy/classifier execution
  - [ ] Start from portfolio → execution flow

## Phase 4: Convert Components to Stateless (Week 2-3)

### 4.1 Convert Strategies
- [ ] File: `src/strategy/strategies/momentum.py` - Make stateless
  - [ ] Remove container inheritance
  - [ ] Convert to pure functions or minimal state (config only)
  - [ ] Implement StatelessStrategy protocol
  - [ ] Test with universal topology

### 4.2 Convert Classifiers
- [ ] File: `src/strategy/classifiers/hmm_classifier.py` - Make stateless
  - [ ] Remove container inheritance
  - [ ] Convert to pure functions
  - [ ] Implement StatelessClassifier protocol
  - [ ] Test with universal topology

### 4.3 Convert Risk Components
- [ ] File: `src/risk/limits.py` - Make stateless
  - [ ] Convert risk validators to pure functions
  - [ ] Accept portfolio state as parameter
  - [ ] Return validation decisions
  - [ ] No internal state tracking

## Phase 5: Simplify Configuration (Week 3)

### 5.1 Update YAML Schema
- [ ] File: `src/core/config/schemas.py` - Simplify schema
  - [ ] Remove pattern-related fields
  - [ ] Add simple 'mode' field
  - [ ] Keep strategy/classifier parameter grids
  - [ ] Remove communication pattern configs

### 5.2 Create Example Configs
- [ ] File: `config/examples/unified_backtest.yaml`
  - [ ] Simple backtest example with mode: backtest
- [ ] File: `config/examples/unified_signal_gen.yaml`
  - [ ] Signal generation example
- [ ] File: `config/examples/unified_replay.yaml`
  - [ ] Signal replay example

## Phase 6: Delete Complexity (Week 3-4)

### 6.1 Remove Pattern Detection
- [ ] Delete: `src/core/coordinator/workflows/config/pattern_detector.py`
- [ ] Delete: `src/core/coordinator/workflows/patterns/` directory
- [ ] Remove pattern detection from WorkflowManager
- [ ] Remove pattern registry references

### 6.2 Remove Multiple Executors
- [ ] Delete: `src/core/coordinator/workflows/execution/nested_executor.py`
- [ ] Delete: `src/core/coordinator/workflows/execution/pipeline_executor.py`
- [ ] Delete: `src/core/coordinator/workflows/execution/standard_executor.py`
- [ ] Delete: `src/core/coordinator/workflows/execution/multi_pattern_executor.py`
- [ ] Update imports to remove executor references

### 6.3 Clean Up Coordinator
- [ ] File: `src/core/coordinator/coordinator.py` - Remove complexity
  - [ ] Remove pattern-based execution paths
  - [ ] Remove execution mode detection (WorkflowManager handles it)
  - [ ] Simplify to just delegate to WorkflowManager

## Phase 7: Testing & Validation (Week 4)

### 7.1 Unit Tests
- [ ] File: `tests/unit/test_stateless_components.py`
  - [ ] Test stateless strategies with mock data
  - [ ] Test stateless classifiers
  - [ ] Test stateless risk validators
  - [ ] Verify no state mutations

### 7.2 Integration Tests
- [ ] File: `tests/integration/test_unified_topology.py`
  - [ ] Test all three modes with same topology
  - [ ] Verify adapter wiring is consistent
  - [ ] Test parameter expansion
  - [ ] Verify signal storage/replay

### 7.3 Performance Tests
- [ ] File: `tests/performance/test_stateless_efficiency.py`
  - [ ] Measure container count reduction
  - [ ] Test parallel execution of stateless components
  - [ ] Verify memory usage improvements
  - [ ] Benchmark vs old pattern-based system

## Phase 8: Documentation (Week 4)

### 8.1 Update Architecture Docs
- [ ] File: `docs/architecture/UNIFIED_ARCHITECTURE.md`
  - [ ] Document the three modes
  - [ ] Explain universal topology
  - [ ] Show configuration examples
  - [ ] Benefits and trade-offs

### 8.2 Migration Guide
- [ ] File: `docs/MIGRATION_TO_UNIFIED.md`
  - [ ] How to convert existing configs
  - [ ] How to convert custom strategies
  - [ ] Common pitfalls and solutions

## Success Metrics

- [ ] 60% reduction in container count achieved
- [ ] All tests passing with unified architecture
- [ ] 90% reduction in workflow management code
- [ ] Configuration files reduced to <20 lines
- [ ] Performance improvement from parallelization
- [ ] Zero pattern detection code remaining

## Notes

- Keep old system running during migration
- Test each phase thoroughly before proceeding
- Focus on enhancing WorkflowManager, not creating new components
- Ensure backward compatibility until full migration complete
- Delete aggressively once new system is validated