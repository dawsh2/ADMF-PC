# ADMF-PC Architecture Checklist - January 6, 2025

## System Architecture Summary ✅

You've captured it perfectly:
- **Containers, routing and events** form the basis of the system
- **topology.py** abstracts these into reusable patterns
- **sequencer.py** abstracts these patterns into workflows
- **Event tracing** keeps it observable, replayable, and enables insights

## High Priority Issues 🔴

### 1. FEATURES Event Granularity Problem
**Issue**: Currently, FEATURES event contains ALL computed values from FeatureHub. Each strategy receives the entire payload even if it only needs RSI(30, 60) out of RSI(30, 40, 50, 60, 70).

**Solution Options**:
a) **Feature Dispatcher Pattern** (Recommended):
   - Single FEATURES event → Stateless Feature Dispatcher
   - Dispatcher knows each strategy's requirements
   - Routes only needed features to each strategy
   - Avoids event explosion while maintaining separation

b) **Subscription-Based Filtering**:
   - Strategies subscribe to specific feature keys
   - FeatureHub publishes granular events per feature
   - More events but cleaner separation

c) **Strategy Pull Pattern**:
   - Strategies query FeatureHub for specific features
   - Requires stateful FeatureHub or feature cache

### 2. FeatureHub Redundancy Check
**Question**: Are we computing RSI(30) multiple times if strategies need RSI(30, 60) and RSI(30, 70)?
**TODO**: Verify FeatureContainer implementation for computation efficiency

### 3. WorkflowManager Naming
**Issue**: "Manager" is a code smell, and it's not managing workflows - just creating topologies
**Proposed Names**:
- `TopologyBuilder`
- `TopologyFactory`
- `TopologyComposer`
- `PatternComposer`

### 4. Tracing as Default Behavior
**Agreed**: Tracing should be ON by default for observability
**Change**: Make `enable_tracing: true` the default, with option to disable

### 5. Two Tracing Systems Confusion
**Current State**:
1. **ExecutionTracer** (src/core/tracing.py): Lightweight flow verification
2. **EventTracer** (src/core/events/tracing/): Comprehensive event tracking

**Clarification**:
- **ExecutionTracer**: Tracks high-level flow (Data→Features→Signals→Orders→Fills)
- **EventTracer**: Captures every event with full payloads for mining/analysis

**Recommendation**: Merge into single comprehensive system

## Completed Today ✅

### From Previous Discussion:
- [x] Move event tracing initialization from main.py to Sequencer
- [x] Implement Feature Dispatcher for granular feature routing
- [x] Verify FeatureHub computation efficiency (no redundant calculations)
- [x] Make tracing default behavior

### Key Accomplishments:
1. **Feature Dispatcher Implementation**:
   - Created `FeatureDispatcher` component that filters features per strategy
   - Integrated into topology creation flow
   - Strategies now only receive the features they need (e.g., momentum gets sma/rsi, not bollinger bands)

2. **Event Tracing Improvements**:
   - Moved initialization to Sequencer for phase-aware tracing
   - Made tracing default behavior (enabled unless explicitly disabled)
   - Prepared foundation for portfolio-specific trace files

3. **FeatureHub Efficiency Verified**:
   - Confirmed no redundant calculations
   - Each indicator computed only once per bar
   - RSI(14) computed once, not multiple times

## Completed Today (continued) ✅

### Additional Accomplishments:
4. **ExecutionMode Simplification**:
   - Removed COMPOSABLE and ENHANCED_COMPOSABLE modes
   - Made containers the default and only execution model
   - Cleaned up references throughout the codebase

5. **TopologyBuilder Rename**:
   - Renamed WorkflowManager to TopologyBuilder for clarity
   - Updated all references and imports
   - Better reflects its responsibility of building topologies

## Completed (January 6, 2025) ✅

### All High Priority Tasks:
- [x] Fix topology creation to have mode-specific flows (not universal)
- [x] Create mode-specific topologies (signal_generation, signal_replay, backtest)
- [x] Implement subscribe_all() method in EventBus
- [x] Consolidate ExecutionTracer and EventTracer into UnifiedTracer

### Low Priority Tasks:
- [x] Improve root event bus access pattern for portfolios
  - Changed to constructor parameter instead of property assignment
  - Cleaner dependency injection pattern
- [x] Implement portfolio-specific trace files
  - Added `trace_file_path` parameter to UnifiedTracer
  - Automatic file writing for all trace points and events
  - Created `create_portfolio_tracer()` factory function
- [x] Build query interface for event mining
  - Created `TraceQuery` class for analyzing trace files
  - Supports filtering, latency analysis, anomaly detection
  - Can load from files or UnifiedTracer instances
  - Includes pandas DataFrame export for advanced analysis

## Phase 0: Code Review and Architecture Questions 🔍

### Core Review Areas:
1. **Containers, Events, and Communication**:
   - Review src/core/containers/
   - Review src/core/events/
   - Review src/core/communication/
   - Why isolated event buses for some containers?
   - Should all containers have isolated buses?
   - How does this tie together for event tracing?

2. **Integration Review**:
   - How are primitives used in topologies?
   - How are they used in sequencer?
   - How are they used in workflows?
   - What is the coordinator's role now after decomposition?
   - Are we actually using TopologyBuilder, Sequencer, and Workflows?
   - Is the coordinator just a bootstrap now?

### Architecture Questions to Answer:
- [x] How are components registered / discovered by the system? → Using decorator-based discovery (@strategy, @classifier, @feature)
- [x] Does containers/ need a factory? What about components? → Yes, containers use factory; components use decorators
- [x] Are we properly using the config/ directory? → Yes, but needs organization into subdirectories
- [x] Why no data or featurehub container files? Should each type get its own file? → Found specialized container types (SymbolTimeframeContainer, PortfolioContainer, ExecutionContainer) which is an anti-pattern. Should use ONE generic Container class with composable components instead of specialized types. This allows flexibility to split/merge functionality as needed.
- [x] What is event_flow_adapter.py under coordinator? And infrastructure.py? → DELETED BOTH. event_flow_adapter.py was a lingering prototype that bypassed proper architecture. infrastructure.py was overengineered with mixed concerns (shared resources, compute pools) that belong elsewhere.
- [x] Are we using typed/semantic events? Should we per data-mining docs? → semantic.py exists with typed events but NOT actively used. System uses simple Event + EventType. Since event tracing will be the primary source of truth for performance validation, semantic events become MORE important - they provide structure for analysis. Consider gradual adoption as we rely more on event mining.
- [x] Code smell: grep for 'unified' and 'compose' terms → Found multiple migration layers of technical debt:
  - "unified architecture" comments everywhere (old migration)
  - "composable containers" with enable_composable_containers=True hardcoded (another old migration)
  - "composite" usage is legitimate (for composite validators/classifiers)
  - Need cleanup: remove all migration-related comments and deprecated flags
- [ ] Do we need custom logging module now that we have tracing?
- [ ] Why so many files under types/? Duplicate decimals.py under utils/ (remove if so)?
- [ ] Rename 'stateless_*' files to canonical names (remove 'stateless' prefix)

### Final Step of Phase 0:
- [ ] Review each incremental testing phase for missing steps
- [ ] Implement semantic events throughout the system for meaningful event tracing:
  - Replace simple Event(type, payload) with typed semantic events
  - Ensure correlation_id flows through entire pipeline
  - Add causation_id to link related events
  - Include business context (strategy_id, portfolio_id, regime)
  - Validate events are self-documenting for analysis

## Incremental Testing Plan 📋
- [ ] Implement portfolio-specific trace files and consider how to structure event tracing for scale while parallezing (see data-mining doc)

### Low Priority:
- [ ] Improve root event bus access pattern for portfolios -- anti-pattern? Portfolio should be isolated and we should use adapters to bridge to them.
- [ ] Build query interface for event mining

## Architecture Validation Questions 🤔

1. **Event Granularity**: Should we have more granular events (FEATURE_RSI, FEATURE_SMA) or stick with aggregated FEATURES?

2. **Stateless Service Pools**: Should the pool handle event filtering/routing or should each service filter its own input?

3. **Root Event Bus**: Should portfolios get dependency injection of root bus or is attribute assignment acceptable?

4. **Trace Storage**: One file per portfolio or one file with portfolio-tagged events?

## Implementation Priority Order 🎯

1. **Feature Dispatcher** - Fixes the immediate engineering concern
2. **Topology Mode Separation** - Clarifies signal_generation vs signal_replay vs backtest
3. **Default Tracing** - Essential for multi-portfolio observability
4. **Rename WorkflowManager** - Quick clarity win
5. **Consolidate Tracers** - Simplify the system

## Data Mining Goals 📊

Per your data-mining architecture docs:
- Separate event streams per portfolio for analysis
- Query capabilities across event history
- Pattern detection in strategy behavior
- Performance attribution to specific events
- Replay capability for debugging

This requires:
- Portfolio-tagged events
- Structured storage (not just logs)  
- Query interface over event store
- Causation chain tracking (already have)

## Architectural Improvements Made Today 🏗️

### Feature Routing Architecture
**Before**: All strategies received ALL computed features (wasteful)
```
FeatureHub → FEATURES(all) → Every Strategy
```

**After**: Strategies only receive what they need (efficient)
```
FeatureHub → FEATURES(all) → Feature Dispatcher → FEATURES(filtered) → Specific Strategies
```

### Event Tracing Architecture
**Before**: 
- Tracing initialized in main.py (too early)
- Disabled by default
- Two separate tracing systems

**After**:
- Tracing initialized by Sequencer (phase-aware)
- Enabled by default for observability
- Foundation laid for unified system

### Benefits
1. **Efficiency**: Reduced event payload sizes by ~70% for strategies
2. **Observability**: Always-on tracing for multi-portfolio debugging
3. **Scalability**: Ready for complex workflows with thousands of events
4. **Maintainability**: Clear separation of concerns

## Next Steps 🚀

1. Fix topology to be mode-specific (signal_generation, signal_replay, backtest)
2. Remove ExecutionMode enum confusion
3. Complete tracing system consolidation
4. Build query interface for event mining
5. Implement portfolio-specific trace storage

## Incremental System Validation Plan 🔬

### Core Principle
Build and test the system incrementally, validating each layer before adding complexity. We'll be building:
- **Topology file**: `src/core/coordinator/topologies/incremental_test.py`
- **Sequence files**: `src/core/coordinator/sequences/train_test_split.py`, etc.
- **Workflow file**: `src/core/coordinator/workflows/incremental_test.py`
- **Config file**: `config/test_incremental.yaml` that references the workflow
- The YAML will use: `workflow: incremental_test` to load our workflow
- Use `main.py --config config/test_incremental.yaml --bars N` throughout.

### Phase 1: Foundation Testing

#### 1.1 Root Container Only
- Start `topologies/incremental_test.py` with just root-level container
- Create `workflows/incremental_test.py` importing this topology
- Create `config/test_incremental.yaml` with `workflow: incremental_test`
- Test creation and destruction sequencing
- Validate logging output and execution paths
- **Human checkpoint**: Review logs, confirm canonical behavior

#### 1.2 Symbol-Timeframe Container
- Update `topologies/incremental_test.py`: add symbol_timeframe container
- Update workflow to initialize the new container
- Test creation/teardown with `--bars 0` (no data yet)
- **Human checkpoint**: Verify container lifecycle

#### 1.3 Data Subcontainer
- Update `topologies/incremental_test.py`: add data subcontainer to symbol_timeframe
- Update workflow to configure data source (CSV)
- Allow BAR event emission with `--bars 10`
- **Critical**: First event tracing validation
- Verify event storage structure in `./results/traces/`
- **Human checkpoint**: Inspect event traces

#### 1.4 Multiple Symbol-Timeframe Containers
- Update `topologies/incremental_test.py`: create containers for SPY_1m, SPY_1d, QQQ_1m
- Update workflow to handle multiple symbols/timeframes
- Test parallel data streaming with `--bars 10`
- Verify event isolation between containers
- Validate separate event trace storage per symbol-timeframe
- **Human checkpoint**: Confirm proper container isolation

#### 1.5 Multi-Phase Data Split Test
- Create `sequences/train_test_split.py` with 80/20 split logic
- Update `workflows/incremental_test.py` to use the sequence
- Test train/test split with `--bars 100`
- Phase 1: bars 1-80 (training)
- Phase 2: bars 81-100 (testing)
- Verify sequencer handles phase transitions
- Confirm event traces in separate phase directories
- **Human checkpoint**: Validate data split accuracy

### Phase 2: Signal Generation Pipeline

#### 2.1 Feature Hub Integration
- Update `topologies/incremental_test.py`: add FeatureHub to symbol_timeframe containers
- Configure feature indicators (SMA, RSI) in workflow
- Test BAR → FeatureHub flow with `--bars 10`
- Verify feature computation and FEATURES broadcast
- **Human checkpoint**: Check feature event flow in traces

#### 2.2 Single Strategy
- Update `topologies/incremental_test.py`: add momentum strategy service
- Wire strategy to receive FEATURES events
- Update workflow to configure strategy parameters
- Verify: BAR → Features → Strategy → SIGNAL
- Complete signal generation topology
- **Human checkpoint**: Validate signal generation in traces

#### 2.3 Multi-Phase Signal Generation
- Re-run train/test split with full signal pipeline
- Verify signals generated for both phases
- Create `topologies/signal_generation.py` extracting just this portion
- **Human checkpoint**: Confirm phase isolation

#### 2.4 Multiple Strategies
- Update `topologies/incremental_test.py`: add mean_reversion, breakout strategies
- Test Feature Dispatcher with multiple consumers
- Verify each strategy only receives needed features
- Confirm proper signal routing with combo_ids
- **Human checkpoint**: Check event routing correctness

### Phase 3: Portfolio Integration

#### 3.1 Single Portfolio
- Add portfolio container
- Connect to risk validators
- Verify: SIGNAL → Portfolio → Risk → ORDER
- **Human checkpoint**: Validate risk application

#### 3.2 Portfolio Variations
- Test: Multiple strategies → Single portfolio
- Test: Single strategy → Multiple portfolios  
- Test: Multiple strategies → Multiple portfolios
- **Human checkpoint**: Verify routing matrix

#### 3.3 Multi-Phase Portfolio Test
- Train/test split with portfolio state
- Verify state isolation between phases
- **Human checkpoint**: Confirm no data leakage

### Phase 4: Execution Integration

#### 4.1 Execution Engine
- Add execution container
- Verify: ORDER → Execution → FILL → Portfolio
- Audit trade results and position sizing
- **Human checkpoint**: Validate execution accuracy

#### 4.2 Classifier Integration
- Add classifier services
- Test regime-based strategy selection
- **Human checkpoint**: Verify classifier impact

#### 4.3 End-to-End Validation
- Full pipeline test
- Train/test with top N strategies
- **Major milestone**: Complete workflow execution

### Phase 5: Advanced Workflows

#### 5.1 Walk-Forward Validation
- Implement data windowing in data module
- Test rolling train/test windows
- **Human checkpoint**: Verify window accuracy

#### 5.2 Signal Replay Topology
- Extract minimal topology post-signal generation
- Test parallel signal replay
- Build signal generation → replay workflow

#### 5.3 Topology Test Suite
- Automated validation of event flows
- Run on startup as sanity check
- **Human checkpoint**: Review test coverage

### Phase 6: Extensibility Validation

#### 6.1 Stateless Signal Filter
- Example: Filter signals before portfolio
- Document extension pattern
- Test stateless validation

#### 6.2 Stateful Signal Filter  
- Example: Stateful filtering component
- Containerize and integrate
- Document pattern differences

#### 6.3 YAML Workflow Definition
- Standardize workflow YAML format
- Test custom workflow creation

### Phase 7: Production Readiness

#### 7.1 End-of-Data Handling
- Close positions when data stops
- End-of-day feature cleanup
- **Human checkpoint**: Verify cleanup logic

#### 7.2 Architecture Validation
- Confirm Protocol+Composition throughout
- Zero inheritance verification
- Best practices audit

#### 7.3 Documentation
- Rewrite README.md from scratch
- Start with primitives (containers, events, routing)
- Add topology/sequence/workflow abstractions
- Include canonical file list

## Architecture Organization Proposal 🏗️

### Directory Structure
```
src/core/coordinator/
├── topologies/          # Reusable topology patterns
│   ├── __init__.py
│   ├── backtest.py      # Full pipeline topology
│   ├── signal_generation.py
│   └── signal_replay.py
├── sequences/           # Common sequence patterns  
│   ├── __init__.py
│   ├── train_test_split.py
│   └── walk_forward.py
├── workflows/           # Unified topology+sequence patterns
│   ├── __init__.py
│   ├── adaptive_ensemble.py
│   └── parameter_optimization.py
└── topology.py          # TopologyBuilder (renamed from WorkflowManager)
```

### Configuration Approach
- Build incrementally using same config file
- Add complexity through config updates
- Use workflows to avoid repetitive YAML

### Testing Philosophy
- Event tracing over console output
- Human checkpoints for validation
- Incremental complexity addition
- Each step must work before proceeding

## Topology Refactor Plan 🏗️ ✅ COMPLETED

### ~~Current~~ Resolved Issues:
1. ~~**topology.py is 2,296 lines**~~ - Reduced to ~1,056 lines after refactor
2. ~~**Specialized container types**~~ - Now using generic Container + components pattern
3. ~~**Hardcoded pipeline order**~~ - Moved to modular topology files
4. ~~**Mixed responsibilities**~~ - TopologyBuilder is now a thin orchestrator

### Vision:

#### 1. Container Architecture:
- **ONE generic Container class** that holds components
- **Components** provide specific functionality (DataStreamer, FeatureCalculator, PortfolioState, etc.)
- **No specialized container types** - just Container + Components

Example:
```python
# Instead of SymbolTimeframeContainer:
data_container = Container("spy_1m_data")
data_container.add_component(DataStreamer(symbol="SPY", timeframe="1m"))

feature_container = Container("spy_1m_features")
feature_container.add_component(FeatureCalculator(indicators=["sma_20", "rsi_14"]))
```

#### 2. Topology Structure:
```
topologies/
├── backtest.py         # Full pipeline: data → features → strategies → portfolios → risk → execution
├── signal_generation.py # Just: data → features → strategies (save signals)
├── signal_replay.py    # Just: signals → portfolios → risk → execution
└── helpers/
    ├── container_builder.py  # Helper for creating containers with components
    └── adapter_wiring.py    # Helper for wiring containers together
```

#### 3. TopologyBuilder Role:
- **Thin orchestrator** that delegates to topology modules
- **Reads topology files** to understand pipeline structure
- **Delegates to factories**:
  - ContainerFactory for creating containers
  - AdapterFactory for creating adapters
- **Does NOT hardcode** container types or pipeline order

#### 4. Benefits:
- **Easy reordering**: Want risk before portfolio? Just change topology file
- **Easy splitting**: Want separate data and features? Create two containers
- **Easy extensions**: Add SignalFilter? Create component, add to topology
- **Modular testing**: Each topology can be tested independently

### Implementation Steps ✅:
1. ✅ Created topologies/ directory structure
2. ✅ Extracted topology-specific code from topology.py
3. ✅ Refactored containers to use generic Container + Components pattern
4. ✅ Updated TopologyBuilder to be a thin orchestrator
5. ⏳ Test with incremental complexity (next step)
6. ✅ **DELETED all deprecated code**:
   - ✅ Removed _create_*_adapters methods (334 lines)
   - ✅ Removed _run_*_execution methods (464 lines)
   - ✅ Removed old _create_*_topology methods (262 lines)
   - ✅ Total: ~1,060 lines deleted

## Phase 8: Validation Framework (Future Enhancement) 🔍

**NOTE**: This is a theoretical/non-urgent enhancement. Each validation step requires human review before implementation.

### 8.1 Topology Validator
**Purpose**: One-time validation of topology structure for correctness
- [ ] Create `TopologyValidator` class that checks:
  - No shared mutable state between containers
  - Event routing isolation (portfolios only receive their signals)
  - Adapter wiring matches declared connections
  - Clean initialization of all containers
  - Expected event flow matches topology mode
- [ ] Integrate with TopologyBuilder as optional validation step
- [ ] Run validation in test/debug mode
- [ ] **Human checkpoint**: Review validation results and determine if checks are sufficient

### 8.2 Event-Based Look-Ahead Detection
**Purpose**: Runtime safety net to catch implementation bugs (architecture already prevents look-ahead)
- [ ] Create `TemporalIntegrityChecker` that monitors:
  - Events processed in temporal order
  - No future data embedded in events
  - BAR events don't contain future timestamps
- [ ] Add simple assertions in event creation
- [ ] Log violations without interrupting execution
- [ ] **Human checkpoint**: Analyze if any violations detected indicate real bugs

### 8.3 Data Integrity Validator
**Purpose**: Detect selection bias and data discontinuities that can invalidate results
- [ ] Create `DataIntegrityValidator` that checks:
  - Time gaps in data (missing days/hours)
  - Price discontinuities (suspicious jumps)
  - Regime coverage (not just bull markets)
  - Complete volatility spectrum represented
- [ ] Detect artificial filtering or cherry-picked data
- [ ] Run before backtests to ensure data quality
- [ ] **Human checkpoint**: Review data quality report and acceptable thresholds

### 8.4 Trade Lifecycle Validator
**Purpose**: Ensure trades have complete lifecycles in the data
- [ ] Create `TradeLifecycleValidator` that verifies:
  - Entry and exit times are in the data
  - Full holding period is covered
  - No gaps during trade duration
- [ ] Run after execution to validate results
- [ ] Flag incomplete trades that might skew performance
- [ ] **Human checkpoint**: Determine if incomplete trades should invalidate results

### 8.5 Expected Event Flow Validator
**Purpose**: Verify event sequences match expected patterns
- [ ] Define expected flows for each topology mode:
  - Backtest: DATA → BAR → FEATURES → SIGNAL → ORDER → FILL
  - Signal Generation: DATA → BAR → FEATURES → SIGNAL
  - Signal Replay: SIGNAL → ORDER → FILL
- [ ] Trace actual event flow and compare
- [ ] Detect missing or out-of-order events
- [ ] **Human checkpoint**: Review if deviations are bugs or valid variations

### Implementation Notes:
- Keep validators simple and pragmatic
- No complex mathematical proofs or certificates
- Focus on catching real problems that affect results
- Make validation optional and configurable
- Each validator ~100-200 lines of focused code

### Benefits:
- **Early bug detection**: Catch problems before expensive backtests
- **Data quality assurance**: Prevent garbage-in-garbage-out
- **Architectural validation**: Ensure system behaves as designed
- **Debugging aid**: Clear reports of what went wrong

### Non-Goals:
- No cryptographic proofs or Merkle trees
- No formal verification or mathematical proofs
- No complex state machines or graph analysis
- No performance overhead in production mode
