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
