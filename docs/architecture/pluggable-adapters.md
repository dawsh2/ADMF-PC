# ADMF-PC: Adaptive Decision Making Framework - Protocol Components

ADMF-PC is a quantitative trading system built around a simple idea: trading strategies should be configured, not programmed. The system uses container isolation and event-driven architecture to run thousands of backtests in parallel without interference.

## What Makes This Different

### Zero-Code Strategy Development
Instead of writing Python code for each strategy, you describe what you want in a YAML file:

```yaml
strategies:
  - name: "momentum_tech"
    type: "momentum"
    fast_period: 10
    slow_period: 30
    symbols: ["AAPL", "GOOGL", "MSFT"]
```

The system handles all the implementation details: data loading, indicator calculation, signal generation, risk management, and execution simulation.

### Execution Model Selection
Different phases use optimal execution models:

- **Parameter Discovery**: Actor-based for 5000+ parallel evaluations
- **Signal Replay**: Lightweight functions for fast optimization
- **Backtesting**: Full containers for complete isolation
- **Live Trading**: Maximum reliability containers with restart policies

```yaml
execution_models:
  parameter_discovery:
    model: "actors"
    config:
      max_actors: 5000
  signal_replay:
    model: "functions"
    config:
      timeout_seconds: 60
  live_trading:
    model: "containers"
    config:
      isolation_level: "full"
      restart_policy: "always"
```

### Multi-Phase Optimization
Traditional systems test parameters one at a time. ADMF-PC runs systematic multi-phase workflows:

1. **Parameter Discovery**: Test thousands of parameter combinations
2. **Regime Analysis**: Find which parameters work best in different market conditions
3. **Ensemble Optimization**: Combine strategies optimally (100x faster using signal replay)
4. **Validation**: Test the complete system on fresh data

### Schema Evolution and Event Correlation
Events automatically handle versioning and maintain correlation chains:

```python
# Events track their relationships
signal_event.causation_id = indicator_event.event_id  # What caused this
signal_event.correlation_id = workflow_session_id     # Session tracking

# Schema migration happens automatically
v1_signal = TradingSignal(schema_version="1.0.0", ...)
v2_signal = schema_registry.migrate(v1_signal, "2.0.0")  # Automatic upgrade
```

This enables:
- **Full audit trails**: Every decision is traceable to its source
- **Backward compatibility**: Old components work with new event versions  
- **Debugging support**: Follow event chains to understand system behavior
- **Compliance**: Complete record of all trading decisions

## Architecture Overview

### Protocol + Composition Design
Instead of complex inheritance hierarchies, everything is built from composable components:

```python
# Mix any components together
strategy_components = [
    SimpleMovingAverage(20),              # Built-in indicator
    ta.RSI(period=14),                    # TA-Lib indicator  
    sklearn.RandomForestClassifier(),     # ML model
    custom_function,                      # Your function
    ThirdPartyLibrary.signal()            # External library
]
```

No inheritance required. Components just need to follow simple protocols (interfaces).

### Semantic Event-Driven Communication

### Strongly Typed Events
Components communicate through semantic events that carry rich type information:

```python
@dataclass
class TradingSignal(SemanticEventBase):
    symbol: str = ""
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD" 
    strength: float = 0.0  # 0.0 to 1.0
    regime_context: Optional[str] = None
    correlation_id: str = ""  # Links related events
    causation_id: Optional[str] = None  # Parent event
```

### Clean Event Flow with Transformation
```
Market Data → Indicators → Strategies → Risk Management → Execution
```

Each step transforms the previous step's semantic events into new event types. The system automatically handles:
- **Type validation**: Events are checked for correctness
- **Correlation tracking**: Related events are linked together
- **Schema evolution**: New event versions work with old components
- **Automatic transformation**: Events convert between compatible types

### Four Organizational Patterns
The system supports different ways of organizing strategies based on your research focus:

- **Strategy-First**: Traditional approach, good for comparing individual strategies
- **Classifier-First**: Organize by market regime, good for adaptive strategies  
- **Risk-First**: Organize by risk profile, good for capital allocation research
- **Portfolio-First**: Organize by asset allocation, good for institutional approaches

Same underlying system, different configuration styles.

## Container Patterns

### Full Backtest Container
Complete strategy evaluation from market data to execution:
- Loads historical data
- Calculates all indicators
- Runs classifier logic (if used)
- Executes strategy logic
- Applies risk management
- Simulates order execution

### Signal Replay Container
Fast optimization using pre-captured signals:
- Reads signals from previous runs
- Applies different ensemble weights
- Skips expensive calculation steps
- 100x faster than full backtests

### Signal Generation Container
Pure research without execution simulation:
- Generates and analyzes signals
- Calculates signal quality metrics
- Stores signals for later replay
- Good for strategy research

## Workspace Management

Every workflow creates a structured workspace on disk:

```
./results/workflow_123/
├── signals/              # Trading signals (streaming format)
├── performance/          # Backtest results (structured data)
├── analysis/             # Research outputs (regime analysis, etc.)
├── metadata/             # Configuration and logs
└── checkpoints/          # Resumability data
```

### Benefits of File-Based Communication
- **Natural checkpointing**: Resume from any failed step
- **Debugging**: Inspect all intermediate results
- **Parallel processing**: Multiple processes can read/write safely
- **Manual intervention**: Edit results between phases if needed

## Flexible Communication Adapters

Different organizational patterns need different communication flows. The system uses pluggable adapters:

- **Pipeline**: Linear data flow (A → B → C)
- **Hierarchical**: Parent-child relationships (regime classifier → risk profiles)
- **Broadcast**: One-to-many (data feed → multiple strategies)
- **Selective**: Conditional routing based on content

All configured through YAML, no code changes needed.

## Multi-Phase Workflow Example

Here's how a complete optimization workflow actually runs:

### Phase 1: Parameter Discovery
```bash
Coordinator: "Test these 1000 parameter combinations"
Optimizer: "Here are the combinations to test"
Coordinator: "Run 1000 backtests, save signals to workspace"
Backtester: "Executed all tests, signals saved"
```

### Phase 2: Regime Analysis  
```bash
Coordinator: "Analyze which parameters work best per regime"
Optimizer: "Reading performance data, analyzing by regime..."
Optimizer: "Bull markets: use params X, Bear markets: use params Y"
```

### Phase 3: Ensemble Optimization
```bash
Coordinator: "Find optimal strategy weights using signal replay"
Optimizer: "Testing 500 weight combinations..."
Backtester: "Replaying signals with different weights (very fast)"
Optimizer: "Bull markets: 70% strategy A, Bear markets: 30% strategy A"
```

### Phase 4: Validation
```bash
Coordinator: "Test complete adaptive system on fresh data"
Backtester: "Running full test with regime-switching parameters and weights"
Backtester: "Validation complete, system works as expected"
```

## Key Technical Innovations

### Semantic Event System
**Type-safe communication** with automatic validation and transformation:
- Events carry rich metadata (correlation IDs, causation chains, regime context)
- Schema evolution allows backward compatibility
- Automatic type conversion between compatible event types
- Full audit trail of all decisions for compliance

### Signal Replay Optimization  
**100x faster ensemble optimization** through captured signal streams:
- Parameter discovery phase captures all trading signals
- Ensemble optimization replays signals without recalculation
- Eliminates expensive indicator computation in optimization loops
- Enables rapid testing of thousands of weight combinations

### Execution Model Flexibility
**Optimal execution model per phase**:
- Actors for massive parallelization (parameter sweeps)
- Functions for lightweight tasks (signal replay)
- Containers for full isolation (backtesting, live trading)
- Automatic selection based on workload characteristics

### Workspace-Aware Communication
**File-based coordination** between workflow phases:
- Semantic events serialize to structured workspace files
- Multi-phase workflows coordinate through file handoffs
- Natural checkpointing and resumability
- Complete intermediate result inspection

## What This Enables

### Systematic Strategy Research
Instead of testing strategies one at a time, run systematic research programs that automatically:
- Test parameter spaces exhaustively
- Find regime-specific optimal parameters
- Discover ensemble combinations
- Validate on out-of-sample data

### Reproducible Results
Given the same configuration file, the system produces identical results every time. No hidden state, no race conditions, no surprises.

### Massive Parallelization
Run thousands of backtests simultaneously without interference. Each test is completely isolated from others.

### Flexible Experimentation
Change organizational patterns, communication flows, and optimization strategies through configuration changes, not code rewrites.

## Getting Started

1. **Define your strategy** in a YAML configuration file
2. **Specify your data** (symbols, date ranges, sources)
3. **Choose your workflow type** (backtest, optimization, analysis)
4. **Run the system** and find results in the workspace
5. **Inspect intermediate files** to understand what happened
6. **Iterate** by modifying configuration and re-running

No programming required for most use cases. The system handles the complexity while exposing simple, declarative interfaces.

## Design Philosophy

The system is built around several core principles:

- **Configuration over code**: Describe what you want, not how to do it
- **Composition over inheritance**: Mix components freely without framework lock-in
- **Semantic events over raw data**: Rich, typed communication with correlation
- **Execution model flexibility**: Optimal runtime per workload (actors/containers/functions)
- **Isolation over sharing**: Prevent interference through strong boundaries
- **Files over databases**: Simple, inspectable, portable state management
- **Schema evolution**: Backward compatible changes without breaking existing components
- **Protocols over frameworks**: Work with any compatible components

This creates a system that's both powerful enough for sophisticated research and simple enough for everyday use, while maintaining the semantic richness needed for complex trading workflows.
