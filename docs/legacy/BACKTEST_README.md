# ADMF-PC Backtest Architecture & Standardization

## Overview
The ADMF-PC (Adaptive Decision Making Framework - Protocol Components) system implements a sophisticated multi-symbol trading architecture with nested container hierarchy, protocol-based design, and complete separation of concerns between orchestration and execution.

## Architectural Motivations

### Why Standardization Matters
In complex trading systems, inconsistencies in backtest setup can lead to:
- **Irreproducible results**: Different initialization orders or missing steps
- **State leakage**: Data or configuration bleeding between test runs  
- **Hidden biases**: Execution path variations that affect outcomes
- **Debugging nightmares**: Each backtest might fail differently

### Our Solution: Universal Container Pattern
Every execution instance (backtest or live) is created through an identical, enforced pattern that:
1. **Guarantees consistency**: Same configuration always produces same container structure
2. **Prevents errors**: No way to skip steps or initialize incorrectly
3. **Enables massive parallelization**: Run 1000s of backtests confidently
4. **Simplifies debugging**: Every container follows the same blueprint

## Primary Container Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              COORDINATOR                                      │
│  • Orchestrates workflows across all containers                               │
│  • Manages phase transitions and data flow                                    │
│  • Handles checkpointing and resumability                                     │
│  • Creates and manages top-level containers (Backtest, Optimization, Live)    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          BACKTEST CONTAINER                                   │
│  (Encapsulates entire backtest process - ensures clean creation/disposal)     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────┐                                                          │
│  │ Historical Data│                                                          │
│  │    Streamer    │─────────┐                                               │
│  └────────────────┘         │                                               │
│                             │                                               │
│  ┌──────────────────────────▼───────────────────────────────────────────┐   │
│  │                    Shared Indicator Architecture                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              Indicator Hub (Shared Computation)              │    │   │
│  │  │  • MA, RSI, ATR, etc. computed once from streamed data      │    │   │
│  │  │  • Caches results for efficiency                             │    │   │
│  │  │  • Emits indicator events to downstream consumers            │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                             │                                         │   │
│  │                             │ Indicator Events                       │   │
│  │                             ▼                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                              │
│                                │ Indicator Events                            │
│                ┌───────────────┴────────────────┐                            │
│                │                                │                            │
│  ┌─────────────▼─────────────────┐  ┌──────────▼─────────────────────┐     │
│  │   HMM Classifier Container    │  │  Pattern Classifier Container   │     │
│  ├───────────────────────────────┤  ├────────────────────────────────┤     │
│  │ ┌───────────────────────────┐ │  │ ┌────────────────────────────┐ │     │
│  │ │   HMM Regime Classifier   │ │  │ │ Pattern Regime Classifier  │ │     │
│  │ │ • Consumes indicator data │ │  │ │ • Consumes indicator data  │ │     │
│  │ │ • Determines regime state │ │  │ │ • Detects market patterns  │ │     │
│  │ │ • Bull/Bear/Neutral       │ │  │ │ • Breakout/Range/Trending │ │     │
│  │ └────────────┬──────────────┘ │  │ └─────────────┬──────────────┘ │     │
│  │              │                │  │               │                │     │
│  │              ▼                │  │               ▼                │     │
│  │ ┌───────────────────────────┐ │  │ ┌────────────────────────────┐ │     │
│  │ │    Risk Container Pool    │ │  │ │    Risk Container Pool     │ │     │
│  │ │     (Subcontainers)       │ │  │ │     (Subcontainers)        │ │     │
│  │ ├───────────────────────────┤ │  │ ├────────────────────────────┤ │     │
│  │ │ ┌───────────────────────┐ │ │  │ │ ┌────────────────────────┐ │ │     │
│  │ │ │ Conservative Risk     │ │ │  │ │ │ Balanced Risk          │ │ │     │
│  │ │ │ Container             │ │ │  │ │ │ Container              │ │ │     │
│  │ │ │ • Max 2% per position │ │ │  │ │ │ • Max 3% per position │ │ │     │
│  │ │ │ • 10% total exposure  │ │ │  │ │ │ • 20% total exposure  │ │ │     │
│  │ │ │ ┌─────────────────┐   │ │ │  │ │ │ ┌──────────────────┐  │ │ │     │
│  │ │ │ │ Portfolio A     │   │ │ │  │ │ │ │ Portfolio C      │  │ │ │     │
│  │ │ │ │ $50K allocation │   │ │ │  │ │ │ │ $100K allocation │  │ │ │     │
│  │ │ │ │ ┌─────────────┐ │   │ │ │  │ │ │ │ ┌──────────────┐ │  │ │ │     │
│  │ │ │ │ │ Momentum    │ │   │ │ │  │ │ │ │ │ Pattern      │ │  │ │ │     │
│  │ │ │ │ │ Strategy    │ │   │ │ │  │ │ │ │ │ Strategy     │ │  │ │ │     │
│  │ │ │ │ │ AAPL(40%)   │ │   │ │ │  │ │ │ │ │ SPY(60%)     │ │  │ │ │     │
│  │ │ │ │ │ GOOGL(30%)  │ │   │ │ │  │ │ │ │ │ QQQ(40%)     │ │  │ │ │     │
│  │ │ │ │ │ MSFT(30%)   │ │   │ │ │  │ │ │ │ └──────────────┘ │  │ │ │     │
│  │ │ │ │ └─────────────┘ │   │ │ │  │ │ │ └──────────────────┘  │ │ │     │
│  │ │ │ └─────────────────┘   │ │ │  │ │ └────────────────────────┘ │ │     │
│  │ │ │ ┌─────────────────┐   │ │ │  │ │                            │ │     │
│  │ │ │ │ Portfolio B     │   │ │ │  │ │ ┌────────────────────────┐ │ │     │
│  │ │ │ │ $30K allocation │   │ │ │  │ │ │ Aggressive Risk        │ │ │     │
│  │ │ │ │ ┌─────────────┐ │   │ │ │  │ │ │ Container              │ │ │     │
│  │ │ │ │ │ Mean Rev    │ │   │ │ │  │ │ │ • Max 5% per position │ │ │     │
│  │ │ │ │ │ Strategy    │ │   │ │ │  │ │ │ • 30% total exposure  │ │ │     │
│  │ │ │ │ │ SPY(100%)   │ │   │ │ │  │ │ │ ┌──────────────────┐  │ │ │     │
│  │ │ │ │ └─────────────┘ │   │ │ │  │ │ │ │ Portfolio D      │  │ │ │     │
│  │ │ │ └─────────────────┘   │ │ │  │ │ │ │ $200K allocation │  │ │ │     │
│  │ │ └───────────────────────┘ │ │  │ │ │ │ ┌──────────────┐ │  │ │ │     │
│  │ │                           │ │  │ │ │ │ │ Breakout     │ │  │ │ │     │
│  │ │ ┌───────────────────────┐ │ │  │ │ │ │ │ Strategy     │ │  │ │ │     │
│  │ │ │ Moderate Risk         │ │ │  │ │ │ │ │ BTC(50%)     │ │  │ │ │     │
│  │ │ │ Container             │ │ │  │ │ │ │ │ ETH(30%)     │ │  │ │ │     │
│  │ │ │ • Max 3% per position │ │ │  │ │ │ │ │ SOL(20%)     │ │  │ │ │     │
│  │ │ │ • 15% total exposure  │ │ │  │ │ │ │ └──────────────┘ │  │ │ │     │
│  │ │ │ ┌─────────────────┐   │ │ │  │ │ │ └──────────────────┘  │ │ │     │
│  │ │ │ │ Portfolio E     │   │ │ │  │ │ └────────────────────────┘ │ │     │
│  │ │ │ │ $75K allocation │   │ │ │  │ └────────────────────────────┘ │     │
│  │ │ │ │ ┌─────────────┐ │   │ │ │  │                                │     │
│  │ │ │ │ │ Pairs Trade │ │   │ │ │  │                                │     │
│  │ │ │ │ │ Strategy    │ │   │ │ │  │                                │     │
│  │ │ │ │ │ (Multi-sub) │ │   │ │ │  │                                │     │
│  │ │ │ │ │ XLE/XLF     │ │   │ │ │  │                                │     │
│  │ │ │ │ └─────────────┘ │   │ │ │  │                                │     │
│  │ │ │ └─────────────────┘   │ │ │  │                                │     │
│  │ │ └───────────────────────┘ │ │  │ Output: Signals & Performance │     │
│  │ └───────────────────────────┘ │  └────────────────────────────────┘     │
│  │                               │                                          │
│  │ Output: Signals & Performance │                                          │
│  └───────────────────────────────┘                                          │
│                │                                │                            │
│                └────────────────┬───────────────┘                            │
│                                 │                                            │
│  ┌──────────────────────────────▼───────────────────────────────────────┐   │
│  │                         Backtest Engine                               │   │
│  │  • Executes trades based on aggregated signals                       │   │
│  │  • Manages portfolio state and position tracking                     │   │
│  │  • Calculates performance metrics                                    │   │
│  │  • Handles multi-symbol data alignment                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Results & Performance Storage                    │   │
│  │  • Streams results to disk during execution                          │   │
│  │  • Maintains in-memory cache of top performers only                  │   │
│  │  • Provides aggregated metrics to Coordinator                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Simplified Data Flow View

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          BACKTEST CONTAINER                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│                        Historical Data Stream                                 │
│                                 │                                             │
│                                 ▼                                             │
│                    ┌────────────────────────┐                                │
│                    │    Indicator Hub       │                                │
│                    │ (Compute once, share)  │                                │
│                    └───────────┬────────────┘                                │
│                                │                                              │
│                    ┌───────────┼───────────┐                                 │
│                    ▼           ▼           ▼                                 │
│            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                  │
│            │ Classifier 1│ │ Classifier 2│ │ Classifier N│                  │
│            ├─────────────┤ ├─────────────┤ ├─────────────┤                  │
│            │┌───────────┐│ │┌───────────┐│ │┌───────────┐│                  │
│            ││Risk & Port││ ││Risk & Port││ ││Risk & Port││                  │
│            ││Container  ││ ││Container  ││ ││Container  ││                  │
│            ││┌─────────┐││ ││┌─────────┐││ ││┌─────────┐││                  │
│            │││Strategy │││ │││Strategy │││ │││Strategy │││                  │
│            ││└─────────┘││ ││└─────────┘││ ││└─────────┘││                  │
│            │└───────────┘│ │└───────────┘│ │└───────────┘│                  │
│            └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                  │
│                   │               │               │                          │
│                   └───────────────┼───────────────┘                          │
│                                   ▼                                          │
│                          ┌─────────────────┐                                │
│                          │ Backtest Engine │                                │
│                          └────────┬────────┘                                │
│                                   ▼                                          │
│                          ┌─────────────────┐                                │
│                          │     Results     │                                │
│                          └─────────────────┘                                │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Event Flow: Signals → Orders → Fills

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Event Flow Within Backtest Container                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Strategies           Risk & Portfolio Containers      Backtest Engine       │
│      │                           │                          │                 │
│      │    SIGNAL Event           │                          │                 │
│      │  (Buy AAPL, strength=0.8) │                          │                 │
│      ├──────────────────────────►│                          │                 │
│      │                           │                          │                 │
│      │                    Risk Assessment:                  │                 │
│      │                    - Check position limits           │                 │
│      │                    - Check exposure limits           │                 │
│      │                    - Apply position sizing           │                 │
│      │                    - May VETO signal                 │                 │
│      │                           │                          │                 │
│      │                           │     ORDER Event          │                 │
│      │                           │  (Buy AAPL, 100 shares) │                 │
│      │                           ├─────────────────────────►│                 │
│      │                           │                          │                 │
│      │                           │                   Execute Order:           │
│      │                           │                   - Check market data      │
│      │                           │                   - Apply slippage         │
│      │                           │                   - Update positions       │
│      │                           │                          │                 │
│      │                           │      FILL Event          │                 │
│      │                           │◄─────────────────────────┤                 │
│      │                           │  (Filled @ $150.25)      │                 │
│      │                           │                          │                 │
│      │                 Update Risk & Portfolio:             │                 │
│      │                    - Track positions                 │                 │
│      │                    - Update exposure                 │                 │
│      │                    - Risk metrics                    │                 │
│      │                    - Portfolio state                │                 │
│      │                           │                          │                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Points:
1. **Strategies** generate SIGNAL events based on market data and indicators
2. **Risk & Portfolio Containers** convert SIGNAL events to ORDER events (or veto them)
3. **Backtest Engine** executes ORDER events and generates FILL events
4. **Risk & Portfolio Containers** update portfolio state based on FILL events

## Container Lifecycle Management

### Backtest Container Benefits

1. **Complete Isolation**: All backtest components live within a single container
2. **Clean Lifecycle**: Single point of creation and disposal
3. **Resource Management**: Track and limit resource usage for entire backtest
4. **State Consistency**: Ensures all components share same backtest context
5. **Easy Parallelization**: Run multiple backtests in parallel containers

### Container Creation Flow

```
Coordinator.start_backtest()
        │
        ▼
┌─────────────────────┐
│ Create Backtest     │
│ Container           │
└──────────┬──────────┘
           │
           ▼
    Within Container:
    1. Create Data Streamer
    2. Create Indicator Hub
    3. Create Classifiers
    4. Create Risk & Portfolio Subcontainers
    5. Create Strategies
    6. Create Backtest Engine
    7. Wire up event buses
           │
           ▼
┌─────────────────────┐
│ Execute Backtest    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Stream Results      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Dispose Container   │
│ (Clean up all       │
│  subcomponents)     │
└─────────────────────┘
```

## Multi-Symbol Architecture Details

### Container Hierarchy: Classifier/Risk/Portfolio/Strategy
```
HMM Classifier Container
├── Conservative Risk Container
│   ├── Risk Settings: Max 2% per position, 10% total exposure
│   ├── Portfolio A ($50K allocation)
│   │   └── Momentum Strategy
│   │       ├── AAPL: 40% of portfolio allocation
│   │       ├── GOOGL: 30% of portfolio allocation
│   │       └── MSFT: 30% of portfolio allocation
│   └── Portfolio B ($30K allocation)
│       └── Mean Reversion Strategy
│           └── SPY: 100% of portfolio allocation
│
├── Moderate Risk Container  
│   ├── Risk Settings: Max 3% per position, 15% total exposure
│   └── Portfolio E ($75K allocation)
│       └── Pairs Trading Strategy (Multi-sub strategy)
│           ├── Long XLE Sub-strategy
│           └── Short XLF Sub-strategy
│
└── Aggressive Risk Container
    ├── Risk Settings: Max 5% per position, 30% total exposure
    └── Portfolio D ($200K allocation)
        └── Breakout Strategy
            ├── BTC: 50% of portfolio allocation
            ├── ETH: 30% of portfolio allocation
            └── SOL: 20% of portfolio allocation

Pattern Classifier Container
└── Balanced Risk Container
    ├── Risk Settings: Max 3% per position, 20% total exposure
    └── Portfolio C ($100K allocation)
        └── Pattern Strategy
            ├── SPY: 60% of portfolio allocation
            └── QQQ: 40% of portfolio allocation
```

### Key Benefits of Four-Level Hierarchy

**1. Classifier Level Grouping**
- All components using same regime detection share indicator computations
- HMM classifier computes regime once → shared by all nested components
- Efficient for testing multiple strategies under same market regime detection

**2. Risk Level Separation**  
- Multiple risk profiles can operate under same classifier
- Conservative vs Aggressive risk can use same HMM regime signals
- Independent position sizing and exposure limits per risk container

**3. Portfolio Level Isolation**
- Multiple portfolios can share same risk settings but different allocations
- Portfolio A: $50K momentum focus, Portfolio B: $30K mean reversion focus
- Both use same Conservative risk limits but independent capital allocation

**4. Strategy Level Flexibility**
- Strategies can contain sub-strategies for complex execution
- Pairs trading strategy contains Long XLE + Short XLF sub-strategies
- Cointegration strategies can combine multiple assets in single strategy

**5. Automatic Indicator Inference**
- System scans all nested Strategy and Classifier components
- Automatically determines required indicators (RSI, MA, ATR, etc.)
- Indicator Hub computes each indicator once per bar for all consumers

**6. Scalability Examples**
- Simple: Single MA crossover = Minimal nesting (just Strategy)
- Complex: Multi-classifier, multi-risk, multi-portfolio, multi-strategy = Full nesting
- Same architecture handles both extremes efficiently

## Key Architectural Principles

### 1. Protocol-Based Design
- No inheritance, only protocol implementations
- Clean contracts between components
- Easy testing and mocking

### 2. Event-Driven Communication
- Unidirectional event flow
- No circular dependencies
- Clear data lineage

### 3. Container Isolation
- Each container has its own event bus scope
- Independent configuration namespaces
- Resource tracking per container

### 4. Nested Subcontainers
- Risk & Portfolio containers inherit classifier context
- Strategies inherit risk & portfolio context
- Clean hierarchical data flow

### 5. Separation of Concerns
- **Coordinator**: Orchestration and workflow management
- **Backtest Container**: Lifecycle and resource management  
- **Backtest Engine**: Order execution, fills, and performance calculation
- **Classifiers**: Regime detection and context provision
- **Risk & Portfolio Containers**: Signal → Order conversion, position sizing, exposure management, portfolio state tracking
- **Strategies**: Signal generation only (no position management)

## Why This Architecture Matters for Complex Workflows

### Supporting Multi-Phase Optimization
The Coordinator can orchestrate complex workflows knowing each backtest is identical:

```
Phase 1: Parameter Grid Search
├── Create 1000 backtest containers
├── Each tests different parameter combinations
├── All follow identical creation pattern
└── Results aggregated by Coordinator

Phase 2: Ensemble Weight Optimization  
├── Create 100 backtest containers
├── Each tests different weight combinations
├── Uses SIGNALS from Phase 1 (not raw market data)
└── Simplified container pattern (see Signal Replay Architecture)

Phase 3: Out-of-Sample Validation
├── Create final backtest containers
├── Test on holdout data
├── Identical setup ensures fair comparison
└── No training data leakage possible
```

### Phase Management and Data Flow

### Phase Transitions
```
INITIALIZATION → DATA_PREPARATION → COMPUTATION → VALIDATION → AGGREGATION
      │                │                 │             │            │
      ▼                ▼                 ▼             ▼            ▼
Create Container  Load Data      Execute Trades  Validate Results  Store
& Components      Setup Streams   Run Strategies  Check Metrics    Output
```

### Cross-Phase Data Flow
1. Each phase publishes completion events
2. Next phase subscribes to previous phase events
3. Coordinator manages transitions
4. Results stream to disk continuously
5. Only top performers kept in memory

## Container Naming Strategy

Format: `{container_type}_{phase}_{classifier}_{risk_profile}_{timestamp}`

Examples:
- `backtest_phase1_hmm_conservative_20240115_143052`
- `backtest_phase2_pattern_aggressive_20240115_145523`

This enables:
- Easy identification of container purpose
- Tracking across optimization phases
- Debugging and monitoring
- Result aggregation by type

## Standardized Execution Model

### Universal Backtest Creation Pattern
Every execution instance follows this exact pattern:

```
ExecutionEngine.create_instance(config) → Standardized Container
                                         ├── Data Streamer (Historical/Live)
                                         ├── Indicator Hub
                                         ├── Classifier Containers
                                         │   └── Risk Subcontainers
                                         │       └── Strategies
                                         └── Execution Engine (Backtest/Live)

The ONLY differences between modes:
- Train/Test: Historical data streamer + Backtest execution engine
- Live: Real-time data streamer + Live execution engine (broker connection)
- Everything else remains IDENTICAL
```

## Signal Replay Architecture (Ensemble Optimization)

For Phase 2 ensemble weight optimization, we use a simplified container architecture that replays signals from Phase 1:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     SIGNAL REPLAY CONTAINER                                   │
│  (Simplified architecture for ensemble weight optimization)                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────┐                                                          │
│  │  Signal Log    │                                                          │
│  │   Streamer     │─────────┐                                               │
│  │ (From Phase 1) │         │                                               │
│  └────────────────┘         │                                               │
│                             │                                               │
│                             ▼                                               │
│            ┌────────────────────────────────┐                               │
│            │     Ensemble Optimizer         │                               │
│            │ • Reads strategy signals       │                               │
│            │ • NO indicator computation     │                               │
│            │ • NO classifier needed         │                               │
│            │ • Tests weight combinations    │                               │
│            └────────────────┬───────────────┘                               │
│                             │                                               │
│                             ▼                                               │
│            ┌────────────────────────────────┐                               │
│            │    Risk & Portfolio Container  │                               │
│            │ • Applies ensemble weights     │                               │
│            │ • Combines signals → Orders    │                               │
│            │ • Same risk limits apply      │                               │
│            └────────────────┬───────────────┘                               │
│                             │                                               │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Backtest Engine                               │   │
│  │  • Executes orders (same as before)                                  │   │
│  │  • Tracks performance of ensemble                                    │   │
│  │  • Much faster - no indicator/classifier computation                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences from Standard Architecture:
1. **Input**: Signal logs instead of market data
2. **No Indicator Hub**: Signals already computed in Phase 1
3. **No Classifiers**: Regime context embedded in signals
4. **Simplified Flow**: Signal → Ensemble → Risk → Execution
5. **Speed**: 10-100x faster than recomputing everything
6. **Flexible Testing**: Quickly experiment with different risk and classifier settings

### Signal Log Format:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "signals": [
    {
      "strategy_id": "momentum_001",
      "symbol": "AAPL",
      "direction": "BUY",
      "strength": 0.85,
      "regime_context": "bull",
      "classifier": "hmm"
    },
    {
      "strategy_id": "mean_rev_002",
      "symbol": "SPY",
      "direction": "SELL",
      "strength": 0.65,
      "regime_context": "neutral",
      "classifier": "hmm"
    }
  ]
}
```

## Signal Generation Architecture (Analysis & Optimization)

For signal analysis, MAE/MFE optimization, and classifier tuning, we use an even simpler pattern:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION CONTAINER                                │
│  (Pure signal generation for analysis - NO execution)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────┐                                                          │
│  │ Historical Data│                                                          │
│  │    Streamer    │─────────┐                                               │
│  └────────────────┘         │                                               │
│                             │                                               │
│                             ▼                                               │
│            ┌────────────────────────────────┐                               │
│            │        Indicator Hub           │                               │
│            │  (Standard computation)        │                               │
│            └────────────────┬───────────────┘                               │
│                             │                                               │
│                             ▼                                               │
│            ┌────────────────────────────────┐                               │
│            │    Classifier Container        │                               │
│            │  (Test different classifiers)  │                               │
│            └────────────────┬───────────────┘                               │
│                             │                                               │
│                             ▼                                               │
│            ┌────────────────────────────────┐                               │
│            │    Strategy Container          │                               │
│            │  • Generate signals only       │                               │
│            │  • NO risk assessment         │                               │
│            │  • NO order generation        │                               │
│            └────────────────┬───────────────┘                               │
│                             │                                               │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Signal Analysis Engine                             │   │
│  │  • Capture all signals with metadata                                 │   │
│  │  • Calculate signal quality metrics (win rate, MAE/MFE)             │   │
│  │  • Analyze signal correlation across strategies                      │   │
│  │  • Store signals for later replay                                    │   │
│  │  • NO execution, NO portfolio tracking                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Use Cases for Signal Generation:

1. **Risk & Portfolio Parameter Optimization**
   - Generate signals once
   - Test different risk & portfolio settings offline
   - Find optimal position sizing without re-running strategies

2. **MAE/MFE Analysis**
   - Track Maximum Adverse/Favorable Excursion for each signal
   - Optimize stop-loss and take-profit levels
   - No need for full backtest execution

3. **Classifier Comparison**
   - Run same strategies under different classifiers
   - Compare signal quality across regime detection methods
   - Identify which classifier provides best signals

4. **Signal Processing Research**
   - Test signal filtering algorithms
   - Analyze signal correlation and overlap
   - Develop signal combination strategies

5. **Fast Iteration**
   - 100x faster than full backtest
   - Focus purely on signal quality
   - Rapid experimentation cycle

### Signal Analysis Output Format:
```json
{
  "signal_id": "sig_20240115_103000_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "strategy_id": "momentum_001",
  "symbol": "AAPL",
  "direction": "BUY",
  "strength": 0.85,
  "price_at_signal": 150.00,
  "metadata": {
    "regime": "bull",
    "classifier": "hmm",
    "indicators": {
      "rsi": 65,
      "ma_cross": true
    }
  },
  "forward_returns": {
    "1_bar": 0.002,
    "5_bar": 0.015,
    "10_bar": 0.008,
    "mae": -0.003,
    "mfe": 0.022
  }
}
```

## Three-Pattern Backtest Architecture

The Coordinator can select from three standardized patterns:

```
1. FULL BACKTEST (Standard)
   Market Data → Indicators → Classifiers → Strategies → Risk & Portfolio → Execution
   Use: Complete strategy evaluation, live trading

2. SIGNAL REPLAY (Ensemble)  
   Signal Logs → Ensemble Weights → Risk & Portfolio → Execution
   Use: Ensemble optimization, risk & portfolio parameter tuning

3. SIGNAL GENERATION (Analysis)
   Market Data → Indicators → Classifiers → Strategies → Analysis
   Use: Signal quality research, MAE/MFE optimization, classifier comparison
```

### Coordinator Pattern Selection:
```python
class BacktestCoordinator:
    def create_backtest(self, pattern: BacktestPattern, config: Dict):
        if pattern == BacktestPattern.FULL:
            return BacktestContainerFactory.create_instance(config)
        elif pattern == BacktestPattern.SIGNAL_REPLAY:
            return SignalReplayContainerFactory.create_instance(config)
        elif pattern == BacktestPattern.SIGNAL_GENERATION:
            return SignalGenerationContainerFactory.create_instance(config)
```

### Coordinator vs Backtest Responsibilities

**Coordinator (High-Level Orchestration):**
- Decides WHAT to backtest (configurations)
- Manages optimization workflows
- Handles parameter expansion/validation
- Orchestrates multi-phase processes:
  - Phase 1: Grid search for regime-optimized parameters
  - Phase 2: Ensemble weight optimization using optimal rules
  - Phase 3: Out-of-sample validation
- Manages train/test/live cycles
- Handles checkpointing and recovery

**Backtest Container (Standardized Execution):**
- Creates components in exact same order every time
- Ensures consistent wiring of event buses
- Guarantees no state leakage between instances
- Provides identical execution paths
- Handles all low-level execution details

### Example Workflow Separation

```
Coordinator says:                    Backtest Container does:
"Run grid search with               → Creates 1000 identical containers
 - HMM classifier                   → Each with same component structure
 - Conservative risk & portfolio    → Each with different parameters
 - 10 momentum variants             → Each fully isolated
 - 100 parameter combos"            → Each following standard pattern
```

## Benefits of This Architecture

1. **Clean Isolation**: Backtest container ensures complete isolation
2. **Reusability**: Same architecture works for optimization, live trading
3. **Scalability**: Easy to parallelize multiple backtests
4. **Maintainability**: Clear boundaries and responsibilities
5. **Testability**: Each component can be tested in isolation
6. **Flexibility**: Easy to add new classifiers, risk profiles, strategies
7. **Resource Efficiency**: Shared computation in Indicator Hub
8. **Observability**: Clear event flow for monitoring and debugging
9. **Reproducibility**: Identical setup process eliminates variations
10. **State Safety**: No leakage between backtest instances

### Reproducibility Guarantee

By standardizing the container creation process, we ensure:
- Same configuration → Same container structure → Same results
- No hidden dependencies or setup variations
- Easy to debug: every container follows the same pattern
- Seamless transition from backtest to live (just swap data source and execution engine)

## Implementation Requirements

### Container Factory Pattern
```python
class BacktestContainerFactory:
    """Enforces standardized container creation"""
    
    @staticmethod
    def create_instance(config: BacktestConfig) -> BacktestContainer:
        # ALWAYS creates components in this exact order
        container = BacktestContainer(config.container_id)
        
        # 1. Data layer
        container.add_component(DataStreamer(config.data_config))
        container.add_component(IndicatorHub(config.indicator_config))
        
        # 2. Classifier layer (with risk & portfolio subcontainers)
        for classifier_config in config.classifiers:
            classifier = container.create_subcontainer(classifier_config)
            for risk_config in classifier_config.risk_containers:
                risk_container = classifier.create_subcontainer(risk_config)
                for strategy_config in risk_config.strategies:
                    risk_container.add_component(Strategy(strategy_config))
        
        # 3. Execution layer
        container.add_component(BacktestEngine(config.execution_config))
        
        # 4. Wire event buses (always in same order)
        container.wire_event_flows()
        
        return container


class SignalReplayContainerFactory:
    """Simplified container for signal replay (ensemble optimization)"""
    
    @staticmethod
    def create_instance(config: SignalReplayConfig) -> SignalReplayContainer:
        # Much simpler creation pattern
        container = SignalReplayContainer(config.container_id)
        
        # 1. Signal replay layer (replaces market data + indicators + classifiers)
        container.add_component(SignalLogStreamer(config.signal_log_path))
        
        # 2. Ensemble optimizer (combines signals with weights)
        container.add_component(EnsembleOptimizer(config.weight_config))
        
        # 3. Risk & Portfolio container (same as before - converts signals to orders)
        risk_container = container.create_subcontainer(config.risk_config)
        
        # 4. Execution layer (same backtest engine)
        container.add_component(BacktestEngine(config.execution_config))
        
        # 5. Simplified wiring
        container.wire_signal_replay_flows()
        
        return container


class SignalGenerationContainerFactory:
    """Signal generation only - for analysis and optimization"""
    
    @staticmethod
    def create_instance(config: SignalGenConfig) -> SignalGenContainer:
        # Execution-free pattern
        container = SignalGenContainer(config.container_id)
        
        # 1. Standard data and indicator layers
        container.add_component(DataStreamer(config.data_config))
        container.add_component(IndicatorHub(config.indicator_config))
        
        # 2. Classifier layer (for testing different classifiers)
        for classifier_config in config.classifiers:
            classifier = container.create_subcontainer(classifier_config)
            # Add strategies directly - no risk & portfolio containers needed
            for strategy_config in config.strategies:
                classifier.add_component(Strategy(strategy_config))
        
        # 3. Signal analysis engine (instead of backtest engine)
        container.add_component(SignalAnalysisEngine(config.analysis_config))
        
        # 4. Wire for signal flow only
        container.wire_signal_generation_flows()
        
        return container
```

### Configuration-Driven Architecture
The entire system is driven by configuration, ensuring:
- **Declarative setup**: Describe what you want, not how to build it
- **Version control friendly**: Track all experiment configurations
- **Easy experimentation**: Change parameters without touching code
- **Audit trail**: Every backtest configuration is preserved

### State Isolation Guarantees
Each container provides complete isolation:
- **Independent event buses**: No cross-container event leakage
- **Separate memory spaces**: No shared mutable state
- **Resource limits**: CPU/memory caps per container
- **Clean disposal**: All resources freed on container destruction

## Real-World Benefits

### For Development
- **Faster debugging**: Issues are consistent across runs
- **Easier testing**: Mock any component, container structure stays same
- **Confident refactoring**: Changes can't break initialization order

### For Research
- **True A/B testing**: Only parameters change, not execution
- **Reproducible papers**: Share config files to replicate exactly
- **Fair comparisons**: All strategies run in identical environments

### For Production
- **Seamless deployment**: Same container pattern for backtest and live
- **Risk management**: Guaranteed risk & portfolio container enforcement
- **Audit compliance**: Every execution fully traceable

### For Scale
- **Parallel execution**: Spin up 10,000 identical containers
- **Resource management**: Each container has defined limits
- **Cloud ready**: Containers map directly to cloud instances

## Dual Architecture Benefits

### Why Two Container Types?

Having both standard and signal-replay containers provides:

1. **Phase 1 Optimization Speed**: Standard containers compute everything once
   - Generate signals for all strategy variants
   - Store signals for reuse
   - One pass through market data

2. **Phase 2 Optimization Speed**: Signal-replay containers are blazing fast
   - No indicator recomputation
   - No classifier recomputation  
   - Direct signal → ensemble → execution pipeline
   - 10-100x faster than Phase 1

3. **Storage Efficiency**: 
   - Phase 1: Store signals (small) instead of full market data processing
   - Phase 2: Read signals sequentially, minimal memory footprint

### Example Performance Gains

```
Traditional Approach (everything recomputed every time):
- Strategy Optimization: 1000 params × 5 years × 10ms = 14 hours
- Risk & Portfolio Optimization: 100 risk & portfolio settings × 5 years × 10ms = 1.4 hours  
- Ensemble Optimization: 100 weights × 5 years × 10ms = 1.4 hours
- Total: 16.8 hours

Our Three-Pattern Approach:
- Signal Generation: 1000 params × 5 years × 8ms = 11.2 hours (no execution)
- Risk & Portfolio Optimization: 100 risk & portfolio × 5 years × 0.1ms = 2.5 minutes (signal replay)
- Ensemble Optimization: 100 weights × 5 years × 0.1ms = 2.5 minutes (signal replay)
- Total: 11.3 hours

Speedup: 16.8 hours → 11.3 hours = 33% faster overall
Risk & Portfolio/Ensemble phases: 2.8 hours → 5 minutes = 33x faster!
```

### Workflow Benefits

1. **Iterative Development**: Test signal quality before implementing execution
2. **Rapid Experimentation**: Change risk & portfolio settings without regenerating signals  
3. **Classifier A/B Testing**: Compare classifiers using same signal set
4. **MAE/MFE Optimization**: Analyze excursions without full backtest
5. **Correlation Analysis**: Study signal overlap across strategies