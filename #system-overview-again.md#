# ADMF-PC System Architecture Overview

## Introduction

ADMF-PC is an event-driven quantitative trading and research platform designed from the ground up to enable massively parallel backtesting while maintaining perfect state isolation. The system allows researchers to test thousands of strategy variations simultaneously without worrying about data contamination or race conditions.

At its heart, the system embraces three core principles:
1. **Event-driven communication** - Everything is an event, enabling complete observability
2. **Container-based isolation** - State is isolated where needed, shared where beneficial  
3. **Declarative configuration** - Complex workflows defined in YAML, no coding required

## The Foundation: Events and Isolation

### Event System Architecture

The event system is the nervous system of ADMF-PC. Unlike traditional trading systems where components directly call each other, every interaction in ADMF-PC happens through events. This provides several crucial benefits:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Event-Driven Architecture                     │
│                                                                  │
│  Producer           Event Bus            Consumer               │
│  ┌─────────┐       ┌─────────┐         ┌─────────┐            │
│  │Component│──────▶│  Events │────────▶│Component│            │
│  └─────────┘       └─────────┘         └─────────┘            │
│                          │                                      │
│                          ▼                                      │
│                    ┌───────────┐                               │
│                    │ Observer  │                                │
│                    │ (Tracing) │                                │
│                    └───────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

Every event carries rich metadata:
- **Event Type**: What happened (BAR, SIGNAL, ORDER, FILL, etc.)
- **Payload**: The actual data
- **Source ID**: Which component created it
- **Container ID**: Which container owns the component
- **Correlation ID**: Links related events (e.g., all events for one trade)
- **Timestamp**: When it occurred

### Container Isolation Model

Containers are the fundamental unit of isolation in ADMF-PC. Each container:
- Owns its own event bus
- Manages its own state
- Controls its component lifecycle
- Decides its own tracing policy

```
┌─────────────────────────────────────────────────────────────────┐
│                        Container Hierarchy                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Root Container                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Symbol    │  │   Symbol    │  │   Symbol    │     │   │
│  │  │ Container   │  │ Container   │  │ Container   │     │   │
│  │  │  (SPY_1m)   │  │  (QQQ_1m)   │  │  (IWM_1m)   │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │         │                │                │              │   │
│  │         └────────────────┴────────────────┘              │   │
│  │                          │                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ Portfolio 1 │  │ Portfolio 2 │  │ Portfolio N │     │   │
│  │  │   (Isolated │  │   (Isolated │  │   (Isolated │     │   │
│  │  │  Event Bus) │  │  Event Bus) │  │  Event Bus) │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

This isolation ensures:
- **No state contamination**: Each portfolio tracks its own positions independently
- **Clean parallelization**: Portfolios can run on different CPU cores
- **Surgical debugging**: Issues are contained to specific containers

## Communication: Routers and Adapters

While containers are isolated, they still need to communicate. This is where routers come in. Routers are specialized components that wire up event flows between containers.

### Router Types

```
┌─────────────────────────────────────────────────────────────────┐
│                      Router Patterns                            │
│                                                                  │
│  1. Pipe Router (1:1)                                          │
│     Source ──────▶ Router ──────▶ Target                      │
│                                                                  │
│  2. Broadcast Router (1:N)                                      │
│                    ┌──────▶ Target 1                           │
│     Source ──────▶ Router ├──────▶ Target 2                   │
│                    └──────▶ Target N                           │
│                                                                  │
│  3. Filter Router (Conditional)                                │
│     Source ──────▶ Router ──?───▶ Target                      │
│                      │                                          │
│                    Filter                                       │
│                    Logic                                        │
└─────────────────────────────────────────────────────────────────┘
```

Routers enable sophisticated communication patterns:
- **Feature Distribution**: One feature container broadcasts to many strategies
- **Signal Aggregation**: Multiple strategies feed into one portfolio
- **Selective Routing**: Orders routed based on symbol or portfolio ID

## Building Blocks: Components

Components are the workers of the system. They come in two flavors:

### Stateless Components
Pure functions that process inputs to outputs:
- **Strategies**: `(features, bar) → signal`
- **Risk Validators**: `(order, portfolio_state) → approved_order`
- **Execution Models**: `(order, market_data) → fill`

Stateless components can be shared across containers and run in parallel without synchronization.

### Stateful Components  
Maintain internal state and live inside containers:
- **Data Streamers**: Track position in data files
- **Feature Calculators**: Cache computed indicators
- **Portfolio Trackers**: Maintain positions and P&L

## The Magic: Topologies

Topologies are pre-built patterns that wire together containers, components, and routers for specific workflows. Think of them as blueprints for common trading system architectures.

### Full Backtest Topology

The complete pipeline from data to results:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Full Backtest Topology                       │
│                                                                  │
│  Data Containers          Feature Processing                    │
│  ┌────────────┐           ┌────────────────┐                   │
│  │  SPY Data  │────BAR───▶│Feature Container│                  │
│  └────────────┘           └────────────────┘                   │
│                                   │                             │
│                              FEATURES                           │
│                                   │                             │
│                    ┌──────────────┴───────────────┐             │
│                    ▼                              ▼             │
│            ┌───────────────┐             ┌───────────────┐     │
│            │Strategy Pool  │             │Strategy Pool  │     │
│            │(Momentum, MA) │             │(Mean Rev, BB) │     │
│            └───────────────┘             └───────────────┘     │
│                    │                              │             │
│                 SIGNALS                       SIGNALS          │
│                    │                              │             │
│                    ▼                              ▼             │
│            ┌───────────────┐             ┌───────────────┐     │
│            │  Portfolio 1  │             │  Portfolio 2  │     │
│            │ (Aggressive)  │             │(Conservative) │     │
│            └───────────────┘             └───────────────┘     │
│                    │                              │             │
│                 ORDERS                        ORDERS           │
│                    │                              │             │
│                    └──────────────┬───────────────┘             │
│                                   ▼                             │
│                          ┌─────────────────┐                    │
│                          │ Risk Validation │                    │
│                          └─────────────────┘                    │
│                                   │                             │
│                            VALIDATED_ORDERS                     │
│                                   │                             │
│                          ┌─────────────────┐                    │
│                          │Execution Engine │                    │
│                          └─────────────────┘                    │
│                                   │                             │
│                                FILLS                            │
│                                   │                             │
│                           (Broadcast back to                    │
│                            all portfolios)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Signal Generation Topology

Optimized for capturing signals without execution:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Signal Generation Topology                      │
│                                                                  │
│  Data → Features → Strategies → Signal Storage                 │
│                                       │                         │
│                                   Event Trace                   │
│                                    (Signals)                    │
└─────────────────────────────────────────────────────────────────┘
```

This topology:
- Skips portfolio and execution components
- Captures all signals to disk with event tracing
- Enables massive parameter sweeps efficiently

### Signal Replay Topology  

Replays previously generated signals:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Signal Replay Topology                        │
│                                                                  │
│  Signal Storage → Portfolios → Risk → Execution                │
│        │                                                        │
│   Event Replay                                                  │
│   (From Disk)                                                   │
└─────────────────────────────────────────────────────────────────┘
```

Benefits:
- Test different portfolio configurations without regenerating signals
- 10x+ faster than full backtesting
- Perfect for ensemble weight optimization

## Orchestration: Sequences and Workflows

### Sequences

Sequences define patterns of execution across time windows. They handle the lifecycle of containers across multiple runs.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Sequence Patterns                          │
│                                                                  │
│  Single Pass:     ████████████████████████                     │
│                   └─────── One Run ──────┘                     │
│                                                                  │
│  Train/Test:      ████████████ │ ████████                      │
│                   └── Train ──┘ └─ Test ─┘                     │
│                                                                  │
│  Walk Forward:    ████│███ ████│███ ████│███                   │
│                   Train│Test Train│Test Train│Test              │
│                                                                  │
│  Monte Carlo:     ████ ████ ████ ████ ████                     │
│                   └─── Multiple Random Runs ───┘                │
└─────────────────────────────────────────────────────────────────┘
```

Key insight: Each execution window gets fresh containers, ensuring no state leakage between time periods.

### Workflows

Workflows combine multiple phases to achieve complex objectives:

```
┌─────────────────────────────────────────────────────────────────┐
│              Adaptive Ensemble Workflow Example                 │
│                                                                  │
│  Phase 1: Grid Search (Signal Generation)                      │
│  ┌─────────────────────────────────────┐                       │
│  │ • Run all parameter combinations     │                       │
│  │ • Walk-forward validation            │                       │
│  │ • Save signals to disk               │                       │
│  └─────────────────────────────────────┘                       │
│                     │                                           │
│                     ▼                                           │
│  Phase 2: Regime Analysis                                       │
│  ┌─────────────────────────────────────┐                       │
│  │ • Load Phase 1 results               │                       │
│  │ • Identify regime patterns           │                       │
│  │ • Find best params per regime        │                       │
│  └─────────────────────────────────────┘                       │
│                     │                                           │
│                     ▼                                           │
│  Phase 3: Ensemble Optimization (Signal Replay)                │
│  ┌─────────────────────────────────────┐                       │
│  │ • Replay signals from best strategies│                       │
│  │ • Optimize ensemble weights          │                       │
│  │ • Walk-forward validation            │                       │
│  └─────────────────────────────────────┘                       │
│                     │                                           │
│                     ▼                                           │
│  Phase 4: Final Validation (Full Backtest)                     │
│  ┌─────────────────────────────────────┐                       │
│  │ • Run on out-of-sample data         │                       │
│  │ • Adaptive regime switching          │                       │
│  │ • Complete performance report        │                       │
│  └─────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Architectural Decisions

### 1. Event Tracing as the Metrics System

Rather than maintaining separate systems for metrics and event history, ADMF-PC uses event tracing as the single source of truth for performance metrics.

```
┌─────────────────────────────────────────────────────────────────┐
│              Event Tracing for Metrics                          │
│                                                                  │
│  Traditional Approach:          ADMF-PC Approach:               │
│  ┌─────────────────┐           ┌─────────────────┐             │
│  │   Event Bus     │           │   Event Bus     │             │
│  └────────┬────────┘           └────────┬────────┘             │
│           │                              │                      │
│     ┌─────┴─────┐                 ┌─────┴─────┐                │
│     ▼           ▼                 ▼           ▼                │
│  Metrics    Event Log          Event Tracer                    │
│  System     (Separate)         (Unified)                       │
│                                    │                            │
│                              ┌─────┴─────┐                      │
│                              ▼           ▼                      │
│                          Metrics    Full History                │
│                          (Derived)  (If Needed)                 │
└─────────────────────────────────────────────────────────────────┘
```

Benefits:
- Single source of truth
- Metrics always consistent with event history
- Can reconstruct any metric from events
- Flexible retention policies save memory

### 2. Container-Level Event Tracing Configuration

Each container can have its own tracing policy:

```yaml
# Portfolio container: Only track trades
portfolio_1:
  event_tracing: [POSITION_OPEN, POSITION_CLOSE, FILL]
  retention_policy: trade_complete  # Delete events after trade closes
  
# Analysis container: Track everything  
analysis:
  event_tracing: ALL
  retention_policy: all  # Keep everything
  storage: disk  # Spill to disk if needed
```

This granular control enables:
- Memory-efficient production deployments
- Detailed debugging when needed
- Compliance and audit trails where required

### 3. Smart Data Handling with Bar Indexing

The system uses an innovative approach to data management that eliminates data copying:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Bar Indexing System                          │
│                                                                  │
│  Traditional: Copy data to each consumer                        │
│  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐             │
│  │ Data │────▶│Copy 1│────▶│Strat1│     │Memory│             │
│  │ File │────▶│Copy 2│────▶│Strat2│     │Usage │             │
│  └──────┘────▶│Copy 3│────▶│Strat3│     │ 3x! │             │
│                └──────┘     └──────┘     └──────┘             │
│                                                                  │
│  ADMF-PC: Share data with index tracking                       │
│  ┌──────┐     ┌──────┐     ┌──────┐                           │
│  │ Data │────▶│Shared│◀────│Index1│──▶Strategy 1             │
│  │ File │     │Memory│◀────│Index2│──▶Strategy 2             │
│  └──────┘     └──────┘◀────│Index3│──▶Strategy 3             │
│                             └──────┘                           │
│                                                                  │
│  Each strategy maintains only its position in the data!        │
└─────────────────────────────────────────────────────────────────┘
```

This approach:
- Reduces memory usage by 10-100x for multi-strategy backtests
- Enables true zero-copy data sharing
- Maintains perfect isolation (each strategy has its own index)
- Scales to massive universes

### 4. Hierarchical Result Storage

Results are organized to match the execution hierarchy:

```
./results/
└── workflow_20240106_123456/
    ├── phase1_grid_search/
    │   ├── window_1/
    │   │   ├── portfolio_1_results.json
    │   │   ├── portfolio_2_results.json
    │   │   └── traces/
    │   │       ├── portfolio_1_trace.jsonl
    │   │       └── portfolio_2_trace.jsonl
    │   └── window_2/
    │       └── ...
    ├── phase2_regime_analysis/
    │   └── regime_results.json
    └── phase3_final_validation/
        └── final_report.html
```

This structure:
- Makes results easy to navigate
- Enables partial re-runs
- Supports distributed execution
- Facilitates debugging

## Performance and Scalability

### Memory Management Strategy

The system implements a multi-tiered approach to memory management:

1. **Streaming Calculations**: Metrics computed on-the-fly, not stored
2. **Smart Retention**: Keep only what's needed (e.g., open positions)
3. **Disk Spillover**: Automatic spillover for large traces
4. **Container Limits**: Per-container memory budgets

### Parallelization Model

```
┌─────────────────────────────────────────────────────────────────┐
│                   Parallelization Levels                        │
│                                                                  │
│  Level 1: Portfolio Parallelism                                │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐              │
│  │ Port 1 │  │ Port 2 │  │ Port 3 │  │ Port N │              │
│  │ Core 1 │  │ Core 2 │  │ Core 3 │  │ Core N │              │
│  └────────┘  └────────┘  └────────┘  └────────┘              │
│                                                                  │
│  Level 2: Strategy Parallelism (within portfolio)              │
│  ┌─────────────────────────────┐                               │
│  │      Portfolio Container     │                               │
│  │  ┌─────┐ ┌─────┐ ┌─────┐   │                               │
│  │  │Str1 │ │Str2 │ │Str3 │   │ (Async processing)           │
│  │  └─────┘ └─────┘ └─────┘   │                               │
│  └─────────────────────────────┘                               │
│                                                                  │
│  Level 3: Window Parallelism (walk-forward)                    │
│  Window1 ──▶ Machine 1                                         │
│  Window2 ──▶ Machine 2                                         │
│  Window3 ──▶ Machine 3                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Conclusion

ADMF-PC represents a thoughtful approach to building scalable quantitative trading systems. By embracing events as the fundamental communication primitive, containers as the isolation boundary, and declarative configuration as the user interface, it achieves a rare combination of power and usability.

The system's key innovations:
- **Unified event/metrics system** eliminates redundancy
- **Container isolation** enables massive parallelization
- **Smart data handling** reduces memory usage dramatically
- **Declarative workflows** make complex research accessible

These architectural decisions position ADMF-PC to handle everything from simple single-strategy backtests to complex multi-phase ensemble optimizations across thousands of parameter combinations.