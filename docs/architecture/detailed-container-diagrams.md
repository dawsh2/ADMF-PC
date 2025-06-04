# Detailed Container Diagrams

## Overview

This document provides comprehensive visualizations of ADMF-PC's container architecture, showing how different container types are structured, how they interact, and how they compose into complex trading systems.

## Multi-Phase Coordinator Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              COORDINATOR                                      │
│  Sequential Phase Processing - Each phase uses different container types:     │
│                                                                               │
│  Phase 1: Parameter Discovery     Phase 2: Regime Analysis                   │
│  ┌─────────────────────────┐     ┌─────────────────────────┐                 │
│  │ Multiple Full Backtest  │ ────▶ │ Analysis Container      │                 │
│  │ Containers              │     │ (No execution - reads   │                 │
│  │ (Complete execution)    │     │  results, finds optimal │                 │
│  └─────────────────────────┘     │  parameters per regime) │                 │
│                                  └─────────────────────────┘                 │
│                                                                               │
│  Phase 3: Ensemble Optimization   Phase 4: Validation                        │
│  ┌─────────────────────────┐     ┌─────────────────────────┐                 │
│  │ Signal Replay           │ ────▶ │ Final Full Backtest     │                 │
│  │ Containers              │     │ Container               │                 │
│  │ (No indicators/data -   │     │ (Complete execution     │                 │
│  │  replays saved signals) │     │  with optimizations)    │                 │
│  └─────────────────────────┘     └─────────────────────────┘                 │
│                                                                               │
│  Each phase type requires different computational patterns and components     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Standard Backtest Container Architecture

This detailed container represents **one backtest instance** used in phases like Parameter Discovery or Validation. Other phase types (Analysis, Signal Replay) use different, simpler container architectures optimized for their specific computational patterns.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD BACKTEST CONTAINER                                │
│  (Used in Parameter Discovery & Validation phases)                            │
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
│  │  │               Feature Hub (Shared Computation)                │    │   │
│  │  │  • MA, RSI, ATR, etc. computed once from streamed data      │    │   │
│  │  │  • Caches results for efficiency                             │    │   │
│  │  │  • Emits feature events to downstream consumers               │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                             │                                         │   │
│  │                             │ Feature Events                         │   │
│  │                             ▼                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                              │
│                                │ Feature Events                              │
│                ┌───────────────┴────────────────┐                            │
│                │                                │                            │
│  ┌─────────────▼─────────────────┐  ┌──────────▼─────────────────────┐     │
│  │   HMM Classifier Container    │  │  Pattern Classifier Container   │     │
│  ├───────────────────────────────┤  ├────────────────────────────────┤     │
│  │ ┌───────────────────────────┐ │  │ ┌────────────────────────────┐ │     │
│  │ │   HMM Regime Classifier   │ │  │ │ Pattern Regime Classifier  │ │     │
│  │ │ • Consumes feature data    │ │  │ │ • Consumes feature data     │ │     │
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
│  │ │ └───────────────────────┘ │ │  │ └────────────────────────────┘ │     │
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

## Container Type Specialization

### Analysis Container (Phase 2)
```
┌──────────────────────────────────────────────────────────────────┐
│                      ANALYSIS CONTAINER                          │
│  (Statistical analysis only - no trading execution)              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────┐    ┌─────────────────────────┐     │
│  │   Performance Reader    │    │    Regime Analyzer      │     │
│  │  • Loads trial results  │───▶│ • Classifies regimes    │     │
│  │  • Aggregates metrics   │    │ • Groups by performance │     │
│  └─────────────────────────┘    └───────────┬─────────────┘     │
│                                              │                   │
│  ┌─────────────────────────┐                │                   │
│  │  Statistical Analyzer   │◀───────────────┘                   │
│  │  • Correlation analysis │                                    │
│  │  • Significance testing │                                    │
│  │  • Parameter sensitivity│                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │    Report Generator     │                                    │
│  │  • Optimal parameters   │                                    │
│  │  • Regime performance   │                                    │
│  │  • Recommendations      │                                    │
│  └─────────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Signal Replay Container (Phase 3)
```
┌──────────────────────────────────────────────────────────────────┐
│                   SIGNAL REPLAY CONTAINER                        │
│  (Replays saved signals - no indicator computation)             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────┐    ┌─────────────────────────┐     │
│  │    Signal Reader        │    │   Weight Optimizer      │     │
│  │  • Loads signal logs    │───▶│ • Tests weight combos   │     │
│  │  • Streams in order     │    │ • Applies to signals    │     │
│  └─────────────────────────┘    └───────────┬─────────────┘     │
│                                              │                   │
│                                              ▼                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                  Risk & Execution Layer                   │   │
│  │  • Same risk management as full backtest                  │   │
│  │  • Same execution engine                                  │   │
│  │  • 10-100x faster (no feature computation)                │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Nested Container Hierarchies

```
┌─────────────────────────────────────────────────────────────┐
│                    Classifier Container                     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 Risk Container                          ││
│  │  ┌─────────────────────────────────────────────────────┐││
│  │  │             Portfolio Container                     │││
│  │  │  ┌─────────────────────────────────────────────────┐│││
│  │  │  │           Strategy Container                    ││││
│  │  │  │     (Individual strategy logic)                ││││
│  │  │  └─────────────────────────────────────────────────┘│││
│  │  │     (Portfolio allocation logic)                   │││
│  │  └─────────────────────────────────────────────────────┘││
│  │     (Risk management logic)                            ││
│  └─────────────────────────────────────────────────────────┘│
│     (Regime classification logic)                           │
└─────────────────────────────────────────────────────────────┘
```

### Hierarchy Benefits:
- **Modularity**: Each level handles specific concerns
- **Reusability**: Inner containers can be used independently
- **Testing**: Each level can be tested in isolation
- **Flexibility**: Mix and match different implementations

## Event Flow Through Container Hierarchy

```
Market Data
    │
    ▼
Feature Hub ────────────────────┐
    │                           │
    ▼                           ▼
Classifier Container        Classifier Container
    │                           │
    ├─ Regime: BULL            ├─ Regime: BEAR
    │                           │
    ▼                           ▼
Risk Container              Risk Container
    │                           │
    ├─ Conservative            ├─ Aggressive
    │                           │
    ▼                           ▼
Portfolio Container         Portfolio Container
    │                           │
    ├─ Strategy A             ├─ Strategy B
    ├─ Strategy B             ├─ Strategy C
    │                           │
    ▼                           ▼
SIGNALS                     SIGNALS
    │                           │
    └───────────┬───────────────┘
                │
                ▼
         Execution Engine
```

## Container Communication Patterns

### Parent-Child Event Bridging
```
┌─────────────────────────────────────────────────────┐
│              Parent Container                       │
│  ┌─────────────────────────────────────────────┐   │
│  │           Parent Event Bus                   │   │
│  │  ┌───────────────┐  ┌───────────────┐       │   │
│  │  │ Subscription  │  │ Subscription  │       │   │
│  │  │ to Child A    │  │ to Child B    │       │   │
│  │  └───────┬───────┘  └───────┬───────┘       │   │
│  └───────────┼──────────────────┼───────────────┘   │
│              │                  │                   │
│  ┌───────────▼────────┐ ┌──────▼────────────┐     │
│  │   Child A Event    │ │  Child B Event    │     │
│  │       Bus          │ │      Bus          │     │
│  │  ┌─────────────┐   │ │  ┌─────────────┐  │     │
│  │  │ Component 1 │   │ │  │ Component 2 │  │     │
│  │  └─────────────┘   │ │  └─────────────┘  │     │
│  └────────────────────┘ └───────────────────┘     │
└─────────────────────────────────────────────────────┘
```

### Rules:
- Children never know about parents
- Parents can subscribe to child events
- No lateral communication between siblings
- Events flow up the hierarchy

## Container Factory Patterns

```python
class ContainerFactory:
    """Creates containers based on phase requirements"""
    
    def create_container(self, phase_type: str, config: Dict) -> Container:
        if phase_type == "backtest":
            return self._create_backtest_container(config)
        elif phase_type == "analysis":
            return self._create_analysis_container(config)
        elif phase_type == "signal_replay":
            return self._create_replay_container(config)
        elif phase_type == "signal_generation":
            return self._create_generation_container(config)
            
    def _create_backtest_container(self, config: Dict) -> Container:
        """Full backtest with all components"""
        container = BacktestContainer()
        
        # Add all layers
        container.add_component(DataStreamer(config['data']))
        container.add_component(FeatureHub(config['features']))
        container.add_component(StrategyLayer(config['strategies']))
        container.add_component(RiskManager(config['risk']))
        container.add_component(ExecutionEngine())
        
        return container
```

## Container Composition Examples

### Simple Single-Strategy Container
```
┌──────────────────────────────────┐
│      Simple Container            │
│  ┌────────────────────────────┐  │
│  │     Data Streamer          │  │
│  └──────────┬─────────────────┘  │
│             │                    │
│  ┌──────────▼─────────────────┐  │
│  │    Moving Average Strategy │  │
│  └──────────┬─────────────────┘  │
│             │                    │
│  ┌──────────▼─────────────────┐  │
│  │    Fixed Risk Manager      │  │
│  └──────────┬─────────────────┘  │
│             │                    │
│  ┌──────────▼─────────────────┐  │
│  │    Execution Engine        │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

### Complex Multi-Regime Container
```
┌────────────────────────────────────────────────────┐
│            Complex Container                       │
│  ┌──────────────────────────────────────────────┐  │
│  │           Feature Hub (20+ features)         │  │
│  └────────────────┬─────────────────────────────┘  │
│                   │                                │
│  ┌────────────────┴─────────────────┐             │
│  │                                  │             │
│  ▼                                  ▼             │
│ HMM Classifier                Pattern Classifier  │
│  │                                  │             │
│  ├─ 3 Risk Profiles                ├─ 2 Risk     │
│  │                                  │   Profiles  │
│  ├─ 5 Portfolios each              ├─ 3 Portfolios│
│  │                                  │   each      │
│  └─ 10 Strategies total            └─ 6 Strategies│
│                                        total      │
│              │                          │         │
│              └────────┬─────────────────┘         │
│                       │                           │
│           ┌───────────▼────────────┐              │
│           │   Signal Aggregator    │              │
│           └───────────┬────────────┘              │
│                       │                           │
│           ┌───────────▼────────────┐              │
│           │   Ensemble Optimizer   │              │
│           └───────────┬────────────┘              │
│                       │                           │
│           ┌───────────▼────────────┐              │
│           │   Execution Engine     │              │
│           └────────────────────────┘              │
└────────────────────────────────────────────────────┘
```

## Performance Characteristics by Container Type

| Container Type | Components | Memory Usage | CPU Usage | Typical Use |
|----------------|------------|--------------|-----------|-------------|
| Full Backtest | All layers | High (1-2GB) | High | Initial testing |
| Signal Replay | Risk + Exec | Low (100MB) | Low | Optimization |
| Analysis Only | Statistics | Medium (500MB) | Medium | Post-processing |
| Signal Generation | Strategy only | Low (200MB) | Medium | Research |

## Container Design Principles

1. **Single Responsibility**: Each container type optimized for one purpose
2. **Composability**: Containers can be nested and combined
3. **Isolation**: No shared state between containers
4. **Standardization**: All containers follow same lifecycle patterns
5. **Efficiency**: Only include necessary components for the task

## Summary

The container architecture provides:

- **Flexibility**: Different container types for different needs
- **Performance**: Optimized components for each phase
- **Modularity**: Clear separation of concerns
- **Scalability**: Parallel execution of independent containers
- **Maintainability**: Standardized patterns across all types

Understanding these diagrams helps in:
- Choosing the right container type for each task
- Composing complex systems from simple components
- Debugging by understanding component relationships
- Optimizing performance by selecting minimal containers





---------
Alternative Diagram:

# Improved Standard Backtest Container Structure

## Logical Container Hierarchy

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD BACKTEST CONTAINER                                │
│  (Main orchestrator - manages component lifecycle and data flow)              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          DATA CONTAINER                                 │  │
│  │  (Responsible for all data ingestion and preprocessing)                 │  │
│  │                                                                         │  │
│  │  ┌────────────────┐    ┌─────────────────────────────────────────────┐ │  │
│  │  │ Historical Data│    │          Indicator Container                 │ │  │
│  │  │    Streamer    │───▶│  (Encapsulates all feature computation)       │ │  │
│  │  └────────────────┘    │                                             │ │  │
│  │                        │  ┌─────────────────────────────────────┐    │ │  │
│  │                        │  │         Feature Hub                 │    │ │  │
│  │                        │  │  • MA, RSI, ATR, MACD, etc.        │    │ │  │
│  │                        │  │  • Manages computation order       │    │ │  │
│  │                        │  │  • Handles dependencies            │    │ │  │
│  │                        │  │  • Caches results efficiently      │    │ │  │
│  │                        │  └─────────────────────────────────────┘    │ │  │
│  │                        └─────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                     │
│                                         │ Enriched Market Data                │
│                                         ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      CLASSIFICATION CONTAINER                          │  │
│  │  (Responsible for market regime identification and signal generation)  │  │
│  │                                                                         │  │
│  │  ┌───────────────────────────────┐  ┌─────────────────────────────────┐ │  │
│  │  │   HMM Classifier Container    │  │  Pattern Classifier Container   │ │  │
│  │  │                               │  │                                 │ │  │
│  │  │ ┌───────────────────────────┐ │  │ ┌─────────────────────────────┐ │ │  │
│  │  │ │   HMM Regime Engine       │ │  │ │  Pattern Recognition Engine │ │ │  │
│  │  │ │ • State transition logic  │ │  │ │ • Chart pattern detection   │ │ │  │
│  │  │ │ • Bull/Bear/Neutral       │ │  │ │ • Breakout/Range/Trending   │ │ │  │
│  │  │ │ • Confidence scoring      │ │  │ │ • Pattern strength scoring  │ │ │  │
│  │  │ └─────────────┬─────────────┘ │  │ └──────────────┬──────────────┘ │ │  │
│  │  │               │               │  │                │                │ │  │
│  │  │               ▼               │  │                ▼                │ │  │
│  │  │ ┌───────────────────────────┐ │  │ ┌─────────────────────────────┐ │ │  │
│  │  │ │    Signal Generator       │ │  │ │    Signal Generator         │ │ │  │
│  │  │ │ • Regime-based signals    │ │  │ │ • Pattern-based signals     │ │ │  │
│  │  │ │ • Timing and strength     │ │  │ │ • Entry/exit points         │ │ │  │
│  │  │ └───────────────────────────┘ │  │ └─────────────────────────────┘ │ │  │
│  │  └───────────────────────────────┘  └─────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                     │
│                                         │ Raw Signals                         │
│                                         ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       EXECUTION CONTAINER                              │  │
│  │  (Responsible for risk management, portfolio allocation, and execution) │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    Signal Processing Layer                      │   │  │
│  │  │  • Signal aggregation and conflict resolution                   │   │  │
│  │  │  • Signal strength normalization                                │   │  │
│  │  │  • Timing coordination between classifiers                      │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                         │                               │  │
│  │                                         ▼                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                     Risk Management Layer                       │   │  │
│  │  │                                                                 │   │  │
│  │  │  ┌───────────────────────┐    ┌───────────────────────────┐    │   │  │
│  │  │  │ Conservative Risk     │    │ Balanced Risk             │    │   │  │
│  │  │  │ Profile Container     │    │ Profile Container         │    │   │  │
│  │  │  │                       │    │                           │    │   │  │
│  │  │  │ • Max 2% per position │    │ • Max 3% per position     │    │   │  │
│  │  │  │ • 10% total exposure  │    │ • 20% total exposure      │    │   │  │
│  │  │  │ • Stop loss: 1.5%     │    │ • Stop loss: 2%           │    │   │  │
│  │  │  └───────────────────────┘    └───────────────────────────┘    │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                         │                               │  │
│  │                                         ▼                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                   Portfolio Management Layer                    │   │  │
│  │  │                                                                 │   │  │
│  │  │  ┌─────────────────────────┐    ┌─────────────────────────┐    │   │  │
│  │  │  │ Portfolio A Container   │    │ Portfolio B Container   │    │   │  │
│  │  │  │ $50K allocation         │    │ $100K allocation        │    │   │  │
│  │  │  │                         │    │                         │    │   │  │
│  │  │  │ ┌─────────────────────┐ │    │ ┌─────────────────────┐ │    │   │  │
│  │  │  │ │ Strategy Container  │ │    │ │ Strategy Container  │ │    │   │  │
│  │  │  │ │                     │ │    │ │                     │ │    │   │  │
│  │  │  │ │ Momentum Strategy   │ │    │ │ Pattern Strategy    │ │    │   │  │
│  │  │  │ │ • AAPL (40%)        │ │    │ │ • SPY (60%)         │ │    │   │  │
│  │  │  │ │ • GOOGL (30%)       │ │    │ │ • QQQ (40%)         │ │    │   │  │
│  │  │  │ │ • MSFT (30%)        │ │    │ │                     │ │    │   │  │
│  │  │  │ └─────────────────────┘ │    │ └─────────────────────┘ │    │   │  │
│  │  │  └─────────────────────────┘    └─────────────────────────┘    │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                         │                               │  │
│  │                                         ▼                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                      Execution Engine                          │   │  │
│  │  │  • Order generation and management                              │   │  │
│  │  │  • Position tracking and reconciliation                        │   │  │
│  │  │  • Performance calculation and reporting                       │   │  │
│  │  │  • Transaction cost modeling                                   │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                     │
│                                         ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        RESULTS CONTAINER                               │  │
│  │  (Responsible for data persistence and performance tracking)           │  │
│  │                                                                         │  │
│  │  • Real-time performance streaming to disk                             │  │
│  │  • In-memory caching of key metrics                                    │  │
│  │  • Aggregated results for coordinator reporting                        │  │
│  │  • Trade logs and position history                                     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Improvements

### 1. Clear Separation of Concerns
- **Data Container**: Handles all data ingestion and preprocessing
- **Classification Container**: Focuses purely on regime identification and signal generation
- **Execution Container**: Manages risk, portfolios, and trade execution
- **Results Container**: Handles persistence and reporting

### 2. Proper Encapsulation
- **Indicator Container** is now properly contained within the Data Container
- Each container has a single, well-defined responsibility
- Clear data flow between containers

### 3. Logical Component Hierarchy
```
Main Container
├── Data Container
│   ├── Data Streamer
│   └── Indicator Container
│       └── Indicator Hub
├── Classification Container
│   ├── HMM Classifier Container
│   │   ├── Regime Engine
│   │   └── Signal Generator
│   └── Pattern Classifier Container
│       ├── Pattern Engine
│       └── Signal Generator
├── Execution Container
│   ├── Signal Processing Layer
│   ├── Risk Management Layer
│   ├── Portfolio Management Layer
│   └── Execution Engine
└── Results Container
```

### 4. Benefits of This Structure

**Modularity**: Each container can be developed, tested, and optimized independently

**Reusability**: Containers can be composed differently for different phases
- Analysis phase: Data + Classification containers only
- Signal Replay phase: Execution + Results containers only

**Performance**: Clear boundaries allow for targeted optimization
- Data Container can focus on efficient indicator computation
- Execution Container can optimize for low-latency trade processing

**Maintainability**: Clear ownership and responsibility boundaries

**Testing**: Each container can be unit tested with mock dependencies

## Container Interface Contracts

```python
class DataContainer:
    def get_enriched_data(self) -> EnrichedMarketData:
        """Returns market data with all computed features"""
        
class ClassificationContainer:
    def process_signals(self, data: EnrichedMarketData) -> List[Signal]:
        """Generates trading signals from market data"""
        
class ExecutionContainer:
    def execute_signals(self, signals: List[Signal]) -> ExecutionResults:
        """Processes signals through risk management and execution"""
        
class ResultsContainer:
    def store_results(self, results: ExecutionResults) -> None:
        """Persists execution results and performance metrics"""
```

This structure provides much clearer boundaries and makes it easier to understand what each component is responsible for.