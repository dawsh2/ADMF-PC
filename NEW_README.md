# ADMF-PC Event Flow Architecture: The Next Evolution

## Executive Summary

The core idea of the system is using containers to isolate state where necassary in a parallelized backtest, and leveraging that the data transformation pipeline is linear: Data-->BARS-->Features-->FEATURE/BAR-->Strategy & Classifiers-->SIGNAL & ClassifierChange-->Portfolio-->ORDER_REQUEST-->Risk-->ORDER-->Execution (applying stateless functional slippage/commission etc)-->FILL-->PORTFOLIO_UPATE (more or less). It's clearly an event driven system, and to communicate with otherwise isolated containers, we had to devise an adapter, these are under src/core/communication (but since we've simplified the topology into a universal linear flow, we may not need these more extravagent adapters now, but still need a way to communicate from root event bus to containers). We take these primitives - containers, events and cross-communication - and use them to create standardized topologies that dynamicaly generate and wire up the proper system for whatever we're testing (e.g, a multi-asset, multi-strategy, multi-portfolio backtest -- since it uses the same fundemental topology, it should be as easy to wire up as a single backtest). We then leverage this concept to use standardize topologies sequentially to simplify the handling of e.g, train test splits, while maintaing reproducability through the standardization of the process and no state leakage since containers are destroyed after every sequence. At a higher level we combine these sequenced topologies into what we call a Workflow, see the Adaptive Ensemble for an example. Notably, the system is interfaced through YAML files (which ensures consistent, well-tested execution paths and proper creation/destruction of containers to guarentee reproducability), which the Coordinator interprets and delegates appropriately. To keep the parallelization tractable we've implemented event tracing. 

NOTE: The sequencer will be adapted to handle batching for memory intensive operations. The coordinator may be expanded to handle computing across multiple computers. Maybe it is indeed best that Coordinator handle data storage locations. 

## Part 1: The Architecture Evolution Story

### Linear Event Flow

The new architecture introduces a **standard linear event flow** that works for all use cases and configurations:

```
         Linear Event Flow with Symbol Containers
┌─────────────────────────────────────────────────────────────┐
│                 UNIVERSAL EVENT FLOW PATTERN                │
│                                                             │
│  Symbol Containers → Stateless Services → Portfolio        │
│  (Data + Features)   (Strategies, Risk)   Containers       │
│                                                             │
│  • No complex adapters needed                               │
│  • Natural parallelization                                  │
│  • Clear data flow                                          │
│  • 60% fewer containers                                     │
└─────────────────────────────────────────────────────────────┘
```

## Part 2: Core Architecture Components

### 1. Symbol Containers (Stateful, Isolated)

**Purpose**: Encapsulate all data and feature computation for a specific symbol+timeframe combination.

```
Symbol Container: SPY_1m
┌─────────────────────────────────────────────────────────────┐
│  Components:                                                │
│  ├── Data Module (streaming market data)                    │
│  └── Feature Hub (indicator calculations)                   │
│                                                             │
│  Responsibilities:                                          │
│  • Maintain streaming position in data                      │
│  • Cache computed indicators                                │
│  • Broadcast BAR and FEATURES events                        │
│                                                             │
│  Isolation Benefit:                                         │
│  • Multi-symbol backtests have clean separation             │
│  • No event contamination between symbols                   │
└─────────────────────────────────────────────────────────────┘
```

### 2. Stateless Service Pools (Pure Functions)

**Purpose**: Process data without maintaining state, enabling massive parallelization.

```
Stateless Service Pools
┌─────────────────────────────────────────────────────────────┐
│  Strategy Services:                                         │
│  • momentum_strategy(features) → signal                     │
│  • mean_reversion(features) → signal                        │
│  • breakout_strategy(features) → signal                     │
│                                                             │
│  Classifier Services:                                       │
│  • hmm_regime(features) → classification                    │
│  • volatility_regime(features) → classification             │
│                                                             │
│  Risk Services:                                             │
│  • validate_order(order, portfolio_state) → approved_order  │
│  • calculate_position_size(signal, risk_params) → size      │
│                                                             │
│  Execution Services:                                        │
│  • calculate_slippage(order, market_data) → slippage        │
│  • calculate_commission(order) → commission                 │
│  • model_market_impact(order, market_depth) → impact        │
│                                                             │
│  Benefits:                                                  │
│  • Zero state contamination                                 │
│  • Perfect parallelization                                  │
│  • Trivial testing (pure functions)                         │
└─────────────────────────────────────────────────────────────┘
```

### 3. Portfolio Containers (Stateful, Isolated)

**Purpose**: Maintain trading state and aggregate signals from multiple strategies.

```
Portfolio Container
┌─────────────────────────────────────────────────────────────┐
│  State Management:                                          │
│  • Positions and cash tracking                              │
│  • Order lifecycle management                               │
│  • P&L calculation                                          │
│                                                             │
│  Signal Aggregation:                                        │
│  • Receives signals from multiple strategies                │
│  • Applies portfolio-level risk management                  │
│  • Generates orders                                         │
│                                                             │
│  Execution:                                                 │
│  • Uses stateless execution functions                       │
│  • Manages order state                                      │
│  • Updates positions on fills                               │
└─────────────────────────────────────────────────────────────┘
```

### 4. Execution Container (Stateful, Shared)

**Purpose**: Manage order lifecycle and generate fills for all portfolios.

```
Execution Container
┌─────────────────────────────────────────────────────────────┐
│  Stateful Components:                                       │
│  • Order lifecycle state machine (NEW → PENDING → FILLED)   │
│  • Active order tracking                                    │
│  • Fill generation and distribution                         │
│  • Execution statistics                                     │
│                                                             │
│  Stateless Components:                                      │
│  • Slippage calculation functions                           │
│  • Market impact modeling                                   │
│  • Commission calculation                                   │
│                                                             │
│  Event Flow:                                                │
│  • Receives: ORDER events from all portfolios               │
│  • Broadcasts: FILL events to all portfolios               │
│  • Portfolios filter fills by portfolio_id                  │
└─────────────────────────────────────────────────────────────┘
```

## Part 3: Complete Event Flow Diagram



```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                          Root-Level Container                                      │
│                                                                                    │
│  Symbol Containers (Isolated)                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                        │
│  │   SPY_1m       │  │   SPY_1d       │  │   QQQ_1m       │                        │
│  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │                        │
│  │ │    Data    │ │  │ │    Data    │ │  │ │    Data    │ │                        │
│  │ └─────┬──────┘ │  │ └─────┬──────┘ │  │ └─────┬──────┘ │                        │
│  │       ↓ BAR    │  │       ↓ BAR    │  │       ↓ BAR    │                        │
│  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │                        │
│  │ │  Features  │ │  │ │  Features  │ │  │ │  Features  │ │                        │
│  │ └─────┬──────┘ │  │ └─────┬──────┘ │  │ └─────┬──────┘ │                        │
│  └───────┼────────┘  └───────┼────────┘  └───────┼────────┘                        │
│          │                   │                   │                                 │
│          └───────────────────┴───────────────────┘                                 │
│                              ↓                                                     │
│                    FEATURES (Broadcast)                                            │
│                              ↓                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                    Stateless Service Pools                                   │  │
│  │  ┌─────────────────────────────────┬───────────────────────────────────────┐ │  │
│  │  │   Strategy Services             │   Classifier Services                 │ │  │
│  │  │  • momentum_strategy            │  • hmm_regime_classifier              │ │  │
│  │  │  • mean_reversion_strategy      │  • volatility_classifier              │ │  │
│  │  │  • breakout_strategy            │  • trend_classifier                   │ │  │
│  │  └─────────────────────────────────┴───────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                                     │
│                     SIGNALS (Targeted)                                             │
│                              ↓                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                      Portfolio Containers (Isolated)                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │  │
│  │  │  Portfolio_1    │  │  Portfolio_2    │  │  Portfolio_n    │               │  │
│  │  │                 │  │                 │  │                 │               │  │
│  │  │ STRATEGIES:     │  │ STRATEGIES:     │  │ STRATEGIES:     │               │  │
│  │  │ • Momentum(10,20)│ │ • Momentum(5,15)│  │ • MeanRev(20)   │               │  │
│  │  │ • Breakout(14)  │  │ • MeanRev(10)   │  │ • Breakout(21)  │               │  │
│  │  │                 │  │ • Breakout(7)   │  │ • Momentum(50)  │               │  │
│  │  │ CLASSIFIERS:    │  │                 │  │                 │               │  │
│  │  │ • HMM Regime    │  │ CLASSIFIERS:    │  │ CLASSIFIERS:    │               │  │
│  │  │ • Volatility    │  │ • Trend         │  │ • HMM Regime    │               │  │
│  │  │                 │  │ • Volatility    │  │ • Trend         │               │  │
│  │  │ State:          │  │ • HMM Regime    │  │ • Volatility    │               │  │
│  │  │ • Positions     │  │                 │  │                 │               │  │
│  │  │ • Orders        │  │ State:          │  │ State:          │               │  │
│  │  │ • Positions     │  │                 │  │                 │               │  │
│  │  │ • Orders        │  │ • Positions     │  │ • Positions     │               │  │
│  │  │ • P&L           │  │ • Orders        │  │ • Orders        │               │  │
│  │  │ • Strategy Wts  │  │ • P&L           │  │ • P&L           │               │  │
│  │  │ • Regime State  │  │ • Strategy Wts  │  │ • Strategy Wts  │               │  │
│  │  └─────────────────┘  │ • Regime State  │  │ • Regime State  │               │  │
│                          └─────────────────┘  └─────────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                              ↓                               ↑                     │
│                      ORDER REQUEST                          FILL                   │
│                              ↓                               ↑                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                    Stateless Service Pools (Level 2)                         │  │
│  │  ┌─────────────────────────────────────┬───────────────────────────────────┐ │  │
│  │  │        Risk Pool                    │      Execution Pool               │ │  │
│  │  │                                     │                                   │ │  │
│  │  │ Risk Validation Services:           │ Execution Calculation Services:   │ │  │
│  │  │ • Portfolio-level risk checks       │ • Slippage calculation            │ │  │
│  │  │ • System-wide risk limits           │ • Market impact modeling          │ │  │
│  │  │ • Liquidity validation              │ • Commission & fee calculation    │ │  │
│  │  │ • Correlation analysis              │ • Execution cost analysis         │ │  │
│  │  │ • Real-time VaR calculation         │ • Price improvement detection     │ │  │
│  │  │ • Margin requirement validation     │                                   │ │  │
│  │  │ • Order size optimization           │                                   │ │  │
│  │  └─────────────────────────────────────┴───────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                              ↓                               ↑                     │
│                        VALIDATED ORDER              CALCULATED FILLS               │
│                              ↓                               ↑                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                      Execution Container (Shared)                            │  │
│  │                                                                              │  │
│  │  Order Lifecycle Management:                                                 │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ STATEFUL:                                                              │  │  │
│  │  │ • Order tracking & state machine                                       │  │  │
│  │  │ • Fill generation & matching                                           │  │  │
│  │  │ • Position reconciliation                                              │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────┘
```

NOTE: We also maintain two alternative topologies: one for signal generation, which is identical to above but terminates at the 'SIGNALS' line, and other for signal replay, containing the remainder. New topologies can be introduced to incorporate new components, see the 'Signal Filter' examples below, where we overview how to extend the system.

## Part 4: Key Architectural Benefits

### 1. Dynamic Generation
# Fix this section - rewrite to emphasize dynamic generation of containers and event wiring 
**Before**: Complex adapter patterns (Pipeline, Broadcast, Hierarchical, etc.)
**After**: Single linear flow that handles all use cases

```yaml
# Old: Complex adapter configuration
adapters:
  - type: hierarchical
    parent: classifier
    children: [risk_containers]
    routing: complex_rules
  - type: pipeline
    stages: [data, indicators, strategies, risk, execution]

# New: No adapter configuration needed!
# The linear flow is implicit in the architecture
```

### 2. Natural Multi-Asset Support

Symbol containers make multi-asset backtesting trivial:

```
Multi-Asset Configuration:
├── SPY_1m Container
├── SPY_5m Container  
├── SPY_1d Container
├── QQQ_1m Container
├── QQQ_1d Container
└── Each broadcasts to relevant strategies
```

### 3. Resource Efficiency

Comparison for 24 strategy combinations (6 strategies × 4 classifiers):

```
Old Architecture:                      New Architecture:
├── 24 Strategy Containers            ├── 0 Strategy Containers (stateless!)
├── 24 Risk Containers                ├── 0 Risk Containers (stateless!)
├── 24 Portfolio Containers           ├── 24 Portfolio Containers
├── 4 Classifier Containers           ├── 0 Classifier Containers (stateless!)
├── Shared Infrastructure             ├── 2-3 Symbol Containers
└── Total: 75+ Containers             └── Total: ~27 Containers (60% reduction!)
```

### 4. Perfect Isolation Where Needed

- **Symbol Containers**: Isolated for clean multi-symbol separation
- **Portfolio Containers**: Isolated for parallel parameter testing
- **Stateless Services**: No isolation needed (no state to contaminate!)

## Part 5: Common Use Cases

### Simple Single-Asset Backtest

```yaml
symbols:
  - symbol: SPY
    timeframes: [1d]

strategies:
  - type: momentum
    parameters:
      fast_period: 10
      slow_period: 20

portfolios:
  - name: simple_test
    risk:
      max_position_size: 0.1
```

Creates:
- 1 Symbol Container (SPY_1d)
- 1 Portfolio Container
- Stateless momentum strategy service

### Multi-Timeframe Cross-Asset Strategy

```yaml
symbols:
  - symbol: SPY
    timeframes: [1m, 5m, 1d]
  - symbol: QQQ
    timeframes: [1m, 1d]
  - symbol: VIX
    timeframes: [5m]

strategies:
  - type: multi_timeframe_pairs
    subscriptions: [SPY_1m, QQQ_1m, SPY_1d, QQQ_1d, VIX_5m]
```

Creates:
- 6 Symbol Containers (one per symbol+timeframe)
- Strategies automatically receive all subscribed data
- Natural handling of complex multi-timeframe logic

### Parameter Optimization Grid Search

```yaml
strategies:
  - type: momentum
    parameters:
      fast_period: [5, 10, 15]
      slow_period: [20, 30, 40]
      
portfolios:
  - risk_levels: [conservative, moderate, aggressive]
```

Creates:
- 27 Portfolio Containers (9 parameter combos × 3 risk levels)
- All share the same Symbol Containers
- Massive parallelization with minimal overhead

### Adaptive Ensemble Workflow

The architecture naturally supports complex multi-phase workflows:

```
Phase 1: Grid Search (Signal Generation Mode - Walk-Forward)
├── Run all parameter combinations using walk-forward validation
├── Capture signals to event store across all walk-forward windows
└── Group results by regime within each validation period

Phase 2: Regime Analysis (using Phase 1 performance data)
├── Find best parameters per detected regime
└── Store regime-optimal configurations

Phase 3: Adaptive Ensemble Weight Optimization (Signal Replay Mode - Walk-Forward)
├── Load only signals from each regime across walk-forward windows
├── Find optimal ensemble weights per regime using walk-forward validation
└── 3x+ faster than replaying all signals

Phase 4: Final Validation (Full Backtest Mode)
├── Deploy final regime-adaptive ensemble strategy over OOS/test data
└── Dynamically switch parameters based on regime
  ```

## Part 6: Configuration Examples

### Basic YAML Configuration

```yaml
# Simple backtest configuration
workflow: full_backtest

symbols:
  - symbol: SPY
    timeframes: [1m, 5m, 1d]

data:
  start_date: '2023-01-01'
  end_date: '2023-12-31'
  source: csv

# Features auto-inferred from strategies
features:
  auto_infer: true

# Stateless services
strategies:
  - type: momentum
    parameters:
      momentum_threshold: [0.01, 0.02]

classifiers:
  - type: volatility_regime
    parameters:
      window: [20, 50]

# Portfolio configuration
portfolios:
  - name: test_portfolio
    strategies: [momentum]  # Can list multiple strategies
    risk:
      max_position_size: [0.1, 0.2]

execution:
  slippage_bps: 5
  commission_per_share: 0.01
```

## Part 7: Implementation Guidelines

### Symbol Container Pattern

```python
class SymbolContainer:
    """Standard pattern for symbol+timeframe data management"""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.identifier = f"{symbol}_{timeframe}"
        
        # Isolated event bus
        self.event_bus = EventBus(self.identifier)
        
        # Stateful components
        self.data = DataModule(symbol, timeframe)
        self.features = FeatureHub()
        
        # Wire internal events
        self.data.on_bar = self.features.update
        self.features.on_features = self.broadcast_features
        
    def broadcast_features(self, features):
        """Broadcast to all interested services"""
        self.event_bus.publish(Event(
            type='FEATURES',
            data={
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'features': features
            }
        ))
```

### Stateless Service Pattern

```python
def momentum_strategy(event: Event) -> Optional[Signal]:
    """Pure function - no state maintained"""
    features = event.data['features']
    
    if features['sma_fast'] > features['sma_slow']:
        return Signal(
            symbol=event.data['symbol'],
            direction='BUY',
            strength=calculate_strength(features)
        )
    return None
```

### Portfolio Container Pattern

```python
class PortfolioContainer:
    """Maintains trading state for a specific configuration"""
    
    def __init__(self, portfolio_id: str, config: Dict):
        self.portfolio_id = portfolio_id
        self.positions = {}
        self.cash = config['initial_capital']
        
        # Isolated event bus for internal events
        self.event_bus = EventBus(f"portfolio_{portfolio_id}")
        
    def process_signal(self, signal: Signal):
        """Aggregate signals and generate orders"""
        # Check risk limits
        if self.risk_check(signal):
            order = self.create_order(signal)
            self.execute_order(order)
            
    def execute_order(self, order: Order):
        """Use stateless execution functions"""
        fill = simulate_execution(order, self.current_market_data)
        self.update_position(fill)
```

## Part 8: Migration Path

For teams migrating from the old architecture:

1. **Remove complex adapters** - the linear flow handles everything
2. **Keep portfolio containers** - they still manage state
3. **Simplify configuration** - no adapter patterns needed

## Summary

The new EVENT_FLOW_ARCHITECTURE represents a major simplification while maintaining all the power of the original Protocol + Composition design:

- **No complex adapters** - linear flow handles all cases
- **Natural multi-asset support** through Symbol containers
- **Perfect isolation** only where needed
- **Massive parallelization** with minimal overhead

The result is a system that's easier to understand, debug, and extend while maintaining perfect reproducibility and scalability.

For more details on the original architecture philosophy, see `docs/new/arch-101.md`.
For implementation examples, see `examples/` directory.
