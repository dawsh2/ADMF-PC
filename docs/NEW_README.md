# ADMF-PC Event Flow Architecture: The Next Evolution

## Executive Summary
This is an event-driven quantitative trading framework that's under active development. We've strived to perfect the architecture to allow parallelization with isolation of state between seperate instances. We've worked hard to ensure the system is modular and flexible, so it's able to accomplish whatever a research can throw at it elegantly. The system should stay out of the users way, but enable maximal flexibility and reproducability. 

The core idea of the system is using containers to isolate state where necassary in a parallelized backtest, and leveraging the idea that the data transformation pipeline is linear, e.g: Data-->BARS-->Features-->FEATURE/BAR-->Strategy & Classifiers-->SIGNAL & ClassifierChange-->Portfolio-->ORDER_REQUEST-->Risk-->ORDER-->Execution (applying stateless functional slippage/commission etc)-->FILL-->PORTFOLIO_UPATE (more or less). We take these primitives - containers, events and cross-container-communication-adapters - and use them to create standardized topologies that dynamicaly generate and wire up the proper structure for whatever we're testing. To handle more complex workflows, such as train/test splitting, walkforward validation, we simply recycle the topologies between phases, and use sequencer to orchestrate (which further delegates responsbilities to other modules when appropriate, i.e optimization or analysis). This ensures each phase is created identically to the previous one, while guarenteeing no leakge of state since all components are destroyed at the end of each phase. At a higher level we combine these sequenced topologies into what we call a Workflow, see the Adaptive Ensemble for an example. Notably, the system is interfaced through YAML files (which ensures consistent, well-tested execution paths and proper creation/destruction of containers to guarentee reproducability), which the Coordinator interprets and delegates appropriately to the TopologyBuilder and Sequencer, depending on the Workflow. Lastly, to improve observability nad keep the parallelization tractable we've implemented event tracing, which introduced many tangential benefits. 

NOTE: The sequencer will be adapted to handle batching for memory intensive operations as it already has all the logic necassary. The Coordinator may be expanded to handle computing across multiple computers. Maybe it is indeed best that Coordinator handle data storage locations. 


## UPDATES & TODO 
- optimization is done by specifying parameter search space in config, which is passed to the to TopologyBuilder, which then passes it to the optimization module for parameter expansion, and uses this to determine number of feature/strategy/classifier/risk/execution configurations to run by using the optimization module for expansion and objective function determination, which portfolio containers then subscribe to accordingly (pretty elegant!) -- confirm objective function is passed correctly
- containers now enable event tracing (with portfolio containers defaulting to it for performance metrics), with varying degrees of info storage -- this keeps us tied to the event tracing system, which will come in handy later, but also keeps it light to store things in mememory -- CRITICAL insight and upgrade, now we're no longer running two parallel data/metric systems and ensure our tracing is plugged in from day one 
- !!: still need to formalize how this data is referenced when not stored in memory between runs, and how we determine when to run in memory, and when to write to disk -- my gut tells me the 'move fast and break things' approach is best here, let's just store in memory until it's a problem, but also setup the event tracing to write to disk so I can analyze when running with more detail. We may want to also look into building memory and performance monitoring, so when I do ineivitably push it to it's limit and it crashes, I at least know what caused it. this is an important design decision and i need to help myself clear things up later. I think when if I do start pushing the memory limit, we'll just set a memory limit per container and write to disk if it gets full. simple and dynamic, doesn't require advanced user configurations. and because our root level container encompasses all containers, we can easily set this to a reasonable maximum. 
- !!: should also be easy to impleement a 'signal_storage' event tracing mode so the container traces it's signal history
- !!: in the future, we'll run in signal generation mode with no portfolios, only strategies/signal generators generating signals, this will have to be traced on the root event bus since that's where strategies live -- should be easy to implement as a workflow default for the signal-generation topology. 
- !!: need to complete final code review of all modules under src/core, file by file. some immediate smells are src/core/analysis, src/core/logging, rename src/core/communication to src/core/adapters/, src/core/tracing.py. double check events module and make sure it's good. 
- !!: then we can proceed with checklist_*.md.



Once the container tracing and metrics streaming is complete, we'll have crystalized the design.




## Part 1: Core Architecture Components

### 1. Stateless Service Pools (Pure Functions)

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


### 2. Symbol Containers (Stateful, Isolated)

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
│  • Broadcasts: FILL events to all portfolios                │
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


## Topologies 
The preceeding diagram is an example of what we call a 'topology'. It's a set of containers and stateless functions configured with a linear event flow, but with any number of subcomponents at each step. For example, in the topology above we may have any number of strategy components running. These are configured dynamically by the TopologyBuilder which accepts a specification file as input and creates the resultant data flow and object graph. The allows us to extend the system to include new components (e.g a signal filtering module) or run in alternate modes (e.g, signal generation or signal replay) arbitrarily, while ensuring consistent and reusable execution paths and object lifecycles. When running an optimization, we simply pass the TopologyBuilder the expanded parameter set, and it constructs it (for _any_ component, not just strategies). This makes it extremely easy and reliable to interface. The TopologyBuilder just creates whatever it's passed, with the same event flow and object lifecycle, regardless of parameter count or variations.

## Sequences and Phases
The sequencer is the next object in the abstraction hierarchy. A sequence is responsible for executing a process, or set of processes, over a given topology, or set of topologies. For example, a train/test split using the standard topology above is, in this system, considered two seperate executions managed as one sequence. A simple backtest (one iteration over a data set) is a single execution and single sequence. This is important so object lifecycle is managed properly, ensuring no leakage of state. At the end of each execution, containers are destroyed (this ensures state is cleaned). A Phase is a complete process that uses sequences. Phases are the high-level, objective-achieving processes. For example, an end-to-end optimization using walk-forward validation and applying parameters to an OOS test set. The system works by breaking these down into reusable sequences and topology patterns to simplify execution and development while ensuring no leakage of state and increase reproducablity. 

Lastly, phases are composed into workflows, for multi-step processes like below:

### Adaptive Ensemble Workflow

The architecture naturally supports complex multi-phase workflows:

```
Phase 1: Grid Search (Signal Generation Mode - Walk-Forward) <-- signal generation topology
├── Run all parameter combinations using walk-forward validation <-- set of sequences 
├── Capture signals to event store across all walk-forward windows
└── Group results by regime within each validation period

Phase 2: Regime Analysis (using Phase 1 performance data)
├── Find best parameters per detected regime
└── Store regime-optimal configurations

Phase 3: Adaptive Ensemble Weight Optimization (Signal Replay Mode - Walk-Forward) <-- signal replay topology
├── Load only signals from each regime across walk-forward windows <-- set of sequences
├── Find optimal ensemble weights per regime using walk-forward validation
└── 3x+ faster than replaying all signals

Phase 4: Final Validation (Full Backtest Mode on OOS)
├── Deploy final regime-adaptive ensemble strategy over OOS/test data
└── Dynamically switch parameters based on regime
  ```




### In summary:
  - Topology = Event flow diagram (Signals, Orders, etc)
  - Execution Window = Container lifecycle (creation → destruction)
  - Sequence = Pattern of execution windows (e.g., train_test = 2 windows)
  - Phase = High-level workflow step that uses a sequence
  - Workflow = 


Sequences are composed into phases, for example a walk-forward validation is just many train/test splits. Phases are the highest level process substep in a _Workflow_.

  I need to add sections about:
  1. Understanding phases vs sequences vs execution windows
  2. How container lifecycle writing works
  
  
## Containers, Routers & Events
These are the building blocks of the system. Events are custom types

  
## Understanding 


## Also need a section on tracing, results and memory management

### 1. Discovery System

All components self-register through decorators, making them automatically discoverable:

```python
# Strategies self-register
@strategy(features=['sma', 'rsi'])
def momentum_strategy(features, bar, params):
    return {'direction': 'long', 'strength': 0.8}

# Classifiers self-register 
@classifier(features=['volatility'])
def volatility_classifier(features, params):
    return {'regime': 'high_vol', 'confidence': 0.9}

# Execution models self-register
@execution_model(model_type='slippage')
class VolumeImpactSlippageModel:
    def calculate_slippage(self, order, market_price, market_data):
        # Slippage calculation logic
```


### 3. Dynamic Parameter Grid Expansion

The system automatically generates all parameter combinations:

```
Strategies × Symbols × Timeframes × Risk Profiles × Execution Models = Total Combinations
```

Each combination gets its own isolated portfolio container with unique routing:

```yaml
strategies:
  - type: momentum
    params: [fast: 10, slow: 20]
  - type: mean_reversion
    params: [period: 14]
    
risk_profiles:
  - type: conservative
  - type: aggressive
  
execution_models:
  - type: retail      # Regular broker costs
  - type: zero_cost   # Ideal execution
```

This creates 2 × 2 × 2 = 8 unique portfolio containers, each testing a different combination.

### 4. Event-Based Routing

Components communicate purely through events with intelligent routing:

**Feature Routing**: The FeatureDispatcher ensures each strategy only receives the features it needs:
```
SPY_1m Features {sma_20, rsi_14, volume, ...} 
    ↓ (filtered by FeatureDispatcher)
Momentum Strategy receives only {sma_20, rsi_14}
Mean Reversion receives only {rsi_14, bollinger_bands}
```

**Signal Routing**: Signals are tagged with `combo_id` for precise delivery:
```
Strategy generates signal → Tagged with combo_id:"c0001" → Only Portfolio_c0001 processes it
```

**Fill Routing**: Fills include `portfolio_id` for accurate position updates:
```
Execution generates fill → Tagged with portfolio_id:"portfolio_c0001" → Only that portfolio updates
```

### 5. Scaling Without Code Changes

To add new capabilities, simply create decorated functions:

```python
# Add a new strategy - system automatically discovers it
@strategy(features=['atr', 'adx'])
def trend_following_strategy(features, bar, params):
    # Strategy logic
    
# Add a new execution model - automatically available
@execution_model(model_type='slippage', name='adverse_market')
class AdverseMarketSlippage:
    # High slippage for stress testing
```

The YAML configuration automatically makes these available:
```yaml
strategies:
  - type: trend_following  # New strategy automatically available
  
execution_models:
  - type: adverse_market   # New execution model automatically available
```

### 7. Natural Multi-Asset Support

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

For 24 strategy combinations (6 strategies × 4 classifiers):

```
Architecture Overview:
├── 0 Strategy Containers (stateless!)
├── 0 Risk Containers (stateless!)
├── 24 Portfolio Containers
├── 0 Classifier Containers (stateless!)
├── 2-3 Symbol Containers
└── Total: ~27 Containers
```

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

Notably, workflows can be composed and extended with branching decision logc. The workflow above could become part of a larger workflow which runs this iteratively until returns diminish. 

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


## Summary


- **Natural multi-asset support** through Symbol containers
- **Perfect isolation** only where needed
- **Massive parallelization** with minimal overhead

The result is a system that's easier to understand, debug, and extend while maintaining perfect reproducibility and scalability.

For more details on the original architecture philosophy, see `docs/new/arch-101.md`.
For implementation examples, see `examples/` directory.
