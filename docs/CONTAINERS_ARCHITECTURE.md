# Core Containers

This module contains THE container implementation for ADMF-PC.

## Files

- `container.py` - The canonical container implementation
- `protocols.py` - Container protocols and interfaces
- `factory.py` - Container creation and lifecycle

## No "Enhanced" Versions

Do not create enhanced_container.py, improved_container.py, etc.
Use composition and configuration to add capabilities.

## Unified Architecture: Stateful vs Stateless Components

In the unified stateless architecture, we maintain a clear separation between:
- **Stateful Containers** (4 types): Handle data streaming, caching, and lifecycle
- **Stateless Components**: Pure functions for strategies, classifiers, and risk validation

### Containers That Remain Stateful

#### 1. Data Container
**Role**: Streaming data management and timeline coordination
- Maintains current market data position
- Manages data streaming and buffering
- Provides consistent timeline across all components
- Caches historical data for efficient access

**Why Stateful**: Data streaming requires maintaining position in the data stream and buffering for efficient access.

#### 2. FeatureHub Container 
**Role**: Indicator calculation and feature caching
- Calculates technical indicators incrementally
- Maintains rolling windows for efficiency
- Caches computed features to avoid recalculation
- Provides feature values to stateless strategies

**Why Stateful**: Incremental indicator calculation requires maintaining historical state (e.g., 20-day moving average needs last 20 values).

#### 3. Portfolio Container
**Role**: Position tracking and P&L calculation
- Tracks open positions and cash balance
- Calculates real-time P&L and risk metrics
- Maintains transaction history
- Provides portfolio state to risk validators

**Why Stateful**: Portfolio management inherently requires tracking positions, cash, and historical transactions.

#### 4. Execution Container
**Role**: Order lifecycle and fill management
- Manages order state machine (pending → filled/cancelled)
- Tracks partial fills
- Handles order routing and broker communication
- Maintains execution history

**Why Stateful**: Order management requires tracking order states and handling asynchronous fill notifications.

### Components That Become Stateless

#### Strategies → StatelessStrategy Protocol
- Pure functions: `generate_signal(features, bar, params) → signal`
- No internal state, all data passed as parameters
- Can run 1000s in parallel with different parameters

#### Classifiers → StatelessClassifier Protocol  
- Pure functions: `classify_regime(features, params) → regime`
- No regime history, just current classification
- Perfect for parallel regime detection

#### Risk Validators → StatelessRiskValidator Protocol
- Pure functions: `validate_order(order, portfolio_state, limits, market_data) → result`
- Portfolio state passed as parameter
- Enables testing multiple risk configurations simultaneously

### Universal Topology

The unified architecture creates the same 4 containers for ALL workflow modes:

```
Data Container → FeatureHub Container
                        ↓
              [Stateless Strategies]
              [Stateless Classifiers]
                        ↓
               Portfolio Containers (1 per parameter combo)
                        ↓
              [Stateless Risk Validators]
                        ↓
               Execution Container
```

### Benefits

1. **60% Container Reduction**: From ~10 containers to 4 + stateless components
2. **Perfect Parallelization**: Stateless components can run in parallel
3. **Single Topology**: Same structure for backtest, signal generation, and replay
4. **No Pattern Detection**: Eliminates complex workflow pattern matching
5. **Simplified Configuration**: Just specify mode and parameters

### Parameter Expansion

For multiple strategies/risk configs, we create multiple portfolio containers:
- Each portfolio container gets a unique `combo_id`
- Stateless strategies/classifiers are shared across all portfolios
- RoutingAdapter directs signals to correct portfolio by `combo_id`

Example with 20 strategies × 3 risk configs:
- 1 Data Container
- 1 FeatureHub Container
- 60 Portfolio Containers (one per combination)
- 1 Execution Container
- 20 Stateless Strategy instances (shared)
- 3 Stateless Risk Validator instances (shared)

Total: 64 containers + 23 stateless components (vs 180 full containers in old system!)