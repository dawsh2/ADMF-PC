# Event Reference

Complete reference for ADMF-PC's event system including all event types, schemas, flow patterns, and adapter configurations.

## 📡 Event System Overview

ADMF-PC uses a sophisticated event-driven architecture where all components communicate through strongly-typed events. The event system supports:

- **Type-Safe Events**: All events have defined schemas with validation
- **Event Correlation**: Track related events through correlation and causation IDs
- **Flexible Routing**: Configurable event adapters for different communication patterns
- **Schema Evolution**: Backward-compatible event schema versioning
- **Performance Tiers**: Different delivery guarantees based on event importance

## 🎯 Core Event Types

### Market Data Events

#### BarEvent
**Description**: Historical or real-time OHLCV price data

```python
@dataclass
class BarEvent(SemanticEventBase):
    # Market data
    symbol: str                      # Trading symbol (e.g., "SPY")
    timestamp: datetime              # Bar timestamp
    open: float                      # Opening price
    high: float                      # High price
    low: float                       # Low price
    close: float                     # Closing price
    volume: int                      # Volume traded
    
    # Optional fields
    vwap: Optional[float] = None     # Volume-weighted average price
    trades: Optional[int] = None     # Number of trades
    
    # Metadata
    timeframe: str = "1m"            # Bar timeframe (1m, 5m, 1h, etc.)
    data_source: str = ""            # Data provider
    
schema_version: "1.1.0"
performance_tier: "fast"             # High frequency, low latency
```

**Validation Rules**:
- `open`, `high`, `low`, `close` must be positive
- `high` >= max(`open`, `close`)
- `low` <= min(`open`, `close`)
- `volume` >= 0

**Example**:
```python
bar_event = BarEvent(
    symbol="SPY",
    timestamp=datetime(2023, 12, 1, 9, 30, 0),
    open=450.25,
    high=451.80,
    low=450.10,
    close=451.50,
    volume=1500000,
    timeframe="1m",
    event_id="bar_spy_20231201_0930",
    correlation_id="trading_session_001"
)
```

#### TickEvent
**Description**: Individual trade or quote data

```python
@dataclass
class TickEvent(SemanticEventBase):
    symbol: str                      # Trading symbol
    timestamp: datetime              # Tick timestamp
    price: float                     # Trade price
    size: int                        # Trade size
    
    # Quote data (optional)
    bid: Optional[float] = None      # Best bid price
    ask: Optional[float] = None      # Best ask price
    bid_size: Optional[int] = None   # Bid size
    ask_size: Optional[int] = None   # Ask size
    
    # Trade metadata
    tick_type: Literal["trade", "quote"] = "trade"
    exchange: str = ""               # Exchange identifier
    
schema_version: "1.0.0"
performance_tier: "fast"
```

### Indicator Events

#### IndicatorEvent
**Description**: Technical indicator calculation results

```python
@dataclass
class IndicatorEvent(SemanticEventBase):
    # Indicator identification
    indicator_name: str              # Indicator name (e.g., "SMA_20", "RSI_14")
    symbol: str                      # Symbol for calculation
    value: float                     # Indicator value
    timestamp: datetime              # Calculation timestamp
    
    # Optional metadata
    confidence: Optional[float] = None    # Indicator confidence (0-1)
    parameters: Dict[str, Any] = field(default_factory=dict)  # Indicator parameters
    metadata: Dict[str, Any] = field(default_factory=dict)    # Additional metadata
    
    # Calculation context
    calculation_period: Optional[int] = None  # Lookback period used
    data_points_used: Optional[int] = None    # Number of data points in calculation
    
schema_version: "1.2.0"
performance_tier: "standard"
```

**Common Indicator Names**:
- Moving Averages: `SMA_20`, `EMA_12`, `WMA_10`
- Momentum: `RSI_14`, `MACD_12_26_9`, `STOCH_14_3`
- Volatility: `ATR_14`, `BB_20_2`, `VIX`
- Volume: `OBV`, `VWAP`, `AD_LINE`

**Example**:
```python
indicator_event = IndicatorEvent(
    indicator_name="RSI_14",
    symbol="SPY",
    value=65.4,
    timestamp=datetime(2023, 12, 1, 9, 30, 0),
    confidence=0.85,
    parameters={"period": 14, "method": "wilder"},
    calculation_period=14,
    data_points_used=14
)
```

### Trading Signal Events

#### TradingSignal
**Description**: Buy/sell/hold signals from strategies

```python
@dataclass
class TradingSignal(SemanticEventBase):
    # Signal identification
    symbol: str                      # Trading symbol
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD"  # Signal action
    strength: float = 0.0            # Signal strength (0.0-1.0)
    timestamp: datetime              # Signal generation time
    
    # Price targets (optional)
    price_target: Optional[float] = None      # Target price
    stop_loss: Optional[float] = None         # Stop loss price
    take_profit: Optional[float] = None       # Take profit price
    
    # Strategy context
    strategy_id: str = ""            # Strategy that generated signal
    strategy_type: str = ""          # Strategy type
    regime_context: Optional[str] = None      # Market regime
    
    # Risk assessment
    risk_score: float = 0.5          # Risk assessment (0.0-1.0)
    confidence: float = 0.5          # Signal confidence (0.0-1.0)
    
    # Additional metadata
    indicators_used: List[str] = field(default_factory=list)  # Indicators contributing to signal
    features: Dict[str, Any] = field(default_factory=dict)    # Feature values
    
schema_version: "2.1.0"
performance_tier: "standard"
```

**Signal Strength Interpretation**:
- `0.0-0.2`: Weak signal
- `0.2-0.5`: Moderate signal  
- `0.5-0.8`: Strong signal
- `0.8-1.0`: Very strong signal

**Example**:
```python
trading_signal = TradingSignal(
    symbol="SPY",
    action="BUY",
    strength=0.75,
    timestamp=datetime(2023, 12, 1, 9, 30, 0),
    price_target=452.0,
    stop_loss=449.0,
    strategy_id="momentum_strategy_001",
    strategy_type="momentum",
    risk_score=0.3,
    confidence=0.8,
    indicators_used=["SMA_20", "RSI_14"],
    features={"sma_slope": 0.05, "rsi_divergence": True}
)
```

### Order Events

#### OrderEvent
**Description**: Orders to be executed by the execution engine

```python
@dataclass
class OrderEvent(SemanticEventBase):
    # Order identification
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str                      # Trading symbol
    side: Literal["BUY", "SELL"]     # Order side
    quantity: int                    # Order quantity (shares)
    
    # Order details
    order_type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"] = "MARKET"
    price: Optional[float] = None    # Limit price (for limit orders)
    stop_price: Optional[float] = None  # Stop price (for stop orders)
    
    # Time in force
    time_in_force: Literal["DAY", "GTC", "IOC", "FOK"] = "DAY"
    expire_time: Optional[datetime] = None  # Order expiration
    
    # Risk management
    position_size_pct: float = 0.02  # Position size as % of portfolio
    max_position_value: Optional[float] = None  # Maximum position value
    
    # Execution preferences
    execution_algorithm: Optional[str] = None  # TWAP, VWAP, etc.
    urgency: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    
    # Source tracking
    source_signal_id: Optional[str] = None     # Originating signal
    strategy_id: str = ""            # Strategy that generated order
    
schema_version: "1.3.0"
performance_tier: "reliable"
```

**Order Type Descriptions**:
- `MARKET`: Execute immediately at market price
- `LIMIT`: Execute only at specified price or better
- `STOP`: Convert to market order when stop price reached
- `STOP_LIMIT`: Convert to limit order when stop price reached

**Example**:
```python
order_event = OrderEvent(
    symbol="SPY",
    side="BUY",
    quantity=100,
    order_type="LIMIT",
    price=451.25,
    time_in_force="DAY",
    position_size_pct=0.02,
    execution_algorithm="TWAP",
    urgency="MEDIUM",
    source_signal_id="signal_spy_20231201_001",
    strategy_id="momentum_strategy_001"
)
```

### Execution Events

#### FillEvent
**Description**: Executed trade confirmations

```python
@dataclass
class FillEvent(SemanticEventBase):
    # Fill identification
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str                    # Original order ID
    symbol: str                      # Trading symbol
    side: Literal["BUY", "SELL"]     # Trade side
    
    # Execution details
    quantity: int                    # Filled quantity
    price: float                     # Execution price
    timestamp: datetime              # Execution timestamp
    
    # Costs and fees
    commission: float = 0.0          # Commission paid
    fees: float = 0.0               # Other fees
    
    # Market data at execution
    market_price: Optional[float] = None      # Market price at execution
    bid: Optional[float] = None      # Bid price at execution
    ask: Optional[float] = None      # Ask price at execution
    
    # Execution metadata
    venue: str = ""                  # Execution venue
    liquidity_flag: Optional[str] = None     # "A"dd or "R"emove liquidity
    execution_algorithm: Optional[str] = None # Algorithm used
    
    # Performance metrics
    slippage_bps: Optional[float] = None      # Execution slippage
    implementation_shortfall: Optional[float] = None  # Implementation shortfall
    
schema_version: "1.1.0"
performance_tier: "reliable"
```

**Example**:
```python
fill_event = FillEvent(
    order_id="order_spy_20231201_001",
    symbol="SPY",
    side="BUY",
    quantity=100,
    price=451.30,
    timestamp=datetime(2023, 12, 1, 9, 30, 15),
    commission=0.50,
    market_price=451.25,
    bid=451.20,
    ask=451.35,
    venue="NYSE",
    liquidity_flag="A",
    slippage_bps=1.1
)
```

### Portfolio Events

#### PortfolioUpdateEvent
**Description**: Portfolio state changes and position updates

```python
@dataclass
class PortfolioUpdateEvent(SemanticEventBase):
    # Portfolio snapshot
    timestamp: datetime              # Update timestamp
    total_value: float               # Total portfolio value
    cash: float                      # Available cash
    
    # Position details
    positions: Dict[str, int] = field(default_factory=dict)        # Symbol -> quantity
    position_values: Dict[str, float] = field(default_factory=dict) # Symbol -> market value
    unrealized_pnl: Dict[str, float] = field(default_factory=dict) # Symbol -> unrealized P&L
    
    # Portfolio metrics
    total_unrealized_pnl: float = 0.0       # Total unrealized P&L
    total_realized_pnl: float = 0.0         # Total realized P&L
    gross_exposure: float = 0.0             # Gross market exposure
    net_exposure: float = 0.0               # Net market exposure
    
    # Risk metrics
    portfolio_beta: Optional[float] = None   # Portfolio beta
    portfolio_volatility: Optional[float] = None  # Portfolio volatility
    var_1d_95: Optional[float] = None        # 1-day 95% VaR
    
    # Update trigger
    update_trigger: str = ""         # What caused the update
    related_fill_id: Optional[str] = None   # Related fill (if applicable)
    
schema_version: "1.0.0"
performance_tier: "standard"
```

**Example**:
```python
portfolio_update = PortfolioUpdateEvent(
    timestamp=datetime(2023, 12, 1, 9, 30, 15),
    total_value=101250.0,
    cash=55125.0,
    positions={"SPY": 100, "QQQ": -50},
    position_values={"SPY": 45125.0, "QQQ": -18750.0},
    unrealized_pnl={"SPY": 125.0, "QQQ": -250.0},
    total_unrealized_pnl=-125.0,
    gross_exposure=63875.0,
    net_exposure=26375.0,
    update_trigger="fill_processed",
    related_fill_id="fill_spy_20231201_001"
)
```

### Regime and Context Events

#### RegimeChangeEvent
**Description**: Market regime changes and context updates

```python
@dataclass
class RegimeChangeEvent(SemanticEventBase):
    # Regime identification
    previous_regime: str             # Previous regime name
    new_regime: str                  # New regime name
    confidence: float                # Regime confidence (0.0-1.0)
    timestamp: datetime              # Regime change timestamp
    
    # Classification details
    classifier_type: str             # Type of regime classifier
    classifier_id: str = ""          # Specific classifier instance
    
    # Regime characteristics
    regime_features: Dict[str, float] = field(default_factory=dict)  # Regime features
    transition_probability: Optional[float] = None  # Transition probability
    expected_duration: Optional[int] = None         # Expected regime duration (days)
    
    # Historical context
    regime_history: List[str] = field(default_factory=list)  # Recent regime sequence
    last_regime_duration: Optional[int] = None      # Previous regime duration
    
schema_version: "1.0.0"
performance_tier: "reliable"
```

**Common Regime Types**:
- Trend Regimes: `"BULL"`, `"BEAR"`, `"SIDEWAYS"`
- Volatility Regimes: `"LOW_VOL"`, `"HIGH_VOL"`, `"EXTREME_VOL"`
- Correlation Regimes: `"DECOUPLED"`, `"COUPLED"`, `"CRISIS"`

**Example**:
```python
regime_change = RegimeChangeEvent(
    previous_regime="LOW_VOL",
    new_regime="HIGH_VOL",
    confidence=0.85,
    timestamp=datetime(2023, 12, 1, 9, 0, 0),
    classifier_type="hidden_markov",
    classifier_id="hmm_vol_3state",
    regime_features={"volatility": 0.28, "trend": -0.05},
    transition_probability=0.75,
    expected_duration=15,
    regime_history=["LOW_VOL", "LOW_VOL", "HIGH_VOL"]
)
```

## 🔄 Event Flow Patterns

### Standard Trading Flow

```
BarEvent → IndicatorEvent → TradingSignal → OrderEvent → FillEvent → PortfolioUpdateEvent
```

**Event Sequence**:
1. **BarEvent**: New market data arrives
2. **IndicatorEvent**: Technical indicators calculated
3. **TradingSignal**: Strategy generates signal
4. **OrderEvent**: Risk management creates order
5. **FillEvent**: Order executed by broker
6. **PortfolioUpdateEvent**: Portfolio state updated

### Regime-Aware Flow

```
BarEvent → RegimeChangeEvent → IndicatorEvent → TradingSignal (regime-adjusted) → OrderEvent
```

**Enhanced Flow**:
- Regime detection runs on market data
- Regime changes trigger strategy adjustments
- Signals are generated with regime context
- Risk management adapts to regime

### Multi-Strategy Flow

```
                    ┌─ Strategy A → TradingSignal A ─┐
BarEvent → IndicatorEvent ├─ Strategy B → TradingSignal B ─┼→ EnsembleSignal → OrderEvent
                    └─ Strategy C → TradingSignal C ─┘
```

**Ensemble Processing**:
- Multiple strategies process same data
- Each strategy generates independent signals
- Ensemble aggregates signals into final decision

## 🔌 Event Adapter Configuration

### Pipeline Adapter

**Use Case**: Sequential processing through containers

```yaml
event_adapters:
  - type: "pipeline"
    name: "main_trading_flow"
    containers: ["data", "indicators", "strategy", "risk", "execution"]
    
    # Performance configuration
    buffer_size: 1000
    batch_processing: true
    timeout_seconds: 30
    
    # Error handling
    error_policy: "propagate"        # or "skip", "retry"
    retry_attempts: 3
    dead_letter_queue: true
```

**Event Flow**:
```
Data Container → Indicators Container → Strategy Container → Risk Container → Execution Container
```

### Broadcast Adapter

**Use Case**: One-to-many event distribution

```yaml
event_adapters:
  - type: "broadcast"
    name: "market_data_distribution"
    source: "data_container"
    targets: ["strategy_a", "strategy_b", "strategy_c"]
    
    # Filtering per target
    target_filters:
      strategy_a:
        event_types: ["BarEvent"]
        symbols: ["SPY", "QQQ"]
      strategy_b:
        event_types: ["BarEvent", "IndicatorEvent"]
        symbols: ["SPY"]
        
    # Delivery configuration
    delivery_guarantee: "at_least_once"
    async_delivery: true
```

### Hierarchical Adapter

**Use Case**: Parent-child relationships with context propagation

```yaml
event_adapters:
  - type: "hierarchical"
    name: "regime_hierarchy"
    parent: "regime_classifier"
    children: ["strategy_bull", "strategy_bear", "strategy_neutral"]
    
    # Context propagation
    context_events: ["RegimeChangeEvent"]
    propagate_to_all_children: true
    
    # Aggregation
    aggregate_child_results: true
    aggregation_method: "weighted_average"
    child_weights:
      strategy_bull: 0.4
      strategy_bear: 0.3
      strategy_neutral: 0.3
```

### Selective Adapter

**Use Case**: Content-based routing with complex rules

```yaml
event_adapters:
  - type: "selective"
    name: "intelligent_routing"
    source: "signal_generator"
    
    # Routing rules
    routing_rules:
      - condition: "event.strength > 0.8 and event.confidence > 0.7"
        target: "high_conviction_execution"
        
      - condition: "event.strength > 0.5 and event.risk_score < 0.3"
        target: "moderate_conviction_execution"
        
      - condition: "event.action == 'HOLD' or event.strength < 0.2"
        target: "signal_archive"
        
    # Default routing
    default_target: "standard_execution"
    route_unmatched: true
```

## 📊 Event Performance Tiers

### Performance Tier Configuration

```yaml
event_performance:
  # Fast tier - high frequency, low latency
  fast_tier:
    events: ["BarEvent", "TickEvent"]
    delivery_guarantee: "at_most_once"
    max_latency_ms: 10
    batch_size: 100
    compression: false
    
  # Standard tier - balanced performance
  standard_tier:
    events: ["IndicatorEvent", "TradingSignal", "PortfolioUpdateEvent"]
    delivery_guarantee: "at_least_once"
    max_latency_ms: 100
    batch_size: 50
    compression: true
    
  # Reliable tier - guaranteed delivery
  reliable_tier:
    events: ["OrderEvent", "FillEvent", "RegimeChangeEvent"]
    delivery_guarantee: "exactly_once"
    max_latency_ms: 1000
    batch_size: 1
    persistence: true
    acknowledgments: true
```

### Tier Characteristics

| Tier | Latency | Throughput | Reliability | Use Cases |
|------|---------|------------|-------------|-----------|
| Fast | <10ms | Very High | Best Effort | Market data, quotes |
| Standard | <100ms | High | At Least Once | Indicators, signals |
| Reliable | <1000ms | Medium | Exactly Once | Orders, fills, regime changes |

## 🔍 Event Correlation and Tracing

### Correlation ID Pattern

```python
# Generate correlation ID for related events
correlation_id = f"trade_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Market data event
bar_event = BarEvent(
    symbol="SPY",
    correlation_id=correlation_id,
    # ... other fields
)

# Derived indicator event
indicator_event = IndicatorEvent(
    indicator_name="RSI_14",
    correlation_id=correlation_id,  # Same correlation ID
    causation_id=bar_event.event_id,  # Links to causing event
    # ... other fields
)

# Trading signal
trading_signal = TradingSignal(
    symbol="SPY",
    correlation_id=correlation_id,  # Same correlation ID
    causation_id=indicator_event.event_id,  # Links to causing event
    # ... other fields
)
```

### Event Tracing Configuration

```yaml
event_tracing:
  enabled: true
  
  # Trace collection
  trace_collection:
    correlation_tracking: true
    causation_tracking: true
    timing_analysis: true
    
  # Trace storage
  trace_storage:
    backend: "elasticsearch"        # or "database", "file"
    retention_days: 90
    compression: true
    
  # Trace analysis
  trace_analysis:
    latency_analysis: true
    bottleneck_detection: true
    error_correlation: true
    
  # Visualization
  trace_visualization:
    real_time_dashboard: true
    flow_diagrams: true
    performance_heatmaps: true
```

## 🛠️ Custom Event Types

### Creating Custom Events

```python
from src.core.events.base import SemanticEventBase
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CustomMarketEvent(SemanticEventBase):
    """Custom event for specialized market data"""
    
    # Required fields
    symbol: str
    custom_metric: float
    calculation_method: str
    
    # Optional fields
    confidence_interval: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Schema information
    schema_version: str = "1.0.0"
    performance_tier: str = "standard"
    
    def validate(self) -> bool:
        """Custom validation logic"""
        return (
            self.custom_metric >= 0.0 and
            self.calculation_method in ["method_a", "method_b"] and
            (self.confidence_interval is None or 0.0 <= self.confidence_interval <= 1.0)
        )

# Register custom event type
from src.core.events.registry import event_registry
event_registry.register_event_type(CustomMarketEvent)
```

### Custom Event Configuration

```yaml
# Using custom events in configuration
custom_events:
  - type: "CustomMarketEvent"
    source: "custom_data_provider"
    target_containers: ["strategy_container"]
    
    # Validation settings
    validation:
      strict_mode: true
      log_validation_errors: true
      
    # Performance settings
    performance_tier: "standard"
    batch_processing: true
```

## 📋 Event Validation

### Schema Validation

```python
def validate_event(event: SemanticEventBase) -> ValidationResult:
    """Validate event against its schema"""
    
    # Type validation
    if not isinstance(event, SemanticEventBase):
        return ValidationResult(False, ["Event must inherit from SemanticEventBase"])
    
    # Required field validation
    errors = []
    required_fields = get_required_fields(type(event))
    for field in required_fields:
        if not hasattr(event, field) or getattr(event, field) is None:
            errors.append(f"Required field '{field}' is missing or None")
    
    # Custom validation
    if hasattr(event, 'validate') and callable(event.validate):
        if not event.validate():
            errors.append("Custom validation failed")
    
    # Business logic validation
    business_errors = validate_business_logic(event)
    errors.extend(business_errors)
    
    return ValidationResult(len(errors) == 0, errors)
```

### Validation Configuration

```yaml
event_validation:
  # Global validation settings
  strict_mode: true                  # Reject invalid events
  log_validation_errors: true       # Log validation failures
  
  # Per-event-type validation
  event_type_validation:
    BarEvent:
      price_sanity_check: true       # Check price reasonableness
      volume_validation: true        # Validate volume data
      
    TradingSignal:
      strength_bounds_check: true    # Ensure strength in [0,1]
      strategy_id_required: true     # Require strategy ID
      
    OrderEvent:
      position_size_check: true      # Validate position sizing
      risk_limit_check: true         # Check against risk limits
      
  # Validation error handling
  error_handling:
    invalid_event_action: "reject"   # or "log_and_continue", "transform"
    max_validation_errors: 100       # Stop processing after N errors
    error_notification: true         # Send alerts on validation failures
```

## 🚀 Performance Optimization

### Event Serialization

```yaml
event_serialization:
  # Serialization format
  default_format: "msgpack"         # or "json", "protobuf", "pickle"
  
  # Compression
  compression:
    enabled: true
    algorithm: "lz4"                 # or "gzip", "zstd"
    compression_level: 3
    
  # Performance optimization
  optimization:
    event_pooling: true              # Reuse event objects
    batch_serialization: true       # Serialize in batches
    lazy_deserialization: true      # Deserialize on demand
    
  # Format-specific settings
  msgpack_settings:
    use_bin_type: true
    datetime_format: "timestamp"
    
  protobuf_settings:
    schema_registry: "confluent"
    schema_evolution: true
```

### Event Batching

```yaml
event_batching:
  # Batching configuration
  default_batch_size: 100
  max_batch_wait_ms: 50
  
  # Per-tier batching
  tier_batching:
    fast_tier:
      batch_size: 1000
      max_wait_ms: 10
      
    standard_tier:
      batch_size: 100
      max_wait_ms: 50
      
    reliable_tier:
      batch_size: 1
      max_wait_ms: 1000
      
  # Adaptive batching
  adaptive_batching:
    enabled: true
    min_batch_size: 10
    max_batch_size: 1000
    latency_target_ms: 100
```

---

Continue to [Coordinator Modes](coordinator-modes.md) for detailed coordinator configuration →