# Event Tracing Architecture: Unified Metrics Through Event Sourcing

## Overview

This document explains a critical architectural decision in ADMF-PC: using event tracing as the single source of truth for portfolio metrics, with intelligent memory management through pruning. This approach elegantly solves multiple challenges in parallel backtesting while laying the foundation for future ML and analysis capabilities.

## The Problem Space

When building a parallel backtesting system, we face several interconnected challenges:

### 1. Metrics Calculation in Parallel Environments
- Multiple portfolio containers running simultaneously
- Each portfolio manages different strategies
- Need accurate, isolated metrics per portfolio
- Must avoid cross-contamination between parallel runs

### 2. Memory Constraints at Scale
- Full event traces can consume gigabytes in large-scale backtests
- Need to track enough data for accurate metrics
- Must maintain reasonable memory footprint
- Cannot sacrifice data fidelity for performance

### 3. Data Consistency and Auditability
- Metrics must match actual trading events
- Need ability to audit/replay specific trades
- Traditional approach: maintain separate metrics database
- Problem: metrics can drift from actual events

### 4. Future Extensibility
- Eventually need full event traces for ML/analysis
- Don't want to rebuild infrastructure later
- Need gradual path from metrics-only to full tracing

## The Solution: Event-Sourced Metrics with Intelligent Pruning

### Core Insight
Instead of maintaining metrics as separate state, derive them directly from the event stream. Use memory-aware pruning to keep only what's needed for active calculations.

```python
# Traditional approach - separate metrics tracking
class Portfolio:
    def __init__(self):
        self.metrics = MetricsCalculator()
        self.positions = {}
    
    def on_trade(self, trade):
        self.positions[trade.id] = trade
        self.metrics.update(trade)  # Parallel bookkeeping - can drift!

# Our approach - metrics derived from events
class Portfolio:
    def __init__(self):
        self.event_bus = EventBus()
        self.event_bus.attach_observer(MetricsObserver(
            retention_policy="trade_complete"  # Prune closed trades
        ))
    
    def on_trade(self, trade):
        self.event_bus.publish(trade)  # Single source of truth!
```

### Architecture Components

#### 1. Mandatory Event Filtering
```python
# Enforce correctness in parallel environments
bus.subscribe(
    EventType.SIGNAL.value,
    portfolio.receive_event,
    filter_func=strategy_filter(['momentum_1', 'pairs_1'])  # REQUIRED!
)
```
This prevents cross-contamination between portfolios by construction, not convention.

#### 2. Observer-Based Metrics Collection
```python
@dataclass
class MetricsObserver(EventObserverProtocol):
    calculator: MetricsCalculatorProtocol  # Composed, not inherited
    retention_policy: str = "trade_complete"
    active_trades: Dict[str, List[Event]] = field(default_factory=dict)
    
    def on_publish(self, event: Event) -> None:
        if event.event_type == EventType.POSITION_OPEN:
            # Track events for this trade
            self.active_trades[event.correlation_id] = [event]
            
        elif event.event_type == EventType.POSITION_CLOSE:
            # Calculate metrics from complete trade history
            trade_events = self.active_trades[event.correlation_id]
            self.calculator.update_from_trade(trade_events)
            
            # CRITICAL: Prune completed trade from memory
            del self.active_trades[event.correlation_id]
```

#### 3. Memory-Aware Retention Policies
```python
# Different policies for different use cases
retention_policies = {
    "trade_complete": "Keep events until trade closes, then prune",
    "sliding_window": "Keep last N events",
    "minimal": "Keep only open positions",
    "all": "Keep everything (for development/analysis)"
}
```

### The Elegance: Multiple Problems, One Solution

This design solves all our challenges with a single, cohesive approach:

1. **Parallel Safety**: Mandatory filters prevent cross-contamination
2. **Memory Efficiency**: Pruning keeps memory bounded
3. **Data Consistency**: Metrics derived from events, not parallel state
4. **Future-Proof**: Same infrastructure scales from metrics to full ML

## Real-World Example: Parallel Backtest

```yaml
# Configuration automatically sets up isolated, efficient portfolios
portfolios:
  - id: momentum_portfolio
    strategies: [momentum_1, momentum_2]
    event_tracing:
      retention_policy: trade_complete
      
  - id: pairs_portfolio  
    strategies: [pairs_1, pairs_2]
    event_tracing:
      retention_policy: trade_complete
```

Each portfolio:
- Only receives events for its strategies (forced filtering)
- Tracks complete history of open trades (memory efficient)
- Calculates metrics from actual events (accurate)
- Prunes completed trades automatically (scalable)

## Benefits Realized

### 1. Correctness by Construction
- Impossible to accidentally process wrong strategy's signals
- Metrics always match actual events
- No drift between "what happened" and "what we measured"

### 2. Memory Efficiency
```python
# Example: 1000 parallel portfolios, 100 trades each
# Traditional: ~100MB per portfolio (all trades) = 100GB total
# With pruning: ~1MB per portfolio (open trades only) = 1GB total
# 100x memory reduction!
```

### 3. Rich Debugging and Analysis
```python
# The same event system monitors itself!
memory_observer = ContainerMemoryObserver()
latency_observer = LatencyObserver()

# Debug memory leaks, performance issues, etc.
# Using the same infrastructure that runs the backtest
```

### 4. Natural Evolution Path
```python
# Today: Metrics only
retention_policy: "trade_complete"

# Tomorrow: ML features
retention_policy: "sliding_window"
window_size: 1000

# Future: Full analysis
retention_policy: "all"
storage_backend: "disk"
```

## Design Decisions and Trade-offs

### Why Observer Pattern?
- Separates concerns (observation vs calculation)
- Allows multiple observers without changing core code
- Easy to test with mock observers
- Natural fit for event-driven architecture

### Why Mandatory Filtering for SIGNAL Events?
- SIGNAL events are the most critical for correctness
- In parallel backtests, wrong signals → wrong results
- Better to fail at subscription than silently process wrong data
- Other event types (FILL, BAR) less dangerous to broadcast

### Why Prune at Trade Completion?
- Natural boundary for metrics calculation
- Minimizes memory while preserving completeness
- Can always replay from persistent storage if needed
- Matches mental model of trading (position lifecycle)

## Implementation Guidelines

### 1. Container Setup
```python
class PortfolioContainer:
    def __init__(self, config):
        # Set up event bus with tracing
        self.event_bus = EventBus(self.container_id)
        
        # Create metrics observer with pruning
        metrics_observer = MetricsObserver(
            calculator=StreamingMetrics(initial_capital),
            retention_policy=config.get('retention_policy', 'trade_complete')
        )
        
        # Attach observer
        self.event_bus.attach_observer(metrics_observer)
        
        # Subscribe with mandatory filter
        strategies = config.get('strategies', [])
        root_bus.subscribe_to_signals(
            self.receive_event,
            strategy_ids=strategies
        )
```

### 2. Metrics Retrieval
```python
def get_metrics(self) -> Dict[str, Any]:
    """Metrics are already calculated by observer."""
    return self.metrics_observer.get_metrics()
```

### 3. Memory Monitoring
```python
# Add memory observer in development/staging
if config.get('monitor_memory', False):
    memory_observer = ContainerMemoryObserver(
        emit_warnings=True,
        threshold_mb=50.0
    )
    self.event_bus.attach_observer(memory_observer)
```

## Future Extensions

### 1. ML Feature Extraction
```python
# New observer for feature extraction
class FeatureExtractionObserver(EventObserverProtocol):
    def on_publish(self, event: Event):
        if event.event_type == EventType.FEATURES:
            self.feature_buffer.append(event.payload)
            # Extract patterns, sequences, etc.
```

### 2. Execution Analysis
```python
# Track slippage, latency, etc.
class ExecutionQualityObserver(EventObserverProtocol):
    def on_publish(self, event: Event):
        if event.event_type == EventType.FILL:
            latency = event.timestamp - event.causation_timestamp
            slippage = event.payload['fill_price'] - event.payload['expected_price']
```

### 3. Distributed Tracing
```python
# Extend to distributed systems
class DistributedTracingObserver(EventObserverProtocol):
    def on_publish(self, event: Event):
        span = self.tracer.start_span(event.event_type)
        span.set_tag('correlation_id', event.correlation_id)
```

## Conclusion

This architecture turns what could be a complex distributed systems problem into an elegant event-sourcing solution. By making event tracing the foundation rather than an afterthought, we get:

1. **Correctness**: Metrics derived from events, not parallel state
2. **Efficiency**: Bounded memory through intelligent pruning  
3. **Extensibility**: Same infrastructure scales from metrics to ML
4. **Simplicity**: One pattern (events + observers) solves multiple problems

The key insight is that portfolio metrics are just a projection of the event stream. By embracing this fully and adding memory-aware pruning, we get a system that's both powerful today and ready for tomorrow's requirements.

## References

- `/src/core/events/refactor.md` - Implementation details
- `/src/core/events/observer.md` - Observer pattern examples
- `/docs/architecture/data-mining-architecture.md` - Future data mining plans
- `/src/core/events/` - Core implementation

---

*"The best architectures solve multiple problems with a single, elegant abstraction. Event tracing with intelligent pruning is that abstraction for ADMF-PC."*