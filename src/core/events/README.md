# Event System Architecture

## Overview

The ADMF-PC event system provides a clean, composable infrastructure for event-driven communication between containers. Built on Protocol + Composition principles, it serves as the **single source of truth** for all system data while maintaining minimal memory footprint through intelligent retention policies.

## Core Design Principles

### 1. **Events as Data**
Every piece of data in the system flows through events:
- Market data → `BAR` events
- Calculations → `FEATURES` events  
- Decisions → `SIGNAL` events
- Executions → `ORDER`, `FILL` events
- State changes → `POSITION_OPEN`, `POSITION_CLOSE` events

### 2. **Container Isolation**
Each container gets its own `EventBus` instance:
```python
container = Container(container_id="portfolio_1")
container.event_bus = EventBus(container.id)  # Isolated bus
```

### 3. **Composition Over Inheritance**
Functionality is added via observers, not subclassing:
```python
# Pure event bus
bus = EventBus("my_bus")

# Add tracing via composition
tracer = EventTracer(trace_id, storage)
bus.attach_observer(tracer)

# Add metrics via composition  
metrics = MetricsObserver()
bus.attach_observer(metrics)
```

## Memory-Efficient Architecture

### The Challenge
Running 1000+ parallel portfolio containers during optimization could consume massive memory if every event is retained.

### The Solution: Configurable Retention

#### 1. **Minimal Mode** (Portfolio Containers)
Only tracks open positions, automatically pruning closed trades:

```yaml
# In container config
event_tracing:
  mode: minimal
  events_to_trace: ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL']
  retention_policy: trade_complete  # Auto-prune when positions close
```

**Memory Profile:**
- Position opens → Event stored temporarily
- Position closes → Metrics updated, events pruned
- Result: Only N open positions in memory at once

#### 2. **Full Mode** (Analysis Containers)
Retains complete history for signal analysis:

```yaml
event_tracing:
  mode: full
  events_to_trace: ALL
  storage_backend: disk  # Stream to disk if needed
```

#### 3. **Sliding Window** (Feature Containers)
Keeps recent history only:

```yaml
event_tracing:
  mode: sliding_window
  max_events: 1000
  events_to_trace: ['BAR', 'FEATURES']
```

## Container Integration

### How Containers Use Events for Metrics

Portfolio containers demonstrate the pattern:

```python
class PortfolioContainer:
    def __init__(self, config):
        self.event_bus = EventBus(self.container_id)
        
        # Permanent metrics storage
        self.metrics = PortfolioMetrics()
        
        # Configure minimal event tracing
        self.event_bus.enable_tracing({
            'retention_policy': 'trade_complete',
            'events_to_trace': ['POSITION_OPEN', 'POSITION_CLOSE']
        })
    
    def on_position_close(self, event):
        # 1. Find correlated POSITION_OPEN event
        open_event = self.event_bus.tracer.find_correlated(
            event.correlation_id, 
            'POSITION_OPEN'
        )
        
        # 2. Calculate metrics from event pair
        pnl = event.exit_price - open_event.entry_price
        duration = event.timestamp - open_event.timestamp
        
        # 3. Update permanent metrics
        self.metrics.total_trades += 1
        self.metrics.total_pnl += pnl
        
        # 4. Events automatically pruned!
        # Memory freed, only metrics remain
```

### Results Extraction

After execution, containers provide metrics computed from events:

```python
def get_results(self):
    return {
        'container_id': self.container_id,
        'correlation_id': self.correlation_id,
        'metrics': self.metrics.to_dict(),  # Computed from events
        'memory_usage': len(self.event_bus.tracer.events)  # Should be minimal
    }
```

## Configuration-Driven Behavior

Everything is configured via YAML:

```yaml
containers:
  - id: portfolio_1
    type: portfolio
    event_tracing:
      enabled: true
      mode: minimal
      retention_policy: trade_complete
      events_to_trace: 
        - POSITION_OPEN
        - POSITION_CLOSE
        - FILL
      storage_path: ./results/portfolio_1/
      
  - id: signal_analyzer
    type: analysis
    event_tracing:
      enabled: true
      mode: full
      events_to_trace: ALL
      storage_backend: disk
      storage_config:
        compression: true
        max_file_size_mb: 100
```

## Event Flow Architecture

```
Data Container          Portfolio Container       Execution Container
     |                         |                          |
     | BAR event              |                          |
     |----------------------->|                          |
     |                        |                          |
     |                   Generate Signal                 |
     |                        |                          |
     |                        | ORDER_REQUEST            |
     |                        |------------------------->|
     |                        |                          |
     |                        |          ORDER           |
     |                        |<-------------------------|
     |                        |                          |
     |                        | Track Position           |
     |                        | (POSITION_OPEN)          |
     |                        |                          |
     |                        |          FILL            |
     |                        |<-------------------------|
     |                        |                          |
     |                   Update Metrics                  |
     |                   Prune Events                    |
     |                        |                          |
```

## Key Benefits

### 1. **Unified Data Layer**
- Single source of truth for all data
- Reproducible results from event replay
- No parallel data structures

### 2. **Massive Scalability**
- Minimal memory mode for portfolio containers
- 1000+ containers can run in parallel
- Each tracks only its open positions

### 3. **Flexible Analysis**
- Full tracing where needed (signals, features)
- Minimal tracing where not (portfolios)
- Configurable per container type

### 4. **Clean Architecture**
- EventBus knows nothing about tracing
- Tracing knows nothing about portfolios
- Storage knows nothing about trading

## Common Patterns

### Pattern 1: Ephemeral Event Memory
```python
# Events as temporary working memory
tracer.trace_event(position_open)  # Store temporarily
# ... time passes ...
tracer.trace_event(position_close)  # Trigger calculation
metrics.update(calculate_pnl(...))  # Extract value
tracer.prune_correlated(...)       # Free memory
```

### Pattern 2: Streaming Results
```python
# For large backtests, stream results to disk
if self.metrics.total_trades % 100 == 0:
    self.save_partial_results()
    self.event_bus.tracer.clear()  # Free memory
```

### Pattern 3: Correlation-Based Analysis
```python
# Use correlation IDs to link related events
signal_event.correlation_id = "trade_123"
order_event.correlation_id = "trade_123"  
fill_event.correlation_id = "trade_123"
close_event.correlation_id = "trade_123"

# Later: analyze complete trade lifecycle
trade_events = tracer.get_events_by_correlation("trade_123")
```

## Storage and Results

### During Execution
- Each container with tracing saves to: `./results/{container_id}/`
- Minimal mode: Only metrics in memory
- Full mode: Events streamed to disk

### After Execution
```
./results/
├── root_container/
│   ├── metrics.json
│   └── events.jsonl.gz (if full tracing)
├── portfolio_1/
│   └── metrics.json (minimal mode = no events)
├── portfolio_2/
│   └── metrics.json
└── signal_analyzer/
    ├── metrics.json
    └── events.jsonl.gz (full tracing)
```

## Configuration Examples

### Selective Container Tracing

#### Only Portfolio Containers Trace
```yaml
# Memory Impact: Minimal - only portfolio containers allocate tracing memory
trace_settings:
  container_settings:
    "portfolio_*":
      enabled: true
      events_to_trace: "ALL"
    "*":  # Everything else
      enabled: false
```

#### All Containers Trace, But Filter Events
```yaml
# Memory Impact: Higher - all containers have tracers but limited events
trace_settings:
  # All containers trace, but filter events
  events_to_trace: ["POSITION_OPEN", "POSITION_CLOSE", "FILL", "PORTFOLIO_UPDATE"]
  retention_policy: "trade_complete"
```

#### Mixed Approach - Optimized Per Container Type
```yaml
# Memory Impact: Optimized - each container type traces only what it needs
trace_settings:
  container_settings:
    "portfolio_*":
      enabled: true
      events_to_trace: ["POSITION_OPEN", "POSITION_CLOSE", "FILL"]
      retention_policy: "trade_complete"  # Auto-prune closed trades
      max_events: 1000  # ~10MB per container
    "strategy_*":
      enabled: true  
      events_to_trace: ["SIGNAL"]  # Only trace signals
      retention_policy: "sliding_window"
      max_events: 100  # ~1MB per container
    "data_*":
      enabled: false  # No memory allocated
```

### Memory Optimization Patterns

#### Ultra-Minimal Portfolio Tracing
```yaml
# Memory Impact: ~1MB per portfolio container
# Perfect for 1000+ parallel portfolios
portfolio_settings:
  event_tracing: ["POSITION_OPEN", "POSITION_CLOSE"]
  retention_policy: "trade_complete"  # Auto-prunes when trades close
  sliding_window_size: 0  # Don't keep any history
```

**Memory Calculation:**
- Average position event: ~1KB
- Max open positions: 20
- Memory per container: ~20KB active + overhead = ~1MB
- 1000 containers: ~1GB total

### Wildcard Pattern Matching

#### Multi-Phase Optimization
```yaml
trace_settings:
  container_settings:
    "portfolio_phase1_*":  # Exploration phase - minimal tracing
      enabled: true
      events_to_trace: ["FILL", "POSITION_CLOSE"]
      retention_policy: "trade_complete"
      max_events: 100
    "portfolio_phase2_*":  # Validation phase - full tracing
      enabled: true
      events_to_trace: "ALL"
      retention_policy: "sliding_window"
      max_events: 10000
    "portfolio_final_*":   # Final candidates - persist everything
      enabled: true
      events_to_trace: "ALL"
      storage_backend: "disk"
```

#### Strategy-Specific Tracing
```yaml
trace_settings:
  container_settings:
    "backtest_*_momentum_*":  # Momentum strategies
      enabled: true
      events_to_trace: ["BAR", "SIGNAL", "FILL"]
    "backtest_*_hmm_*":  # HMM classifier strategies
      enabled: true
      events_to_trace: ["REGIME_CHANGE", "SIGNAL", "POSITION_CLOSE"]
    "backtest_*_ml_*":  # ML strategies - need more data
      enabled: true
      events_to_trace: "ALL"
      storage_backend: "disk"  # Too much for memory
```

### Environment-Based Configuration

#### Development vs Production
```yaml
# Development - Full visibility
development:
  trace_settings:
    events_to_trace: "ALL"
    retention_policy: "all"
    max_events: 100000
    storage_backend: "memory"  # Fast access for debugging

# Production - Minimal overhead
production:
  trace_settings:
    container_settings:
      "portfolio_*":
        enabled: true
        events_to_trace: ["FILL", "RISK_BREACH", "ERROR"]
        retention_policy: "sliding_window"
        max_events: 1000
      "execution_*":
        enabled: true
        events_to_trace: ["ORDER", "FILL", "ERROR"]
        retention_policy: "sliding_window"
        max_events: 5000
      "*":
        enabled: false  # No tracing for other containers
```

### Memory Impact Guidelines

| Configuration | Memory per Container | Use Case |
|--------------|---------------------|----------|
| No tracing | 0 MB | Data containers, pure compute |
| Minimal portfolio | 1-2 MB | Large-scale optimization (1000+ portfolios) |
| Sliding window (100 events) | 1-5 MB | Signal analysis, feature tracking |
| Sliding window (1000 events) | 10-50 MB | Detailed analysis, debugging |
| Full memory tracing | 100+ MB | Development, single backtest analysis |
| Disk streaming | 10-20 MB buffer | Long-running production systems |

### Advanced Patterns

#### Conditional Tracing Based on Performance
```yaml
# Enable detailed tracing only for profitable strategies
trace_settings:
  container_settings:
    "portfolio_*":
      enabled: true
      events_to_trace: ["POSITION_CLOSE"]
      # Dynamically enable full tracing if profitable
      dynamic_rules:
        - condition: "metrics.total_pnl > 0"
          action: 
            events_to_trace: "ALL"
            storage_backend: "disk"
```

#### Hierarchical Configuration
```yaml
# Base settings with overrides
trace_settings:
  # Default for all containers
  default:
    enabled: false
    
  # Override by container type
  overrides:
    - pattern: "portfolio_*"
      enabled: true
      events_to_trace: ["POSITION_OPEN", "POSITION_CLOSE"]
      
    # More specific pattern takes precedence
    - pattern: "portfolio_aggressive_*"
      events_to_trace: "ALL"  # These need more monitoring
      retention_policy: "all"
```

## Best Practices

### 1. **Choose Appropriate Retention**
- Portfolio containers: `trade_complete` or `minimal`
- Feature containers: `sliding_window`
- Analysis containers: `full` with disk storage
- Data containers: No tracing

### 2. **Use Correlation IDs**
- Link related events across containers
- Enable trade lifecycle analysis
- Support event pruning

### 3. **Monitor Memory Usage**
```python
# In container
def get_memory_stats(self):
    return {
        'events_in_memory': self.event_bus.tracer.storage.count(),
        'open_positions': len(self.open_positions),
        'metrics_size': sys.getsizeof(self.metrics)
    }
```

### 4. **Batch Operations**
- Update metrics in batches
- Prune events periodically
- Stream results incrementally

### 5. **Pattern-Based Configuration**
- Use wildcards to configure groups of containers
- More specific patterns override general ones
- Test patterns with small runs first

## Summary

The event system provides a powerful, memory-efficient foundation for ADMF-PC's data architecture. By using events as both the communication mechanism AND the data layer, with configurable retention policies, we achieve:

- **Correctness**: Single source of truth
- **Scalability**: Minimal memory footprint
- **Flexibility**: Full tracing when needed
- **Simplicity**: Clean protocol-based design

This design enables running thousands of parallel containers during optimization while maintaining complete data fidelity and minimal memory usage.