# Step 10.8: Event Tracing & Data Mining

## 🎯 Objectives

Implement comprehensive event tracing and data mining infrastructure based on the [Data Mining Architecture](../../architecture/data-mining-architecture.md) to enable:

1. Complete event lineage tracking (signal → order → fill → P&L)
2. Performance pattern discovery and validation
3. Real-time failure pattern detection
4. Institutional-grade trade analytics
5. Foundation for multi-portfolio debugging

## 📋 Prerequisites

Before starting this step:
- [ ] Step 10.1-10.7 complete (or at least basic backtesting working)
- [ ] PostgreSQL or TimescaleDB available
- [ ] Understanding of correlation IDs and event causation
- [ ] Familiarity with Parquet storage format

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Event Tracing Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Live Events → TracedEventBus → EventStore → Analytics DB   │
│       ↓              ↓              ↓             ↓          │
│  Correlation ID   Lineage     Parquet Files   SQL Metrics   │
│       ↓              ↓              ↓             ↓          │
│  Event Grouping  Causation   Compression    Fast Queries    │
│       ↓              ↓              ↓             ↓          │
│   Debug Power   Root Cause   Long Storage  Pattern Mining   │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Implementation Guide

### 1. Enhanced Event Structure

```python
# src/core/events/traced_event.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class TracedEvent:
    """Event with full lineage and performance tracking"""
    # Identity
    event_id: str
    event_type: str
    timestamp: datetime
    
    # Lineage
    correlation_id: str      # Groups related events (entire backtest)
    causation_id: str        # What caused this event
    source_container: str    # Who emitted it
    
    # Performance tracking
    created_at: datetime
    emitted_at: datetime
    received_at: Optional[datetime]
    processed_at: Optional[datetime]
    
    # Payload
    data: Dict[str, Any]
    
    # Metadata
    version: str = "1.0"
    sequence_number: int = 0
    partition_key: str = ""  # For future sharding
```

### 2. Event Store Implementation

```python
# src/core/events/event_store.py
class EventStore:
    """Stores events in Parquet format for efficient analysis"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.buffer = []
        self.buffer_size = 1000
        
    def store_event(self, event: TracedEvent):
        """Buffer events for batch writing"""
        self.buffer.append(self._event_to_dict(event))
        if len(self.buffer) >= self.buffer_size:
            self.flush()
            
    def flush(self):
        """Write buffered events to Parquet"""
        # Partition by date and event type for efficiency
        # Compress with Snappy for balance of speed/size
```

### 3. Event Tracer Integration

```python
# src/core/events/event_tracer.py
class EventTracer:
    """Traces event flow through the system"""
    
    def trace_event(self, event: Event, source: Container):
        """Add tracing metadata to events"""
        traced = TracedEvent(
            event_id=self.generate_event_id(),
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            correlation_id=self.get_correlation_id(),
            causation_id=self.get_causation_id(),
            source_container=source.metadata.name,
            created_at=datetime.now(),
            data=event.payload
        )
        self.event_store.store_event(traced)
        return traced
        
    def find_signal_to_fill_path(self, fill_event_id: str):
        """Trace back from fill to original signal"""
        # Critical for debugging and attribution
```

### 4. Smart Event Sampling

```python
# src/core/events/event_sampler.py
class IntelligentEventSampler:
    """Sample events while preserving critical information"""
    
    def sample_events(self, events: List[TracedEvent], rate=0.1):
        # Always keep: SIGNAL, ORDER, FILL, RISK_BREACH
        # Keep context window around critical events
        # Adaptively sample high-activity periods
```

### 5. SQL Analytics Schema

```sql
-- Optimization runs table
CREATE TABLE optimization_runs (
    run_id UUID PRIMARY KEY,
    correlation_id VARCHAR(100) UNIQUE,
    strategy_type VARCHAR(50),
    parameters JSONB,
    
    -- Performance metrics
    sharpe_ratio DECIMAL(5,3),
    max_drawdown DECIMAL(5,3),
    total_return DECIMAL(10,3),
    
    -- Market context
    market_regime VARCHAR(20),
    volatility_regime VARCHAR(20),
    
    -- Event linkage
    event_storage_path VARCHAR(500)
);

-- Pattern discovery table
CREATE TABLE discovered_patterns (
    pattern_id UUID PRIMARY KEY,
    pattern_type VARCHAR(50),
    pattern_signature JSONB,
    success_rate DECIMAL(5,3),
    last_validated TIMESTAMP
);
```

### 6. Pattern Mining Pipeline

```python
# src/analytics/pattern_miner.py
class PatternMiner:
    """Discover profitable patterns from event streams"""
    
    def mine_optimization_results(self, criteria):
        # Phase 1: SQL discovery (what worked)
        promising_runs = self.sql_query(criteria)
        
        # Phase 2: Event analysis (why it worked)
        for run in promising_runs:
            events = self.load_events(run.correlation_id)
            patterns = self.extract_patterns(events)
            
        # Phase 3: Pattern validation
        validated = self.validate_patterns(patterns)
        
        return validated
```

## ✅ Implementation Checklist

### Core Infrastructure
- [ ] Create TracedEvent dataclass with all fields
- [ ] Implement EventStore with Parquet backend
- [ ] Add correlation ID generation
- [ ] Integrate EventTracer with existing EventBus
- [ ] Implement smart event sampling

### Storage & Analytics
- [ ] Set up PostgreSQL/TimescaleDB schema
- [ ] Create optimization_runs table
- [ ] Create discovered_patterns table
- [ ] Implement event → SQL ETL pipeline
- [ ] Add indexes for common queries

### Pattern Discovery
- [ ] Implement signal-to-fill path tracing
- [ ] Create latency analysis tools
- [ ] Build pattern extraction algorithms
- [ ] Add pattern validation framework
- [ ] Create failure pattern detection

### Integration
- [ ] Update all containers to use TracedEventBus
- [ ] Add correlation ID to backtest initialization
- [ ] Modify event publishing to include tracing
- [ ] Update results storage to include SQL metrics
- [ ] Create debugging utilities

### Testing
- [ ] Test event lineage tracking
- [ ] Verify zero event loss
- [ ] Benchmark performance overhead (<5%)
- [ ] Test pattern discovery accuracy
- [ ] Validate storage compression

## 🧪 Testing Requirements

### Unit Tests
```python
def test_event_lineage():
    """Verify we can trace from fill back to signal"""
    # Create signal → order → fill chain
    # Verify complete lineage preserved
    
def test_smart_sampling():
    """Test intelligent event sampling"""
    # Verify critical events always kept
    # Check context window preservation
    # Validate compression ratios
```

### Integration Tests
```python
def test_end_to_end_tracing():
    """Full backtest with event tracing"""
    # Run complete backtest
    # Verify all events captured
    # Check SQL metrics match
    # Test pattern discovery
```

### Performance Tests
- Event tracing overhead: <5% on backtest time
- Storage efficiency: <100MB per million events
- Query performance: <1s for correlation_id lookup
- Pattern matching: <10ms per pattern

## 🎯 Success Criteria

### Functionality
- [ ] Can trace any fill back to originating signal
- [ ] All critical events (SIGNAL, ORDER, FILL) captured
- [ ] Pattern discovery identifies known strategies
- [ ] Failure patterns detected before losses

### Performance
- [ ] Less than 5% overhead on backtest execution
- [ ] Event storage compressed 10:1 or better
- [ ] SQL queries return in <1 second
- [ ] Can handle 1M+ events per backtest

### Quality
- [ ] Zero event loss under all conditions
- [ ] Complete lineage for every trade
- [ ] Accurate latency measurements
- [ ] Reproducible pattern discovery

## 🔗 Integration Points

### With Existing System
- EventBus → TracedEventBus
- Container initialization → Add tracer
- Results storage → Include SQL metrics
- Coordinator → Set correlation ID

### With Future Steps
- Multi-portfolio → Portfolio-specific correlation IDs
- ML integration → Pattern features for models
- Production → Real-time pattern monitoring

## 📊 Pattern Discovery Examples

### Success Patterns
```python
# Discovery: "Momentum works best with low entry volatility"
pattern = {
    'type': 'entry_condition',
    'strategy': 'momentum',
    'condition': 'vix < 20',
    'success_rate': 0.73,
    'avg_return': 0.025
}
```

### Failure Patterns
```python
# Discovery: "Mean reversion fails during regime transitions"
pattern = {
    'type': 'failure_condition',
    'strategy': 'mean_reversion',
    'condition': 'regime_change_detected',
    'failure_rate': 0.78,
    'avg_loss': -0.032
}
```

## 🚨 Common Pitfalls

### 1. Event Loss
**Problem**: Events lost during high-frequency periods
**Solution**: Use buffering and async writes

### 2. Storage Growth
**Problem**: Event storage grows unbounded
**Solution**: Implement retention policies and compression

### 3. Query Performance
**Problem**: Slow pattern discovery queries
**Solution**: Proper indexing and materialized views

### 4. Memory Usage
**Problem**: Large event buffers consume memory
**Solution**: Streaming processing and smart sampling

## 📈 Performance Optimization

### Event Processing
- Batch writes to Parquet
- Async event storage
- Circular buffers for streaming
- Compression for older events

### Query Optimization
- Partition by date and event type
- Create indexes on correlation_id
- Use materialized views for metrics
- Cache frequently accessed patterns

## 🎯 Next Steps

After completing event tracing:
1. Use pattern discovery to improve strategies
2. Implement real-time pattern monitoring
3. Build automated failure detection
4. Create performance attribution reports
5. Prepare for [Step 10.9: Multi-Portfolio Architecture](step-10.9-multi-portfolio-architecture.md)

## 📚 Additional Resources

- [Data Mining Architecture](../../architecture/data-mining-architecture.md)
- [Event-Driven Architecture](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)
- [Testing Framework](../testing-framework/README.md)
- [SQL Query Optimization Guide](../../references/sql-optimization.md)