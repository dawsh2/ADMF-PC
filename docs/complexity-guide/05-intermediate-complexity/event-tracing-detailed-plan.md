# Phase 1: Event Tracing Infrastructure Implementation Plan

## Overview
Implement comprehensive event tracing based on docs/architecture/data-mining-architecture.md to enable deep analysis and pattern discovery.

## 1.1 Enhanced Event Structure

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
    correlation_id: str      # Groups related events (entire backtest run)
    causation_id: str        # What caused this event
    source_container: str    # Who emitted it
    
    # Performance tracking
    created_at: datetime     # When event was created
    emitted_at: datetime     # When published to bus
    received_at: datetime    # When received by target
    processed_at: datetime   # When processing completed
    
    # Payload
    data: Dict[str, Any]
    
    # Metadata
    version: str = "1.0"
    sequence_number: int = 0
    partition_key: str = ""  # For future sharding
    
    @property
    def latency_ms(self) -> float:
        """Total processing latency in milliseconds"""
        return (self.processed_at - self.created_at).total_seconds() * 1000
    
    @property
    def queue_time_ms(self) -> float:
        """Time spent in event bus"""
        return (self.received_at - self.emitted_at).total_seconds() * 1000
```

## 1.2 Event Store Implementation

```python
# src/core/events/event_store.py
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq

class EventStore:
    """Stores events in Parquet format for efficient analysis"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.buffer_size = 1000
        
    def store_event(self, event: TracedEvent):
        """Store single event"""
        self.buffer.append(self._event_to_dict(event))
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
            
    def flush(self):
        """Write buffered events to Parquet"""
        if not self.buffer:
            return
            
        df = pd.DataFrame(self.buffer)
        
        # Partition by date and event type for efficient queries
        for (date, event_type), group in df.groupby([
            df['timestamp'].dt.date,
            df['event_type']
        ]):
            path = self.base_path / f"date={date}" / f"type={event_type}"
            path.mkdir(parents=True, exist_ok=True)
            
            file_path = path / f"events_{datetime.now().timestamp()}.parquet"
            group.to_parquet(file_path, compression='snappy')
            
        self.buffer.clear()
        
    def query_events(self, 
                    correlation_id: Optional[str] = None,
                    event_type: Optional[str] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Query events with filters"""
        # Use PyArrow for efficient filtering
        filters = []
        if correlation_id:
            filters.append(('correlation_id', '=', correlation_id))
        if event_type:
            filters.append(('event_type', '=', event_type))
            
        dataset = pq.ParquetDataset(
            self.base_path,
            filters=filters,
            use_legacy_dataset=False
        )
        
        df = dataset.read().to_pandas()
        
        # Apply time filters
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
            
        return df.sort_values('timestamp')
```

## 1.3 Event Tracer Integration

```python
# src/core/events/event_tracer.py
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import uuid

class EventTracer:
    """Traces event flow through the system"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.active_traces = {}  # correlation_id -> events
        self.correlation_map = defaultdict(list)
        self.critical_events = {'SIGNAL', 'ORDER', 'FILL', 'RISK_BREACH'}
        
    def create_correlation_id(self) -> str:
        """Generate new correlation ID for a backtest run"""
        return f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
    def trace_event(self, event: Event, source: Container, destination: Optional[Container] = None):
        """Trace event with full context"""
        traced_event = TracedEvent(
            event_id=f"{event.event_type}_{uuid.uuid4().hex[:8]}",
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            correlation_id=event.metadata.get('correlation_id', 'unknown'),
            causation_id=event.metadata.get('causation_id', ''),
            source_container=source.metadata.name,
            created_at=datetime.now(),
            emitted_at=datetime.now(),
            received_at=None,
            processed_at=None,
            data=event.payload,
            sequence_number=self._get_next_sequence(event.metadata.get('correlation_id'))
        )
        
        # Store in active traces
        correlation_id = traced_event.correlation_id
        if correlation_id not in self.active_traces:
            self.active_traces[correlation_id] = deque(maxlen=10000)
        self.active_traces[correlation_id].append(traced_event)
        
        # Store persistently
        self.event_store.store_event(traced_event)
        
        return traced_event
        
    def find_signal_to_fill_path(self, fill_event_id: str) -> List[TracedEvent]:
        """Trace back from fill to original signal"""
        # Load events
        fill_event = self._get_event(fill_event_id)
        if not fill_event:
            return []
            
        correlation_id = fill_event.correlation_id
        events = self.event_store.query_events(correlation_id=correlation_id)
        
        # Build causation chain
        path = []
        current = fill_event
        
        while current:
            path.append(current)
            if current.causation_id:
                current = events[events['event_id'] == current.causation_id].iloc[0] if len(events[events['event_id'] == current.causation_id]) > 0 else None
            else:
                break
                
        return list(reversed(path))
        
    def analyze_latency(self, correlation_id: str) -> Dict[str, Any]:
        """Analyze processing latency for a run"""
        events = self.event_store.query_events(correlation_id=correlation_id)
        
        # Calculate latencies by event type
        latencies = defaultdict(list)
        for _, event in events.iterrows():
            if pd.notna(event['processed_at']) and pd.notna(event['created_at']):
                latency_ms = (event['processed_at'] - event['created_at']).total_seconds() * 1000
                latencies[event['event_type']].append(latency_ms)
                
        # Calculate statistics
        stats = {}
        for event_type, values in latencies.items():
            if values:
                stats[event_type] = {
                    'mean_ms': np.mean(values),
                    'median_ms': np.median(values),
                    'p95_ms': np.percentile(values, 95),
                    'max_ms': max(values),
                    'count': len(values)
                }
                
        return stats
```

## 1.4 Smart Event Sampling

```python
# src/core/events/event_sampler.py
class IntelligentEventSampler:
    """Sample events while preserving critical information"""
    
    def __init__(self):
        self.critical_events = {'SIGNAL', 'ORDER', 'FILL', 'RISK_BREACH', 'PORTFOLIO'}
        self.context_window = 5  # Keep N events before/after critical
        
    def sample_events(self, events: List[TracedEvent], sample_rate: float = 0.1) -> List[TracedEvent]:
        """Smart sampling that preserves important events and context"""
        
        # Always keep critical events
        critical_indices = [
            i for i, e in enumerate(events) 
            if e.event_type in self.critical_events
        ]
        
        # Keep context around critical events
        context_indices = set()
        for idx in critical_indices:
            start = max(0, idx - self.context_window)
            end = min(len(events), idx + self.context_window + 1)
            context_indices.update(range(start, end))
            
        # Sample remaining events based on activity
        remaining = [i for i in range(len(events)) if i not in context_indices]
        
        # Detect high activity periods
        activity_scores = self._calculate_activity_scores(events)
        
        sampled = []
        for idx in remaining:
            # Increase sampling during high activity
            adjusted_rate = sample_rate * (1 + activity_scores[idx])
            if random.random() < adjusted_rate:
                sampled.append(idx)
                
        # Combine all indices
        keep_indices = sorted(context_indices | set(sampled))
        
        return [events[i] for i in keep_indices]
        
    def _calculate_activity_scores(self, events: List[TracedEvent]) -> List[float]:
        """Calculate activity score for each event"""
        scores = []
        window_size = 10
        
        for i, event in enumerate(events):
            # Count events in surrounding window
            start = max(0, i - window_size // 2)
            end = min(len(events), i + window_size // 2)
            
            # More events = higher activity
            window_events = events[start:end]
            event_density = len(window_events) / window_size
            
            # Critical events increase activity score
            critical_count = sum(1 for e in window_events if e.event_type in self.critical_events)
            critical_density = critical_count / len(window_events) if window_events else 0
            
            # Combined score
            score = event_density + critical_density * 2
            scores.append(min(score, 3.0))  # Cap at 3x base rate
            
        return scores
```

## 1.5 Integration with Existing System

```python
# src/core/events/traced_event_bus.py
class TracedEventBus(EventBus):
    """EventBus with integrated tracing"""
    
    def __init__(self, name: str, tracer: Optional[EventTracer] = None):
        super().__init__(name)
        self.tracer = tracer
        self.correlation_id = None
        
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for all events in this bus"""
        self.correlation_id = correlation_id
        
    def publish(self, event: Event, source: Container):
        """Publish event with tracing"""
        # Add correlation ID
        if self.correlation_id:
            event.metadata['correlation_id'] = self.correlation_id
            
        # Add causation ID if this event was triggered by another
        if hasattr(self, '_current_processing_event'):
            event.metadata['causation_id'] = self._current_processing_event.event_id
            
        # Trace if tracer available
        if self.tracer:
            traced_event = self.tracer.trace_event(event, source)
            event.metadata['event_id'] = traced_event.event_id
            
        # Normal publish
        super().publish(event)
```

## 1.6 Testing Infrastructure

```python
# tests/test_event_tracing.py
import pytest
from datetime import datetime, timedelta
import pandas as pd

class TestEventTracing:
    
    def test_event_lineage(self, event_tracer):
        """Test we can trace from fill back to signal"""
        # Create event chain
        signal_event = create_signal_event()
        order_event = create_order_event(causation_id=signal_event.event_id)
        fill_event = create_fill_event(causation_id=order_event.event_id)
        
        # Trace events
        for event in [signal_event, order_event, fill_event]:
            event_tracer.trace_event(event, mock_container())
            
        # Verify lineage
        path = event_tracer.find_signal_to_fill_path(fill_event.event_id)
        
        assert len(path) == 3
        assert path[0].event_type == 'SIGNAL'
        assert path[1].event_type == 'ORDER'
        assert path[2].event_type == 'FILL'
        
    def test_latency_analysis(self, event_tracer):
        """Test latency calculations"""
        correlation_id = event_tracer.create_correlation_id()
        
        # Create events with known latencies
        events = []
        for i in range(100):
            event = TracedEvent(
                event_id=f"test_{i}",
                event_type='BAR',
                timestamp=datetime.now(),
                correlation_id=correlation_id,
                causation_id='',
                source_container='test',
                created_at=datetime.now(),
                emitted_at=datetime.now(),
                received_at=datetime.now() + timedelta(milliseconds=5),
                processed_at=datetime.now() + timedelta(milliseconds=10),
                data={}
            )
            events.append(event)
            event_tracer.trace_event(event, mock_container())
            
        # Analyze latencies
        stats = event_tracer.analyze_latency(correlation_id)
        
        assert 'BAR' in stats
        assert abs(stats['BAR']['mean_ms'] - 10) < 1  # ~10ms latency
        assert stats['BAR']['count'] == 100
        
    def test_smart_sampling(self, event_sampler):
        """Test intelligent event sampling"""
        # Create event stream with critical events
        events = []
        for i in range(1000):
            if i % 100 == 0:
                # Critical event every 100
                event_type = 'SIGNAL'
            else:
                event_type = 'BAR'
                
            event = create_event(event_type=event_type)
            events.append(event)
            
        # Sample at 10% rate
        sampled = event_sampler.sample_events(events, sample_rate=0.1)
        
        # All critical events should be kept
        critical_count = sum(1 for e in sampled if e.event_type == 'SIGNAL')
        assert critical_count == 10  # All 10 signals kept
        
        # Should have context around critical events
        for i, event in enumerate(sampled):
            if event.event_type == 'SIGNAL':
                # Check context window
                window_start = max(0, i - 5)
                window_end = min(len(sampled), i + 6)
                window = sampled[window_start:window_end]
                assert len(window) >= 6  # At least some context
                
        # Total should be more than just critical due to sampling
        assert len(sampled) > 10
        assert len(sampled) < 200  # But much less than original 1000
```

## 1.7 SQL Schema Setup

```sql
-- Create database schema for analytics
CREATE SCHEMA IF NOT EXISTS backtest_analytics;

-- Main optimization runs table (simplified for Phase 1)
CREATE TABLE backtest_analytics.backtest_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Basic info
    config_file VARCHAR(255),
    strategy_type VARCHAR(50),
    
    -- Performance metrics (to be filled by post-processing)
    total_return DECIMAL(10,3),
    sharpe_ratio DECIMAL(5,3),
    max_drawdown DECIMAL(5,3),
    total_trades INTEGER,
    
    -- Event metadata
    event_count INTEGER,
    event_storage_path VARCHAR(500),
    
    -- Status
    status VARCHAR(20) DEFAULT 'running',
    completed_at TIMESTAMP
);

-- Event latency tracking
CREATE TABLE backtest_analytics.event_latencies (
    run_id UUID REFERENCES backtest_analytics.backtest_runs(run_id),
    event_type VARCHAR(50),
    mean_latency_ms DECIMAL(8,2),
    p95_latency_ms DECIMAL(8,2),
    max_latency_ms DECIMAL(8,2),
    event_count INTEGER,
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, event_type)
);

-- Indexes for performance
CREATE INDEX idx_correlation_lookup ON backtest_analytics.backtest_runs(correlation_id);
CREATE INDEX idx_run_status ON backtest_analytics.backtest_runs(status, created_at);
```

## Integration Points

### 1. BacktestContainer Initialization
```python
# In BacktestContainer.__init__
self.event_tracer = EventTracer(
    event_store=EventStore(Path(f"events/{self.correlation_id}"))
)
self.correlation_id = self.event_tracer.create_correlation_id()

# Set correlation ID on all event buses
for container in self.all_containers:
    if hasattr(container, 'event_bus'):
        container.event_bus.set_correlation_id(self.correlation_id)
```

### 2. Event Publishing
```python
# Replace all event publishing with traced version
# Before:
self.event_bus.publish(event)

# After:
self.event_bus.publish(event, source=self)
```

### 3. Results Storage
```python
# At end of backtest
def store_results(self):
    # Flush event store
    self.event_tracer.event_store.flush()
    
    # Calculate and store metrics
    metrics = self.calculate_performance_metrics()
    latencies = self.event_tracer.analyze_latency(self.correlation_id)
    
    # Store in SQL
    self.store_run_metadata(metrics, latencies)
```

## Success Criteria

1. **Event Capture**: 100% of critical events (SIGNAL, ORDER, FILL, PORTFOLIO) captured
2. **Lineage Tracking**: Can trace any fill back to its originating signal
3. **Performance Impact**: Less than 5% overhead on backtest execution time
4. **Storage Efficiency**: Event storage compressed to <100MB per 1M events
5. **Query Performance**: Can retrieve all events for a correlation_id in <1 second

## Next Steps

After Phase 1 is complete, we'll have:
- Complete event lineage for every trade
- Performance metrics for system bottlenecks
- Foundation for pattern discovery
- Ability to debug any unexpected behavior

This sets us up perfectly for Phase 2 (Enhanced Testing & Attribution) and eventually Phase 3 (Multi-Portfolio Support).