# Event Tracing Implementation - Day 1 Action Plan

## 🎯 Today's Goal
Create the core TracedEvent structure and integrate it with the existing EventBus to start capturing event lineage.

## 📋 Day 1 Tasks (4-6 hours)

### 1. Create TracedEvent Structure (30 min)
```bash
# Create the new events module structure
mkdir -p src/core/events/tracing
touch src/core/events/tracing/__init__.py
touch src/core/events/tracing/traced_event.py
```

**File: `src/core/events/tracing/traced_event.py`**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

@dataclass
class TracedEvent:
    """Event with full lineage and performance tracking"""
    # Identity
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Lineage
    correlation_id: str = ""  # Groups all events in a backtest
    causation_id: str = ""    # ID of event that caused this one
    source_container: str = "" # Container that emitted event
    
    # Performance tracking
    created_at: datetime = field(default_factory=datetime.now)
    emitted_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    # Payload
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    version: str = "1.0"
    sequence_number: int = 0
    partition_key: str = ""
    
    @property
    def latency_ms(self) -> float:
        """Total processing latency in milliseconds"""
        if self.processed_at and self.created_at:
            return (self.processed_at - self.created_at).total_seconds() * 1000
        return 0.0
```

### 2. Create Event Tracer (45 min)
**File: `src/core/events/tracing/event_tracer.py`**
```python
import logging
from typing import Dict, List, Optional
from collections import defaultdict, deque
import uuid
from datetime import datetime

from ..types import Event
from .traced_event import TracedEvent

logger = logging.getLogger(__name__)

class EventTracer:
    """Traces event flow through the system"""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.traced_events = deque(maxlen=10000)  # Keep last 10k events in memory
        self.event_index = {}  # event_id -> TracedEvent for fast lookup
        self.sequence_counter = 0
        
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for this backtest run"""
        return f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
    def trace_event(self, event: Event, source_container: str) -> TracedEvent:
        """Convert regular event to traced event"""
        self.sequence_counter += 1
        
        traced = TracedEvent(
            event_id=f"{event.event_type.value}_{uuid.uuid4().hex[:8]}",
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            correlation_id=self.correlation_id,
            causation_id=event.metadata.get('causation_id', ''),
            source_container=source_container,
            created_at=datetime.now(),
            data=event.payload,
            sequence_number=self.sequence_counter
        )
        
        # Store in memory for quick access
        self.traced_events.append(traced)
        self.event_index[traced.event_id] = traced
        
        # Add event_id to original event for causation tracking
        event.metadata['event_id'] = traced.event_id
        event.metadata['correlation_id'] = self.correlation_id
        
        return traced
```

### 3. Create TracedEventBus (1 hour)
**File: `src/core/events/tracing/traced_event_bus.py`**
```python
from typing import Optional, Callable, Any
import logging

from ..event_bus import EventBus
from ..types import Event, EventType
from .event_tracer import EventTracer
from ...containers.protocols import Container

logger = logging.getLogger(__name__)

class TracedEventBus(EventBus):
    """EventBus with integrated event tracing"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.tracer: Optional[EventTracer] = None
        self._current_processing_event: Optional[str] = None
        
    def set_tracer(self, tracer: EventTracer):
        """Attach event tracer to this bus"""
        self.tracer = tracer
        logger.info(f"EventBus '{self.name}' now tracing with correlation_id: {tracer.correlation_id}")
        
    def publish(self, event: Event, source: Optional[Container] = None):
        """Publish event with tracing"""
        # Add causation if we're processing another event
        if self._current_processing_event:
            event.metadata['causation_id'] = self._current_processing_event
            
        # Trace the event if tracer attached
        if self.tracer and source:
            traced = self.tracer.trace_event(event, source.metadata.name)
            traced.emitted_at = datetime.now()
            
        # Normal publish
        super().publish(event)
        
    def _dispatch_event(self, event: Event, handler: Callable):
        """Override to track processing context"""
        old_event = self._current_processing_event
        self._current_processing_event = event.metadata.get('event_id')
        
        try:
            # Mark received time
            if self.tracer and self._current_processing_event:
                if self._current_processing_event in self.tracer.event_index:
                    self.tracer.event_index[self._current_processing_event].received_at = datetime.now()
                    
            # Process event
            handler(event)
            
            # Mark processed time
            if self.tracer and self._current_processing_event:
                if self._current_processing_event in self.tracer.event_index:
                    self.tracer.event_index[self._current_processing_event].processed_at = datetime.now()
                    
        finally:
            self._current_processing_event = old_event
```

### 4. Integration Test (1 hour)
**File: `tests/test_event_tracing/test_basic_tracing.py`**
```python
import pytest
from datetime import datetime
import time

from src.core.events.types import Event, EventType
from src.core.events.tracing.traced_event_bus import TracedEventBus
from src.core.events.tracing.event_tracer import EventTracer
from src.core.containers.universal import UniversalContainer

def test_event_lineage():
    """Test we can trace event causation"""
    # Setup
    tracer = EventTracer()
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    # Create mock containers
    strategy_container = UniversalContainer(name="StrategyContainer")
    strategy_container.event_bus = bus
    
    risk_container = UniversalContainer(name="RiskContainer") 
    risk_container.event_bus = bus
    
    # Track received events
    received_events = []
    
    def risk_handler(event):
        received_events.append(event)
        # Risk container creates ORDER from SIGNAL
        if event.event_type == EventType.SIGNAL:
            order_event = Event(
                event_type=EventType.ORDER,
                timestamp=datetime.now(),
                payload={"action": "BUY", "quantity": 100}
            )
            bus.publish(order_event, source=risk_container)
    
    # Subscribe risk container to signals
    bus.subscribe(EventType.SIGNAL, risk_handler)
    
    # Strategy emits signal
    signal_event = Event(
        event_type=EventType.SIGNAL,
        timestamp=datetime.now(),
        payload={"symbol": "SPY", "strength": 0.8}
    )
    
    bus.publish(signal_event, source=strategy_container)
    
    # Give time for processing
    time.sleep(0.1)
    
    # Verify causation chain
    assert len(tracer.traced_events) == 2
    
    signal_traced = tracer.traced_events[0]
    order_traced = tracer.traced_events[1]
    
    # Order should have signal as causation
    assert order_traced.causation_id == signal_traced.event_id
    assert signal_traced.source_container == "StrategyContainer"
    assert order_traced.source_container == "RiskContainer"
    
def test_latency_tracking():
    """Test event processing latency measurement"""
    tracer = EventTracer()
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    container = UniversalContainer(name="TestContainer")
    container.event_bus = bus
    
    # Slow handler
    def slow_handler(event):
        time.sleep(0.05)  # 50ms delay
        
    bus.subscribe(EventType.BAR, slow_handler)
    
    # Publish event
    event = Event(
        event_type=EventType.BAR,
        timestamp=datetime.now(),
        payload={"price": 100}
    )
    
    bus.publish(event, source=container)
    time.sleep(0.1)
    
    # Check latency
    traced = tracer.traced_events[0]
    assert traced.latency_ms >= 50  # At least 50ms
    assert traced.latency_ms < 100  # But not too long
```

### 5. Update Existing Containers (1.5 hours)

We need to update key containers to use TracedEventBus. Start with:

**Update `src/execution/containers_pipeline.py`:**
```python
# In BacktestContainer.__init__
from ...core.events.tracing.traced_event_bus import TracedEventBus
from ...core.events.tracing.event_tracer import EventTracer

class BacktestContainer(UniversalContainer):
    def __init__(self, config: Dict[str, Any]):
        # Create traced event bus
        event_bus = TracedEventBus("backtest_main")
        super().__init__(
            name="BacktestContainer",
            config=config,
            event_bus=event_bus
        )
        
        # Initialize tracer
        self.event_tracer = EventTracer()
        event_bus.set_tracer(self.event_tracer)
        
        # Store correlation ID for reporting
        self.correlation_id = self.event_tracer.correlation_id
        logger.info(f"Backtest starting with correlation_id: {self.correlation_id}")
```

### 6. Run Integration Test (30 min)

```bash
# Run our existing multi-strategy test with tracing
python main.py --config config/multi_strategy_test.yaml --bars 50

# Check that we see correlation_id in logs
# Verify no performance degradation
```

### 7. Create Simple Analysis Script (30 min)

**File: `analyze_event_trace.py`**
```python
"""Quick script to analyze traced events from a backtest run"""
from src.core.events.tracing.event_tracer import EventTracer

def analyze_latest_run(tracer: EventTracer):
    """Analyze events from the tracer"""
    print(f"\nEvent Trace Analysis")
    print(f"Correlation ID: {tracer.correlation_id}")
    print(f"Total Events: {len(tracer.traced_events)}")
    
    # Count by type
    event_types = {}
    for event in tracer.traced_events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    print("\nEvent Counts by Type:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    
    # Find signal->fill chains
    signals = [e for e in tracer.traced_events if e.event_type == "SIGNAL"]
    print(f"\nSignals Generated: {len(signals)}")
    
    # Trace each signal
    for signal in signals:
        chain = trace_forward(signal, tracer)
        if any(e.event_type == "FILL" for e in chain):
            print(f"\nSignal {signal.event_id} -> Fill (chain length: {len(chain)})")

def trace_forward(event, tracer):
    """Trace events forward from given event"""
    chain = [event]
    
    # Find all events caused by this one
    for traced in tracer.traced_events:
        if traced.causation_id == event.event_id:
            chain.extend(trace_forward(traced, tracer))
            
    return chain

if __name__ == "__main__":
    # This will be integrated with backtest results
    print("Run a backtest first, then we'll analyze its events")
```

## 🎯 End of Day 1 Goals

By end of day, we should have:
- [x] TracedEvent dataclass created
- [x] EventTracer tracking events in memory
- [x] TracedEventBus publishing traced events
- [x] Basic integration test passing
- [x] Correlation IDs in backtest logs
- [x] Simple analysis showing event counts

## 🚀 Tomorrow's Plan

Day 2 will focus on:
1. Event persistence to Parquet
2. SQL schema setup
3. Smart event sampling
4. Performance optimization

## 💡 Quick Wins

If we finish early, we can:
1. Add more event metadata (portfolio values at each event)
2. Create a simple visualization of event flow
3. Add latency histograms
4. Test with larger backtests

## 🧪 Testing Commands

```bash
# Run unit tests
pytest tests/test_event_tracing/test_basic_tracing.py -v

# Run integration with existing backtest
python main.py --config config/multi_strategy_test.yaml --bars 50

# Verify no performance regression
time python main.py --config config/multi_strategy_test.yaml --bars 500
```

## 📝 Notes

- Start simple - just trace in memory
- Focus on correctness over performance initially
- Ensure zero impact on existing functionality
- Build incrementally and test often