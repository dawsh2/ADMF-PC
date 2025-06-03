"""
Basic tests for event tracing functionality

These tests verify:
1. Event lineage tracking (causation chains)
2. Latency measurement
3. Correlation ID propagation
4. Integration with existing event bus
"""

from datetime import datetime
import time
from typing import Any

from src.core.events.types import Event, EventType
from src.core.events.tracing.traced_event_bus import TracedEventBus
from src.core.events.tracing.event_tracer import EventTracer


class MockContainer:
    """Mock container that implements the minimal protocol for testing"""
    def __init__(self, name: str):
        self.metadata = type('Metadata', (), {'name': name})()


def test_event_lineage():
    """Test we can trace event causation through the system"""
    # Setup
    tracer = EventTracer()
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    # Create mock containers
    strategy_container = MockContainer("StrategyContainer")
    risk_container = MockContainer("RiskContainer")
    
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
    
    # Verify we can trace the chain
    chain = tracer.trace_causation_chain(order_traced.event_id)
    assert len(chain) == 2
    assert chain[0].event_id == signal_traced.event_id
    assert chain[1].event_id == order_traced.event_id
    

def test_latency_tracking():
    """Test event processing latency measurement"""
    tracer = EventTracer()
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    container = MockContainer("TestContainer")
    
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
    assert traced.processing_time_ms >= 50  # Processing took at least 50ms
    

def test_correlation_id_propagation():
    """Test that correlation IDs are properly propagated"""
    # Create tracer with specific correlation ID
    correlation_id = "test_backtest_20240101_123456"
    tracer = EventTracer(correlation_id=correlation_id)
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    container = MockContainer("TestContainer")
    
    # Publish event
    event = Event(
        event_type=EventType.SIGNAL,
        timestamp=datetime.now(),
        payload={"test": "data"}
    )
    
    bus.publish(event, source=container)
    
    # Check correlation ID
    traced = tracer.traced_events[0]
    assert traced.correlation_id == correlation_id
    assert event.metadata['correlation_id'] == correlation_id
    

def test_event_type_counting():
    """Test event type statistics tracking"""
    tracer = EventTracer()
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    container = MockContainer("TestContainer")
    
    # Publish various event types
    for _ in range(5):
        bus.publish(Event(EventType.BAR, datetime.now(), {}), source=container)
    
    for _ in range(3):
        bus.publish(Event(EventType.SIGNAL, datetime.now(), {}), source=container)
        
    for _ in range(2):
        bus.publish(Event(EventType.ORDER, datetime.now(), {}), source=container)
    
    # Check statistics
    summary = tracer.get_summary()
    assert summary['total_events'] == 10
    assert summary['event_counts']['BAR'] == 5
    assert summary['event_counts']['SIGNAL'] == 3
    assert summary['event_counts']['ORDER'] == 2
    assert summary['container_counts']['TestContainer'] == 10
    

def test_nested_event_handling():
    """Test handling of nested event publication"""
    tracer = EventTracer()
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    container = MockContainer("TestContainer")
    
    # Handler that publishes another event
    def nested_handler(event):
        if event.event_type == EventType.SIGNAL:
            # This handler publishes an ORDER event
            order_event = Event(
                event_type=EventType.ORDER,
                timestamp=datetime.now(),
                payload={"nested": True}
            )
            bus.publish(order_event, source=container)
            
            # And then publishes a FILL event
            fill_event = Event(
                event_type=EventType.FILL,
                timestamp=datetime.now(),
                payload={"nested": True, "level": 2}
            )
            bus.publish(fill_event, source=container)
    
    bus.subscribe(EventType.SIGNAL, nested_handler)
    bus.subscribe(EventType.ORDER, lambda e: None)  # Just to process it
    bus.subscribe(EventType.FILL, lambda e: None)   # Just to process it
    
    # Publish initial event
    signal = Event(
        event_type=EventType.SIGNAL,
        timestamp=datetime.now(),
        payload={"initial": True}
    )
    bus.publish(signal, source=container)
    
    # Give time for processing
    time.sleep(0.1)
    
    # Should have 3 events total
    assert len(tracer.traced_events) == 3
    
    # Check causation chain
    signal_traced = tracer.traced_events[0]
    order_traced = tracer.traced_events[1]
    fill_traced = tracer.traced_events[2]
    
    assert order_traced.causation_id == signal_traced.event_id
    assert fill_traced.causation_id == signal_traced.event_id  # Both caused by signal
    

def test_tracer_memory_limit():
    """Test that tracer respects memory limits"""
    # Create tracer with small limit
    tracer = EventTracer(max_events=100)
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    container = MockContainer("TestContainer")
    
    # Publish more events than limit
    for i in range(150):
        event = Event(
            event_type=EventType.BAR,
            timestamp=datetime.now(),
            payload={"index": i}
        )
        bus.publish(event, source=container)
    
    # Should only have last 100 events
    assert len(tracer.traced_events) == 100
    assert len(tracer.event_index) == 100
    
    # First event should be index 50 (0-49 were dropped)
    first_event = tracer.traced_events[0]
    assert first_event.data['index'] == 50
    

def test_find_events_by_type():
    """Test finding events by type"""
    tracer = EventTracer()
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    container = MockContainer("TestContainer")
    
    # Publish mixed events
    bus.publish(Event(EventType.BAR, datetime.now(), {}), source=container)
    bus.publish(Event(EventType.SIGNAL, datetime.now(), {}), source=container)
    bus.publish(Event(EventType.BAR, datetime.now(), {}), source=container)
    bus.publish(Event(EventType.ORDER, datetime.now(), {}), source=container)
    bus.publish(Event(EventType.SIGNAL, datetime.now(), {}), source=container)
    
    # Find by type
    bars = tracer.find_events_by_type('BAR')
    signals = tracer.find_events_by_type('SIGNAL')
    orders = tracer.find_events_by_type('ORDER')
    
    assert len(bars) == 2
    assert len(signals) == 2
    assert len(orders) == 1
    

def test_latency_statistics():
    """Test latency statistics calculation"""
    tracer = EventTracer()
    bus = TracedEventBus("test_bus")
    bus.set_tracer(tracer)
    
    container = MockContainer("TestContainer")
    
    # Create handlers with different delays
    def fast_handler(event):
        time.sleep(0.01)  # 10ms
        
    def slow_handler(event):
        time.sleep(0.05)  # 50ms
    
    bus.subscribe(EventType.BAR, fast_handler)
    bus.subscribe(EventType.SIGNAL, slow_handler)
    
    # Publish events
    for _ in range(3):
        bus.publish(Event(EventType.BAR, datetime.now(), {}), source=container)
        
    for _ in range(2):
        bus.publish(Event(EventType.SIGNAL, datetime.now(), {}), source=container)
    
    time.sleep(0.2)  # Wait for processing
    
    # Get statistics
    stats = tracer.calculate_latency_stats()
    
    # BAR events should be fast
    assert 'BAR' in stats
    assert stats['BAR']['avg_ms'] >= 10
    assert stats['BAR']['avg_ms'] < 20
    assert stats['BAR']['count'] == 3
    
    # SIGNAL events should be slow
    assert 'SIGNAL' in stats
    assert stats['SIGNAL']['avg_ms'] >= 50
    assert stats['SIGNAL']['avg_ms'] < 60
    assert stats['SIGNAL']['count'] == 2