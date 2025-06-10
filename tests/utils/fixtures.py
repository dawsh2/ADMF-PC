"""
Test fixtures and utilities for ADMF-PC testing.

Provides reusable test data, mock objects, and setup utilities
for all test levels.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock


# === TEST DATA FIXTURES ===

class TestData:
    """Standard test data for consistent testing."""
    
    @staticmethod
    def sample_bar_data(symbol: str = "AAPL", close: float = 100.0) -> Dict[str, Any]:
        """Create sample bar data."""
        return {
            "symbol": symbol,
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def sample_signal_data(symbol: str = "AAPL", direction: str = "BUY") -> Dict[str, Any]:
        """Create sample signal data."""
        return {
            "symbol": symbol,
            "direction": direction,
            "strength": 0.8,
            "strategy_id": "test_strategy",
            "confidence": 0.9
        }
    
    @staticmethod
    def sample_order_data(symbol: str = "AAPL", quantity: int = 100) -> Dict[str, Any]:
        """Create sample order data."""
        return {
            "symbol": symbol,
            "quantity": quantity,
            "side": "BUY",
            "order_type": "MARKET",
            "order_id": str(uuid.uuid4())
        }
    
    @staticmethod
    def sample_fill_data(symbol: str = "AAPL", quantity: int = 100) -> Dict[str, Any]:
        """Create sample fill data."""
        return {
            "symbol": symbol,
            "quantity": quantity,
            "price": 100.50,
            "fill_id": str(uuid.uuid4()),
            "order_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }


class TestTimeSequence:
    """Generate time-sequenced test data."""
    
    def __init__(self, start_time: datetime = None, interval: timedelta = None):
        self.start_time = start_time or datetime.now()
        self.interval = interval or timedelta(minutes=1)
        self.current_time = self.start_time
    
    def next_timestamp(self) -> datetime:
        """Get next timestamp in sequence."""
        timestamp = self.current_time
        self.current_time += self.interval
        return timestamp
    
    def generate_bar_sequence(self, symbol: str, count: int, base_price: float = 100.0) -> List[Dict[str, Any]]:
        """Generate sequence of bar data."""
        bars = []
        price = base_price
        
        for i in range(count):
            # Simple random walk
            price_change = (i % 3 - 1) * 0.5  # -0.5, 0, 0.5 pattern
            price += price_change
            
            bar = {
                "symbol": symbol,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0, 
                "close": price + 0.25,
                "volume": 1000000 + (i * 10000),
                "timestamp": self.next_timestamp().isoformat()
            }
            bars.append(bar)
        
        return bars


# === MOCK OBJECTS ===

class MockEventBus:
    """Mock EventBus for testing."""
    
    def __init__(self):
        self.published_events = []
        self.subscriptions = {}
        self.subscription_counter = 0
    
    def subscribe(self, event_type: str, handler, filter_func=None):
        """Mock subscribe."""
        sub_id = f"sub_{self.subscription_counter}"
        self.subscription_counter += 1
        
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        
        self.subscriptions[event_type].append({
            'id': sub_id,
            'handler': handler,
            'filter': filter_func
        })
        
        return sub_id
    
    def publish(self, event):
        """Mock publish."""
        self.published_events.append(event)
        
        # Call handlers if they exist
        if event.event_type in self.subscriptions:
            for sub in self.subscriptions[event.event_type]:
                try:
                    if sub['filter'] is None or sub['filter'](event):
                        sub['handler'](event)
                except Exception as e:
                    print(f"Mock handler error: {e}")
    
    def clear_published(self):
        """Clear published events for testing."""
        self.published_events.clear()
    
    def get_published_count(self, event_type: str = None) -> int:
        """Get count of published events."""
        if event_type is None:
            return len(self.published_events)
        return len([e for e in self.published_events if e.event_type == event_type])


class MockContainer:
    """Mock Container for testing."""
    
    def __init__(self, container_id: str = None, role: str = "test"):
        self.container_id = container_id or f"test_container_{uuid.uuid4().hex[:8]}"
        self.role = role
        self.event_bus = MockEventBus()
        self.components = {}
        self.state = "created"
    
    def add_component(self, name: str, component: Any):
        """Add component."""
        self.components[name] = component
    
    def get_component(self, name: str):
        """Get component."""
        return self.components.get(name)
    
    def publish_event(self, event, target_scope: str = "local"):
        """Publish event."""
        self.event_bus.publish(event)


class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.container = None
        self.initialized = False
        self.started = False
        self.events_received = []
    
    def initialize(self, container):
        """Initialize with container."""
        self.container = container
        self.initialized = True
    
    def start(self):
        """Start component."""
        self.started = True
    
    def stop(self):
        """Stop component."""
        self.started = False
    
    def get_state(self):
        """Get state."""
        return {
            'name': self.name,
            'initialized': self.initialized,
            'started': self.started,
            'events_received': len(self.events_received)
        }
    
    def on_event(self, event):
        """Event handler."""
        self.events_received.append(event)


# === TEST UTILITIES ===

class EventCollector:
    """Utility to collect and analyze events in tests."""
    
    def __init__(self):
        self.events = []
    
    def handler(self, event):
        """Event handler that collects events."""
        self.events.append(event)
    
    def clear(self):
        """Clear collected events."""
        self.events.clear()
    
    def count(self, event_type: str = None) -> int:
        """Count events by type."""
        if event_type is None:
            return len(self.events)
        return len([e for e in self.events if e.event_type == event_type])
    
    def get_events(self, event_type: str = None) -> List:
        """Get events by type."""
        if event_type is None:
            return self.events.copy()
        return [e for e in self.events if e.event_type == event_type]
    
    def get_latest(self, event_type: str = None):
        """Get latest event of type."""
        events = self.get_events(event_type)
        return events[-1] if events else None
    
    def assert_count(self, expected: int, event_type: str = None):
        """Assert event count."""
        actual = self.count(event_type)
        if actual != expected:
            raise AssertionError(f"Expected {expected} events of type {event_type}, got {actual}")
    
    def assert_event_sequence(self, expected_types: List[str]):
        """Assert sequence of event types."""
        actual_types = [e.event_type for e in self.events]
        if actual_types != expected_types:
            raise AssertionError(f"Expected event sequence {expected_types}, got {actual_types}")


class TestTimer:
    """Utility to track timing in tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing."""
        self.start_time = datetime.now()
    
    def stop(self):
        """Stop timing."""
        self.end_time = datetime.now()
    
    def elapsed(self) -> timedelta:
        """Get elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed().total_seconds() * 1000


def create_test_event(event_type: str, source_id: str = "test", **payload_kwargs):
    """Create test event with standard structure."""
    from datetime import datetime
    import uuid
    
    # This is a simplified version - real implementation would import from events module
    class TestEvent:
        def __init__(self, event_type, source_id, payload):
            self.event_type = event_type
            self.source_id = source_id
            self.payload = payload
            self.timestamp = datetime.now()
            self.event_id = str(uuid.uuid4())
    
    return TestEvent(event_type, source_id, payload_kwargs)


def setup_test_container(role: str = "test", name: str = None) -> MockContainer:
    """Setup a standard test container."""
    container_name = name or f"test_{uuid.uuid4().hex[:8]}"
    return MockContainer(role=role, container_id=container_name)


def setup_strategy_test_scenario():
    """Setup a complete strategy testing scenario."""
    container = setup_test_container("backtest", "strategy_test")
    
    # Add mock components
    data_component = MockComponent("data_streamer")
    strategy_component = MockComponent("strategy")
    
    container.add_component("data", data_component)
    container.add_component("strategy", strategy_component)
    
    # Initialize components
    data_component.initialize(container)
    strategy_component.initialize(container)
    
    # Create event collectors
    bar_collector = EventCollector()
    signal_collector = EventCollector()
    
    # Wire up event flow
    container.event_bus.subscribe("BAR", bar_collector.handler)
    container.event_bus.subscribe("SIGNAL", signal_collector.handler)
    
    return {
        'container': container,
        'data_component': data_component,
        'strategy_component': strategy_component,
        'bar_collector': bar_collector,
        'signal_collector': signal_collector
    }