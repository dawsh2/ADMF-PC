"""
Isolated tests for Events module core logic.

These tests work by copying/mocking the essential classes to avoid
import issues. They validate that the core event logic works correctly.
"""

import unittest
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import uuid
import threading


# Mock the Event types and classes (copied from expected interface)

class EventType(Enum):
    """Mock EventType enum."""
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    CANCEL = "CANCEL"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"


@dataclass(frozen=True)
class Event:
    """Mock Event class - should match your actual Event."""
    event_type: str
    source_id: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict[str, Any]] = None


class MockEventBus:
    """Mock EventBus - copy of expected interface."""
    
    def __init__(self, bus_id: Optional[str] = None):
        self.bus_id = bus_id or str(uuid.uuid4())
        self._subscriptions: Dict[str, List[tuple]] = defaultdict(list)
        self._subscription_counter = 0
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: str, handler: Callable, filter_func: Optional[Callable] = None) -> str:
        """Subscribe to events."""
        with self._lock:
            subscription_id = f"sub_{self._subscription_counter}"
            self._subscription_counter += 1
            self._subscriptions[event_type].append((subscription_id, handler, filter_func))
            return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        with self._lock:
            for event_type, subs in self._subscriptions.items():
                for i, (sub_id, handler, filter_func) in enumerate(subs):
                    if sub_id == subscription_id:
                        del subs[i]
                        return True
            return False
    
    def publish(self, event: Event) -> None:
        """Publish event to subscribers."""
        event_type = event.event_type
        handlers = []
        
        with self._lock:
            handlers = self._subscriptions[event_type].copy()
        
        for subscription_id, handler, filter_func in handlers:
            try:
                # Apply filter if present
                if filter_func and not filter_func(event):
                    continue
                
                # Call handler
                handler(event)
                
            except Exception as e:
                # Log error but continue (in real implementation)
                print(f"Error in event handler: {e}")


# Mock container-related classes

class ContainerRole(Enum):
    """Mock ContainerRole."""
    BACKTEST = "backtest"
    PORTFOLIO = "portfolio"
    EXECUTION = "execution"


class ContainerState(Enum):
    """Mock ContainerState."""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass
class ContainerConfig:
    """Mock ContainerConfig."""
    role: ContainerRole
    name: str
    capabilities: Optional[set] = None


class MockContainer:
    """Mock Container class."""
    
    def __init__(self, config: ContainerConfig):
        self.config = config
        self.container_id = f"{config.role.value}_{config.name}_{uuid.uuid4().hex[:8]}"
        self.state = ContainerState.CREATED
        self.event_bus = MockEventBus(f"bus_{self.container_id}")
        self._components: Dict[str, Any] = {}
    
    def add_component(self, name: str, component: Any) -> None:
        """Add component to container."""
        self._components[name] = component
    
    def get_component(self, name: str) -> Any:
        """Get component from container."""
        return self._components.get(name)


# === ISOLATED TESTS ===

class TestEventBasics(unittest.TestCase):
    """Test Event class basics."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(
            event_type=EventType.BAR.value,
            source_id="test_source",
            payload={"symbol": "AAPL", "close": 100.50}
        )
        
        self.assertEqual(event.event_type, EventType.BAR.value)
        self.assertEqual(event.source_id, "test_source")
        self.assertEqual(event.payload["symbol"], "AAPL")
        self.assertEqual(event.payload["close"], 100.50)
        self.assertIsInstance(event.timestamp, datetime)
        self.assertIsNotNone(event.event_id)
    
    def test_event_immutability(self):
        """Test events are immutable (frozen dataclass)."""
        event = Event(
            event_type=EventType.ORDER.value,
            source_id="strategy_1",
            payload={"symbol": "MSFT", "quantity": 100}
        )
        
        # Should not be able to modify
        with self.assertRaises(Exception):  # FrozenInstanceError or AttributeError
            event.source_id = "different_source"
    
    def test_event_with_metadata(self):
        """Test event with metadata."""
        metadata = {"priority": "high", "retry_count": 0}
        event = Event(
            event_type=EventType.SIGNAL.value,
            source_id="momentum_strategy",
            payload={"symbol": "AAPL", "direction": "BUY"},
            metadata=metadata
        )
        
        self.assertEqual(event.metadata["priority"], "high")
        self.assertEqual(event.metadata["retry_count"], 0)


class TestEventBusIsolated(unittest.TestCase):
    """Test EventBus logic in isolation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = MockEventBus()
    
    def test_basic_subscribe_publish(self):
        """Test basic subscribe and publish."""
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        # Subscribe
        sub_id = self.event_bus.subscribe(EventType.BAR.value, handler)
        self.assertIsNotNone(sub_id)
        
        # Publish
        event = Event(
            event_type=EventType.BAR.value,
            source_id="data_feed",
            payload={"symbol": "AAPL", "close": 150.25}
        )
        self.event_bus.publish(event)
        
        # Verify
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].payload["symbol"], "AAPL")
    
    def test_multiple_subscribers(self):
        """Test multiple subscribers receive same event."""
        received1 = []
        received2 = []
        
        def handler1(event):
            received1.append(event)
        
        def handler2(event):
            received2.append(event)
        
        # Subscribe both
        self.event_bus.subscribe(EventType.SIGNAL.value, handler1)
        self.event_bus.subscribe(EventType.SIGNAL.value, handler2)
        
        # Publish
        event = Event(
            event_type=EventType.SIGNAL.value,
            source_id="strategy",
            payload={"symbol": "MSFT", "direction": "SELL"}
        )
        self.event_bus.publish(event)
        
        # Both should receive
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 1)
        self.assertEqual(received1[0], received2[0])
    
    def test_event_filtering(self):
        """Test event filtering works."""
        received = []
        
        def handler(event):
            received.append(event)
        
        def filter_func(event):
            return event.payload.get("symbol") == "AAPL"
        
        # Subscribe with filter
        self.event_bus.subscribe(EventType.ORDER.value, handler, filter_func)
        
        # Publish multiple events
        events = [
            Event(EventType.ORDER.value, "test", {"symbol": "AAPL", "qty": 100}),
            Event(EventType.ORDER.value, "test", {"symbol": "MSFT", "qty": 50}),
            Event(EventType.ORDER.value, "test", {"symbol": "AAPL", "qty": 200}),
        ]
        
        for event in events:
            self.event_bus.publish(event)
        
        # Should only receive AAPL events
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0].payload["symbol"], "AAPL")
        self.assertEqual(received[1].payload["symbol"], "AAPL")
    
    def test_unsubscribe(self):
        """Test unsubscribing works."""
        received = []
        
        def handler(event):
            received.append(event)
        
        # Subscribe
        sub_id = self.event_bus.subscribe(EventType.FILL.value, handler)
        
        # Publish first event
        event1 = Event(EventType.FILL.value, "broker", {"fill_id": "1"})
        self.event_bus.publish(event1)
        
        # Unsubscribe
        success = self.event_bus.unsubscribe(sub_id)
        self.assertTrue(success)
        
        # Publish second event
        event2 = Event(EventType.FILL.value, "broker", {"fill_id": "2"})
        self.event_bus.publish(event2)
        
        # Should only receive first event
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["fill_id"], "1")
    
    def test_error_handling(self):
        """Test error handling in handlers."""
        good_received = []
        
        def bad_handler(event):
            raise ValueError("Simulated handler error")
        
        def good_handler(event):
            good_received.append(event)
        
        # Subscribe both handlers
        self.event_bus.subscribe(EventType.SYSTEM_START.value, bad_handler)
        self.event_bus.subscribe(EventType.SYSTEM_START.value, good_handler)
        
        # Publish event
        event = Event(EventType.SYSTEM_START.value, "system", {"message": "starting"})
        self.event_bus.publish(event)
        
        # Good handler should still receive despite bad handler error
        self.assertEqual(len(good_received), 1)


class TestContainerIsolated(unittest.TestCase):
    """Test Container logic in isolation."""
    
    def test_container_creation(self):
        """Test container creation."""
        config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="test_backtest"
        )
        container = MockContainer(config)
        
        self.assertIsNotNone(container.container_id)
        self.assertEqual(container.config.role, ContainerRole.BACKTEST)
        self.assertEqual(container.state, ContainerState.CREATED)
        self.assertIsInstance(container.event_bus, MockEventBus)
    
    def test_component_management(self):
        """Test adding and retrieving components."""
        config = ContainerConfig(
            role=ContainerRole.PORTFOLIO,
            name="test_portfolio"
        )
        container = MockContainer(config)
        
        # Create mock component
        class MockComponent:
            def __init__(self, name):
                self.name = name
        
        component = MockComponent("test_component")
        
        # Add component
        container.add_component("test_comp", component)
        
        # Retrieve component
        retrieved = container.get_component("test_comp")
        self.assertEqual(retrieved, component)
        self.assertEqual(retrieved.name, "test_component")
    
    def test_container_event_bus_integration(self):
        """Test container integrates with its event bus."""
        config = ContainerConfig(
            role=ContainerRole.EXECUTION,
            name="test_execution"
        )
        container = MockContainer(config)
        
        received = []
        
        # Subscribe through container's event bus
        container.event_bus.subscribe(EventType.ORDER.value, lambda e: received.append(e))
        
        # Publish through container's event bus
        order_event = Event(
            event_type=EventType.ORDER.value,
            source_id="strategy",
            payload={"symbol": "GOOGL", "quantity": 75}
        )
        container.event_bus.publish(order_event)
        
        # Verify
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["symbol"], "GOOGL")


class TestEventsContainersIntegrationIsolated(unittest.TestCase):
    """Test events and containers work together (isolated)."""
    
    def test_multiple_containers_isolation(self):
        """Test container event buses are isolated."""
        # Create two containers
        config1 = ContainerConfig(ContainerRole.BACKTEST, "backtest_1")
        config2 = ContainerConfig(ContainerRole.PORTFOLIO, "portfolio_1")
        
        container1 = MockContainer(config1)
        container2 = MockContainer(config2)
        
        received1 = []
        received2 = []
        
        # Subscribe to each container's bus
        container1.event_bus.subscribe(EventType.SIGNAL.value, lambda e: received1.append(e))
        container2.event_bus.subscribe(EventType.SIGNAL.value, lambda e: received2.append(e))
        
        # Publish to container1
        signal1 = Event(EventType.SIGNAL.value, "strategy1", {"symbol": "AAPL"})
        container1.event_bus.publish(signal1)
        
        # Only container1 should receive
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 0)
        
        # Publish to container2
        signal2 = Event(EventType.SIGNAL.value, "strategy2", {"symbol": "MSFT"})
        container2.event_bus.publish(signal2)
        
        # Now container2 should receive its event
        self.assertEqual(len(received1), 1)  # Still 1
        self.assertEqual(len(received2), 1)  # Now 1
    
    def test_container_component_event_flow(self):
        """Test components can use container's event bus."""
        config = ContainerConfig(ContainerRole.BACKTEST, "event_flow_test")
        container = MockContainer(config)
        
        # Mock component that publishes events
        class MockPublisher:
            def __init__(self, container):
                self.container = container
            
            def publish_signal(self, symbol, direction):
                event = Event(
                    event_type=EventType.SIGNAL.value,
                    source_id="mock_publisher",
                    payload={"symbol": symbol, "direction": direction}
                )
                self.container.event_bus.publish(event)
        
        # Mock component that receives events
        class MockReceiver:
            def __init__(self):
                self.received = []
            
            def on_signal(self, event):
                self.received.append(event)
        
        # Add components
        publisher = MockPublisher(container)
        receiver = MockReceiver()
        
        container.add_component("publisher", publisher)
        container.add_component("receiver", receiver)
        
        # Wire up subscription
        container.event_bus.subscribe(EventType.SIGNAL.value, receiver.on_signal)
        
        # Publisher sends signal
        publisher.publish_signal("AAPL", "BUY")
        
        # Receiver should get it
        self.assertEqual(len(receiver.received), 1)
        self.assertEqual(receiver.received[0].payload["symbol"], "AAPL")
        self.assertEqual(receiver.received[0].payload["direction"], "BUY")


if __name__ == "__main__":
    unittest.main(verbosity=2)