"""
Unit tests for event system.

Tests event bus, subscriptions, and event isolation.
"""

import unittest
import asyncio
from unittest.mock import Mock, MagicMock, call
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.events.event_bus import EventBus
from src.core.events.types import Event, EventType
from src.core.events.subscription_manager import SubscriptionManager
from src.core.events.isolation import EventIsolation


class TestEvent(unittest.TestCase):
    """Test Event dataclass."""
    
    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            event_type=EventType.ORDER,
            source_id="test_source",
            payload={"order_id": "123", "symbol": "AAPL"}
        )
        
        self.assertEqual(event.event_type, EventType.ORDER)
        self.assertEqual(event.source_id, "test_source")
        self.assertEqual(event.payload["order_id"], "123")
        self.assertIsInstance(event.timestamp, datetime)
        self.assertIsNotNone(event.event_id)
    
    def test_event_immutability(self):
        """Test that events are immutable."""
        event = Event(
            event_type=EventType.FILL,
            source_id="broker",
            payload={"fill_id": "456"}
        )
        
        # Should not be able to modify
        with self.assertRaises(AttributeError):
            event.source_id = "different"
    
    def test_event_metadata(self):
        """Test event metadata."""
        metadata = {"priority": "high", "retry_count": 0}
        event = Event(
            event_type=EventType.SYSTEM,
            source_id="system",
            payload={"message": "test"},
            metadata=metadata
        )
        
        self.assertEqual(event.metadata["priority"], "high")
        self.assertEqual(event.metadata["retry_count"], 0)


class TestEventBus(unittest.TestCase):
    """Test EventBus functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
    
    def test_subscribe_and_publish(self):
        """Test basic subscribe and publish."""
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        # Subscribe to ORDER events
        self.event_bus.subscribe(EventType.ORDER, handler)
        
        # Publish ORDER event
        event = Event(
            event_type=EventType.ORDER,
            source_id="test",
            payload={"test": "data"}
        )
        self.event_bus.publish(event)
        
        # Check event was received
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0], event)
    
    def test_multiple_subscribers(self):
        """Test multiple subscribers to same event type."""
        handler1_events = []
        handler2_events = []
        
        def handler1(event):
            handler1_events.append(event)
        
        def handler2(event):
            handler2_events.append(event)
        
        # Subscribe both handlers
        self.event_bus.subscribe(EventType.FILL, handler1)
        self.event_bus.subscribe(EventType.FILL, handler2)
        
        # Publish event
        event = Event(
            event_type=EventType.FILL,
            source_id="broker",
            payload={"fill": "data"}
        )
        self.event_bus.publish(event)
        
        # Both should receive
        self.assertEqual(len(handler1_events), 1)
        self.assertEqual(len(handler2_events), 1)
    
    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        received = []
        
        def handler(event):
            received.append(event)
        
        # Subscribe
        subscription_id = self.event_bus.subscribe(EventType.SYSTEM, handler)
        
        # Publish first event
        event1 = Event(EventType.SYSTEM, "test", {"msg": "1"})
        self.event_bus.publish(event1)
        
        # Unsubscribe
        self.event_bus.unsubscribe(subscription_id)
        
        # Publish second event
        event2 = Event(EventType.SYSTEM, "test", {"msg": "2"})
        self.event_bus.publish(event2)
        
        # Should only have received first event
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["msg"], "1")
    
    def test_event_filtering(self):
        """Test event filtering by source."""
        received = []
        
        def handler(event):
            received.append(event)
        
        # Subscribe with source filter
        self.event_bus.subscribe(
            EventType.ORDER,
            handler,
            source_filter="strategy_1"
        )
        
        # Publish from different sources
        event1 = Event(EventType.ORDER, "strategy_1", {"id": "1"})
        event2 = Event(EventType.ORDER, "strategy_2", {"id": "2"})
        event3 = Event(EventType.ORDER, "strategy_1", {"id": "3"})
        
        self.event_bus.publish(event1)
        self.event_bus.publish(event2)
        self.event_bus.publish(event3)
        
        # Should only receive from strategy_1
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0].payload["id"], "1")
        self.assertEqual(received[1].payload["id"], "3")
    
    def test_wildcard_subscription(self):
        """Test subscribing to all event types."""
        all_events = []
        
        def handler(event):
            all_events.append(event)
        
        # Subscribe to all events
        self.event_bus.subscribe_all(handler)
        
        # Publish various events
        events = [
            Event(EventType.ORDER, "test", {"type": "order"}),
            Event(EventType.FILL, "test", {"type": "fill"}),
            Event(EventType.SYSTEM, "test", {"type": "system"})
        ]
        
        for event in events:
            self.event_bus.publish(event)
        
        # Should receive all
        self.assertEqual(len(all_events), 3)
    
    def test_error_handling(self):
        """Test error handling in subscribers."""
        good_events = []
        
        def bad_handler(event):
            raise ValueError("Handler error")
        
        def good_handler(event):
            good_events.append(event)
        
        # Subscribe both
        self.event_bus.subscribe(EventType.ORDER, bad_handler)
        self.event_bus.subscribe(EventType.ORDER, good_handler)
        
        # Publish event
        event = Event(EventType.ORDER, "test", {"data": "test"})
        self.event_bus.publish(event)
        
        # Good handler should still receive despite bad handler error
        self.assertEqual(len(good_events), 1)
    
    def test_async_handlers(self):
        """Test async event handlers."""
        received = []
        
        async def async_handler(event):
            await asyncio.sleep(0.01)  # Simulate async work
            received.append(event)
        
        # Subscribe async handler
        self.event_bus.subscribe(EventType.FILL, async_handler)
        
        # Publish event
        event = Event(EventType.FILL, "test", {"async": True})
        
        # Run in event loop
        async def test():
            self.event_bus.publish(event)
            await asyncio.sleep(0.02)  # Wait for async handler
        
        asyncio.run(test())
        
        # Should have received
        self.assertEqual(len(received), 1)


class TestSubscriptionManager(unittest.TestCase):
    """Test subscription management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = SubscriptionManager()
    
    def test_add_subscription(self):
        """Test adding subscriptions."""
        handler = Mock()
        
        sub_id = self.manager.add_subscription(
            EventType.ORDER,
            handler,
            source_filter="strategy_1",
            priority=10
        )
        
        self.assertIsNotNone(sub_id)
        
        # Get handlers for event type
        handlers = self.manager.get_handlers(EventType.ORDER)
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0][0], handler)
    
    def test_remove_subscription(self):
        """Test removing subscriptions."""
        handler = Mock()
        
        sub_id = self.manager.add_subscription(EventType.FILL, handler)
        self.manager.remove_subscription(sub_id)
        
        handlers = self.manager.get_handlers(EventType.FILL)
        self.assertEqual(len(handlers), 0)
    
    def test_priority_ordering(self):
        """Test handlers are ordered by priority."""
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()
        
        # Add with different priorities
        self.manager.add_subscription(EventType.SYSTEM, handler1, priority=5)
        self.manager.add_subscription(EventType.SYSTEM, handler2, priority=10)
        self.manager.add_subscription(EventType.SYSTEM, handler3, priority=1)
        
        # Get handlers - should be ordered by priority (high to low)
        handlers = self.manager.get_handlers(EventType.SYSTEM)
        
        self.assertEqual(handlers[0][0], handler2)  # Priority 10
        self.assertEqual(handlers[1][0], handler1)  # Priority 5
        self.assertEqual(handlers[2][0], handler3)  # Priority 1
    
    def test_subscription_filtering(self):
        """Test subscription filtering."""
        handler1 = Mock()
        handler2 = Mock()
        
        # Add with different filters
        self.manager.add_subscription(
            EventType.ORDER,
            handler1,
            source_filter="strategy_1"
        )
        self.manager.add_subscription(
            EventType.ORDER,
            handler2,
            source_filter="strategy_2"
        )
        
        # Get all handlers
        all_handlers = self.manager.get_handlers(EventType.ORDER)
        self.assertEqual(len(all_handlers), 2)
        
        # Filter by source
        event1 = Event(EventType.ORDER, "strategy_1", {})
        filtered1 = self.manager.get_handlers_for_event(event1)
        self.assertEqual(len(filtered1), 1)
        self.assertEqual(filtered1[0][0], handler1)
        
        event2 = Event(EventType.ORDER, "strategy_2", {})
        filtered2 = self.manager.get_handlers_for_event(event2)
        self.assertEqual(len(filtered2), 1)
        self.assertEqual(filtered2[0][0], handler2)
    
    def test_subscription_stats(self):
        """Test subscription statistics."""
        # Add various subscriptions
        self.manager.add_subscription(EventType.ORDER, Mock())
        self.manager.add_subscription(EventType.ORDER, Mock())
        self.manager.add_subscription(EventType.FILL, Mock())
        self.manager.add_subscription(EventType.SYSTEM, Mock())
        
        stats = self.manager.get_statistics()
        
        self.assertEqual(stats["total_subscriptions"], 4)
        self.assertEqual(stats["by_event_type"][EventType.ORDER], 2)
        self.assertEqual(stats["by_event_type"][EventType.FILL], 1)
        self.assertEqual(stats["by_event_type"][EventType.SYSTEM], 1)


class TestEventIsolation(unittest.TestCase):
    """Test event isolation between containers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.isolation = EventIsolation()
    
    def test_create_isolated_bus(self):
        """Test creating isolated event bus."""
        bus1 = self.isolation.create_isolated_bus("container1")
        bus2 = self.isolation.create_isolated_bus("container2")
        
        # Buses should be different instances
        self.assertIsNot(bus1, bus2)
        
        # Events in bus1 shouldn't affect bus2
        received1 = []
        received2 = []
        
        bus1.subscribe(EventType.ORDER, lambda e: received1.append(e))
        bus2.subscribe(EventType.ORDER, lambda e: received2.append(e))
        
        event = Event(EventType.ORDER, "test", {"data": "test"})
        bus1.publish(event)
        
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 0)
    
    def test_bridge_buses(self):
        """Test bridging isolated buses."""
        bus1 = self.isolation.create_isolated_bus("container1")
        bus2 = self.isolation.create_isolated_bus("container2")
        
        # Bridge specific event types
        self.isolation.bridge_buses(
            bus1, bus2,
            event_types=[EventType.SYSTEM],
            bidirectional=True
        )
        
        received1 = []
        received2 = []
        
        bus1.subscribe(EventType.SYSTEM, lambda e: received1.append(e))
        bus2.subscribe(EventType.SYSTEM, lambda e: received2.append(e))
        
        # Publish from bus1
        event1 = Event(EventType.SYSTEM, "bus1", {"msg": "from bus1"})
        bus1.publish(event1)
        
        # Should be received in both
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 1)
        
        # Clear
        received1.clear()
        received2.clear()
        
        # Publish from bus2 (bidirectional)
        event2 = Event(EventType.SYSTEM, "bus2", {"msg": "from bus2"})
        bus2.publish(event2)
        
        # Should be received in both
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 1)
    
    def test_unidirectional_bridge(self):
        """Test unidirectional bridge."""
        bus1 = self.isolation.create_isolated_bus("source")
        bus2 = self.isolation.create_isolated_bus("target")
        
        # Bridge one way only
        self.isolation.bridge_buses(
            bus1, bus2,
            event_types=[EventType.ORDER],
            bidirectional=False
        )
        
        received1 = []
        received2 = []
        
        bus1.subscribe(EventType.ORDER, lambda e: received1.append(e))
        bus2.subscribe(EventType.ORDER, lambda e: received2.append(e))
        
        # Publish from bus1
        event1 = Event(EventType.ORDER, "bus1", {})
        bus1.publish(event1)
        
        # Should be received in both
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 1)
        
        # Clear
        received1.clear()
        received2.clear()
        
        # Publish from bus2
        event2 = Event(EventType.ORDER, "bus2", {})
        bus2.publish(event2)
        
        # Should only be received in bus2
        self.assertEqual(len(received1), 0)
        self.assertEqual(len(received2), 1)
    
    def test_event_filtering_in_bridge(self):
        """Test event filtering in bridges."""
        bus1 = self.isolation.create_isolated_bus("filtered")
        bus2 = self.isolation.create_isolated_bus("target")
        
        # Bridge with filter
        def filter_func(event):
            return event.payload.get("forward", False)
        
        self.isolation.bridge_buses(
            bus1, bus2,
            event_types=[EventType.SYSTEM],
            filter_func=filter_func
        )
        
        received = []
        bus2.subscribe(EventType.SYSTEM, lambda e: received.append(e))
        
        # Publish events
        event1 = Event(EventType.SYSTEM, "test", {"forward": True})
        event2 = Event(EventType.SYSTEM, "test", {"forward": False})
        event3 = Event(EventType.SYSTEM, "test", {"other": "data"})
        
        bus1.publish(event1)
        bus1.publish(event2)
        bus1.publish(event3)
        
        # Only event1 should be forwarded
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["forward"], True)


if __name__ == "__main__":
    unittest.main()