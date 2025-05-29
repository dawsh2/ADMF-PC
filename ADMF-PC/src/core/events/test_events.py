"""
Tests for the containerized event system.

This module contains comprehensive tests for event isolation,
subscription management, and container lifecycle.
"""

import unittest
from datetime import datetime
import threading
import time
from typing import List, Dict, Any

from .types import Event, EventType, create_market_event, create_signal_event
from .event_bus import EventBus, ContainerEventBus
from .subscription_manager import SubscriptionManager, WeakSubscriptionManager
from .isolation import EventIsolationManager, get_isolation_manager


class TestEventTypes(unittest.TestCase):
    """Test event types and creation helpers."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(
            event_type=EventType.BAR,
            payload={"symbol": "AAPL", "close": 150.0},
            source_id="test_source"
        )
        
        self.assertEqual(event.event_type, EventType.BAR)
        self.assertEqual(event.payload["symbol"], "AAPL")
        self.assertEqual(event.source_id, "test_source")
        self.assertIsInstance(event.timestamp, datetime)
    
    def test_market_event_helper(self):
        """Test market event creation helper."""
        timestamp = datetime.now()
        event = create_market_event(
            EventType.BAR,
            symbol="AAPL",
            timestamp=timestamp,
            data={"open": 149.0, "close": 150.0},
            container_id="test_container"
        )
        
        self.assertEqual(event.event_type, EventType.BAR)
        self.assertEqual(event.payload["symbol"], "AAPL")
        self.assertEqual(event.payload["data"]["close"], 150.0)
        self.assertEqual(event.container_id, "test_container")
        self.assertEqual(event.timestamp, timestamp)
    
    def test_signal_event_helper(self):
        """Test signal event creation helper."""
        event = create_signal_event(
            symbol="AAPL",
            signal_type="BUY",
            strength=0.8,
            timestamp=datetime.now(),
            container_id="test_container",
            stop_loss=145.0
        )
        
        self.assertEqual(event.event_type, EventType.SIGNAL)
        self.assertEqual(event.payload["signal_type"], "BUY")
        self.assertEqual(event.payload["strength"], 0.8)
        self.assertEqual(event.payload["stop_loss"], 145.0)


class TestEventBus(unittest.TestCase):
    """Test EventBus functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bus = EventBus("test_container")
        self.received_events: List[Event] = []
    
    def handler(self, event: Event):
        """Test event handler."""
        self.received_events.append(event)
    
    def test_subscribe_publish(self):
        """Test basic subscribe and publish."""
        self.bus.subscribe(EventType.BAR, self.handler)
        
        event = Event(EventType.BAR, {"test": "data"})
        self.bus.publish(event)
        
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].payload["test"], "data")
    
    def test_multiple_handlers(self):
        """Test multiple handlers for same event type."""
        handler2_events = []
        
        def handler2(event):
            handler2_events.append(event)
        
        self.bus.subscribe(EventType.BAR, self.handler)
        self.bus.subscribe(EventType.BAR, handler2)
        
        event = Event(EventType.BAR, {"test": "data"})
        self.bus.publish(event)
        
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(len(handler2_events), 1)
    
    def test_unsubscribe(self):
        """Test unsubscribe functionality."""
        self.bus.subscribe(EventType.BAR, self.handler)
        self.bus.unsubscribe(EventType.BAR, self.handler)
        
        event = Event(EventType.BAR, {"test": "data"})
        self.bus.publish(event)
        
        self.assertEqual(len(self.received_events), 0)
    
    def test_unsubscribe_all(self):
        """Test unsubscribe_all functionality."""
        self.bus.subscribe(EventType.BAR, self.handler)
        self.bus.subscribe(EventType.SIGNAL, self.handler)
        self.bus.unsubscribe_all(self.handler)
        
        self.bus.publish(Event(EventType.BAR, {}))
        self.bus.publish(Event(EventType.SIGNAL, {}))
        
        self.assertEqual(len(self.received_events), 0)
    
    def test_error_handling(self):
        """Test that errors in handlers don't stop propagation."""
        def error_handler(event):
            raise ValueError("Test error")
        
        self.bus.subscribe(EventType.BAR, error_handler)
        self.bus.subscribe(EventType.BAR, self.handler)
        
        event = Event(EventType.BAR, {"test": "data"})
        self.bus.publish(event)
        
        # Handler should still receive event despite error
        self.assertEqual(len(self.received_events), 1)
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        def publisher():
            for i in range(100):
                self.bus.publish(Event(EventType.BAR, {"i": i}))
        
        def subscriber():
            for i in range(50):
                self.bus.subscribe(EventType.BAR, lambda e: None)
                time.sleep(0.001)
                self.bus.unsubscribe(EventType.BAR, lambda e: None)
        
        threads = [
            threading.Thread(target=publisher),
            threading.Thread(target=publisher),
            threading.Thread(target=subscriber),
            threading.Thread(target=subscriber)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without deadlock or exception
        self.assertTrue(True)


class TestSubscriptionManager(unittest.TestCase):
    """Test SubscriptionManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bus = EventBus("test_container")
        self.manager = SubscriptionManager(self.bus, "test_component")
        self.received_events: List[Event] = []
    
    def handler(self, event: Event):
        """Test event handler."""
        self.received_events.append(event)
    
    def test_managed_subscribe(self):
        """Test subscription through manager."""
        self.manager.subscribe(EventType.BAR, self.handler)
        
        event = Event(EventType.BAR, {"test": "data"})
        self.bus.publish(event)
        
        self.assertEqual(len(self.received_events), 1)
    
    def test_unsubscribe_all(self):
        """Test unsubscribe_all cleans up properly."""
        self.manager.subscribe(EventType.BAR, self.handler)
        self.manager.subscribe(EventType.SIGNAL, self.handler)
        
        self.manager.unsubscribe_all()
        
        self.bus.publish(Event(EventType.BAR, {}))
        self.bus.publish(Event(EventType.SIGNAL, {}))
        
        self.assertEqual(len(self.received_events), 0)
    
    def test_context_manager(self):
        """Test context manager usage."""
        with SubscriptionManager(self.bus, "test") as manager:
            manager.subscribe(EventType.BAR, self.handler)
            self.bus.publish(Event(EventType.BAR, {"test": "data"}))
            self.assertEqual(len(self.received_events), 1)
        
        # After context exit, should be unsubscribed
        self.bus.publish(Event(EventType.BAR, {"test": "data2"}))
        self.assertEqual(len(self.received_events), 1)  # No new events


class TestEventIsolation(unittest.TestCase):
    """Test container event isolation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.isolation_manager = EventIsolationManager()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up any created containers
        for container_id in list(self.isolation_manager._active_containers):
            self.isolation_manager.remove_container_bus(container_id)
    
    def test_container_isolation(self):
        """Test that containers are isolated from each other."""
        # Create two containers
        bus1 = self.isolation_manager.create_container_bus("container1")
        bus2 = self.isolation_manager.create_container_bus("container2")
        
        events1 = []
        events2 = []
        
        bus1.subscribe(EventType.BAR, lambda e: events1.append(e))
        bus2.subscribe(EventType.BAR, lambda e: events2.append(e))
        
        # Publish to container 1
        event1 = Event(EventType.BAR, {"data": "container1"})
        bus1.publish(event1)
        
        # Publish to container 2
        event2 = Event(EventType.BAR, {"data": "container2"})
        bus2.publish(event2)
        
        # Each container should only receive its own events
        self.assertEqual(len(events1), 1)
        self.assertEqual(len(events2), 1)
        self.assertEqual(events1[0].payload["data"], "container1")
        self.assertEqual(events2[0].payload["data"], "container2")
    
    def test_container_context(self):
        """Test container context manager."""
        bus = self.isolation_manager.create_container_bus("test_container")
        
        events = []
        bus.subscribe(EventType.BAR, lambda e: events.append(e))
        
        with self.isolation_manager.container_context("test_container") as context_bus:
            self.assertEqual(context_bus, bus)
            context_bus.publish(Event(EventType.BAR, {"test": "data"}))
        
        self.assertEqual(len(events), 1)
    
    def test_container_cleanup(self):
        """Test container cleanup."""
        bus = self.isolation_manager.create_container_bus("cleanup_test")
        
        events = []
        bus.subscribe(EventType.BAR, lambda e: events.append(e))
        
        # Remove container
        self.isolation_manager.remove_container_bus("cleanup_test")
        
        # Bus should be cleared
        self.assertEqual(bus.get_stats()["subscription_count"], 0)


class TestContainerEventBus(unittest.TestCase):
    """Test ContainerEventBus extended features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bus = ContainerEventBus("test_container")
    
    def test_source_filtering(self):
        """Test source-based event filtering."""
        received = []
        
        # Subscribe with source filter
        self.bus.subscribe_filtered(
            EventType.BAR,
            lambda e: received.append(e),
            source_filter="source1"
        )
        
        # Publish from different sources
        event1 = Event(EventType.BAR, {"data": 1}, source_id="source1")
        event2 = Event(EventType.BAR, {"data": 2}, source_id="source2")
        event3 = Event(EventType.BAR, {"data": 3}, source_id="source1")
        
        self.bus.publish(event1)
        self.bus.publish(event2)
        self.bus.publish(event3)
        
        # Should only receive events from source1
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0].payload["data"], 1)
        self.assertEqual(received[1].payload["data"], 3)
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        self.bus.subscribe(EventType.BAR, lambda e: None)
        self.bus.subscribe(EventType.SIGNAL, lambda e: None)
        
        # Publish various events
        for _ in range(5):
            self.bus.publish(Event(EventType.BAR, {}))
        for _ in range(3):
            self.bus.publish(Event(EventType.SIGNAL, {}))
        
        metrics = self.bus.get_metrics()
        
        self.assertEqual(metrics["metrics"]["events_by_type"][EventType.BAR], 5)
        self.assertEqual(metrics["metrics"]["events_by_type"][EventType.SIGNAL], 3)
        self.assertEqual(metrics["metrics"]["total_events"], 8)


if __name__ == "__main__":
    unittest.main()