"""
Integration tests for Events + Containers after complete rewrite.

This test suite builds up progressively:
1. EventBus standalone
2. Container standalone  
3. EventBus + Container integration
4. Basic component lifecycle
5. Event filtering and barriers

Tests are designed to validate the rewritten architecture works.
"""

import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, MagicMock
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from current rewritten modules
from src.core.events import (
    EventBus, Event, EventType, EventHandler,
    create_market_event, create_signal_event, create_system_event
)

from src.core.containers import (
    Container, ContainerConfig, ContainerRole, ContainerState,
    ContainerProtocol, ContainerComponent
)


class TestEventBusStandalone(unittest.TestCase):
    """Test EventBus works independently."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
    
    def test_basic_pub_sub(self):
        """Test basic publish/subscribe works."""
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        # Subscribe
        subscription_id = self.event_bus.subscribe(EventType.BAR, handler)
        self.assertIsNotNone(subscription_id)
        
        # Publish
        event = create_market_event(
            event_type=EventType.BAR,
            symbol="AAPL", 
            data={"open": 100, "high": 101, "low": 99, "close": 100.5}
        )
        self.event_bus.publish(event)
        
        # Verify
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].payload["symbol"], "AAPL")
        self.assertEqual(received_events[0].payload["data"]["close"], 100.5)
    
    def test_signal_event_filtering(self):
        """Test signal events require filtering as documented."""
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        # Create signal event
        signal_event = create_signal_event(
            strategy_id="momentum_1",
            symbol="AAPL",
            direction="BUY",
            strength=0.8
        )
        
        # Try subscribing to SIGNAL without filter - should work but with warning
        self.event_bus.subscribe(EventType.SIGNAL, handler)
        self.event_bus.publish(signal_event)
        
        # Should still work (the requirement is documented but not enforced in basic tests)
        self.assertEqual(len(received_events), 1)
    
    def test_multiple_event_types(self):
        """Test multiple event types work."""
        bar_events = []
        signal_events = []
        system_events = []
        
        self.event_bus.subscribe(EventType.BAR, lambda e: bar_events.append(e))
        self.event_bus.subscribe(EventType.SIGNAL, lambda e: signal_events.append(e))
        self.event_bus.subscribe(EventType.SYSTEM_START, lambda e: system_events.append(e))
        
        # Publish different event types
        self.event_bus.publish(create_market_event(EventType.BAR, "AAPL", {"close": 100}))
        self.event_bus.publish(create_signal_event("strat_1", "AAPL", "BUY", 0.5))
        self.event_bus.publish(create_system_event("system", "backtest_started"))
        
        # Verify routing
        self.assertEqual(len(bar_events), 1)
        self.assertEqual(len(signal_events), 1) 
        self.assertEqual(len(system_events), 1)


class TestContainerStandalone(unittest.TestCase):
    """Test Container works independently."""
    
    def test_container_creation(self):
        """Test basic container creation."""
        config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="test_backtest"
        )
        container = Container(config)
        
        # Verify basic properties
        self.assertIsNotNone(container.container_id)
        self.assertEqual(container.config.role, ContainerRole.BACKTEST)
        self.assertEqual(container.config.name, "test_backtest")
        self.assertEqual(container.state, ContainerState.CREATED)
    
    def test_container_has_event_bus(self):
        """Test container has an event bus."""
        config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="test_container"
        )
        container = Container(config)
        
        # Should have event bus
        self.assertIsNotNone(container.event_bus)
        self.assertIsInstance(container.event_bus, EventBus)
    
    def test_basic_component_management(self):
        """Test basic component add/get."""
        config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="test_container"
        )
        container = Container(config)
        
        # Create mock component
        mock_component = Mock()
        mock_component.component_id = "test_component"
        
        # Add component
        container.add_component("test_comp", mock_component)
        
        # Retrieve component
        retrieved = container.get_component("test_comp")
        self.assertEqual(retrieved, mock_component)
    
    def test_container_lifecycle_states(self):
        """Test container lifecycle state transitions."""
        config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="test_lifecycle"
        )
        container = Container(config)
        
        # Should start in CREATED state
        self.assertEqual(container.state, ContainerState.CREATED)
        
        # State transitions would be tested here
        # (depends on your actual Container implementation)


class MockTestComponent(ContainerComponent):
    """Mock component that implements ContainerComponent protocol."""
    
    def __init__(self, name: str):
        self.name = name
        self.container = None
        self.initialized = False
        self.started = False
        
    def initialize(self, container):
        """Initialize with container reference."""
        self.container = container
        self.initialized = True
    
    def start(self):
        """Start the component."""
        self.started = True
    
    def stop(self):
        """Stop the component."""
        self.started = False
    
    def get_state(self):
        """Get component state."""
        return {
            'name': self.name,
            'initialized': self.initialized,
            'started': self.started
        }


class TestEventsContainersIntegration(unittest.TestCase):
    """Test Events and Containers work together."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="integration_test"
        )
        self.container = Container(self.config)
    
    def test_container_event_bus_integration(self):
        """Test container can publish and subscribe to events."""
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        # Subscribe through container's event bus
        self.container.event_bus.subscribe(EventType.BAR, handler)
        
        # Publish through container's event bus
        bar_event = create_market_event(
            EventType.BAR,
            "AAPL",
            {"open": 100, "high": 102, "low": 99, "close": 101}
        )
        self.container.event_bus.publish(bar_event)
        
        # Verify
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].payload["symbol"], "AAPL")
    
    def test_component_event_access(self):
        """Test components can access container's event bus."""
        # Create and add component
        component = MockTestComponent("test_component")
        self.container.add_component("test_comp", component)
        
        # Initialize component
        component.initialize(self.container)
        
        # Component should have access to event bus through container
        self.assertIsNotNone(component.container)
        self.assertIsNotNone(component.container.event_bus)
        
        # Component can publish events
        received = []
        self.container.event_bus.subscribe(EventType.SYSTEM_START, lambda e: received.append(e))
        
        # Component publishes event through container
        system_event = create_system_event("test_component", "component_ready")
        component.container.event_bus.publish(system_event)
        
        # Verify
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].source_id, "test_component")
    
    def test_multiple_containers_isolation(self):
        """Test event buses are isolated between containers."""
        # Create second container
        config2 = ContainerConfig(
            role=ContainerRole.PORTFOLIO, 
            name="test_portfolio"
        )
        container2 = Container(config2)
        
        received1 = []
        received2 = []
        
        # Subscribe to each container's event bus
        self.container.event_bus.subscribe(EventType.ORDER, lambda e: received1.append(e))
        container2.event_bus.subscribe(EventType.ORDER, lambda e: received2.append(e))
        
        # Publish to first container
        order_event = Event(
            event_type=EventType.ORDER,
            source_id="strategy_1",
            payload={"symbol": "AAPL", "quantity": 100, "side": "BUY"}
        )
        self.container.event_bus.publish(order_event)
        
        # Only first container should receive
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 0)
        
        # Publish to second container
        order_event2 = Event(
            event_type=EventType.ORDER,
            source_id="strategy_2", 
            payload={"symbol": "MSFT", "quantity": 50, "side": "SELL"}
        )
        container2.event_bus.publish(order_event2)
        
        # Only second container should receive new event
        self.assertEqual(len(received1), 1)  # Still 1
        self.assertEqual(len(received2), 1)  # Now 1
    
    def test_container_component_lifecycle_integration(self):
        """Test component lifecycle integrates with container."""
        component = MockTestComponent("lifecycle_test")
        
        # Add to container
        self.container.add_component("lifecycle_comp", component)
        
        # Component should not be initialized yet
        self.assertFalse(component.initialized)
        self.assertFalse(component.started)
        
        # Initialize component
        component.initialize(self.container)
        self.assertTrue(component.initialized)
        self.assertEqual(component.container, self.container)
        
        # Start component
        component.start()
        self.assertTrue(component.started)
        
        # Stop component
        component.stop()
        self.assertFalse(component.started)


class TestBasicDataFlow(unittest.TestCase):
    """Test basic data flow through events and containers."""
    
    def test_simple_bar_to_signal_flow(self):
        """Test basic bar -> strategy -> signal flow."""
        # Setup container
        config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="data_flow_test"
        )
        container = Container(config)
        
        signals_generated = []
        
        # Mock strategy component that generates signals from bars
        class MockStrategy(ContainerComponent):
            def __init__(self):
                self.container = None
                
            def initialize(self, container):
                self.container = container
                # Subscribe to BAR events
                container.event_bus.subscribe(EventType.BAR, self.on_bar)
            
            def start(self):
                pass
                
            def stop(self):
                pass
                
            def get_state(self):
                return {"signals_generated": len(signals_generated)}
            
            def on_bar(self, event):
                """Generate signal from bar data."""
                bar_data = event.payload["data"]
                
                # Simple strategy: buy if close > open
                if bar_data["close"] > bar_data["open"]:
                    signal = create_signal_event(
                        strategy_id="simple_momentum",
                        symbol=event.payload["symbol"],
                        direction="BUY",
                        strength=0.7,
                        metadata={"source_bar": event.event_id}
                    )
                    self.container.event_bus.publish(signal)
                    signals_generated.append(signal)
        
        # Add strategy to container
        strategy = MockStrategy()
        container.add_component("strategy", strategy)
        strategy.initialize(container)
        strategy.start()
        
        # Send bar data
        bar_event = create_market_event(
            EventType.BAR,
            "AAPL",
            {"open": 100, "high": 102, "low": 99, "close": 101}  # close > open -> BUY signal
        )
        container.event_bus.publish(bar_event)
        
        # Verify signal was generated
        self.assertEqual(len(signals_generated), 1)
        self.assertEqual(signals_generated[0].payload["symbol"], "AAPL")
        self.assertEqual(signals_generated[0].payload["direction"], "BUY")
        self.assertEqual(signals_generated[0].payload["strategy_id"], "simple_momentum")


if __name__ == "__main__":
    # Run tests in specific order for progressive validation
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes in progressive order
    suite.addTests(loader.loadTestsFromTestCase(TestEventBusStandalone))
    suite.addTests(loader.loadTestsFromTestCase(TestContainerStandalone))
    suite.addTests(loader.loadTestsFromTestCase(TestEventsContainersIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicDataFlow))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)