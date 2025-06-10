"""
Integration tests using the REAL rewritten Events and Containers modules.

Now that we've confirmed imports work, these tests validate that your
rewritten architecture actually functions correctly together.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from the actual rewritten modules
from src.core.events import (
    EventBus, Event, EventType, EventHandler,
    create_market_event, create_signal_event, create_system_event
)

from src.core.containers import (
    Container, ContainerConfig, ContainerRole, ContainerState,
    ContainerProtocol, ContainerComponent
)

from src.core.events.barriers import (
    BarrierProtocol, create_standard_barriers, DataAlignmentBarrier
)

# Import test utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.fixtures import TestData, EventCollector, MockComponent


class TestRealEventBus(unittest.TestCase):
    """Test the real EventBus implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
    
    def test_real_event_creation(self):
        """Test creating events with real event creation functions."""
        # Test market event
        bar_event = create_market_event(
            event_type=EventType.BAR,
            symbol="AAPL",
            data={"open": 100, "high": 102, "low": 99, "close": 101}
        )
        
        self.assertEqual(bar_event.event_type, EventType.BAR.value)
        self.assertEqual(bar_event.payload["symbol"], "AAPL")
        self.assertEqual(bar_event.payload["close"], 101)
        
        # Test signal event
        signal_event = create_signal_event(
            symbol="MSFT",
            direction="BUY", 
            strength=0.8,
            strategy_id="momentum_1"
        )
        
        self.assertEqual(signal_event.event_type, EventType.SIGNAL.value)
        self.assertEqual(signal_event.payload["strategy_id"], "momentum_1")
        self.assertEqual(signal_event.payload["symbol"], "MSFT")
    
    def test_real_event_bus_pub_sub(self):
        """Test real EventBus publish/subscribe."""
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        # Subscribe (new interface doesn't return subscription_id)
        self.event_bus.subscribe(EventType.BAR.value, handler)
        # Verify subscription was successful by publishing and checking
        
        # Publish real event
        bar_event = create_market_event(
            EventType.BAR,
            "AAPL",
            TestData.sample_bar_data("AAPL", 150.0)
        )
        self.event_bus.publish(bar_event)
        
        # Verify
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].payload["symbol"], "AAPL")
    
    def test_signal_event_filtering_requirement(self):
        """Test that SIGNAL events have filtering requirements (as documented)."""
        signal_events = []
        
        def signal_handler(event):
            signal_events.append(event)
        
        # Subscribe to signals (filtering is REQUIRED for SIGNAL events)
        try:
            # This should fail without filter
            self.event_bus.subscribe(EventType.SIGNAL.value, signal_handler)
            self.fail("Expected ValueError for SIGNAL subscription without filter")
        except ValueError as e:
            # Expected - SIGNAL events require filters
            self.assertIn("filter", str(e).lower())
            
        # Now subscribe with proper filter
        self.event_bus.subscribe(
            EventType.SIGNAL.value, 
            signal_handler,
            filter_func=lambda e: e.payload.get('strategy_id') == 'test_strategy'
        )
        
        # Publish signal  
        signal = create_signal_event("AAPL", "BUY", 0.7, "test_strategy")
        self.event_bus.publish(signal)
        
        # Should receive filtered signal
        self.assertEqual(len(signal_events), 1)


class TestRealContainer(unittest.TestCase):
    """Test the real Container implementation."""
    
    def test_real_container_creation(self):
        """Test creating real containers."""
        config = ContainerConfig(
            name="real_test_backtest",
            components=["portfolio_manager"]  # Will infer 'portfolio' type
        )
        container = Container(config)
        
        # Verify container properties
        self.assertIsNotNone(container.container_id)
        self.assertEqual(container.container_type, "portfolio")
        self.assertEqual(container.config.name, "real_test_backtest")
        
        # Should have an event bus
        self.assertIsInstance(container.event_bus, EventBus)
    
    def test_real_container_component_management(self):
        """Test adding components to real container."""
        config = ContainerConfig(
            name="real_portfolio_test",
            components=["portfolio_manager"]
        )
        container = Container(config)
        
        # Create a component that follows the real protocol
        class TestStrategy:
            def __init__(self, name):
                self.name = name
                self.container = None
            
            def initialize(self, container):
                self.container = container
            
            def get_state(self):
                return {"name": self.name, "initialized": self.container is not None}
        
        strategy = TestStrategy("momentum_strategy")
        
        # Add to container
        container.add_component("strategy", strategy)
        
        # Retrieve
        retrieved = container.get_component("strategy")
        self.assertEqual(retrieved, strategy)
        self.assertEqual(retrieved.name, "momentum_strategy")


class RealTestComponent(ContainerComponent):
    """Real test component implementing ContainerComponent protocol."""
    
    def __init__(self, name: str):
        self.name = name
        self.container = None
        self.initialized = False
        self.started = False
        self.events_processed = []
    
    def initialize(self, container):
        """Initialize with container."""
        self.container = container
        self.initialized = True
        
        # Subscribe to events through container
        self.container.event_bus.subscribe(EventType.BAR.value, self.on_bar)
        # SIGNAL events require filter
        self.container.event_bus.subscribe(
            EventType.SIGNAL.value, 
            self.on_signal,
            filter_func=lambda e: True  # Accept all signals for test component
        )
    
    def start(self):
        """Start component."""
        self.started = True
    
    def stop(self):
        """Stop component."""
        self.started = False
    
    def get_state(self):
        """Get component state."""
        return {
            'name': self.name,
            'initialized': self.initialized,
            'started': self.started,
            'events_processed': len(self.events_processed)
        }
    
    def on_bar(self, event):
        """Handle bar events."""
        self.events_processed.append(('BAR', event))
        
        # Generate signal if close > open (simple strategy)
        if event.payload.get('close', 0) > event.payload.get('open', 0):
            signal = create_signal_event(
                symbol=event.payload['symbol'],
                direction="BUY",
                strength=0.7,
                strategy_id=f"{self.name}_strategy"
            )
            self.container.event_bus.publish(signal)
    
    def on_signal(self, event):
        """Handle signal events."""
        self.events_processed.append(('SIGNAL', event))


class TestRealEventsContainersIntegration(unittest.TestCase):
    """Test real Events + Containers integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ContainerConfig(
            name="integration_test",
            components=["portfolio_manager"]
        )
        self.container = Container(self.config)
    
    def test_real_container_event_flow(self):
        """Test real container event flow."""
        # Add real component
        component = RealTestComponent("test_strategy")
        self.container.add_component("strategy", component)
        
        # Initialize and start
        component.initialize(self.container)
        component.start()
        
        # Verify component is set up
        self.assertTrue(component.initialized)
        self.assertTrue(component.started)
        self.assertEqual(component.container, self.container)
        
        # Send bar data that should trigger signal
        bar_event = create_market_event(
            EventType.BAR,
            "AAPL",
            {"open": 100, "high": 102, "low": 99, "close": 101}  # close > open
        )
        self.container.event_bus.publish(bar_event)
        
        # Component should have processed bar and generated signal
        self.assertGreater(len(component.events_processed), 0)
        
        # Check for BAR event
        bar_events = [e for e in component.events_processed if e[0] == 'BAR']
        self.assertEqual(len(bar_events), 1)
        self.assertEqual(bar_events[0][1].payload['symbol'], 'AAPL')
        
        # Check for generated SIGNAL event
        signal_events = [e for e in component.events_processed if e[0] == 'SIGNAL']
        self.assertEqual(len(signal_events), 1)
        self.assertEqual(signal_events[0][1].payload['symbol'], 'AAPL')
        self.assertEqual(signal_events[0][1].payload['direction'], 'BUY')
    
    def test_real_multi_container_isolation(self):
        """Test that real containers are properly isolated."""
        # Create second container
        config2 = ContainerConfig(
            name="execution_test",
            components=["data_streamer"]
        )
        container2 = Container(config2)
        
        # Add components to each
        component1 = RealTestComponent("strategy_1")
        component2 = RealTestComponent("strategy_2")
        
        self.container.add_component("strategy", component1)
        container2.add_component("strategy", component2)
        
        # Initialize
        component1.initialize(self.container)
        component2.initialize(container2)
        
        # Publish to first container
        bar_event = create_market_event(
            EventType.BAR,
            "AAPL", 
            {"open": 100, "high": 101, "low": 99, "close": 100.5}
        )
        self.container.event_bus.publish(bar_event)
        
        # Only first component should receive
        self.assertGreater(len(component1.events_processed), 0)
        self.assertEqual(len(component2.events_processed), 0)
        
        # Publish to second container
        bar_event2 = create_market_event(
            EventType.BAR,
            "MSFT",
            {"open": 50, "high": 51, "low": 49, "close": 50.5}
        )
        container2.event_bus.publish(bar_event2)
        
        # Now second component should receive its event
        self.assertGreater(len(component2.events_processed), 0)
        # First component should still only have its events
        component1_bar_count = len([e for e in component1.events_processed if e[0] == 'BAR'])
        self.assertEqual(component1_bar_count, 1)


class TestRealBarriers(unittest.TestCase):
    """Test real barrier system."""
    
    def test_real_barrier_creation(self):
        """Test creating real barriers."""
        barrier = create_standard_barriers(
            container_id="test_container",
            symbols=["AAPL", "MSFT"],
            timeframes=["1m", "5m"],
            prevent_duplicates=True
        )
        
        self.assertIsNotNone(barrier)
        self.assertTrue(hasattr(barrier, 'should_proceed'))
        self.assertTrue(hasattr(barrier, 'update_state'))
        self.assertTrue(hasattr(barrier, 'reset'))
    
    def test_data_alignment_barrier(self):
        """Test data alignment barrier logic."""
        barrier = DataAlignmentBarrier(
            required_symbols=["AAPL"],
            required_timeframes=["1m"]
        )
        
        # Create bar event
        bar_event = create_market_event(
            EventType.BAR,
            "AAPL",
            {"timeframe": "1m", "close": 100}
        )
        
        # First check should succeed since we have the required data
        result = barrier.should_proceed(bar_event)
        self.assertTrue(result)  # Should pass once we have AAPL_1m data


class TestRealDataFlow(unittest.TestCase):
    """Test realistic data flow scenarios."""
    
    def test_complete_bar_to_signal_flow(self):
        """Test complete data flow: Bar → Strategy → Signal."""
        # Setup
        config = ContainerConfig(
            name="data_flow_test",
            components=["strategy"]
        )
        container = Container(config)
        
        # Add strategy component
        strategy = RealTestComponent("momentum_strategy")
        container.add_component("strategy", strategy)
        strategy.initialize(container)
        strategy.start()
        
        # Set up signal collector
        signals_received = []
        container.event_bus.subscribe(
            EventType.SIGNAL.value,
            lambda e: signals_received.append(e),
            filter_func=lambda e: True  # Accept all signals for test
        )
        
        # Send sequence of bars
        test_data = TestData()
        bar_sequence = [
            {"open": 100, "close": 99},    # close < open, no signal
            {"open": 99, "close": 101},    # close > open, BUY signal
            {"open": 101, "close": 102},   # close > open, BUY signal
        ]
        
        for i, bar_data in enumerate(bar_sequence):
            bar_event = create_market_event(
                EventType.BAR,
                "AAPL",
                bar_data
            )
            container.event_bus.publish(bar_event)
        
        # Should have generated 2 signals (for bars where close > open)
        self.assertEqual(len(signals_received), 2)
        
        # All signals should be BUY for AAPL from momentum_strategy
        for signal in signals_received:
            self.assertEqual(signal.payload['symbol'], 'AAPL')
            self.assertEqual(signal.payload['direction'], 'BUY')
            self.assertEqual(signal.payload['strategy_id'], 'momentum_strategy_strategy')
    
    def test_multi_symbol_flow(self):
        """Test handling multiple symbols."""
        config = ContainerConfig(
            name="multi_symbol_test",
            components=["strategy"]
        )
        container = Container(config)
        
        strategy = RealTestComponent("multi_symbol_strategy")
        container.add_component("strategy", strategy)
        strategy.initialize(container)
        
        # Send bars for different symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            bar_event = create_market_event(
                EventType.BAR,
                symbol,
                {"open": 100, "close": 101}  # Will trigger signal
            )
            container.event_bus.publish(bar_event)
        
        # Should have processed all symbols
        bar_events = [e for e in strategy.events_processed if e[0] == 'BAR']
        processed_symbols = {e[1].payload['symbol'] for e in bar_events}
        self.assertEqual(processed_symbols, set(symbols))


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)