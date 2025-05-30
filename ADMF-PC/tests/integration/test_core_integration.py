"""
Integration tests for core infrastructure.

Tests interaction between containers, components, events, and configuration.
"""

import unittest
import asyncio
from datetime import datetime
from unittest.mock import Mock, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.containers.universal import UniversalScopedContainer
from src.core.components.registry import ComponentRegistry
from src.core.components.factory import ComponentFactory
from src.core.events.event_bus import EventBus
from src.core.events.types import Event, EventType
from src.core.config.schemas import ConfigSchema, SchemaField
from src.core.coordinator.coordinator import SystemCoordinator
from src.core.components.protocols import Component, ComponentMetadata


class MockTradingComponent(Component):
    """Mock trading component for testing."""
    
    def __init__(self, component_id: str, event_bus: EventBus = None):
        self.id = component_id
        self.event_bus = event_bus
        self.received_events = []
        self.initialized = False
        self.started = False
        
    def get_metadata(self):
        return ComponentMetadata(
            id=self.id,
            type="trading",
            capabilities=["event_processing", "data_handling"]
        )
    
    async def initialize(self, config=None):
        self.initialized = True
        self.config = config or {}
        
        # Subscribe to events if event bus provided
        if self.event_bus:
            self.event_bus.subscribe(EventType.ORDER, self.handle_event)
            self.event_bus.subscribe(EventType.FILL, self.handle_event)
    
    async def start(self):
        self.started = True
        
        # Emit startup event
        if self.event_bus:
            self.event_bus.publish(Event(
                event_type=EventType.SYSTEM,
                source_id=self.id,
                payload={"message": f"{self.id} started"}
            ))
    
    def handle_event(self, event: Event):
        self.received_events.append(event)
    
    def process_data(self, data):
        # Simulate processing and emit event
        if self.event_bus:
            self.event_bus.publish(Event(
                event_type=EventType.ORDER,
                source_id=self.id,
                payload={"data": data, "processed_by": self.id}
            ))


class TestContainerComponentIntegration(unittest.TestCase):
    """Test integration between containers and components."""
    
    def test_container_component_lifecycle(self):
        """Test component lifecycle within containers."""
        # Create container hierarchy
        root = UniversalScopedContainer("root", "system")
        trading = UniversalScopedContainer("trading", "subsystem")
        risk = UniversalScopedContainer("risk", "subsystem")
        
        root.add_child(trading)
        root.add_child(risk)
        
        # Add components
        event_bus = EventBus()
        
        trading_comp = MockTradingComponent("trader1", event_bus)
        risk_comp = MockTradingComponent("risk1", event_bus)
        
        trading.add_component("trader1", trading_comp)
        risk.add_component("risk1", risk_comp)
        
        # Start container hierarchy
        async def test():
            await root.start()
            
            # All components should be initialized and started
            self.assertTrue(trading_comp.initialized)
            self.assertTrue(trading_comp.started)
            self.assertTrue(risk_comp.initialized)
            self.assertTrue(risk_comp.started)
            
            # Components should be accessible
            self.assertEqual(trading.get_component("trader1"), trading_comp)
            self.assertEqual(risk.get_component("risk1"), risk_comp)
            
            # Stop everything
            await root.stop()
        
        asyncio.run(test())
    
    def test_cross_container_communication(self):
        """Test communication between components in different containers."""
        # Shared event bus
        event_bus = EventBus()
        
        # Create containers
        container1 = UniversalScopedContainer("container1", "trading")
        container2 = UniversalScopedContainer("container2", "risk")
        
        # Create components with shared event bus
        comp1 = MockTradingComponent("comp1", event_bus)
        comp2 = MockTradingComponent("comp2", event_bus)
        
        container1.add_component("comp1", comp1)
        container2.add_component("comp2", comp2)
        
        async def test():
            # Initialize components
            await comp1.initialize()
            await comp2.initialize()
            
            # comp1 processes data, should emit event
            comp1.process_data({"test": "data"})
            
            # comp2 should receive the event
            self.assertEqual(len(comp2.received_events), 1)
            self.assertEqual(comp2.received_events[0].source_id, "comp1")
        
        asyncio.run(test())


class TestComponentRegistryFactory(unittest.TestCase):
    """Test component registry and factory integration."""
    
    def test_registry_factory_workflow(self):
        """Test complete workflow with registry and factory."""
        registry = ComponentRegistry()
        factory = ComponentFactory()
        
        # Register component creator
        def create_trading_component(config):
            return MockTradingComponent(
                component_id=config.get("id", "default"),
                event_bus=config.get("event_bus")
            )
        
        factory.register_creator("trading", create_trading_component)
        
        # Create multiple components
        event_bus = EventBus()
        
        for i in range(3):
            config = {"id": f"trader_{i}", "event_bus": event_bus}
            component = factory.create("trading", config)
            registry.register(component)
        
        # Query registry
        all_traders = registry.get_by_type("trading")
        self.assertEqual(len(all_traders), 3)
        
        # Get by capability
        event_processors = registry.get_by_capability("event_processing")
        self.assertEqual(len(event_processors), 3)
        
        # Get specific component
        trader_1 = registry.get("trader_1")
        self.assertIsNotNone(trader_1)
        self.assertEqual(trader_1.id, "trader_1")


class TestEventSystemIntegration(unittest.TestCase):
    """Test event system integration with components."""
    
    def test_event_flow_through_system(self):
        """Test event flow through multiple components."""
        event_bus = EventBus()
        events_processed = []
        
        # Create processing chain
        class ProcessingComponent:
            def __init__(self, name, next_component=None):
                self.name = name
                self.next = next_component
                
            def process(self, event):
                events_processed.append((self.name, event))
                
                # Transform and forward
                if self.next:
                    new_event = Event(
                        event_type=event.event_type,
                        source_id=self.name,
                        payload={
                            **event.payload,
                            f"processed_by_{self.name}": True
                        }
                    )
                    event_bus.publish(new_event)
        
        # Create chain
        comp3 = ProcessingComponent("comp3")
        comp2 = ProcessingComponent("comp2", comp3)
        comp1 = ProcessingComponent("comp1", comp2)
        
        # Subscribe components
        event_bus.subscribe(EventType.ORDER, comp1.process)
        event_bus.subscribe(EventType.ORDER, comp2.process, source_filter="comp1")
        event_bus.subscribe(EventType.ORDER, comp3.process, source_filter="comp2")
        
        # Start the chain
        initial_event = Event(
            event_type=EventType.ORDER,
            source_id="start",
            payload={"value": 100}
        )
        
        event_bus.publish(initial_event)
        
        # Check all components processed in order
        self.assertEqual(len(events_processed), 3)
        self.assertEqual(events_processed[0][0], "comp1")
        self.assertEqual(events_processed[1][0], "comp2")
        self.assertEqual(events_processed[2][0], "comp3")
        
        # Check event transformation
        final_event = events_processed[2][1]
        self.assertTrue(final_event.payload.get("processed_by_comp1"))
        self.assertTrue(final_event.payload.get("processed_by_comp2"))
    
    def test_event_isolation_between_containers(self):
        """Test event isolation between containers."""
        from src.core.events.isolation import EventIsolation
        
        isolation = EventIsolation()
        
        # Create isolated event buses
        bus1 = isolation.create_isolated_bus("container1")
        bus2 = isolation.create_isolated_bus("container2")
        
        received1 = []
        received2 = []
        
        bus1.subscribe(EventType.ORDER, lambda e: received1.append(e))
        bus2.subscribe(EventType.ORDER, lambda e: received2.append(e))
        
        # Publish to bus1
        event = Event(EventType.ORDER, "test", {"data": "test"})
        bus1.publish(event)
        
        # Only bus1 should receive
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 0)
        
        # Now bridge for specific events
        isolation.bridge_buses(bus1, bus2, event_types=[EventType.FILL])
        
        # ORDER events still isolated
        bus1.publish(Event(EventType.ORDER, "test", {"data": "test2"}))
        self.assertEqual(len(received1), 2)
        self.assertEqual(len(received2), 0)
        
        # But FILL events are bridged
        received1.clear()
        received2.clear()
        
        bus1.subscribe(EventType.FILL, lambda e: received1.append(e))
        bus2.subscribe(EventType.FILL, lambda e: received2.append(e))
        
        fill_event = Event(EventType.FILL, "test", {"fill": "data"})
        bus1.publish(fill_event)
        
        # Both should receive FILL
        self.assertEqual(len(received1), 1)
        self.assertEqual(len(received2), 1)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration system integration."""
    
    def test_component_configuration(self):
        """Test configuring components with schemas."""
        # Define schema
        schema = ConfigSchema(
            name="TradingComponentConfig",
            fields=[
                SchemaField("id", "string", required=True),
                SchemaField("max_positions", "integer", default=10, min_value=1),
                SchemaField("risk_limit", "float", default=0.02, min_value=0.0, max_value=1.0),
                SchemaField("features", "object", properties={
                    "use_stops": SchemaField("use_stops", "boolean", default=True),
                    "stop_loss_pct": SchemaField("stop_loss_pct", "float", default=0.05)
                })
            ]
        )
        
        # User config (minimal)
        user_config = {
            "id": "trader_1",
            "risk_limit": 0.01
        }
        
        # Validate and apply defaults
        validation = schema.validate(user_config)
        self.assertTrue(validation.is_valid)
        
        full_config = schema.apply_defaults(user_config)
        
        # Check defaults were applied
        self.assertEqual(full_config["max_positions"], 10)
        self.assertEqual(full_config["features"]["use_stops"], True)
        self.assertEqual(full_config["features"]["stop_loss_pct"], 0.05)
        
        # Create component with config
        component = MockTradingComponent(full_config["id"])
        asyncio.run(component.initialize(full_config))
        
        self.assertEqual(component.config["risk_limit"], 0.01)
        self.assertEqual(component.config["max_positions"], 10)


class TestSystemCoordinator(unittest.TestCase):
    """Test system coordinator integration."""
    
    def test_coordinator_orchestration(self):
        """Test coordinator orchestrating multiple subsystems."""
        coordinator = SystemCoordinator()
        
        # Create subsystems
        trading_container = UniversalScopedContainer("trading", "subsystem")
        risk_container = UniversalScopedContainer("risk", "subsystem")
        data_container = UniversalScopedContainer("data", "subsystem")
        
        # Add to coordinator
        coordinator.add_subsystem("trading", trading_container)
        coordinator.add_subsystem("risk", risk_container)
        coordinator.add_subsystem("data", data_container)
        
        # Create shared event bus
        event_bus = EventBus()
        
        # Add components to subsystems
        trading_comp = MockTradingComponent("trader", event_bus)
        risk_comp = MockTradingComponent("risk_manager", event_bus)
        data_comp = MockTradingComponent("data_feed", event_bus)
        
        trading_container.add_component("trader", trading_comp)
        risk_container.add_component("risk_manager", risk_comp)
        data_container.add_component("data_feed", data_comp)
        
        # Test lifecycle coordination
        async def test():
            # Start all subsystems
            await coordinator.start_all()
            
            # All components should be started
            self.assertTrue(trading_comp.started)
            self.assertTrue(risk_comp.started)
            self.assertTrue(data_comp.started)
            
            # Simulate data flow
            data_comp.process_data({"symbol": "AAPL", "price": 150.00})
            
            # Other components should receive events
            self.assertGreater(len(trading_comp.received_events), 0)
            self.assertGreater(len(risk_comp.received_events), 0)
            
            # Stop all
            await coordinator.stop_all()
        
        asyncio.run(test())
    
    def test_coordinator_error_handling(self):
        """Test coordinator handling component errors."""
        coordinator = SystemCoordinator()
        
        # Create failing component
        class FailingComponent(Component):
            def get_metadata(self):
                return ComponentMetadata(id="failing", type="test")
            
            async def initialize(self, config=None):
                raise RuntimeError("Initialization failed")
        
        container = UniversalScopedContainer("test", "subsystem")
        container.add_component("failing", FailingComponent())
        
        coordinator.add_subsystem("test", container)
        
        # Coordinator should handle the error
        async def test():
            with self.assertRaises(RuntimeError):
                await coordinator.start_all()
            
            # Should track failed subsystem
            self.assertIn("test", coordinator.get_failed_subsystems())
        
        asyncio.run(test())


if __name__ == "__main__":
    unittest.main()