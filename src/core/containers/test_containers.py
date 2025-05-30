"""
Tests for the container system.
"""

import unittest
from unittest.mock import Mock, patch
import time
from datetime import datetime

from .universal import (
    UniversalScopedContainer,
    ContainerState,
    ContainerType,
    ComponentSpec
)
from .lifecycle import (
    ContainerLifecycleManager,
    LifecycleEvent
)
from .factory import ContainerFactory
from .bootstrap import ContainerBootstrap
from ..events import EventBus, EventType, Event


# Mock components for testing

class MockStrategy:
    """Mock strategy component."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.initialized = False
        self.started = False
        self.stopped = False
        
    @property
    def component_id(self):
        return "mock_strategy"
    
    def initialize(self, context):
        self.initialized = True
        self.context = context
    
    def start(self):
        self.started = True
    
    def stop(self):
        self.stopped = True
    
    def reset(self):
        self.started = False
        self.stopped = False
    
    def generate_signal(self, data):
        return {"action": "buy", "confidence": 0.8}


class MockPortfolio:
    """Mock portfolio component."""
    
    def __init__(self, initial_cash: float = 100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
    
    @property
    def component_id(self):
        return "mock_portfolio"
    
    def reset(self):
        self.cash = self.initial_cash
        self.positions = {}


# Tests

class TestUniversalScopedContainer(unittest.TestCase):
    """Test UniversalScopedContainer functionality."""
    
    def setUp(self):
        self.container = UniversalScopedContainer(
            container_id="test_container",
            container_type="test"
        )
    
    def tearDown(self):
        if hasattr(self, 'container'):
            self.container.dispose()
    
    def test_container_creation(self):
        """Test container is created properly."""
        self.assertEqual(self.container.container_id, "test_container")
        self.assertEqual(self.container.container_type, "test")
        self.assertEqual(self.container.state, ContainerState.CREATED)
        self.assertIsInstance(self.container.event_bus, EventBus)
    
    def test_component_creation(self):
        """Test creating components in container."""
        # Create component spec
        spec = ComponentSpec(
            name="Strategy",
            class_name=MockStrategy,
            params={"fast_period": 5, "slow_period": 20},
            capabilities=["lifecycle"]
        )
        
        # Create component
        self.container.create_component(spec)
        
        # Verify spec stored
        self.assertIn("Strategy", self.container._component_specs)
        self.assertEqual(len(self.container._component_order), 1)
    
    def test_component_initialization(self):
        """Test component initialization."""
        # Create component with direct class reference
        spec = ComponentSpec(
            name="Strategy",
            class_name=MockStrategy,  # Direct class reference
            params={"fast_period": 5}
        )
        self.container.create_component(spec)
        
        # Initialize
        strategy = self.container.initialize_component("Strategy")
        
        self.assertIsInstance(strategy, MockStrategy)
        self.assertEqual(strategy.fast_period, 5)
        self.assertIn("Strategy", self.container._initialized_components)
    
    def test_container_lifecycle(self):
        """Test container state transitions."""
        # Create component with direct class reference
        self.container.create_component(ComponentSpec(
            name="Strategy",
            class_name=MockStrategy
        ))
        
        # Initialize
        self.container.initialize_scope()
        self.assertEqual(self.container.state, ContainerState.INITIALIZED)
        
        # Start
        self.container.start()
        self.assertEqual(self.container.state, ContainerState.RUNNING)
        
        # Stop
        self.container.stop()
        self.assertEqual(self.container.state, ContainerState.STOPPED)
        
        # Reset
        self.container.reset()
        self.assertEqual(self.container.state, ContainerState.INITIALIZED)
    
    def test_shared_services(self):
        """Test shared services registration."""
        # Create container with shared services
        shared_data = Mock()
        container = UniversalScopedContainer(
            container_id="test_shared",
            shared_services={"DataProvider": shared_data}
        )
        
        try:
            # Verify shared service is registered
            self.assertTrue(container._dependency_container.has("DataProvider"))
            
            # Resolve shared service
            resolved = container._dependency_container.resolve("DataProvider")
            self.assertIs(resolved, shared_data)
        finally:
            container.dispose()
    
    def test_event_isolation(self):
        """Test event bus isolation between containers."""
        # Create two containers
        container1 = UniversalScopedContainer("container1")
        container2 = UniversalScopedContainer("container2")
        
        try:
            # Subscribe to events in each container
            events1 = []
            events2 = []
            
            container1.event_bus.subscribe(EventType.BAR, lambda e: events1.append(e))
            container2.event_bus.subscribe(EventType.BAR, lambda e: events2.append(e))
            
            # Publish event to container1
            event = Event(EventType.BAR, {"data": "test"})
            container1.event_bus.publish(event)
            
            # Only container1 should receive event
            self.assertEqual(len(events1), 1)
            self.assertEqual(len(events2), 0)
        finally:
            container1.dispose()
            container2.dispose()
    
    def test_container_stats(self):
        """Test container statistics."""
        stats = self.container.get_stats()
        
        self.assertEqual(stats['container_id'], "test_container")
        self.assertEqual(stats['container_type'], "test")
        self.assertEqual(stats['state'], "CREATED")
        self.assertEqual(stats['component_count'], 0)
        self.assertIn('event_stats', stats)


class TestContainerLifecycleManager(unittest.TestCase):
    """Test ContainerLifecycleManager functionality."""
    
    def setUp(self):
        self.manager = ContainerLifecycleManager(max_containers=5)
    
    def tearDown(self):
        self.manager.dispose_all()
    
    def test_create_container(self):
        """Test container creation through manager."""
        container_id = self.manager.create_container(
            container_type="test",
            initialize=False
        )
        
        self.assertIsNotNone(container_id)
        self.assertIn(container_id, self.manager._containers)
        
        # Get container
        container = self.manager.get_container(container_id)
        self.assertIsInstance(container, UniversalScopedContainer)
    
    def test_container_lifecycle_management(self):
        """Test managing container lifecycle."""
        # Create container
        container_id = self.manager.create_container(
            container_type="test",
            initialize=False
        )
        
        # Initialize
        self.manager.initialize_container(container_id)
        container = self.manager.get_container(container_id)
        self.assertEqual(container.state, ContainerState.INITIALIZED)
        
        # Start
        self.manager.start_container(container_id)
        self.assertEqual(container.state, ContainerState.RUNNING)
        
        # Stop
        self.manager.stop_container(container_id)
        self.assertEqual(container.state, ContainerState.STOPPED)
        
        # Dispose
        self.manager.dispose_container(container_id)
        self.assertNotIn(container_id, self.manager._containers)
    
    def test_container_limit(self):
        """Test container limit enforcement."""
        # Create max containers
        container_ids = []
        for i in range(5):
            cid = self.manager.create_container(f"test_{i}")
            container_ids.append(cid)
        
        # Access some containers to update access time
        time.sleep(0.01)
        self.manager.get_container(container_ids[2])
        self.manager.get_container(container_ids[3])
        
        # Create one more - should evict oldest
        new_id = self.manager.create_container("test_new")
        
        # Should have evicted the first (oldest accessed)
        self.assertNotIn(container_ids[0], self.manager._containers)
        self.assertIn(new_id, self.manager._containers)
        self.assertEqual(len(self.manager._containers), 5)
    
    def test_list_containers(self):
        """Test listing containers with filters."""
        # Create different types
        self.manager.create_container(container_type="backtest")
        self.manager.create_container(container_type="backtest")
        self.manager.create_container(container_type="optimization")
        
        # List all
        all_containers = self.manager.list_containers()
        self.assertEqual(len(all_containers), 3)
        
        # List by type
        backtest_containers = self.manager.list_containers(container_type="backtest")
        self.assertEqual(len(backtest_containers), 2)
        
        opt_containers = self.manager.list_containers(container_type="optimization")
        self.assertEqual(len(opt_containers), 1)
    
    def test_lifecycle_hooks(self):
        """Test lifecycle hook functionality."""
        hook_calls = []
        
        def test_hook(container):
            hook_calls.append((container.container_id, container.state))
        
        # Add hooks
        self.manager.add_lifecycle_hook(LifecycleEvent.INITIALIZED, test_hook)
        self.manager.add_lifecycle_hook(LifecycleEvent.STARTED, test_hook)
        
        # Create and start container
        container_id = self.manager.create_container(
            container_type="test",
            initialize=True,
            start=True
        )
        
        # Verify hooks were called
        self.assertEqual(len(hook_calls), 2)
        self.assertEqual(hook_calls[0][1], ContainerState.INITIALIZED)
        self.assertEqual(hook_calls[1][1], ContainerState.RUNNING)


class TestContainerFactory(unittest.TestCase):
    """Test ContainerFactory functionality."""
    
    def setUp(self):
        self.factory = ContainerFactory()
        
        # Register mock components
        from ..components import get_registry
        registry = get_registry()
        registry.register(MockStrategy, name="MockStrategy", override=True)
        registry.register(MockPortfolio, name="Portfolio", override=True)
    
    def tearDown(self):
        self.factory.lifecycle_manager.dispose_all()
    
    def test_create_backtest_container(self):
        """Test creating backtest container."""
        container_id = self.factory.create_backtest_container(
            strategy_spec={
                'class': 'MockStrategy',
                'parameters': {'fast_period': 15}
            }
        )
        
        self.assertIsNotNone(container_id)
        
        # Verify container type
        container = self.factory.get_container(container_id)
        self.assertEqual(container.container_type, ContainerType.BACKTEST.value)
        
        # Verify components created
        self.assertIn("Strategy", container._component_specs)
        self.assertIn("Portfolio", container._component_specs)
    
    def test_create_optimization_container(self):
        """Test creating optimization container."""
        container_id = self.factory.create_optimization_container(
            strategy_spec={
                'class': 'MockStrategy',
                'parameters': {'fast_period': 20}
            },
            trial_id="trial_001"
        )
        
        self.assertEqual(container_id, "opt_trial_trial_001")
        
        # Verify container
        container = self.factory.get_container(container_id)
        self.assertEqual(container.container_type, ContainerType.OPTIMIZATION.value)
    
    def test_custom_container(self):
        """Test creating custom container."""
        components = [
            {
                'name': 'CustomComponent',
                'class_name': 'MockStrategy',
                'params': {'fast_period': 25}
            }
        ]
        
        container_id = self.factory.create_custom_container(
            container_type="custom",
            components=components
        )
        
        container = self.factory.get_container(container_id)
        self.assertEqual(container.container_type, "custom")
        self.assertIn("CustomComponent", container._component_specs)


class TestContainerBootstrap(unittest.TestCase):
    """Test ContainerBootstrap functionality."""
    
    def setUp(self):
        self.bootstrap = ContainerBootstrap()
        
        # Register mock components
        from ..components import get_registry
        registry = get_registry()
        registry.register(MockStrategy, name="MockStrategy", override=True)
        registry.register(MockPortfolio, name="Portfolio", override=True)
    
    def tearDown(self):
        self.bootstrap.factory.lifecycle_manager.dispose_all()
    
    def test_create_container_from_spec(self):
        """Test creating container from specification."""
        spec = {
            'type': 'backtest',
            'strategy': {
                'class': 'MockStrategy',
                'parameters': {'fast_period': 12}
            }
        }
        
        container_id = self.bootstrap.create_container(spec)
        self.assertIsNotNone(container_id)
        
        # Verify container
        container = self.bootstrap.factory.get_container(container_id)
        self.assertEqual(container.container_type, 'backtest')
    
    def test_create_batch(self):
        """Test creating batch of containers."""
        batch_spec = {
            'base': {
                'type': 'optimization',
                'strategy': {
                    'class': 'MockStrategy'
                }
            },
            'variations': [
                {'id': 'opt_1', 'strategy': {'parameters': {'fast_period': 5}}},
                {'id': 'opt_2', 'strategy': {'parameters': {'fast_period': 10}}},
                {'id': 'opt_3', 'strategy': {'parameters': {'fast_period': 15}}}
            ]
        }
        
        container_ids = self.bootstrap.create_batch(batch_spec)
        
        self.assertEqual(len(container_ids), 3)
        # Container IDs are prefixed with "opt_trial_"
        self.assertIn('opt_trial_opt_1', container_ids)
        self.assertIn('opt_trial_opt_2', container_ids)
        self.assertIn('opt_trial_opt_3', container_ids)
    
    def test_shared_services(self):
        """Test shared service management."""
        # Add shared service
        mock_data = Mock()
        self.bootstrap.add_shared_service("DataProvider", mock_data)
        
        # Create container
        container_id = self.bootstrap.create_container({
            'type': 'backtest',
            'strategy': {'class': 'MockStrategy'}
        })
        
        # Verify shared service is available
        container = self.bootstrap.factory.get_container(container_id)
        self.assertTrue(container._dependency_container.has("DataProvider"))


if __name__ == "__main__":
    unittest.main()