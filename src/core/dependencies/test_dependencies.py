"""
Tests for the dependency management system.
"""

import unittest
from typing import Dict, Any

from .graph import DependencyGraph, DependencyValidator
from .container import DependencyContainer, ScopedContainer
from ..events import EventBus


# Test components

class DataService:
    """Mock data service."""
    
    def __init__(self):
        self.data = {"test": "data"}
        self.initialized = False
    
    @property
    def component_id(self):
        return "data_service"
    
    def initialize(self, context: Dict[str, Any]):
        self.initialized = True


class StrategyService:
    """Mock strategy that depends on data."""
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.signals = []
    
    @property
    def component_id(self):
        return "strategy_service"


class PortfolioService:
    """Mock portfolio that depends on event bus."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.positions = {}
    
    @property
    def component_id(self):
        return "portfolio_service"


class RiskService:
    """Mock risk service that depends on portfolio."""
    
    def __init__(self, portfolio: PortfolioService):
        self.portfolio = portfolio
    
    @property
    def component_id(self):
        return "risk_service"


# Circular dependency components
class ServiceA:
    def __init__(self, service_b: 'ServiceB'):
        self.service_b = service_b
    
    @property
    def component_id(self):
        return "service_a"


class ServiceB:
    def __init__(self, service_a: ServiceA):
        self.service_a = service_a
    
    @property
    def component_id(self):
        return "service_b"


# Tests

class TestDependencyGraph(unittest.TestCase):
    """Test dependency graph functionality."""
    
    def setUp(self):
        self.graph = DependencyGraph("test")
    
    def test_add_component(self):
        """Test adding components to graph."""
        node = self.graph.add_component("DataService", DataService)
        
        self.assertEqual(node.name, "DataService")
        self.assertEqual(node.component_type, DataService)
        self.assertFalse(node.is_resolved)
    
    def test_add_dependency(self):
        """Test adding dependencies."""
        self.graph.add_dependency("Strategy", "DataService")
        
        # Check forward edge
        deps = self.graph.get_dependencies("Strategy")
        self.assertIn("DataService", deps)
        
        # Check reverse edge
        dependents = self.graph.get_dependents("DataService")
        self.assertIn("Strategy", dependents)
    
    def test_detect_cycles(self):
        """Test circular dependency detection."""
        # No cycles initially
        self.assertEqual(len(self.graph.detect_cycles()), 0)
        
        # Add circular dependency
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("B", "C")
        self.graph.add_dependency("C", "A")
        
        cycles = self.graph.detect_cycles()
        self.assertEqual(len(cycles), 1)
        self.assertEqual(set(cycles[0][:3]), {"A", "B", "C"})
    
    def test_initialization_order(self):
        """Test topological sort for initialization."""
        # Build dependency chain: Data -> Strategy -> Risk
        self.graph.add_dependency("Risk", "Strategy")
        self.graph.add_dependency("Strategy", "Data")
        
        order = self.graph.get_initialization_order()
        
        # Data should come before Strategy
        self.assertLess(order.index("Data"), order.index("Strategy"))
        # Strategy should come before Risk  
        self.assertLess(order.index("Strategy"), order.index("Risk"))
    
    def test_initialization_order_with_cycles(self):
        """Test that initialization order fails with cycles."""
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("B", "A")
        
        with self.assertRaises(ValueError) as ctx:
            self.graph.get_initialization_order()
        
        self.assertIn("cycles", str(ctx.exception))
    
    def test_transitive_dependencies(self):
        """Test getting all transitive dependencies."""
        # Chain: A -> B -> C -> D
        self.graph.add_dependency("A", "B")
        self.graph.add_dependency("B", "C")
        self.graph.add_dependency("C", "D")
        
        deps = self.graph.get_all_dependencies("A")
        self.assertEqual(deps, {"B", "C", "D"})
    
    def test_resolution_tracking(self):
        """Test tracking resolved components."""
        self.graph.add_component("Data", DataService)
        
        self.assertFalse(self.graph.is_resolved("Data"))
        
        instance = DataService()
        self.graph.set_instance("Data", instance)
        
        self.assertTrue(self.graph.is_resolved("Data"))
        self.assertEqual(self.graph.get_instance("Data"), instance)
    
    def test_can_resolve(self):
        """Test checking if component can be resolved."""
        self.graph.add_dependency("Strategy", "Data")
        
        # Can't resolve Strategy without Data
        self.assertFalse(self.graph.can_resolve("Strategy"))
        
        # Resolve Data
        self.graph.add_component("Data", DataService)
        self.graph.set_instance("Data", DataService())
        
        # Now Strategy can be resolved
        self.assertTrue(self.graph.can_resolve("Strategy"))
    
    def test_validation(self):
        """Test graph validation."""
        # Add valid dependencies
        self.graph.add_component("Data")
        self.graph.add_component("Strategy")
        self.graph.add_dependency("Strategy", "Data")
        
        errors = self.graph.validate()
        self.assertEqual(len(errors), 0)
        
        # Add circular dependency
        self.graph.add_dependency("Data", "Strategy")
        
        errors = self.graph.validate()
        self.assertIn("circular_dependencies", errors)
        
        # Test orphaned component warning instead
        self.graph.add_component("OrphanedComponent")
        
        errors = self.graph.validate()
        self.assertIn("warnings", errors)
        self.assertTrue(any("OrphanedComponent" in w for w in errors.get("warnings", [])))


class TestDependencyContainer(unittest.TestCase):
    """Test dependency injection container."""
    
    def setUp(self):
        self.container = DependencyContainer("test")
        self.event_bus = EventBus("test")
    
    def test_register_type(self):
        """Test registering component types."""
        self.container.register_type("DataService", DataService)
        
        self.assertTrue(self.container.has("DataService"))
    
    def test_register_instance(self):
        """Test registering existing instances."""
        instance = DataService()
        self.container.register_instance("DataService", instance)
        
        resolved = self.container.resolve("DataService")
        self.assertIs(resolved, instance)
    
    def test_register_factory(self):
        """Test registering factory functions."""
        def create_data_service():
            return DataService()
        
        self.container.register_factory("DataService", create_data_service)
        
        resolved = self.container.resolve("DataService")
        self.assertIsInstance(resolved, DataService)
    
    def test_dependency_injection(self):
        """Test automatic dependency injection."""
        # Register dependencies
        self.container.register_instance("EventBus", self.event_bus)
        self.container.register_type("DataService", DataService)
        self.container.register_type("StrategyService", StrategyService,
                                    dependencies=["DataService"])
        
        # Resolve strategy
        strategy = self.container.resolve("StrategyService")
        
        # Should have data service injected
        self.assertIsInstance(strategy, StrategyService)
        self.assertIsInstance(strategy.data_service, DataService)
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies during resolution."""
        self.container.register_type("ServiceA", ServiceA, dependencies=["ServiceB"])
        self.container.register_type("ServiceB", ServiceB, dependencies=["ServiceA"])
        
        with self.assertRaises(ValueError) as ctx:
            self.container.resolve("ServiceA")
        
        self.assertIn("Circular dependency", str(ctx.exception))
    
    def test_singleton_behavior(self):
        """Test singleton registration."""
        self.container.register_type("DataService", DataService, singleton=True)
        
        # Multiple resolves should return same instance
        instance1 = self.container.resolve("DataService")
        instance2 = self.container.resolve("DataService")
        
        self.assertIs(instance1, instance2)
    
    def test_non_singleton_behavior(self):
        """Test non-singleton registration."""
        self.container.register_type("DataService", DataService, singleton=False)
        
        # Multiple resolves should return different instances
        instance1 = self.container.resolve("DataService")
        instance2 = self.container.resolve("DataService")
        
        self.assertIsNot(instance1, instance2)
    
    def test_resolve_all(self):
        """Test resolving multiple components in order."""
        # Register chain of dependencies
        self.container.register_instance("EventBus", self.event_bus)
        self.container.register_type("PortfolioService", PortfolioService,
                                    dependencies=["EventBus"])
        self.container.register_type("RiskService", RiskService,
                                    dependencies=["PortfolioService"])
        
        # Resolve all
        resolved = self.container.resolve_all(["RiskService", "PortfolioService"])
        
        self.assertIn("PortfolioService", resolved)
        self.assertIn("RiskService", resolved)
        self.assertIsInstance(resolved["RiskService"], RiskService)
    
    def test_reset(self):
        """Test container reset."""
        self.container.register_type("DataService", DataService)
        
        # Resolve to create instance
        instance1 = self.container.resolve("DataService")
        
        # Reset container
        self.container.reset()
        
        # Should create new instance
        instance2 = self.container.resolve("DataService")
        self.assertIsNot(instance1, instance2)
    
    def test_parent_container(self):
        """Test hierarchical resolution with parent container."""
        # Parent has shared services
        parent = DependencyContainer("parent")
        parent.register_instance("EventBus", self.event_bus)
        
        # Child container
        child = DependencyContainer("child", parent=parent)
        child.register_type("PortfolioService", PortfolioService,
                           dependencies=["EventBus"])
        
        # Should resolve EventBus from parent
        portfolio = child.resolve("PortfolioService")
        self.assertIs(portfolio.event_bus, self.event_bus)


class TestScopedContainer(unittest.TestCase):
    """Test scoped container for isolation."""
    
    def setUp(self):
        # Shared container with read-only services
        self.shared = DependencyContainer("shared")
        self.shared.register_instance("EventBus", EventBus("shared"))
        
        # Create scoped container
        self.scoped = ScopedContainer("trial_001", self.shared)
    
    def test_shared_resolution(self):
        """Test resolution of shared components."""
        self.scoped.register_shared("EventBus")
        
        # Should resolve from parent
        event_bus = self.scoped.resolve("EventBus")
        self.assertEqual(event_bus.container_id, "shared")
    
    def test_scoped_isolation(self):
        """Test that scoped components are isolated."""
        # Register same component in both containers
        self.shared.register_type("DataService", DataService)
        self.scoped.register_type("DataService", DataService)
        
        # Resolve in both
        shared_data = self.shared.resolve("DataService")
        scoped_data = self.scoped.resolve("DataService")
        
        # Should be different instances
        self.assertIsNot(shared_data, scoped_data)
    
    def test_teardown(self):
        """Test scoped container teardown."""
        # Register component with teardown
        class TeardownComponent:
            def __init__(self):
                self.torn_down = False
            
            @property
            def component_id(self):
                return "teardown"
            
            def teardown(self):
                self.torn_down = True
        
        self.scoped.register_type("TeardownComponent", TeardownComponent)
        component = self.scoped.resolve("TeardownComponent")
        
        # Teardown container
        self.scoped.teardown()
        
        # Component should be torn down
        self.assertTrue(component.torn_down)


class TestDependencyValidator(unittest.TestCase):
    """Test dependency direction validation."""
    
    def setUp(self):
        self.validator = DependencyValidator()
    
    def test_valid_direction(self):
        """Test valid dependency directions."""
        # Higher level depending on lower level is OK
        error = self.validator.validate_direction(
            "StrategyService", "DataService",
            "strategy", "data"
        )
        self.assertIsNone(error)
    
    def test_invalid_direction(self):
        """Test invalid dependency directions."""
        # Lower level depending on higher level is not OK
        error = self.validator.validate_direction(
            "DataService", "StrategyService",
            "data", "strategy"
        )
        self.assertIsNotNone(error)
        self.assertIn("Invalid dependency direction", error)


if __name__ == "__main__":
    unittest.main()