"""
Unit tests for container system.

Tests universal containers, lifecycle management, and isolation.
"""

import unittest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.containers.universal import UniversalScopedContainer
from src.core.containers.lifecycle import ContainerLifecycle, LifecycleState
from src.core.containers.factory import ContainerFactory
from src.core.containers.bootstrap import ContainerBootstrap
from src.core.components.protocols import Component, ComponentMetadata, Capability


class MockComponent(Component):
    """Mock component for testing."""
    
    def __init__(self, component_id="mock", component_type="test"):
        self.id = component_id
        self.type = component_type
        self.initialized = False
        self.started = False
        self.stopped = False
    
    def get_metadata(self):
        return ComponentMetadata(
            id=self.id,
            type=self.type,
            capabilities=["test_capability"]
        )
    
    async def initialize(self, config=None):
        self.initialized = True
        self.config = config or {}
    
    async def start(self):
        if not self.initialized:
            raise RuntimeError("Not initialized")
        self.started = True
    
    async def stop(self):
        self.stopped = True


class TestUniversalScopedContainer(unittest.TestCase):
    """Test UniversalScopedContainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.container = UniversalScopedContainer(
            container_id="test_container",
            container_type="test"
        )
    
    def test_container_creation(self):
        """Test creating a container."""
        self.assertEqual(self.container.container_id, "test_container")
        self.assertEqual(self.container.container_type, "test")
        self.assertIsNone(self.container.parent)
        self.assertEqual(len(self.container._children), 0)
    
    def test_add_component(self):
        """Test adding components to container."""
        component = MockComponent("comp1")
        
        self.container.add_component("comp1", component)
        
        self.assertIn("comp1", self.container._components)
        self.assertEqual(self.container.get_component("comp1"), component)
    
    def test_add_duplicate_component(self):
        """Test adding duplicate component."""
        component1 = MockComponent("comp1")
        component2 = MockComponent("comp1")
        
        self.container.add_component("comp1", component1)
        
        with self.assertRaises(ValueError):
            self.container.add_component("comp1", component2)
    
    def test_remove_component(self):
        """Test removing components."""
        component = MockComponent("comp1")
        
        self.container.add_component("comp1", component)
        removed = self.container.remove_component("comp1")
        
        self.assertEqual(removed, component)
        self.assertNotIn("comp1", self.container._components)
        self.assertIsNone(self.container.get_component("comp1"))
    
    def test_child_containers(self):
        """Test parent-child container relationships."""
        child1 = UniversalScopedContainer("child1", "child")
        child2 = UniversalScopedContainer("child2", "child")
        
        self.container.add_child(child1)
        self.container.add_child(child2)
        
        self.assertEqual(len(self.container._children), 2)
        self.assertEqual(child1.parent, self.container)
        self.assertEqual(child2.parent, self.container)
    
    def test_remove_child(self):
        """Test removing child containers."""
        child = UniversalScopedContainer("child", "child")
        
        self.container.add_child(child)
        removed = self.container.remove_child("child")
        
        self.assertEqual(removed, child)
        self.assertIsNone(child.parent)
        self.assertEqual(len(self.container._children), 0)
    
    def test_component_lookup_hierarchy(self):
        """Test component lookup through hierarchy."""
        # Create parent-child structure
        parent = UniversalScopedContainer("parent", "parent")
        child = UniversalScopedContainer("child", "child")
        parent.add_child(child)
        
        # Add components at different levels
        parent_comp = MockComponent("shared", "parent_type")
        child_comp = MockComponent("local", "child_type")
        
        parent.add_component("shared", parent_comp)
        child.add_component("local", child_comp)
        
        # Child can access its own components
        self.assertEqual(child.get_component("local"), child_comp)
        
        # Child can access parent components
        self.assertEqual(child.get_component("shared", search_parent=True), parent_comp)
        
        # Parent cannot access child components
        self.assertIsNone(parent.get_component("local"))
    
    def test_async_lifecycle(self):
        """Test async lifecycle management."""
        component = MockComponent("async_comp")
        self.container.add_component("async_comp", component)
        
        async def test_lifecycle():
            # Start container
            await self.container.start()
            self.assertTrue(component.initialized)
            self.assertTrue(component.started)
            
            # Stop container
            await self.container.stop()
            self.assertTrue(component.stopped)
        
        asyncio.run(test_lifecycle())
    
    def test_lifecycle_propagation(self):
        """Test lifecycle propagation to children."""
        parent = UniversalScopedContainer("parent", "parent")
        child = UniversalScopedContainer("child", "child")
        parent.add_child(child)
        
        parent_comp = MockComponent("p_comp")
        child_comp = MockComponent("c_comp")
        
        parent.add_component("p_comp", parent_comp)
        child.add_component("c_comp", child_comp)
        
        async def test_propagation():
            # Start parent should start child
            await parent.start()
            
            self.assertTrue(parent_comp.started)
            self.assertTrue(child_comp.started)
            
            # Stop parent should stop child
            await parent.stop()
            
            self.assertTrue(parent_comp.stopped)
            self.assertTrue(child_comp.stopped)
        
        asyncio.run(test_propagation())
    
    def test_isolation(self):
        """Test container isolation."""
        container1 = UniversalScopedContainer("cont1", "isolated")
        container2 = UniversalScopedContainer("cont2", "isolated")
        
        comp1 = MockComponent("comp", "type1")
        comp2 = MockComponent("comp", "type2")
        
        container1.add_component("comp", comp1)
        container2.add_component("comp", comp2)
        
        # Components should be isolated
        self.assertEqual(container1.get_component("comp"), comp1)
        self.assertEqual(container2.get_component("comp"), comp2)
        self.assertIsNot(comp1, comp2)
    
    def test_get_all_components(self):
        """Test getting all components."""
        comp1 = MockComponent("comp1", "type1")
        comp2 = MockComponent("comp2", "type2")
        comp3 = MockComponent("comp3", "type1")
        
        self.container.add_component("comp1", comp1)
        self.container.add_component("comp2", comp2)
        self.container.add_component("comp3", comp3)
        
        # Get all components
        all_comps = self.container.get_all_components()
        self.assertEqual(len(all_comps), 3)
        
        # Get by type
        type1_comps = self.container.get_components_by_type("type1")
        self.assertEqual(len(type1_comps), 2)
        self.assertIn(comp1, type1_comps)
        self.assertIn(comp3, type1_comps)


class TestContainerLifecycle(unittest.TestCase):
    """Test container lifecycle management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lifecycle = ContainerLifecycle()
    
    def test_lifecycle_states(self):
        """Test lifecycle state transitions."""
        # Initial state
        self.assertEqual(self.lifecycle.state, LifecycleState.CREATED)
        
        # Valid transitions
        asyncio.run(self.lifecycle.initialize())
        self.assertEqual(self.lifecycle.state, LifecycleState.INITIALIZED)
        
        asyncio.run(self.lifecycle.start())
        self.assertEqual(self.lifecycle.state, LifecycleState.STARTED)
        
        asyncio.run(self.lifecycle.stop())
        self.assertEqual(self.lifecycle.state, LifecycleState.STOPPED)
    
    def test_invalid_transitions(self):
        """Test invalid lifecycle transitions."""
        # Can't start from CREATED
        with self.assertRaises(RuntimeError):
            asyncio.run(self.lifecycle.start())
        
        # Initialize first
        asyncio.run(self.lifecycle.initialize())
        
        # Can't initialize again
        with self.assertRaises(RuntimeError):
            asyncio.run(self.lifecycle.initialize())
        
        # Start
        asyncio.run(self.lifecycle.start())
        
        # Can't initialize when started
        with self.assertRaises(RuntimeError):
            asyncio.run(self.lifecycle.initialize())
    
    def test_lifecycle_hooks(self):
        """Test lifecycle hooks."""
        events = []
        
        async def on_init():
            events.append("init")
        
        async def on_start():
            events.append("start")
        
        async def on_stop():
            events.append("stop")
        
        self.lifecycle.add_hook(LifecycleState.INITIALIZED, on_init)
        self.lifecycle.add_hook(LifecycleState.STARTED, on_start)
        self.lifecycle.add_hook(LifecycleState.STOPPED, on_stop)
        
        async def run_lifecycle():
            await self.lifecycle.initialize()
            await self.lifecycle.start()
            await self.lifecycle.stop()
        
        asyncio.run(run_lifecycle())
        
        self.assertEqual(events, ["init", "start", "stop"])
    
    def test_lifecycle_error_handling(self):
        """Test error handling in lifecycle."""
        async def failing_hook():
            raise ValueError("Hook failed")
        
        self.lifecycle.add_hook(LifecycleState.INITIALIZED, failing_hook)
        
        # Should handle error and continue
        try:
            asyncio.run(self.lifecycle.initialize())
            # Should still transition
            self.assertEqual(self.lifecycle.state, LifecycleState.INITIALIZED)
        except ValueError:
            self.fail("Lifecycle should handle hook errors")


class TestContainerFactory(unittest.TestCase):
    """Test container factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = ContainerFactory()
    
    def test_register_container_type(self):
        """Test registering container types."""
        def create_test_container(config):
            return UniversalScopedContainer(
                container_id=config.get("id", "test"),
                container_type="test"
            )
        
        self.factory.register_type("test", create_test_container)
        
        # Create container
        container = self.factory.create("test", {"id": "my_test"})
        
        self.assertIsNotNone(container)
        self.assertEqual(container.container_id, "my_test")
        self.assertEqual(container.container_type, "test")
    
    def test_create_with_components(self):
        """Test creating container with components."""
        def create_with_components(config):
            container = UniversalScopedContainer(
                container_id=config["id"],
                container_type="complex"
            )
            
            # Add components based on config
            for comp_config in config.get("components", []):
                comp = MockComponent(
                    comp_config["id"],
                    comp_config["type"]
                )
                container.add_component(comp_config["id"], comp)
            
            return container
        
        self.factory.register_type("complex", create_with_components)
        
        # Create with components
        config = {
            "id": "complex_container",
            "components": [
                {"id": "comp1", "type": "processor"},
                {"id": "comp2", "type": "handler"}
            ]
        }
        
        container = self.factory.create("complex", config)
        
        self.assertEqual(len(container._components), 2)
        self.assertIsNotNone(container.get_component("comp1"))
        self.assertIsNotNone(container.get_component("comp2"))
    
    def test_create_hierarchy(self):
        """Test creating container hierarchy."""
        def create_parent(config):
            parent = UniversalScopedContainer(
                config["id"],
                "parent"
            )
            
            # Create children
            for child_config in config.get("children", []):
                child = self.factory.create(
                    child_config["type"],
                    child_config
                )
                parent.add_child(child)
            
            return parent
        
        def create_child(config):
            return UniversalScopedContainer(
                config["id"],
                "child"
            )
        
        self.factory.register_type("parent", create_parent)
        self.factory.register_type("child", create_child)
        
        # Create hierarchy
        config = {
            "id": "root",
            "children": [
                {"id": "child1", "type": "child"},
                {"id": "child2", "type": "child"}
            ]
        }
        
        root = self.factory.create("parent", config)
        
        self.assertEqual(len(root._children), 2)
        self.assertIsNotNone(root._children.get("child1"))
        self.assertIsNotNone(root._children.get("child2"))


class TestContainerBootstrap(unittest.TestCase):
    """Test container bootstrap process."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bootstrap = ContainerBootstrap()
    
    def test_bootstrap_phases(self):
        """Test bootstrap phases."""
        phases_executed = []
        
        def phase1():
            phases_executed.append("phase1")
        
        def phase2():
            phases_executed.append("phase2")
            
        def phase3():
            phases_executed.append("phase3")
        
        self.bootstrap.add_phase("init", phase1)
        self.bootstrap.add_phase("configure", phase2)
        self.bootstrap.add_phase("start", phase3)
        
        # Execute bootstrap
        self.bootstrap.execute()
        
        # Phases should execute in order
        self.assertEqual(phases_executed, ["phase1", "phase2", "phase3"])
    
    def test_bootstrap_with_dependencies(self):
        """Test bootstrap with phase dependencies."""
        executed = []
        
        def database():
            executed.append("database")
        
        def cache():
            executed.append("cache")
        
        def app():
            executed.append("app")
        
        # Add phases with dependencies
        self.bootstrap.add_phase("app", app, dependencies=["database", "cache"])
        self.bootstrap.add_phase("cache", cache, dependencies=["database"])
        self.bootstrap.add_phase("database", database)
        
        # Execute
        self.bootstrap.execute()
        
        # Should respect dependencies
        self.assertEqual(executed, ["database", "cache", "app"])
    
    def test_bootstrap_error_handling(self):
        """Test error handling in bootstrap."""
        def failing_phase():
            raise RuntimeError("Phase failed")
        
        def normal_phase():
            pass
        
        self.bootstrap.add_phase("failing", failing_phase)
        self.bootstrap.add_phase("normal", normal_phase)
        
        # Should raise error
        with self.assertRaises(RuntimeError):
            self.bootstrap.execute()
    
    def test_bootstrap_rollback(self):
        """Test rollback on bootstrap failure."""
        rolled_back = []
        
        def phase1():
            pass
        
        def phase1_rollback():
            rolled_back.append("phase1")
        
        def phase2():
            raise RuntimeError("Phase 2 failed")
        
        self.bootstrap.add_phase("phase1", phase1, rollback=phase1_rollback)
        self.bootstrap.add_phase("phase2", phase2)
        
        # Execute with rollback
        try:
            self.bootstrap.execute(rollback_on_error=True)
        except RuntimeError:
            pass
        
        # Should have rolled back phase1
        self.assertEqual(rolled_back, ["phase1"])


if __name__ == "__main__":
    unittest.main()