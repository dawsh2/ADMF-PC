"""
Unit tests for core component system.

Tests component registry, factory, discovery, and protocols.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.components.registry import ComponentRegistry
from src.core.components.factory import ComponentFactory
from src.core.components.discovery import ComponentDiscovery
from src.core.components.protocols import (
    Component,
    ComponentMetadata,
    Capability,
    LifecycleComponent
)


class TestComponentMetadata(unittest.TestCase):
    """Test component metadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating component metadata."""
        metadata = ComponentMetadata(
            id="test_component",
            type="processor",
            version="1.0.0",
            capabilities=["data_processing", "event_handling"],
            dependencies=["logger", "config"],
            configuration={"buffer_size": 1024}
        )
        
        self.assertEqual(metadata.id, "test_component")
        self.assertEqual(metadata.type, "processor")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertIn("data_processing", metadata.capabilities)
        self.assertEqual(len(metadata.dependencies), 2)
    
    def test_metadata_defaults(self):
        """Test metadata with default values."""
        metadata = ComponentMetadata(
            id="simple",
            type="basic"
        )
        
        self.assertEqual(metadata.version, "0.0.0")
        self.assertEqual(len(metadata.capabilities), 0)
        self.assertEqual(len(metadata.dependencies), 0)
        self.assertEqual(len(metadata.configuration), 0)


class TestComponentRegistry(unittest.TestCase):
    """Test component registry."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ComponentRegistry()
    
    def test_register_component(self):
        """Test registering a component."""
        mock_component = Mock(spec=Component)
        mock_component.get_metadata.return_value = ComponentMetadata(
            id="test_comp",
            type="processor"
        )
        
        self.registry.register(mock_component)
        
        self.assertTrue(self.registry.has("test_comp"))
        self.assertEqual(self.registry.get("test_comp"), mock_component)
    
    def test_register_duplicate(self):
        """Test registering duplicate component."""
        mock_component = Mock(spec=Component)
        mock_component.get_metadata.return_value = ComponentMetadata(
            id="test_comp",
            type="processor"
        )
        
        self.registry.register(mock_component)
        
        # Should raise error on duplicate
        with self.assertRaises(ValueError):
            self.registry.register(mock_component)
    
    def test_unregister_component(self):
        """Test unregistering a component."""
        mock_component = Mock(spec=Component)
        mock_component.get_metadata.return_value = ComponentMetadata(
            id="test_comp",
            type="processor"
        )
        
        self.registry.register(mock_component)
        self.registry.unregister("test_comp")
        
        self.assertFalse(self.registry.has("test_comp"))
    
    def test_get_by_type(self):
        """Test getting components by type."""
        # Register multiple components
        for i in range(3):
            mock_comp = Mock(spec=Component)
            mock_comp.get_metadata.return_value = ComponentMetadata(
                id=f"processor_{i}",
                type="processor"
            )
            self.registry.register(mock_comp)
        
        # Register different type
        other_comp = Mock(spec=Component)
        other_comp.get_metadata.return_value = ComponentMetadata(
            id="handler_1",
            type="handler"
        )
        self.registry.register(other_comp)
        
        processors = self.registry.get_by_type("processor")
        self.assertEqual(len(processors), 3)
        
        handlers = self.registry.get_by_type("handler")
        self.assertEqual(len(handlers), 1)
    
    def test_get_by_capability(self):
        """Test getting components by capability."""
        # Component with capabilities
        capable_comp = Mock(spec=Component)
        capable_comp.get_metadata.return_value = ComponentMetadata(
            id="capable",
            type="processor",
            capabilities=["async", "batch"]
        )
        self.registry.register(capable_comp)
        
        # Component without capability
        basic_comp = Mock(spec=Component)
        basic_comp.get_metadata.return_value = ComponentMetadata(
            id="basic",
            type="processor",
            capabilities=["sync"]
        )
        self.registry.register(basic_comp)
        
        async_comps = self.registry.get_by_capability("async")
        self.assertEqual(len(async_comps), 1)
        self.assertEqual(async_comps[0].get_metadata().id, "capable")


class TestComponentFactory(unittest.TestCase):
    """Test component factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = ComponentFactory()
    
    def test_register_creator(self):
        """Test registering component creator."""
        def create_processor(config):
            mock_comp = Mock(spec=Component)
            mock_comp.get_metadata.return_value = ComponentMetadata(
                id=config.get("id", "processor"),
                type="processor"
            )
            return mock_comp
        
        self.factory.register_creator("processor", create_processor)
        
        # Create component
        comp = self.factory.create("processor", {"id": "test_proc"})
        self.assertIsNotNone(comp)
        self.assertEqual(comp.get_metadata().id, "test_proc")
    
    def test_create_unknown_type(self):
        """Test creating unknown component type."""
        with self.assertRaises(ValueError):
            self.factory.create("unknown_type", {})
    
    def test_creator_with_dependencies(self):
        """Test creator with dependency injection."""
        def create_with_deps(config, dependencies):
            mock_comp = Mock(spec=Component)
            mock_comp.dependencies = dependencies
            return mock_comp
        
        self.factory.register_creator("dependent", create_with_deps)
        
        deps = {"logger": Mock(), "config": Mock()}
        comp = self.factory.create("dependent", {}, dependencies=deps)
        
        self.assertEqual(comp.dependencies, deps)


class TestComponentDiscovery(unittest.TestCase):
    """Test component discovery."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ComponentRegistry()
        self.discovery = ComponentDiscovery(self.registry)
    
    def test_discover_by_interface(self):
        """Test discovering components by interface."""
        # Create components with different interfaces
        class DataProcessor:
            def process(self, data): pass
        
        class EventHandler:
            def handle(self, event): pass
        
        # Mock components
        processor = Mock(spec=[Component, DataProcessor])
        processor.get_metadata.return_value = ComponentMetadata(
            id="proc1",
            type="processor"
        )
        
        handler = Mock(spec=[Component, EventHandler])
        handler.get_metadata.return_value = ComponentMetadata(
            id="handler1",
            type="handler"
        )
        
        self.registry.register(processor)
        self.registry.register(handler)
        
        # Discover by interface
        processors = self.discovery.find_by_interface(DataProcessor)
        self.assertEqual(len(processors), 1)
        self.assertEqual(processors[0].get_metadata().id, "proc1")
    
    def test_discover_compatible_components(self):
        """Test discovering compatible components."""
        # Component that requires certain capabilities
        consumer = Mock(spec=Component)
        consumer.get_metadata.return_value = ComponentMetadata(
            id="consumer",
            type="consumer",
            dependencies=["data_source"]
        )
        
        # Compatible provider
        provider = Mock(spec=Component)
        provider.get_metadata.return_value = ComponentMetadata(
            id="provider",
            type="provider",
            capabilities=["data_source"]
        )
        
        # Incompatible component
        other = Mock(spec=Component)
        other.get_metadata.return_value = ComponentMetadata(
            id="other",
            type="other",
            capabilities=["something_else"]
        )
        
        self.registry.register(consumer)
        self.registry.register(provider)
        self.registry.register(other)
        
        # Find compatible components
        compatible = self.discovery.find_compatible(consumer)
        self.assertEqual(len(compatible), 1)
        self.assertEqual(compatible[0].get_metadata().id, "provider")


class TestLifecycleComponent(unittest.TestCase):
    """Test lifecycle component protocol."""
    
    def test_lifecycle_methods(self):
        """Test component lifecycle methods."""
        class TestComponent(LifecycleComponent):
            def __init__(self):
                self.initialized = False
                self.started = False
                self.stopped = False
            
            def initialize(self, config):
                self.initialized = True
                self.config = config
            
            def start(self):
                if not self.initialized:
                    raise RuntimeError("Not initialized")
                self.started = True
            
            def stop(self):
                self.stopped = True
            
            def get_metadata(self):
                return ComponentMetadata(id="test", type="test")
        
        comp = TestComponent()
        
        # Test lifecycle
        comp.initialize({"key": "value"})
        self.assertTrue(comp.initialized)
        
        comp.start()
        self.assertTrue(comp.started)
        
        comp.stop()
        self.assertTrue(comp.stopped)
    
    def test_lifecycle_order_enforcement(self):
        """Test that lifecycle order is enforced."""
        class StrictComponent(LifecycleComponent):
            def __init__(self):
                self.state = "created"
            
            def initialize(self, config):
                if self.state != "created":
                    raise RuntimeError(f"Invalid state for init: {self.state}")
                self.state = "initialized"
            
            def start(self):
                if self.state != "initialized":
                    raise RuntimeError(f"Invalid state for start: {self.state}")
                self.state = "started"
            
            def stop(self):
                if self.state != "started":
                    raise RuntimeError(f"Invalid state for stop: {self.state}")
                self.state = "stopped"
            
            def get_metadata(self):
                return ComponentMetadata(id="strict", type="test")
        
        comp = StrictComponent()
        
        # Can't start without init
        with self.assertRaises(RuntimeError):
            comp.start()
        
        # Proper sequence
        comp.initialize({})
        comp.start()
        comp.stop()
        
        # Can't stop twice
        with self.assertRaises(RuntimeError):
            comp.stop()


class TestCapabilities(unittest.TestCase):
    """Test capability system."""
    
    def test_capability_definition(self):
        """Test defining capabilities."""
        class DataProcessingCapability(Capability):
            """Capability for data processing."""
            
            def get_required_methods(self):
                return ["process_data", "validate_data"]
            
            def validate(self, component):
                # Check if component has required methods
                for method in self.get_required_methods():
                    if not hasattr(component, method):
                        return False, f"Missing method: {method}"
                return True, None
        
        capability = DataProcessingCapability()
        
        # Valid component
        class ValidProcessor:
            def process_data(self, data): pass
            def validate_data(self, data): pass
        
        valid_comp = ValidProcessor()
        is_valid, error = capability.validate(valid_comp)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Invalid component
        class InvalidProcessor:
            def process_data(self, data): pass
            # Missing validate_data
        
        invalid_comp = InvalidProcessor()
        is_valid, error = capability.validate(invalid_comp)
        self.assertFalse(is_valid)
        self.assertIn("validate_data", error)
    
    def test_capability_composition(self):
        """Test composing multiple capabilities."""
        class AsyncCapability(Capability):
            def validate(self, component):
                if hasattr(component, "async_process"):
                    return True, None
                return False, "Missing async_process"
        
        class BatchCapability(Capability):
            def validate(self, component):
                if hasattr(component, "batch_process"):
                    return True, None
                return False, "Missing batch_process"
        
        # Component with both capabilities
        class FullProcessor:
            def async_process(self): pass
            def batch_process(self): pass
        
        comp = FullProcessor()
        
        async_cap = AsyncCapability()
        batch_cap = BatchCapability()
        
        # Both should validate
        self.assertTrue(async_cap.validate(comp)[0])
        self.assertTrue(batch_cap.validate(comp)[0])


if __name__ == "__main__":
    unittest.main()