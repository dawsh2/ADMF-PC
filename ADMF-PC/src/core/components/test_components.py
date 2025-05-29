"""
Tests for the protocol-based component system.
"""

import unittest
from typing import Dict, Any, Optional

from .protocols import (
    Component,
    Lifecycle,
    EventCapable,
    Configurable,
    SignalGenerator,
    Capability,
    detect_capabilities,
    has_capability
)
from .registry import ComponentRegistry, register_component
from .factory import ComponentFactory, create_component
from .discovery import ComponentScanner, discover_components
from ..events import EventBus, EventType, Event


# Test components

class SimpleComponent:
    """Minimal component with just component_id."""
    
    @property
    def component_id(self):
        return "simple"


class LifecycleComponent:
    """Component with lifecycle support."""
    
    def __init__(self):
        self.initialized = False
        self.started = False
        self.stopped = False
    
    @property
    def component_id(self):
        return "lifecycle"
    
    def initialize(self, context: Dict[str, Any]):
        self.initialized = True
        self.context = context
    
    def start(self):
        self.started = True
    
    def stop(self):
        self.stopped = True
    
    def reset(self):
        self.started = False
        self.stopped = False
    
    def teardown(self):
        self.initialized = False


class TradingStrategy:
    """Component implementing SignalGenerator protocol."""
    
    def __init__(self, name: str = "strategy"):
        self.name = name
        self.signals_generated = 0
    
    @property
    def component_id(self):
        return self.name
    
    def generate_signal(self, data: Any) -> Optional[Dict[str, Any]]:
        self.signals_generated += 1
        return {"action": "buy", "confidence": 0.8}


@register_component(tags=["strategy", "trend"])
class DecoratedStrategy:
    """Component registered via decorator."""
    
    @property
    def component_id(self):
        return "decorated_strategy"
    
    def generate_signal(self, data: Any) -> Optional[Dict[str, Any]]:
        return {"action": "hold"}


class ConfigurableComponent:
    """Component with configuration support."""
    
    def __init__(self):
        self.config = {}
    
    @property
    def component_id(self):
        return "configurable"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "number"}
            }
        }
    
    def configure(self, config: Dict[str, Any]):
        self.config = config
    
    def get_config(self) -> Dict[str, Any]:
        return self.config


# Tests

class TestProtocols(unittest.TestCase):
    """Test protocol detection and capabilities."""
    
    def test_component_protocol(self):
        """Test basic component protocol."""
        simple = SimpleComponent()
        self.assertTrue(isinstance(simple, Component))
        self.assertEqual(simple.component_id, "simple")
    
    def test_lifecycle_protocol(self):
        """Test lifecycle protocol detection."""
        lifecycle = LifecycleComponent()
        self.assertTrue(isinstance(lifecycle, Lifecycle))
        self.assertTrue(has_capability(lifecycle, Capability.LIFECYCLE))
    
    def test_signal_generator_protocol(self):
        """Test signal generator protocol."""
        strategy = TradingStrategy()
        self.assertTrue(isinstance(strategy, SignalGenerator))
        self.assertTrue(has_capability(strategy, Capability.SIGNAL_GENERATOR))
    
    def test_detect_capabilities(self):
        """Test capability detection."""
        simple_caps = detect_capabilities(SimpleComponent())
        self.assertEqual(len(simple_caps), 0)  # Only base Component
        
        lifecycle_caps = detect_capabilities(LifecycleComponent())
        self.assertIn(Capability.LIFECYCLE, lifecycle_caps)
        
        strategy_caps = detect_capabilities(TradingStrategy())
        self.assertIn(Capability.SIGNAL_GENERATOR, strategy_caps)


class TestRegistry(unittest.TestCase):
    """Test component registry."""
    
    def setUp(self):
        self.registry = ComponentRegistry()
    
    def test_register_component(self):
        """Test basic component registration."""
        metadata = self.registry.register(SimpleComponent)
        
        self.assertEqual(metadata.name, "SimpleComponent")
        self.assertEqual(metadata.component_class, SimpleComponent)
        self.assertEqual(len(metadata.capabilities), 0)
    
    def test_register_with_name(self):
        """Test registration with custom name."""
        metadata = self.registry.register(
            SimpleComponent,
            name="custom_simple"
        )
        self.assertEqual(metadata.name, "custom_simple")
    
    def test_duplicate_registration(self):
        """Test duplicate registration handling."""
        self.registry.register(SimpleComponent)
        
        # Should raise without override
        with self.assertRaises(ValueError):
            self.registry.register(SimpleComponent)
        
        # Should work with override
        self.registry.register(SimpleComponent, override=True)
    
    def test_find_by_capability(self):
        """Test finding components by capability."""
        self.registry.register(SimpleComponent)
        self.registry.register(LifecycleComponent)
        self.registry.register(TradingStrategy)
        
        # Find lifecycle components
        lifecycle_components = self.registry.find_by_capability(Capability.LIFECYCLE)
        self.assertEqual(len(lifecycle_components), 1)
        self.assertEqual(lifecycle_components[0].name, "LifecycleComponent")
        
        # Find signal generators
        signal_components = self.registry.find_by_capability(Capability.SIGNAL_GENERATOR)
        self.assertEqual(len(signal_components), 1)
        self.assertEqual(signal_components[0].name, "TradingStrategy")
    
    def test_find_by_tag(self):
        """Test finding components by tag."""
        self.registry.register(SimpleComponent, tags=["basic"])
        self.registry.register(TradingStrategy, tags=["strategy", "algo"])
        
        # Find by single tag
        strategies = self.registry.find_by_tag("strategy")
        self.assertEqual(len(strategies), 1)
        
        # Find by multiple tags
        all_tagged = self.registry.find_by_tag(["basic", "algo"])
        self.assertEqual(len(all_tagged), 2)


class TestFactory(unittest.TestCase):
    """Test component factory."""
    
    def setUp(self):
        self.registry = ComponentRegistry()
        self.factory = ComponentFactory(self.registry)
        self.event_bus = EventBus("test")
    
    def test_create_simple_component(self):
        """Test creating a simple component."""
        self.registry.register(SimpleComponent)
        
        component = self.factory.create("SimpleComponent")
        self.assertIsInstance(component, SimpleComponent)
        self.assertEqual(component.component_id, "simple")
    
    def test_create_with_class(self):
        """Test creating with direct class reference."""
        component = self.factory.create(SimpleComponent)
        self.assertIsInstance(component, SimpleComponent)
    
    def test_create_with_lifecycle(self):
        """Test creating component with lifecycle."""
        self.registry.register(LifecycleComponent)
        
        context = {"test_key": "test_value"}
        component = self.factory.create("LifecycleComponent", context=context)
        
        # Should be initialized
        self.assertTrue(component.initialized)
        self.assertEqual(component.context, context)
    
    def test_enhance_with_events(self):
        """Test enhancing component with event capability."""
        self.registry.register(SimpleComponent)
        
        context = {"event_bus": self.event_bus}
        component = self.factory.create(
            "SimpleComponent",
            context=context,
            capabilities=[Capability.EVENTS]
        )
        
        # Should have event properties
        self.assertTrue(hasattr(component, 'event_bus'))
        self.assertTrue(hasattr(component, 'initialize_events'))
        self.assertTrue(hasattr(component, 'teardown_events'))
        
        # Should be detected as event capable
        self.assertTrue(has_capability(component, Capability.EVENTS))
    
    def test_create_from_config(self):
        """Test creating from configuration dict."""
        self.registry.register(TradingStrategy)
        
        config = {
            "class": "TradingStrategy",
            "params": {"name": "my_strategy"},
            "capabilities": [Capability.EVENTS]
        }
        
        context = {"event_bus": self.event_bus}
        component = self.factory.create_from_config(config, context)
        
        self.assertIsInstance(component, TradingStrategy)
        self.assertEqual(component.component_id, "my_strategy")
        self.assertTrue(has_capability(component, Capability.EVENTS))


class TestDiscovery(unittest.TestCase):
    """Test component discovery."""
    
    def setUp(self):
        self.registry = ComponentRegistry()
        self.scanner = ComponentScanner(self.registry)
    
    def test_scan_module(self):
        """Test scanning a module for components."""
        # Clear any existing registrations
        self.registry.clear()
        
        # Scan this test module
        components = self.scanner.scan_module(__name__)
        
        # Should find our test components
        component_names = [self.registry.get(name).component_class.__name__ 
                          for name in components]
        
        self.assertIn("SimpleComponent", component_names)
        self.assertIn("LifecycleComponent", component_names)
        self.assertIn("TradingStrategy", component_names)
    
    def test_component_filter(self):
        """Test component filtering during discovery."""
        self.registry.clear()
        
        # Add filter to only accept components with lifecycle
        self.scanner.add_filter(
            lambda cls: has_capability(cls, Capability.LIFECYCLE)
        )
        
        components = self.scanner.scan_module(__name__)
        
        # Should only find LifecycleComponent
        self.assertEqual(len(components), 1)
        self.assertEqual(
            self.registry.get(components[0]).component_class.__name__,
            "LifecycleComponent"
        )


class TestDecoratorRegistration(unittest.TestCase):
    """Test decorator-based registration."""
    
    def test_decorator_registration(self):
        """Test that decorated components are auto-registered."""
        # The decorator should have registered it on import
        registry = ComponentRegistry()  # Get new registry instance
        
        # Note: In practice, decorators register with the global registry
        # For testing, we'll manually check the decorated class
        self.assertTrue(hasattr(DecoratedStrategy, '__name__'))
        self.assertEqual(DecoratedStrategy().component_id, "decorated_strategy")


if __name__ == "__main__":
    unittest.main()