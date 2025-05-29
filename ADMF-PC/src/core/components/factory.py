"""
Component factory with capability enhancement for ADMF-PC.

This module provides the factory system that creates components with
automatic capability detection and enhancement. Components get exactly
the capabilities they need without unnecessary overhead.
"""

from __future__ import annotations
from typing import Dict, Type, Any, Optional, List, Union, Callable
import inspect
import logging
from functools import wraps

from .protocols import (
    Component,
    Lifecycle,
    EventCapable,
    Configurable,
    Optimizable,
    Monitorable,
    Stateful,
    Capability,
    detect_capabilities,
    has_capability
)
from .registry import ComponentRegistry, get_registry
from ..events import EventBusProtocol, SubscriptionManager


logger = logging.getLogger(__name__)


class CapabilityEnhancer:
    """Base class for capability enhancers."""
    
    def can_enhance(self, component: Any, capability: str) -> bool:
        """Check if this enhancer can add the capability to the component."""
        return False
    
    def enhance(self, component: Any, context: Dict[str, Any]) -> Any:
        """Enhance the component with the capability."""
        return component


class LifecycleEnhancer(CapabilityEnhancer):
    """Adds lifecycle management to components."""
    
    def can_enhance(self, component: Any, capability: str) -> bool:
        return capability == Capability.LIFECYCLE
    
    def enhance(self, component: Any, context: Dict[str, Any]) -> Any:
        """Add lifecycle methods if not present."""
        if has_capability(component, Capability.LIFECYCLE):
            return component
        
        # Add default lifecycle methods
        if not hasattr(component, 'initialize'):
            component.initialize = lambda ctx: None
        if not hasattr(component, 'start'):
            component.start = lambda: None
        if not hasattr(component, 'stop'):
            component.stop = lambda: None
        if not hasattr(component, 'reset'):
            component.reset = lambda: None
        if not hasattr(component, 'teardown'):
            component.teardown = lambda: None
        
        return component


class EventEnhancer(CapabilityEnhancer):
    """Adds event system integration to components."""
    
    def can_enhance(self, component: Any, capability: str) -> bool:
        return capability == Capability.EVENTS
    
    def enhance(self, component: Any, context: Dict[str, Any]) -> Any:
        """Add event system support."""
        if has_capability(component, Capability.EVENTS):
            return component
        
        # Get event bus from context
        event_bus = context.get('event_bus')
        if not event_bus:
            raise ValueError("Event bus not provided in context")
        
        # Add event properties and methods
        component.event_bus = event_bus
        component._subscription_manager = SubscriptionManager(
            event_bus,
            getattr(component, 'component_id', 'unknown')
        )
        
        # Add event methods
        def initialize_events():
            # Call original method if exists
            if hasattr(component, '_original_initialize_events'):
                component._original_initialize_events()
        
        def teardown_events():
            component._subscription_manager.unsubscribe_all()
            # Call original method if exists
            if hasattr(component, '_original_teardown_events'):
                component._original_teardown_events()
        
        component.initialize_events = initialize_events
        component.teardown_events = teardown_events
        
        return component


class ComponentFactory:
    """
    Factory for creating protocol-based components with capabilities.
    
    This factory creates components and automatically enhances them with
    requested capabilities, ensuring minimal overhead for simple components.
    """
    
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        """
        Initialize the factory.
        
        Args:
            registry: Component registry to use (defaults to global)
        """
        self.registry = registry or get_registry()
        self._enhancers: Dict[str, CapabilityEnhancer] = {
            Capability.LIFECYCLE: LifecycleEnhancer(),
            Capability.EVENTS: EventEnhancer()
        }
        
        # Register infrastructure capabilities if available
        try:
            from ..infrastructure.capabilities import (
                LoggingCapability,
                MonitoringCapability,
                ErrorHandlingCapability,
                DebuggingCapability,
                ValidationCapability
            )
            
            self._enhancers.update({
                'logging': LoggingCapability(),
                'monitoring': MonitoringCapability(),
                'error_handling': ErrorHandlingCapability(),
                'debugging': DebuggingCapability(),
                'validation': ValidationCapability()
            })
            logger.debug("Infrastructure capabilities registered")
        except ImportError:
            logger.debug("Infrastructure capabilities not available")
        
        logger.debug("ComponentFactory initialized")
    
    def create(
        self,
        component_spec: Union[str, Type[Any], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Create a component instance with requested capabilities.
        
        Args:
            component_spec: Component name, class, or configuration dict
            context: Container context (event_bus, container_id, etc.)
            capabilities: Additional capabilities to add
            **kwargs: Arguments passed to component constructor
            
        Returns:
            Component instance with requested capabilities
            
        Raises:
            ValueError: If component cannot be created
        """
        context = context or {}
        
        # Parse component specification
        if isinstance(component_spec, str):
            # Look up by name in registry
            metadata = self.registry.get(component_spec)
            if not metadata:
                raise ValueError(f"Component '{component_spec}' not found in registry")
            component_class = metadata.component_class
            
        elif isinstance(component_spec, type):
            # Direct class reference
            component_class = component_spec
            
        elif isinstance(component_spec, dict):
            # Configuration dictionary
            return self._create_from_config(component_spec, context)
            
        else:
            raise ValueError(f"Invalid component specification: {component_spec}")
        
        # Create component instance
        component = self._instantiate(component_class, context, kwargs)
        
        # Detect existing capabilities
        existing_capabilities = detect_capabilities(component)
        
        # Enhance with requested capabilities
        if capabilities:
            for capability in capabilities:
                if capability not in existing_capabilities:
                    component = self._enhance_capability(component, capability, context)
        
        # Initialize if lifecycle capable
        if has_capability(component, Capability.LIFECYCLE) and context:
            component.initialize(context)
        
        # Initialize events if event capable
        if has_capability(component, Capability.EVENTS):
            component.initialize_events()
        
        logger.debug(
            f"Created component {getattr(component, 'component_id', 'unknown')} "
            f"with capabilities: {detect_capabilities(component)}"
        )
        
        return component
    
    def create_from_config(
        self,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a component from a configuration dictionary.
        
        Config format:
        {
            "class": "ComponentName" or ComponentClass,
            "params": {...},  # Constructor parameters
            "capabilities": ["lifecycle", "events", ...],
            "config": {...}   # Component configuration
        }
        """
        return self._create_from_config(config, context or {})
    
    def add_enhancer(self, capability: str, enhancer: CapabilityEnhancer) -> None:
        """Add a custom capability enhancer."""
        self._enhancers[capability] = enhancer
    
    # Private methods
    
    def _instantiate(
        self,
        component_class: Type[Any],
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Any:
        """Instantiate a component with proper arguments."""
        # Get constructor signature
        sig = inspect.signature(component_class.__init__)
        params = sig.parameters
        
        # Build constructor arguments
        init_args = {}
        
        # Add context items if constructor accepts them
        for param_name, param in params.items():
            if param_name == 'self':
                continue
                
            # Check context for matching parameter
            if param_name in context:
                init_args[param_name] = context[param_name]
            elif param_name == 'container_id' and 'container_id' in context:
                init_args['container_id'] = context['container_id']
            elif param_name == 'event_bus' and 'event_bus' in context:
                init_args['event_bus'] = context['event_bus']
        
        # Add explicit kwargs
        init_args.update(kwargs)
        
        # Create instance
        return component_class(**init_args)
    
    def _create_from_config(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Create component from configuration dictionary."""
        # Extract configuration
        component_class = config.get('class')
        params = config.get('params', {})
        capabilities = config.get('capabilities', [])
        component_config = config.get('config', {})
        
        # Create component
        component = self.create(
            component_class,
            context=context,
            capabilities=capabilities,
            **params
        )
        
        # Apply configuration if configurable
        if component_config and has_capability(component, Capability.CONFIGURABLE):
            component.configure(component_config)
        
        return component
    
    def _enhance_capability(
        self,
        component: Any,
        capability: str,
        context: Dict[str, Any]
    ) -> Any:
        """Enhance a component with a capability."""
        enhancer = self._enhancers.get(capability)
        
        if not enhancer:
            logger.warning(f"No enhancer available for capability: {capability}")
            return component
        
        if not enhancer.can_enhance(component, capability):
            logger.warning(
                f"Cannot enhance {component} with capability: {capability}"
            )
            return component
        
        return enhancer.enhance(component, context)


# Convenience functions

def create_component(
    component_spec: Union[str, Type[Any], Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    capabilities: Optional[List[str]] = None,
    **kwargs
) -> Any:
    """
    Create a component using the global factory.
    
    This is a convenience function that uses the default factory
    and registry to create components.
    """
    factory = ComponentFactory()
    return factory.create(component_spec, context, capabilities, **kwargs)


def create_minimal_component(
    component_class: Type[Any],
    **kwargs
) -> Any:
    """
    Create a minimal component without any capabilities.
    
    This creates the component with zero framework overhead,
    perfect for simple components that don't need lifecycle,
    events, or other framework features.
    """
    return component_class(**kwargs)