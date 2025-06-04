"""
Component factory with capability enhancement and registry for ADMF-PC.

This module provides the factory system that creates components with
automatic capability detection and enhancement. Components get exactly
the capabilities they need without unnecessary overhead.

It also includes the registry for discovering and managing components
based on their implemented protocols rather than class inheritance.
"""

from __future__ import annotations
from typing import Dict, Type, Any, Optional, List, Union, Callable, Set
import inspect
import logging
from functools import wraps
from dataclasses import dataclass, field
import importlib
from pathlib import Path

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
    has_capability,
    CAPABILITY_PROTOCOLS
)
from ..events import EventBusProtocol, SubscriptionManager


logger = logging.getLogger(__name__)


@dataclass
class ComponentMetadata:
    """Metadata about a registered component."""
    
    name: str
    component_class: Type[Any]
    capabilities: List[str]
    module: str
    description: Optional[str] = None
    version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    
    def has_capability(self, capability: str) -> bool:
        """Check if component has a specific capability."""
        return capability in self.capabilities
    
    def has_all_capabilities(self, capabilities: List[str]) -> bool:
        """Check if component has all specified capabilities."""
        return all(cap in self.capabilities for cap in capabilities)
    
    def has_any_capability(self, capabilities: List[str]) -> bool:
        """Check if component has any of the specified capabilities."""
        return any(cap in self.capabilities for cap in capabilities)


class ComponentRegistry:
    """
    Registry for protocol-based components.
    
    This registry allows components to be discovered and registered based on
    the protocols they implement, enabling flexible component composition.
    """
    
    def __init__(self):
        """Initialize the component registry."""
        self._components: Dict[str, ComponentMetadata] = {}
        self._by_capability: Dict[str, Set[str]] = {}
        self._by_tag: Dict[str, Set[str]] = {}
        
        # Validators for component registration
        self._validators: List[Callable[[Type[Any]], bool]] = []
        
        logger.debug("ComponentRegistry initialized")
    
    def register(
        self,
        component_class: Type[Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        override: bool = False
    ) -> ComponentMetadata:
        """
        Register a component in the registry.
        
        Args:
            component_class: The component class to register
            name: Optional name (defaults to class name)
            description: Optional description
            version: Optional version string
            tags: Optional list of tags
            override: Whether to override existing registration
            
        Returns:
            ComponentMetadata for the registered component
            
        Raises:
            ValueError: If component is already registered and override=False
            TypeError: If component doesn't implement Component protocol
        """
        # Validate component
        if not self._validate_component(component_class):
            raise TypeError(
                f"Component {component_class} does not implement Component protocol"
            )
        
        # Generate name if not provided
        if name is None:
            name = component_class.__name__
        
        # Check for existing registration
        if name in self._components and not override:
            raise ValueError(f"Component '{name}' is already registered")
        
        # Detect capabilities
        capabilities = detect_capabilities(component_class)
        
        # Extract config schema if available
        config_schema = None
        if has_capability(component_class, "configurable"):
            try:
                # Try to get schema from class method
                if hasattr(component_class, 'get_config_schema'):
                    config_schema = component_class.get_config_schema()
            except:
                pass
        
        # Create metadata
        metadata = ComponentMetadata(
            name=name,
            component_class=component_class,
            capabilities=capabilities,
            module=component_class.__module__,
            description=description or component_class.__doc__,
            version=version,
            tags=tags or [],
            config_schema=config_schema
        )
        
        # Register component
        self._components[name] = metadata
        
        # Update capability index
        for capability in capabilities:
            if capability not in self._by_capability:
                self._by_capability[capability] = set()
            self._by_capability[capability].add(name)
        
        # Update tag index
        for tag in metadata.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = set()
            self._by_tag[tag].add(name)
        
        logger.info(
            f"Registered component '{name}' with capabilities: {capabilities}"
        )
        
        return metadata
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a component.
        
        Args:
            name: The component name to unregister
            
        Returns:
            True if component was unregistered, False if not found
        """
        if name not in self._components:
            return False
        
        metadata = self._components[name]
        
        # Remove from capability index
        for capability in metadata.capabilities:
            self._by_capability[capability].discard(name)
            if not self._by_capability[capability]:
                del self._by_capability[capability]
        
        # Remove from tag index
        for tag in metadata.tags:
            self._by_tag[tag].discard(name)
            if not self._by_tag[tag]:
                del self._by_tag[tag]
        
        # Remove component
        del self._components[name]
        
        logger.info(f"Unregistered component '{name}'")
        return True
    
    def get(self, name: str) -> Optional[ComponentMetadata]:
        """Get metadata for a specific component."""
        return self._components.get(name)
    
    def get_class(self, name: str) -> Optional[Type[Any]]:
        """Get the class for a specific component."""
        metadata = self.get(name)
        return metadata.component_class if metadata else None
    
    def find_by_capability(
        self,
        capability: Union[str, List[str]],
        match_all: bool = True
    ) -> List[ComponentMetadata]:
        """
        Find components by capability.
        
        Args:
            capability: Single capability or list of capabilities
            match_all: If True, components must have all capabilities.
                      If False, components must have at least one.
                      
        Returns:
            List of matching component metadata
        """
        capabilities = [capability] if isinstance(capability, str) else capability
        
        if match_all:
            # Find components with all capabilities
            matching_names = None
            for cap in capabilities:
                cap_components = self._by_capability.get(cap, set())
                if matching_names is None:
                    matching_names = cap_components.copy()
                else:
                    matching_names &= cap_components
            
            if matching_names is None:
                return []
        else:
            # Find components with any capability
            matching_names = set()
            for cap in capabilities:
                matching_names |= self._by_capability.get(cap, set())
        
        return [self._components[name] for name in matching_names]
    
    def find_by_tag(self, tag: Union[str, List[str]]) -> List[ComponentMetadata]:
        """Find components by tag(s)."""
        tags = [tag] if isinstance(tag, str) else tag
        
        matching_names = set()
        for t in tags:
            matching_names |= self._by_tag.get(t, set())
        
        return [self._components[name] for name in matching_names]
    
    def list_all(self) -> List[ComponentMetadata]:
        """List all registered components."""
        return list(self._components.values())
    
    def list_capabilities(self) -> List[str]:
        """List all available capabilities."""
        return list(self._by_capability.keys())
    
    def list_tags(self) -> List[str]:
        """List all used tags."""
        return list(self._by_tag.keys())
    
    def add_validator(self, validator: Callable[[Type[Any]], bool]) -> None:
        """
        Add a custom validator for component registration.
        
        Args:
            validator: Function that returns True if component is valid
        """
        self._validators.append(validator)
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._components.clear()
        self._by_capability.clear()
        self._by_tag.clear()
        logger.info("ComponentRegistry cleared")
    
    # Private methods
    
    def _validate_component(self, component_class: Type[Any]) -> bool:
        """Validate that a component meets requirements."""
        # Check basic Component protocol
        if not isinstance(component_class, type):
            return False
        
        # Component must at least have component_id property
        if not hasattr(component_class, 'component_id'):
            return False
        
        # Run custom validators
        for validator in self._validators:
            if not validator(component_class):
                return False
        
        return True


def get_registry() -> ComponentRegistry:
    """Get a component registry instance.
    
    Note: This now creates a new instance each time to avoid global state.
    Callers should manage their own registry instance.
    """
    return ComponentRegistry()


# Decorator for automatic registration
def register_component(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator for automatic component registration.
    
    Example:
        @register_component(tags=["strategy", "trend"])
        class MyTrendStrategy:
            @property
            def component_id(self):
                return "my_trend_strategy"
            
            def generate_signal(self, data):
                # Strategy logic
                pass
    """
    def decorator(cls: Type[Any]) -> Type[Any]:
        get_registry().register(
            cls,
            name=name,
            description=description,
            version=version,
            tags=tags
        )
        return cls
    
    return decorator


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
            registry: Component registry to use (creates new if not provided)
        """
        self.registry = registry if registry is not None else ComponentRegistry()
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