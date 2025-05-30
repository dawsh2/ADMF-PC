"""
Protocol-aware component registry for ADMF-PC.

This module provides a registry for discovering and managing components
based on their implemented protocols rather than class inheritance.
"""

from __future__ import annotations
from typing import Dict, Type, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
import inspect
import importlib
import logging
from pathlib import Path

from .protocols import (
    Component,
    detect_capabilities,
    has_capability,
    CAPABILITY_PROTOCOLS
)


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


# Global registry instance
_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _registry


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