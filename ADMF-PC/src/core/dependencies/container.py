"""
Dependency injection container for ADMF-PC.

This module provides the container that manages component instances,
handles dependency injection, and integrates with the dependency graph
for proper resolution order.
"""

from __future__ import annotations
from typing import Dict, Type, Any, Optional, List, Callable, Union, Set
from dataclasses import dataclass, field
import inspect
import logging
from functools import wraps

from .graph import DependencyGraph
from ..components.protocols import Component, detect_capabilities
from ..components.factory import ComponentFactory


logger = logging.getLogger(__name__)


@dataclass
class Registration:
    """Represents a component registration in the container."""
    
    name: str
    registration_type: str  # "type", "instance", "factory"
    target: Any  # Type, instance, or factory function
    metadata: Dict[str, Any] = field(default_factory=dict)
    singleton: bool = True
    dependencies: List[str] = field(default_factory=list)


class DependencyContainer:
    """
    Dependency injection container with protocol support.
    
    This container manages component lifecycles, handles dependency
    injection, and ensures proper initialization order using the
    dependency graph.
    """
    
    def __init__(
        self,
        container_id: Optional[str] = None,
        parent: Optional['DependencyContainer'] = None
    ):
        """
        Initialize the container.
        
        Args:
            container_id: Optional identifier for this container
            parent: Optional parent container for hierarchical resolution
        """
        self.container_id = container_id
        self.parent = parent
        
        self._registrations: Dict[str, Registration] = {}
        self._instances: Dict[str, Any] = {}
        self._graph = DependencyGraph(container_id)
        self._factory = ComponentFactory()
        
        # Track resolution to detect circular dependencies
        self._resolving: Set[str] = set()
        
        logger.debug(f"DependencyContainer created: {container_id}")
    
    def register_type(
        self,
        name: str,
        component_type: Type[Any],
        singleton: bool = True,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a component type.
        
        Args:
            name: Name to register under
            component_type: The component class
            singleton: Whether to create a single instance
            dependencies: List of dependency names
            metadata: Additional metadata
        """
        dependencies = dependencies or self._infer_dependencies(component_type)
        
        registration = Registration(
            name=name,
            registration_type="type",
            target=component_type,
            singleton=singleton,
            dependencies=dependencies,
            metadata=metadata or {}
        )
        
        self._registrations[name] = registration
        
        # Update dependency graph
        self._graph.add_component(name, component_type, metadata)
        if dependencies:
            self._graph.add_dependencies(name, dependencies)
        
        logger.debug(f"Registered type '{name}': {component_type.__name__}")
    
    def register_instance(
        self,
        name: str,
        instance: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an existing instance.
        
        Args:
            name: Name to register under
            instance: The instance to register
            metadata: Additional metadata
        """
        registration = Registration(
            name=name,
            registration_type="instance",
            target=instance,
            singleton=True,
            metadata=metadata or {}
        )
        
        self._registrations[name] = registration
        self._instances[name] = instance
        
        # Update dependency graph
        self._graph.add_component(name, type(instance), metadata)
        self._graph.set_instance(name, instance)
        
        logger.debug(f"Registered instance '{name}': {type(instance).__name__}")
    
    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],
        singleton: bool = True,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a factory function.
        
        Args:
            name: Name to register under
            factory: Function that creates the component
            singleton: Whether to cache the result
            dependencies: List of dependency names
            metadata: Additional metadata
        """
        dependencies = dependencies or self._infer_dependencies(factory)
        
        registration = Registration(
            name=name,
            registration_type="factory",
            target=factory,
            singleton=singleton,
            dependencies=dependencies,
            metadata=metadata or {}
        )
        
        self._registrations[name] = registration
        
        # Update dependency graph
        self._graph.add_component(name, None, metadata)
        if dependencies:
            self._graph.add_dependencies(name, dependencies)
        
        logger.debug(f"Registered factory '{name}'")
    
    def resolve(self, name: str, resolution_path: Optional[List[str]] = None) -> Any:
        """
        Resolve a component by name.
        
        Args:
            name: Component name to resolve
            resolution_path: Path of components being resolved (for cycle detection)
            
        Returns:
            The resolved component instance
            
        Raises:
            ValueError: If component not found or circular dependency detected
        """
        resolution_path = resolution_path or []
        
        # Check for circular dependencies
        if name in self._resolving:
            cycle = resolution_path + [name]
            raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        # Check if already resolved
        if name in self._instances:
            return self._instances[name]
        
        # Check parent container
        if name not in self._registrations and self.parent:
            return self.parent.resolve(name, resolution_path)
        
        if name not in self._registrations:
            raise ValueError(f"Component '{name}' not registered")
        
        # Mark as resolving
        self._resolving.add(name)
        resolution_path.append(name)
        
        try:
            registration = self._registrations[name]
            
            if registration.registration_type == "instance":
                instance = registration.target
                
            elif registration.registration_type == "type":
                # Resolve dependencies first
                deps = self._resolve_dependencies(registration.dependencies, resolution_path)
                
                # Create instance with dependencies and params
                instance = self._create_instance(registration.target, deps, registration.metadata)
                
            elif registration.registration_type == "factory":
                # Resolve dependencies first
                deps = self._resolve_dependencies(registration.dependencies, resolution_path)
                
                # Call factory with dependencies
                instance = self._call_with_dependencies(registration.target, deps)
                
            else:
                raise ValueError(f"Unknown registration type: {registration.registration_type}")
            
            # Cache if singleton
            if registration.singleton:
                self._instances[name] = instance
                self._graph.set_instance(name, instance)
            
            return instance
            
        finally:
            self._resolving.remove(name)
            resolution_path.pop()
    
    def resolve_all(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Resolve multiple components in proper order.
        
        Args:
            names: Component names to resolve (None for all)
            
        Returns:
            Dictionary mapping names to instances
        """
        if names is None:
            names = list(self._registrations.keys())
        
        # Get resolution order from dependency graph
        resolution_order = self._graph.get_resolution_order(names)
        
        resolved = {}
        for name in resolution_order:
            if name in self._registrations:
                resolved[name] = self.resolve(name)
        
        return resolved
    
    def get(self, name: str) -> Optional[Any]:
        """Get a component if it exists, otherwise return None."""
        try:
            return self.resolve(name)
        except ValueError:
            return None
    
    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        if name in self._registrations:
            return True
        return self.parent.has(name) if self.parent else False
    
    def reset(self) -> None:
        """
        Reset the container, clearing cached instances.
        
        This is useful for ensuring fresh instances in new scopes.
        """
        self._instances.clear()
        
        # Reset graph instances
        for name in self._graph._nodes:
            self._graph._nodes[name].instance = None
        
        logger.debug(f"Container {self.container_id} reset")
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """
        Validate all registered dependencies.
        
        Returns:
            Dictionary of validation errors
        """
        errors = self._graph.validate()
        
        # Check that all dependencies can be resolved
        for name, registration in self._registrations.items():
            for dep in registration.dependencies:
                if not self.has(dep):
                    errors.setdefault("missing_dependencies", []).append(
                        f"Component '{name}' depends on unregistered component '{dep}'"
                    )
        
        return errors
    
    def get_dependency_graph(self) -> DependencyGraph:
        """Get the underlying dependency graph."""
        return self._graph
    
    # Private methods
    
    def _infer_dependencies(self, target: Union[Type[Any], Callable]) -> List[str]:
        """
        Infer dependencies from constructor or function parameters.
        
        Args:
            target: Class or function to analyze
            
        Returns:
            List of inferred dependency names
        """
        if inspect.isclass(target):
            # Look at __init__ parameters
            init_method = getattr(target, '__init__', None)
            if not init_method:
                return []
            sig = inspect.signature(init_method)
        else:
            # Function or callable
            sig = inspect.signature(target)
        
        dependencies = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
                
            # Check for type annotations
            if param.annotation != inspect.Parameter.empty:
                # Only consider non-builtin types as dependencies
                type_name = getattr(param.annotation, '__name__', None)
                if type_name and not isinstance(param.annotation, type(int)):
                    # Skip builtin types
                    if param.annotation.__module__ != 'builtins':
                        dependencies.append(type_name)
                elif param_name in self._registrations:
                    # Fall back to parameter name if registered
                    dependencies.append(param_name)
        
        return dependencies
    
    def _resolve_dependencies(
        self,
        dependencies: List[str],
        resolution_path: List[str]
    ) -> Dict[str, Any]:
        """Resolve a list of dependencies."""
        resolved = {}
        
        for dep in dependencies:
            resolved[dep] = self.resolve(dep, resolution_path.copy())
        
        return resolved
    
    def _create_instance(
        self,
        component_type: Type[Any],
        dependencies: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create an instance with resolved dependencies."""
        # Get constructor signature
        sig = inspect.signature(component_type.__init__)
        
        # Build kwargs from dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Try to match by parameter name
            if param_name in dependencies:
                kwargs[param_name] = dependencies[param_name]
            # Try to match by type annotation
            elif param.annotation != inspect.Parameter.empty:
                type_name = getattr(param.annotation, '__name__', None)
                if type_name and type_name in dependencies:
                    kwargs[param_name] = dependencies[type_name]
        
        # Add params from metadata if available
        if metadata and 'params' in metadata:
            # Params override dependency-injected values
            kwargs.update(metadata['params'])
        
        # Add container context
        context = {
            'container': self,
            'container_id': self.container_id,
            'event_bus': dependencies.get('EventBus'),
            **dependencies
        }
        
        # Use factory to create with capabilities
        return self._factory.create(
            component_type,
            context=context,
            **kwargs
        )
    
    def _call_with_dependencies(
        self,
        factory: Callable,
        dependencies: Dict[str, Any]
    ) -> Any:
        """Call a factory function with resolved dependencies."""
        sig = inspect.signature(factory)
        
        # Build kwargs from dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in dependencies:
                kwargs[param_name] = dependencies[param_name]
            elif param.annotation != inspect.Parameter.empty:
                type_name = getattr(param.annotation, '__name__', None)
                if type_name and type_name in dependencies:
                    kwargs[param_name] = dependencies[type_name]
        
        return factory(**kwargs)


class ScopedContainer(DependencyContainer):
    """
    Scoped container for isolated component execution.
    
    This container type is designed for complete state isolation,
    perfect for parallel backtests or optimization trials.
    """
    
    def __init__(
        self,
        scope_id: str,
        shared_container: Optional[DependencyContainer] = None
    ):
        """
        Initialize a scoped container.
        
        Args:
            scope_id: Unique identifier for this scope
            shared_container: Optional container with shared services
        """
        super().__init__(container_id=f"scope_{scope_id}", parent=shared_container)
        self.scope_id = scope_id
        
        # Track shared vs scoped registrations
        self._shared_names: Set[str] = set()
        
        logger.info(f"Created scoped container: {scope_id}")
    
    def register_shared(self, name: str) -> None:
        """
        Mark a component as shared from parent container.
        
        Shared components are resolved from the parent and not
        recreated in this scope.
        """
        if not self.parent or not self.parent.has(name):
            raise ValueError(f"Cannot share non-existent component '{name}'")
        
        self._shared_names.add(name)
        logger.debug(f"Component '{name}' marked as shared in scope {self.scope_id}")
    
    def resolve(self, name: str, resolution_path: Optional[List[str]] = None) -> Any:
        """Override to handle shared components."""
        # Always resolve shared components from parent
        if name in self._shared_names and self.parent:
            return self.parent.resolve(name, resolution_path)
        
        return super().resolve(name, resolution_path)
    
    def teardown(self) -> None:
        """
        Teardown the scoped container.
        
        This should be called when the scope completes to ensure
        proper cleanup of all scoped components.
        """
        # Get teardown order
        teardown_order = self._graph.get_teardown_order()
        
        # Teardown components in reverse order
        for name in teardown_order:
            if name in self._instances and name not in self._shared_names:
                instance = self._instances[name]
                
                # Call teardown if lifecycle capable
                if hasattr(instance, 'teardown'):
                    try:
                        instance.teardown()
                    except Exception as e:
                        logger.error(f"Error tearing down {name}: {e}")
        
        # Clear instances
        self.reset()
        
        logger.info(f"Scoped container {self.scope_id} torn down")