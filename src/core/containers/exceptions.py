"""
Container-specific exceptions for better error handling.

This module provides specific exception types for container operations,
making error handling more precise and debugging easier.
"""

from typing import List, Optional
from .protocols import ContainerState


class ContainerError(Exception):
    """Base exception for all container-related errors."""
    pass


class ComponentError(ContainerError):
    """Base exception for component-related errors."""
    pass


class ComponentAlreadyExistsError(ComponentError):
    """Raised when trying to add a component that already exists."""
    
    def __init__(self, component_name: str):
        super().__init__(f"Component '{component_name}' already exists")
        self.component_name = component_name


class ComponentNotFoundError(ComponentError):
    """Raised when trying to access a component that doesn't exist."""
    
    def __init__(self, component_name: str):
        super().__init__(f"Component '{component_name}' not found")
        self.component_name = component_name


class ComponentDependencyError(ComponentError):
    """Raised when component dependency cannot be resolved."""
    
    def __init__(self, component_name: str, dependency_name: str):
        super().__init__(
            f"Component '{component_name}' dependency '{dependency_name}' not found"
        )
        self.component_name = component_name
        self.dependency_name = dependency_name


class ContainerStateError(ContainerError):
    """Base exception for container state-related errors."""
    pass


class InvalidContainerStateError(ContainerStateError):
    """Raised when container is in wrong state for requested operation."""
    
    def __init__(
        self, 
        container_name: str, 
        current_state: ContainerState, 
        expected_states: List[ContainerState]
    ):
        expected_str = ", ".join(s.value for s in expected_states)
        super().__init__(
            f"Container {container_name} in state {current_state.value}, "
            f"expected one of: [{expected_str}]"
        )
        self.container_name = container_name
        self.current_state = current_state
        self.expected_states = expected_states


class ContainerConfigError(ContainerError):
    """Base exception for configuration-related errors."""
    pass


class UnknownContainerRoleError(ContainerConfigError):
    """Raised when unknown container role is specified."""
    
    def __init__(self, role: str, available_roles: Optional[List[str]] = None):
        if available_roles:
            msg = f"Unknown container role: '{role}'. Available: {available_roles}"
        else:
            msg = f"Unknown container role: '{role}'"
        super().__init__(msg)
        self.role = role
        self.available_roles = available_roles


class InvalidContainerConfigError(ContainerConfigError):
    """Raised when container configuration is invalid."""
    
    def __init__(self, reason: str, config_key: Optional[str] = None):
        if config_key:
            msg = f"Invalid container configuration for '{config_key}': {reason}"
        else:
            msg = f"Invalid container configuration: {reason}"
        super().__init__(msg)
        self.reason = reason
        self.config_key = config_key


class ContainerHierarchyError(ContainerError):
    """Base exception for container hierarchy errors."""
    pass


class CircularContainerDependencyError(ContainerHierarchyError):
    """Raised when circular dependency detected in container hierarchy."""
    
    def __init__(self, container_path: List[str]):
        path_str = " -> ".join(container_path)
        super().__init__(f"Circular container dependency detected: {path_str}")
        self.container_path = container_path


class ParentContainerNotSetError(ContainerHierarchyError):
    """Raised when operation requires parent container but none is set."""
    
    def __init__(self, container_id: str, operation: str):
        super().__init__(
            f"Container {container_id} has no parent container for operation: {operation}"
        )
        self.container_id = container_id
        self.operation = operation