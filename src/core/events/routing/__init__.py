"""
Event routing system for cross-container communication.

This module provides the infrastructure for containers to communicate
across isolation boundaries while maintaining the benefits of isolated
event buses.
"""

from .protocols import (
    EventQoS,
    EventScope,
    EventFilter,
    BatchingConfig,
    EventPublication,
    EventSubscription,
    ValidationResult,
    EventRouterProtocol
)

from .router import EventRouter
from .interface import ContainerEventInterface

__all__ = [
    # Enums
    'EventQoS',
    'EventScope',
    # Data classes
    'EventFilter',
    'BatchingConfig', 
    'EventPublication',
    'EventSubscription',
    'ValidationResult',
    # Protocols
    'EventRouterProtocol',
    # Implementations
    'EventRouter',
    'ContainerEventInterface'
]