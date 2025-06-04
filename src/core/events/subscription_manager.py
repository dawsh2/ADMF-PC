"""
Subscription management for containerized components.

This module provides the SubscriptionManager that tracks all event
subscriptions for a component and ensures proper cleanup during
component teardown.
"""

from __future__ import annotations
from typing import Dict, List, Set, Union, Optional, Callable, Any
from dataclasses import dataclass, field
from weakref import WeakMethod, ref
import logging

from ..types.events import Event, EventType, EventHandler, EventBusProtocol


logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """Represents a single event subscription."""
    event_type: Union[EventType, str]
    handler: EventHandler
    source_filter: Optional[Set[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubscriptionManager:
    """
    Manages event subscriptions for a component.
    
    This class tracks all subscriptions made by a component and provides
    a clean way to unsubscribe from all events during teardown. It's
    designed to work with the container's EventBus to ensure proper
    cleanup when containers are disposed.
    """
    
    def __init__(
        self,
        event_bus: EventBusProtocol,
        component_id: Optional[str] = None
    ):
        """
        Initialize the subscription manager.
        
        Args:
            event_bus: The event bus to subscribe to
            component_id: Optional identifier for the component
        """
        self.event_bus = event_bus
        self.component_id = component_id
        self._subscriptions: List[Subscription] = []
        self._active = True
        
        logger.debug(f"SubscriptionManager created for component: {component_id}")
    
    def subscribe(
        self,
        event_type: Union[EventType, str],
        handler: EventHandler,
        source_filter: Optional[Union[str, List[str]]] = None,
        **metadata
    ) -> None:
        """
        Subscribe to an event type and track the subscription.
        
        Args:
            event_type: The type of events to subscribe to
            handler: The handler function
            source_filter: Optional source ID(s) to filter by
            **metadata: Additional metadata to store with the subscription
        """
        if not self._active:
            logger.warning(
                f"Attempted to subscribe after teardown in component {self.component_id}"
            )
            return
        
        # Handle source filtering if the event bus supports it
        if source_filter and hasattr(self.event_bus, 'subscribe_filtered'):
            self.event_bus.subscribe_filtered(event_type, handler, source_filter)
            sources = {source_filter} if isinstance(source_filter, str) else set(source_filter)
        else:
            self.event_bus.subscribe(event_type, handler)
            sources = None
        
        # Track the subscription
        subscription = Subscription(
            event_type=event_type,
            handler=handler,
            source_filter=sources,
            metadata=metadata
        )
        self._subscriptions.append(subscription)
        
        logger.debug(
            f"Component {self.component_id} subscribed to {event_type} "
            f"(total subscriptions: {len(self._subscriptions)})"
        )
    
    def unsubscribe(
        self,
        event_type: Union[EventType, str],
        handler: EventHandler
    ) -> bool:
        """
        Unsubscribe from a specific event/handler combination.
        
        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove
            
        Returns:
            True if the subscription was found and removed
        """
        if not self._active:
            return False
        
        # Find and remove the subscription
        for i, sub in enumerate(self._subscriptions):
            if sub.event_type == event_type and sub.handler == handler:
                self.event_bus.unsubscribe(event_type, handler)
                self._subscriptions.pop(i)
                logger.debug(
                    f"Component {self.component_id} unsubscribed from {event_type}"
                )
                return True
        
        return False
    
    def unsubscribe_all(self) -> None:
        """
        Unsubscribe from all events.
        
        This is typically called during component teardown to ensure
        no event handlers remain registered.
        """
        if not self._active:
            return
        
        subscription_count = len(self._subscriptions)
        
        # Unsubscribe in reverse order (LIFO)
        while self._subscriptions:
            sub = self._subscriptions.pop()
            try:
                self.event_bus.unsubscribe(sub.event_type, sub.handler)
            except Exception as e:
                logger.error(
                    f"Error unsubscribing {sub.handler} from {sub.event_type}: {e}"
                )
        
        self._active = False
        
        logger.debug(
            f"Component {self.component_id} unsubscribed from all "
            f"{subscription_count} subscriptions"
        )
    
    def get_subscriptions(self) -> List[Subscription]:
        """Get a list of all active subscriptions."""
        return self._subscriptions.copy()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.unsubscribe_all()
        return False


class WeakSubscriptionManager(SubscriptionManager):
    """
    Subscription manager that uses weak references to handlers.
    
    This variant automatically removes subscriptions when the handler
    object is garbage collected, preventing memory leaks in long-running
    systems.
    """
    
    def subscribe(
        self,
        event_type: Union[EventType, str],
        handler: EventHandler,
        source_filter: Optional[Union[str, List[str]]] = None,
        **metadata
    ) -> None:
        """
        Subscribe with weak reference to the handler.
        
        Note: This works best with bound methods. For lambda or function
        handlers, use the regular SubscriptionManager.
        """
        # Try to create a weak reference
        try:
            if hasattr(handler, '__self__'):
                # Bound method - use WeakMethod
                weak_handler = WeakMethod(handler, self._cleanup_callback)
                
                # Create wrapper that calls the weak reference
                def wrapper(event: Event):
                    strong_handler = weak_handler()
                    if strong_handler:
                        strong_handler(event)
                
                wrapper._weak_ref = weak_handler
                wrapper._original = handler
                
                super().subscribe(event_type, wrapper, source_filter, **metadata)
            else:
                # Not a bound method, use regular subscription
                super().subscribe(event_type, handler, source_filter, **metadata)
                
        except TypeError:
            # Can't create weak reference, use regular subscription
            super().subscribe(event_type, handler, source_filter, **metadata)
    
    def _cleanup_callback(self, weak_ref):
        """Called when a weak reference is garbage collected."""
        # Find and remove subscriptions with this weak reference
        to_remove = []
        
        for i, sub in enumerate(self._subscriptions):
            if hasattr(sub.handler, '_weak_ref') and sub.handler._weak_ref == weak_ref:
                to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(to_remove):
            sub = self._subscriptions.pop(i)
            try:
                self.event_bus.unsubscribe(sub.event_type, sub.handler)
                logger.debug(
                    f"Auto-removed subscription for garbage collected handler "
                    f"in component {self.component_id}"
                )
            except:
                pass


def create_subscription_manager(
    event_bus: EventBusProtocol,
    component_id: Optional[str] = None,
    use_weak_refs: bool = False
) -> SubscriptionManager:
    """
    Factory function to create the appropriate subscription manager.
    
    Args:
        event_bus: The event bus to use
        component_id: Optional component identifier
        use_weak_refs: Whether to use weak references for handlers
        
    Returns:
        SubscriptionManager instance
    """
    if use_weak_refs:
        return WeakSubscriptionManager(event_bus, component_id)
    else:
        return SubscriptionManager(event_bus, component_id)