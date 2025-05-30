"""
Container-aware EventBus implementation for ADMF-PC.

This module provides the core event bus that handles event routing within
a container. Each container gets its own EventBus instance, ensuring
complete isolation between different backtests or optimization trials.
"""

from __future__ import annotations
from typing import Dict, List, Set, Union, Optional, Callable
from collections import defaultdict
import threading
import logging
import asyncio
from weakref import WeakSet, ref
import traceback

from .types import Event, EventType, EventHandler, EventBusProtocol


logger = logging.getLogger(__name__)


class EventBus(EventBusProtocol):
    """
    Container-isolated event bus implementation.
    
    Each container gets its own EventBus instance, preventing any
    cross-contamination between parallel executions. Thread-safe
    for use within a single container.
    """
    
    def __init__(self, container_id: Optional[str] = None):
        """
        Initialize the EventBus.
        
        Args:
            container_id: Optional identifier for the container this bus belongs to
        """
        self.container_id = container_id
        self._subscribers: Dict[Union[EventType, str], List[EventHandler]] = defaultdict(list)
        self._handler_refs: Dict[EventHandler, Set[Union[EventType, str]]] = defaultdict(set)
        self._lock = threading.RLock()
        self._event_count = 0
        self._error_count = 0
        
        # Performance optimization: cache handler lists
        self._handler_cache: Dict[Union[EventType, str], tuple] = {}
        self._cache_dirty = False
        
        logger.debug(f"EventBus created for container: {container_id}")
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all registered handlers.
        
        Events are delivered synchronously in the order handlers were registered.
        Errors in handlers are caught and logged but don't stop propagation.
        
        Args:
            event: The event to publish
        """
        if event.container_id is None and self.container_id:
            event.container_id = self.container_id
        
        handlers = self._get_handlers(event.event_type)
        
        if not handlers:
            return
        
        self._event_count += 1
        
        # Execute handlers without holding the lock
        for handler in handlers:
            try:
                # Check if handler is async and handle appropriately
                if asyncio.iscoroutinefunction(handler):
                    # For async handlers, create a task if we're in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler(event))
                    except RuntimeError:
                        # No event loop running, call synchronously (not ideal but works)
                        import warnings
                        warnings.warn("Async handler called without event loop", RuntimeWarning)
                        asyncio.run(handler(event))
                else:
                    # Synchronous handler
                    handler(event)
            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"Error in event handler {handler} for event {event.event_type}: {e}",
                    exc_info=True
                )
                # Optionally publish error event (avoid infinite recursion)
                if event.event_type != EventType.ERROR:
                    self._publish_error_event(e, handler, event)
    
    def subscribe(self, event_type: Union[EventType, str], handler: EventHandler) -> None:
        """
        Subscribe a handler to events of a specific type.
        
        Args:
            event_type: The type of events to subscribe to
            handler: The handler function to call when events are published
        """
        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                self._handler_refs[handler].add(event_type)
                self._cache_dirty = True
                
                logger.debug(
                    f"Handler {handler} subscribed to {event_type} "
                    f"in container {self.container_id}"
                )
    
    def unsubscribe(self, event_type: Union[EventType, str], handler: EventHandler) -> None:
        """
        Unsubscribe a handler from events of a specific type.
        
        Args:
            event_type: The type of events to unsubscribe from
            handler: The handler to remove
        """
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                self._handler_refs[handler].discard(event_type)
                
                # Clean up empty entries
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
                if not self._handler_refs[handler]:
                    del self._handler_refs[handler]
                
                self._cache_dirty = True
                
                logger.debug(
                    f"Handler {handler} unsubscribed from {event_type} "
                    f"in container {self.container_id}"
                )
    
    def unsubscribe_all(self, handler: EventHandler) -> None:
        """
        Unsubscribe a handler from all event types.
        
        Args:
            handler: The handler to remove from all subscriptions
        """
        with self._lock:
            # Get all event types this handler is subscribed to
            event_types = list(self._handler_refs.get(handler, set()))
            
            # Unsubscribe from each
            for event_type in event_types:
                self.unsubscribe(event_type, handler)
    
    def clear(self) -> None:
        """Clear all subscriptions. Used during container teardown."""
        with self._lock:
            self._subscribers.clear()
            self._handler_refs.clear()
            self._handler_cache.clear()
            self._cache_dirty = False
            logger.debug(f"EventBus cleared for container {self.container_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the event bus."""
        with self._lock:
            return {
                "container_id": self.container_id,
                "event_count": self._event_count,
                "error_count": self._error_count,
                "subscription_count": sum(len(handlers) for handlers in self._subscribers.values()),
                "event_types": list(self._subscribers.keys()),
                "handler_count": len(self._handler_refs)
            }
    
    # Private methods
    
    def _get_handlers(self, event_type: Union[EventType, str]) -> tuple:
        """Get handlers for an event type, using cache if possible."""
        with self._lock:
            if self._cache_dirty or event_type not in self._handler_cache:
                # Create immutable tuple for thread-safe access
                handlers = tuple(self._subscribers.get(event_type, []))
                self._handler_cache[event_type] = handlers
                if event_type in self._handler_cache:
                    self._cache_dirty = False
                return handlers
            return self._handler_cache[event_type]
    
    def _publish_error_event(self, error: Exception, handler: EventHandler, original_event: Event) -> None:
        """Publish an error event when a handler fails."""
        try:
            error_event = Event(
                event_type=EventType.ERROR,
                payload={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "handler": str(handler),
                    "original_event_type": str(original_event.event_type),
                    "traceback": traceback.format_exc()
                },
                source_id="event_bus",
                container_id=self.container_id,
                metadata={
                    "category": "handler_error",
                    "original_event_id": original_event.metadata.get("event_id")
                }
            )
            # Publish without going through the normal flow to avoid recursion
            error_handlers = self._subscribers.get(EventType.ERROR, [])
            for error_handler in error_handlers:
                try:
                    error_handler(error_event)
                except:
                    # If error handlers fail, just log it
                    logger.exception("Error handler failed")
        except:
            logger.exception("Failed to publish error event")


class ContainerEventBus(EventBus):
    """
    Enhanced EventBus with container-specific features.
    
    This extends the basic EventBus with additional features useful
    for containerized execution:
    - Automatic cleanup on container disposal
    - Event filtering by source
    - Performance metrics
    """
    
    def __init__(self, container_id: str):
        super().__init__(container_id)
        self._source_filters: Dict[Union[EventType, str], Set[str]] = defaultdict(set)
        self._metrics = {
            "events_by_type": defaultdict(int),
            "handler_execution_times": defaultdict(list)
        }
    
    def subscribe_filtered(
        self,
        event_type: Union[EventType, str],
        handler: EventHandler,
        source_filter: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Subscribe to events with optional source filtering.
        
        Args:
            event_type: The type of events to subscribe to
            handler: The handler function
            source_filter: Optional source ID(s) to filter by
        """
        # Wrap handler with filter
        if source_filter:
            sources = [source_filter] if isinstance(source_filter, str) else source_filter
            filtered_handler = self._create_filtered_handler(handler, set(sources))
            self.subscribe(event_type, filtered_handler)
            # Store original handler reference for unsubscribe
            self._source_filters[event_type].add(handler)
        else:
            self.subscribe(event_type, handler)
    
    def _create_filtered_handler(
        self,
        handler: EventHandler,
        sources: Set[str]
    ) -> EventHandler:
        """Create a filtered wrapper for a handler."""
        def filtered_handler(event: Event) -> None:
            if event.source_id in sources:
                handler(event)
        
        # Preserve original handler reference
        filtered_handler.__wrapped__ = handler
        return filtered_handler
    
    def publish(self, event: Event) -> None:
        """Override to collect metrics."""
        self._metrics["events_by_type"][event.event_type] += 1
        super().publish(event)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics about event processing."""
        stats = self.get_stats()
        stats["metrics"] = {
            "events_by_type": dict(self._metrics["events_by_type"]),
            "total_events": sum(self._metrics["events_by_type"].values())
        }
        return stats