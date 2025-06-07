"""
Container-aware EventBus implementation for ADMF-PC.

This module provides the core event bus that handles event routing within
a container. Each container gets its own EventBus instance, ensuring
complete isolation between different backtests or optimization trials.
"""

from __future__ import annotations
from typing import Dict, List, Set, Union, Optional, Callable, Any
from collections import defaultdict
import threading
import logging
import asyncio
from weakref import WeakSet, ref
import traceback
from datetime import datetime

from ..types.events import Event, EventType, EventHandler, EventBusProtocol


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
        
        # Optional tracing support
        self._tracer: Optional['EventTracer'] = None
        self._current_processing_event: Optional[str] = None
        self._processing_stack: List[str] = []  # Stack for nested event handling
        self._current_correlation_id: Optional[str] = None
        
        logger.debug(f"EventBus created for container: {container_id}")
    
    def enable_tracing(self, trace_config: Dict[str, Any]) -> None:
        """
        Enable event tracing for this event bus.
        
        Args:
            trace_config: Configuration for tracing including:
                - correlation_id: Optional correlation ID for this trace session
                - max_events: Maximum events to keep in memory
        """
        from .tracing.event_tracer import EventTracer
        
        self._tracer = EventTracer(
            correlation_id=trace_config.get('correlation_id'),
            max_events=trace_config.get('max_events', 10000)
        )
        self._current_correlation_id = self._tracer.correlation_id
        logger.info(f"Tracing enabled for EventBus '{self.container_id}' with correlation_id: {self._current_correlation_id}")
    
    def disable_tracing(self) -> None:
        """Disable event tracing."""
        self._tracer = None
        self._current_correlation_id = None
        logger.info(f"Tracing disabled for EventBus '{self.container_id}'")
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set the current correlation ID for new events."""
        self._current_correlation_id = correlation_id
        if self._tracer:
            self._tracer.correlation_id = correlation_id
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all registered handlers.
        
        Events are delivered synchronously in the order handlers were registered.
        Errors in handlers are caught and logged but don't stop propagation.
        
        Args:
            event: The event to publish
        """
        # Set container_id if not set
        if event.container_id is None and self.container_id:
            event.container_id = self.container_id
        
        # Set correlation_id if not set
        if event.correlation_id is None and self._current_correlation_id:
            event.correlation_id = self._current_correlation_id
        
        # Set causation_id if we're currently processing another event
        if event.causation_id is None and self._current_processing_event:
            event.causation_id = self._current_processing_event
        
        # Generate event_id if not present
        if 'event_id' not in event.metadata:
            import uuid
            event.metadata['event_id'] = f"{event.event_type.value if hasattr(event.event_type, 'value') else event.event_type}_{uuid.uuid4().hex[:8]}"
        
        # Trace the event if tracer is enabled
        if self._tracer is not None:
            traced = self._tracer.trace_event(event, event.source_id or self.container_id)
            # Update timing in metadata
            event.metadata['trace_timing'] = {
                'emitted_at': datetime.now().isoformat()
            }
            logger.debug(f"Event traced: {event.event_type} [{event.metadata['event_id']}]")
        
        handlers = self._get_handlers(event.event_type)
        
        if not handlers:
            return
        
        self._event_count += 1
        
        # Execute handlers without holding the lock
        for handler in handlers:
            self._dispatch_event(event, handler)
    
    def _dispatch_event(self, event: Event, handler: EventHandler) -> None:
        """
        Dispatch event to handler with error handling and optional tracing.
        """
        # Save current processing context
        old_event = self._current_processing_event
        current_event_id = event.metadata.get('event_id')
        
        # Push to processing stack for nested handling
        if current_event_id:
            self._processing_stack.append(current_event_id)
            
        self._current_processing_event = current_event_id
        
        try:
            # Mark received time if tracing
            if self._tracer and current_event_id:
                event.metadata.setdefault('trace_timing', {})['received_at'] = datetime.now().isoformat()
            
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
            
            # Mark processed time if tracing
            if self._tracer and current_event_id:
                event.metadata.setdefault('trace_timing', {})['processed_at'] = datetime.now().isoformat()
                
        except Exception as e:
            self._error_count += 1
            logger.error(
                f"Error in event handler {handler} for event {event.event_type}: {e}",
                exc_info=True
            )
            # Optionally publish error event (avoid infinite recursion)
            if event.event_type != EventType.ERROR:
                self._publish_error_event(e, handler, event)
                
        finally:
            # Restore previous processing context
            self._current_processing_event = old_event
            
            # Pop from processing stack
            if current_event_id and self._processing_stack:
                self._processing_stack.pop()
    
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
                    f"in container {self.container_id} (bus id: {id(self)})"
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
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe a handler to ALL event types (wildcard subscription).
        
        This handler will receive every event published to this event bus,
        regardless of type. Useful for logging, debugging, or event capture.
        
        Args:
            handler: The handler function to call for all events
        """
        with self._lock:
            # Use a special wildcard key
            wildcard_key = '*'  # Special key for all events
            
            if handler not in self._subscribers[wildcard_key]:
                self._subscribers[wildcard_key].append(handler)
                self._handler_refs[handler].add(wildcard_key)
                self._cache_dirty = True
                
                logger.debug(
                    f"Handler {handler} subscribed to ALL events "
                    f"in container {self.container_id} (bus id: {id(self)})"
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
            stats = {
                "container_id": self.container_id,
                "event_count": self._event_count,
                "error_count": self._error_count,
                "subscription_count": sum(len(handlers) for handlers in self._subscribers.values()),
                "event_types": list(self._subscribers.keys()),
                "handler_count": len(self._handler_refs)
            }
            
            # Add tracer stats if available
            if self._tracer:
                stats["tracer_summary"] = self._tracer.get_summary()
                
            return stats
    
    def get_processing_stack(self) -> List[str]:
        """
        Get current event processing stack.
        
        Useful for debugging nested event handling.
        """
        return self._processing_stack.copy()
    
    def get_tracer_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics from the tracer.
        
        Returns:
            Dictionary with event statistics or None if no tracer attached
        """
        if self._tracer:
            return self._tracer.get_summary()
        return None
    
    def trace_causation_chain(self, event_id: str) -> List[Any]:
        """
        Trace the complete causation chain for an event.
        
        Args:
            event_id: ID of the event to trace
            
        Returns:
            List of events in the causation chain, empty if tracing not enabled
        """
        if self._tracer:
            return self._tracer.trace_causation_chain(event_id)
        return []
    
    # Private methods
    
    def _get_handlers(self, event_type: Union[EventType, str]) -> tuple:
        """Get handlers for an event type, using cache if possible."""
        with self._lock:
            if self._cache_dirty or event_type not in self._handler_cache:
                # Get specific handlers for this event type
                specific_handlers = list(self._subscribers.get(event_type, []))
                
                # Add wildcard handlers (subscribed to all events)
                wildcard_handlers = list(self._subscribers.get('*', []))
                
                # Combine them, with specific handlers first
                all_handlers = specific_handlers + wildcard_handlers
                
                # Create immutable tuple for thread-safe access
                handlers = tuple(all_handlers)
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
                correlation_id=original_event.correlation_id,  # Maintain correlation
                causation_id=original_event.metadata.get("event_id"),  # Error caused by original event
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