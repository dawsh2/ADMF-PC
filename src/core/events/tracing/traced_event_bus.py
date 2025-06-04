"""
TracedEventBus - EventBus with integrated event tracing capabilities

This module provides TracedEventBus which extends the standard EventBus
to automatically trace all events with full lineage and performance metrics.
"""

from typing import Optional, Callable, Any, Dict
import logging
from datetime import datetime

from ..event_bus import EventBus
from ...types.events import Event, EventType
from .event_tracer import EventTracer
from ...containers.protocols import Container

logger = logging.getLogger(__name__)


class TracedEventBus(EventBus):
    """
    EventBus with integrated event tracing.
    
    This enhanced EventBus automatically:
    1. Traces all published events with correlation IDs
    2. Tracks event causation chains
    3. Measures event processing latency
    4. Maintains event lineage for debugging
    
    The TracedEventBus is a drop-in replacement for EventBus,
    adding tracing capabilities without changing the interface.
    """
    
    def __init__(self, name: str = "traced_bus"):
        """
        Initialize TracedEventBus.
        
        Args:
            name: Name for this event bus instance
        """
        super().__init__(name)
        self.tracer: Optional[EventTracer] = None
        self._current_processing_event: Optional[str] = None
        self._processing_stack: list[str] = []  # Stack for nested event handling
        
    def set_tracer(self, tracer: EventTracer):
        """
        Attach event tracer to this bus.
        
        Args:
            tracer: EventTracer instance to use for tracing events
        """
        self.tracer = tracer
        logger.info(f"EventBus '{self.name}' now tracing with correlation_id: {tracer.correlation_id}")
        
    def set_correlation_id(self, correlation_id: str):
        """
        Set correlation ID for the event bus.
        
        This is used when we already have a tracer but want to update
        the correlation ID (e.g., for multi-portfolio scenarios).
        """
        if self.tracer:
            self.tracer.correlation_id = correlation_id
            logger.info(f"EventBus '{self.name}' correlation_id updated to: {correlation_id}")
            
    def publish(self, event: Event, source: Optional[Container] = None):
        """
        Publish event with automatic tracing.
        
        Args:
            event: Event to publish
            source: Optional container that is publishing the event
        """
        # Add causation if we're currently processing another event
        if self._current_processing_event:
            event.metadata['causation_id'] = self._current_processing_event
            
        # Trace the event if tracer is attached and source is provided
        if self.tracer and source:
            traced = self.tracer.trace_event(event, source.metadata.name)
            traced.emitted_at = datetime.now()
            
            # Log trace info at debug level
            logger.debug(
                f"Event traced: {traced.event_type} "
                f"[{traced.event_id}] from {traced.source_container}"
            )
            
        # Normal publish through parent class
        super().publish(event)
        
    def _dispatch_event(self, event: Event, handler: Callable):
        """
        Override to track processing context and measure latency.
        
        This method wraps the actual event handler to:
        1. Track which event is currently being processed
        2. Measure processing time
        3. Update traced event with timing information
        """
        # Save current processing context
        old_event = self._current_processing_event
        current_event_id = event.metadata.get('event_id')
        
        # Push to processing stack for nested handling
        if current_event_id:
            self._processing_stack.append(current_event_id)
            
        self._current_processing_event = current_event_id
        
        try:
            # Mark received time
            if self.tracer and current_event_id:
                traced_event = self.tracer.get_event(current_event_id)
                if traced_event:
                    traced_event.received_at = datetime.now()
                    
            # Process event through handler
            handler(event)
            
            # Mark processed time
            if self.tracer and current_event_id:
                traced_event = self.tracer.get_event(current_event_id)
                if traced_event:
                    traced_event.processed_at = datetime.now()
                    
                    # Log performance metrics at debug level
                    if traced_event.latency_ms > 100:  # Warn if >100ms
                        logger.warning(
                            f"High latency detected: {traced_event.event_type} "
                            f"[{traced_event.event_id}] took {traced_event.latency_ms:.1f}ms"
                        )
                    
        except Exception as e:
            # Log error with event context
            logger.error(
                f"Error processing {event.event_type} [{current_event_id or 'unknown'}]: {e}",
                exc_info=True
            )
            raise
            
        finally:
            # Restore previous processing context
            self._current_processing_event = old_event
            
            # Pop from processing stack
            if current_event_id and self._processing_stack:
                self._processing_stack.pop()
                
    def get_processing_stack(self) -> list[str]:
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
        if self.tracer:
            return self.tracer.get_summary()
        return None
        
    def trace_causation_chain(self, event_id: str) -> list:
        """
        Trace the complete causation chain for an event.
        
        Args:
            event_id: ID of the event to trace
            
        Returns:
            List of events in the causation chain
        """
        if self.tracer:
            return self.tracer.trace_causation_chain(event_id)
        return []
        
    def find_signal_to_fill_chain(self, fill_event_id: str) -> list:
        """
        Specialized method to trace from a fill back to its originating signal.
        
        Args:
            fill_event_id: ID of the fill event
            
        Returns:
            List of events from signal to fill
        """
        if not self.tracer:
            return []
            
        chain = self.trace_causation_chain(fill_event_id)
        
        # Filter to show only key events in the trading flow
        key_types = {'SIGNAL', 'ORDER', 'FILL', 'RISK_CHECK', 'POSITION_UPDATE'}
        return [e for e in chain if e.event_type in key_types]
        
    def __repr__(self) -> str:
        tracer_info = f", tracer={self.tracer.correlation_id}" if self.tracer else ""
        return f"TracedEventBus(name={self.name}{tracer_info})"