"""Event tracer for debugging and analysis."""

from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
import json
import logging

from ..protocols import EventObserverProtocol
from ..types import Event, EventType

logger = logging.getLogger(__name__)


class EventTracer:
    """Simple event tracer that records all events for analysis."""
    
    def __init__(self, trace_id: str = "default", max_events: int = 10000):
        """
        Initialize event tracer.
        
        Args:
            trace_id: Identifier for this trace session
            max_events: Maximum events to retain
        """
        self.trace_id = trace_id
        self.max_events = max_events
        self.events: List[Event] = []
        self.event_counts = defaultdict(int)
        self.start_time = datetime.now()
        
    def on_publish(self, event: Event) -> None:
        """Record published event."""
        if len(self.events) < self.max_events:
            self.events.append(event)
        self.event_counts[event.event_type] += 1
        
    def on_delivered(self, event: Event, handler: Any) -> None:
        """Record event delivery."""
        pass
        
    def on_error(self, event: Event, handler: Any, error: Exception) -> None:
        """Record handler error."""
        logger.error(f"Handler error for {event.event_type}: {error}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get trace summary."""
        return {
            'trace_id': self.trace_id,
            'total_events': sum(self.event_counts.values()),
            'event_counts': dict(self.event_counts),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'events_stored': len(self.events)
        }
        
    def save_to_file(self, filepath: str) -> None:
        """Save trace to file."""
        data = {
            'trace_id': self.trace_id,
            'summary': self.get_summary(),
            'events': [
                {
                    'event_type': e.event_type,
                    'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                    'correlation_id': e.correlation_id,
                    'payload': e.payload
                }
                for e in self.events
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()
        self.event_counts.clear()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage breakdown."""
        import sys
        
        # Calculate approximate memory usage
        events_memory = sum(sys.getsizeof(event) for event in self.events)
        counts_memory = sys.getsizeof(self.event_counts) + sum(
            sys.getsizeof(k) + sys.getsizeof(v) 
            for k, v in self.event_counts.items()
        )
        
        return {
            'stored_events': len(self.events),
            'max_events': self.max_events,
            'memory_usage_bytes': events_memory + counts_memory,
            'event_types_tracked': len(self.event_counts),
            'memory_efficiency': len(self.events) / self.max_events if self.max_events > 0 else 0
        }